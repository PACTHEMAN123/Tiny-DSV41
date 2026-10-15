import copy
import gc
import os
import random
import tempfile
import threading
import time
import unittest
from pathlib import Path

from rollout_manager.input_cache import CacheClient, InputCache, cache_key, canonical
from rollout_manager.adapters.input_reuse import InputReuse, layout, restore_rng, rng_state


SOURCE = """import torch
def get_inputs():
    x = torch.rand(3, 4)
    return [x, x.t(), x, torch.randint(0, 10, (8,)), 3]
"""


def identity():
    return dict(task_sha256="task", generator_sha256="generator", configuration={}, namespace="development",
                seed=42, trial_index=0, input_contract="contract", adapter_version="v1", runtime={}, rng_before="rng")


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ric-", dir="/tmp")
        self.cache = InputCache(Path(self.temp.name)/"c", ram_bytes=0, disk_bytes=4096)
        self.client = CacheClient(self.cache.socket_path)
        self.consumer = self.client.control("begin", copy_limit=4096)["consumer"]

    def tearDown(self):
        for consumer in list(self.cache.consumers):
            self.client.control("end", consumer=consumer)
        self.cache.close()
        self.temp.cleanup()

    def publish(self, key, size=1024):
        result, fd = self.client.call("acquire", consumer=self.consumer, identity=key, size=size)
        if result["status"] == "miss":
            os.pwrite(fd, b"a"*size, 0)
            os.close(fd)
            self.client.control("publish", consumer=self.consumer, key=result["key"], size=size,
                metadata={"generation_seconds": 1, "publication_copy_bytes": size, "publication_copy_seconds": .1})
        elif fd is not None:
            os.close(fd)
        return result

    def test_every_identity_field_changes_key_and_final_is_separate(self):
        base = identity()
        for field in base:
            changed = copy.deepcopy(base)
            changed[field] = ("final-verification" if field == "namespace" else {"changed": True})
            self.assertNotEqual(cache_key(base), cache_key(changed), field)
        del base["runtime"]
        with self.assertRaises(ValueError):
            cache_key(base)

    def test_references_prevent_eviction_and_are_accounted_separately(self):
        self.publish(identity(), 4096)
        hit, fd = self.client.call("acquire", consumer=self.consumer, identity=identity(), size=4096)
        self.assertEqual(hit["status"], "hit")
        os.close(fd)
        snap = self.cache.snapshot()
        self.assertEqual(snap["shared_storage_bytes"]["disk"], 4096)
        self.assertEqual(snap["consumer_cow_reserved_bytes"], 4096)
        self.assertEqual(self.publish(dict(identity(), seed=43))["status"], "bypass")
        self.client.control("release", consumer=self.consumer, lease=hit["lease"])
        self.assertEqual(self.publish(dict(identity(), seed=43))["status"], "miss")
        self.assertEqual(self.cache.snapshot()["metrics"]["evictions"], 1)

    def test_single_flight_and_abandoned_generation_cleanup(self):
        miss, fd = self.client.call("acquire", consumer=self.consumer, identity=identity(), size=4096)
        os.close(fd)
        other = self.client.control("begin", copy_limit=4096)["consumer"]
        reply = []
        def waiter():
            result, descriptor = self.client.call("acquire", consumer=other, identity=identity(), size=4096)
            os.close(descriptor)
            reply.append(result)
        thread = threading.Thread(target=waiter)
        thread.start()
        self.client.control("end", consumer=self.consumer)
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(reply[0]["status"], "miss")
        self.client.control("end", consumer=other)
        self.assertEqual(self.cache.snapshot()["generating_bytes"], 0)

    def test_no_clear_with_live_consumers(self):
        with self.assertRaises(RuntimeError):
            self.cache.clear()

    def test_copy_limit_bypasses_without_returning_an_unsafe_mapping(self):
        self.publish(identity(), 4096)
        other = self.client.control("begin", copy_limit=16)["consumer"]
        result, fd = self.client.call("acquire", consumer=other, identity=identity(), size=4096)
        self.assertEqual(result["reason"], "consumer_copy_budget")
        self.assertIsNone(fd)

    def test_ram_seal_prevents_writes(self):
        if not os.path.isdir("/proc/self/fd"):
            self.skipTest("Linux sealed memfd check")
        self.client.control("end", consumer=self.consumer)
        self.cache.close()
        self.cache = InputCache(Path(self.temp.name)/"ram", ram_bytes=4096)
        self.client = CacheClient(self.cache.socket_path)
        self.consumer = self.client.control("begin", copy_limit=4096)["consumer"]
        self.publish(identity())
        _, fd = self.client.call("acquire", consumer=self.consumer, identity=identity(), size=1024)
        try:
            with self.assertRaises(OSError):
                os.pwrite(fd, b"x", 0)
        finally:
            os.close(fd)


class GeneratorReuseTests(unittest.TestCase):
    def setUp(self):
        import torch
        from rollout_manager.adapters.input_memory import make_memory_plan
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory(prefix="ric-", dir="/tmp")
        self.cache = InputCache(Path(self.temp.name)/"c", ram_bytes=0, disk_bytes=1024*1024)
        self.client = CacheClient(self.cache.socket_path)
        self.consumer = self.client.control("begin", copy_limit=1024*1024)["consumer"]
        context = {}
        exec(SOURCE, context)
        self.generator = context["get_inputs"]
        # The cache accepts a reviewed plan; this fixture includes an alias/view.
        import hashlib
        self.plan = {"source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
                     "torch_version": torch.__version__, "allocation_bytes": 4096}

    def tearDown(self):
        gc.collect()
        for c in list(self.cache.consumers):
            self.client.control("end", consumer=c)
        self.cache.close()
        self.temp.cleanup()

    def adapter(self, namespace="development", seed=42):
        return InputReuse(self.torch, self.cache.socket_path, self.consumer, SOURCE, self.plan,
                          namespace=namespace, seed=seed, mode="kernelbench", qualifying=True)

    def test_values_aliases_rng_and_mutation_isolation(self):
        torch = self.torch
        import numpy as np
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        before = rng_state(torch)
        native = self.generator()
        after = rng_state(torch)
        restore_rng(torch, before)
        first = self.adapter()
        cold = first.get_inputs(self.generator, 0)
        self.assertEqual(first.events[0]["status"], "miss")
        self.assertEqual(canonical(after), canonical(rng_state(torch)))
        restore_rng(torch, before)
        warm = self.adapter()
        shared = warm.get_inputs(lambda: self.fail("hit regenerated native inputs"), 0)
        self.assertEqual(warm.events[0]["status"], "hit")
        self.assertEqual(canonical(after), canonical(rng_state(torch)))
        for a, b in zip(native, shared):
            if isinstance(a, torch.Tensor):
                self.assertTrue(torch.equal(a, b))
                self.assertEqual(a.stride(), b.stride())
        self.assertIs(shared[0], shared[2])
        self.assertEqual(shared[0].untyped_storage()._cdata, shared[1].untyped_storage()._cdata)
        shared[0].fill_(-1)
        self.assertTrue(torch.equal(shared[1], shared[0].t()))
        restore_rng(torch, before)
        independent = self.adapter().get_inputs(self.generator, 0)
        self.assertTrue(torch.equal(native[0], independent[0]))
        self.assertTrue(torch.equal(native[0], cold[0]))
        self.assertEqual(self.cache.snapshot()["references"], 2)
        del shared, independent, a, b
        gc.collect()
        self.assertEqual(self.cache.snapshot()["references"], 0)

    def test_python_numpy_rng_and_namespace(self):
        import numpy as np
        torch = self.torch
        def generator():
            return [torch.rand(8), random.random(), float(np.random.random())]
        torch.manual_seed(7)
        state = rng_state(torch)
        cold = self.adapter().get_inputs(generator, 2)
        after = rng_state(torch)
        restore_rng(torch, state)
        hit = self.adapter()
        warm = hit.get_inputs(generator, 2)
        self.assertEqual(cold[1:], warm[1:])
        self.assertEqual(canonical(after), canonical(rng_state(torch)))
        restore_rng(torch, state)
        final = self.adapter("final-verification", 42)
        final.get_inputs(generator, 2)
        self.assertEqual(final.events[0]["status"], "miss")
        restore_rng(torch, state)
        different_trial = self.adapter()
        different_trial.get_inputs(generator, 3)
        self.assertEqual(different_trial.events[0]["status"], "miss")

    def test_failure_aborts_reservation_and_remains_an_infrastructure_error(self):
        adapter = self.adapter()
        def broken():
            raise ValueError("bad generator")
        with self.assertRaisesRegex(RuntimeError, "bad generator"):
            adapter.get_inputs(broken, 0)
        self.assertEqual(self.cache.snapshot()["generating_bytes"], 0)
        with self.assertRaises(RuntimeError):
            adapter.metrics()

    def test_unqualified_generator_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "qualification"):
            InputReuse(self.torch, self.cache.socket_path, self.consumer, SOURCE, self.plan,
                       namespace="development", seed=42, mode="kernelbench")

    def test_bitwise_checker_covers_signed_zero_and_nan(self):
        from rollout_manager.analysis.qualify_input_reuse import equal_inputs
        torch = self.torch
        a = torch.tensor([0., float("nan")])
        self.assertTrue(equal_inputs(torch, [a], [a.clone()]))
        changed = a.clone()
        changed[0] = -0.
        self.assertFalse(equal_inputs(torch, [a], [changed]))

    def test_checker_rejects_lost_storage_and_object_aliases(self):
        from rollout_manager.analysis.qualify_input_reuse import equal_inputs
        torch = self.torch
        a = torch.arange(6.).reshape(2, 3)
        b = a.clone()
        self.assertTrue(equal_inputs(torch, [a, a.t(), a], [b, b.t(), b]))
        self.assertFalse(equal_inputs(torch, [a, a.t()], [b, b.t().clone()]))
        self.assertFalse(equal_inputs(torch, [a, a], [b, b.view_as(b)]))

    def test_resolved_generator_configuration_is_qualified_and_keyed(self):
        import hashlib
        source = "import torch\nsize = 4\ndef get_inputs():\n    return [torch.rand(size)]\n"
        context = {}
        exec(source, context)
        plan = dict(self.plan, source_sha256=hashlib.sha256(source.encode()).hexdigest())
        args = (self.torch, self.cache.socket_path, self.consumer, source, plan)
        options = dict(namespace="development", seed=42, mode="kernelbench")
        cold = InputReuse(*args, **options, qualifying=True)
        cold.get_inputs(context["get_inputs"], 0)
        self.assertEqual(cold.identity["configuration"]["generator_globals"], {"size": 4})
        qualification = {"passed": True, "identity": copy.deepcopy(cold.identity)}
        context["size"] = 8
        changed = InputReuse(*args, **options, qualification=qualification)
        with self.assertRaisesRegex(RuntimeError, "configuration changed"):
            changed.get_inputs(context["get_inputs"], 0)


if __name__ == "__main__":
    unittest.main()
