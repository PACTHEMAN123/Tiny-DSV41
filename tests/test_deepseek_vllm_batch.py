"""CPU regressions for atomic batched LoRA generation and request binding."""
import copy
import json
import socket
import sys
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

from rollout_manager.backends.vllm_actor import (
    LoRAActorEngine, lora_batch_id, receive, send, MAX_LORA_BATCH_RPC_BYTES,
)
import test_deepseek_vllm_controller as fixtures
from test_deepseek_vllm_remote import specs
from test_deepseek_vllm_transport import update


class BatchActorTests(unittest.TestCase):
    def engine(self):
        manifest, _, _ = update()
        engine = LoRAActorEngine.__new__(LoRAActorEngine)
        engine.bootstrap = {"max_model_len": 512, "max_num_seqs": 32, "eos_token_ids": [1]}
        engine.ack, engine.activation = {k: manifest[k] for k in ("version", "digest", "lineage")}, manifest
        engine.adapter_request = NS(lora_int_id=1, lora_name="immutable")
        engine.failed = False
        worker = {"pid": 1, "start_ticks": 2, "cmdline_sha256": "cmd", "device_uuid": "GPU-fixture"}
        calls = []
        class LLM:
            llm_engine = NS(vllm_config=NS(cache_config=NS(
                kv_cache_size_tokens=32 * 512, kv_cache_max_concurrency=32., effective_attention_block_size=16),
                scheduler_config=NS(max_num_batched_tokens=16384)))
            def __init__(self):
                self.pending, self.paused = None, False
                self.llm_engine.has_unfinished_requests = lambda: self.pending is not None
                self.llm_engine.engine_core = NS(call_utility=self.utility)
                self.llm_engine.output_processor = NS(request_states={})
            def utility(self, method, *args):
                if method == "pause_scheduler":
                    self.paused = True
                elif method == "resume_scheduler":
                    self.paused = False
                elif method == "is_scheduler_paused":
                    return self.paused
                else:
                    raise AssertionError(method)
            def generate(self, prompts, params, *, use_tqdm, lora_request):
                calls.append((prompts, params, lora_request))
                return self.complete(prompts)
            def enqueue(self, prompts, params, *, use_tqdm, lora_request):
                assert self.paused
                calls.append((prompts, params, lora_request))
                self.pending = prompts
                ids = [f"{100 + i}-abcdef{i:02x}" for i in range(len(prompts))]
                self.llm_engine.output_processor.request_states = {
                    value: NS(request_id=value, external_req_id=str(100 + i),
                              prompt_token_ids=prompt["prompt_token_ids"], lora_request=lora_request)
                    for i, (value, prompt) in enumerate(zip(ids, prompts))}
                return ids
            def wait_for_completion(self, *, use_tqdm):
                assert not self.paused
                prompts, self.pending = self.pending, None
                return self.complete(prompts)
            def complete(self, prompts):
                return [NS(request_id=str(100 + i), finished=True,
                    prompt_token_ids=prompt["prompt_token_ids"],
                    outputs=[NS(token_ids=[5, 1], logprobs=[{5: NS(logprob=-.2)}, {1: NS(logprob=-.4)}], finish_reason="stop")],
                    metrics=NS(queued_ts=10. + i * .001, scheduled_ts=11., first_token_ts=12. + i * .01, last_token_ts=15. + i * .01))
                    for i, prompt in enumerate(prompts)]
        engine.llm = LLM()
        engine.receipt = lambda: copy.deepcopy({"workers": [worker], "loaded_policy_ack": engine.ack})
        return engine, calls

    def request(self, engine, count=32):
        values = specs(count)
        for i, spec in enumerate(values):
            spec["prompt_ids"] = [i, 4]
        return {"specs": values, "policy_ack": dict(engine.ack), "adapter_id": 1,
                "batch_id": lora_batch_id(values, engine.ack)}

    def dispatch(self, engine, request):
        module = ModuleType("vllm")
        module.SamplingParams = lambda **values: NS(**values)
        with patch.dict(sys.modules, {"vllm": module}):
            return engine.generate_batch(request)

    def test_one_enqueue32_individual_seeds_and_exact_order(self):
        engine, calls = self.engine()
        request = self.request(engine)
        reply = self.dispatch(engine, request)
        self.assertEqual(len(calls), 1)
        self.assertEqual([p.seed for p in calls[0][1]], list(range(32)))
        self.assertTrue(all(p.n == 1 and p.top_p == 1 and p.top_k == -1 for p in calls[0][1]))
        self.assertIs(calls[0][2], engine.adapter_request)
        self.assertEqual(reply["batch_id"], request["batch_id"])
        for i, result in enumerate(reply["results"]):
            metrics = result["generation_metrics"]
            self.assertEqual(result["context_token_ids"], [i, 4])
            self.assertEqual(result["action_token_ids"], [5, 1])
            self.assertEqual(result["behavior_log_probs"], [-.2, -.4])
            self.assertEqual(result["stop_token_count"], 1)
            self.assertEqual((metrics["request_id"], metrics["sampling_seed"], metrics["batch_request_index"]), (f"{i:032x}", i, i))
            self.assertEqual(metrics["engine_request_id"], str(100 + i))
            self.assertNotIn("worker_graphs", metrics)
            self.assertNotIn("engine_receipt", metrics)  # one shared actor frame

    def test_capacity_policy_context_fail_before_dispatch_and_partial_output_latches(self):
        for mutation in ("sequences", "context", "kv", "prefill", "policy", "membership"):
            engine, calls = self.engine(); request = self.request(engine)
            if mutation == "sequences": engine.bootstrap["max_num_seqs"] = 1
            if mutation == "context": engine.bootstrap["max_model_len"] = 2
            if mutation == "kv": engine.llm.llm_engine.vllm_config.cache_config.kv_cache_max_concurrency = 31.
            if mutation == "prefill": engine.llm.llm_engine.vllm_config.scheduler_config.max_num_batched_tokens = 63
            if mutation == "policy": request["policy_ack"]["version"] = 99
            if mutation == "membership": request["specs"][0]["sampling_seed"] = 999
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, RuntimeError)):
                self.dispatch(engine, request)
            self.assertEqual(calls, [])
        engine, _ = self.engine(); request = self.request(engine)
        original = engine.llm.wait_for_completion
        engine.llm.wait_for_completion = lambda *a, **kw: original(*a, **kw)[:-1]
        with self.assertRaisesRegex(RuntimeError, "admitted request IDs"):
            self.dispatch(engine, request)
        self.assertTrue(engine.failed)
        with self.assertRaisesRegex(RuntimeError, "immutable"):
            self.dispatch(engine, request)

    def test_duplicate_or_reordered_engine_ids_rejected_even_when_prompts_equal(self):
        for ids in (("100", "100"), ("101", "100")):
            engine, _ = self.engine(); request = self.request(engine, 2)
            original = engine.llm.wait_for_completion
            def outputs(*args, **kwargs):
                result = original(*args, **kwargs)
                for output, value in zip(result, ids): output.request_id = value
                return result
            engine.llm.wait_for_completion = outputs
            with self.assertRaisesRegex(RuntimeError, "admitted request IDs"):
                self.dispatch(engine, request)
            self.assertTrue(engine.failed)

    def test_singleton_keeps_public_generate(self):
        engine, calls = self.engine()
        engine.llm.enqueue = lambda *a, **kw: self.fail("singleton enqueued")
        result = self.dispatch(engine, self.request(engine, 1))
        self.assertEqual(len(calls), 1)
        self.assertEqual(result["results"][0]["action_token_ids"], [5, 1])

    def test_repeated_g32_replies_do_not_accumulate_prior_batch_history(self):
        engine, _ = self.engine()
        sizes = []
        for group in range(64):
            request = self.request(engine)
            for index, spec in enumerate(request["specs"]):
                spec["request_id"] = f"{group * 32 + index:032x}"
            request["batch_id"] = lora_batch_id(request["specs"], engine.ack)
            reply = self.dispatch(engine, request)
            sizes.append(len(json.dumps(reply)))
        self.assertEqual(min(sizes), max(sizes))
        self.assertLess(max(sizes), 64 * 1024)

    def test_large_batch_frame_is_bounded_and_legacy_default_stays2MiB(self):
        value = {"payload": "x" * (2 * 1024**2)}
        left, right = socket.socketpair()
        with left, right:
            with self.assertRaisesRegex(ValueError, "byte budget"):
                send(left, value)
            thread = threading.Thread(target=send, args=(left, value), kwargs={"max_bytes": MAX_LORA_BATCH_RPC_BYTES})
            thread.start()
            self.assertEqual(receive(right, max_bytes=MAX_LORA_BATCH_RPC_BYTES), value)
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())


class BatchControllerTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.LoRAControllerTests.asyncSetUp

    async def test_batch_identity_replay_and_deadline(self):
        from dataclasses import replace
        self.controller.config = replace(self.config, max_num_seqs=32)
        await self.controller.activate(*update(base_identity=self.controller.base_identity)[:2])
        await self.controller.resume()
        values = specs(32)
        results = self.controller.sample_batch(values)
        self.assertEqual(len(results), 32)
        self.assertEqual(list((self.root / "work").glob("*.json")), [])
        self.assertTrue(all("engine_receipt" not in result["generation_metrics"] for result in results))
        with self.assertRaisesRegex(RuntimeError, "replay"):
            self.controller.sample_batch(values)
        self.assertEqual(self.controller._active, 0)
        with patch.object(self.controller, "_timeout", side_effect=TimeoutError("expired")), self.assertRaises(TimeoutError):
            self.controller.sample_batch([{**values[0], "request_id": "f" * 32}])
        self.assertNotIn("f" * 32, self.controller._seen)

    async def test_batch_wrong_seed_fails_controller(self):
        from dataclasses import replace
        self.controller.config = replace(self.config, max_num_seqs=32)
        await self.controller.activate(*update(base_identity=self.controller.base_identity)[:2])
        await self.controller.resume()
        original = self.controller.actor.call
        def wrong(*args, **kwargs):
            reply = original(*args, **kwargs)
            reply["results"][0]["generation_metrics"]["sampling_seed"] = 999
            return reply
        with patch.object(self.controller.actor, "call", side_effect=wrong), self.assertRaises(RuntimeError):
            self.controller.sample_batch(specs(32))
        self.assertEqual(self.controller._state, "failed")
        self.assertEqual(list((self.root / "work").glob("*.sampled.json")), [])
        self.assertEqual(self.controller._active, 0)


if __name__ == "__main__":
    unittest.main()
