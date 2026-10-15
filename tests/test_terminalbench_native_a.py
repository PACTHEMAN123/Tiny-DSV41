"""A experiment control regression checks; no fabricated benchmark episodes."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


from rollout_manager.terminal_bench import native_workflow as runner
from rollout_manager.terminal_bench import sealed_provider


class NativeExperimentControlTests(unittest.TestCase):
    def provider(self):
        provider = runner.SealedCPUGroupProvider.__new__(runner.SealedCPUGroupProvider)
        provider.config = SimpleNamespace(worker_timeout_seconds=.03)
        provider._loop = provider._thread = None
        provider._control_futures = []
        provider._error = None
        provider._closed = provider._paused = False
        provider._next = 0
        provider._retained, provider._ack = {}, {}
        provider._automatic_artifacts = False
        provider._historical_delivery = None
        provider._expected_max_turns = None
        return provider

    def test_missing_cpu_seal_waits_and_failed_control_is_not_silently_empty(self):
        self.assertIs(runner.SealedCPUGroupProvider, sealed_provider.SealedCPUGroupProvider)
        with tempfile.TemporaryDirectory() as temporary:
            provider = self.provider()
            provider.manifests = (Path(temporary) / "group-0.seal.json",)
            started = time.monotonic()
            with self.assertRaises(queue.Empty):
                provider.get(timeout=.05)
            self.assertGreaterEqual(time.monotonic() - started, .04)
            provider._error = RuntimeError("real control failure")
            with self.assertRaisesRegex(RuntimeError, "real control failure"):
                provider.get(timeout=.05)

    def test_python310_future_timeout_is_normalized_without_cancelling_operation(self):
        class LegacyFutureTimeout(Exception):
            pass  # Python 3.10's future timeout is separate from builtin TimeoutError.
        provider = self.provider()
        future = Mock()
        future.result.side_effect = LegacyFutureTimeout()
        async def operation():
            return None
        coroutine = operation()
        try:
            with patch.object(sealed_provider, "FutureTimeoutError", LegacyFutureTimeout), patch.object(
                    sealed_provider.asyncio, "run_coroutine_threadsafe", return_value=future):
                with self.assertRaisesRegex(RuntimeError, "acknowledgement timed out"):
                    provider._call(coroutine)
            future.cancel.assert_not_called()
            self.assertEqual(provider._control_futures, [future])
            self.assertIsInstance(provider._error, RuntimeError)
        finally:
            coroutine.close()

    def test_control_observer_timeout_waits_for_actual_operation_before_cleanup(self):
        provider = self.provider()
        ready, entered, release, operation_done, closed = (threading.Event() for _ in range(5))
        def control_loop():
            provider._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(provider._loop)
            ready.set()
            provider._loop.run_forever()
            provider._loop.close()
        provider._thread = threading.Thread(target=control_loop)
        provider._thread.start()
        self.assertTrue(ready.wait(2))
        def operation():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("test operation was not released")
            operation_done.set()
        class Shim:
            def close(self):
                if not operation_done.is_set():
                    raise AssertionError("owned operation detached before shim cleanup")
        class Native:
            async def close(self):
                closed.set()
        provider.shim, provider.native = Shim(), Native()
        async def owned_operation():
            from rollout_manager.backends.native_qwen_loopback import _owned_call
            await _owned_call(operation)
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                with self.assertRaisesRegex(RuntimeError, "acknowledgement timed out"):
                    provider._call(owned_operation())
                self.assertTrue(entered.is_set())
                closing = executor.submit(provider.close)
                time.sleep(.04)
                self.assertFalse(closing.done())
                self.assertFalse(closed.is_set())
                self.assertTrue(provider._thread.is_alive())
                release.set()
                closing.result(timeout=2)
                self.assertTrue(operation_done.is_set() and closed.is_set())
                self.assertFalse(provider._thread.is_alive())
            finally:
                release.set()
                provider.close()

    def test_updated_policy_wait_uses_fresh_singleton_lease_and_proves_delivery(self):
        import json
        from rollout_manager.terminal_bench.sealed import PolicyEvaluation
        from rollout_manager.training import PolicyRef
        with tempfile.TemporaryDirectory() as temporary:
            provider = self.provider()
            provider.output = Path(temporary)
            provider.post_update_manifest = provider.output / "group-1.seal.json"
            provider.post_update_manifest.write_text("immutable fixture seal")
            provider._post_update_done = False
            provider._next = 1
            provider.task_ids = ("sparql-university",)
            policy = PolicyRef(1, "a" * 64, "lineage", {}, {"temperature": .8, "top_p": 1, "top_k": 0})
            provider.native = SimpleNamespace(policy=policy)
            ack = {"version": 1, "digest": policy.digest, "lineage": policy.lineage}
            lease = {"group_size": 1, "purpose": "policy_evaluation", "policy": policy.to_dict()}
            provider._build_lease = Mock(return_value=lease)
            provider._weight_update = {"weights_changed": True}
            provider._expected_original_seal_sha256 = None
            provider.pause = Mock(return_value={"delivery_drain": {"complete": True}})
            calls = []
            class Controller:
                async def resume(self):
                    calls.append("resume")
                    return ack
                async def validate_group_delivery(self, value):
                    calls.append("delivery")
                    return {"complete": True}
            provider._delivery_controller = lambda: Controller()
            provider._call = asyncio.run
            sample = SimpleNamespace(reward=1., to_dict=lambda: {"fixture": True})
            evaluation = PolicyEvaluation(1, "sparql-university", policy, (sample,))
            with patch("rollout_manager.terminal_bench.sealed.validate_sealed_group", return_value=evaluation) as validate:
                self.assertEqual(provider.resume(), ack)
            validate.assert_called_once_with(provider.post_update_manifest, policy, 1,
                expected_lease=lease, expected_group_size=1, evaluation=True)
            self.assertEqual(calls, ["resume", "delivery", "resume"])
            self.assertTrue(provider._post_update_done)
            self.assertFalse(provider._paused)
            self.assertEqual(json.loads((provider.output / "cpu-policy-lease-v00000001-evaluation.json").read_text()), lease)
            receipt = json.loads((provider.output / "updated-policy-episode.json").read_text())
            self.assertFalse(receipt["second_optimizer_step"])
            self.assertTrue(receipt["actual_gpu_delivery"]["complete"])

    def test_cpu_shape_keeps_legacy_g2_and_requires_explicit_singleton_evaluation(self):
        from rollout_manager.terminal_bench.cpu_workflow import execution_shape
        from rollout_manager.terminal_bench.sealed import TASKS
        self.assertEqual(execution_shape({}), (TASKS, 2))
        self.assertEqual(execution_shape({"task_ids": ["sparql-university"], "group_size": 8}),
                         (("sparql-university",), 8))
        with self.assertRaises(ValueError):
            execution_shape({"group_size": 1})
        self.assertEqual(execution_shape({"group_size": 1, "purpose": "policy_evaluation"}), (TASKS, 1))

    def test_turn_budget_declaration_preserves_original_policy_identity(self):
        original = {"identity": {"rollout_profile": {"max_turns": 128}}}
        self.assertEqual(runner.turn_budget_declaration(None), (128, 128))
        self.assertEqual(runner.turn_budget_declaration(16), (16, 16))
        self.assertEqual(runner.turn_budget_declaration(None, original), (128, None))
        self.assertEqual(runner.turn_budget_declaration(16, original), (128, 16))
        self.assertEqual(original["identity"]["rollout_profile"]["max_turns"], 128)
        for value in (True, 0, 1025, "16"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "agent_max_turns"):
                runner.turn_budget_declaration(value)

    def test_zero_variance_g8_optimizer_counter_does_not_prove_weight_change(self):
        import torch
        from dsv41_train.rl.config import RLConfig
        from dsv41_train.rl.runtime import train_groups
        from rollout_manager.training import PolicyRef, TrainingGroup, TrainingTrajectory, TurnTrace, token_digest
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.transitions = torch.nn.Parameter(torch.arange(36, dtype=torch.float32).view(6, 6) / 36)
                self.config = SimpleNamespace(model_type="fixture")
            def forward(self, input_ids, attention_mask=None):
                return self.transitions[input_ids]
            def assert_engram_invariants(self):
                pass
        model = Model()
        policy = PolicyRef(0, "a" * 64, "lineage", {}, {"temperature": .8, "top_p": 1, "top_k": 0})
        context = token_digest([1, 2])
        behavior = float((model.transitions[2] / .8).log_softmax(-1)[3].detach())
        samples = [TrainingTrajectory(0, "fixture", context, "b" * 64, index,
            (TurnTrace(1, "%032x" % (index + 1), [1, 2], [3], [behavior], policy, "stop", 0),),
            0.) for index in range(8)]
        group = TrainingGroup(0, "fixture", context, "b" * 64, policy, 8, samples)
        before = runner.trainable_tensor_digest(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.1, weight_decay=0, foreach=False)
        metrics = train_groups(model, None, optimizer, [group], RLConfig(rollout_engine="native-pytorch", temperature=.8, kl_beta=0, weight_decay=0))
        self.assertEqual(metrics["nonzero_advantages"], 0)
        self.assertEqual(metrics["grad_norm"], 0.)
        self.assertEqual(metrics["zero_variance_group_fraction"], 1.)
        self.assertEqual(float(optimizer.state[model.transitions]["step"]), 1.)
        self.assertEqual(before, runner.trainable_tensor_digest(model))


if __name__ == "__main__":
    unittest.main()
