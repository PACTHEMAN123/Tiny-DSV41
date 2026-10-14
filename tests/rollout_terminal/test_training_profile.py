"""Independent training attempts preserve the original evaluation contract."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import ModuleType
import unittest
from unittest.mock import AsyncMock, patch

from rollout_manager.backends.terminal_bench import TerminalBenchBackend, _identity_key
from rollout_manager.contracts import CancellationToken, EpisodeIdentity, EpisodePlan, ResourceRequirements, TaskSpec
from rollout_manager.generation_training import TrainingGenerationContract
from rollout_manager.training import PolicyRef, TurnTrace
from rollout_manager.terminal_bench import admission, controller
if __package__:
    from .test_classification import trial
else:
    from test_classification import trial


def profile(root):
    policy = PolicyRef(2, "a" * 64, "fixture", identities={
        "tokenizer_sha256": "b" * 64, "chat_template_sha256": "c" * 64})
    receipt = {"schema_version": 1, "policy": policy.to_dict(), "logprob_convention": "sampled_distribution",
        **{name: True for name in TrainingGenerationContract.REQUIRED}}
    return {"execution_profile": "training", "training_policy": policy.to_dict(),
        "training_capabilities": receipt, "output_dir": str(root), "relay_root": str(root),
        "model_guard_evidence": str(root / "guard.json"), "payload_dir": str(root)}


def plans(root):
    task = TaskSpec("terminal-bench", "merge-diff-arc-agi-task", "a" * 64,
        {"task_path": str(root / "tasks/merge-diff-arc-agi-task")},
        ResourceRequirements("cpu-only", 1, 4096 * 1024**2, 10240 * 1024**2))
    return [EpisodePlan(EpisodeIdentity("run", task.benchmark, task.task_id, "group", i), task) for i in range(2)]


class TrainingProfileTests(unittest.IsolatedAsyncioTestCase):
    async def test_training_samples_share_gate_but_have_independent_episode_identities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = TerminalBenchBackend(profile(root))
            backend._started = True
            samples = plans(root)
            with patch.object(backend, "_checked_plan", return_value={"required_service_seconds": 3060}), \
                 patch.object(admission, "check_empty_spool"), \
                 patch.object(admission, "check_readiness", return_value={"gpu": {}}) as ready, \
                 patch.object(controller, "read_json", return_value={}), \
                 patch.object(controller, "guard_gate", return_value={}):
                receipt = await backend.admit_group(samples)
                self.assertEqual(len(set(receipt["episode_ids"])), 2)
                ready.assert_called_once_with(root, controller.MODEL, 2, 3060, required_generation_seconds=360)
                await backend.finish_group(samples, [])
                with self.assertRaisesRegex(ValueError, "replayed"):
                    await backend.admit_group(samples)

    async def test_evaluation_duplicate_task_and_training_mixed_policy_remain_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = TerminalBenchBackend({"output_dir": str(root)})
            backend._started = True
            with self.assertRaisesRegex(ValueError, "unique pinned"):
                await backend.admit_group(plans(root))
            backend = TerminalBenchBackend(profile(root))
            backend._started = True
            samples = plans(root)
            other = profile(root)
            other["training_policy"]["version"] = 3
            other["training_capabilities"]["policy"]["version"] = 3
            samples[1] = replace(samples[1], payload={"training_policy": other["training_policy"],
                "training_capabilities": other["training_capabilities"]})
            with self.assertRaisesRegex(ValueError, "same leased policy"):
                await backend.admit_group(samples)

    def test_training_profile_requires_real_serving_evidence_contract(self):
        with self.assertRaisesRegex(ValueError, "leased PolicyRef"):
            TerminalBenchBackend({"execution_profile": "training"})._training_contract()

    def test_agent_turn_budget_default_and_quick_run_reach_harbor_config(self):
        module = ModuleType("harbor.models.job.config")
        module.JobConfig = type("JobConfig", (), {"model_validate": staticmethod(lambda value: value)})
        bootstrap = ModuleType("rollout_manager.terminal_bench.harbor_bootstrap_ubuntu_merge_sparql")
        bootstrap.IMAGE_MAP_PIN = hashlib.sha256(b"{}").hexdigest()
        bootstrap.PAYLOAD_PIN = "fixture"
        with tempfile.TemporaryDirectory() as tmp, patch.dict("sys.modules", {module.__name__: module, bootstrap.__name__: bootstrap}):
            root = Path(tmp)
            image_map = root / "image-map.json"
            image_map.write_bytes(b"{}")
            for turns in (None, 16):
                with self.subTest(turns=turns):
                    settings = profile(root)
                    settings["image_map_path"] = str(image_map)
                    if turns is not None:
                        settings["agent_max_turns"] = turns
                    config = TerminalBenchBackend(settings).build_config(plans(root)[0], root)
                    kwargs = config["agents"][0]["kwargs"]
                    self.assertEqual(kwargs["max_turns"], 128 if turns is None else 16)
                    self.assertEqual((kwargs["max_tokens"], kwargs["temperature"], kwargs["command_seconds"]), (2048, 1, 30))
                    self.assertEqual(config["retry"]["max_retries"], 0)
                    self.assertEqual(config["verifier"], {"disable": False})

    def test_agent_turn_budget_requires_bounded_integer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = {**profile(root), "executor_root": str(root), "task_copy_dir": str(root / "tasks"),
                "task_provenance_sha256": "a" * 64, "payload_sha256": "b" * 64,
                "bootstrap_qualifications": {"sparql-university": str(root / "baseline")}}
            for value in (True, False, 0, -1, 1025, 16.0, "16", None):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, "agent_max_turns"):
                    TerminalBenchBackend.validate_config({**settings, "agent_max_turns": value})
            for value in (1, 16, 128, 1024):
                TerminalBenchBackend.validate_config({**settings, "agent_max_turns": value})
            TerminalBenchBackend.validate_config(settings)

    def test_new_checkpoint_receipt_preserves_negotiated_tokenizer_and_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = profile(root)
            backend = TerminalBenchBackend(settings)
            current = profile(root)
            current["training_policy"]["version"] = 3
            current["training_capabilities"]["policy"]["version"] = 3
            sample = replace(plans(root)[0], payload={"training_policy": current["training_policy"],
                "training_capabilities": current["training_capabilities"]})
            self.assertEqual(backend._training_contract(sample).policy.version, 3)
            current["training_policy"]["identities"]["tokenizer_sha256"] = "d" * 64
            current["training_capabilities"]["policy"]["identities"]["tokenizer_sha256"] = "d" * 64
            bad = replace(sample, payload={"training_policy": current["training_policy"],
                "training_capabilities": current["training_capabilities"]})
            with self.assertRaisesRegex(ValueError, "negotiated"):
                backend._training_contract(bad)


@unittest.skipUnless(importlib.util.find_spec("harbor"), "optional Harbor contract environment unavailable")
class HarborTrainingConfigTests(unittest.TestCase):
    def test_each_sample_remains_one_attempt_no_retry_and_original_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = TerminalBenchBackend(profile(root))
            for sample in plans(root):
                config = backend.build_config(sample, root / str(sample.identity.sample_index))
                self.assertEqual((config.n_attempts, config.n_concurrent_trials, config.retry.max_retries), (1, 1, 0))
                self.assertFalse(config.verifier.disable)
                self.assertIsNone(config.verifier.import_path)
                self.assertEqual(config.agents[0].kwargs["temperature"], 1)
                self.assertEqual(config.agents[0].kwargs["command_seconds"], 30)
                self.assertEqual(config.agents[0].kwargs["generation_seconds"], 360)


@unittest.skipUnless(importlib.util.find_spec("harbor"), "optional Harbor contract environment unavailable")
class HarborTrainingResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_training_marker_requires_matching_durable_trace_and_preserves_original_state(self):
        for valid in (True, False):
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sample = plans(root)[0]
                settings = profile(root)
                trace = TurnTrace(1, "d" * 32, [1, 2], [3], [-.5], PolicyRef.from_dict(settings["training_policy"]))
                class Transport:
                    training_traces = [trace]
                    def with_training(self, contract): return self
                    def harbor_relay(self, output): return None
                backend = TerminalBenchBackend(settings, Transport())
                backend._admitted[_identity_key(sample.identity)] = {}
                native = root / ("episode-" + _identity_key(sample.identity)) / "jobs/episode/trial"
                class Job:
                    create = AsyncMock()
                    def add_hook(self, *_): pass
                    async def run(self):
                        native.mkdir(parents=True)
                        trial(native, name=sample.task.task_id)
                        agent = controller.read_json(native / "agent/file-relay/agent-result.json")
                        agent["turns"] = 1
                        (native / "agent/file-relay/agent-result.json").write_text(json.dumps(agent))
                        controller.write_new(native / "agent/file-relay" / (trace.request_id + ".state.json"),
                            {"id": trace.request_id, "state": "response_received"})
                        saved = trace.to_dict()
                        if not valid:
                            saved["action_token_ids"] = [4]
                        controller.write_new(native / "agent/file-relay" / (trace.request_id + ".training.json"), saved)
                Job.create.return_value = Job()
                module = ModuleType("harbor.job")
                module.Job = Job
                with patch.dict("sys.modules", {"harbor.job": module}), patch.object(backend, "_checked_plan", return_value={}):
                    result = await backend.execute(sample, CancellationToken())
                self.assertEqual(result.score_eligible, valid)
                state = result.request_state["requests"][trace.request_id]
                self.assertEqual(state.get("training_evidence"), "validated" if valid else None)
                original = controller.read_json(native / "agent/file-relay" / (trace.request_id + ".state.json"))
                self.assertNotIn("training_evidence", original)

    async def test_missing_token_evidence_preserves_verifier_reward_but_blocks_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = plans(root)[0]
            class Transport:
                training_traces = []
                def with_training(self, contract): return self
                def harbor_relay(self, output): return None
            backend = TerminalBenchBackend(profile(root), Transport())
            backend._admitted[_identity_key(sample.identity)] = {}
            class Job:
                create = AsyncMock()
                def add_hook(self, *_): pass
                async def run(self):
                    native = root / ("episode-" + _identity_key(sample.identity)) / "jobs/episode/trial"
                    native.mkdir(parents=True)
                    trial(native, name=sample.task.task_id)
                    agent = controller.read_json(native / "agent/file-relay/agent-result.json")
                    agent["turns"] = 1
                    (native / "agent/file-relay/agent-result.json").write_text(json.dumps(agent))
                    controller.write_new(native / "agent/file-relay" / ("d" * 32 + ".state.json"),
                        {"id": "d" * 32, "state": "response_received"})
            Job.create.return_value = Job()
            module = ModuleType("harbor.job")
            module.Job = Job
            with patch.dict("sys.modules", {"harbor.job": module}), patch.object(backend, "_checked_plan", return_value={}):
                result = await backend.execute(sample, CancellationToken())
            self.assertEqual(result.native_outcome["official_reward"], 1)
            self.assertFalse(result.score_eligible)
            self.assertEqual(result.execution_state, "infrastructure_error")
            self.assertEqual(result.cleanup_state, "verified")
            self.assertTrue(backend._stopped)
