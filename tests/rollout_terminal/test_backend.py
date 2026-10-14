"""Offline bridge contracts: synthetic jobs never call a model or Pouch daemon."""
import asyncio
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import hashlib
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from rollout_manager.backends.terminal_bench import TerminalBenchBackend, _identity_key
from rollout_manager.contracts import CancellationToken, EpisodeIdentity, EpisodePlan, ResourceRequirements, TaskSpec
from rollout_manager.terminal_bench import admission, controller
if __package__:
    from .test_classification import trial
else:
    from test_classification import trial


def plan(root):
    task = TaskSpec("terminal-bench", "merge-diff-arc-agi-task", "a" * 64,
        {"task_path": str(root / "tasks/merge-diff-arc-agi-task")},
        ResourceRequirements("cpu-orchestrator", 1, 4096 * 1024**2, 10240 * 1024**2))
    return EpisodePlan(EpisodeIdentity("run", "terminal-bench", task.task_id, "group", 0), task)


class GroupAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_readiness_gate_preserves_generation_and_uncertainty_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = {"model": controller.MODEL, "boot_id": "boot", "max_inflight": 64,
                    "expires_unix": 11000, "generation_seconds": 360}
            ready = {"schema_version": 1, "protocol": 2, "paired_unix": 999, "valid_until_unix": 1010,
                "cpu": {**base, "pid": 1, "pid_start_ticks": 17},
                "gpu": {**base, "uncertain": 0, "model_max_num_seqs": 64, "model_capacity_source": "vllm_server_info"}}
            controller.write_new(root / "readiness.json", ready)
            checked = admission.check_readiness(root, controller.MODEL, 2, 3060, now=1000,
                identity_reader=lambda pid: (17, "S"), required_generation_seconds=360)
            self.assertEqual(checked["gpu"]["model_max_num_seqs"], 64)
            ready["gpu"]["uncertain"] = 1
            (root / "readiness.json").write_text(json.dumps(ready))
            with self.assertRaises(admission.episode.EpisodeError):
                admission.check_readiness(root, controller.MODEL, 2, 3060, now=1000,
                    identity_reader=lambda pid: (17, "S"), required_generation_seconds=360)

    async def test_one_shared_gate_and_terminal_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = TerminalBenchBackend({"output_dir": str(root), "relay_root": str(root),
                                           "model_guard_evidence": str(root / "guard.json")})
            backend._started = True
            episode = plan(root)
            row = {"required_service_seconds": 3060}
            with patch.object(backend, "_checked_plan", return_value=row), \
                 patch.object(admission, "check_empty_spool") as empty, \
                 patch.object(admission, "check_readiness", return_value={"gpu": {}}) as ready, \
                 patch.object(controller, "read_json", return_value={}), \
                 patch.object(controller, "guard_gate", return_value={}) as guard:
                receipt = await backend.admit_group([episode])
                self.assertEqual(receipt["resources_owned_by"], "shared_coordinator")
                ready.assert_called_once_with(root, controller.MODEL, 1, 3060, required_generation_seconds=360)
                self.assertEqual(guard.call_args.args[2], 3060)
                with self.assertRaises(ValueError):
                    await backend.admit_group([episode])
                closed = await backend.finish_group([episode], [])
                self.assertTrue(closed["spool_terminal"])
                self.assertEqual(empty.call_count, 2)

    async def test_unknown_spool_stops_admission(self):
        backend = TerminalBenchBackend({"relay_root": "/unused"})
        backend._admitted = {"id": {}}
        with patch.object(admission, "check_empty_spool", side_effect=ValueError("uncertain")):
            result = await backend.finish_group([], [])
        self.assertFalse(result["spool_terminal"])
        self.assertTrue(backend._stopped)

    async def test_local_cancellation_drains_harbor_cleanup(self):
        token = CancellationToken()
        entered, cleanup = asyncio.Event(), asyncio.Event()
        class Job:
            async def run(self):
                try:
                    entered.set()
                    await asyncio.Event().wait()
                finally:
                    await asyncio.sleep(0.01)
                    cleanup.set()
        backend = TerminalBenchBackend({})
        running = asyncio.create_task(backend._run_job(Job(), token))
        await entered.wait()
        token.cancel("test")
        await running
        self.assertTrue(cleanup.is_set())
        self.assertFalse(backend._jobs)


@unittest.skipUnless(importlib.util.find_spec("harbor"), "optional Harbor contract environment unavailable")
class HarborBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_original_bootstrap_evidence_is_required_without_rerun(self):
        from rollout_manager.terminal_bench.task_assets import qualification_proof
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve() / "baseline"
            trial(root, name="merge-diff-arc-agi-task", failed=True)
            raw = (root / "verifier/ctrf.json").read_bytes()
            evidence = {"task_id": "merge-diff-arc-agi-task", "task_provenance_sha256": "a" * 64,
                "payload_manifest_sha256": "b" * 64, "model_evaluation": False,
                "agent_executed": False, "oracle_executed": False, "attempts": 1, "retries": 0,
                "bootstrap_check_status": "native_tools_and_uvx_plugin_passed", "cleanup_verified": True,
                "original_verifier_terminal": {"terminal_proven_by_provider": True, "return_code": 1},
                "verifier_result": {"rewards": {"reward": 0}},
                "ctrf": {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}}
            controller.write_new(root / "bootstrap-qualification.json", evidence)
            proof = qualification_proof(root, evidence["task_id"], "a" * 64, "b" * 64)
            self.assertEqual(proof["original_baseline_reward"], 0)
            self.assertIsNone(proof["model_score"])
            with self.assertRaises(ValueError):
                qualification_proof(root, evidence["task_id"], "c" * 64, "b" * 64)
            evidence["cleanup_verified"] = False
            (root / "bootstrap-qualification.json").write_text(json.dumps(evidence))
            with self.assertRaises(ValueError):
                qualification_proof(root, evidence["task_id"], "a" * 64, "b" * 64)

    async def test_stock_job_config_preserves_one_attempt_original_verifier_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            backend = TerminalBenchBackend({"relay_root": str(root), "payload_dir": str(root)})
            config = backend.build_config(plan(root), root)
            self.assertEqual((config.n_attempts, config.retry.max_retries, config.n_concurrent_trials), (1, 0, 1))
            self.assertEqual(config.timeout_multiplier, 1)
            self.assertFalse(config.verifier.disable)
            self.assertIsNone(config.verifier.import_path)
            self.assertIsNone(config.verifier.override_timeout_sec)
            options = config.agents[0].kwargs
            self.assertEqual((options["generation_seconds"], options["max_turns"], options["command_seconds"], options["max_tokens"]), (360, 128, 30, 2048))
            self.assertFalse(config.environment.kwargs["allow_public_network"])
            self.assertEqual(len(config.environment.kwargs["image_id_by_reference"]), 89)

    async def test_real_episode_dispatch_classification_without_second_scheduler(self):
        from rollout_manager.terminal_bench.harbor_filerelay_agent import relay_factory
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            episode = plan(root)
            backend = TerminalBenchBackend({"output_dir": str(root), "relay_root": str(root), "payload_dir": str(root)})
            backend._admitted[_identity_key(episode.identity)] = {}
            class Job:
                create = AsyncMock()
                def add_hook(self, *_):
                    pass
                async def run(self):
                    native = root / ("episode-" + _identity_key(episode.identity)) / "jobs/episode/native-trial"
                    native.mkdir(parents=True)
                    trial(native, name=episode.task.task_id)
                    controller.write_new(native / "agent/file-relay" / ("d" * 32 + ".state.json"),
                        {"id": "d" * 32, "state": "response_received", "remote_cancellation_proven": False})
            Job.create.return_value = Job()
            module = ModuleType("harbor.job")
            module.Job = Job
            with patch.dict("sys.modules", {"harbor.job": module}), patch.object(backend, "_checked_plan", return_value={}):
                result = await backend.execute(episode, CancellationToken())
            Job.create.assert_awaited_once()
            self.assertEqual(result.identity, episode.identity)
            self.assertTrue(result.score_eligible)
            self.assertEqual(result.native_outcome["official_reward"], 1)
            self.assertEqual(result.cleanup_state, "verified")
            self.assertIsNone(relay_factory.get())
            self.assertFalse(result.request_state["remote_cancellation_proven"])
