"""Native serving identity gates; no GPU, browser pairing or Pouch is simulated as live."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rollout_manager.backends.terminal_bench import TerminalBenchBackend
from rollout_manager.generation_training import TrainingGenerationContract
from rollout_manager.training import PolicyRef
from rollout_manager.terminal_bench import admission, controller
from .test_backend import plan

MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"


def proof(now=1000):
    policy = PolicyRef(0, "a" * 64, "native-test", identities={
        "tokenizer_sha256": "b" * 64, "chat_template_sha256": "c" * 64})
    capabilities = {"schema_version": 1, "policy": policy.to_dict(),
        "logprob_convention": "sampled_distribution",
        **{name: True for name in TrainingGenerationContract.REQUIRED}}
    return {"schema_version": 1, "serving_profile": controller.NATIVE_PROFILE, "served_model": MODEL,
        "model_pid": 123, "model_pid_start_ticks": 456, "model_cmdline_sha256": "d" * 64,
        "generation_slots": 1, "started_unix": 900, "expires_unix": 12000, "observed_unix": now,
        "model_dir": "/mnt/fuse/Qwen/Qwen3-30B-A3B-Instruct-2507",
        "python_executable": "/workspace/venv/bin/python", "netns_inode": 77,
        "controller_state": "running", "active_generations": 0, "policy": policy.to_dict(),
        "policy_ack": {"version": 0, "digest": "a" * 64, "lineage": "native-test"},
        "capabilities": capabilities}


def readiness(value):
    base = {"serving_profile": controller.NATIVE_PROFILE, "model": MODEL, "boot_id": "boot",
        "max_inflight": 1, "expires_unix": 12000, "generation_seconds": 360}
    return {"schema_version": 1, "protocol": 2, "paired_unix": 999, "valid_until_unix": 1010,
        "cpu": {**base, "pid": 1, "pid_start_ticks": 17},
        "gpu": {**base, "uncertain": 0, "native_controller": value}}


class NativeProofTests(unittest.TestCase):
    def test_actual_loaded_policy_receipt_is_required(self):
        value = proof()
        self.assertIs(controller.validate_native_controller(value, MODEL, now=1000,
            required_seconds=3060, expected_policy=value["policy"]), value)
        for field, replacement in (("generation_slots", 64), ("controller_state", "paused"),
                ("observed_unix", 980), ("expires_unix", 2000), ("model_pid", True),
                ("python_executable", "/workspace/bin/vllm")):
            with self.subTest(field=field):
                changed = copy.deepcopy(value)
                changed[field] = replacement
                with self.assertRaises(ValueError):
                    controller.validate_native_controller(changed, MODEL, now=1000, required_seconds=3060)
        bad = copy.deepcopy(value)
        bad["capabilities"]["token_ids"] = False
        with self.assertRaises(ValueError):
            controller.validate_native_controller(bad, MODEL, now=1000)
        bad = copy.deepcopy(value)
        bad["policy_ack"]["digest"] = "e" * 64
        with self.assertRaises(ValueError):
            controller.validate_native_controller(bad, MODEL, now=1000)

    def test_guard_binds_current_policy_and_process_without_forging_vllm(self):
        live = proof()
        snapshot = copy.deepcopy(live)
        snapshot["observed_unix"] = 901
        evidence = {"schema_version": 2, "serving_profile": controller.NATIVE_PROFILE,
            "model": MODEL, "native_controller": snapshot}
        result = controller.guard_gate(evidence, readiness(live), 3060, 1000,
            serving_profile=controller.NATIVE_PROFILE, model=MODEL, expected_policy=live["policy"])
        self.assertEqual(result["generation_slots"], 1)
        self.assertEqual(result["policy_ack"], live["policy_ack"])
        for field in ("model_cmdline_sha256", "model_pid_start_ticks", "model_dir"):
            with self.subTest(field=field):
                wrong = copy.deepcopy(evidence)
                wrong["native_controller"][field] = "e" * 64 if field.endswith("sha256") else (
                    999 if field.endswith("ticks") else "/mnt/fuse/different")
                with self.assertRaises(ValueError):
                    controller.guard_gate(wrong, readiness(live), 3060, 1000,
                        serving_profile=controller.NATIVE_PROFILE, model=MODEL)
        other = copy.deepcopy(live["policy"])
        other["version"] = 1
        with self.assertRaises(ValueError):
            controller.guard_gate(evidence, readiness(live), 3060, 1000,
                serving_profile=controller.NATIVE_PROFILE, model=MODEL, expected_policy=other)

    def test_browser_readiness_preserves_quota_ttl_uncertainty_and_cpu_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready = readiness(proof())
            controller.write_new(root / "readiness.json", ready)
            def check(concurrency=1):
                return admission.check_readiness(root, MODEL, concurrency, 3060, now=1000,
                    identity_reader=lambda pid: (17, "S"), serving_profile=controller.NATIVE_PROFILE,
                    expected_policy=proof()["policy"], required_generation_seconds=360)
            self.assertEqual(check()["gpu"]["native_controller"]["generation_slots"], 1)
            with self.assertRaises(admission.episode.EpisodeError):
                check(2)
            mutations = [("gpu", "uncertain", 1), ("gpu", "model_max_num_seqs", 64),
                ("cpu", "serving_profile", "vllm-v1"), ("cpu", "pid_start_ticks", 18),
                ("gpu", "generation_seconds", 120)]
            for role, field, value in mutations:
                with self.subTest(field=field):
                    bad = copy.deepcopy(ready)
                    bad[role][field] = value
                    (root / "readiness.json").write_text(json.dumps(bad))
                    with self.assertRaises(admission.episode.EpisodeError):
                        check()


class NativeAdmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_native_backend_uses_the_registered_policy_lease(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            live = proof()
            backend = TerminalBenchBackend({"execution_profile": "training",
                "serving_profile": controller.NATIVE_PROFILE, "model": MODEL,
                "training_policy": live["policy"], "training_capabilities": live["capabilities"],
                "output_dir": str(root), "relay_root": str(root),
                "model_guard_evidence": str(root / "guard.json")})
            backend._started = True
            with patch.object(backend, "_checked_plan", return_value={"required_service_seconds": 3060}), \
                    patch.object(admission, "check_empty_spool"), \
                    patch.object(admission, "check_readiness", return_value=readiness(live)) as ready, \
                    patch.object(controller, "read_json", return_value={}), \
                    patch.object(controller, "guard_gate", return_value={}) as guard:
                await backend.admit_group([plan(root)])
            self.assertEqual(ready.call_args.kwargs["expected_policy"], live["policy"])
            self.assertEqual(guard.call_args.kwargs["model"], MODEL)

    async def test_qwen_cannot_silently_use_legacy_deepseek_guard(self):
        backend = TerminalBenchBackend({"model": MODEL, "relay_root": "/unused"})
        backend._started = True
        with patch.object(backend, "_checked_plan", return_value={"required_service_seconds": 3060}), \
                patch.object(admission, "check_empty_spool"):
            with self.assertRaisesRegex(ValueError, "recorded served model"):
                await backend.admit_group([plan(Path("/unused"))])


if __name__ == "__main__":
    unittest.main()
