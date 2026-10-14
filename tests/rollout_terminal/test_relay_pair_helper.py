"""Launch plans and real-readiness guard derivation; no live browser/GPU claims."""
import argparse
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rollout_manager.terminal_bench import admission, controller
from .test_native_profile import MODEL, proof, readiness

REPO = Path(__file__).absolute().parents[2]
spec = importlib.util.spec_from_file_location(
    "terminalbench_relay_pair_test_entrypoint", REPO / "examples/rl/terminalbench_relay_pair.py")
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


class RelayPairHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name).resolve()
        self.root = self.project / "new-native-relay"
        python = self.project / "venv/bin/python"
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        self.args = argparse.Namespace(
            role="cpu", source_root=str(REPO), python=str(python),
            public_base="https://cpu.example/webide/33349/proxy/18092/",
            peer_origin="https://gpu.example", proxy_origin=None, model=MODEL,
            max_seconds=30000, native_controller_receipt=None,
            native_controller_token_file=None, execute=False,
            output=str(self.project / "new-native-guard.json"),
            lease_file=str(self.project / "cpu-policy-lease.json"))

    def plan(self):
        with patch.object(helper.socket, "socket") as socket:
            value = helper.launch_plan(self.args, self.project, self.root)
            socket.return_value.bind.assert_called_once_with(("127.0.0.1", 18092))
            return value

    def test_independent_cpu_entrypoint_uses_explicit_wheel_source(self):
        source = self.project / "worker/site"
        package = source / "rollout_manager/__init__.py"
        package.parent.mkdir(parents=True)
        package.write_text("")
        with patch.dict(sys.modules, {"rollout_manager": SimpleNamespace(__file__=str(package))}), \
                patch.object(sys, "path", list(sys.path)):
            self.assertEqual(helper.configure_source(str(source)), source)
            self.assertEqual(sys.path[0], str(source))
        with patch.object(sys, "path", list(sys.path)):
            with self.assertRaisesRegex(ValueError, "differs from"):
                helper.configure_source(str(source))

    def test_cpu_plan_keeps_venv_symlink_and_fixed_native_capacity(self):
        source, context, log, receipt, command = self.plan()
        self.assertEqual(source, REPO)
        self.assertEqual(command[0], self.args.python)
        self.assertNotEqual(command[0], str(Path(self.args.python).resolve()))
        self.assertEqual(command[command.index("--port") + 1], "18092")
        self.assertEqual(command[command.index("--serving-profile") + 1], helper.PROFILE)
        self.assertEqual(command[command.index("--max-inflight") + 1], "1")
        self.assertEqual(command[command.index("--generation-seconds") + 1], "360")
        self.assertEqual(command[command.index("--max-seconds") + 1], "30000")
        self.assertFalse(self.root.exists())
        self.assertTrue(all(self.project in path.parents for path in (context, log, receipt)))

    def test_exact_observed_private_http_gateway_origin_is_explicit(self):
        self.args.proxy_origin = "http://verified-private-gateway:18093"
        command = self.plan()[-1]
        self.assertEqual(command[command.index("--proxy-origin") + 1], self.args.proxy_origin)
        for invalid in ("http://gateway/", "http://user:private@gateway", "http://gateway?token=private", "ws://gateway"):
            with self.subTest(origin=invalid), patch.object(helper.socket, "socket"):
                self.args.proxy_origin = invalid
                with self.assertRaises(ValueError):
                    helper.launch_plan(self.args, self.project, self.root)

    def test_explicit_cpu_port_binds_matching_proxy_and_launch_command(self):
        self.args.port = 18096
        self.args.public_base = "https://cpu.example/webide/33349/proxy/18096/"
        with patch.object(helper.socket, "socket") as socket:
            command = helper.launch_plan(self.args, self.project, self.root)[-1]
        socket.return_value.bind.assert_called_once_with(("127.0.0.1", 18096))
        self.assertEqual(command[command.index("--port") + 1], "18096")
        for value in (True, 0, 1023, 65536):
            self.args.port = value
            with self.subTest(port=value), self.assertRaises(ValueError):
                helper.launch_plan(self.args, self.project, self.root)
        self.args.port = 18095
        with self.assertRaisesRegex(ValueError, "selected relay port"):
            helper.launch_plan(self.args, self.project, self.root)

    def test_existing_launch_and_provider_fragments_require_inspection(self):
        self.root.mkdir()
        with self.assertRaisesRegex(ValueError, "inspection"):
            self.plan()
        self.root.rmdir()
        for field, value in (
                ("public_base", "https://cpu.example/webide/33349/proxy/18092/#token=private"),
                ("public_base", "https://cpu.example/webide/33349/proxy/18090/"),
                ("peer_origin", "https://gpu.example/proxy/18093/"),
                ("proxy_origin", "https://gateway.example/path")):
            with self.subTest(field=field, value=value):
                args = copy.copy(self.args)
                setattr(args, field, value)
                with self.assertRaises(ValueError), patch.object(helper.socket, "socket"):
                    helper.launch_plan(args, self.project, self.root)

    def test_guard_derives_schema2_only_from_actual_readiness_and_expected_lease(self):
        value, ready = proof(), readiness(proof())
        lease = {"kind": "terminalbench-policy-lease", "served_model": MODEL, "policy": value["policy"]}
        config = {"role": "cpu", "serving_profile": helper.PROFILE, "model": MODEL,
                  "boot_id": ready["cpu"]["boot_id"]}
        with patch.object(helper, "private_json", return_value=lease), \
                patch.object(helper, "local_api", return_value=config) as local, \
                patch.object(admission, "check_readiness", return_value=ready) as check, \
                patch.object(helper.time, "time", return_value=1000):
            plan = helper.snapshot_guard(self.args, self.project, self.root)
            self.assertTrue(plan["actual_browser_roundtrip_validated"])
            self.assertFalse(Path(self.args.output).exists())
            self.assertEqual(check.call_args.kwargs["expected_policy"], value["policy"])
            local.assert_called_with(self.root, 18092, "/api/config")
            self.args.execute = True
            helper.snapshot_guard(self.args, self.project, self.root)
        evidence = controller.read_json(Path(self.args.output))
        self.assertEqual(evidence["schema_version"], 2)
        self.assertEqual(evidence["native_controller"], value)
        self.assertEqual(evidence["cpu_relay_boot_id"], ready["cpu"]["boot_id"])
        self.assertEqual(Path(self.args.output).stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, "new file"):
            helper.snapshot_guard(self.args, self.project, self.root)

    def test_explicit_vllm_launch_uses_distinct_controller_flags(self):
        from rollout_manager.terminal_bench.browser_relay import relay_server
        self.args.serving_profile = "vllm-training-v1"
        self.args.role = "gpu"
        self.args.port = 18095
        self.args.public_base = "https://gpu.example/webide/current/proxy/18095/"
        self.args.training_controller_receipt = str(self.project / "actual-startup.json")
        self.args.training_controller_token_file = str(self.project / "private-token.txt")
        with patch.object(helper.socket, "socket"), patch.object(relay_server, "owned_training_controller") as owned:
            command = helper.launch_plan(self.args, self.project, self.root)[-1]
        self.assertEqual(command[command.index("--serving-profile") + 1], "vllm-training-v1")
        self.assertEqual(command[command.index("--port") + 1], "18095")
        self.assertIn("--training-controller-receipt", command)
        self.assertIn("--training-controller-token-file", command)
        self.assertNotIn("--native-controller-receipt", command)
        self.assertEqual(owned.call_args.kwargs["serving_profile"], "vllm-training-v1")
        self.args.native_controller_receipt = str(self.project / "wrong-native.json")
        with patch.object(helper.socket, "socket"), self.assertRaises(ValueError):
            helper.launch_plan(self.args, self.project, self.root)

    def test_vllm_guard_retains_actual_engine_proof_and_loaded_policy(self):
        from .test_browser_relay_vllm_training import vllm_proof
        value = vllm_proof()
        ready = readiness(value)
        for role in ("cpu", "gpu"):
            ready[role]["serving_profile"] = "vllm-training-v1"
        ready["gpu"]["training_controller"] = ready["gpu"].pop("native_controller")
        lease = {"kind": "terminalbench-policy-lease", "served_model": MODEL, "policy": value["policy"]}
        config = {"role": "cpu", "serving_profile": "vllm-training-v1", "model": MODEL,
                  "boot_id": ready["cpu"]["boot_id"]}
        self.args.serving_profile = "vllm-training-v1"
        self.args.execute = True
        with patch.object(helper, "private_json", return_value=lease), \
                patch.object(helper, "local_api", return_value=config), \
                patch.object(admission, "check_readiness", return_value=ready) as check, \
                patch.object(helper.time, "time", return_value=value["observed_unix"]):
            result = helper.snapshot_guard(self.args, self.project, self.root)
        self.assertEqual(check.call_args.kwargs["serving_profile"], "vllm-training-v1")
        evidence = controller.read_json(Path(self.args.output))
        self.assertNotIn("native_controller", evidence)
        self.assertEqual(evidence["training_controller"]["vllm_engine"], value["vllm_engine"])
        self.assertEqual(result["policy_ack"], value["policy_ack"])

    def test_stale_or_different_boot_readiness_never_creates_guard(self):
        value = proof()
        lease = {"kind": "terminalbench-policy-lease", "served_model": MODEL, "policy": value["policy"]}
        self.args.execute = True
        with patch.object(helper, "private_json", return_value=lease), \
                patch.object(helper, "local_api", return_value={}), \
                patch.object(admission, "check_readiness", side_effect=admission.episode.EpisodeError("stale")):
            with self.assertRaisesRegex(admission.episode.EpisodeError, "stale"):
                helper.snapshot_guard(self.args, self.project, self.root)
        with patch.object(helper, "private_json", return_value=lease), \
                patch.object(helper, "local_api", return_value={"role": "cpu", "serving_profile": helper.PROFILE,
                    "model": MODEL, "boot_id": "wrong-boot"}), \
                patch.object(admission, "check_readiness", return_value=readiness(value)):
            with self.assertRaisesRegex(ValueError, "identity"):
                helper.snapshot_guard(self.args, self.project, self.root)
        self.assertFalse(Path(self.args.output).exists())


if __name__ == "__main__":
    unittest.main()
