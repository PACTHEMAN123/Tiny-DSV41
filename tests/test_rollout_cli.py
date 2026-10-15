import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rollout_manager.cli import main, prepare
from rollout_manager.datasets import TERMINAL_REVISION


class OfflineCliTests(unittest.TestCase):
    def config(self, root, benchmark="kernelbench"):
        code = "class Model: pass\n"
        case = {"task": {"task_id": benchmark + "/level_1/0", "dataset": benchmark,
                "revision": "ca1464e5d4723fd14f87b8e68306aaa1fe66b81b",
                "code": code, "source_sha256": hashlib.sha256(code.encode()).hexdigest()},
                "request": {"model": "pinned-model", "messages": []}, "files": {}}
        (root / "cases.json").write_text(json.dumps([case]))
        config = {"schema_version": 1, "run_id": "offline-test", "benchmark": benchmark,
                  "backend": "native-cuda", "transport": "tls-rpc", "output_dir": "new-output",
                  "native": {"mode": "inference", "cases": "cases.json", "assets": "external-assets",
                    "credentials": "external-credentials", "catalog": "external-approvals.json",
                    "url": "https://relay.example", "concurrency": [8]}}
        path = root / "config.json"; path.write_text(json.dumps(config)); return path

    def test_validate_and_plan_read_only_without_transport(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); config = self.config(root)
            args = ["--benchmark", "kernelbench", "--backend", "native-cuda",
                    "--transport", "tls-rpc", "--config", str(config)]
            with patch("socket.socket", side_effect=AssertionError("network forbidden")):
                for action in ("validate", "plan", "inspect"):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        self.assertEqual(main([action, *args]), 0)
                    value = json.loads(out.getvalue())
                    if action == "plan":
                        self.assertEqual(len(value["episodes"]), 8)
                        self.assertEqual({p["identity"]["sample_index"] for p in value["episodes"]}, set(range(8)))
                    if action == "inspect":
                        self.assertNotIn("native_case", value["tasks"][0]["payload"])
            self.assertFalse((root / "new-output").exists())

    def test_explicit_combination_and_config_must_agree(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = self.config(root)
            with self.assertRaisesRegex(ValueError, "combination"):
                prepare(path, "kernelbench", "harbor-pouch", "browser-filerelay")
            config = json.loads(path.read_text()); config["transport"] = "browser-filerelay"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "differs"):
                prepare(path, "kernelbench", "native-cuda", "tls-rpc")

    def test_terminal_catalog_inspection_does_not_import_harbor(self):
        command = [sys.executable, "-B", "-c",
            "import sys; from rollout_manager.cli import main; "
            "main(['inspect','--benchmark','terminal-bench','--backend','harbor-pouch','--transport','browser-filerelay']); "
            "assert not any(k == 'harbor' or k.startswith('harbor.') for k in sys.modules)"]
        run = subprocess.run(command, capture_output=True, text=True, check=True)
        tasks = json.loads(run.stdout)
        self.assertEqual(len(tasks), 89)
        self.assertEqual({t["provenance"]["revision"] for t in tasks}, {TERMINAL_REVISION})

    def test_offline_validation_rejects_unestablished_rpc_routes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = self.config(root)
            config = json.loads(path.read_text())
            for url in ("http://relay.example", "https://user:password@relay.example", "https://relay.example/generate", "https://relay.example?token=secret"):
                config["native"]["url"] = url; path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, "HTTPS origin"):
                    prepare(path, "kernelbench", "native-cuda", "tls-rpc")

    def test_frozen_inference_cannot_be_mislabeled_as_fresh_coupled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); path = self.config(root)
            config = json.loads(path.read_text())
            config["native"].update(mode="coupled", frozen_rollouts="external-frozen-manifest.json")
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "tools-only"):
                prepare(path, "kernelbench", "native-cuda", "tls-rpc")

    def test_terminal_configuration_paths_remain_on_explicit_cpu_executor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); executor = root / "cpu-executor"
            config = {"schema_version": 1, "run_id": "tb-path-test", "benchmark": "terminal-bench",
                "backend": "harbor-pouch", "transport": "browser-filerelay",
                "output_dir": str(executor / "new-run"), "executor": "cpu",
                "capacity": {"cpu": 2, "memory_bytes": 6*1024**3, "storage_bytes": 20*1024**3},
                "terminal_bench": {"executor_root": str(executor), "relay_root": str(executor/"relay"),
                    "model_guard_evidence": str(executor/"guard.json"), "payload_dir": str(executor/"payload"),
                    "task_copy_dir": str(executor/"tasks"), "task_provenance_sha256": "pin", "payload_sha256": "pin",
                    "bootstrap_qualifications": {"merge-diff-arc-agi-task": str(executor/"merge-proof"),
                                                 "sparql-university": str(executor/"sparql-proof")}}}
            path = root / "config.json"; path.write_text(json.dumps(config))
            prepared = prepare(path, "terminal-bench", "harbor-pouch", "browser-filerelay")
            self.assertEqual(len(prepared[4]), 2)
            arguments = ["run", "--benchmark", "terminal-bench", "--backend", "harbor-pouch",
                         "--transport", "browser-filerelay", "--config", str(path)]
            with patch.dict(os.environ, {}, clear=True), \
                 patch("rollout_manager.execution_environment.prepare_worker_environment", return_value={"child": "private"}) as prepare_child, \
                 patch("rollout_manager.cli.os.umask") as umask, \
                 patch("rollout_manager.cli.os.execve", side_effect=SystemExit(47)) as execute_child, \
                 patch("rollout_manager.cli.EventStore", side_effect=AssertionError("no coordinator before reexec")):
                with self.assertRaises(SystemExit) as stopped:
                    main(arguments)
            self.assertEqual(stopped.exception.code, 47)
            prepare_child.assert_called_once()
            umask.assert_called_once_with(0o077)
            self.assertEqual(execute_child.call_args.args[1], [sys.executable, "-B", "-m", "rollout_manager", *arguments])
            config["terminal_bench"]["relay_root"] = str(root/"different-executor-relay")
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "CPU executor root"):
                prepare(path, "terminal-bench", "harbor-pouch", "browser-filerelay")


if __name__ == "__main__":
    unittest.main()
