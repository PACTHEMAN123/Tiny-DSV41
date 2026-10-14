"""Private worker setup is tested locally without exec, Harbor, or a daemon."""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from rollout_manager.backends.terminal_bench import TerminalBenchBackend
from rollout_manager.execution_environment import (
    MARKER, prepare_worker_environment, validate_worker_environment,
)


class WorkerEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runtime = {"executor_root": str(self.root), "output_dir": str(self.root / "new-run/terminal-bench")}
        self.sha = "a" * 64

    def validation(self, env, runtime=None, sha=None):
        with patch.dict(os.environ, env, clear=True), patch.object(sys, "dont_write_bytecode", True), \
             patch.object(tempfile, "tempdir", None):
            return validate_worker_environment(runtime or self.runtime, sha or self.sha)

    def test_fresh_child_is_filtered_private_and_does_not_modify_parent(self):
        parent = {"HOME": "/original-home", "TMPDIR": "/old-tmp", "PYTHONPATH": "/old-code",
                  "PIP_CONFIG_FILE": "/old-pip", "HARBOR_TELEMETRY": "1", "PATH": "/bin"}
        with patch.dict(os.environ, parent, clear=True):
            env = prepare_worker_environment(self.runtime, self.sha)
            self.assertEqual(dict(os.environ), parent)
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PIP_CONFIG_FILE", env)
        self.assertEqual(env["HARBOR_TELEMETRY"], "0")
        self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertFalse(Path(self.runtime["output_dir"]).parent.exists())
        worker = Path(env[MARKER])
        self.assertEqual(worker, self.root / "new-run-worker")
        self.assertEqual({p.name for p in worker.iterdir()}, {"home", "cache", "tmp", "worker-environment.json"})
        self.validation(env)
        with self.assertRaises(FileExistsError):
            prepare_worker_environment(self.runtime, self.sha)

    def test_marker_telemetry_and_configuration_binding_are_enforced(self):
        env = prepare_worker_environment(self.runtime, self.sha)
        for key in (MARKER, "HOME", "TMPDIR", "HARBOR_TELEMETRY"):
            changed = dict(env)
            changed.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validation(changed)
        with self.assertRaises(ValueError):
            self.validation(env, sha="b" * 64)
        changed = dict(env, PYTHONPATH="/historical-source")
        with self.assertRaises(ValueError):
            self.validation(changed)
        receipt = Path(env[MARKER]) / "worker-environment.json"
        binding = json.loads(receipt.read_text())
        binding["output_dir"] = str(self.root / "different")
        receipt.write_text(json.dumps(binding))
        with self.assertRaises(ValueError):
            self.validation(env)

    def test_private_paths_and_cached_temporary_directory_are_enforced(self):
        env = prepare_worker_environment(self.runtime, self.sha)
        Path(env["TMPDIR"]).chmod(0o755)
        with self.assertRaises(ValueError):
            self.validation(env)
        Path(env["TMPDIR"]).chmod(0o700)
        with patch.dict(os.environ, env, clear=True), patch.object(sys, "dont_write_bytecode", True), \
             patch.object(tempfile, "tempdir", "/unscoped-tmp"), self.assertRaises(ValueError):
            validate_worker_environment(self.runtime, self.sha)
        with self.assertRaises(ValueError):
            prepare_worker_environment({**self.runtime, "worker_root": str(self.root.parent / "outside")}, self.sha)

    def test_backend_refuses_unprepared_environment_before_harbor_import(self):
        runtime = {**self.runtime, "relay_root": str(self.root / "relay"),
            "model_guard_evidence": str(self.root / "guard.json"), "payload_dir": str(self.root / "payload"),
            "task_copy_dir": str(self.root / "tasks"), "task_provenance_sha256": "c" * 64,
            "payload_sha256": "d" * 64, "configuration_sha256": self.sha,
            "bootstrap_qualifications": {"merge-diff-arc-agi-task": str(self.root / "qualification")}}
        with patch.dict(os.environ, {}, clear=True), \
             patch("rollout_manager.backends.terminal_bench.metadata.version") as version, self.assertRaises(ValueError):
            asyncio.run(TerminalBenchBackend(runtime).start())
        version.assert_not_called()
        self.assertFalse(Path(self.runtime["output_dir"]).exists())
