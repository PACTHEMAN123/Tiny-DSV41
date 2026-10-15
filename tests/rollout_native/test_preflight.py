import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rollout_manager import preflight as driver


class PreflightTests(unittest.TestCase):
    def invoke(self, root, inventory, owners=""):
        with patch("rollout_manager.telemetry.snapshot", return_value={"gpu_error": "telemetry timeout"}), \
                patch("rollout_manager.telemetry.read", return_value="unavailable"), \
                patch.object(driver.os, "sched_getaffinity", return_value={0}, create=True), \
                patch.object(Path, "read_text", return_value=""), \
                patch.object(driver.subprocess, "check_output", side_effect=[owners, "storage", inventory]):
            return driver.preflight(root)

    def test_inventory_survives_optional_telemetry_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(self.invoke(root, "0, GPU-a\n1, GPU-b\n"), ["GPU-a", "GPU-b"])
            evidence = json.loads((root / "preflight.json").read_text())
            self.assertEqual(evidence["gpu_error"], "telemetry timeout")
            self.assertIn("GPU-a", evidence["gpu_inventory_csv"])

    def test_required_inventory_timeout_blocks_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "not admitted"):
                self.invoke(root, subprocess.TimeoutExpired("nvidia-smi", 15))
            self.assertIn("gpu_inventory_error", json.loads((root / "preflight.json").read_text()))

    def test_existing_owner_still_blocks_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "owners must be reconciled"):
                self.invoke(Path(directory), "0, GPU-a\n", owners="123, GPU-a, 1024\n")
