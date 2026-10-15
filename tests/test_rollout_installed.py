"""Run after an offline wheel installation, with ROLLOUT_INSTALLED_PYTHON set."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("ROLLOUT_INSTALLED_PYTHON"), "requires independently installed wheel")
class InstalledPackageTests(unittest.TestCase):
    def run_outside(self, arguments):
        with tempfile.TemporaryDirectory() as root:
            return subprocess.run([os.environ["ROLLOUT_INSTALLED_PYTHON"], "-I", *arguments],
                                  cwd=root, capture_output=True, text=True, check=True).stdout

    def test_native_imports_without_harbor_or_training_dependencies(self):
        output = self.run_outside(["-c", "import json,importlib.util,rollout_manager; "
            "from rollout_manager.backends.native import NativeBackend; "
            "print(json.dumps({'path':rollout_manager.__file__,"
            "'harbor':importlib.util.find_spec('harbor') is not None,"
            "'torch':importlib.util.find_spec('torch') is not None}))"])
        value = json.loads(output)
        self.assertIn("site-packages", value["path"])
        self.assertFalse(value["harbor"])
        self.assertFalse(value["torch"])

    def test_installed_catalog_resources_and_offline_cli(self):
        tasks = json.loads(self.run_outside(["-m", "rollout_manager", "inspect", "--benchmark", "terminal-bench",
                        "--backend", "harbor-pouch", "--transport", "browser-filerelay"]))
        self.assertEqual(len(tasks), 89)
        self.assertTrue(all(t["resources"]["gpu_count"] == 0 for t in tasks))
        output = self.run_outside(["-m", "rollout_manager", "--help"])
        self.assertIn("validate", output)

    def test_legacy_native_cli_and_installed_child_help(self):
        output = self.run_outside(["-m", "rollout_manager", "native", "--help"])
        self.assertIn("--rollouts-per-task", output)
        self.assertIn("--frozen-rollouts", output)
        output = self.run_outside(["-m", "rollout_manager.adapters.harness_child", "--help"])
        self.assertIn("--upstream", output)
        entrypoint = Path(os.environ["ROLLOUT_INSTALLED_PYTHON"]).parent / "rollout-manager"
        with tempfile.TemporaryDirectory() as root:
            run = subprocess.run([str(entrypoint), "--help"], cwd=root, capture_output=True, text=True, check=True)
        self.assertIn("browser-filerelay", run.stdout)


if __name__ == "__main__":
    unittest.main()
