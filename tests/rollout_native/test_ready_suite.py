import hashlib
import tempfile
import unittest
from pathlib import Path

from rollout_manager.analysis.select_ready_suite import model_dependency, select, stage


class ReadySuiteTests(unittest.TestCase):
    def test_named_dependency_and_explicit_deferral(self):
        independent = {"task": {"task_id": "plain", "code": "def get_inputs(): return []"}}
        source = "model_name = 'gpt2'\nconfig = AutoConfig.from_pretrained(model_name)\n"
        pretrained = {"task": {"task_id": "pretrained", "code": source}}
        self.assertEqual(model_dependency(source), "gpt2")
        selected, deferred = select([independent, pretrained], set())
        self.assertEqual(selected, [independent])
        self.assertEqual(deferred[0]["task_id"], "pretrained")
        self.assertEqual(select([independent, pretrained], {"gpt2"}), ([independent, pretrained], []))

    def test_dynamic_dependency_is_not_silently_included(self):
        with self.assertRaisesRegex(ValueError, "unresolved"):
            model_dependency("config = AutoConfig.from_pretrained(os.environ['MODEL'])")

    def test_mount_alias_is_staged_only_after_digest_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mounts/openai-community/gpt2/model.safetensors"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"weights")
            model = {"model": "gpt2", "revision": "pinned", "files": [{"name": source.name,
                     "bytes": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}]}
            result = stage(model, root / "assets", root / "mounts")
            self.assertTrue(result["ready"])
            cache = root / "assets/hf-home/hub/models--gpt2"
            self.assertEqual((cache / "snapshots/pinned/model.safetensors").read_bytes(), b"weights")
            self.assertEqual((cache / "refs/main").read_text(), "pinned")
            self.assertEqual(source.read_bytes(), b"weights")

    def test_incomplete_or_wrong_model_never_becomes_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "mounts/model/weights.bin"
            source.parent.mkdir(parents=True)
            model = {"model": "model", "revision": "pinned", "files": [{"name": source.name,
                     "bytes": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}]}
            source.write_bytes(b"partial!")
            self.assertFalse(stage(model, root / "assets", root / "mounts")["ready"])
            source.write_bytes(b"invalid")
            self.assertIn("digest mismatch", stage(model, root / "assets", root / "mounts")["reason"])
            cache = root / "assets/hf-home/hub/models--model"
            self.assertFalse((cache / "snapshots/pinned/weights.bin").exists())
            self.assertFalse((cache / "refs/main").exists())
            self.assertEqual(list((cache / "snapshots/pinned").iterdir()), [])
            source.write_bytes(b"weights")
            self.assertTrue(stage(model, root / "assets", root / "mounts")["ready"])


if __name__ == "__main__":
    unittest.main()
