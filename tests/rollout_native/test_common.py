import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rollout_manager.common import digest, save_json, wire


class EvidenceIdentityTests(unittest.TestCase):
    def test_digest_preserves_original_canonical_source_encoding(self):
        value = {"model_new.py": "class ModelNew: pass\n", "kernel.cu": "// source\n"}
        original = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.assertEqual(digest(value), hashlib.sha256(original).hexdigest())
        self.assertEqual(digest(dict(reversed(list(value.items())))), digest(value))

    def test_invalid_identity_data_fails_closed(self):
        for value in ({1: "x"}, float("nan"), object()):
            with self.assertRaises(ValueError):
                wire(value)
        with self.assertRaises(ValueError):
            wire("data", limit=1)

    def test_atomic_record_replaces_only_the_requested_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"nested/result.json"
            save_json(path, {"state": "uncertain"})
            save_json(path, {"state": "finished", "correct": False})
            self.assertEqual(json.loads(path.read_text()), {"state": "finished", "correct": False})
            self.assertFalse(path.with_suffix(".json.tmp").exists())
