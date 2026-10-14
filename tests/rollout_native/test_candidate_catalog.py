import json
import tempfile
import unittest
from pathlib import Path

from rollout_manager.common import digest
from rollout_manager.analysis.build_candidate_catalog import build_catalog


class CandidateCatalogTests(unittest.TestCase):
    def test_only_explicit_decisions_for_exact_available_sources_are_published(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "jobs/a").mkdir(parents=True)
            (root / "jobs/b").mkdir(parents=True)
            a, b = {"model_new.py": "a"}, {"model_new.py": "b"}
            (root / "jobs/a/candidate.json").write_text(json.dumps(a))
            (root / "jobs/b/candidate.json").write_text(json.dumps(b))
            path = root / "review.json"
            review = {"scope": "reviewed-development-pilot", "review_method": "read complete files",
                      "approved_sha256": [digest(a)]}
            path.write_text(json.dumps(review))
            catalog = build_catalog([root], [path])
            self.assertEqual(set(catalog), {digest(a)})
            self.assertEqual(catalog[digest(a)]["decision"], "approve")
            review["rejected"] = {digest(b): "candidate uses a forbidden fallback"}
            path.write_text(json.dumps(review))
            catalog = build_catalog([root], [path])
            self.assertEqual(catalog[digest(b)]["decision"], "reject")
            self.assertEqual(catalog[digest(a)]["evidence"][0]["file"], path.name)
            review["approved_sha256"].append(digest({"model_new.py": "unseen"}))
            path.write_text(json.dumps(review))
            with self.assertRaisesRegex(ValueError, "no matching source"):
                build_catalog([root], [path])

    def test_conflicting_approval_and_rejection_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "jobs/a").mkdir(parents=True)
            files = {"model_new.py": "a"}
            (root / "jobs/a/candidate.json").write_text(json.dumps(files))
            path = root / "review.json"
            path.write_text(json.dumps({"scope": "reviewed-development-pilot", "review_method": "read files",
                "approved_sha256": [digest(files)], "rejected": {digest(files): "reject"}}))
            with self.assertRaisesRegex(ValueError, "conflicting"):
                build_catalog([root], [path])
