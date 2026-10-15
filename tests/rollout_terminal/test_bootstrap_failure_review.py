"""Explicit baseline review contract fixtures, never benchmark execution/reward."""
import copy
import hashlib
import json
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest.mock import patch

from rollout_manager.terminal_bench import task_assets, controller
from rollout_manager.backends.terminal_bench import TerminalBenchBackend


class BootstrapFailureReviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.project = Path(self.tmp.name).resolve()
        self.root = self.project / "preserved-baseline"
        (self.root / "verifier").mkdir(parents=True)
        tests = [{"name": name, "status": "failed", "raw_status": "call_failed",
                  "message": error} for name, error in task_assets._EXPECTED_SPARQL_FAILURES.items()]
        report = {"results": {"tool": {"name": "pytest"}, "summary": {
            "tests": 3, "passed": 0, "failed": 3, "skipped": 0, "pending": 0, "other": 0}, "tests": tests}}
        controller.write_new(self.root / "verifier/ctrf.json", report)
        stdout = "\n".join(name + " " + error for name, error in task_assets._EXPECTED_SPARQL_FAILURES.items())
        (self.root / "verifier/test-stdout.txt").write_text(stdout + "\nFileNotFoundError '/app/solution.sparql'\n")
        raw = (self.root / "verifier/ctrf.json").read_bytes()
        terminal = {"terminal_proven_by_provider": True, "return_code": 1, "command": "original verifier"}
        self.evidence = {"task_id": "sparql-university", "task_provenance_sha256": "a" * 64,
            "payload_manifest_sha256": "b" * 64, "model_evaluation": False, "agent_executed": False,
            "oracle_executed": False, "attempts": 1, "retries": 0, "bootstrap_check_status":
            "native_tools_and_uvx_plugin_passed", "cleanup_verified": True, "failure_type": None,
            "original_verifier_terminal": terminal, "verifier_result": {"rewards": {"reward": 0}},
            "ctrf": {"sha256": self.sha(raw), "bytes": len(raw)}}
        controller.write_new(self.root / "bootstrap-qualification.json", self.evidence)
        self.cleanup = {"provider": "pouch", "container_id": "c" * 64, "started": True, "stopped": True,
            "deleted": True, "cleanup_verified": True, "network_mode": "none",
            "public_network_explicitly_allowed": False}
        controller.write_new(self.root / "pouch-environment.json", self.cleanup)
        self.absence_path = self.project / "fresh-absence.json"
        self.absence = {"schema_version": 1, "kind": "pouch-container-absence-observation", "read_only": True,
            "hostname": socket.gethostname(), "container_id": self.cleanup["container_id"],
            "socket_path": "/var/run/pouchd.sock", "method": "GET",
            "path": "/containers/" + self.cleanup["container_id"] + "/json", "http_status": 404,
            "observed_unix": time.time() - 1}
        controller.write_new(self.absence_path, self.absence)
        # Fixture pins are local test data only. Production hardcodes the exact
        # independently reviewed historical bytes and rejects this fixture.
        self.pins = {"task_id": "sparql-university", "qualification_sha256": self.sha_file("bootstrap-qualification.json"),
            "ctrf_sha256": self.sha_file("verifier/ctrf.json"), "stdout_sha256": self.sha_file("verifier/test-stdout.txt")}
        self.review = {"schema_version": 1, "kind": "original-bootstrap-failure-review", "review_only": True,
            "model_score": None, **self.pins, "qualification_directory": str(self.root),
            "failure_scope": "expected-unsolved-task-output", "missing_artifact": "/app/solution.sparql",
            "expected_failures": dict(task_assets._EXPECTED_SPARQL_FAILURES), "original_verifier_terminal": terminal,
            "cleanup_sha256": self.sha_file("pouch-environment.json"),
            "container_absence_proof": str(self.absence_path),
            "container_absence_proof_sha256": self.sha(self.absence_path.read_bytes()), "reviewed_unix": time.time()}
        self.review_path = self.project / "fresh-review.json"
        controller.write_new(self.review_path, self.review)

    @staticmethod
    def sha(raw):
        return hashlib.sha256(raw).hexdigest()

    def sha_file(self, relative):
        return self.sha((self.root / relative).read_bytes())

    def qualified(self):
        with patch.object(task_assets, "_REVIEWED_SPARQL_BASELINE", self.pins):
            return task_assets.qualification_proof(self.root, "sparql-university", "a" * 64, "b" * 64,
                                                    failure_review=self.review_path)

    def test_explicit_review_preserves_reward_and_original_files(self):
        before = {str(path): self.sha(path.read_bytes()) for path in self.root.rglob("*") if path.is_file()}
        proof = self.qualified()
        self.assertEqual(proof["original_baseline_reward"], 0)
        self.assertIsNone(proof["model_score"])
        self.assertEqual(proof["bootstrap_failure_review_sha256"], self.sha(self.review_path.read_bytes()))
        self.assertEqual(proof["container_absence_proof_sha256"], self.sha(self.absence_path.read_bytes()))
        self.assertEqual(before, {str(path): self.sha(path.read_bytes()) for path in self.root.rglob("*") if path.is_file()})

    def test_default_legacy_gate_and_exact_production_pin_are_retained(self):
        with self.assertRaisesRegex(ValueError, "require original log review"):
            task_assets.qualification_proof(self.root, "sparql-university", "a" * 64, "b" * 64)
        with self.assertRaisesRegex(ValueError, "pinned original SPARQL"):
            task_assets.qualification_proof(self.root, "sparql-university", "a" * 64, "b" * 64,
                                            failure_review=self.review_path)
        self.assertEqual(task_assets._REVIEWED_SPARQL_BASELINE["qualification_sha256"],
            "92a20307c315d0c4e22955e638c111da8dff6d151fc73b29e7fc813134bc8caf")

    def test_review_tampering_other_task_or_invented_failure_is_rejected(self):
        changes = {"task_id": "merge-diff-arc-agi-task", "stdout_sha256": "f" * 64,
            "missing_artifact": "/app/other", "model_score": 0,
            "expected_failures": {"invented": "AssertionError"},
            "original_verifier_terminal": {"terminal_proven_by_provider": False}}
        for field, replacement in changes.items():
            with self.subTest(field=field):
                changed = {**self.review, field: replacement}
                self.review_path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    self.qualified()

    def test_original_stdout_cleanup_or_reward_change_is_rejected(self):
        for relative in ("verifier/test-stdout.txt", "pouch-environment.json", "bootstrap-qualification.json"):
            with self.subTest(relative=relative):
                path = self.root / relative
                raw = path.read_bytes()
                path.write_bytes(raw + b" ")
                with self.assertRaises(ValueError):
                    self.qualified()
                path.write_bytes(raw)

    def test_current_absence_identity_freshness_and_private_files_are_required(self):
        for field, replacement in (("http_status", 200), ("container_id", "d" * 64),
                ("observed_unix", time.time() - 43201), ("method", "DELETE"), ("hostname", "other-machine")):
            with self.subTest(field=field):
                changed = {**self.absence, field: replacement}
                self.absence_path.write_text(json.dumps(changed))
                self.review_path.write_text(json.dumps({**self.review,
                    "container_absence_proof_sha256": self.sha(self.absence_path.read_bytes())}))
                with self.assertRaises(ValueError):
                    self.qualified()
        self.review_path.chmod(0o644)
        with self.assertRaisesRegex(ValueError, "private regular file"):
            self.qualified()

    def test_optional_runtime_map_stays_below_explicit_executor(self):
        runtime = {"executor_root": str(self.project), "output_dir": str(self.project / "output"),
            "relay_root": str(self.project / "relay"), "model_guard_evidence": str(self.project / "guard"),
            "payload_dir": str(self.project / "payload"), "task_copy_dir": str(self.project / "tasks"),
            "task_provenance_sha256": "a" * 64, "payload_sha256": "b" * 64,
            "bootstrap_qualifications": {"sparql-university": str(self.root)}}
        TerminalBenchBackend.validate_config(runtime)
        TerminalBenchBackend.validate_config({**runtime,
            "bootstrap_failure_reviews": {"sparql-university": str(self.review_path)}})
        for reviews in ({"sparql-university": "/tmp/outside"}, {"unknown-task": str(self.review_path)}, None):
            with self.assertRaises(ValueError):
                TerminalBenchBackend.validate_config({**runtime, "bootstrap_failure_reviews": reviews})

    def merge_fixture(self):
        report = controller.read_json(self.root / "verifier/ctrf.json")
        report["results"]["tests"] = [{"name": "test_outputs.py::" + name,
            "status": "failed", "raw_status": "call_failed", "message": error}
            for name, error in task_assets._EXPECTED_MERGE_FAILURES.items()]
        report["results"]["summary"].update(tests=5, failed=5)
        (self.root / "verifier/ctrf.json").write_text(json.dumps(report))
        (self.root / "verifier/test-stdout.txt").write_text("\n".join(
            name + " " + error for name, error in task_assets._EXPECTED_MERGE_FAILURES.items())
            + "\n" + "\n".join(task_assets._MERGE_MISSING_ARTIFACTS))
        self.evidence["task_id"] = "merge-diff-arc-agi-task"
        self.evidence["original_verifier_terminal"]["return_code"] = 0
        raw = (self.root / "verifier/ctrf.json").read_bytes()
        self.evidence["ctrf"] = {"sha256": self.sha(raw), "bytes": len(raw)}
        (self.root / "bootstrap-qualification.json").write_text(json.dumps(self.evidence))
        pins = {"task_id": "merge-diff-arc-agi-task",
            "qualification_sha256": self.sha_file("bootstrap-qualification.json"),
            "ctrf_sha256": self.sha_file("verifier/ctrf.json"),
            "stdout_sha256": self.sha_file("verifier/test-stdout.txt")}
        self.review.pop("missing_artifact")
        self.review.update(pins, missing_artifacts=list(task_assets._MERGE_MISSING_ARTIFACTS),
            expected_failures=dict(task_assets._EXPECTED_MERGE_FAILURES))
        self.review_path.write_text(json.dumps(self.review))
        return pins

    def test_exact_merge_review_preserves_baseline_zero_and_rejects_unpinned_fixture(self):
        pins = self.merge_fixture()
        with self.assertRaisesRegex(ValueError, "require original log review"):
            task_assets.qualification_proof(self.root, pins["task_id"], "a" * 64, "b" * 64)
        with self.assertRaisesRegex(ValueError, "pinned original Merge"):
            task_assets.qualification_proof(self.root, pins["task_id"], "a" * 64, "b" * 64,
                                            failure_review=self.review_path)
        before = {str(p): self.sha(p.read_bytes()) for p in self.root.rglob("*") if p.is_file()}
        with patch.object(task_assets, "_REVIEWED_MERGE_BASELINE", pins):
            proof = task_assets.qualification_proof(self.root, pins["task_id"], "a" * 64, "b" * 64,
                                                  failure_review=self.review_path)
        self.assertEqual(proof["original_baseline_reward"], 0)
        self.assertIsNone(proof["model_score"])
        self.assertEqual(before, {str(p): self.sha(p.read_bytes()) for p in self.root.rglob("*") if p.is_file()})
        self.assertEqual(task_assets._REVIEWED_MERGE_BASELINE["qualification_sha256"],
            "c682dcd8017a3c8615f7ff49410812d0927382ffe4eaaa4abb1d4ce563a8e53f")

    def test_merge_review_rejects_changed_outputs_error_mapping_or_other_baseline(self):
        pins = self.merge_fixture()
        for field, replacement in (("missing_artifacts", ["/app/repo/algo.py"]),
                ("expected_failures", {**self.review["expected_failures"], "test_both_bundles_fetched": "AssertionError"}),
                ("task_id", "sparql-university"), ("stdout_sha256", "d" * 64)):
            with self.subTest(field=field), patch.object(task_assets, "_REVIEWED_MERGE_BASELINE", pins):
                self.review_path.write_text(json.dumps({**self.review, field: replacement}))
                with self.assertRaises(ValueError):
                    task_assets.qualification_proof(self.root, pins["task_id"], "a" * 64, "b" * 64,
                                                   failure_review=self.review_path)


if __name__ == "__main__":
    unittest.main()
