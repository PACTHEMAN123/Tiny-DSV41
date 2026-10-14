import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rollout_manager.reference_qualification import WRAPPER, build_catalog, load_catalog
from rollout_manager.workloads import GroupSchedule, suite_groups


class ReferenceQualificationTests(unittest.TestCase):
    def test_unscorable_reference_stays_in_workload_coverage(self):
        schedule = GroupSchedule(suite_groups([{"task": {"task_id": "dropout"}, "request": {}}], 1, 1, 42), 1)
        case = schedule.take(lambda: False)
        schedule.complete(case, {"job": "job", "started_unix_ns": 1, "finished_unix_ns": 2,
                                 "final_correct": True, "score_eligible": False, "verified_correct": False})
        self.assertEqual(schedule.summary()["completed_groups"], 1)
        self.assertEqual(schedule.summary()["groups"][0]["episodes"], 1)
        self.assertEqual(schedule.summary()["groups"][0]["verified"], 0)

    def test_catalog_requires_complete_cohort_and_matching_runtime_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cases = [{"task": {"task_id": key, "source_sha256": key}} for key in ("float", "integer", "dropout")]
            path = root / "cases.json"
            path.write_text(json.dumps(cases))
            sha = hashlib.sha256(path.read_bytes()).hexdigest()

            def evidence(name, corrected, values):
                run = root / name
                run.mkdir()
                (run / "settings.json").write_text(json.dumps({"cases_sha256": sha, "candidate_wrapper": WRAPPER,
                    "preserve_integer_inputs": corrected, "seed": 42, "correctness_trials": 5,
                    "performance_trials": 10, "generated_candidates": False}))
                (run / "summary.json").write_text(json.dumps({"fatal": False, "completed": len(values),
                    "expected": len(values), "results": [{"task_id": key, "passed": passed, "result":
                    {"correct": passed, "process_group_stopped": True, "gpu_contexts_released": True}}
                    for key, passed in values.items()]}))
                return run

            baseline = evidence("baseline", False, {"float": True, "integer": False, "dropout": False})
            correction = evidence("correction", True, {"integer": True, "dropout": False})
            catalog = build_catalog(path, baseline, correction)
            self.assertTrue(catalog["tasks"]["integer"]["score_eligible"])
            self.assertFalse(catalog["tasks"]["dropout"]["score_eligible"])
            target = root / "catalog.json"
            target.write_text(json.dumps(catalog))
            self.assertEqual(load_catalog(target, path, cases, preserve_integer_inputs=True), catalog)
            with self.assertRaisesRegex(ValueError, "policy differs"):
                load_catalog(target, path, cases, preserve_integer_inputs=False)
            partial = evidence("partial", True, {"integer": True})
            with self.assertRaisesRegex(ValueError, "cover"):
                build_catalog(path, baseline, partial)
            cases[0]["task"]["source_sha256"] = "changed"
            with self.assertRaisesRegex(ValueError, "source"):
                load_catalog(target, path, cases, preserve_integer_inputs=True)
