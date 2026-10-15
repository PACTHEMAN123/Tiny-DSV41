import copy
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rollout_manager.frozen_rollouts import freeze, load
from rollout_manager.common import digest
from rollout_manager.workloads import suite_groups


class FrozenRolloutTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.cases = [{"task": {"task_id": str(i)}, "files": {}, "request": {"model": "test"}}
                      for i in range(2)]
        self.cases_path = self.root / "cases.json"
        self.cases_path.write_text(json.dumps(self.cases))
        self.groups = suite_groups(self.cases, 8, 1, 42)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "settings.json").write_text(json.dumps({"mode": "inference",
            "schedule": "full-suite", "turns": 1, "rollouts_per_task": 8,
            "repeats": 1, "sampling_seed": 42}))
        self.files = {"model_new.py": "class ModelNew:\n    pass\n",
                      "kernels/solution.cu": "cuda", "kernels/solution_binding.cpp": "binding"}

    def populate(self, groups=None):
        for group, cases in self.groups if groups is None else groups:
            for case in cases:
                sample = case["rollout"]["sample_index"]
                job = group + "-" + str(sample)
                path = self.source / "jobs" / job
                path.mkdir(parents=True)
                content = json.dumps(self.files) if sample else "invalid JSON"
                result = {"task": case["task"]["task_id"], "rollout": case["rollout"],
                          "job": job, "completed": True, "stages": {"inference": {"service_seconds": 5}}}
                if sample:
                    result["candidate_sha256"] = digest(self.files)
                    (path / "candidate.json").write_text(json.dumps(self.files))
                else:
                    result["candidate_format_error"] = "format error recorded by collection"
                (path / "result.json").write_text(json.dumps(result))
                (path / "dispatch.json").write_text(json.dumps({"request": case["request"]}))
                (path / "inference.json").write_text(json.dumps({"response": {
                    "choices": [{"message": {"content": content}}]}}))

    def bind(self, manifest, **kwargs):
        path = self.root / "frozen.json"
        path.write_text(json.dumps(manifest))
        return load(path, self.cases_path, self.groups, **kwargs)

    def test_full_groups_keep_format_failures_and_independent_seeds(self):
        self.populate()
        manifest = freeze(self.source, self.cases_path)
        groups, metadata = self.bind(manifest)
        self.assertEqual(len(groups), 2)
        self.assertFalse(metadata["partial_qualification"])
        self.assertEqual(metadata["expected_episodes"], 16)
        self.assertEqual(len({c["rollout"]["sampling_seed"] for _, samples in groups for c in samples}), 16)
        self.assertEqual(sum("candidate_format_error" in c["frozen_rollout"]
                             for _, samples in groups for c in samples), 2)
        self.assertNotIn("frozen_rollout", self.groups[0][1][0])
        self.assertEqual(groups[0][1][1]["frozen_rollout"]["source"]["stages"],
                         {"inference": {"service_seconds": 5}})

    def test_partial_groups_require_explicit_qualification_and_are_never_sliced(self):
        self.populate([self.groups[0], (self.groups[1][0], self.groups[1][1][:7])])
        with self.assertRaisesRegex(ValueError, "partial replay"):
            freeze(self.source, self.cases_path)
        manifest = freeze(self.source, self.cases_path, allow_partial=True)
        self.assertEqual(manifest["observed_responses"], 15)
        self.assertEqual(len(manifest["groups"]), 1)
        with self.assertRaisesRegex(ValueError, "explicit qualification"):
            self.bind(manifest)
        groups, metadata = self.bind(manifest, allow_partial=True)
        self.assertEqual(len(groups[0][1]), 8)
        self.assertTrue(metadata["partial_qualification"])

    def test_uncertain_request_cannot_be_converted_to_candidate_failure(self):
        self.populate()
        path = next(self.source.glob("jobs/*/result.json"))
        result = json.loads(path.read_text())
        result.update(completed=False, communication_failure={"dispatch_may_have_occurred": True})
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "infrastructure-failed"):
            freeze(self.source, self.cases_path, allow_partial=True)

    def test_tampered_source_or_sampling_is_rejected_before_execution(self):
        self.populate()
        original = freeze(self.source, self.cases_path)
        for mutation in ("content", "format", "seed", "request", "settings", "coverage"):
            with self.subTest(mutation=mutation):
                manifest = copy.deepcopy(original)
                item = manifest["groups"][0][1]
                if mutation == "content":
                    item["content"] = json.dumps(self.files | {"kernels/solution.cu": "changed"})
                elif mutation == "format":
                    item["candidate_format_error"] = "mislabelled"
                elif mutation == "seed":
                    item["rollout"]["sampling_seed"] += 1
                elif mutation == "request":
                    item["request_sha256"] = "bad"
                elif mutation == "settings":
                    manifest["sampling_seed"] += 1
                else:
                    manifest["observed_responses"] = 0
                with self.assertRaises(ValueError):
                    self.bind(manifest)

    def test_duplicate_and_incomplete_groups_cannot_claim_full_coverage(self):
        self.populate()
        original = freeze(self.source, self.cases_path)
        for groups in ([original["groups"][0]] * 2,
                       [original["groups"][0][:-1], original["groups"][1]]):
            with self.assertRaisesRegex(ValueError, "incomplete frozen group"):
                self.bind(original | {"groups": groups})

    def test_tools_run_evaluates_exact_candidates_without_regenerating_or_dropping_failures(self):
        from rollout_manager.runner import run

        self.populate()
        self.bind(freeze(self.source, self.cases_path))
        for case in self.cases:
            case["task"]["dataset"] = "kernelbench"
        # Dataset metadata is part of the cohort digest, so rebuild the frozen input.
        self.cases_path.write_text(json.dumps(self.cases))
        self.groups = suite_groups(self.cases, 8, 1, 42)
        self.bind(freeze(self.source, self.cases_path))
        (self.root / "manifest.json").write_text(json.dumps({"upstream_files": {}, "architecture": "10.0"}))
        for name in ("contracts", "models"):
            (self.root / (name + ".json")).write_text("{}")
        evaluations = []

        class Client:
            def __init__(self, *args, **kwargs):
                pass

            def call(self, *args):
                raise AssertionError("frozen replay must not call inference")

            def close(self):
                pass

        class Harness:
            def __init__(self, *args, **kwargs):
                pass

            def compile(self, workspace):
                return {"compiled": True, "process_group_stopped": True}

            def evaluate(self, workspace, dataset, final=False):
                evaluations.append(final)
                return {"correct": True, "process_group_stopped": True,
                        "gpu_quiescent": True, "feedback": "native result"}

        args = Namespace(root=self.root / "run", assets=self.root, credentials=self.root,
            cases=self.cases_path, catalog=self.root / "catalog.json", mode="tools",
            schedule="full-suite", frozen_rollouts=self.root / "frozen.json",
            turns=1, repeats=1, rollouts_per_task=8, sampling_seed=42, max_inflight_groups=2,
            gpus="gpu", build_slots=1, cores_per_worker=1, concurrency=[4],
            evaluation_host_gib=1, model_manifest=self.root / "models.json", ttl=60,
            input_memory_plan=self.root / "contracts.json",
            review_timeout=1, instrumentation="none", url="https://example.invalid")
        eligibility = {"tasks": {str(i): {"score_eligible": True} for i in range(2)}}
        with patch("rollout_manager.runner.os.sched_getaffinity", return_value=set(range(16)), create=True), \
                patch("rollout_manager.reference_qualification.load_catalog", return_value=eligibility), \
                patch("rollout_manager.runner.evaluator_estimates", return_value={str(i): {"bytes": 1} for i in range(2)}), \
                patch("rollout_manager.runner.available_bytes", return_value=1024**4), \
                patch("rollout_manager.runner.gpu_processes", return_value=[]), \
                patch("rollout_manager.runner.check_headroom"), \
                patch("rollout_manager.runner.monitor"), patch("rollout_manager.runner.probe_loop"), \
                patch("rollout_manager.runner.RpcClient", Client), \
                patch("rollout_manager.runner.HarnessRunner", Harness), \
                patch("rollout_manager.runner.prepare_workspace") as prepare, \
                patch("rollout_manager.runner.await_review", return_value=0):
            run(args)
        summary = json.loads((args.root / "summary.json").read_text())
        self.assertFalse(summary["fatal"])
        self.assertTrue(summary["requested_levels_completed"])
        self.assertEqual(summary["measurement"], "tools_only_frozen_real_inference")
        level = summary["levels"][0]
        self.assertEqual((level["jobs"], level["correct"], level["errors"]), (16, 14, 2))
        self.assertEqual(level["completion_tokens"], 0)
        self.assertEqual(evaluations.count(True), 14)
        self.assertEqual(prepare.call_count, 14)
        self.assertTrue(all(call.args[3] == self.files for call in prepare.call_args_list))
        self.assertTrue(all("source_inference" in item and "communication" not in item
                            for item in level["results"]))
