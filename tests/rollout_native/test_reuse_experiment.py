import unittest
from pathlib import Path

from rollout_manager.analysis.reuse_experiment import build_command, model_files, compare_jobs
from rollout_manager.analysis.check_input_reuse_cuda import check_result


class ReuseExperimentTests(unittest.TestCase):
    def test_changed_development_result_stops_even_when_both_final_flags_are_false(self):
        baseline = {"job": {"verified": False, "development_correct": False, "final_correct": None,
            "error_type": None, "candidate_sha256": "exact", "source_inference": {},
            "rollout": {}, "score_eligible": True}}
        self.assertEqual(compare_jobs(baseline, baseline), [])
        changed = {"job": dict(baseline["job"], development_correct=True)}
        self.assertEqual(compare_jobs(baseline, changed), ["job"])

    def test_workload_identity_cannot_be_downgraded_to_an_outcome_change(self):
        baseline = {"job": {"candidate_sha256": "exact", "source_inference": {},
                            "rollout": {}, "score_eligible": True}}
        with self.assertRaisesRegex(ValueError, "workload changed"):
            compare_jobs(baseline, {"job": dict(baseline["job"], candidate_sha256="different")})

    def settings(self):
        fields = ("mode assets credentials cases catalog frozen_rollouts url rpc_timeout reference_qualification "
                  "concurrency repeats schedule rollouts_per_task max_inflight_groups sampling_seed turns gpus "
                  "build_slots cores_per_worker evaluation_cores evaluation_threads gpu_slots gpu_settle_seconds "
                  "evaluation_host_gib model_manifest review_timeout ttl").split()
        values = {key: key+"_value" for key in fields}
        values.update(mode="tools", concurrency=[64], repeats=1, rollouts_per_task=8,
                      max_inflight_groups=8, gpu_slots=1, evaluation_threads=1)
        return values

    def test_only_cache_switches_differ_between_comparable_commands(self):
        kwargs = dict(settings=self.settings(), root=Path("run"), registry=Path("plan.json"),
                      qualification=Path("qualified.json"), socket_path=Path("/tmp/cache.sock"))
        off = build_command(**kwargs, state="off")
        for state in ("cold", "warm"):
            enabled = build_command(**kwargs, state=state)
            self.assertEqual(enabled[:-6], off)
            self.assertEqual(enabled[-6:], ["--input-cache-socket", "/tmp/cache.sock",
                "--input-cache-qualification", "qualified.json", "--input-cache-state", state])
        self.assertNotIn("--preparation-workers", off)
        self.assertNotIn("--preparation-gib", off)

    def test_cuda_integration_requires_hits_trials_and_cleanup(self):
        import copy
        result = {"process_group_stopped": True, "gpu_contexts_released": True, "correct": True,
            "native": {"stage_metrics": {"input_calls": 7, "logical_input_indices": list(range(7)),
                "input_reuse": {"bypasses": 0, "hits": 7, "events": [{"key": str(i)} for i in range(7)]}}}}
        self.assertEqual(check_result(result, True, True), list(map(str, range(7))))
        changed = copy.deepcopy(result)
        changed["native"]["stage_metrics"]["input_reuse"]["hits"] = 0
        with self.assertRaises(ValueError):
            check_result(changed, True, True)
        changed = copy.deepcopy(result)
        changed["native"]["stage_metrics"]["logical_input_indices"] = [0]*7
        with self.assertRaises(ValueError):
            check_result(changed, False, False)
        changed = copy.deepcopy(result)
        changed["gpu_contexts_released"] = False
        with self.assertRaises(RuntimeError):
            check_result(changed, False, False)

    def test_only_required_offline_models_are_checked_and_corruption_is_rejected(self):
        import hashlib
        import json
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            cases = root/"cases.json"
            cases.write_text(json.dumps([{"task": {"code":
                "model_name = 'gpt2'\nx = Model.from_pretrained(model_name)\n"}}]))
            assets = root/"model-assets.json"
            digest = hashlib.sha256(b"model").hexdigest()
            assets.write_text(json.dumps([
                {"model": "gpt2", "revision": "pinned", "files": [
                    {"name": "weights.bin", "bytes": 5, "sha256": digest}]},
                {"model": "not-ready", "revision": "pinned", "files": []}]))
            weights = root/"hf-home/hub/models--gpt2/snapshots/pinned/weights.bin"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"model")
            settings = {"cases": str(cases), "model_manifest": str(assets)}
            home, pins = model_files(settings)
            self.assertEqual(home, root/"hf-home")
            self.assertEqual(pins, {str(weights): digest})
            weights.write_bytes(b"wrong")
            with self.assertRaisesRegex(ValueError, "asset changed"):
                model_files(settings)


if __name__ == "__main__":
    unittest.main()
