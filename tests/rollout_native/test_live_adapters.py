import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rollout_manager.adapters.candidates import FILES, parse_candidate, require_review
from rollout_manager.adapters.harnesses import HarnessRunner
from rollout_manager.common import digest
from rollout_manager.adapters.native_timing import HarnessInstrumentationError


class LiveAdapterTests(unittest.TestCase):
    def test_cpu_only_compiler_does_not_query_gpu_ownership(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("rollout_manager.adapters.harnesses.os.sched_getaffinity", return_value={0}, create=True):
                runner = HarnessRunner(root, "", "10.0", lambda: None, cores=[0])
            with patch("rollout_manager.adapters.harnesses.run_process", return_value={"process_group_stopped": True}), \
                    patch("rollout_manager.adapters.harnesses.gpu_processes") as query:
                result = runner.child(root, "compile", ["-m", "utils.compile"], gpu=False, timeout=240)
            query.assert_not_called()
            self.assertEqual(result["cuda_visible_devices"], "")
            self.assertTrue(result["gpu_contexts_released"])

    def test_native_evaluation_keeps_separate_seeds_and_cleanup_evidence(self):
        for final in (False, True):
            with self.subTest(final=final), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "source-manifest.json").write_text("{}")
                prefix = "final" if final else "evaluation"
                (root / f"{prefix}-kernelbench.log").write_text("")
                (root / f"{prefix}-kernelbench.json").write_text('{"correctness": true}')
                process = {"process_group_stopped": True, "gpu_contexts_released": False,
                           "gpu_quiescent": True, "returncode": 0, "stopped_for": None}
                with patch("rollout_manager.adapters.harnesses.os.sched_getaffinity",
                           return_value={0}, create=True):
                    runner = HarnessRunner(root, "gpu", "10.0", lambda: None, cores=[0])
                with patch.object(runner, "child", return_value=process) as child:
                    result = runner.evaluate(root, "kernelbench", final=final, observe_inputs=True)
                argv = child.call_args.args[2]
                self.assertEqual(argv[argv.index("--seed")+1], "1042" if final else "42")
                self.assertIn("--observe-inputs", argv)
                self.assertFalse(result["gpu_contexts_released"])
                self.assertEqual(result["seed"], 1042 if final else 42)

    def test_instrumentation_failure_does_not_become_candidate_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "source-manifest.json").write_text("{}")
            (root / "evaluation-kernelbench.log").write_text("native input count mismatch")
            with patch("rollout_manager.adapters.harnesses.os.sched_getaffinity", return_value={0}, create=True):
                runner = HarnessRunner(root, "gpu", "10.0", lambda: None, cores=[0])
            with patch.object(runner, "child", return_value={"returncode": 78}):
                with self.assertRaises(HarnessInstrumentationError):
                    runner.evaluate(root, "kernelbench", observe_inputs=True)

    def test_native_thread_limit_cannot_exceed_reserved_affinity(self):
        with patch("rollout_manager.adapters.harnesses.os.sched_getaffinity",
                   return_value={0, 1, 2, 3}, create=True):
            for count in (0, 4):
                with self.assertRaisesRegex(ValueError, "thread count"):
                    HarnessRunner(Path("."), "gpu", "10.3", lambda: None, cores=[0, 1, 2], cpu_threads=count)
            runner = HarnessRunner(Path("."), "gpu", "10.3", lambda: None, cores=[0, 1, 2], cpu_threads=3)
            self.assertEqual(runner.env["OMP_NUM_THREADS"], "3")
            self.assertEqual(runner.env["MKL_NUM_THREADS"], "3")
            self.assertEqual(runner.cores, [0, 1, 2])

    def test_candidate_paths_cannot_replace_harness_or_escape(self):
        files = {name: "pass" for name in FILES}
        files["model_new.py"] = "class ModelNew: pass"
        self.assertEqual(parse_candidate(json.dumps({"files": files})), files)
        for name in ("../model.py", "utils/verification.py", "/tmp/x"):
            with self.assertRaises(ValueError):
                parse_candidate(json.dumps({"files": dict(files, **{name: "x"})}))

    def test_missing_entry_point_is_candidate_feedback_before_compilation(self):
        files = {name: "pass" for name in FILES}
        files["model_new.py"] = "class Model: pass"
        with self.assertRaisesRegex(ValueError, "ModelNew"):
            parse_candidate(json.dumps({"files": files}))
        with self.assertRaisesRegex(ValueError, "must be text"):
            parse_candidate(None)

    def test_review_is_bound_to_exact_source(self):
        files = {name: "pass" for name in FILES}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "review.json"
            path.write_text(json.dumps({"sha256": digest(files), "scope": "reviewed-development-pilot"}))
            require_review(path, files)
            files["model_new.py"] += "\n# changed"
            with self.assertRaises(ValueError):
                require_review(path, files)
