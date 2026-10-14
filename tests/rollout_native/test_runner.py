import concurrent.futures
import json
import tempfile
import threading
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from rollout_manager.runner import CandidateRejected, Journal, await_review, feedback_request, make_pool, reservation, reviewed, run
from rollout_manager.common import digest
from rollout_manager.network import tcp_info


class LoadTestTests(unittest.TestCase):
    def test_exact_rejection_ends_review_without_authorizing_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = {"model_new.py": "candidate"}
            sha = digest(files)
            catalog = root / "catalog.json"
            catalog.write_text(json.dumps({sha: {"scope": "reviewed-development-pilot",
                "decision": "reject", "reason": "PyTorch computation in forward violates CUDA-only contract"}}))
            journal = Journal(root)
            try:
                self.assertFalse(reviewed(files, catalog))
                with patch("rollout_manager.runner.time.sleep") as sleep:
                    with self.assertRaises(CandidateRejected) as caught:
                        await_review(files, root, catalog, 60, journal, "job")
                    sleep.assert_not_called()
                self.assertEqual(caught.exception.decision["sha256"], sha)
                self.assertEqual(caught.exception.decision["decision"], "reject")
                self.assertEqual(journal.counts["review"], {"queued": 0, "active": 0, "completed": 0})
                self.assertFalse(reviewed({"model_new.py": "different candidate"}, catalog))
            finally:
                journal.close()

    def test_rpc_failure_stops_full_suite_and_keeps_missing_coverage(self):
        deadlines = []
        class Client:
            def __init__(self, *args, timeout=900):
                deadlines.append(timeout)

            def call(self, route, request):
                error = RuntimeError("relay HTTP 502")
                error.rpc_failure = {"http_status": 502, "dispatch_may_have_occurred": True}
                raise error

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "manifest.json").write_text(json.dumps({"upstream_files": {}, "architecture": "10.0"}))
            cases = [{"task": {"task_id": str(i)}, "files": {}, "request": {}} for i in range(2)]
            (base / "cases.json").write_text(json.dumps(cases))
            args = Namespace(root=base / "run", assets=base, credentials=base, catalog=base / "catalog.json",
                cases=base / "cases.json", mode="inference", gpus="", build_slots=1, cores_per_worker=1,
                ttl=60, turns=1, repeats=1, concurrency=[1], url="https://example.invalid",
                review_timeout=1, instrumentation="none", schedule="full-suite", rollouts_per_task=8,
                max_inflight_groups=1, rpc_timeout=1860)
            with patch("rollout_manager.runner.os.sched_getaffinity", return_value=set(range(16)), create=True), \
                    patch("rollout_manager.runner.check_headroom"), \
                    patch("rollout_manager.runner.monitor"), patch("rollout_manager.runner.probe_loop"), \
                    patch("rollout_manager.runner.RpcClient", Client):
                run(args)
            summary = json.loads((base / "run/summary.json").read_text())
            self.assertTrue(summary["fatal"])
            self.assertEqual(deadlines, [1860])
            self.assertFalse(summary["requested_levels_completed"])
            self.assertEqual(summary["levels"][0]["jobs"], 1)
            self.assertEqual(summary["levels"][0]["coverage"]["expected_episodes"], 16)
            self.assertEqual(summary["levels"][0]["results"][0]["failure_class"], "infrastructure")

    def test_review_wait_cancels_when_other_work_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / "catalog.json"
            catalog.write_text("{}")
            fatal = threading.Event()
            journal = Journal(root)
            def fail_during_wait(_):
                fatal.set()
            try:
                with patch("rollout_manager.runner.time.sleep", side_effect=fail_during_wait):
                    with self.assertRaisesRegex(RuntimeError, "stopped while awaiting source review"):
                        await_review({"model_new.py": "unreviewed"}, root, catalog, 60, journal, "job", fatal)
                self.assertEqual(journal.counts["review"], {"queued": 0, "active": 0, "completed": 0})
                self.assertEqual(json.loads(catalog.read_text()), {})
            finally:
                journal.close()

    def test_two_turn_run_uses_feedback_and_verifies_prior_correct_candidate(self):
        calls, finals = [], []

        class Client:
            def __init__(self, *args, timeout=900):
                pass

            def call(self, route, request):
                calls.append(request)
                return {"response": {"choices": [{"finish_reason": "stop", "message": {
                    "role": "assistant", "content": "candidate"}}], "usage": {"completion_tokens": 10}},
                    "communication": dict.fromkeys(("client_total_seconds", "server_processing_seconds",
                        "residual_seconds", "connect_tls_seconds", "request_body_bytes", "response_body_bytes",
                        "tcp_retrans_delta"), 0)}

            def close(self):
                pass

        class Harness:
            def __init__(self, *args, **kwargs):
                pass

            def compile(self, workspace):
                return {"compiled": workspace.parent.name.endswith("-t1"), "process_group_stopped": True,
                        "stopped_for": None, "feedback": "compiler feedback"}

            def evaluate(self, workspace, dataset, final=False):
                if final:
                    finals.append(workspace)
                return {"correct": True, "process_group_stopped": True, "gpu_quiescent": True,
                        "processes": [], "timing": {"candidate_us": 42}, "feedback": "native feedback"}

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "manifest.json").write_text(json.dumps({"upstream_files": {}, "architecture": "10.3"}))
            cases = [{"task": {"task_id": "test", "dataset": "kernelbench"}, "files": {},
                      "request": {"model": "test", "messages": [{"role": "user", "content": "task"}]}}]
            (base / "cases.json").write_text(json.dumps(cases))
            args = Namespace(root=base / "run", assets=base, credentials=base, catalog=base / "catalog.json",
                             cases=base / "cases.json", mode="coupled", gpus="gpu", build_slots=1,
                             cores_per_worker=1, ttl=60, turns=2, repeats=1, concurrency=[1],
                             url="https://example.invalid", review_timeout=1, instrumentation="none")
            with patch("rollout_manager.runner.os.sched_getaffinity", return_value=set(range(16)), create=True), \
                    patch("rollout_manager.runner.gpu_processes", return_value=[]), \
                    patch("rollout_manager.runner.check_headroom"), \
                    patch("rollout_manager.runner.monitor"), patch("rollout_manager.runner.probe_loop"), \
                    patch("rollout_manager.runner.RpcClient", Client), \
                    patch("rollout_manager.runner.HarnessRunner", Harness), \
                    patch("rollout_manager.runner.prepare_workspace"), \
                    patch("rollout_manager.runner.parse_candidate", return_value={"model_new.py": "reviewed"}), \
                    patch("rollout_manager.runner.await_review", return_value=0):
                run(args)
            level = json.loads((base / "run/level-1.json").read_text())
            self.assertEqual(level["correct"], 1)
            self.assertEqual(level["errors"], 0)
            self.assertEqual(level["completion_tokens"], 20)
            self.assertEqual(len(calls), 2)
            self.assertIn("native feedback", calls[1]["messages"][-1]["content"])
            self.assertTrue(finals[0].parent.name.endswith("-t1"))
            self.assertEqual(level["results"][0]["selected_turn"], 1)

    def test_feedback_extends_actual_conversation_without_mutating_it(self):
        original = {"model": "model", "messages": [{"role": "user", "content": "task"}]}
        response = {"choices": [{"message": {"role": "assistant", "content": "candidate"}}]}
        request = feedback_request(original, response, {"compiled": True, "correct": False})
        self.assertEqual(len(original["messages"]), 1)
        self.assertEqual(request["messages"][-2], response["choices"][0]["message"])
        self.assertIn('"correct": false', request["messages"][-1]["content"])
        self.assertEqual(request["model"], "model")

    def test_parallel_reservations_return_capacity_after_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            journal = Journal(root)
            pool = make_pool(["gpu-0", "gpu-1"])

            def job(index):
                try:
                    with reservation(pool, journal, str(index), "evaluation"):
                        raise ValueError("candidate rejected")
                except ValueError:
                    pass
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(job, range(32)))
            journal.close()
            self.assertEqual(pool.qsize(), 2)
            records = [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]
            self.assertTrue(all(0 <= r["queues"]["evaluation"]["active"] <= 2 for r in records))
            self.assertEqual(records[-1]["queues"]["evaluation"], {"queued": 0, "active": 0, "completed": 32})

    def test_review_catalog_cannot_authorize_modified_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "catalog.json"
            files = {"model_new.py": "reviewed source"}
            path.write_text(json.dumps({digest(files): {"scope": "reviewed-development-pilot"}}))
            self.assertTrue(reviewed(files, path))
            self.assertFalse(reviewed({"model_new.py": "other source"}, path))

    def test_linux_tcp_info_field_offsets(self):
        import struct
        from unittest.mock import patch
        class Socket:
            def getsockopt(self, *args):
                return bytes(8) + struct.pack("=24I", *range(24))
        with patch("socket.TCP_INFO", 11, create=True):
            record = tcp_info(Socket())
        self.assertEqual(record["rtt_us"], 15)
        self.assertEqual(record["total_retrans"], 23)
        self.assertEqual(record["snd_cwnd"], 18)
