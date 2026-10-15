import copy
import unittest

from rollout_manager.analysis.report_coupled import clock_bounds, engine_match, without_review
from rollout_manager.analysis.analyze_graph_trace import coverage


class CoupledEvidenceTests(unittest.TestCase):
    def test_clock_offset_with_asymmetric_delay(self):
        row = {"client_started_unix_ns": 1000, "server_received_unix_ns": 1320,
               "server_response_unix_ns": 1340, "client_received_unix_ns": 1100}
        result = clock_bounds([row], [row])
        self.assertEqual(result["before_bounds_ns"], (240, 320))
        self.assertEqual(result["A_minus_B_ns"], 280)

    def test_clock_step_widens_display_uncertainty(self):
        row = {"client_started_unix_ns": 1000, "server_received_unix_ns": 1320,
               "server_response_unix_ns": 1340, "client_received_unix_ns": 1100}
        later = dict(row, server_received_unix_ns=2320, server_response_unix_ns=2340)
        self.assertEqual(clock_bounds([row], [later])["uncertainty_ns"], 540)

    def test_ambiguous_engine_join_is_rejected(self):
        record = {"response_metrics": {"decode": 1}, "completion_tokens": 3}
        response = {"metrics": {"decode": 1}, "usage": {"completion_tokens": 3}}
        with self.assertRaises(ValueError):
            engine_match([record, record], response)

    def test_nested_cuda_calls_are_not_double_counted(self):
        self.assertEqual(coverage([(0, 10), (2, 5), (8, 15), (20, 22)]), 17)

    def test_review_projection_preserves_measurements_and_recomputes_group_tail(self):
        def interval(job, machine, stage, start, end, scope="stage"):
            return {"job": job, "machine": machine, "stage": stage, "scope": scope,
                    "start_seconds": start, "end_seconds": end, "duration_seconds": end - start}
        run = {"clock": {}, "origin_unix_ns": 100,
               "jobs": [{"job": name, "rollout": {}, "verified_correct": True} for name in ("a", "b")],
               "intervals": [
                   interval("a", "A", "decode", 0, 2),
                   interval("a", "B", "rpc", 0, 2),
                   interval("a", "B", "review", 2, 22),
                   interval("a", "B", "build", 22, 32),
                   interval("a", "B", "compile/process", 22.5, 31, "process"),
                   interval("a", "B", "result", 32, 32, "boundary"),
                   interval("b", "A", "decode", 0, 4),
                   interval("b", "B", "review", 4, 14),
                   interval("b", "B", "build", 14, 26),
                   interval("b", "B", "result", 26, 26, "boundary"),
               ]}
        original = copy.deepcopy(run)
        projected = without_review(run)
        self.assertEqual(run, original)
        self.assertEqual(projected["projected_span_seconds"], 16)
        self.assertFalse(any(row["stage"] == "review" for row in projected["intervals"]))
        for row in projected["intervals"]:
            source, = [r for r in original["intervals"] if (r["job"], r["stage"]) == (row["job"], row["stage"])]
            self.assertEqual(row["duration_seconds"], source["duration_seconds"])
            if row["machine"] == "A" or row["stage"] == "rpc":
                self.assertEqual(row["start_seconds"], source["start_seconds"])
        nested, = [r for r in projected["intervals"] if r["scope"] == "process"]
        self.assertEqual(nested["start_seconds"], 2.5)

    def test_review_projection_rejects_ambiguous_input(self):
        with self.assertRaisesRegex(ValueError, "exactly one review"):
            without_review({"jobs": [{"job": "a"}], "intervals": []})
