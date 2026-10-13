import unittest

from summarize_memory_timeline import host_totals, occupancy, select_events


class MemoryTimelineTest(unittest.TestCase):
    def test_gpu_scope_annotations_do_not_duplicate_host_time(self):
        trace = {"traceEvents": [
            {"ph": "X", "cat": "user_annotation", "name": "train/backward", "ts": 0, "dur": 2000},
            {"ph": "X", "cat": "gpu_user_annotation", "name": "train/backward", "ts": 0, "dur": 1800},
            {"ph": "X", "cat": "kernel", "name": "ncclAllReduce", "ts": 5, "dur": 1000},
            {"ph": "X", "cat": "kernel", "name": "gemm", "ts": 10, "dur": 150},
            {"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD", "ts": 20, "dur": 10}]}
        events = select_events(trace)
        self.assertEqual([e["kind"] for e in events], ["host", "nccl", "compute", "h2d"])
        self.assertEqual(host_totals(events, 2)["train/backward"],
                         {"calls_per_step": .5, "inclusive_ms_per_step": 1.0})

    def test_overlap_is_unioned_and_partial_bins_are_clipped(self):
        self.assertEqual(occupancy([(0, 15), (10, 20), (25, 40)], 0, 10, 4), [100, 100, 50, 100])
        self.assertEqual(occupancy([(-5, 5), (35, 50)], 0, 10, 4), [50, 0, 0, 50])
        self.assertEqual(occupancy([], 0, 10, 2), [0, 0])

    def test_loading_keeps_domain_and_training_combines_tables(self):
        events = [{"kind": "host", "name": name, "duration_us": 1000} for name in
                  ("dsv41/load_decode/engram", "dsv41/load_decode/experts",
                   "dsv41/engram_store/model.tables.1", "dsv41/engram_store/model.tables.3")]
        totals = host_totals(events, 1)
        self.assertEqual(len(totals), 3)
        self.assertEqual(totals["dsv41/engram_store"]["inclusive_ms_per_step"], 2)


if __name__ == "__main__":
    unittest.main()
