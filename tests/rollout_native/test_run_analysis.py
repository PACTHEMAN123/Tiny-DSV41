import unittest


from rollout_manager.analysis import analyze_run as analysis


class MeasurementAnalysisTests(unittest.TestCase):
    def test_gpu_query_failure_is_missing_coverage_not_zero_activity(self):
        samples = [{"unix_ns": i*5_000_000_000, "memory": "MemAvailable: 100 kB\n",
                    "stat": "cpu 0 0 0 0 0 0 0 0\ncpu0 0 0 0 0 0 0 0 0\n",
                    "cpu_pressure": "", "memory_pressure": "", "io_pressure": ""}
                   for i in range(3)]
        samples[0]["gpu_csv"] = "0, GPU-test, 100, 0, 10, 20"
        samples[1].update(gpu_csv=None, gpu_error="query timeout")
        samples[2]["gpu_csv"] = "0, GPU-test, 50, 0, 10, 20"
        result = analysis.telemetry_summary(samples,
            {"cpu_affinity": [0], "effective_driver_affinity": [0]}, 0, 10_000_000_000)
        self.assertEqual(result["time_weighted_mean"]["gpu_0_util_percent"], 100)
        self.assertEqual(result["metric_covered_seconds"]["gpu_0_util_percent"], 5)
        self.assertEqual(result["covered_seconds"], 10)
        self.assertEqual(result["gpu_query_error_samples"], 1)

    def test_cpu_busy_excludes_idle_iowait_steal_and_guest_duplicates(self):
        raw = "cpu 10 2 3 100 20 5 7 9 8 2\ncpu0 4 0 1 70 8 2 3 1 0 0\n"
        self.assertEqual(analysis.cpu_ticks(raw), 27)
        self.assertEqual(analysis.cpu_ticks(raw, [0]), 10)

    def test_interval_clipping_keeps_state_from_before_window(self):
        rows = [{"unix_ns": 0}, {"unix_ns": 10_000_000_000}, {"unix_ns": 20_000_000_000}]
        self.assertEqual([(row["unix_ns"], seconds) for row, seconds in
                          analysis.intervals(rows, 5_000_000_000, 15_000_000_000)],
                         [(0, 5), (10_000_000_000, 5)])

    def test_persistent_latency_excludes_connection_setup(self):
        result = analysis.communication([
            {"connect_tls_seconds": .02, "residual_seconds": .021, "tcp_retrans_delta": 1},
            {"connect_tls_seconds": 0, "residual_seconds": .001, "tcp_retrans_delta": 0}])
        self.assertEqual(result["persistent_residual_seconds"]["count"], 1)
        self.assertEqual(result["persistent_residual_seconds"]["mean"], .001)
        self.assertEqual(result["tcp_retransmissions"], 1)

    def test_virtual_cpu_numbering_is_not_treated_as_affinity_mapping(self):
        def sample(timestamp, ticks):
            return {"unix_ns": timestamp, "memory": "MemAvailable: 100 kB\n",
                    "stat": f"cpu {ticks} 0 0 0 0 0 0 0\ncpu0 {ticks} 0 0 0 0 0 0 0\n",
                    "cpu_pressure": "some avg10=0 avg60=0 avg300=0 total=0",
                    "memory_pressure": "some avg10=0 avg60=0 avg300=0 total=0",
                    "io_pressure": "some avg10=0 avg60=0 avg300=0 total=0"}
        result = analysis.telemetry_summary([sample(0, 0), sample(1_000_000_000, 100)],
            {"cpu_affinity": [192], "effective_driver_affinity": [192]}, 0, 1_000_000_000)
        self.assertFalse(result["proc_stat_cpu_ids_match_affinity"])
        self.assertNotIn("allocated_cpu_busy_core_equivalents", result)
        self.assertEqual(result["visible_aggregate_cpu_busy_core_equivalents"], 1)
