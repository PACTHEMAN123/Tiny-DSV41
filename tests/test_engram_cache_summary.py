import json
from pathlib import Path
import tempfile
import unittest

from summarize_engram_cache import cache_workload, route_accounting
from test_memory_benchmark_summary import trial


class CacheAccountingTest(unittest.TestCase):
    def test_includes_index_traffic_and_compares_matching_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            entries = []
            for variant, speed in (("baseline", 100), ("cpu_async", 90), ("cpu_cache", 80)):
                report = trial(variant, speed)
                if variant == "cpu_cache":
                    for rank in report["ranks"]:
                        rank["engram"]["1"].update(h2d_bytes=4, cache_metadata_h2d_bytes=6,
                                                     cache_hits=1, cache_misses=1)
                path = folder / f"{variant}.json"
                path.write_text(json.dumps(report))
                entries.append({"case": variant, "repeat": 0, "exit_code": 0, "report": str(path)})
            (folder / "summary.json").write_text(json.dumps(entries))
            result = cache_workload(folder)
            self.assertAlmostEqual(result["cache_to_async_throughput_ratio"]["median"], 80 / 90)
            metrics = result["cache_accounting"]
            self.assertEqual(metrics["cpu_cache"]["hit_fraction"]["median"], .5)
            self.assertEqual(metrics["cpu_cache"]["row_and_cache_metadata_h2d_bytes_per_step"]["median"], 10)
            self.assertEqual(metrics["cpu_async"]["row_and_cache_metadata_h2d_bytes_per_step"]["median"], 8)

    def test_route_accounting_excludes_other_threads_and_outside_waits(self):
        with tempfile.TemporaryDirectory() as directory:
            events = [
                {"name": "dsv41/engram_id_route/table", "cat": "user_annotation", "ts": 100, "dur": 200, "tid": 1},
                {"name": "cudaStreamSynchronize", "cat": "cuda_runtime", "ts": 110, "dur": 30, "tid": 1},
                {"name": "cudaStreamSynchronize", "cat": "cuda_runtime", "ts": 110, "dur": 30, "tid": 2},
                {"name": "cudaStreamSynchronize", "cat": "cuda_runtime", "ts": 290, "dur": 30, "tid": 1},
            ]
            (Path(directory) / "rank-0.json").write_text(json.dumps({"traceEvents": events}))
            result = route_accounting(directory)["ranks"][0]
            self.assertEqual(result["route_host_ms"], .2)
            self.assertEqual(result["operations"]["cudaStreamSynchronize"], {"calls": 1, "host_ms": .03})


if __name__ == "__main__":
    unittest.main()
