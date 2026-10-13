import copy
import json
from pathlib import Path
import tempfile
import unittest

from summarize_memory_benchmark import summarize, summarize_trial


def trial(variant, speed):
    return {"variant": variant, "world": 2, "cp": 1, "ep": 1, "steps": 2,
            "warmup": 1, "microbatches": 2, "sequence_length": 16, "config": {},
            "torch": "fixture", "gpu": "fixture", "tokens_per_second": speed,
            "ranks": [{"rank": rank, "load_seconds": 1 + rank, "load_peak_bytes": 100 + rank,
                       "train_peak_bytes": 200 + rank, "allocated_bytes": 150 + rank,
                       "reserved_bytes": 300 + rank, "cpu_max_rss_bytes": 400 + rank,
                       "loading": {"logical_bytes_read": 10}, "losses": [2., 1.],
                       "step_ms": [10. + rank, 12. + rank],
                       "engram": {"1": {"h2d_bytes": 8, "d2h_bytes": 8, "lookup_rows": 2,
                                         "resolve_wait_seconds": .002}}} for rank in range(2)]}


class MemoryBenchmarkSummaryTest(unittest.TestCase):
    def test_peak_is_rank_max_and_transfers_are_global_per_step(self):
        result = summarize_trial(trial("baseline", 100))
        self.assertEqual(result["train_peak_bytes"], 201)
        self.assertEqual(result["engram_h2d_bytes_per_step"], 8)
        self.assertEqual(result["slowest_rank_step_ms"], [11., 13.])

    def test_rejects_missing_ranks_steps_and_nonfinite_losses(self):
        valid = trial("baseline", 100)
        for broken in (lambda row: row["ranks"].pop(),
                       lambda row: row["ranks"][0]["losses"].pop(),
                       lambda row: row["ranks"][0]["losses"].__setitem__(0, float("nan"))):
            report = copy.deepcopy(valid)
            broken(report)
            with self.assertRaises(ValueError):
                summarize_trial(report)

    def test_repeats_loss_comparison_and_incomplete_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            entries = []
            for variant, speeds in (("baseline", [100, 120]), ("cpu", [110, 144])):
                for repeat, speed in enumerate(speeds):
                    report = trial(variant, speed)
                    if variant == "cpu":
                        report["ranks"][0]["losses"][0] += .125
                    path = folder / f"{variant}-{repeat}.json"
                    path.write_text(json.dumps(report))
                    entries.append({"case": variant, "repeat": repeat, "exit_code": 0, "report": str(path)})
            (folder / "summary.json").write_text(json.dumps(entries))
            (folder / "manifest.json").write_text(json.dumps({"schedule": [
                {"case": item["case"], "repeat": item["repeat"]} for item in entries
            ] + [{"case": "direct", "repeat": 0}]}))
            result = summarize(folder)
            cpu = result["variants"]["cpu"]
            self.assertEqual(cpu["metrics"]["tokens_per_second"]["median"], 127)
            self.assertAlmostEqual(cpu["throughput_ratio_to_baseline"]["median"], 1.15)
            self.assertEqual(cpu["max_loss_difference_from_baseline"], .125)
            self.assertEqual(len(result["failures"]), 1)
            self.assertEqual(result["planned_trials"], 5)
            self.assertEqual(result["completed_trials"], 4)
            report["load_profiled"] = True
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "different benchmark workloads"):
                summarize(folder)


if __name__ == "__main__":
    unittest.main()
