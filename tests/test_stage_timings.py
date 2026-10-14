import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rollout_manager.analysis.stage_timings import StageTimings


class StageTimingTests(unittest.TestCase):
    def test_wall_clock_rollback_does_not_change_measured_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stages.jsonl"
            timings = StageTimings(path, clock_domain="test_process_monotonic")
            with patch("rollout_manager.analysis.stage_timings.time.time_ns", side_effect=[50, 40]), \
                    patch("rollout_manager.analysis.stage_timings.time.monotonic_ns", side_effect=[100, 2_000_000_100]):
                self.assertEqual(timings.call("operation", lambda: 7, details={"step": 1}), 7)
            record = json.loads(path.read_text())
            self.assertEqual(record["duration_seconds"], 2.)
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["completion"], "host_call_return")
            self.assertEqual(record["details"], {"step": 1})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_failure_record_preserves_the_original_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stages.jsonl"
            timings = StageTimings(path, clock_domain="test_process_monotonic")
            error = RuntimeError("private diagnostic text")
            with self.assertRaises(RuntimeError) as raised:
                with timings.interval("failed_operation"):
                    raise error
            self.assertIs(raised.exception, error)
            record = json.loads(path.read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["error_type"], "RuntimeError")
            self.assertNotIn("private diagnostic text", path.read_text())
            self.assertGreaterEqual(record["duration_seconds"], 0.)


if __name__ == "__main__":
    unittest.main()
