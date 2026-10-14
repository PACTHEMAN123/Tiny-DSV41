import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rollout_manager.adapters.process import run_process
from rollout_manager.cpu_layout import controlled_layout
from rollout_manager.runner import Journal, make_pool, reservation


class WorkerResourceTests(unittest.TestCase):
    def layout(self, evaluation_cores):
        cpus = list(range(64, 128)) + list(range(192, 256))
        def topology(path):
            if path.name == "physical_package_id":
                return "0"
            return str(int(path.parent.parent.name[3:]) % 128)
        with patch.object(Path, "read_text", autospec=True, side_effect=topology):
            return controlled_layout(cpus, 8, 4, 8, evaluation_cores)

    def test_baseline_cpu_layout_and_larger_affinity_use_distinct_cores(self):
        for cores, expected in ((1, 48), (2, 56), (3, 64)):
            builds, evaluators, control = self.layout(cores)
            used = [cpu for chunk in builds + evaluators for cpu in chunk] + control
            self.assertEqual(len(used), expected)
            self.assertEqual(len({cpu % 128 for cpu in used}), expected)
            self.assertEqual([len(chunk) for chunk in evaluators], [cores] * 8)
        with self.assertRaisesRegex(ValueError, "physical cores"):
            self.layout(4)

    def test_uncertain_gpu_is_quarantined_and_waiters_cancel(self):
        with tempfile.TemporaryDirectory() as root:
            journal = Journal(Path(root))
            try:
                pool = make_pool(["gpu"])
                with reservation(pool, journal, "job", "evaluation") as (_, timing):
                    timing["quarantined"] = True
                self.assertTrue(pool.empty())
                with self.assertRaises(RuntimeError):
                    with reservation(pool, journal, "later", "evaluation", lambda: True):
                        self.fail("cancelled job acquired a GPU")
                self.assertEqual(journal.counts["evaluation"]["queued"], 0)
            finally:
                journal.close()

    def test_cancellation_reaps_child_before_return(self):
        with tempfile.TemporaryDirectory() as root:
            start = time.monotonic()
            result = run_process([sys.executable, "-c", "import time;time.sleep(30)"], Path(root),
                dict(os.environ), Path(root)/"child.log", timeout=5, heartbeat=lambda: None,
                cancelled=lambda: time.monotonic()-start > .1)
            self.assertEqual(result["stopped_for"], "cancelled")
            self.assertTrue(result["process_group_stopped"])
            self.assertLess(result["wall_seconds"], 3)
