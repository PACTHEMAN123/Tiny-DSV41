import unittest

from rollout_manager.analysis.report_inference_trace import union_ns


class TraceCoverageTests(unittest.TestCase):
    def test_overlap_is_counted_once(self):
        self.assertEqual(union_ns([(7, 10), (1, 6), (2, 4), (5, 8), (12, 15)]), 12)
        self.assertEqual(union_ns([]), 0)

    def test_negative_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            union_ns([(8, 4)])
