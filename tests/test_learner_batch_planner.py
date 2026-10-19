"""Capacity cells cover actual lengths without guessing GPU memory."""
import unittest

from dsv41_train.rl.learner_batch_planner import parse_learner_batch_shapes, plan_learner_microbatches


class LearnerBatchPlannerTests(unittest.TestCase):
    def test_fixed_default_preserves_original_order_and_ragged_tail(self):
        plan = plan_learner_microbatches([2047 - index for index in range(32)], 6)
        self.assertEqual([len(row["response_indices"]) for row in plan], [6, 6, 6, 6, 6, 2])
        self.assertEqual([index for row in plan for index in row["response_indices"]], list(range(32)))
        self.assertTrue(all(row["qualified_shape_cell"] is None for row in plan))

    def test_stable_sorted_cells_cover_once_and_respect_actual_bounds(self):
        lengths = [2047, 100, 511, 100, 1023, 255] * 5 + [255, 2047]
        cells = parse_learner_batch_shapes("4:2047,8:1023,16:511,32:255")
        plan = plan_learner_microbatches(lengths, 4, cells)
        order = [index for row in plan for index in row["response_indices"]]
        self.assertEqual(order, sorted(range(32), key=lambda i: (lengths[i], i)))
        self.assertEqual(sorted(order), list(range(32)))
        self.assertEqual(plan, plan_learner_microbatches(lengths, 4, list(reversed(cells))))
        self.assertTrue(any(len(row["response_indices"]) < row["qualified_shape_cell"]["max_batch_size"]
                            for row in plan))
        for row in plan:
            self.assertLessEqual(len(row["response_indices"]), row["qualified_shape_cell"]["max_batch_size"])
            self.assertLessEqual(max(lengths[i] for i in row["response_indices"]),
                                 row["qualified_shape_cell"]["max_model_input_tokens"])

    def test_missing_coverage_and_invalid_cells_fail_closed(self):
        self.assertIsNone(parse_learner_batch_shapes(None))
        for text in ("", "4:0", "33:255", "4:2048", "4:1023,4:1023", "4.0:255", "4:255,", "4: 255"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_learner_batch_shapes(text)
        with self.assertRaisesRegex(ValueError, "cover every actual response"):
            plan_learner_microbatches([255] * 31 + [256], 4, parse_learner_batch_shapes("32:255"))


if __name__ == "__main__":
    unittest.main()
