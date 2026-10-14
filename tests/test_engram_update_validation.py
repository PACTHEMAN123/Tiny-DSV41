import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from scripts.engram_update_validation import capture_touched_rows, verify_touched_updates


def table(rows=1024, width=8, dtype=torch.bfloat16):
    return SimpleNamespace(weight=torch.nn.Parameter(torch.ones(rows, width, dtype=dtype)))


def sparse_gradient(target, row_ids, values):
    target.weight.grad = torch.sparse_coo_tensor(torch.tensor([row_ids]), values.to(target.weight.dtype), target.weight.shape)


class FullTouchedUpdateTest(unittest.TestCase):
    def test_update_outside_first256_rows_is_proved_without_changing_bounded_semantics(self):
        target = table()
        sparse_gradient(target, list(range(512)), torch.ones(512, 8))
        snapshot = capture_touched_rows({"1": target}, chunk_rows=17)
        with torch.no_grad():
            target.weight[511, 7] -= .25
        self.assertTrue(torch.equal(target.weight[:256], snapshot["1"].before[:256]))
        result = verify_touched_updates({"1": target}, snapshot, chunk_rows=13)["1"]
        self.assertEqual(result["global_rows_checked"], 512)
        self.assertEqual(result["global_elements_checked"], 4096)
        self.assertEqual(result["global_changed_elements"], 1)
        self.assertEqual(result["global_max_abs_delta"], .25)
        self.assertTrue(result["all_unique_rows_checked"])

    def test_duplicate_rows_snapshot_original_dtype_and_sparse_working_set_only(self):
        target = table(rows=100000, width=16)
        sparse_gradient(target, [3, 99999, 3], torch.ones(3, 16))
        snapshots = capture_touched_rows({"1": target}, chunk_rows=1)
        self.assertEqual(snapshots["1"].before.dtype, torch.bfloat16)
        self.assertEqual(snapshots["1"].before.shape, (2, 16))
        with torch.no_grad():
            target.weight[3, 2] = .5
            target.weight[99999, 8] = .25
        result = verify_touched_updates({"1": target}, snapshots, chunk_rows=1)["1"]
        self.assertEqual(result["before_snapshot_bytes_local"], 64)
        self.assertLessEqual(result["snapshot_allocated_bytes_local"], 88)
        self.assertEqual(result["global_changed_elements"], 2)
        self.assertEqual(result["global_raw_gradient_nnz"], 3)

    def test_real_bfloat16_rounding_to_no_update_still_fails(self):
        target = table(rows=4)
        sparse_gradient(target, [2], torch.full((1, 8), 1e-8))
        snapshot = capture_touched_rows({"1": target})
        torch.optim.SGD([target.weight], lr=.01).step()
        with self.assertRaisesRegex(RuntimeError, "no actual numerical update in ALL 1"):
            verify_touched_updates({"1": target}, snapshot)

    def test_nonfinite_touched_values_and_dense_gradients_fail(self):
        target = table(rows=4)
        target.weight.grad = torch.ones_like(target.weight)
        with self.assertRaisesRegex(RuntimeError, "row-sparse"):
            capture_touched_rows({"1": target})
        sparse_gradient(target, [2], torch.ones(1, 8))
        snapshot = capture_touched_rows({"1": target})
        with torch.no_grad():
            target.weight[2, 5] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            verify_touched_updates({"1": target}, snapshot)
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            capture_touched_rows({"1": target})

    def test_no_gradient_and_signed_zero_only_are_not_false_update_proofs(self):
        target = table(rows=4)
        snapshot = capture_touched_rows({"1": target})
        with self.assertRaisesRegex(RuntimeError, "no actual numerical update in ALL 0"):
            verify_touched_updates({"1": target}, snapshot)
        with torch.no_grad():
            target.weight[2].zero_()
        sparse_gradient(target, [2], torch.ones(1, 8))
        snapshot = capture_touched_rows({"1": target})
        with torch.no_grad():
            target.weight[2] = -target.weight[2]
        with self.assertRaisesRegex(RuntimeError, "no actual numerical update"):
            verify_touched_updates({"1": target}, snapshot)

    def test_each_table_must_have_actual_update(self):
        first, second = table(rows=4), table(rows=4)
        for target in (first, second):
            sparse_gradient(target, [0], torch.ones(1, 8))
        tables = {"first": first, "second": second}
        snapshot = capture_touched_rows(tables)
        with torch.no_grad():
            first.weight[0, 0] = 2
        with self.assertRaisesRegex(RuntimeError, "table second: no actual"):
            verify_touched_updates(tables, snapshot)

    def test_global_proof_accepts_empty_local_owner_only_when_other_owner_updated(self):
        target = table(rows=4)
        runtime = SimpleNamespace(device=torch.device("cpu"), world_size=2)
        def other_rank(value, op):
            if op == torch.distributed.ReduceOp.SUM:
                value.add_(torch.tensor([1, 8, 2, 16, 8, 1]))
            elif op == torch.distributed.ReduceOp.MAX:
                value.fill_(.25)
        with patch("torch.distributed.is_initialized", return_value=True), patch("torch.distributed.all_reduce", side_effect=other_rank):
            snapshot = capture_touched_rows({"1": target}, runtime)
            record = verify_touched_updates({"1": target}, snapshot, runtime)["1"]
        self.assertEqual(record["local_changed_elements"], 0)
        self.assertEqual(record["global_changed_elements"], 2)
        self.assertEqual(record["global_rows_checked"], 1)
        self.assertEqual(record["global_max_abs_delta"], .25)
        self.assertEqual(record["snapshot_allocated_bytes_global"], 24)

    def test_reference_requires_full_proof_even_when_legacy_bounded_delta_is_zero(self):
        import copy
        from test_dsv4_engram_sft_benchmark import fixture_summary
        from scripts.benchmark_dsv4_engram_sft import compare_summaries
        summary = fixture_summary()
        summary["workload"]["update_validation"] = {"method": "all-unique-touched-rows-v1"}
        step = summary["ranks"][0]["steps"][0]
        step["table_updates"]["1"]["parameter_delta_max"] = 0.
        step["full_table_updates"] = {"1": {"scope": "all_unique_sparse_touched_rows_all_columns",
            "all_unique_rows_checked": True, "finite_before_after": True, "global_changed_elements": 5,
            "global_max_abs_delta": .01, "global_rows_checked": 500, "global_elements_checked": 4000}}
        self.assertTrue(compare_summaries(summary, copy.deepcopy(summary))["validation_passed"])
        candidate = copy.deepcopy(summary)
        candidate["ranks"][0]["steps"][0]["full_table_updates"]["1"]["global_changed_elements"] = 0
        with self.assertRaisesRegex(ValueError, "full touched-update proof failed"):
            compare_summaries(candidate, summary)


if __name__ == "__main__":
    unittest.main()
