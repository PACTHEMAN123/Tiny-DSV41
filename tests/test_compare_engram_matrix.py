"""Standard-library tests; no model, torch, network or GPU needed."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.compare_engram_matrix import CASES, compare_matrix, difference, main


def trial(case):
    cached = case.endswith("cache")
    cpu = case.startswith("cpu")
    variant = "cpu" if case in ("cpu_sync", "cpu_async") else case
    ranks = []
    for rank in range(2):
        updates = {name: dict(max_update=.01, nonzero_updates=4, gradient_nnz=2,
                             global_max_update=.01, global_nonzero_updates=8,
                             global_probed_rows=4, global_gradient_nnz=4) for name in ("1", "24")}
        metrics = {name: dict(lookup_rows=20, h2d_bytes=80 if cpu else 0,
                             d2h_bytes=80 if cpu else 0, resolve_wait_seconds=.001,
                             cache_hits=10 if cached else 0, cache_misses=10 if cached else 0,
                             cache_fill_rows=10 if cached else 0, cache_evictions=0,
                             cache_invalidated_rows=4 if cached else 0,
                             cache_metadata_h2d_bytes=80 if cached else 0) for name in ("1", "24")}
        ranks.append(dict(rank=rank, variant=variant, validation_passed=True,
                          world_size=2, ep=2, cp=1, batch_size=1, microbatches=1, steps=2,
                          sequence_length=128, global_batch=2, optimizer_foreach=False,
                          data={"purpose": "execution-only", "source_data_identity": "same-hash",
                                "quality_evaluation_data": False, "source_repeated": True},
                          engram_config={"layer_ids": [1, 24], "head_dim": 64, "vocab_per_head": 4096},
                          engram_parameters=1000000, losses=[3., 2.9], warmup_losses=[3.1],
                          validation_checks=dict(finite_loss_before_backward=3, finite_gradients_before_step=3,
                                                 update_steps_verified=1),
                          update_validation=dict(dense_global_max_update=.01, tables=updates),
                          table_updates_first_step=copy.deepcopy(updates),
                          parameter_samples={"lm_head.weight": [.02, -.03], "model.layers.1.engram.proj.weight": [.01, .04]},
                          lm_head_sample_update=.01,
                          table_samples={name: {"ids": [1, 3], "values": [[.01, .02], [-.03, .04]]} for name in ("1", "24")},
                          policy=dict(device="cpu" if cpu else "cuda", prefetch="async" if case in ("cpu_async", "cpu_cache") else "sync",
                                      deduplicate=True, staging_bytes=4096, cache_bytes=1024 if cached else 0),
                          cache_allocated_bytes=1024 if cached else 0, engram_metrics=metrics))
    return dict(variant=variant, validation_passed=True, optimizer_foreach=False,
                tokens_per_second=100., supervised_tokens_per_second=30.,
                peak_allocated_gib=50., peak_reserved_gib=52., global_losses=[3., 2.9], ranks=ranks)


class MatrixComparisonTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.summaries = {case: trial(case) for case in CASES}

    def compare(self):
        for case, data in self.summaries.items():
            folder = self.root / case
            folder.mkdir(exist_ok=True)
            (folder / "summary.json").write_text(json.dumps(data))
        return compare_matrix(self.root)

    def test_five_cases_pass_and_rank_and_row_order_are_immaterial(self):
        self.summaries["cpu_async"]["ranks"].reverse()
        for row in self.summaries["cpu_cache"]["ranks"]:
            for table in row["table_samples"].values():
                table["ids"].reverse()
                table["values"].reverse()
        result = self.compare()
        self.assertTrue(result["validation_passed"], result["errors"])
        self.assertEqual(len(result["comparisons"]), 5)
        self.assertIn("not bare Qwen", result["scope"])

    def test_bf16_roundoff_passes_but_large_parameter_drift_fails(self):
        # Adjacent BF16 values around .02: tolerate modest backend rounding.
        self.summaries["cpu_sync"]["ranks"][0]["parameter_samples"]["lm_head.weight"][0] += .000244140625
        self.assertTrue(self.compare()["validation_passed"])
        self.summaries["cpu_sync"]["ranks"][0]["parameter_samples"]["lm_head.weight"][0] += .02
        result = self.compare()
        self.assertFalse(result["validation_passed"])
        self.assertGreater(result["comparisons"]["cpu_sync"]["ranks"]["0"]["parameters"]["lm_head.weight"]["outside_tolerance"], 0)

    def test_loss_drift_detected_independently_of_samples(self):
        summary = self.summaries["gpu_cache"]
        summary["global_losses"][0] += .2
        for row in summary["ranks"]:
            row["losses"][0] += .2
        result = self.compare()
        self.assertFalse(result["validation_passed"])
        self.assertFalse(result["comparisons"]["gpu_cache"]["global_losses"]["passed"])

    def test_missing_rank_failed_checks_nonfinite_and_noop_each_fail(self):
        mutations = (
            lambda summary: summary["ranks"].pop(),
            lambda summary: summary["ranks"][0]["validation_checks"].update(finite_gradients_before_step=0),
            lambda summary: summary["ranks"][0]["parameter_samples"]["lm_head.weight"].append(float("nan")),
            lambda summary: summary["ranks"][0]["update_validation"].update(dense_global_max_update=0),
            lambda summary: summary.update(validation_passed=False),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.summaries = {case: trial(case) for case in CASES}
                mutation(self.summaries["cpu_async"])
                self.assertFalse(self.compare()["validation_passed"])

    def test_every_table_must_update_and_samples_must_identify_same_rows(self):
        for row in self.summaries["cpu_cache"]["ranks"]:
            row["update_validation"]["tables"]["24"]["global_max_update"] = 0
            row["table_updates_first_step"] = copy.deepcopy(row["update_validation"]["tables"])
        self.assertFalse(self.compare()["validation_passed"])
        self.summaries["cpu_cache"] = trial("cpu_cache")
        self.summaries["cpu_cache"]["ranks"][0]["table_samples"]["24"]["ids"] = [2, 3]
        result = self.compare()
        self.assertFalse(result["validation_passed"])
        self.assertTrue(any("row ID sets differ" in error for error in result["errors"]))

    def test_cache_traffic_and_invalidation_required_but_hits_optional(self):
        for row in self.summaries["cpu_cache"]["ranks"]:
            for metrics in row["engram_metrics"].values():
                metrics.update(cache_hits=0, cache_misses=20)
        result = self.compare()
        self.assertTrue(result["validation_passed"], result["errors"])
        self.assertEqual(len(result["variants"]["cpu_cache"]["warnings"]), 2)
        for row in self.summaries["cpu_cache"]["ranks"]:
            row["engram_metrics"]["24"]["cache_invalidated_rows"] = 0
        self.assertFalse(self.compare()["validation_passed"])

    def test_cache_over_budget_wrong_policy_and_counter_mismatch_fail(self):
        mutations = (
            lambda row: row.update(cache_allocated_bytes=3000),
            lambda row: row["engram_metrics"]["1"].update(cache_hits=19),
            lambda row: row["policy"].update(device="cuda"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                self.summaries["cpu_cache"] = trial("cpu_cache")
                mutation(self.summaries["cpu_cache"]["ranks"][0])
                self.assertFalse(self.compare()["validation_passed"])

    def test_workload_identity_and_global_loss_accounting_are_checked(self):
        for row in self.summaries["cpu_sync"]["ranks"]:
            row["data"]["source_data_identity"] = "wrong-hash"
        result = self.compare()
        self.assertFalse(result["validation_passed"])
        self.assertTrue(any("identity differs" in error for error in result["errors"]))
        self.summaries["cpu_sync"] = trial("cpu_sync")
        self.summaries["cpu_sync"]["global_losses"][0] += .01
        self.assertFalse(self.compare()["validation_passed"])

    def test_missing_or_malformed_file_is_failure_and_cli_writes_report(self):
        self.assertFalse(compare_matrix(self.root)["validation_passed"])
        self.compare()
        (self.root / "gpu_cache" / "summary.json").write_text("broken json")
        with patch("builtins.print"):
            self.assertEqual(main([str(self.root)]), 1)
        self.assertFalse(json.loads((self.root / "comparison.json").read_text())["validation_passed"])
        self.compare()
        with patch("builtins.print"):
            self.assertEqual(main([str(self.root)]), 0)

    def test_zero_tolerances_and_malformed_sample_shapes(self):
        self.assertTrue(difference([0.], [0.], atol=0, rtol=0)["passed"])
        self.assertFalse(difference([.1], [0.], atol=0, rtol=0)["passed"])
        with self.assertRaisesRegex(ValueError, "shapes differ"):
            difference([[1, 2]], [1, 2], atol=.01, rtol=.01)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            compare_matrix(self.root, sample_rtol=-1)


if __name__ == "__main__":
    unittest.main()
