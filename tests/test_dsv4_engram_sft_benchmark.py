"""CPU validation of the real-checkpoint benchmark's policies and evidence helpers."""

import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.models.dsv4.engram.hash import NgramHash
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from dsv41_train.runtime import Runtime
from post_training_helpers import small_config
from scripts.benchmark_dsv4_engram_sft import (
    _engram_update_metrics, _engram_update_probes, _finite_gradients,
    compare_summaries, dense_probes, dense_updates, final_samples, gradient_summary,
    numeric_difference, parse_args, policies, selected_tables, validate_storage_metrics,
    hash_layout, metric_delta, metric_snapshot, timing_statistics,
)


def fixture_summary():
    row = {"rank": 0,
           "final_samples": {"dense_nonembedding": {"model.norm.weight": {"indices": [0, 3], "values": [1., 1.1]}},
                             "tables": {"1": {"global_row_ids": [4], "columns": [0, 1], "values": [[.01, .02]]}}},
           "steps": [{"step": 0, "global_supervised_tokens": 4, "global_input_tokens": 8,
                      "gradients": {"dense_bounded_samples": {"model.norm.weight": {"sample_l2": .1, "sample_max_abs": .02}},
                                    "tables_full_touched_gradients": {"1": {"l2": .2, "max_abs": .03}}},
                      "dense_update_global_max": .001,
                      "table_updates": {"1": {"parameter_delta_max": .0001}}}]}
    return {"validation_passed": True, "workload": {"source": "actual-source-hash", "data": "same-token-hash"},
            "global_losses_all_steps": [3.], "ranks": [row]}


def storage_row(cached=True):
    return {"cache_allocated_bytes": 1024 if cached else 0, "pinned_staging_allocated_bytes": 2048,
            "engram_metrics": {"1": dict(lookup_rows=10, h2d_bytes=1024, d2h_bytes=2048, resolve_wait_seconds=.01,
                                         cache_hits=4 if cached else 0, cache_misses=6 if cached else 0,
                                         cache_fill_rows=5 if cached else 0, cache_evictions=1 if cached else 0,
                                         cache_invalidated_rows=3 if cached else 0,
                                         cache_metadata_h2d_bytes=64 if cached else 0)}}


class DSV4EngramBenchmarkTest(unittest.TestCase):
    def test_long_sequence_defaults_and_measured_step_statistics(self):
        args = parse_args(["--model-path", "m", "--data", "d", "--output", "o", "--variant", "gpu"])
        self.assertEqual((args.seq_len, args.warmup, args.steps), (4096, 2, 20))
        self.assertTrue(args.require_full_sequences)
        result = timing_statistics([1., 2., 9., 3.])
        self.assertEqual(result["median_seconds"], 2.5)
        self.assertEqual(result["total_seconds"], 15.)
        self.assertAlmostEqual(result["p90_seconds"], 7.2)
        for bad in ([], [0.], [float("nan")]):
            with self.assertRaises(ValueError):
                timing_statistics(bad)

    def test_metrics_are_copied_and_differenced_without_silent_resets(self):
        table = SimpleNamespace(metrics={"lookup_rows": 5, "cache_hits": 1})
        before = metric_snapshot({"1": table})
        table.metrics.update(lookup_rows=8, cache_hits=2)
        self.assertEqual(metric_delta(metric_snapshot({"1": table}), before),
                         {"1": {"lookup_rows": 3, "cache_hits": 1}})
        table.metrics["lookup_rows"] = 0
        with self.assertRaisesRegex(ValueError, "reset"):
            metric_delta(metric_snapshot({"1": table}), before)

    def test_hash_layout_identifies_exact_rows_and_token_map(self):
        hashing = NgramHash(small_config())
        rows = int(hashing.primes.sum())
        tables = {"1": SimpleNamespace(global_num_embeddings=rows, embedding_dim=8)}
        first = hash_layout(hashing, [1], tables)
        self.assertEqual(first["table_rows"], {"1": rows})
        self.assertEqual(len(first["token_map_sha256_int64"]), 64)
        hashing.token_map[1] = 0
        self.assertNotEqual(first["token_map_sha256_int64"], hash_layout(hashing, [1], tables)["token_map_sha256_int64"])
        tables["1"].global_num_embeddings += 1
        with self.assertRaisesRegex(ValueError, "partition sum"):
            hash_layout(hashing, [1], tables)

    def test_real_prefix_requires_an_actual_engram_table_and_preserves_rows(self):
        config = small_config(engram_num_embeddings=[384006168])
        self.assertEqual(selected_tables(config, 0, 2), {"1": 384006168})
        with self.assertRaisesRegex(ValueError, "contains no Engram"):
            selected_tables(config, 0, 1)
        with self.assertRaisesRegex(ValueError, "KV source"):
            selected_tables(config, 2, 1)
        with self.assertRaisesRegex(ValueError, "outside checkpoint"):
            selected_tables(config, 0, 6)

    def test_policies_keep_backbone_loading_and_only_change_storage(self):
        base = ["--model-path", "model", "--data", "data", "--output", "out"]
        expected = {"gpu": ("cuda", "sync", 0), "cpu_sync": ("cpu", "sync", 0),
                    "cpu_async": ("cpu", "async", 0), "cpu_cache": ("cpu", "async", 8 * 1024**2)}
        for variant, wanted in expected.items():
            args = parse_args(base + ["--variant", variant])
            storage, loading = policies(args)
            self.assertEqual((storage.device, storage.prefetch, storage.cache_bytes), wanted)
            self.assertEqual(loading.mode, "direct_shard")
            self.assertTrue(loading.pipeline)
            self.assertFalse(args.optimizer_foreach)
        args = parse_args(base + ["--variant", "cpu_cache", "--cache-mib", "128"])
        self.assertEqual(policies(args)[0].cache_bytes, 128 * 1024**2)

    def test_native_masked_sft_sparse_sgd_produces_finite_gradients_and_actual_updates(self):
        torch.manual_seed(8)
        config = small_config(num_hidden_layers=2, compress_ratios=[0, 0], kv_source_layer_ids=[],
                              index_source_layer_ids=[], candidate_source_layer_id=-1, router_aux_loss_coef=0)
        model = DeepSeekV41ForCausalLM(config, sparse_engram_gradients=True,
                                      engram_policy=EngramPolicy("cpu", "sync"))
        self.addCleanup(model.close)
        optimizer = build_optimizer(model, OptimizerPolicy("sgd", .1))
        probes = dense_probes(model)
        self.assertTrue(all("embedding" not in name and "lm_head" not in name for name, *_ in probes))
        ids = torch.tensor([[1, 2, 3, 4, 0, 0]])
        labels = torch.tensor([[-100, -100, 3, 4, -100, -100]])
        mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
        result = model(ids, labels=labels, attention_mask=mask)
        expected = torch.nn.functional.cross_entropy(result.logits[:, 1:3].reshape(-1, 32), torch.tensor([3, 4]))
        torch.testing.assert_close(result.loss, expected)
        result.loss.backward()
        runtime = Runtime(torch.device("cpu"))
        tables = model.model.engram_tables
        with patch("torch.distributed.all_reduce"):
            _finite_gradients(model, runtime)
            stats = gradient_summary(probes, tables)
            touched = _engram_update_probes(tables)
            optimizer.step()
            updates = _engram_update_metrics(tables, touched, runtime)
        self.assertGreater(stats["tables_full_touched_gradients"]["1"]["l2"], 0)
        self.assertGreater(updates["1"]["parameter_delta_max"], 0)
        self.assertGreater(max(dense_updates(probes).values()), 0)
        samples = final_samples(probes, tables, touched)
        self.assertEqual(len(samples["bounded_sample_sha256"]), 64)
        self.assertTrue(samples["tables"]["1"]["global_row_ids"])
        parameter = probes[0][1]
        parameter.grad.reshape(-1)[0] = float("nan")
        with patch("torch.distributed.all_reduce"), self.assertRaisesRegex(RuntimeError, "cancelled"):
            _finite_gradients(model, runtime)

    def test_storage_metrics_require_transfer_invalidation_and_budget_compliance(self):
        policy = EngramPolicy("cpu", "async", staging_bytes=4096, cache_bytes=2048)
        row = storage_row()
        result = validate_storage_metrics([row], policy, ["1"])
        self.assertEqual(result["1"]["cache_hit_fraction"], .4)
        self.assertEqual(result["1"]["row_and_cache_metadata_h2d_bytes"], 1088)
        for key, value in (("h2d_bytes", 0), ("cache_invalidated_rows", 0), ("cache_hits", 99)):
            bad = copy.deepcopy(row)
            bad["engram_metrics"]["1"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_storage_metrics([bad], policy, ["1"])
        with self.assertRaisesRegex(ValueError, "budget"):
            validate_storage_metrics([dict(row, cache_allocated_bytes=4096)], policy, ["1"])
        no_hits = copy.deepcopy(row)
        no_hits["engram_metrics"]["1"].update(cache_hits=0, cache_misses=10)
        self.assertEqual(validate_storage_metrics([no_hits], policy, ["1"])["1"]["cache_hit_fraction"], 0)

    def test_no_cache_rejects_stale_cache_counters(self):
        policy = EngramPolicy("cpu", "sync", staging_bytes=4096)
        row = storage_row(False)
        validate_storage_metrics([row], policy, ["1"])
        row["engram_metrics"]["1"]["cache_hits"] = 1
        with self.assertRaisesRegex(ValueError, "disabled"):
            validate_storage_metrics([row], policy, ["1"])

    def test_reference_comparison_detects_loss_gradient_update_and_sample_drift(self):
        reference = fixture_summary()
        self.assertTrue(compare_summaries(copy.deepcopy(reference), reference)["validation_passed"])
        mutations = (
            lambda value: value["global_losses_all_steps"].__setitem__(0, 4.),
            lambda value: value["ranks"][0]["steps"][0]["gradients"]["tables_full_touched_gradients"]["1"].update(l2=.4),
            lambda value: value["ranks"][0]["steps"][0].update(dense_update_global_max=.1),
            lambda value: value["ranks"][0]["final_samples"]["tables"]["1"].update(values=[[.5, .02]]),
        )
        for mutation in mutations:
            actual = copy.deepcopy(reference)
            mutation(actual)
            self.assertFalse(compare_summaries(actual, reference)["validation_passed"])
        actual = copy.deepcopy(reference)
        actual["ranks"][0]["final_samples"]["tables"]["1"]["values"][0][0] += .0001
        self.assertTrue(compare_summaries(actual, reference)["validation_passed"])

    def test_reference_requires_same_data_and_touched_row_ids(self):
        reference = fixture_summary()
        actual = copy.deepcopy(reference)
        actual["workload"]["data"] = "wrong"
        with self.assertRaisesRegex(ValueError, "identity"):
            compare_summaries(actual, reference)
        actual = copy.deepcopy(reference)
        actual["ranks"][0]["final_samples"]["tables"]["1"]["global_row_ids"] = [5]
        with self.assertRaisesRegex(ValueError, "locations"):
            compare_summaries(actual, reference)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            numeric_difference([float("nan")], [1.], .01, .01)

    def test_reference_accepts_qwen_step_records_and_rejects_hash_or_column_drift(self):
        reference = fixture_summary()
        row = reference["ranks"][0]
        row["step_records"] = row.pop("steps")
        row["steps"] = 1
        self.assertTrue(compare_summaries(copy.deepcopy(reference), reference)["validation_passed"])
        changed = copy.deepcopy(reference)
        changed["engram_format"] = {"token_map_sha256_int64": "different"}
        with self.assertRaisesRegex(ValueError, "layout"):
            compare_summaries(changed, reference)
        changed = copy.deepcopy(reference)
        changed["ranks"][0]["final_samples"]["tables"]["1"]["columns"] = [1, 2]
        with self.assertRaisesRegex(ValueError, "columns"):
            compare_summaries(changed, reference)


if __name__ == "__main__":
    unittest.main()
