"""CPU failure-path tests for the CUDA benchmark's distributed validation."""

import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn

from dsv41_train.runtime import Runtime
from scripts.engram_update_validation import capture_touched_rows
from qwen_engram_sft_benchmark import (
    DenseUpdateValidationError, _append_step_record, _benchmark_dense_probes,
    _capture_dense_updates, _compare_qwen_summaries, _selected_dense_parameters,
    _configure_determinism,
    _verify_dense_updates, _write_error_record,
    _check_backward_inputs, _finite_gradients, _table_update_probes,
    _update_probes, _validate_data, _verify_updates, parse_args,
)


class BenchmarkChecksTest(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime(torch.device("cpu"))
        reduction = patch("torch.distributed.all_reduce")
        self.reduce = reduction.start()
        self.addCleanup(reduction.stop)

    def fixture(self):
        model = nn.Module()
        model.dense = nn.Linear(2, 3)
        model.engram_tables = nn.ModuleDict({"1": nn.Embedding(7, 4, sparse=True),
                                            "24": nn.Embedding(7, 4, sparse=True)})
        optimizer = torch.optim.SGD(model.parameters(), lr=.125)
        loss = model.dense(torch.ones(1, 2)).sum()
        for table in model.engram_tables.values():
            loss = loss + table(torch.tensor([1, 3, 3])).sum()
        loss.backward()
        dense = _update_probes(model)
        tables = _table_update_probes(model.engram_tables, self.runtime)
        full = capture_touched_rows(model.engram_tables, self.runtime)
        selected = {name: p for name, p, _, _ in dense}
        snapshots = _capture_dense_updates(selected, self.runtime)
        return model, optimizer, dense, tables, full, selected, snapshots

    def test_foreach_defaults_on_and_can_be_explicitly_disabled(self):
        required = ["--model-path", "model", "--data", "data.pt", "--output", "out", "--variant", "cpu"]
        self.assertTrue(parse_args(required).optimizer_foreach)
        self.assertFalse(parse_args(required + ["--no-optimizer-foreach"]).optimizer_foreach)
        self.assertTrue(parse_args(required + ["--optimizer-foreach"]).optimizer_foreach)
        args = parse_args(required)
        self.assertEqual((args.seq_len, args.steps, args.warmup, args.ep), (4096, 20, 2, 2))
        self.assertEqual((args.layers, args.vocab_per_head, args.head_dim), ([1], 16000000, 256))
        self.assertFalse(args.deduplicate)
        self.assertFalse(args.deterministic)
        self.assertTrue(parse_args(required + ["--deterministic"]).deterministic)
        self.assertIsNone(args.engram_init_manifest)
        self.assertEqual(parse_args(required + ["--engram-init-manifest", "cache/manifest.json"]).engram_init_manifest,
                         Path("cache/manifest.json"))

    def test_deterministic_default_preserves_existing_process_mode(self):
        prior = (torch.are_deterministic_algorithms_enabled(),
                 torch.is_deterministic_algorithms_warn_only_enabled())
        try:
            for enabled, warn_only in ((False, False), (True, True), (True, False)):
                torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
                with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}), \
                     patch("torch.use_deterministic_algorithms") as setter:
                    effective = _configure_determinism(False)
                setter.assert_not_called()
                self.assertEqual(effective, {"requested": False, "algorithms_enabled": enabled,
                                            "warn_only": warn_only, "cublas_workspace_config": ":16:8"})
        finally:
            torch.use_deterministic_algorithms(prior[0], warn_only=prior[1])

    def test_strict_deterministic_requires_launcher_env_and_uninitialized_cuda(self):
        for config in (None, ":16:8", ""):
            with patch.dict(os.environ, {}, clear=True), patch("torch.cuda.is_initialized", return_value=False), \
                 patch("torch.use_deterministic_algorithms") as setter:
                if config is not None:
                    os.environ["CUBLAS_WORKSPACE_CONFIG"] = config
                with self.assertRaisesRegex(RuntimeError, "launcher CUBLAS"):
                    _configure_determinism(True)
                setter.assert_not_called()
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}), \
             patch("torch.cuda.is_initialized", return_value=True), \
             patch("torch.use_deterministic_algorithms") as setter:
            with self.assertRaisesRegex(RuntimeError, "before CUDA"):
                _configure_determinism(True)
            setter.assert_not_called()

    def test_strict_deterministic_sets_effective_mode_without_changing_rng(self):
        prior = (torch.are_deterministic_algorithms_enabled(),
                 torch.is_deterministic_algorithms_warn_only_enabled())
        rng = torch.random.get_rng_state().clone()
        try:
            torch.use_deterministic_algorithms(False, warn_only=True)
            with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}), \
                 patch("torch.cuda.is_initialized", return_value=False):
                effective = _configure_determinism(True)
            self.assertEqual(effective, {"requested": True, "algorithms_enabled": True,
                                        "warn_only": False, "cublas_workspace_config": ":4096:8"})
            self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
        finally:
            torch.use_deterministic_algorithms(prior[0], warn_only=prior[1])

    def test_strict_mode_errors_propagate_without_fallback(self):
        with patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}), \
             patch("torch.cuda.is_initialized", return_value=False), \
             patch("torch.use_deterministic_algorithms", side_effect=RuntimeError("unsupported")):
            with self.assertRaisesRegex(RuntimeError, "unsupported"):
                _configure_determinism(True)

    def test_backward_requires_shifted_targets_and_finite_scalar_loss(self):
        labels = torch.tensor([[-100, 3, -100]])
        _check_backward_inputs(torch.tensor(1., requires_grad=True), labels, self.runtime)
        with self.assertRaisesRegex(RuntimeError, "shifted supervised"):
            _check_backward_inputs(torch.tensor(1.), torch.tensor([[3, -100, -100]]), self.runtime)
        with self.assertRaisesRegex(RuntimeError, "scalar supervised"):
            _check_backward_inputs(None, labels, self.runtime)
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(loss=invalid), self.assertRaisesRegex(RuntimeError, "non-finite.*before backward"):
                _check_backward_inputs(torch.tensor(invalid), labels, self.runtime)

    def test_data_rejects_insufficient_rows_and_zero_shifted_targets(self):
        data = {"input_ids": torch.tensor([[1, 2, 3]]),
                "labels": torch.tensor([[-100, 2, 3]]), "metadata": {}}
        _validate_data(data, 1, self.runtime)
        with self.assertRaisesRegex(RuntimeError, "enough rows"):
            _validate_data(data, 2, self.runtime)
        data["labels"] = torch.tensor([[1, -100, -100]])
        with self.assertRaisesRegex(RuntimeError, "shifted supervised"):
            _validate_data(data, 1, self.runtime)

    def test_packed_sequence_width_alignment_and_padding_are_enforced(self):
        data = {"input_ids": torch.tensor([[1, 2, 3]]),
                "labels": torch.tensor([[-100, 2, 3]]), "metadata": {"packing_padding_tokens": 0}}
        _validate_data(data, 1, self.runtime, 3)
        with self.assertRaisesRegex(RuntimeError, "seq-len"):
            _validate_data(data, 1, self.runtime, 4096)
        data["labels"][0, 2] = 4
        with self.assertRaisesRegex(RuntimeError, "align"):
            _validate_data(data, 1, self.runtime, 3)
        data["labels"][0, 2] = 3
        data["attention_mask"] = torch.tensor([[True, True, False]])
        with self.assertRaisesRegex(RuntimeError, "without padding"):
            _validate_data(data, 1, self.runtime, 3)
        del data["attention_mask"]
        data["metadata"]["packing_padding_tokens"] = 1
        with self.assertRaisesRegex(RuntimeError, "reports padding"):
            _validate_data(data, 1, self.runtime, 3)

    def test_dense_and_uncoalesced_sparse_nonfinite_gradients_fail_before_step(self):
        model, *_ = self.fixture()
        _finite_gradients(model, self.runtime)
        model.dense.weight.grad[0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "optimizer step cancelled"):
            _finite_gradients(model, self.runtime)
        model.dense.weight.grad.fill_(1.)
        table = model.engram_tables["1"]
        table.weight.grad = torch.sparse_coo_tensor(torch.tensor([[1, 1]]),
                                                   torch.full((2, 4), float("inf")), table.weight.shape)
        self.assertFalse(table.weight.grad.is_coalesced())
        with self.assertRaisesRegex(RuntimeError, "optimizer step cancelled"):
            _finite_gradients(model, self.runtime)

    def test_peer_rank_failure_is_propagated_even_when_local_gradients_are_finite(self):
        model, *_ = self.fixture()
        self.reduce.side_effect = lambda tensor, **kwargs: tensor.zero_()
        with self.assertRaisesRegex(RuntimeError, "optimizer step cancelled"):
            _finite_gradients(model, self.runtime)

    def test_real_sparse_sgd_updates_pass_but_noop_and_frozen_table_fail(self):
        model, optimizer, dense, tables, full, selected, snapshots = self.fixture()
        with self.assertRaisesRegex(RuntimeError, "ALL selected dense"):
            _verify_updates(dense, model.engram_tables, tables, self.runtime, full_snapshots=full,
                            dense_parameters=selected, dense_snapshots=snapshots)
        optimizer.step()
        result = _verify_updates(dense, model.engram_tables, tables, self.runtime, full_snapshots=full,
                            dense_parameters=selected, dense_snapshots=snapshots)
        self.assertGreater(result["dense_global_max_update"], 0)
        for record in result["tables"].values():
            self.assertGreater(record["global_max_update"], 0)
            self.assertGreater(record["global_nonzero_updates"], 0)
            self.assertEqual(record["global_probed_rows"], 2)
        rows, before, _ = tables["24"]
        with torch.no_grad():
            model.engram_tables["24"].weight[rows] = before
        with self.assertRaisesRegex(RuntimeError, "24"):
            _verify_updates(dense, model.engram_tables, tables, self.runtime, full_snapshots=full,
                            dense_parameters=selected, dense_snapshots=snapshots)

    def test_idle_local_table_owner_can_pass_when_peer_has_a_real_update(self):
        model = nn.Module()
        model.dense = nn.Linear(2, 3)
        tables = {"1": nn.Embedding(7, 4, sparse=True)}
        probes = _update_probes(model)
        table_probes = _table_update_probes(tables, self.runtime)
        self.assertEqual(table_probes, {})
        with torch.no_grad():
            model.dense.weight.add_(.25)

        def peer_update(tensor, op):
            if op == dist.ReduceOp.MAX and tensor.item() == 0:
                tensor.fill_(.5)
            elif op == dist.ReduceOp.SUM:
                tensor.copy_(torch.tensor([4, 1, 1]))
        self.reduce.side_effect = peer_update
        # Global full-row reductions are covered by the shared helper's tests;
        # this checks that Qwen still reports local and peer bounded diagnostics.
        full = object()
        with patch("qwen_engram_sft_benchmark._verify_dense_updates", return_value={}), \
             patch("qwen_engram_sft_benchmark.verify_touched_updates", return_value={"1": {}}) as verify:
            record = _verify_updates(probes, tables, table_probes, self.runtime,
                                     full_snapshots=full, dense_parameters={}, dense_snapshots={})["tables"]["1"]
            verify.assert_called_once_with(tables, full, self.runtime, chunk_rows=8192)
        self.assertEqual(record["max_update"], 0.)
        self.assertEqual(record["global_max_update"], .5)
        self.assertEqual(record["global_nonzero_updates"], 4)

    def test_nonfinite_update_fails_even_after_finite_gradients(self):
        model, optimizer, dense, tables, full, selected, snapshots = self.fixture()
        _finite_gradients(model, self.runtime)
        optimizer.step()
        rows, _, _ = tables["1"]
        with torch.no_grad():
            model.engram_tables["1"].weight[rows[0], 0] = float("inf")
        with self.assertRaisesRegex(RuntimeError, "non-finite Engram table 1 update"):
            _verify_updates(dense, model.engram_tables, tables, self.runtime, full_snapshots=full,
                            dense_parameters=selected, dense_snapshots=snapshots)

    def large_bf16_fixture(self, *, one_updating_row):
        model = nn.Module()
        model.dense = nn.Linear(2, 3)
        model.engram_tables = nn.ModuleDict({"1": nn.Embedding(1025, 4, sparse=True,
                                                             dtype=torch.bfloat16)})
        table = model.engram_tables["1"]
        with torch.no_grad():
            table.weight.fill_(1.)
        model.dense(torch.ones(1, 2)).sum().backward()
        values = torch.full((1025, 4), 1e-5, dtype=torch.bfloat16)
        if one_updating_row:
            values[1] = .25
        table.weight.grad = torch.sparse_coo_tensor(torch.arange(1025).view(1, -1),
                                                    values, table.weight.shape)
        dense = _update_probes(model)
        bounded = _table_update_probes(model.engram_tables, self.runtime)
        self.assertNotIn(1, bounded["1"][0].tolist())
        full = capture_touched_rows(model.engram_tables, self.runtime)
        selected = {name: p for name, p, _, _ in dense}
        snapshots = _capture_dense_updates(selected, self.runtime)
        optimizer = torch.optim.SGD(model.parameters(), lr=.125)
        optimizer.step()
        return model, dense, bounded, full, selected, snapshots

    def test_real_bf16_update_outside_bounded_rows_passes_complete_proof(self):
        model, dense, bounded, full, selected, snapshots = self.large_bf16_fixture(one_updating_row=True)
        result = _verify_updates(dense, model.engram_tables, bounded, self.runtime,
                                 full_snapshots=full, dense_parameters=selected, dense_snapshots=snapshots)
        self.assertEqual(result["tables"]["1"]["global_max_update"], 0.)
        proof = result["full_table_updates"]["1"]
        self.assertTrue(proof["all_unique_rows_checked"])
        self.assertEqual(proof["global_rows_checked"], 1025)
        self.assertEqual(proof["global_elements_checked"], 4100)
        self.assertEqual(proof["global_changed_elements"], 4)
        self.assertEqual(proof["global_max_abs_delta"], .03125)

    def test_full_bf16_noop_is_rejected_despite_nonzero_sparse_gradient(self):
        model, dense, bounded, full, selected, snapshots = self.large_bf16_fixture(one_updating_row=False)
        self.assertTrue(model.engram_tables["1"].weight.grad._values().ne(0).all())
        with self.assertRaisesRegex(RuntimeError, "update"):
            _verify_updates(dense, model.engram_tables, bounded, self.runtime,
                            full_snapshots=full, dense_parameters=selected, dense_snapshots=snapshots)

    def test_nonfinite_unsampled_row_is_rejected_by_complete_proof(self):
        model, dense, bounded, full, selected, snapshots = self.large_bf16_fixture(one_updating_row=True)
        with torch.no_grad():
            model.engram_tables["1"].weight[1, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "finite"):
            _verify_updates(dense, model.engram_tables, bounded, self.runtime,
                            full_snapshots=full, dense_parameters=selected, dense_snapshots=snapshots)

    def selected_dense_fixture(self, *, one_update=True, dtype=torch.bfloat16):
        model = nn.Module()
        model.lm_head = nn.Linear(1025, 1, bias=False, dtype=dtype)
        model.model = nn.Module()
        model.model.norm = nn.LayerNorm(2, bias=False, dtype=dtype)
        layer = nn.Module()
        layer.engram = nn.Module()
        layer.engram.proj = nn.Linear(2, 2, bias=False, dtype=dtype)
        model.model.layers = nn.ModuleList([layer])
        # An unrelated trainable tensor must remain outside the stated scope.
        model.other = nn.Linear(2, 2, bias=False, dtype=dtype)
        selected = _selected_dense_parameters(model)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.fill_(1.)
                parameter.grad = torch.full_like(parameter, 1e-5)
            if one_update:
                model.lm_head.weight.grad[0, 1] = .25
        bounded = _benchmark_dense_probes(model)
        self.assertNotIn(1, bounded[0][2].tolist())
        snapshots = _capture_dense_updates(selected, self.runtime, chunk_elements=7)
        torch.optim.SGD(model.parameters(), lr=.125).step()
        return model, selected, bounded, snapshots

    def test_unsampled_dense_bf16_head_update_passes_when_entire_norm_and_proj_noop(self):
        model, selected, bounded, snapshots = self.selected_dense_fixture()
        for _, parameter, indices, before in bounded:
            self.assertTrue(torch.equal(parameter.detach().flatten()[indices].float(), before))
        proof = _verify_dense_updates(selected, snapshots, self.runtime, chunk_elements=7)
        self.assertEqual(proof["global_changed_elements"], 1)
        self.assertEqual(proof["global_max_abs_delta"], .03125)
        self.assertEqual(proof["global_elements_checked"], 1031)
        self.assertEqual(proof["snapshot_allocated_bytes_local"], 1031 * 2)
        self.assertNotIn("other.weight", proof["parameter_names"])
        self.assertEqual(proof["parameters"]["lm_head.weight"]["local_shape"], [1, 1025])
        for name in ("model.norm.weight", "model.layers.0.engram.proj.weight"):
            self.assertEqual(proof["parameters"][name]["global_changed_elements"], 0)

    def test_full_dense_bf16_noop_rejects_nonzero_gradient_and_preserves_proof(self):
        model, selected, _, snapshots = self.selected_dense_fixture(one_update=False)
        self.assertTrue(all(parameter.grad.ne(0).all() for parameter in selected.values()))
        with self.assertRaises(DenseUpdateValidationError) as context:
            _verify_dense_updates(selected, snapshots, self.runtime, chunk_elements=7)
        proof = context.exception.diagnostics
        self.assertEqual(proof["global_elements_checked"], 1031)
        self.assertEqual(proof["global_changed_elements"], 0)
        self.assertEqual(proof["global_max_abs_delta"], 0)

    def test_dense_unsampled_nonfinite_and_storage_replacement_are_rejected(self):
        for invalid in (float("nan"), float("inf")):
            model, selected, _, snapshots = self.selected_dense_fixture()
            with torch.no_grad():
                model.lm_head.weight[0, 1] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(RuntimeError, "non-finite weight/update"):
                _verify_dense_updates(selected, snapshots, self.runtime, chunk_elements=7)
            with self.assertRaisesRegex(RuntimeError, "before optimizer"):
                _capture_dense_updates(selected, self.runtime, chunk_elements=7)
        model, selected, _, snapshots = self.selected_dense_fixture()
        model.lm_head.weight.data = model.lm_head.weight.detach().clone()
        with self.assertRaisesRegex(RuntimeError, "storage changed"):
            _verify_dense_updates(selected, snapshots, self.runtime)

    def test_chunked_fp32_comparison_does_not_modify_live_parameters_or_snapshots(self):
        _, selected, _, snapshots = self.selected_dense_fixture(dtype=torch.float32)
        current = {name: parameter.detach().clone() for name, parameter in selected.items()}
        original = {name: record[0].clone() for name, record in snapshots.items()}
        for _ in range(2):
            proof = _verify_dense_updates(selected, snapshots, self.runtime, chunk_elements=7)
            self.assertGreater(proof["global_changed_elements"], 0)
            for name, parameter in selected.items():
                self.assertTrue(torch.equal(parameter, current[name]))
                self.assertTrue(torch.equal(snapshots[name][0], original[name]))
                self.assertNotEqual(parameter.data_ptr(), snapshots[name][0].data_ptr())

    def test_durable_steps_survive_later_failure_and_error_keeps_complete_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            _append_step_record(directory, 0, {"step": 0, "warmup": True, "finite": True})
            _append_step_record(directory, 0, {"step": 1, "warmup": False, "finite": True})
            _, selected, _, snapshots = self.selected_dense_fixture(one_update=False)
            try:
                _verify_dense_updates(selected, snapshots, self.runtime)
            except DenseUpdateValidationError as error:
                _write_error_record(directory, 0, {"step": 2, "phase": "after_update_validation"}, error)
                with self.assertRaises(FileExistsError):
                    _write_error_record(directory, 0, {"step": 2}, error)
            rows = [json.loads(line) for line in (Path(directory) / "steps-rank-0.jsonl").read_text().splitlines()]
            self.assertEqual([row["step"] for row in rows], [0, 1])
            failure = json.loads((Path(directory) / "error-rank-0.json").read_text())
            self.assertEqual(failure["step"], 2)
            self.assertEqual(failure["diagnostics"]["global_changed_elements"], 0)
            self.assertEqual(failure["diagnostics"]["global_elements_checked"], 1031)
            self.assertIn("DenseUpdateValidationError", failure["traceback"])

    def test_three_step_cuda_smoke_cli_preserves_4096_and_two_warmups(self):
        common = ["--model-path", "model", "--data", "data.pt", "--output", "out", "--steps", "3",
                  "--warmup", "2", "--seq-len", "4096", "--vocab-per-head", "32768", "--ep", "2",
                  "--layers", "1", "--head-dim", "256", "--staging-mib", "128", "--microbatches", "2",
                  "--no-optimizer-foreach", "--no-deduplicate"]
        for variant in ("gpu", "cpu"):
            args = parse_args([*common, "--variant", variant, "--prefetch", "sync"])
            self.assertEqual((args.steps, args.warmup, args.seq_len, args.ep), (3, 2, 4096, 2))
            self.assertEqual(args.learning_rate, .01)
            self.assertFalse(args.optimizer_foreach)

    def test_full_dense_reference_keeps_old_tests_and_tolerances(self):
        _, selected, _, snapshots = self.selected_dense_fixture()
        proof = _verify_dense_updates(selected, snapshots, self.runtime)
        expected = {"ranks": [{"rank": 0, "step_records": [{"step": 0, "full_dense_updates": proof}]}]}
        actual = copy.deepcopy(expected)
        tolerances = dict(loss_atol=.02, loss_rtol=.01, sample_atol=.0002, sample_rtol=.02)
        with patch("qwen_engram_sft_benchmark.compare_summaries", return_value={"validation_passed": True}) as previous:
            result = _compare_qwen_summaries(actual, expected, **tolerances)
            previous.assert_called_once_with(actual, expected, **tolerances)
        self.assertTrue(result["validation_passed"])
        with patch("qwen_engram_sft_benchmark.compare_summaries", return_value={"validation_passed": False}):
            self.assertFalse(_compare_qwen_summaries(actual, expected, **tolerances)["validation_passed"])
        actual["ranks"][0]["step_records"][0]["full_dense_updates"]["parameters"]["lm_head.weight"]["global_max_abs_delta"] = .5
        with patch("qwen_engram_sft_benchmark.compare_summaries", return_value={"validation_passed": True}):
            self.assertFalse(_compare_qwen_summaries(actual, expected, **tolerances)["validation_passed"])
        actual = copy.deepcopy(expected)
        actual["ranks"][0]["step_records"][0]["full_dense_updates"]["global_changed_elements"] = 0
        with patch("qwen_engram_sft_benchmark.compare_summaries", return_value={"validation_passed": True}), \
             self.assertRaisesRegex(ValueError, "proof failed"):
            _compare_qwen_summaries(actual, expected, **tolerances)


if __name__ == "__main__":
    unittest.main()
