import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.dispatch import TokenDispatcher
from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from dsv41_train.parallel import ContextParallel
from dsv41_train.runtime import Runtime
from dsv41_train.sharded_state import StateAdapter
from qwen_helpers import tiny_config, write_hf_checkpoint
from post_training_helpers import small_config
from train_sft import (
    _engram_gradient_rows, _engram_summary, _engram_update_metrics, _engram_update_probes,
    _load_model, _probe_indices, _trainable_engram_tables,
    _qwen_engram_metadata, _synchronize_engram, _truncate_resumed_metrics,
    parse_args, qwen_engram_options, validate_args,
)


class SFTProbeIndicesTest(unittest.TestCase):
    def test_exact_integer_indices_cover_large_parameter_boundaries(self):
        # Only the <=256 indices are allocated, never these large parameter tensors.
        for numel in (0, 1, 2, 255, 256, 257, 2**24 - 1, 2**24, 2**24 + 1,
                      2**24 + 2, 155582464, 2**31 + 19, 2**63 - 1):
            with self.subTest(numel=numel):
                indices = _probe_indices(numel)
                size = min(numel, 256)
                expected = ([index * (numel - 1) // (size - 1) for index in range(size)]
                            if size > 1 else list(range(size)))
                self.assertEqual(indices.dtype, torch.int64)
                self.assertEqual(indices.tolist(), expected)
                self.assertEqual(indices.numel(), size)
                if size:
                    self.assertEqual(indices[0].item(), 0)
                    self.assertEqual(indices[-1].item(), numel - 1)
                    self.assertTrue(indices.ge(0).all().item())
                    self.assertTrue(indices.lt(numel).all().item())
                    self.assertEqual(indices.unique().numel(), size)


class SFTArgumentTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        (self.folder / "config.json").write_text(json.dumps(tiny_config().to_dict()))
        (self.folder / "model.safetensors.index.json").write_text('{"weight_map": {}}')
        data = self.folder / "data.jsonl"
        data.write_text('{"input_ids": [1, 2], "labels": [-100, 2]}\n')
        self.args = parse_args([
            "--model-type", "qwen", "--model-path", str(self.folder),
            "--data", str(data), "--output-dir", str(self.folder / "run"),
            "--seq-len", "8",
        ])

    def test_qwen_accepts_full_checkpointed_training(self):
        self.args.checkpoint_mode = "full"
        self.args.ep_size = 2
        config = validate_args(self.args, 2)
        self.assertEqual(config.num_experts, 4)
        self.assertEqual(qwen_engram_options(self.args, config), (None, None))
        self.assertEqual(_qwen_engram_metadata(self.args, config), {})

    def test_identity_run_requires_twenty_steps_and_verifiable_adapter(self):
        self.args.identity_eval_prompts = str(self.folder / "heldout.json")
        Path(self.args.identity_eval_prompts).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'requires --lora --verify-lora --adapter-out'):
            validate_args(self.args, 1)
        self.args.lora = self.args.verify_lora = True
        self.args.adapter_out = str(self.folder / 'adapter.pt')
        with self.assertRaisesRegex(ValueError, 'at least 20'):
            validate_args(self.args, 1)
        self.args.steps = 20
        validate_args(self.args, 1)

    def test_long_training_can_use_bounded_lora_verification(self):
        config = tiny_config().to_dict()
        config['max_position_embeddings'] = 16384
        (self.folder / 'config.json').write_text(json.dumps(config))
        self.args.seq_len = self.args.min_active_tokens_per_example = 8192
        self.args.lora = self.args.verify_lora = True
        self.args.lora_probe_tokens = 128
        validate_args(self.args, 1)

    def test_qwen_engram_accepts_cpu_async_cache_and_gpu_policies(self):
        self.args.qwen_engram = True
        self.args.qwen_engram_layers = [0, 1]
        self.args.qwen_engram_vocab_per_head = 17
        self.args.qwen_engram_head_dim = 4
        self.args.qwen_engram_heads = 2
        self.args.qwen_engram_max_ngram = 3
        self.args.qwen_engram_seed = 91
        self.args.engram_deduplicate = True
        for device, prefetch, cache in (("cuda", "sync", 0), ("cuda", "sync", 1),
                                        ("cpu", "sync", 0), ("cpu", "async", 1)):
            with self.subTest(device=device, prefetch=prefetch, cache=cache):
                self.args.engram_device, self.args.engram_prefetch = device, prefetch
                self.args.engram_cache_mib = cache
                model_config = validate_args(self.args, 1)
                config, policy = qwen_engram_options(self.args, model_config)
                self.assertEqual(config.layer_ids, (0, 1))
                self.assertEqual(config.seed, 91)
                self.assertEqual(policy.cache_bytes, cache * 1024**2)
                self.assertEqual(policy.device, device)
                self.assertEqual(policy.prefetch, prefetch)

    def test_qwen_engram_rejects_unsafe_or_ambiguous_options_before_loading(self):
        for change, message in (
            ({"qwen_engram_layers": [0]}, "requires --qwen-engram"),
            ({"engram_device": "cpu"}, "require --qwen-engram"),
            ({"engram_cache_mib": 1}, "require --qwen-engram"),
            ({"qwen_engram": True}, "outside"),
            ({"qwen_engram": True, "qwen_engram_layers": [0, 0]}, "distinct"),
            ({"qwen_engram": True, "qwen_engram_layers": [0], "optimizer": "adamw"}, "requires --optimizer sgd"),
            ({"qwen_engram": True, "qwen_engram_layers": [0], "weight_decay": .01}, "weight-decay 0"),
            ({"qwen_engram": True, "qwen_engram_layers": [0], "engram_prefetch": "async"}, "requires CPU"),
            ({"qwen_engram": True, "qwen_engram_layers": [0], "engram_cache_mib": -1}, "nonnegative"),
            ({"qwen_engram": True, "qwen_engram_layers": [0], "staging_mib": 0}, "at least 4096"),
            ({"qwen_engram": True, "qwen_engram_layers": [0], "qwen_engram_seed": -1}, "nonnegative"),
            ({"qwen_engram": True, "qwen_engram_layers": [0], "qwen_engram_head_dim": 0}, "positive"),
            ({"qwen_engram": True, "qwen_engram_layers": [0], "qwen_engram_max_ngram": 1}, "n-grams"),
            ({"qwen_engram": True, "qwen_engram_layers": [0], "load_mode": "direct_shard"}, "unsupported"),
            ({"qwen_engram": True, "qwen_engram_layers": [0], "fsdp_offload_layers": "0"}, "unsupported"),
        ):
            with self.subTest(change=change):
                args = copy.copy(self.args)
                for key, value in change.items():
                    setattr(args, key, value)
                with self.assertRaisesRegex(ValueError, message):
                    validate_args(args, 1)

    def test_qwen_engram_identity_captures_hash_and_storage_configuration(self):
        self.args.qwen_engram = True
        self.args.qwen_engram_layers = [0, 1]
        original = _qwen_engram_metadata(self.args, tiny_config())
        for key, value in (("qwen_engram_layers", [1, 0]), ("qwen_engram_seed", 17),
                           ("qwen_engram_vocab_per_head", 17), ("engram_device", "cpu"),
                           ("engram_cache_mib", 1)):
            with self.subTest(key=key):
                args = copy.copy(self.args)
                setattr(args, key, value)
                self.assertNotEqual(_qwen_engram_metadata(args, tiny_config()), original)

    def test_rejects_unsupported_topology_or_training_options(self):
        for change, message in (
            ({"cp_size": 2}, "cp-size"),
            ({"ep_size": 3}, "ep-size"),
            ({"gradient_accumulation_steps": 0}, "accumulation"),
            ({"learning_rate": float("nan")}, "learning-rate"),
            ({"weight_decay": float("inf")}, "weight-decay"),
            ({"resume": True, "adapter_in": "adapter.pt"}, "mutually exclusive"),
            ({"verify_lora": True}, "verify-lora requires --lora"),
            ({"checkpoint_state": "lora"}, "checkpoint-state lora requires --lora"),
            ({"lora_probe_tokens": 4097}, "lora-probe-tokens"),
            ({"min_active_tokens_per_example": 9}, "min-active-tokens"),
            ({"num_layers": 1}, "DSV4-only"),
            ({"lora": True, "qwen_engram": True}, "LoRA with added Engram"),
            ({"lora": True, "lora_targets": "mlp.experts"}, "targets only"),
            ({"checkpoint_mode": "selective"}, "none/full"),
            ({"seq_len": 33}, "max_position_embeddings"),
        ):
            with self.subTest(change=change):
                args = copy.copy(self.args)
                for name, value in change.items():
                    setattr(args, name, value)
                with self.assertRaisesRegex(ValueError, message):
                    validate_args(args, 2)

    def test_dsv4_prefix_and_invalid_window_are_checked(self):
        config = DeepSeekV41Config.tiny()
        (self.folder / "config.json").write_text(json.dumps(config.to_dict()))
        self.args.model_type = "dsv4"
        validate_args(self.args, 1)
        self.assertEqual(self.args.num_layers, 1)
        self.args.num_layers = config.num_hidden_layers + 1
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_args(self.args, 1)

    def test_qwen_lora_defaults_and_verification_are_accepted(self):
        self.args.lora = self.args.verify_lora = True
        self.args.checkpoint_state = "lora"
        self.args.optimizer = "adamw"
        self.args.adapter_out = str(self.folder / "adapter.pt")
        validate_args(self.args, 2)
        self.assertEqual(self.args.lora_targets,
                         "self_attn.q_proj,self_attn.k_proj,self_attn.v_proj,self_attn.o_proj")

    def test_dsv4_does_not_silently_accept_qwen_engram_flags(self):
        (self.folder / "config.json").write_text(json.dumps(DeepSeekV41Config.tiny().to_dict()))
        self.args.model_type = "dsv4"
        self.args.qwen_engram = True
        with self.assertRaisesRegex(ValueError, "require --model-type qwen"):
            validate_args(self.args, 1)

    def test_dsv4_lora_verification_accepts_fresh_and_resumed_training(self):
        (self.folder / "config.json").write_text(json.dumps(DeepSeekV41Config.tiny().to_dict()))
        self.args.model_type = "dsv4"
        self.args.lora = self.args.verify_lora = True
        validate_args(self.args, 1)
        self.args.resume = True
        self.args.adapter_out = str(self.folder / "adapter.pt")
        validate_args(self.args, 1)


class SFTQwenEngramIntegrationTest(unittest.TestCase):
    """Exercise real checkpoint loading and DCP on CPU; FSDP calls are isolated."""

    def test_loader_failure_closes_registered_prefetch_workers(self):
        args = parse_args([
            "--model-type", "qwen", "--model-path", "unused", "--data", "unused",
            "--output-dir", "unused", "--qwen-engram", "--qwen-engram-layers", "0",
            "--qwen-engram-vocab-per-head", "17", "--qwen-engram-head-dim", "4",
            "--qwen-engram-heads", "2", "--qwen-engram-max-ngram", "3",
            "--engram-device", "cpu", "--engram-prefetch", "async",
        ])
        tables = []
        def failing_loader(*_args, setup_callback, **_kwargs):
            model = Qwen3MoeForCausalLM(tiny_config())
            setup_callback(model)
            tables.extend(model.model.engram_tables.values())
            for table in tables:
                table.resolve(table.prefetch(torch.zeros((1, 1), dtype=torch.long)))
                self.assertIsNotNone(table._executor)
            raise OSError("simulated checkpoint read failure")
        with ExitStack() as cleanup:
            cleanup.enter_context(patch("train_sft.load_qwen3_moe", side_effect=failing_loader))
            cleanup.enter_context(patch("train_sft._record"))
            with self.assertRaisesRegex(OSError, "checkpoint read failure"):
                _load_model(args, Runtime(torch.device("cpu")), SimpleNamespace(engram=None),
                            ContextParallel(), TokenDispatcher(), cleanup)
        self.assertTrue(tables)
        for table in tables:
            self.assertIsNone(table._executor)
            self.assertFalse(table._pending)

    def test_layerwise_augmentation_sparse_update_and_resume(self):
        torch.manual_seed(713)
        with tempfile.TemporaryDirectory() as directory, ExitStack() as cleanup:
            folder = Path(directory)
            source = Qwen3MoeForCausalLM(tiny_config())
            write_hf_checkpoint(folder, source)
            args = parse_args([
                "--model-type", "qwen", "--model-path", str(folder),
                "--data", str(folder / "data.jsonl"), "--output-dir", str(folder / "run"),
                "--qwen-engram", "--qwen-engram-layers", "0", "1",
                "--qwen-engram-vocab-per-head", "17", "--qwen-engram-head-dim", "4",
                "--qwen-engram-heads", "2", "--qwen-engram-max-ngram", "3",
                "--engram-device", "cpu", "--engram-prefetch", "async",
                "--engram-deduplicate", "--engram-cache-mib", "1",
                "--checkpoint-mode", "full", "--seq-len", "8", "--learning-rate", "0.1",
            ])
            runtime = Runtime(device=torch.device("cpu"))
            meshes = SimpleNamespace(engram=None)
            callbacks = []
            def component(name, module, _meshes):
                callbacks.append(name)
                if name.startswith("model.layers."):
                    self.assertTrue(hasattr(module, "engram"))
            # This test validates CPU loading/data/checkpoint integration, not CUDA FSDP.
            cleanup.enter_context(patch("train_sft.shard_qwen_component", side_effect=component))
            root = cleanup.enter_context(patch("train_sft.shard_qwen_root"))
            cleanup.enter_context(patch("train_sft._record"))
            model = _load_model(args, runtime, meshes, ContextParallel(), TokenDispatcher(), cleanup)
            root.assert_called_once_with(model, meshes)
            self.assertEqual(len(callbacks), tiny_config().num_hidden_layers + 3)
            self.assertEqual(set(model.model.engram_tables), {"0", "1"})
            torch.testing.assert_close(model.model.embed_tokens.weight,
                                       source.model.embed_tokens.weight.bfloat16(), rtol=0, atol=0)
            optimizer = build_optimizer(model, OptimizerPolicy("sgd", .1))
            ids = torch.tensor([[1, 3, 4, 5, 2, 0], [1, 5, 4, 3, 2, 0]])
            labels = ids.masked_fill(ids.eq(0), -100)
            labels[:, :2] = -100
            def update(current_model, current_optimizer):
                current_optimizer.zero_grad(set_to_none=True)
                result = current_model(ids, attention_mask=ids.ne(0), labels=labels)
                self.assertTrue(torch.isfinite(result.loss).item())
                result.loss.backward()
                _synchronize_engram(current_model)
                probes = _engram_update_probes(current_model.model.engram_tables)
                current_optimizer.step()
                with patch("train_sft.dist.all_reduce"):
                    metrics = _engram_update_metrics(current_model.model.engram_tables, probes, runtime)
                for table_metrics in metrics.values():
                    self.assertGreater(table_metrics["parameter_delta_max"], 0)
                    self.assertGreater(table_metrics["probed_rows"], 0)
                return result.loss.detach()
            update(model, optimizer)
            metadata = json.loads(json.dumps(_qwen_engram_metadata(args, model.config)))
            def manager(current_model, current_optimizer, training_config=metadata):
                return CheckpointManager(folder / "run", TrainingState(
                    current_model, current_optimizer, model_config=current_model.config.to_dict(),
                    data_generator=torch.Generator(), training_config=training_config,
                    state_adapter=StateAdapter(current_model),
                ))
            manager(model, optimizer).save(1)
            saved = {name: value.detach().clone() for name, value in model.state_dict().items()}
            expected_loss = update(model, optimizer)
            expected_next = {name: value.detach().clone() for name, value in model.state_dict().items()}
            restored = _load_model(args, runtime, meshes, ContextParallel(), TokenDispatcher(), cleanup)
            restored_optimizer = build_optimizer(restored, OptimizerPolicy("sgd", .1))
            table_parameter_ids = {name: id(table.weight) for name, table in restored.model.engram_tables.items()}
            self.assertEqual(manager(restored, restored_optimizer).load(), 1)
            self.assertEqual(table_parameter_ids,
                             {name: id(table.weight) for name, table in restored.model.engram_tables.items()})
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, saved[name], rtol=0, atol=0)
            torch.testing.assert_close(update(restored, restored_optimizer), expected_loss, rtol=0, atol=0)
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected_next[name], rtol=0, atol=0)
            # These settings have equal column counts and parameter layouts,
            # but different hash addressing: shape checks alone are insufficient.
            args.qwen_engram_heads, args.qwen_engram_max_ngram = 4, 2
            incompatible_model = _load_model(args, runtime, meshes, ContextParallel(), TokenDispatcher(), cleanup)
            self.assertEqual(StateAdapter(incompatible_model).signature(), StateAdapter(restored).signature())
            incompatible = json.loads(json.dumps(_qwen_engram_metadata(args, incompatible_model.config)))
            incompatible_optimizer = build_optimizer(incompatible_model, OptimizerPolicy("sgd", .1))
            with self.assertRaisesRegex(ValueError, "configuration is incompatible"):
                manager(incompatible_model, incompatible_optimizer, incompatible).load()


class SFTDeepSeekEngramTest(unittest.TestCase):
    def make_model(self, *, sparse=True):
        config = small_config(num_hidden_layers=2, compress_ratios=[0, 2],
                              kv_source_layer_ids=[1], index_source_layer_ids=[1],
                              candidate_source_layer_id=1)
        policy = EngramPolicy("cpu", "async", True) if sparse else EngramPolicy()
        model = DeepSeekV41ForCausalLM(config, sparse_engram_gradients=sparse, engram_policy=policy)
        self.addCleanup(model.close)
        return model

    def update(self, model, optimizer):
        ids = torch.tensor([[1, 3, 4, 3, 2, 0], [1, 4, 3, 4, 2, 0]])
        labels = ids.masked_fill(ids.eq(0), -100)
        labels[:, :2] = -100
        optimizer.zero_grad(set_to_none=True)
        loss = model(ids, attention_mask=ids.ne(0), labels=labels).loss
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        _synchronize_engram(model)
        tables = _trainable_engram_tables(model)
        self.assertEqual(set(tables), {"1"})
        # Sparse verification must never densify a pretrained table's gradient.
        with patch.object(torch.Tensor, "to_dense", side_effect=AssertionError("no dense gradient copy")):
            probes = _engram_update_probes(tables)
            for _, before, _ in probes.values():
                self.assertLessEqual(before.shape[0], 256)
                self.assertLessEqual(before.numel(), 256 * model.config.engram_head_dim)
            optimizer.step()
            with patch("train_sft.dist.all_reduce"):
                metrics = _engram_update_metrics(tables, probes, Runtime(torch.device("cpu")))
        self.assertGreater(metrics["1"]["parameter_delta_max"], 0)
        self.assertGreater(metrics["1"]["probed_rows"], 0)
        return loss.detach(), metrics

    def test_sparse_cpu_table_update_checkpoint_resume_and_generic_summary(self):
        torch.manual_seed(307)
        model = self.make_model()
        optimizer = build_optimizer(model, OptimizerPolicy("sgd", .1))
        _, metrics = self.update(model, optimizer)
        self.assertTrue(model.model.engram_tables["1"].weight.grad.is_sparse)
        self.assertEqual(metrics["1"]["gradient_layout"], "sparse")
        self.assertGreater(metrics["1"]["gradient_nnz"], 0)
        summary = _engram_summary(model, {"1": metrics["1"]["parameter_delta_max"]})
        self.assertEqual(summary["engram_trainable_tables"], ["1"])
        self.assertEqual(summary["engram_policies"]["1"]["device"], "cpu")
        self.assertEqual(summary["engram_table_parameters"], 128 * 8)
        self.assertNotIn("qwen_engram", summary)
        with tempfile.TemporaryDirectory() as folder:
            def manager(current, current_optimizer):
                return CheckpointManager(folder, TrainingState(
                    current, current_optimizer, model_config=current.config.to_dict(),
                    data_generator=torch.Generator(), training_config={"model_type": "dsv4"},
                    state_adapter=StateAdapter(current)))
            manager(model, optimizer).save(1)
            expected_loss, _ = self.update(model, optimizer)
            expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
            torch.manual_seed(998)
            restored = self.make_model()
            restored_optimizer = build_optimizer(restored, OptimizerPolicy("sgd", .1))
            self.assertEqual(manager(restored, restored_optimizer).load(), 1)
            actual_loss, _ = self.update(restored, restored_optimizer)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)

    def test_probes_accept_existing_dense_adamw_table_gradients(self):
        torch.manual_seed(308)
        model = self.make_model(sparse=False)
        # Isolate dense probe semantics on CPU. The production placement policy
        # correctly permits dense Engram AdamW only on CUDA.
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        _, metrics = self.update(model, optimizer)
        self.assertFalse(model.model.engram_tables["1"].weight.grad.is_sparse)
        self.assertEqual(metrics["1"]["gradient_layout"], "dense")
        self.assertNotIn("gradient_nnz", metrics["1"])

    def test_frozen_lora_tables_are_reported_but_not_required_to_update(self):
        model = self.make_model()
        model.model.engram_tables["1"].weight.requires_grad_(False)
        self.assertEqual(_trainable_engram_tables(model), {})
        self.assertEqual(_engram_update_probes(_trainable_engram_tables(model)), {})
        summary = _engram_summary(model, {})
        self.assertEqual(summary["engram_trainable_tables"], [])
        self.assertEqual(summary["engram_parameter_delta_max"], {})
        self.assertEqual(summary["engram_table_parameters"], 128 * 8)

    def test_dense_row_discovery_crosses_chunks_and_stops_at_sample_limit(self):
        gradient = torch.zeros((10000, 256))
        gradient[4200:4500, 17] = 1
        rows = _engram_gradient_rows(gradient)
        self.assertEqual(rows.tolist(), list(range(4200, 4456)))


class SFTMetricsRecoveryTest(unittest.TestCase):
    def test_removes_uncheckpointed_steps_and_torn_final_append(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metrics.jsonl"
            saved = {"step": 1, "loss": 3.25}
            path.write_text(json.dumps(saved) + '\n{"step": 2, "loss": 3.0}\n{"step": 3,\n\n')
            _truncate_resumed_metrics(path, 1)
            self.assertEqual([json.loads(row) for row in path.read_text().splitlines()], [saved])

    def test_rejects_internal_corruption_without_rewriting_history(self):
        for content in (
            '{"step": 1}\n{torn}\n{"step": 2}\n',
            '{"step": "1"}\n',
            '{"step": true}\n',
        ):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "metrics.jsonl"
                path.write_text(content)
                with self.assertRaises(ValueError):
                    _truncate_resumed_metrics(path, 1)
                self.assertEqual(path.read_text(), content)


if __name__ == "__main__":
    unittest.main()
