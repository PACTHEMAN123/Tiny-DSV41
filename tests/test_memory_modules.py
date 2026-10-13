import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import torch

from dsv41_train.loading import LoadPolicy
from dsv41_train.layout import parameter_memory, parameter_layouts
from dsv41_train.models.dsv4 import load_dsv41_backbone_window, DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.engram import EngramPolicy, RowShardedEmbedding
from dsv41_train.optimization import build_optimizer, OptimizerPolicy
from dsv41_train.sharded_state import StateAdapter
from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.models.dsv4.model import CausalLMOutput
from post_training_helpers import small_config
from test_dsv4_loading import synthetic_checkpoint
from memory_validation_cases import fixture_dimensions, validation_cases


class LocalMaterializer:
    def prepare(self, model, *, dtype, device, direct):
        model.to(dtype=dtype)
        for layer in model.model.layers:
            layer.attention_hc.float()
            layer.moe_hc.float()
            layer.attention.sinks.float()
        model.to_empty(device=device)

    def roots_loaded(self, model): pass
    def layer_loaded(self, layer): pass
    def finish(self, model): pass


def memory_config(**overrides):
    return small_config(q_lora_rank=32, o_lora_rank=32, engram_head_dim=32, **overrides)


class MemoryModulesTest(unittest.TestCase):
    def test_eight_rank_matrix_includes_full_cp_and_ep_sizes(self):
        experts, length = fixture_dimensions(8)
        cases = validation_cases(8, all_topologies=True)
        expected = {(cp, ep) for cp in (1, 2, 4, 8) for ep in (1, 2, 4, 8)}
        self.assertEqual({(case[0], case[1]) for case in cases}, expected)
        self.assertTrue(all(experts % ep == length % cp == 0 for cp, ep in expected))
        self.assertEqual(len(validation_cases(6, all_topologies=True)), 16)

    def test_output_tensors_are_visible_to_fsdp_hooks(self):
        from torch.utils._pytree import tree_flatten, tree_unflatten
        loss, logits, aux, router = (torch.randn(2, requires_grad=True) for _ in range(4))
        output = CausalLMOutput(loss, logits, aux, (router,))
        values, specification = tree_flatten(output)
        self.assertEqual([id(value) for value in values], [id(loss), id(logits), id(aux), id(router)])
        restored = tree_unflatten(values, specification)
        self.assertIs(restored.loss, loss)
        self.assertIs(restored.router_logits[0], router)

    def test_storage_dedup_async_accumulation_and_empty(self):
        for prefetch in ("sync", "async"):
            for dedup in (False, True):
                table = RowShardedEmbedding(11, 4, sparse_gradients=True,
                    policy=EngramPolicy("cpu", prefetch, dedup, 4096))
                reference = torch.nn.Embedding(11, 4, sparse=True)
                reference.weight.data.copy_(table.weight.detach().normal_())
                opt = torch.optim.SGD(table.parameters(), lr=.01)
                ref_opt = torch.optim.SGD(reference.parameters(), lr=.01)
                try:
                    for _ in range(3):
                        opt.zero_grad(set_to_none=True)
                        ref_opt.zero_grad(set_to_none=True)
                        for indices in (torch.tensor([[0, 4, 4, 10]]), torch.empty((1, 0), dtype=torch.long)):
                            actual, expected = table(indices), reference(indices)
                            torch.testing.assert_close(actual, expected)
                            (actual.square().sum() * .5).backward()
                            (expected.square().sum() * .5).backward()
                        self.assertTrue(table.weight.grad.is_sparse)
                        torch.testing.assert_close(table.weight.grad.to_dense(), reference.weight.grad.to_dense())
                        table.synchronize()
                        opt.step(); ref_opt.step()
                        torch.testing.assert_close(table.weight, reference.weight)
                finally:
                    table.close()

    def test_sparse_sgd_updates_duplicate_bfloat16_rows_once(self):
        table = RowShardedEmbedding(11, 4, sparse_gradients=True, policy=EngramPolicy("cpu"))
        table.to(dtype=torch.bfloat16)
        self.addCleanup(table.close)
        table.weight.data.fill_(1)
        optimizer = build_optimizer(table, OptimizerPolicy("sgd", .01))
        table(torch.ones(200, dtype=torch.long)).sum().backward()
        expected = table.weight.detach().float().add_(table.weight.grad.to_dense().float(), alpha=-.01).to(table.weight.dtype)
        optimizer.step()
        torch.testing.assert_close(table.weight, expected, rtol=0, atol=0)

    def test_deduplication_accumulates_bfloat16_gradients_in_opmath(self):
        table = RowShardedEmbedding(11, 4, sparse_gradients=True,
            policy=EngramPolicy("cpu", deduplicate=True)).to(dtype=torch.bfloat16)
        self.addCleanup(table.close)
        table.weight.data.fill_(1)
        gradient = torch.full((513, 4), .1, dtype=torch.bfloat16)
        table(torch.ones(513, dtype=torch.long)).backward(gradient)
        expected = gradient.float().sum(0).bfloat16()
        torch.testing.assert_close(table.weight.grad.coalesce().values()[0], expected, rtol=0, atol=0)

    def test_dense_cpu_sgd_uses_opmath_without_changing_storage(self):
        torch.manual_seed(1)
        model = torch.nn.Linear(100, 100, bias=False, dtype=torch.bfloat16)
        model.weight.grad = torch.randn_like(model.weight) * .1
        original_gradient = model.weight.grad
        expected = model.weight.detach().float().add_(original_gradient.float(), alpha=-.01).bfloat16()
        optimizer = build_optimizer(model, OptimizerPolicy("sgd", .01, True))
        optimizer.step()
        torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
        self.assertIs(model.weight.grad, original_gradient)
        self.assertEqual(model.weight.dtype, torch.bfloat16)
        self.assertEqual(len(optimizer.state), 0)

    def test_direct_loading_rejects_an_undersized_decode_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            synthetic_checkpoint(Path(folder), memory_config())
            with self.assertRaisesRegex(ValueError, "quantization block"):
                load_dsv41_backbone_window(folder, num_layers=3, dtype=torch.float32,
                    load_policy=LoadPolicy("direct_shard", 7, 4096), materializer=LocalMaterializer())

    def test_adamw_state_restores_into_a_fresh_model(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.manual_seed(42)
            model = DeepSeekV41ForCausalLM(memory_config())
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            def manager(target, opt):
                return CheckpointManager(folder, TrainingState(target, opt,
                    model_config=target.config.to_dict(), data_generator=torch.Generator(),
                    state_adapter=StateAdapter(target)))
            tokens = torch.randint(3, 32, (1, 8))
            model(tokens, labels=tokens).loss.backward()
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            manager(model, optimizer).save(1)
            model(tokens, labels=tokens).loss.backward()
            optimizer.step()
            restored = DeepSeekV41ForCausalLM(memory_config())
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
            manager(restored, restored_optimizer).load()
            restored(tokens, labels=tokens).loss.backward()
            restored_optimizer.step()
            for (name, expected), (_, actual) in zip(model.named_parameters(), restored.named_parameters()):
                torch.testing.assert_close(actual, expected, msg=name)

    def test_tickets_are_single_use_and_boundary_rejects_pending(self):
        table = RowShardedEmbedding(11, 4, sparse_gradients=True, policy=EngramPolicy("cpu", "async"))
        try:
            ticket = table.prefetch(torch.tensor([1, 1]))
            with self.assertRaisesRegex(RuntimeError, "pending"):
                table.prefetch(torch.tensor([2]))
            with self.assertRaisesRegex(RuntimeError, "unconsumed"):
                table.synchronize()
            with self.assertRaisesRegex(RuntimeError, "unconsumed"):
                build_optimizer(table, OptimizerPolicy()).step()
            table.resolve(ticket)
            with self.assertRaisesRegex(RuntimeError, "already consumed"):
                table.resolve(ticket)
        finally:
            table.close()

    def test_foreign_cancelled_and_failed_tickets_release_resources(self):
        table = RowShardedEmbedding(11, 4, sparse_gradients=True, policy=EngramPolicy("cpu", "async"))
        other = RowShardedEmbedding(11, 4, sparse_gradients=True)
        self.addCleanup(table.close)
        self.addCleanup(other.close)
        ticket = table.prefetch(torch.tensor([1]))
        with self.assertRaisesRegex(ValueError, "different table"):
            other.resolve(ticket)
        table.close()
        with self.assertRaisesRegex(RuntimeError, "consumed"):
            table.resolve(ticket)
        with patch.object(table.store, "lookup", side_effect=RuntimeError("lookup failure")):
            with self.assertRaisesRegex(RuntimeError, "lookup failure"):
                table(torch.tensor([1]))
        table.synchronize()
        self.assertEqual(table(torch.tensor([1])).shape, (1, 4))

    def test_direct_loading_quantized_fusions_and_tail_chunks(self):
        with tempfile.TemporaryDirectory() as folder:
            config, expected = synthetic_checkpoint(Path(folder), memory_config())
            for pipeline in (False, True):
                with patch("dsv41_train.loading.reader.ShardedSafeTensorReader.tensor", side_effect=AssertionError("full tensor read")), \
                     patch("dsv41_train.models.dsv4.checkpoint.ThreadPoolExecutor", wraps=ThreadPoolExecutor) as pools, \
                     patch("dsv41_train.loading.source.ThreadPoolExecutor", side_effect=AssertionError("per-parameter pool")):
                    model = load_dsv41_backbone_window(folder, num_layers=config.num_hidden_layers, dtype=torch.float32,
                        load_policy=LoadPolicy("direct_shard", 7, 131072, pipeline),
                        materializer=LocalMaterializer(), sparse_engram_gradients=True,
                        engram_policy=EngramPolicy("cpu"))
                self.assertEqual(pools.call_count, int(pipeline))
                self.addCleanup(model.close)
                for name, parameter in model.named_parameters():
                    torch.testing.assert_close(parameter, expected[name], msg=name)
                self.assertGreater(model.loading_metrics["logical_bytes_read"], 0)

    def test_checkpoint_adapter_roundtrip_next_update(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.manual_seed(42)
            model = DeepSeekV41ForCausalLM(memory_config(), sparse_engram_gradients=True)
            model.configure_engram(EngramPolicy("cpu", "async", True))
            optimizer = build_optimizer(model, OptimizerPolicy())
            generator = torch.Generator().manual_seed(7)
            state = TrainingState(model, optimizer, model_config=model.config.to_dict(), data_generator=generator,
                                  state_adapter=StateAdapter(model))
            manager = CheckpointManager(folder, state)
            tokens = torch.randint(3,32,(1,8))
            try:
                model(tokens, labels=tokens).loss.backward()
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                manager.save(1)
                saved_rng = torch.get_rng_state().clone()
                saved_generator = generator.get_state().clone()
                before = {name:p.detach().clone() for name,p in model.named_parameters()}
                model(tokens, labels=tokens).loss.backward()
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                expected = {name:p.detach().clone() for name,p in model.named_parameters()}
                torch.rand(17)
                torch.rand(17, generator=generator)
                self.assertEqual(manager.load(), 1)
                torch.testing.assert_close(torch.get_rng_state(), saved_rng)
                torch.testing.assert_close(generator.get_state(), saved_generator)
                for name,p in model.named_parameters():
                    torch.testing.assert_close(p,before[name])
                model(tokens, labels=tokens).loss.backward()
                optimizer.step()
                for name,p in model.named_parameters():
                    torch.testing.assert_close(p,expected[name])
                memory = parameter_memory(model, optimizer)
                self.assertGreater(memory["engram"]["gradients"],0)
            finally:
                model.close()

    def test_stateless_checkpoint_does_not_initialize_optimizer_or_gradients(self):
        with tempfile.TemporaryDirectory() as folder:
            model = DeepSeekV41ForCausalLM(memory_config(), sparse_engram_gradients=True)
            self.addCleanup(model.close)
            optimizer = build_optimizer(model, OptimizerPolicy())
            manager = CheckpointManager(folder, TrainingState(model, optimizer,
                model_config=model.config.to_dict(), data_generator=torch.Generator(),
                state_adapter=StateAdapter(model)))
            with patch.object(optimizer, "step", side_effect=AssertionError("checkpoint performed an update")):
                manager.save(0)
                manager.load()
            self.assertFalse(optimizer.state)
            self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_checkpoint_rejects_legacy_and_incompatible_layout_before_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            model = DeepSeekV41ForCausalLM(memory_config(), sparse_engram_gradients=True)
            self.addCleanup(model.close)
            optimizer = build_optimizer(model, OptimizerPolicy())
            state = TrainingState(model, optimizer, model_config=model.config.to_dict(),
                                  data_generator=torch.Generator())
            manager = CheckpointManager(folder, state)
            manager.save(1)
            state.state_adapter = StateAdapter(model)
            with self.assertRaisesRegex(ValueError, "legacy"):
                manager.load()
            manager.save(2)
            state.state_adapter.topology["cp"] = 2
            with self.assertRaisesRegex(ValueError, "incompatible"):
                manager.load()

    def test_checkpoint_rejects_configuration_before_writing_parameters(self):
        with tempfile.TemporaryDirectory() as folder:
            model = DeepSeekV41ForCausalLM(memory_config())
            self.addCleanup(model.close)
            state = TrainingState(model, torch.optim.SGD(model.parameters(), lr=.01),
                model_config=model.config.to_dict(), data_generator=torch.Generator(),
                state_adapter=StateAdapter(model))
            manager = CheckpointManager(folder, state)
            manager.save(0)
            with torch.no_grad():
                model.model.embedding.weight.add_(1)
            expected = model.model.embedding.weight.detach().clone()
            state.model_config["cache_quantization"] = "hf"
            with self.assertRaisesRegex(ValueError, "configuration"):
                manager.load()
            state.model_config["cache_quantization"] = "none"
            state.optimizer = torch.optim.AdamW(model.parameters())
            with self.assertRaisesRegex(ValueError, "configuration"):
                manager.load()
            torch.testing.assert_close(model.model.embedding.weight, expected, rtol=0, atol=0)

    def test_direct_loading_global_layer_window(self):
        with tempfile.TemporaryDirectory() as folder:
            config, expected = synthetic_checkpoint(Path(folder), memory_config())
            model = load_dsv41_backbone_window(folder, start_layer=1, num_layers=3, dtype=torch.float32,
                load_policy=LoadPolicy("direct_shard", 7, 131072), materializer=LocalMaterializer(),
                sparse_engram_gradients=True, engram_policy=EngramPolicy("cpu"))
            self.addCleanup(model.close)
            layouts = parameter_layouts(model)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, expected[layouts[name].logical_name], msg=name)

    def test_cpu_policy_rejects_dense_adam_and_model_relocation(self):
        with self.assertRaisesRegex(ValueError, "sparse"):
            RowShardedEmbedding(11, 4, policy=EngramPolicy("cpu"))
        table = RowShardedEmbedding(11, 4, sparse_gradients=True, policy=EngramPolicy("cpu"))
        with self.assertRaisesRegex(ValueError, "placement"):
            table.to("meta")
        model = DeepSeekV41ForCausalLM(memory_config(), sparse_engram_gradients=True)
        with self.assertRaisesRegex(ValueError, "AdamW"):
            build_optimizer(model, OptimizerPolicy("adamw"))


if __name__ == "__main__":
    unittest.main()
