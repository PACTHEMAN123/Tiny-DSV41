import unittest
from copy import deepcopy
from unittest.mock import Mock, call, patch

import torch

from dsv41_train.dispatch import DispatchMetadata, TokenDispatcher
from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.model import NgramHash, RowShardedEmbedding
from dsv41_train.models.dsv4.moe import RoutedExperts
from dsv41_train.models.dsv4.parallel import ParallelParameters, apply_fsdp2
from dsv41_train.parallel import ContextParallel, ParallelMeshes


class ModelTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cpu_offloaded_sparse_embedding_backpropagates_from_cuda(self):
        torch.manual_seed(17)
        table = RowShardedEmbedding(32, 8, sparse_gradients=True)
        original = table.weight.detach().clone()
        indices = torch.tensor([1, 3, 1, 7], device="cuda")

        output = table(indices)
        expected = torch.nn.functional.embedding(indices.cpu(), original).cuda()
        output.float().square().sum().backward()

        torch.testing.assert_close(output, expected)
        self.assertEqual(table.weight.device.type, "cpu")
        self.assertIsNotNone(table.weight.grad)
        assert table.weight.grad is not None
        self.assertTrue(table.weight.grad.is_sparse)
        self.assertEqual(table.weight.grad.device.type, "cpu")

    def test_empty_expert_route_does_not_allocate_gradients(self):
        class EmptyReceiveDispatcher(TokenDispatcher):
            def dispatch(self, hidden, expert_ids, weights):
                metadata = DispatchMetadata(
                    token_count=hidden.shape[0], topk=expert_ids.shape[-1]
                )
                return (
                    hidden[:0],
                    expert_ids.reshape(-1)[:0],
                    weights.reshape(-1)[:0],
                    metadata,
                )

            def combine(self, hidden, metadata):
                empty = hidden.new_zeros(metadata.token_count, hidden.shape[-1])
                return empty + hidden.sum() * 0

        config = DeepSeekV41Config.tiny()
        experts = RoutedExperts(config, EmptyReceiveDispatcher())
        hidden = torch.randn(3, config.hidden_size, requires_grad=True)
        expert_ids = torch.zeros(3, config.num_experts_per_tok, dtype=torch.long)
        weights = torch.ones(3, config.num_experts_per_tok)

        experts(hidden, expert_ids, weights).sum().backward()

        self.assertTrue(all(parameter.grad is None for parameter in experts.gate_up))
        self.assertTrue(all(parameter.grad is None for parameter in experts.down))

    def test_expert_gradients_are_allocated_only_for_used_experts(self):
        config = DeepSeekV41Config.tiny()
        experts = RoutedExperts(config, TokenDispatcher())
        hidden = torch.randn(3, config.hidden_size)
        expert_ids = torch.zeros(3, config.num_experts_per_tok, dtype=torch.long)
        weights = torch.ones(3, config.num_experts_per_tok)

        experts(hidden, expert_ids, weights).sum().backward()

        self.assertIsNotNone(experts.gate_up[0].grad)
        self.assertIsNotNone(experts.down[0].grad)
        self.assertTrue(all(parameter.grad is None for parameter in experts.gate_up[1:]))
        self.assertTrue(all(parameter.grad is None for parameter in experts.down[1:]))

    def test_layer_window_selects_global_layers(self):
        config = DeepSeekV41Config.tiny()
        with torch.device("meta"):
            model = DeepSeekV41ForCausalLM(config, layer_ids=[3, 4])

        self.assertEqual([layer.layer_id for layer in model.model.layers], [3, 4])
        self.assertEqual(list(model.model.engram_tables), [])

    def test_selected_engram_hash_preserves_global_prime_sequence(self):
        config = DeepSeekV41Config(
            num_hidden_layers=4,
            compress_ratios=[0, 0, 0, 0],
            kv_source_layer_ids=[],
            index_source_layer_ids=[],
            candidate_source_layer_id=-1,
            engram_layer_ids=[1, 3],
            engram_num_embeddings=[100, 100],
            engram_vocab_size=31,
            engram_max_ngram_size=3,
            engram_n_heads=1,
            engram_compressed_vocab_size=32,
        )

        full = NgramHash(config)
        selected = NgramHash(config, [3])

        torch.testing.assert_close(selected.primes[0], full.primes[1])
        torch.testing.assert_close(selected.offsets[0], full.offsets[1])
        torch.testing.assert_close(selected.multipliers[0], full.multipliers[1])
        torch.testing.assert_close(
            full.multipliers,
            torch.tensor(
                [
                    [237300864419207287, 14987281307456977, 111353620295905115],
                    [85240722207167499, 252868047889091175, 33684873675973757],
                ]
            ),
        )

    def test_dsv4_fsdp_wraps_experts_before_layers_and_root(self):
        with torch.device("meta"):
            model = DeepSeekV41ForCausalLM(DeepSeekV41Config.tiny())
        for expert_parallel in (False, True):
            for target in (model, model.model):
                with self.subTest(expert_parallel=expert_parallel, target=type(target).__name__):
                    meshes = ParallelMeshes(
                        fsdp=Mock(), cp=None,
                        ep=Mock() if expert_parallel else None,
                        expert_fsdp=Mock() if expert_parallel else None,
                        engram=Mock() if expert_parallel else None,
                        engram_fsdp=Mock() if expert_parallel else None,
                    )
                    try:
                        from torch.distributed.fsdp import fully_shard
                    except ImportError:
                        shard_path = "torch.distributed._composable.fsdp.fully_shard"
                    else:
                        shard_path = "torch.distributed.fsdp.fully_shard"
                    with patch(shard_path) as shard:
                        apply_fsdp2(target, meshes, reshard_after_forward=False)
                    expected = []
                    for layer in model.model.layers:
                        if expert_parallel:
                            expected.append(call(
                                layer.moe.routed, mesh=meshes.expert_fsdp,
                                reshard_after_forward=False,
                            ))
                        for fp32_module in (
                            layer.attention_hc,
                            layer.moe_hc,
                            layer.attention.sinks,
                        ):
                            expected.append(call(
                                fp32_module,
                                mesh=meshes.fsdp,
                                reshard_after_forward=False,
                            ))
                        expected.append(call(layer, mesh=meshes.fsdp, reshard_after_forward=False))
                    if expert_parallel:
                        for table in model.model.engram_tables.values():
                            expected.append(call(
                                table, mesh=meshes.engram_fsdp,
                                reshard_after_forward=False,
                            ))
                    expected.append(call(target, mesh=meshes.fsdp))
                    self.assertEqual(shard.call_args_list, expected)

    def test_dsv4_cp_ep_fsdp_disables_implicit_backward_prefetch(self):
        with torch.device("meta"):
            model = DeepSeekV41ForCausalLM(DeepSeekV41Config.tiny())
        meshes = ParallelMeshes(
            fsdp=Mock(),
            cp=Mock(),
            ep=Mock(),
            expert_fsdp=Mock(),
            _dense=Mock(),
        )
        try:
            from torch.distributed.fsdp import fully_shard
        except ImportError:
            shard_path = "torch.distributed._composable.fsdp.fully_shard"
        else:
            shard_path = "torch.distributed.fsdp.fully_shard"

        with (
            patch(shard_path),
            patch(
                "dsv41_train.models.dsv4.parallel._disable_backward_prefetch"
            ) as disable,
        ):
            apply_fsdp2(model, meshes)

        disable.assert_called_once_with(model)

    def test_dsv4_cp_ep_fsdp_replicates_small_fp32_parameters(self):
        with torch.device("meta"):
            model = DeepSeekV41ForCausalLM(DeepSeekV41Config.tiny())
        meshes = ParallelMeshes(
            fsdp=Mock(),
            cp=Mock(),
            ep=Mock(),
            expert_fsdp=Mock(),
            _dense=Mock(),
        )
        try:
            from torch.distributed.fsdp import fully_shard
        except ImportError:
            shard_path = "torch.distributed._composable.fsdp.fully_shard"
        else:
            shard_path = "torch.distributed.fsdp.fully_shard"

        with (
            patch(shard_path) as shard,
            patch("dsv41_train.models.dsv4.parallel._disable_backward_prefetch"),
        ):
            apply_fsdp2(model, meshes)

        replicated = ParallelParameters.collect(model).replicated
        replicated_ids = {id(parameter) for parameter in replicated}
        self.assertTrue(replicated_ids)
        wrapped_ids = {id(args.args[0]) for args in shard.call_args_list}
        for layer in model.model.layers:
            self.assertNotIn(id(layer.attention_hc), wrapped_ids)
            self.assertNotIn(id(layer.moe_hc), wrapped_ids)
            self.assertNotIn(id(layer.attention.sinks), wrapped_ids)
            layer_call = next(
                args for args in shard.call_args_list if args.args[0] is layer
            )
            self.assertEqual(
                {id(parameter) for parameter in layer_call.kwargs["ignored_params"]},
                {
                    id(parameter)
                    for module in (layer.attention_hc, layer.moe_hc, layer.attention.sinks)
                    for parameter in module.parameters(recurse=False)
                },
            )
        root_call = shard.call_args_list[-1]
        self.assertEqual(
            {id(parameter) for parameter in root_call.kwargs["ignored_params"]},
            replicated_ids,
        )

    def test_dsv4_fsdp_leaves_sparse_engram_tables_row_sharded(self):
        with torch.device("meta"):
            model = DeepSeekV41ForCausalLM(
                DeepSeekV41Config.tiny(), sparse_engram_gradients=True
            )
        meshes = ParallelMeshes(
            fsdp=Mock(),
            cp=None,
            ep=Mock(),
            expert_fsdp=Mock(),
            engram=Mock(),
            engram_fsdp=Mock(),
        )
        try:
            from torch.distributed.fsdp import fully_shard
        except ImportError:
            shard_path = "torch.distributed._composable.fsdp.fully_shard"
        else:
            shard_path = "torch.distributed.fsdp.fully_shard"

        with patch(shard_path) as shard:
            apply_fsdp2(model, meshes)

        table = next(iter(model.model.engram_tables.values()))
        self.assertNotIn(table, [args.args[0] for args in shard.call_args_list])
        self.assertEqual(
            shard.call_args_list[-1],
            call(
                model,
                mesh=meshes.fsdp,
                ignored_params={table.weight},
            ),
        )

    def test_forward_and_backward(self):
        config = DeepSeekV41Config(
            vocab_size=32,
            hidden_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            head_dim=16,
            q_lora_rank=16,
            qk_rope_head_dim=8,
            max_position_embeddings=16,
            sliding_window=8,
            compress_ratios=[0, 2, 2],
            kv_source_layer_ids=[1],
            index_source_layer_ids=[1, 2],
            candidate_source_layer_id=1,
            candidate_topk_blocks=2,
            candidate_block_size=2,
            index_n_heads=2,
            index_head_dim=16,
            index_topk=4,
            o_groups=2,
            o_lora_rank=16,
            moe_intermediate_size=32,
            n_routed_experts=4,
            num_experts_per_tok=2,
            hc_mult=2,
            hc_sinkhorn_iters=2,
            engram_layer_ids=[1],
            engram_num_embeddings=[128],
            engram_vocab_size=31,
            engram_max_ngram_size=3,
            engram_n_heads=1,
            engram_head_dim=8,
            engram_compressed_vocab_size=32,
        )
        model = DeepSeekV41ForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        attention_mask = torch.tensor([[1] * 8, [1] * 6 + [0] * 2])

        output = model(input_ids, attention_mask, labels=input_ids)

        self.assertEqual(output.logits.shape, (2, 8, config.vocab_size))
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.model.embedding.weight.grad)

    def test_gradient_checkpointing_matches_direct_backward(self):
        torch.manual_seed(23)
        direct = DeepSeekV41ForCausalLM(DeepSeekV41Config.tiny())
        checkpointed = deepcopy(direct)
        checkpointed.gradient_checkpointing_enable()
        direct.train()
        checkpointed.train()
        input_ids = torch.randint(0, direct.config.vocab_size, (1, 8))

        direct_output = direct(input_ids, labels=input_ids)
        checkpointed_output = checkpointed(input_ids, labels=input_ids)
        assert direct_output.loss is not None
        assert checkpointed_output.loss is not None
        direct_output.loss.backward()
        checkpointed_output.loss.backward()

        torch.testing.assert_close(checkpointed_output.logits, direct_output.logits)
        torch.testing.assert_close(checkpointed_output.loss, direct_output.loss)
        pairs = zip(direct.named_parameters(), checkpointed.named_parameters())
        for (direct_name, direct_parameter), (
            checkpointed_name,
            checkpointed_parameter,
        ) in pairs:
            self.assertEqual(direct_name, checkpointed_name)
            if direct_parameter.grad is None or checkpointed_parameter.grad is None:
                self.assertIs(direct_parameter.grad, checkpointed_parameter.grad)
                continue
            torch.testing.assert_close(
                checkpointed_parameter.grad,
                direct_parameter.grad,
                rtol=2e-5,
                atol=2e-6,
                msg=lambda message, name=direct_name: f"{name}: {message}",
            )

    def test_router_aux_gradient_injection_matches_direct_loss(self):
        torch.manual_seed(31)
        config = DeepSeekV41Config.tiny()
        config.router_aux_loss_coef = 0.125
        injected = DeepSeekV41ForCausalLM(config)
        direct = deepcopy(injected)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))

        injected_output = injected(input_ids, labels=input_ids)
        assert injected_output.loss is not None
        injected_output.loss.backward()

        direct_output = direct(input_ids)
        targets = torch.full_like(input_ids, -100)
        targets[:, :-1] = input_ids[:, 1:]
        task_loss = torch.nn.functional.cross_entropy(
            direct_output.logits.float().reshape(-1, config.vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
        )
        assert direct_output.aux_loss is not None
        direct_loss = task_loss + config.router_aux_loss_coef * direct_output.aux_loss
        direct_loss.backward()

        torch.testing.assert_close(injected_output.loss, direct_loss)
        for (injected_name, injected_parameter), (direct_name, direct_parameter) in zip(
            injected.named_parameters(), direct.named_parameters()
        ):
            self.assertEqual(injected_name, direct_name)
            if injected_parameter.grad is None or direct_parameter.grad is None:
                self.assertIs(injected_parameter.grad, direct_parameter.grad)
                continue
            torch.testing.assert_close(
                injected_parameter.grad,
                direct_parameter.grad,
                rtol=2e-5,
                atol=2e-6,
                msg=lambda message, name=injected_name: f"{name}: {message}",
            )

    def test_explicit_local_parallel_primitives(self):
        dispatcher = TokenDispatcher()
        x = torch.arange(12, dtype=torch.float32).view(3, 4)
        expert_ids = torch.tensor([[0, 1], [1, 0], [0, 1]])
        weights = torch.ones(3, 2)

        routed, routed_ids, routed_weights, metadata = dispatcher.dispatch(
            x, expert_ids, weights
        )
        combined = dispatcher.combine(routed * routed_weights[:, None], metadata)

        self.assertEqual(dispatcher.expert_range(2), (0, 2))
        self.assertTrue(torch.equal(routed_ids, expert_ids.flatten()))
        self.assertTrue(torch.equal(combined, x * 2))

        cp = ContextParallel()
        self.assertIs(cp.shard(x), x)
        self.assertIs(cp.gather(x), x)


if __name__ == "__main__":
    unittest.main()
