import unittest
from unittest.mock import Mock, call, patch

import torch

from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.attention import (
    CSA2Attention,
    CSA2Mode,
    RotaryEmbedding,
)
from dsv41_train.cp import (
    ContextParallel,
    bounded_replay_selection,
    sequence_positions,
)
from dsv41_train.models.dsv4.model import NgramHash
from dsv41_train.models.dsv4.moe import DispatchMetadata, RoutedExperts, TokenDispatcher
from dsv41_train.models.dsv4.parallel import apply_fsdp2
from dsv41_train.parallel import ParallelMeshes


class ModelTest(unittest.TestCase):
    def test_bounded_replay_keeps_each_packed_sequence_tail(self):
        input_ids = torch.arange(8).unsqueeze(0)
        token_mask = torch.ones_like(input_ids, dtype=torch.bool)
        sequence_ids = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]])
        positions = sequence_positions(token_mask, sequence_ids)

        selected = bounded_replay_selection(
            input_ids,
            token_mask,
            sequence_ids,
            positions,
            window=2,
            shard_size=8,
        )

        self.assertEqual(selected.source_indices.shape, (1, 8))
        torch.testing.assert_close(
            selected.source_indices[0, :4], torch.tensor([3, 4, 6, 7])
        )
        torch.testing.assert_close(
            selected.positions[0, :4], torch.tensor([3, 4, 1, 2])
        )
        torch.testing.assert_close(
            selected.sequence_ids[0, :4], torch.tensor([0, 0, 1, 1])
        )
        self.assertTrue(selected.token_mask[0, :4].all())
        self.assertFalse(selected.token_mask[0, 4:].any())

    def test_decoder_bounded_replay_limits_decoder_tokens_but_not_global_kv(self):
        config = DeepSeekV41Config.tiny()
        config.sliding_window = 2
        model = DeepSeekV41ForCausalLM(config)
        model.decoder_swa_bounded_replay_enable()
        model.gradient_checkpointing_enable()
        for layer in model.model.layers:
            layer.attention.indexer = None

        layer_lengths = []
        hooks = [
            layer.register_forward_pre_hook(
                lambda _module, args, lengths=layer_lengths: lengths.append(
                    args[0].shape[1]
                )
            )
            for layer in model.model.layers
        ]
        layer_output_sizes = []
        hooks.extend(
            layer.register_forward_hook(
                lambda _module, _args, output, sizes=layer_output_sizes: sizes.append(
                    len(output)
                )
            )
            for layer in model.model.layers
        )
        compressor_lengths = []
        compressor = model.model.layers[3].attention.compressor
        assert compressor is not None
        hooks.append(
            compressor.register_forward_pre_hook(
                lambda _module, args: compressor_lengths.append(args[0].shape[1])
            )
        )

        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        output = model(input_ids, labels=input_ids)
        for hook in hooks:
            hook.remove()

        self.assertEqual(layer_lengths, [8, 8, 8, 2, 2, 2])
        self.assertEqual(layer_output_sizes, [8, 8, 8, 8, 8, 8])
        self.assertEqual(compressor_lengths, [8])
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertTrue(torch.isfinite(model.model.embedding.weight.grad).all())

    def test_decoder_bounded_replay_requires_the_complete_backbone(self):
        with torch.device("meta"):
            model = DeepSeekV41ForCausalLM(
                DeepSeekV41Config.tiny(), layer_ids=[3, 4, 5]
            )

        with self.assertRaisesRegex(ValueError, "complete backbone"):
            model.decoder_swa_bounded_replay_enable()

    def test_attention_fallback_uses_window_indices(self):
        config = DeepSeekV41Config.tiny()
        attention = CSA2Attention(config, 0, RotaryEmbedding(config)).eval()
        query = torch.randn(1, config.num_attention_heads, 2, config.head_dim)
        kv = torch.randn(1, 4, config.head_dim)
        indices = torch.tensor([[[3, 1], [2, 0]]])
        mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)

        actual = attention._attend(query, kv, indices, mask, None, None)

        selected = kv[torch.arange(1).view(-1, 1, 1), indices]
        logits = torch.einsum("bhld,blwd->bhlw", query, selected)
        logits = logits * config.head_dim**-0.5
        sink = torch.zeros(1, config.num_attention_heads, 2, 1)
        probabilities = torch.softmax(
            torch.cat((logits, sink), dim=-1), dim=-1
        )[..., :-1]
        expected = torch.einsum(
            "bhlw,blwd->bhld", probabilities, selected
        ).transpose(1, 2)

        torch.testing.assert_close(actual, expected)

    def test_csa2_modes_follow_layer_reuse_boundaries(self):
        with torch.device("meta"):
            model = DeepSeekV41ForCausalLM(DeepSeekV41Config.tiny())

        self.assertEqual(
            [layer.attention.mode for layer in model.model.layers],
            [
                None,
                CSA2Mode.FULL,
                CSA2Mode.REUSE,
                CSA2Mode.FULL,
                CSA2Mode.REINDEX,
                CSA2Mode.REUSE,
            ],
        )

    def test_empty_expert_route_does_not_allocate_gradients(self):
        class EmptyReceiveDispatcher(TokenDispatcher):
            def dispatch(self, hidden, expert_ids, weights):
                metadata = DispatchMetadata(
                    token_count=hidden.shape[0],
                    token_indices=expert_ids.reshape(-1)[:0],
                )
                return (
                    hidden[:0],
                    torch.zeros(self.num_experts, dtype=torch.int32),
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

        self.assertEqual(experts.gate_up.grad.abs().sum(), 0)
        self.assertEqual(experts.down.grad.abs().sum(), 0)

    def test_expert_gradients_are_allocated_only_for_used_experts(self):
        config = DeepSeekV41Config.tiny()
        experts = RoutedExperts(config, TokenDispatcher())
        torch.nn.init.normal_(experts.gate_up)
        torch.nn.init.normal_(experts.down)
        hidden = torch.randn(3, config.hidden_size)
        expert_ids = torch.zeros(3, config.num_experts_per_tok, dtype=torch.long)
        weights = torch.ones(3, config.num_experts_per_tok)

        experts(hidden, expert_ids, weights).sum().backward()

        self.assertGreater(experts.gate_up.grad[0].abs().sum(), 0)
        self.assertGreater(experts.down.grad[0].abs().sum(), 0)
        self.assertEqual(experts.gate_up.grad[1:].abs().sum(), 0)
        self.assertEqual(experts.down.grad[1:].abs().sum(), 0)

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

    def test_explicit_local_parallel_primitives(self):
        dispatcher = TokenDispatcher()
        self.assertEqual(dispatcher.expert_range(2), (0, 2))
        x = torch.arange(12, dtype=torch.float32).view(3, 4)
        expert_ids = torch.tensor([[0, 1], [1, 0], [0, 1]])
        weights = torch.ones(3, 2)

        routed, counts, routed_weights, metadata = dispatcher.dispatch(
            x, expert_ids, weights
        )
        combined = dispatcher.combine(routed * routed_weights[:, None], metadata)

        self.assertTrue(torch.equal(counts, torch.tensor([3, 3], dtype=torch.int32)))
        self.assertTrue(torch.equal(combined, x * 2))

        cp = ContextParallel()
        self.assertIs(cp.shard(x), x)
        self.assertIs(cp.gather(x), x)


if __name__ == "__main__":
    unittest.main()
