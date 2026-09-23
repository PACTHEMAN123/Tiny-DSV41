import unittest
from unittest.mock import Mock, call, patch

import torch

from dsv41_train.dispatch import TokenDispatcher
from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.parallel import apply_fsdp2
from dsv41_train.parallel import ContextParallel, ParallelMeshes


class ModelTest(unittest.TestCase):
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
                    expected.append(call(target, mesh=meshes.fsdp))
                    self.assertEqual(shard.call_args_list, expected)

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
