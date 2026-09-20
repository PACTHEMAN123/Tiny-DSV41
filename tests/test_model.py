import unittest

import torch

from dsv41_train import DeepSeekV41Config, DeepSeekV41ForCausalLM


class ModelTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
