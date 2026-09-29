from dsv41_train.models.dsv4 import DeepSeekV41Config


def small_config(**overrides):
    values = dict(
        vocab_size=32, hidden_size=32, num_hidden_layers=5,
        num_attention_heads=2, head_dim=32, q_lora_rank=16, qk_rope_head_dim=16,
        max_position_embeddings=64, sliding_window=8,
        compress_ratios=[0, 2, 2, 1, 1], kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 2, 3, 4], candidate_source_layer_id=3,
        candidate_topk_blocks=2, candidate_block_size=2,
        index_n_heads=2, index_head_dim=32, index_topk=3,
        o_groups=2, o_lora_rank=16, moe_intermediate_size=32,
        n_routed_experts=4, num_experts_per_tok=2,
        hc_mult=2, hc_sinkhorn_iters=3,
        engram_layer_ids=[1], engram_num_embeddings=[128],
        engram_vocab_size=31, engram_max_ngram_size=3,
        engram_n_heads=1, engram_head_dim=8, engram_compressed_vocab_size=32,
    )
    values.update(overrides)
    return DeepSeekV41Config(**values)
