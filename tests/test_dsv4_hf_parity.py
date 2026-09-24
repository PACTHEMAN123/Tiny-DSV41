import unittest

import torch

from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.model import (
    Engram,
    HyperConnection,
    NgramHash,
    RotaryEmbedding,
    _fake_quant_fp4_block,
    _fake_quant_fp8_block,
)

try:
    from transformers.models.deepseek_v41.configuration_deepseek_v41 import (
        DeepseekV41TextConfig,
    )
    from transformers.models.deepseek_v41.modeling_deepseek_v41 import (
        DeepseekV41Engram,
        DeepseekV41HyperConnection,
        DeepseekV41NgramHashState,
        DeepseekV41RotaryEmbedding,
        DeepseekV41TextModel,
        _fake_quant_fp4_block as hf_fake_quant_fp4_block,
        _fake_quant_fp8_block as hf_fake_quant_fp8_block,
    )
except ImportError:
    DeepseekV41TextConfig = None


def config_values(*, engram: bool = False) -> dict:
    return {
        "vocab_size": 32,
        "hidden_size": 32,
        "num_hidden_layers": 4,
        "num_attention_heads": 2,
        "head_dim": 32,
        "q_lora_rank": 16,
        "qk_rope_head_dim": 16,
        "rope_theta": 10000.0,
        "compress_rope_theta": 160000.0,
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 4,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 16,
        },
        "max_position_embeddings": 64,
        "sliding_window": 8,
        "compress_ratios": [0, 2, 2, 1],
        "kv_source_layer_ids": [1, 3],
        "index_source_layer_ids": [1, 2, 3],
        "candidate_source_layer_id": 2,
        "candidate_topk_blocks": 2,
        "candidate_block_size": 2,
        "index_n_heads": 2,
        "index_head_dim": 32,
        "index_topk": 3,
        "o_groups": 2,
        "o_lora_rank": 16,
        "moe_intermediate_size": 16,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "scoring_func": "sqrtsoftplus",
        "norm_topk_prob": True,
        "routed_scaling_factor": 1.5,
        "swiglu_limit": 10.0,
        "hc_mult": 2,
        "hc_sinkhorn_iters": 3,
        "hc_eps": 1e-6,
        "engram_layer_ids": [0] if engram else [],
        "engram_num_embeddings": [128] if engram else [],
        "engram_vocab_size": 31,
        "engram_max_ngram_size": 3,
        "engram_n_heads": 1,
        "engram_head_dim": 8,
        "engram_pad_id": 0,
        "engram_compressed_vocab_size": 16,
        "rms_norm_eps": 1e-20,
        "pad_token_id": 0,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


def native_config(*, engram: bool = False) -> DeepSeekV41Config:
    return DeepSeekV41Config(**config_values(engram=engram))


def hf_config(*, engram: bool = False):
    values = config_values(engram=engram)
    values.update(
        n_shared_experts=1,
        num_nextn_predict_layers=0,
        dspark_target_layer_ids=[],
        use_cache=False,
    )
    return DeepseekV41TextConfig(**values)


def copy_tensor(target: torch.Tensor, source: torch.Tensor) -> None:
    target.copy_(source.reshape_as(target))


@torch.no_grad()
def copy_hf_model(native, reference) -> None:
    copy_tensor(native.embedding.weight, reference.embed_tokens.weight)
    copy_tensor(native.norm.weight, reference.norm.weight)
    for target, source in zip(native.layers, reference.layers):
        for target_hc, source_hc in (
            (target.attention_hc, source.attn_hc),
            (target.moe_hc, source.ffn_hc),
        ):
            copy_tensor(target_hc.fn, source_hc.fn)
            copy_tensor(target_hc.base, source_hc.base)
            copy_tensor(target_hc.scale, source_hc.scale)

        copy_tensor(target.input_norm.weight, source.input_layernorm.weight)
        copy_tensor(target.post_attention_norm.weight, source.post_attention_layernorm.weight)

        attention = target.attention
        reference_attention = source.self_attn
        for target_module, source_module in (
            (attention.q_a, reference_attention.q_a_proj),
            (attention.q_norm, reference_attention.q_a_norm),
            (attention.q_b, reference_attention.q_b_proj),
            (attention.kv_proj, reference_attention.kv_proj),
            (attention.kv_norm, reference_attention.kv_norm),
            (attention.o_b, reference_attention.o_b_proj),
        ):
            copy_tensor(target_module.weight, source_module.weight)
        copy_tensor(attention.o_a.weight, reference_attention.o_a_proj.weight)
        copy_tensor(attention.sinks.weight, reference_attention.sinks)

        if attention.compressor is not None:
            reference_compressor = reference_attention.compressor
            copy_tensor(attention.compressor.kv_proj.weight, reference_compressor.kv_proj.weight)
            copy_tensor(attention.compressor.norm.weight, reference_compressor.kv_norm.weight)
            if attention.compressor.gate_proj is not None:
                copy_tensor(
                    attention.compressor.gate_proj.weight,
                    reference_compressor.gate_proj.weight,
                )

        if attention.indexer is not None:
            reference_indexer = reference_attention.indexer
            copy_tensor(attention.indexer.q_proj.weight, reference_indexer.q_b_proj.weight)
            copy_tensor(
                attention.indexer.weight_proj.weight,
                reference_indexer.weights_proj.weight,
            )
            if attention.indexer.owns_keys:
                copy_tensor(attention.indexer.k_proj.weight, reference_indexer.k_proj.weight)
                copy_tensor(attention.indexer.k_norm.weight, reference_indexer.k_norm.weight)

        moe = target.moe
        reference_moe = source.mlp
        copy_tensor(moe.router.weight, reference_moe.gate.weight)
        copy_tensor(moe.router.selection_bias, reference_moe.gate.e_score_correction_bias)
        for expert_id in range(moe.routed.num_experts):
            copy_tensor(
                moe.routed.gate_up[expert_id],
                reference_moe.experts.gate_up_proj[expert_id],
            )
            copy_tensor(
                moe.routed.down[expert_id],
                reference_moe.experts.down_proj[expert_id],
            )
        for target_module, source_module in (
            (moe.shared.gate, reference_moe.shared_experts.gate_proj),
            (moe.shared.up, reference_moe.shared_experts.up_proj),
            (moe.shared.down, reference_moe.shared_experts.down_proj),
        ):
            copy_tensor(target_module.weight, source_module.weight)


@unittest.skipUnless(DeepseekV41TextConfig is not None, "requires the DeepSeek V4.1 HF reference")
class HuggingFaceParityTest(unittest.TestCase):
    def test_quantization_helpers_match_huggingface(self):
        generator = torch.Generator().manual_seed(123)
        values = torch.randn(2, 3, 64, generator=generator) * 7

        torch.testing.assert_close(
            _fake_quant_fp8_block(values),
            hf_fake_quant_fp8_block(values),
            rtol=0,
            atol=0,
        )
        for block_size, e4m3_scales in ((32, False), (16, True)):
            torch.testing.assert_close(
                _fake_quant_fp4_block(
                    values, block_size, e4m3_scales=e4m3_scales
                ),
                hf_fake_quant_fp4_block(
                    values, block_size, e4m3_scales=e4m3_scales
                ),
                rtol=0,
                atol=0,
            )

    def test_hyper_connection_and_rope_match_huggingface(self):
        native = native_config()
        reference = hf_config()
        native_hc = HyperConnection(native)
        reference_hc = DeepseekV41HyperConnection(reference)
        native_hc.load_state_dict(reference_hc.state_dict(), strict=True)
        streams = torch.randn(2, 7, native.hc_mult, native.hidden_size)

        for actual, expected in zip(native_hc(streams), reference_hc(streams)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        positions = torch.arange(7).expand(2, -1)
        native_rope = RotaryEmbedding(native)
        reference_rope = DeepseekV41RotaryEmbedding(reference)
        for compressed, layer_type in ((False, "main"), (True, "compress")):
            actual = native_rope(streams[:, :, 0], positions, compressed=compressed)
            expected = reference_rope(
                streams[:, :, 0], position_ids=positions, layer_type=layer_type
            )
            for actual_part, expected_part in zip(actual, expected):
                torch.testing.assert_close(actual_part, expected_part, rtol=0, atol=0)

    def test_ngram_hash_and_engram_match_huggingface(self):
        native = native_config(engram=True)
        reference = hf_config(engram=True)
        token_map = torch.arange(native.vocab_size).remainder(
            native.engram_compressed_vocab_size
        )
        native_hash = NgramHash(native, token_map=token_map)
        reference_hash = DeepseekV41NgramHashState(reference)
        reference_hash.token_map = token_map
        reference_hash.pad_id = int(token_map[native.engram_pad_id])

        torch.testing.assert_close(native_hash.primes, reference_hash.primes)
        torch.testing.assert_close(native_hash.offsets, reference_hash.offsets)
        torch.testing.assert_close(native_hash.multipliers, reference_hash.multipliers)
        input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 0, 8, 9]])
        token_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 1, 1]], dtype=torch.bool)
        torch.testing.assert_close(
            native_hash(input_ids, token_mask),
            reference_hash(input_ids, token_mask, None),
        )

        native_engram = Engram(native)
        reference_engram = DeepseekV41Engram(reference, 0)
        with torch.no_grad():
            copy_tensor(native_engram.proj.weight, reference_engram.wkv.weight)
            copy_tensor(native_engram.q_weight, reference_engram.q_weight)
            copy_tensor(native_engram.k_weight, reference_engram.k_weight)
        streams = torch.randn(2, 5, native.hc_mult, native.hidden_size)
        columns = (native.engram_max_ngram_size - 1) * native.engram_n_heads
        rows = torch.randn(2, 5, columns, native.engram_head_dim)
        torch.testing.assert_close(
            native_engram(streams, rows, token_mask),
            reference_engram(streams, rows, token_mask),
            rtol=0,
            atol=0,
        )

    def test_every_decoder_layer_matches_huggingface(self):
        torch.manual_seed(42)
        native = DeepSeekV41ForCausalLM(native_config()).model
        reference = DeepseekV41TextModel(hf_config())
        copy_hf_model(native, reference)
        native.eval()
        reference.eval()

        native_layers = []
        reference_layers = []
        native_handles = [
            layer.register_forward_hook(
                lambda _module, _args, output: native_layers.append(
                    (output[0].detach(), output[1].detach())
                )
            )
            for layer in native.layers
        ]
        reference_handles = [
            layer.register_forward_hook(
                lambda _module, _args, output: reference_layers.append(
                    (output[0].detach(), output[1].detach())
                )
            )
            for layer in reference.layers
        ]
        try:
            input_ids = torch.tensor(
                [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]]
            )
            attention_mask = torch.ones_like(input_ids)
            actual, _ = native(input_ids, attention_mask)
            expected = reference(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).last_hidden_state
        finally:
            for handle in (*native_handles, *reference_handles):
                handle.remove()

        self.assertEqual(len(native_layers), native.config.num_hidden_layers)
        self.assertEqual(len(reference_layers), native.config.num_hidden_layers)
        for layer_id, (actual_layer, expected_layer) in enumerate(
            zip(native_layers, reference_layers)
        ):
            with self.subTest(layer=layer_id):
                for actual_part, expected_part in zip(actual_layer, expected_layer):
                    torch.testing.assert_close(actual_part, expected_part, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
