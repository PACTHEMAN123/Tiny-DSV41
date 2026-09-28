"""Small DSV4 model and released-checkpoint fixture for LoRA integration tests."""

import json
from pathlib import Path

import torch

from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM
from test_dsv4_checkpoint import write_safetensors

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


def synthetic_checkpoint(folder: Path, config=None):
    if config is None:
        config = small_config(q_lora_rank=32, o_lora_rank=32, engram_layer_ids=[], engram_num_embeddings=[])
    template = DeepSeekV41ForCausalLM(config)
    tensors = {
        "embed.weight": template.model.embedding.weight.detach(),
        "norm.weight": template.model.norm.weight.detach(),
        "head.weight": template.lm_head.weight.detach(),
    }
    expected = {"model.embedding.weight": tensors["embed.weight"],
                "model.norm.weight": tensors["norm.weight"], "lm_head.weight": tensors["head.weight"]}

    def add(source, target, tensor, quantized=False, fp4=False):
        tensor = tensor.detach()
        if fp4:
            rows, cols = tensor.shape
            tensors[source + ".weight"] = torch.full((rows, cols // 2), 0x21, dtype=torch.int8)
            tensors[source + ".scale"] = torch.full((rows, cols // 32), 0.01)
            value = torch.tensor([0.005, 0.01] * (cols // 2)).expand(rows, -1).clone()
        elif quantized:
            value = tensor.reshape(-1, tensor.shape[-1]).to(torch.float8_e4m3fn)
            tensors[source + ".weight"] = value
            tensors[source + ".scale"] = torch.ones((value.shape[0] + 31) // 32, (value.shape[1] + 31) // 32)
            value = value.float().reshape_as(tensor)
        else:
            tensors[source] = tensor
            value = tensor
        expected[target] = value

    for layer in template.model.layers:
        prefix, target = f"layers.{layer.layer_id}", f"model.layers.{layer.layer_id}"
        state = layer.state_dict()
        mapping = {
            "attention_hc.base": "hc_attn_base", "attention_hc.fn": "hc_attn_fn",
            "attention_hc.scale": "hc_attn_scale", "moe_hc.base": "hc_ffn_base",
            "moe_hc.fn": "hc_ffn_fn", "moe_hc.scale": "hc_ffn_scale",
            "attention.sinks.weight": "attn.attn_sink",
            "attention.q_norm.weight": "attn.q_norm.weight",
            "attention.kv_norm.weight": "attn.kv_norm.weight",
            "input_norm.weight": "attn_norm.weight", "post_attention_norm.weight": "ffn_norm.weight",
            "moe.router.weight": "ffn.gate.weight", "moe.router.selection_bias": "ffn.gate.bias",
        }
        quantized = {
            "attention.q_a.weight": "attn.wq_a", "attention.q_b.weight": "attn.wq_b",
            "attention.kv_proj.weight": "attn.wkv", "attention.o_a.weight": "attn.wo_a",
            "attention.o_b.weight": "attn.wo_b", "moe.shared.gate.weight": "ffn.shared_experts.w1",
            "moe.shared.up.weight": "ffn.shared_experts.w3", "moe.shared.down.weight": "ffn.shared_experts.w2",
        }
        if layer.attention.compressor is not None:
            mapping.update({"attention.compressor.kv_proj.weight": "attn.compressor.wkv.weight",
                            "attention.compressor.norm.weight": "attn.compressor.norm.weight"})
            if layer.attention.compressor.gate_proj is not None:
                mapping["attention.compressor.gate_proj.weight"] = "attn.compressor.wgate.weight"
        if layer.attention.indexer is not None:
            quantized["attention.indexer.q_proj.weight"] = "attn.indexer.wq_b"
            mapping["attention.indexer.weight_proj.weight"] = "attn.indexer.weights_proj.weight"
            if layer.attention.indexer.owns_keys:
                mapping.update({"attention.indexer.k_proj.weight": "attn.indexer.wk.weight",
                                "attention.indexer.k_norm.weight": "attn.indexer.k_norm.weight"})
        for name, source in mapping.items():
            add(f"{prefix}.{source}", f"{target}.{name}", state[name])
        for name, source in quantized.items():
            add(f"{prefix}.{source}", f"{target}.{name}", state[name], quantized=True)
        for index in range(config.n_routed_experts):
            expert = f"{prefix}.ffn.experts.{index}"
            weight = state[f"moe.routed.gate_up.{index}"]
            add(expert + ".w1", "gate", weight[:config.moe_intermediate_size], fp4=True)
            add(expert + ".w3", "up", weight[config.moe_intermediate_size:], fp4=True)
            expected[f"{target}.moe.routed.gate_up.{index}"] = torch.cat((expected.pop("gate"), expected.pop("up")))
            add(expert + ".w2", f"{target}.moe.routed.down.{index}", state[f"moe.routed.down.{index}"], fp4=True)
    write_safetensors(folder / "model.safetensors", tensors)
    (folder / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "model.safetensors" for name in tensors}}), encoding="utf-8",
    )
    return config, expected
