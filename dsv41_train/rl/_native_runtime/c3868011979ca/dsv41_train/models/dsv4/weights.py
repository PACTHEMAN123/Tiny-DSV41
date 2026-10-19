"""DSV4 checkpoint names and fusion rules; independent of storage placement."""

from ...loading.source import WeightSource


def layer_sources(layer):
    prefix = f"layers.{layer.layer_id}"
    plain = {
        "attention_hc.base": "hc_attn_base", "attention_hc.fn": "hc_attn_fn",
        "attention_hc.scale": "hc_attn_scale", "moe_hc.base": "hc_ffn_base",
        "moe_hc.fn": "hc_ffn_fn", "moe_hc.scale": "hc_ffn_scale",
        "attention.sinks.weight": "attn.attn_sink", "attention.q_norm.weight": "attn.q_norm.weight",
        "attention.kv_norm.weight": "attn.kv_norm.weight", "input_norm.weight": "attn_norm.weight",
        "post_attention_norm.weight": "ffn_norm.weight", "moe.router.weight": "ffn.gate.weight",
        "moe.router.selection_bias": "ffn.gate.bias",
    }
    quantized = {
        "attention.q_a.weight": "attn.wq_a", "attention.q_b.weight": "attn.wq_b",
        "attention.kv_proj.weight": "attn.wkv", "attention.o_a.weight": "attn.wo_a",
        "attention.o_b.weight": "attn.wo_b", "moe.shared.gate.weight": "ffn.shared_experts.w1",
        "moe.shared.up.weight": "ffn.shared_experts.w3", "moe.shared.down.weight": "ffn.shared_experts.w2",
    }
    compressor = layer.attention.compressor
    if compressor is not None:
        plain["attention.compressor.kv_proj.weight"] = "attn.compressor.wkv.weight"
        plain["attention.compressor.norm.weight"] = "attn.compressor.norm.weight"
        if compressor.gate_proj is not None:
            plain["attention.compressor.gate_proj.weight"] = "attn.compressor.wgate.weight"
    indexer = layer.attention.indexer
    if indexer is not None:
        quantized["attention.indexer.q_proj.weight"] = "attn.indexer.wq_b"
        plain["attention.indexer.weight_proj.weight"] = "attn.indexer.weights_proj.weight"
        if indexer.owns_keys:
            plain["attention.indexer.k_proj.weight"] = "attn.indexer.wk.weight"
            plain["attention.indexer.k_norm.weight"] = "attn.indexer.k_norm.weight"
    if layer.engram is not None:
        plain.update({"engram.q_weight": "engram.q_weight", "engram.k_weight": "engram.k_weight"})
        quantized["engram.proj.weight"] = "engram.wkv"
    result = {key: WeightSource((f"{prefix}.{name}",)) for key, name in plain.items()}
    result.update({key: WeightSource((f"{prefix}.{name}",), "fp8_block") for key, name in quantized.items()})
    routed = layer.moe.routed
    for local in range(routed.num_experts):
        source = f"{prefix}.ffn.experts.{routed.expert_start + local}"
        result[f"moe.routed.gate_up.{local}"] = WeightSource((source + ".w1", source + ".w3"), "fp4")
        result[f"moe.routed.down.{local}"] = WeightSource((source + ".w2",), "fp4")
    return result
