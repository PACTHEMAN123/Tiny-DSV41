"""Small model and checkpoint fixtures for Qwen tests."""

import json
from pathlib import Path

import torch

from dsv41_train.models.qwen import Qwen3MoeConfig, Qwen3MoeForCausalLM


def tiny_config(**overrides) -> Qwen3MoeConfig:
    values = dict(
        vocab_size=32, hidden_size=16, intermediate_size=32, moe_intermediate_size=8,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        num_experts=4, num_experts_per_tok=2, max_position_embeddings=32,
        max_window_layers=2, bos_token_id=1, eos_token_id=2,
    )
    return Qwen3MoeConfig(**(values | overrides))


def smoke_config() -> Qwen3MoeConfig:
    return tiny_config(
        vocab_size=64, hidden_size=32, intermediate_size=64, moe_intermediate_size=16,
        num_attention_heads=4, num_key_value_heads=2, num_experts=8,
    )


def to_hf_state_dict(model: Qwen3MoeForCausalLM) -> dict[str, torch.Tensor]:
    converted = {}
    for name, value in model.state_dict().items():
        if name.endswith("experts.gate_up_proj"):
            prefix = name.removesuffix("gate_up_proj")
            for expert_id, expert in enumerate(value):
                gate, up = expert.chunk(2, dim=0)
                converted[f"{prefix}{expert_id}.gate_proj.weight"] = gate.clone()
                converted[f"{prefix}{expert_id}.up_proj.weight"] = up.clone()
        elif name.endswith("experts.down_proj"):
            prefix = name.removesuffix("down_proj")
            for expert_id, expert in enumerate(value):
                converted[f"{prefix}{expert_id}.down_proj.weight"] = expert.clone()
        else:
            converted[name] = value.clone()
    return converted


def write_hf_checkpoint(folder: Path, model: Qwen3MoeForCausalLM) -> None:
    from safetensors.torch import save_file

    state = to_hf_state_dict(model)
    (folder / "config.json").write_text(json.dumps(model.config.to_dict()), encoding="utf-8")
    save_file(state, folder / "model.safetensors")
    index = {"metadata": {}, "weight_map": dict.fromkeys(state, "model.safetensors")}
    (folder / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
