"""Hugging Face safetensors loading for the native Qwen model."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

from ...dispatch import TokenDispatcher
from ...parallel import ContextParallel
from .config import Qwen3MoeConfig
from .model import Qwen3MoeExperts, Qwen3MoeForCausalLM


_EXPERT_WEIGHT = re.compile(
    r"^(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def _load_assigned(model: Qwen3MoeForCausalLM, state: dict[str, torch.Tensor]) -> set[str]:
    if not state:
        return set()
    result = model.load_state_dict(state, strict=False, assign=True)
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys: {result.unexpected_keys[:5]}")
    return set(state)


def _fuse_complete_experts(
    model: Qwen3MoeForCausalLM,
    parts: dict[str, dict[str, list[torch.Tensor | None]]],
    intermediate_size: int,
) -> set[str]:
    loaded = set()
    complete = [
        prefix
        for prefix, group in parts.items()
        if all(value is not None for values in group.values() for value in values)
    ]
    for prefix in complete:
        group = parts.pop(prefix)
        gates = group["gate_proj"]
        ups = group["up_proj"]
        downs = group["down_proj"]
        first = gates[0]
        first_down = downs[0]
        assert first is not None and first_down is not None
        gate_up = torch.empty(
            len(gates),
            2 * intermediate_size,
            first.shape[1],
            dtype=first.dtype,
            device=first.device,
        )
        down = torch.empty(
            len(downs),
            first_down.shape[0],
            intermediate_size,
            dtype=first_down.dtype,
            device=first_down.device,
        )
        for expert_id, (gate, up, down_part) in enumerate(zip(gates, ups, downs)):
            assert gate is not None and up is not None and down_part is not None
            gate_up[expert_id, :intermediate_size].copy_(gate)
            gate_up[expert_id, intermediate_size:].copy_(up)
            down[expert_id].copy_(down_part)
        loaded.update(
            _load_assigned(
                model,
                {
                    f"{prefix}.gate_up_proj": gate_up,
                    f"{prefix}.down_proj": down,
                },
            )
        )
    return loaded


def load_qwen3_moe(
    folder: str | Path,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | None = torch.bfloat16,
    context_parallel: ContextParallel | None = None,
    token_dispatcher: TokenDispatcher | None = None,
) -> Qwen3MoeForCausalLM:
    """Load sharded HF weights without depending on Transformers."""

    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("loading HF weights requires the safetensors package") from error

    folder = Path(folder)
    config = Qwen3MoeConfig.from_json(folder / "config.json")
    index_path = folder / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as file:
        index = json.load(file)
    weight_map: dict[str, str] = index["weight_map"]

    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(config, context_parallel, token_dispatcher)
    expected = set(model.state_dict())
    expert_ranges = {
        name: (module.expert_start, module.num_experts)
        for name, module in model.named_modules()
        if isinstance(module, Qwen3MoeExperts)
    }

    loaded = set()
    expert_parts: dict[str, dict[str, list[torch.Tensor | None]]] = {}
    for shard_name in sorted(set(weight_map.values())):
        state = load_file(str(folder / shard_name), device=str(device))
        if dtype is not None:
            state = {
                name: value.to(dtype=dtype) if value.is_floating_point() else value
                for name, value in state.items()
            }
        regular = {}
        for name, value in state.items():
            match = _EXPERT_WEIGHT.fullmatch(name)
            if match is None:
                regular[name] = value
                continue
            prefix, expert_id_text, projection = match.groups()
            if prefix not in expert_ranges:
                raise RuntimeError(f"checkpoint contains experts for a dense layer: {prefix}")
            global_expert_id = int(expert_id_text)
            expert_start, local_experts = expert_ranges[prefix]
            if not 0 <= global_expert_id < config.num_experts:
                raise RuntimeError(f"checkpoint expert id is out of range: {global_expert_id}")
            if not expert_start <= global_expert_id < expert_start + local_experts:
                continue
            group = expert_parts.setdefault(
                prefix,
                {
                    key: [None] * local_experts
                    for key in ("gate_proj", "up_proj", "down_proj")
                },
            )
            local_expert_id = global_expert_id - expert_start
            if group[projection][local_expert_id] is not None:
                raise RuntimeError(f"duplicate checkpoint weight: {name}")
            group[projection][local_expert_id] = value
        loaded.update(_load_assigned(model, regular))
        loaded.update(
            _fuse_complete_experts(
                model,
                expert_parts,
                config.moe_intermediate_size,
            )
        )

    if expert_parts:
        raise RuntimeError(f"incomplete expert weights: {sorted(expert_parts)[:5]}")
    if loaded != expected:
        missing = sorted(expected - loaded)
        raise RuntimeError(f"checkpoint did not load every parameter: {missing[:5]}")
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("checkpoint loading left meta parameters in the model")
    return model


__all__ = ["load_qwen3_moe"]
