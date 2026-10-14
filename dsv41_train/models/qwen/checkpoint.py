"""Hugging Face safetensors loading for the native Qwen model."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

import torch

from ...dispatch import TokenDispatcher
from ...parallel import ContextParallel
from ...loading.reader import ShardedSafeTensorReader
from ...loading.quantization import dequantize_blockwise_fp8
from ...lora import LoRAConfig, inject_lora, load_adapter_state, read_adapter, validate_adapter_layout
from .config import Qwen3MoeConfig
from .model import Qwen3MoeExperts, Qwen3MoeForCausalLM


_EXPERT_WEIGHT = re.compile(
    r"^(model\.layers\.\d+\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)

QWEN_LORA_TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")


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
    setup_callback=None,
    module_callback=None,
    lora_config: LoRAConfig | None = None,
    adapter_path: str | Path | None = None,
) -> Qwen3MoeForCausalLM:
    """Load sharded HF weights without depending on Transformers."""

    folder = Path(folder)
    raw_config = json.loads((folder / "config.json").read_text())
    if raw_config.get("model_type", "qwen3_moe") != "qwen3_moe":
        raise ValueError("this loader requires a Qwen3 MoE checkpoint")
    quantization = raw_config.get("quantization_config") or {}
    if quantization and (quantization.get("quant_method") != "fp8" or not quantization.get("weight_block_size")):
        raise ValueError("only block-scaled FP8 Qwen checkpoints are supported")
    config = Qwen3MoeConfig.from_json(folder / "config.json")
    if adapter_path is not None and lora_config is None:
        raise ValueError("adapter_path requires lora_config")
    if lora_config is not None and set(lora_config.targets) - set(QWEN_LORA_TARGETS):
        raise ValueError("Qwen LoRA currently supports self_attn q_proj/k_proj/v_proj/o_proj targets only")
    index_path = folder / "model.safetensors.index.json"
    with index_path.open(encoding="utf-8") as file:
        index = json.load(file)
    weight_map: dict[str, str] = index["weight_map"]

    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(config, context_parallel, token_dispatcher)
    expected = set(model.state_dict())
    if setup_callback is not None:
        setup_callback(model)
    payload = read_adapter(adapter_path) if adapter_path is not None else None
    if lora_config is not None:
        if hasattr(model.model, "engram_tables"):
            raise ValueError("Qwen LoRA with added Engram is not supported")
        # assign=True preserves requires_grad from the meta parameters. Freeze
        # root weights and every base layer before any checkpoint allocation.
        model.requires_grad_(False)
        if payload is not None:
            validate_adapter_layout(model, payload)
            modules = {f"model.layers.{layer}.{target}": asdict(lora_config)
                       for layer in model.model.layer_ids for target in lora_config.targets}
            if payload.get("modules") != modules:
                raise ValueError("adapter configuration does not match the complete Qwen backbone")
            expected_weights = {f"{name}.lora_{kind}" for name in modules for kind in ("a", "b")}
            if set(payload.get("weights", {})) != expected_weights:
                raise ValueError("adapter tensor names do not match the complete Qwen backbone")
    loaded = set()
    components = [("model.embed_tokens", model.model.embed_tokens)]
    components += [(f"model.layers.{index}", layer) for index, layer in enumerate(model.model.layers)]
    components += [("model.norm", model.model.norm), ("lm_head", model.lm_head)]
    with ShardedSafeTensorReader(folder) as reader:
        def read(name):
            value = reader.tensor(name, device)
            scale_name = name + "_scale_inv"
            if value.dtype == torch.float8_e4m3fn:
                if not quantization or scale_name not in weight_map:
                    raise ValueError(f"missing FP8 scales for {name}")
                return dequantize_blockwise_fp8(value, reader.tensor(scale_name, device),
                    quantization["weight_block_size"], dtype=dtype or torch.bfloat16)
            return value.to(dtype) if dtype is not None and value.is_floating_point() else value

        for prefix, module in components:
            state = {}
            for name in sorted(expected):
                if not name.startswith(prefix + ".") or ".mlp.experts." in name:
                    continue
                state[name[len(prefix)+1:]] = read(name)
                loaded.add(name)
            experts = getattr(getattr(module, "mlp", None), "experts", None)
            if experts is not None:
                fused = {}
                for local_id in range(experts.num_experts):
                    global_id = experts.expert_start + local_id
                    expert_prefix = f"{prefix}.mlp.experts.{global_id}"
                    for projection in ("gate_proj", "up_proj", "down_proj"):
                        value = read(f"{expert_prefix}.{projection}.weight")
                        key = "down_proj" if projection == "down_proj" else "gate_up_proj"
                        if key not in fused:
                            fused[key] = torch.empty_like(getattr(experts, key), device=device, dtype=value.dtype)
                        if key == "down_proj":
                            fused[key][local_id].copy_(value)
                        else:
                            offset = 0 if projection == "gate_proj" else config.moe_intermediate_size
                            fused[key][local_id, offset:offset+config.moe_intermediate_size].copy_(value)
                for key, value in fused.items():
                    state[f"mlp.experts.{key}"] = value
                    loaded.add(f"{prefix}.mlp.experts.{key}")
            result = module.load_state_dict(state, strict=False, assign=True)
            if result.unexpected_keys:
                raise RuntimeError(f"unexpected checkpoint keys: {result.unexpected_keys[:5]}")
            del state
            if lora_config is not None and prefix.startswith("model.layers."):
                inject_lora(module, lora_config)
                if payload is not None:
                    load_adapter_state(module, payload, prefix=prefix)
            if module_callback is not None:
                module_callback(prefix, module)
        model.load_bytes_read = reader.bytes_read
    if loaded != expected:
        missing = sorted(expected - loaded)
        raise RuntimeError(f"checkpoint did not load every parameter: {missing[:5]}")
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("checkpoint loading left meta parameters in the model")
    return model


__all__ = ["QWEN_LORA_TARGETS", "load_qwen3_moe"]
