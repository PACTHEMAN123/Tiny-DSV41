"""Narrow, in-memory LoRA contract for the installed NVIDIA DeepSeek V4.1.

Only attention wq_a is adapted. vLLM packs it into the first, unsharded
fused_wqa_wkv output slice; the KV slice retains a zero adapter. These helpers
do not import the model implementation or initialize CUDA.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

import torch

# Keep the mounted architecture: vLLM uses this name to construct and validate
# EngramConfig before model loading. Only this process's class registry changes.
LORA_ARCHITECTURE = "DeepseekV41ForCausalLM"
LORA_TARGET_MODULE = "fused_wqa_wkv"
LORA_RANK = 8
LORA_ALPHA = 8
# Preserve the installed text model's full packing metadata for base quantization.
# These mappings do not admit adapter targets; validate_deployment and the tensor
# key guard below enforce the narrower query-only scope independently.
PACKED_MODULES_MAPPING = {
    "gate_up_proj": ["w1", "w3"],
    LORA_TARGET_MODULE: ["wq_a", "wkv"],
    "fused_wkv_wgate": ["wkv", "wgate"],
}
_ADAPTER_NAME = re.compile(
    r"language_model\.model\.layers\.(\d+)\.attn\.wq_a\.lora_([AB])\.weight\Z"
)


def _field(config, name):
    return config[name] if isinstance(config, Mapping) else getattr(config, name)


def text_dimensions(config) -> tuple[int, int, int, int]:
    """Accept mounted JSON or vLLM's flattened HF configuration."""
    try:
        nested = _field(config, "text_config")
    except (AttributeError, KeyError):
        nested = None
    config = nested or config
    values = tuple(int(_field(config, name)) for name in (
        "hidden_size", "q_lora_rank", "head_dim", "num_hidden_layers"
    ))
    if any(value <= 0 for value in values):
        raise ValueError("DeepSeek text dimensions must be positive")
    return values


def lora_target_modules(config) -> list[str]:
    """Use whole runtime paths: packed-child suffix matching also admits Engram.wkv."""
    layers = text_dimensions(config)[3]
    return [f"language_model.model.layers.{layer}.attn.{LORA_TARGET_MODULE}"
            for layer in range(layers)]


def validate_deployment(vllm_config) -> None:
    """Reject unsupported scopes before constructing the base model."""
    config = vllm_config.lora_config
    targets = lora_target_modules(vllm_config.model_config.hf_config)
    if config is None or config.target_modules != targets:
        raise ValueError("DeepSeek LoRA requires exact backbone attention target_modules")
    if config.max_lora_rank != LORA_RANK or config.lora_dtype != torch.bfloat16:
        raise ValueError("DeepSeek LoRA requires rank 8 and BF16 adapters")
    if config.fully_sharded_loras or config.enable_tower_connector_lora:
        raise ValueError("DeepSeek LoRA supports unsharded text adapters only")
    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("DeepSeek LoRA requires pipeline_parallel_size=1")
    if vllm_config.speculative_config is not None:
        raise ValueError("DeepSeek LoRA does not adapt speculative draft models")
    text_dimensions(vllm_config.model_config.hf_config)


def validate_adapter_tensors(tensors, text_config, rank=LORA_RANK) -> tuple[str, ...]:
    """Validate CPU BF16 pairs and return exact adapter-visible module names.

    Tensor keys intentionally use vLLM's runtime namespace and no PEFT prefix;
    callers must pass weights_mapper=None to LoRAModel.from_lora_tensors.
    A subset of backbone layers is allowed. No KV, MTP, vision, or expert key
    is admitted, and no missing or malformed tensor is silently ignored.
    """
    if rank != LORA_RANK or not tensors:
        raise ValueError("DeepSeek LoRA requires a nonempty rank-8 adapter")
    hidden, q_rank, _, layers = text_dimensions(text_config)
    pairs = {}
    for name, tensor in tensors.items():
        match = _ADAPTER_NAME.fullmatch(name)
        if match is None or int(match[1]) >= layers:
            raise ValueError(f"Unsupported DeepSeek LoRA tensor: {name}")
        shape = (rank, hidden) if match[2] == "A" else (q_rank, rank)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"{name} must have shape {shape}")
        if tensor.dtype != torch.bfloat16 or tensor.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU BF16 tensor")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} contains nonfinite values")
        pairs.setdefault(name.rsplit(".lora_", 1)[0], set()).add(match[2])
    if any(sides != {"A", "B"} for sides in pairs.values()):
        raise ValueError("Every adapted DeepSeek layer requires both A and B")
    return tuple(sorted(pairs))


def register_deepseek_v41_lora() -> None:
    """Override the stock name in the owned process's public model registry.

    Register before LLM configuration and in each worker before model loading;
    changing HF architectures would disable vLLM's stock Engram configuration.
    The underlying NVIDIA class remains directly imported by our subclass.
    """
    import vllm
    from vllm import ModelRegistry

    if vllm.__version__ != "0.31.0":
        raise RuntimeError("DeepSeek LoRA extension requires exact vLLM 0.31.0")
    from .vllm_lora_triton import configure_lora_triton_source
    configure_lora_triton_source()
    ModelRegistry.register_model(
        LORA_ARCHITECTURE,
        "rollout_manager.backends.deepseek_v41_lora_model:DeepseekV41LoRAForCausalLM",
    )
