"""Narrow, in-memory LoRA contract for the installed NVIDIA DeepSeek V4.1.

Only attention wq_a is adapted. vLLM packs it into the first, unsharded
fused_wqa_wkv output slice; the KV slice retains a zero adapter. These helpers
do not import the model implementation or initialize CUDA.
"""
from __future__ import annotations

from collections.abc import Mapping
import re

import torch

from dsv41_train.rl.lora_config import DistributedLoRAConfig, SUPPORTED_LORA_RANKS

# Keep the mounted architecture: vLLM uses this name to construct and validate
# EngramConfig before model loading. Only this process's class registry changes.
LORA_ARCHITECTURE = "DeepseekV41ForCausalLM"
LORA_TARGET_MODULE = "fused_wqa_wkv"
LORA_RANK = 8
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


def lora_target_modules(config, lora_config=DistributedLoRAConfig()) -> list[str]:
    """Use whole runtime paths: packed-child suffix matching also admits Engram.wkv."""
    layers = text_dimensions(config)[3]
    return [f"language_model.model.layers.{layer}.attn.{LORA_TARGET_MODULE}"
            for layer in lora_config.selected_layers(layers)]


def deployment_lora_config(vllm_config) -> DistributedLoRAConfig:
    """Read the immutable adapter scope passed from actor to every worker."""
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, Mapping) or "rollout_manager_lora" not in additional:
        raise ValueError("DeepSeek LoRA requires its immutable rollout configuration")
    return DistributedLoRAConfig.from_dict(additional["rollout_manager_lora"])


def peft_config(lora_config: DistributedLoRAConfig) -> dict:
    return {"r": lora_config.rank, "lora_alpha": lora_config.alpha,
            "target_modules": ["wq_a"]}


def validate_deployment(vllm_config) -> None:
    """Reject unsupported scopes before constructing the base model."""
    config = vllm_config.lora_config
    profile = deployment_lora_config(vllm_config)
    targets = lora_target_modules(vllm_config.model_config.hf_config, profile)
    if config is None or config.target_modules != targets:
        raise ValueError("DeepSeek LoRA requires exact backbone attention target_modules")
    if config.max_lora_rank != profile.rank or config.lora_dtype != torch.bfloat16:
        raise ValueError("DeepSeek LoRA requires the configured rank and BF16 adapters")
    if config.fully_sharded_loras or config.enable_tower_connector_lora:
        raise ValueError("DeepSeek LoRA supports unsharded text adapters only")
    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        raise ValueError("DeepSeek LoRA requires pipeline_parallel_size=1")
    if vllm_config.speculative_config is not None:
        raise ValueError("DeepSeek LoRA does not adapt speculative draft models")
    text_dimensions(vllm_config.model_config.hf_config)


def validate_adapter_manifest(manifest, text_config, lora_config) -> None:
    """Require every configured query pair before dispatching a tensor payload."""
    if manifest["peft"] != peft_config(lora_config):
        raise ValueError("DeepSeek LoRA PEFT configuration differs from the rollout configuration")
    hidden, q_rank, _, layers = text_dimensions(text_config)
    expected = {
        f"language_model.model.layers.{layer}.attn.wq_a.lora_{side}.weight": shape
        for layer in lora_config.selected_layers(layers)
        for side, shape in (("A", [lora_config.rank, hidden]), ("B", [q_rank, lora_config.rank]))
    }
    actual = {item["name"]: item for item in manifest["tensors"]}
    if set(actual) != set(expected) or any(
        actual[name]["shape"] != shape or actual[name]["dtype"] != "bfloat16"
        for name, shape in expected.items()
    ):
        raise ValueError("DeepSeek LoRA payload differs from the configured query layers and rank")


def validate_adapter_tensors(
    tensors, text_config, rank=LORA_RANK, *, lora_config=None,
) -> tuple[str, ...]:
    """Validate CPU BF16 pairs and return exact adapter-visible module names.

    Tensor keys intentionally use vLLM's runtime namespace and no PEFT prefix;
    callers must pass weights_mapper=None to LoRAModel.from_lora_tensors.
    With ``lora_config``, every configured layer is required exactly. Without
    it this lower-level helper accepts any complete query-only subset. No KV,
    MTP, vision, expert or malformed tensor is silently ignored.
    """
    rank = lora_config.rank if lora_config is not None else rank
    if type(rank) is not int or rank not in SUPPORTED_LORA_RANKS or not tensors:
        raise ValueError("DeepSeek LoRA requires a nonempty adapter with a supported rank")
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
        if lora_config is not None and match[2] == "B" and not torch.isfinite(
            tensor * (lora_config.alpha / rank)
        ).all().item():
            raise ValueError(f"{name} overflows BF16 after LoRA alpha/rank scaling")
        pairs.setdefault(name.rsplit(".lora_", 1)[0], set()).add(match[2])
    if any(sides != {"A", "B"} for sides in pairs.values()):
        raise ValueError("Every adapted DeepSeek layer requires both A and B")
    if lora_config is not None:
        expected = {f"language_model.model.layers.{layer}.attn.wq_a"
                    for layer in lora_config.selected_layers(layers)}
        if set(pairs) != expected:
            raise ValueError("DeepSeek LoRA tensors differ from the configured query layers")
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
