"""Select existing native model families for the shared RL runtime."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .policy import DSV41Config, build_dsv41_policy


def policy_lora_config(config, model_config):
    if config.trainable_scope != "lora":
        return None
    from dsv41_train.lora import LoRAConfig
    targets = config.lora_targets
    if targets is None:
        targets = (("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
                   if getattr(model_config, "model_type", None) == "qwen3_moe" else LoRAConfig().targets)
    return LoRAConfig(config.lora_rank, config.lora_alpha, config.lora_dropout, targets)


def lora_metadata(config, initial_adapter):
    from .checkpoint import file_sha256
    return json.loads(json.dumps({"config": asdict(config),
        "initial_adapter_sha256": file_sha256(initial_adapter) if initial_adapter is not None else None}))


def initial_sft_source(model_dir, model_config, config, random_init):
    if config.initial_sft_checkpoint is None:
        return None
    config.validate()
    if random_init or getattr(model_config, "model_type", None) == "qwen3_moe":
        raise ValueError("initial_sft_checkpoint requires a pretrained DSV4 base")
    from dsv41_train.sft_source import SFTCheckpointSource
    return SFTCheckpointSource(config.initial_sft_checkpoint, model_dir)


def load_model_config(model_dir):
    path = Path(model_dir) / "config.json"
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("model config must be an object")
    text = values.get("text_config")
    if text is not None and not isinstance(text, dict):
        raise ValueError("text_config must be an object")
    root_type, text_type = values.get("model_type"), (text or {}).get("model_type")
    if root_type == "qwen3_moe":
        if text is not None:
            raise ValueError("native Qwen RL requires a flat qwen3_moe config")
        if values.get("quantization_config"):
            raise ValueError("native Qwen RL requires unquantized mounted safetensors")
        from dsv41_train.models.qwen import Qwen3MoeConfig
        from .qwen_policy import validate_qwen_config

        config = Qwen3MoeConfig.from_json(path)
        validate_qwen_config(config)
        return config
    if text_type == "qwen3_moe":
        raise ValueError("native Qwen RL requires a flat qwen3_moe config")
    for name, model_type in (("model_type", root_type), ("text_config.model_type", text_type)):
        if model_type not in (None, "deepseek_v41", "deepseek_v41_text"):
            raise ValueError(f"unsupported {name}: {model_type!r}")
    return DSV41Config.from_model_dir(model_dir)


def build_policy(model_dir, model_config, *, random_init, devices, dtype,
                 engram_weights=None, engram_offload=True, trainable_scope="all",
                 lora_config=None, initial_adapter=None, train_engram=False, engram_cache_mib=0,
                 initial_engram_delta=None, role="learner", sft_source=None):
    if sft_source is not None and (random_init or trainable_scope != "all" or not engram_offload
            or initial_adapter is not None or initial_engram_delta is not None
            or getattr(model_config, "model_type", None) == "qwen3_moe"):
        raise ValueError("SFT source requires pretrained native DSV4/all with CPU Engram offload")
    if (trainable_scope == "lora") != (lora_config is not None):
        raise ValueError("LoRA scope requires an explicit adapter configuration")
    if initial_adapter is not None and lora_config is None:
        raise ValueError("initial_adapter requires LoRA")
    if lora_config is not None and lora_config.dropout != 0:
        raise ValueError("LoRA RL requires dropout=0")
    if getattr(model_config, "model_type", None) == "qwen3_moe":
        if engram_weights or train_engram or engram_cache_mib or initial_engram_delta is not None:
            raise ValueError("Qwen policies do not have Engram tables")
        from .qwen_policy import build_qwen_policy

        model = build_qwen_policy(model_dir, model_config, random_init=random_init,
                                 devices=devices, dtype=dtype, trainable_scope=trainable_scope,
                                 lora_config=lora_config, initial_adapter=initial_adapter)
    else:
        if trainable_scope not in {"all", "lora"}:
            raise ValueError("DeepSeek RL requires trainable_scope=all or lora")
        model = build_dsv41_policy(model_dir, model_config, random_init=random_init,
                                  devices=devices, dtype=dtype, engram_weights=engram_weights,
                                  engram_offload=engram_offload,
                                  lora_config=lora_config, initial_adapter=initial_adapter,
                                  train_engram=train_engram, engram_cache_mib=engram_cache_mib,
                                  initial_engram_delta=initial_engram_delta, role=role, sft_source=sft_source)
    if lora_config is not None:
        model.lora_metadata = lora_metadata(lora_config, initial_adapter)
    return model
