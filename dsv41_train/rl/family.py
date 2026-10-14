"""Select existing native model families for the shared RL runtime."""
from __future__ import annotations

import json
from pathlib import Path

from .policy import DSV41Config, build_dsv41_policy


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
                 engram_weights=None, engram_offload=True, trainable_scope="all"):
    if getattr(model_config, "model_type", None) == "qwen3_moe":
        if engram_weights:
            raise ValueError("Qwen policies do not have Engram tables")
        from .qwen_policy import build_qwen_policy

        return build_qwen_policy(model_dir, model_config, random_init=random_init,
                                 devices=devices, dtype=dtype, trainable_scope=trainable_scope)
    if trainable_scope != "all":
        raise ValueError("DeepSeek RL currently requires trainable_scope=all")
    return build_dsv41_policy(model_dir, model_config, random_init=random_init,
                              devices=devices, dtype=dtype, engram_weights=engram_weights,
                              engram_offload=engram_offload)
