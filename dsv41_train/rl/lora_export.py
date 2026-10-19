"""Snapshot a live native query adapter into the existing vLLM tensor protocol.

This bridge performs no optimization and writes no checkpoint. The caller owns
the mounted-base provenance, completed optimizer step, and pause/drain barrier.
With FSDP, EVERY world rank must call it at the same boundary and in the same
order. It supports the CP1/world-sharded dense adapters used by the reviewed
native trainer; subgroup-only adapter meshes are rejected. Rank zero receives
the packet; other ranks participate in gathers and return None.
"""
from __future__ import annotations

import re

import torch
import torch.distributed as dist
from torch import nn

from rollout_manager.backends.deepseek_v41_lora import text_dimensions, validate_adapter_tensors
from rollout_manager.backends.vllm_lora_transport import pack_update
from .lora_config import DistributedLoRAConfig

_MODULE = re.compile(r"model\.layers\.(\d+)\.attention\.q_a\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _adapters(model, optimizer, lora_config):
    hidden, q_rank, _, layers = text_dimensions(model.config)
    expected_layers = set(lora_config.selected_layers(layers))
    found_layers = set()
    selected = {}
    for name, module in sorted(model.named_modules()):
        if not (hasattr(module, "lora_a") or hasattr(module, "lora_b")):
            continue
        match = _MODULE.fullmatch(name)
        config = getattr(module, "config", None)
        if (not match or int(match[1]) not in expected_layers or not isinstance(module, nn.Linear)
                or getattr(config, "rank", None) != lora_config.rank
                or getattr(config, "alpha", None) != lora_config.alpha
                or getattr(config, "dropout", None) != 0
                or tuple(getattr(config, "targets", ())) != ("attention.q_a",)
                or getattr(module, "scaling", None) != lora_config.alpha / lora_config.rank
                or not getattr(module, "adapters_enabled", False)
                or getattr(module, "merged", True)
                or getattr(getattr(module, "dropout", None), "p", None) != 0):
            raise ValueError("export requires unmerged native query-only adapters matching the configured profile")
        found_layers.add(int(match[1]))
        prefix = f"language_model.model.layers.{int(match[1])}.attn.wq_a"
        for source, target, shape in (("lora_a", "A", (lora_config.rank, hidden)),
                                      ("lora_b", "B", (q_rank, lora_config.rank))):
            value = getattr(module, source, None)
            if (not isinstance(value, nn.Parameter) or not value.requires_grad
                    or value.dtype != torch.bfloat16 or tuple(value.shape) != shape):
                raise ValueError("native query adapter parameter shape, dtype, or trainable flag differs")
            if hasattr(value, "device_mesh"):
                if not dist.is_initialized() or value.device_mesh.size() != dist.get_world_size():
                    raise ValueError("adapter DTensor mesh must span the whole export world")
            if value.grad is not None:
                gradient = value.grad.to_local() if hasattr(value.grad, "to_local") else value.grad
                if not torch.isfinite(gradient).all().item():
                    raise ValueError("adapter gradient is nonfinite")
            selected[f"{prefix}.lora_{target}.weight"] = value
    if found_layers != expected_layers:
        raise ValueError("native query adapter inventory does not cover every configured layer")
    if not selected or len({id(p) for p in selected.values()}) != len(selected):
        raise ValueError("export requires distinct nonempty native query adapters")
    ids = {id(p) for p in selected.values()}
    if any((p.requires_grad or p.grad is not None) for p in model.parameters() if id(p) not in ids):
        raise ValueError("all base parameters must remain frozen and have no gradients")
    if optimizer is not None:
        optimized = [p for group in optimizer.param_groups for p in group["params"]]
        if len(optimized) != len(ids) or {id(p) for p in optimized} != ids:
            raise ValueError("optimizer must contain every adapter exactly once and no base parameter")
        if any(id(p) not in ids for p in optimizer.state):
            raise ValueError("optimizer state includes a non-adapter parameter")
        for state in optimizer.state.values():
            for value in state.values():
                if isinstance(value, torch.Tensor):
                    value = value.to_local() if hasattr(value, "to_local") else value
                    if not torch.isfinite(value).all().item():
                        raise ValueError("optimizer state is nonfinite")
    return selected


@torch.no_grad()
def export_q_a_update(model, *, version, lineage, adapter_id, base_identity, optimizer=None,
                      lora_config=None):
    """Return the rank-zero (manifest, bytes) snapshot for direct activation.

    The live model must expose the reviewed native LoRALinear interface and its
    real configuration. This validates frozen/trainable scope and optional
    optimizer membership/state, NOT whether a genuine update happened. Initial
    zero-B export is allowed. Base identity must come from the actual mounted
    loader; metadata identity alone does not prove numerical base parity.
    """
    selected, error, profile = {}, None, None
    try:
        profile = getattr(model, "distributed_lora_config", DistributedLoRAConfig())
        if lora_config is not None:
            if hasattr(model, "distributed_lora_config") and lora_config != profile:
                raise ValueError("export configuration differs from the loaded learner profile")
            profile = lora_config
        if not isinstance(profile, DistributedLoRAConfig):
            raise ValueError("export LoRA configuration must be DistributedLoRAConfig")
        if (not isinstance(base_identity, dict)
                or set(base_identity) != {"config_sha256", "index_sha256"}
                or any(not isinstance(v, str) or not _SHA.fullmatch(v) for v in base_identity.values())):
            raise ValueError("export requires exact mounted config/index SHA256 identity")
        selected = _adapters(model, optimizer, profile)
        signature = [(name, list(value.shape), str(value.dtype)) for name, value in selected.items()]
    except Exception as exc:
        error = str(exc)
        signature = []
    if dist.is_initialized():
        # Agree before any tensor collective, so one rank's validation error or
        # different adapter inventory cannot strand peers inside full_tensor().
        statuses = [None] * dist.get_world_size()
        dist.all_gather_object(statuses, {"error": error, "signature": signature,
                                        "policy": [version, lineage, adapter_id, base_identity],
                                        "lora_config": profile.to_dict()
                                        if isinstance(profile, DistributedLoRAConfig) else None})
        if any(item["error"] for item in statuses):
            raise ValueError("a world rank rejected native adapter export validation")
        if any(item != statuses[0] for item in statuses):
            raise ValueError("world ranks disagree on adapter inventory or policy identity")
    elif error:
        raise ValueError(error)
    tensors = {}
    for name, value in selected.items():
        value = value.detach()
        if hasattr(value, "full_tensor"):
            value = value.full_tensor()
        tensors[name] = value.cpu().contiguous().clone()
    validate_adapter_tensors(tensors, model.config, lora_config=profile)
    packet = pack_update(tensors, version=version, lineage=lineage, adapter_id=adapter_id,
                         peft={"r": profile.rank, "lora_alpha": profile.alpha,
                               "target_modules": ["wq_a"]},
                         base_identity=base_identity)
    return packet if not dist.is_initialized() or dist.get_rank() == 0 else None
