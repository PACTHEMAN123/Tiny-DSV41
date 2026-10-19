"""Owned DeepSeek worker: direct tensor activation with GPU buffer readback."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .vllm_lora_ipc import configure_lora_ipc

configure_lora_ipc()

from .deepseek_v41_lora import (
    deployment_lora_config, lora_target_modules,
    register_deepseek_v41_lora, text_dimensions, validate_adapter_manifest,
    validate_adapter_tensors,
)
from dsv41_train.rl.lora_config import DistributedLoRAConfig
from .vllm_lora_transport import policy_ack, unpack_update


def base_identity(model_dir):
    root = Path(model_dir)
    return {key: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for key, name in (("config_sha256", "config.json"),
                              ("index_sha256", "model.safetensors.index.json"))}


def _no_checkpoint_loader(*args, **kwargs):
    raise RuntimeError("in-memory LoRA cache miss; checkpoint loading is forbidden")


def verify_readback(manager, tensors, adapter_id, text_config, lora_config=None):
    """Check both packed slices, padding and every untouched target on this rank."""
    import torch
    from vllm.lora.layers.column_parallel_linear import MergedColumnParallelLinearWithLoRA

    profile = lora_config or DistributedLoRAConfig()
    adapted = validate_adapter_tensors(tensors, text_config, lora_config=lora_config)
    hidden, q_rank, head_dim, _ = text_dimensions(text_config)
    slots = [i for i, value in enumerate(manager.lora_index_to_id) if value == adapter_id]
    if len(slots) != 1:
        raise RuntimeError("adapter has no unique active GPU slot")
    slot = slots[0]
    expected_names = set(lora_target_modules(text_config, profile))
    if set(manager.modules) != expected_names:
        raise RuntimeError("actual LoRA wrappers differ from the narrow DeepSeek profile")
    for name in sorted(expected_names):
        module = manager.modules[name]
        if (type(module) is not MergedColumnParallelLinearWithLoRA
                or module.tp_size != 1 or module.n_slices != 2
                or len(module.lora_a_stacked) != 2 or len(module.lora_b_stacked) != 2):
            raise RuntimeError("actual DeepSeek LoRA wrapper layout changed")
        source = name.rsplit(".", 1)[0] + ".wq_a"
        a = tensors.get(source + ".lora_A.weight")
        b = tensors.get(source + ".lora_B.weight")
        for part, output in enumerate((q_rank, head_dim)):
            for side, buffers, shape in (("A", module.lora_a_stacked, (profile.rank, hidden)),
                                          ("B", module.lora_b_stacked, (output, profile.rank))):
                actual = buffers[part][slot, 0].detach().cpu().contiguous()
                if tuple(actual.shape) != shape or actual.dtype != torch.bfloat16:
                    raise RuntimeError("actual LoRA GPU buffer shape/dtype changed")
                wanted = a if side == "A" else b
                if wanted is not None and side == "B":
                    # vLLM folds alpha/rank into BF16 B in LoRALayerWeights.optimize.
                    wanted = wanted * (profile.alpha / profile.rank)
                if part == 1 or wanted is None:
                    if torch.count_nonzero(actual).item() != 0:
                        raise RuntimeError("untouched DeepSeek LoRA buffer is nonzero")
                elif not torch.equal(actual, wanted):
                    raise RuntimeError("actual LoRA GPU buffer differs from received tensor")
    return {"adapted_modules": list(adapted), "target_count": len(expected_names), "slot": slot}



class DeepseekLoRAExtension:
    """Named vLLM RPC; all state belongs to this actor's existing workers."""

    def _lora_state(self):
        from .vllm_worker import process_identity
        import torch
        runner = self.model_runner
        worker_manager = runner.lora_manager
        manager = worker_manager._adapter_manager
        # A request may only hit an already loaded immutable adapter. Never
        # interpret the request's memory:// label as a checkpoint path.
        worker_manager._load_adapter = _no_checkpoint_loader
        device = manager.device
        props = torch.cuda.get_device_properties(device)
        return worker_manager, manager, {**process_identity(), "device": str(device),
            "device_uuid": str(props.uuid), "device_name": props.name,
            "base_identity": base_identity(self.vllm_config.model_config.model),
            "lora_config": deployment_lora_config(self.vllm_config).to_dict(),
            "checkpoint_loader_disabled": True}

    def rollout_manager_lora_call(self, operation, manifest=None, payload=None):
        if getattr(self, "_direct_lora_failed", False):
            raise RuntimeError("LoRA worker failed; restart the owned actor before reuse")
        worker_manager, manager, receipt = self._lora_state()
        if operation == "stats":
            receipt["policy_ack"] = getattr(self, "_direct_lora_ack", None)
            previous = getattr(self, "_direct_lora_manifest", None)
            if previous is not None:
                receipt.update(adapter_id=previous["adapter_id"], readback_digest=previous["digest"])
            return receipt
        if operation != "load_lora":
            raise ValueError("unsupported direct LoRA worker operation")
        # Validate all input before removing the current policy.
        tensors = unpack_update(manifest, payload)
        if manifest["base_identity"] != receipt["base_identity"]:
            raise ValueError("LoRA base model identity differs from mounted worker assets")
        profile = deployment_lora_config(self.vllm_config)
        validate_adapter_manifest(manifest, self.vllm_config.model_config.hf_config, profile)
        validate_adapter_tensors(tensors, self.vllm_config.model_config.hf_config,
                                 lora_config=profile)
        previous = getattr(self, "_direct_lora_manifest", None)
        if previous is not None and (manifest["version"] != previous["version"] + 1
                or manifest["adapter_id"] <= previous["adapter_id"]
                or manifest["lineage"] != previous["lineage"]):
            raise ValueError("LoRA policy version, adapter id or lineage is stale")
        from vllm.lora.lora_model import LoRAModel
        from vllm.lora.peft_helper import PEFTHelper
        helper = PEFTHelper(**manifest["peft"])
        helper.validate_legal(worker_manager.lora_config)
        adapter = LoRAModel.from_lora_tensors(manifest["adapter_id"],
            {name: tensor.clone() for name, tensor in tensors.items()}, helper,
            device="cpu", dtype=worker_manager.lora_config.lora_dtype,
            weights_mapper=None)
        try:
            worker_manager.remove_all_adapters()
            if not manager.add_adapter(adapter):
                raise RuntimeError("vLLM did not insert the received in-memory adapter")
            manager.activate_adapter(adapter.id)
            manager.pin_adapter(adapter.id)
            verify_readback(manager, tensors, adapter.id,
                            self.vllm_config.model_config.hf_config, profile)
        except BaseException:
            self._direct_lora_failed = True
            raise
        self._direct_lora_manifest = manifest
        self._direct_lora_ack = policy_ack(manifest)
        return {**self._lora_state()[2], "policy_ack": self._direct_lora_ack,
                "adapter_id": adapter.id, "readback_digest": manifest["digest"]}


# Worker extensions are imported before the model is constructed. Re-register
# the lazy external entry in each spawned process as well as the parent actor.
register_deepseek_v41_lora()
