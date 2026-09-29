"""FSDP-specific materialization, outside the checkpoint reader and codecs."""

import torch

from .parallel import apply_fsdp2_layer, apply_fsdp2_root, shard_embedding_and_head
from .policies import FSDPPolicy
from ...layout import parameter_layouts


class FSDPMaterializer:
    def __init__(self, meshes, policy=None):
        self.meshes = meshes
        self.policy = policy or FSDPPolicy()
        self.direct = False

    def prepare(self, model, *, dtype, device, direct):
        self.direct = direct
        if not direct:
            return
        model.to(dtype=dtype)
        for layer in model.model.layers:
            for module in (layer.attention_hc, layer.moe_hc, layer.attention.sinks):
                module.float()
            apply_fsdp2_layer(layer, self.meshes, policy=self.policy)
        apply_fsdp2_root(model, self.meshes, reshard_after_forward=self.policy.reshard_after_forward)
        layouts = parameter_layouts(model)
        for name, module in model.named_modules():
            own = list(module.named_parameters(recurse=False))
            if not own:
                continue
            layout = layouts[(name + "." if name else "") + own[0][0]]
            target = device
            if layout.role == "engram":
                if module.policy.device == "cpu":
                    target = "cpu"
            elif layout.logical_name.startswith("model.layers."):
                layer_id = int(layout.logical_name.split(".")[2])
                if self.policy.offloads(layer_id, layout.role):
                    target = "cpu"
            module.to_empty(device=target, recurse=False)
        # FSDP offloads parameters, not buffers used by GPU computation.
        for module in model.modules():
            for name, buffer in module.named_buffers(recurse=False):
                module._buffers[name] = torch.empty_like(buffer, device=device)

    def roots_loaded(self, model):
        if not self.direct:
            shard_embedding_and_head(model, self.meshes, reshard_after_forward=self.policy.reshard_after_forward)

    def layer_loaded(self, layer):
        if not self.direct:
            apply_fsdp2_layer(layer, self.meshes, policy=self.policy)

    def finish(self, model):
        if not self.direct:
            apply_fsdp2_root(model, self.meshes, reshard_after_forward=self.policy.reshard_after_forward)
