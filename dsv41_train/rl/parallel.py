"""Single-process layer model parallelism for each asynchronous policy."""

from contextlib import nullcontext
from dataclasses import fields, replace

import torch
from torch import nn

from dsv41_train.cp import ModelContext
from dsv41_train.models.dsv4.attention import ShadowIndexers
from dsv41_train.models.dsv4.mhc import HyperConnection


def layer_device_map(num_layers, devices):
    devices = tuple(torch.device(device) for device in devices)
    if not devices or len(devices) > num_layers:
        raise ValueError("use between one device and num_hidden_layers devices per policy")
    if any(device.type == "cpu" for device in devices):
        if devices != (torch.device("cpu"),):
            raise ValueError("CPU execution requires exactly one device")
    elif any(device.type != "cuda" or device.index is None for device in devices):
        raise ValueError("use cpu or explicit CUDA device indices")
    if len(set(devices)) != len(devices):
        raise ValueError("policy devices must not contain duplicates")
    return tuple(devices[index * len(devices) // num_layers] for index in range(num_layers))


def validate_available_devices(devices):
    for value in devices:
        device = torch.device(value)
        if device.type == "cuda" and device.index >= torch.cuda.device_count():
            raise ValueError(f"{device} is unavailable; CUDA exposes {torch.cuda.device_count()} devices")


class DeviceRotaryEmbedding(nn.Module):
    """One canonical FP32 frequency vector shared by layers on different GPUs."""

    def __init__(self, source):
        super().__init__()
        self.register_buffer("main", source.main.float().cpu(), persistent=False)
        self.register_buffer("compressed", source.compressed.float().cpu(), persistent=False)

    def _apply(self, fn, recurse=True):
        return self

    def forward(self, x, positions, *, compressed):
        frequency = self.compressed if compressed else self.main
        angles = positions.float().unsqueeze(-1) * frequency.to(x.device)
        return angles.cos().to(x.dtype), angles.sin().to(x.dtype)


def _move_context(context, device):
    return replace(context, **{
        field.name: value.to(device) for field in fields(context)
        if isinstance(value := getattr(context, field.name), torch.Tensor)
    })


def forward_logits(backend, input_ids, attention_mask=None):
    """Native decoder math with explicit transfers of residual and CSA2 state."""
    model = backend.model
    if input_ids.ndim != 2 or not 0 < input_ids.shape[1] <= model.config.max_position_embeddings:
        raise ValueError("input_ids must be [batch, sequence] within max_position_embeddings")
    input_ids = input_ids.to(model.embedding.weight.device)
    mask = torch.ones_like(input_ids, dtype=torch.bool) if attention_mask is None else attention_mask.to(
        device=input_ids.device, dtype=torch.bool)
    if mask.shape != input_ids.shape:
        raise ValueError("attention_mask must match input_ids")
    sequences = torch.zeros_like(input_ids).masked_fill(~mask, -1)
    context = ModelContext.build(model.context_parallel, input_ids, mask, sequences, model.config.sliding_window)
    contexts = {input_ids.device: context}
    hashes = model.hash(input_ids, mask, sequences) if model.hash is not None else None
    hash_indices = {layer_id: index for index, layer_id in enumerate(model.engram_layer_ids)}
    hidden = model.embedding(input_ids)
    streams = hidden.unsqueeze(2).expand(-1, -1, model.config.hc_mult, -1).contiguous()
    pre_mix = hidden.new_zeros(*input_ids.shape, model.config.hc_mult, dtype=torch.float32)
    pre_mix[..., 0] = 1
    shadow = ShadowIndexers()
    for layer in model.layers:
        device = next(layer.parameters()).device
        # Optional Triton kernels must launch on the device owning this layer.
        with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
            streams, pre_mix = streams.to(device), pre_mix.to(device)
            if device not in contexts:
                contexts[device] = _move_context(context, device)
            for field in fields(shadow):
                value = getattr(shadow, field.name)
                if value is not None:
                    setattr(shadow, field.name, value.to(device))
            rows = None
            if layer.engram is not None:
                rows = model.engram_tables[str(layer.layer_id)](hashes[:, :, hash_indices[layer.layer_id]])
            streams, pre_mix, _ = layer(streams, pre_mix, contexts[device], shadow, rows)
    with torch.cuda.device(streams.device) if streams.is_cuda else nullcontext():
        return backend.lm_head(model.norm(HyperConnection.collapse(streams, pre_mix)))
