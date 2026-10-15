"""Single-process layer model parallelism for each asynchronous policy."""

import torch
from torch import nn


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
