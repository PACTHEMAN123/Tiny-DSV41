"""Device selection and metrics shared by the training entry points."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def choose_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def initialize_runtime(
    device_name: str = "auto",
    local_rank_arg: int | None = None,
    *,
    require_distributed: bool = False,
) -> Runtime:
    """Bind torchrun workers to local GPUs; allow plain CPU/GPU runs otherwise."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        raise ValueError("WORLD_SIZE must be positive")
    if world_size == 1 and not require_distributed:
        return Runtime(device=choose_device(device_name))
    if device_name not in {"auto", "cuda"}:
        raise ValueError("distributed training assigns devices automatically; omit --device")
    if not dist.is_available():
        raise RuntimeError("this PyTorch build does not provide torch.distributed")
    if not torch.cuda.is_available():
        raise RuntimeError("distributed training requires CUDA and the NCCL backend")
    local_rank_value = os.environ.get("LOCAL_RANK")
    if local_rank_value is None and local_rank_arg is None:
        raise RuntimeError("LOCAL_RANK is missing; launch with torchrun")
    local_rank = int(local_rank_value) if local_rank_value is not None else local_rank_arg
    assert local_rank is not None
    device_count = torch.cuda.device_count()
    if not 0 <= local_rank < device_count:
        raise RuntimeError(f"LOCAL_RANK {local_rank} is outside the {device_count} visible CUDA devices")

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        timeout_seconds = float(
            os.environ.get("DSV41_DISTRIBUTED_TIMEOUT_SECONDS", "600")
        )
        if timeout_seconds <= 0:
            raise ValueError("DSV41_DISTRIBUTED_TIMEOUT_SECONDS must be positive")
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=timeout_seconds),
        )
    return Runtime(
        device=torch.device("cuda", local_rank),
        rank=dist.get_rank(),
        local_rank=local_rank,
        world_size=dist.get_world_size(),
    )


def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def distributed_mean(value: torch.Tensor, runtime: Runtime) -> float:
    value = value.detach().float()
    if runtime.distributed:
        value = value.clone()
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= runtime.world_size
    return value.item()
