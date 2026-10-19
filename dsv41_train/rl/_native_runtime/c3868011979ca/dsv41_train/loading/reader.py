"""Indexed safetensors reads without materializing an entire checkpoint."""

from __future__ import annotations

import json
import math
import mmap
import struct
from pathlib import Path

import torch


def _torch_dtype(name: str) -> torch.dtype:
    if name == "BF16":
        return torch.bfloat16
    if name == "F32":
        return torch.float32
    if name == "F16":
        return torch.float16
    if name == "I8":
        return torch.int8
    attribute = {
        "F8_E4M3": "float8_e4m3fn",
        "F8_E8M0": "float8_e8m0fnu",
    }.get(name)
    dtype = None if attribute is None else getattr(torch, attribute, None)
    if dtype is None:
        raise RuntimeError(f"this PyTorch build does not support safetensors dtype {name}")
    return dtype


class _SafeTensorShard:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("rb")
        header_size_bytes = self._file.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        self.header_size = struct.unpack("<Q", header_size_bytes)[0]
        self.header = json.loads(self._file.read(self.header_size))
        self.data_start = 8 + self.header_size
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_COPY)

    def tensor(self, name: str, device: torch.device | str) -> torch.Tensor:
        metadata = self.header[name]
        dtype = _torch_dtype(metadata["dtype"])
        shape = tuple(metadata["shape"])
        start, stop = metadata["data_offsets"]
        numel = math.prod(shape)
        expected_bytes = numel * torch.empty((), dtype=dtype).element_size()
        if stop - start != expected_bytes:
            raise ValueError(
                f"invalid byte range for {name}: expected {expected_bytes}, got {stop - start}"
            )
        source = torch.frombuffer(
            self._mapping,
            dtype=dtype,
            count=numel,
            offset=self.data_start + start,
        ).reshape(shape)
        target = torch.device(device)
        return source.clone() if target.type == "cpu" else source.to(target)

    def tensor_rows(
        self,
        name: str,
        row_start: int,
        row_stop: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        metadata = self.header[name]
        shape = tuple(metadata["shape"])
        if not shape:
            raise ValueError(f"cannot slice rows from scalar tensor {name}")
        if not 0 <= row_start <= row_stop <= shape[0]:
            raise IndexError(
                f"invalid row range for {name}: [{row_start}, {row_stop}) of {shape[0]}"
            )
        dtype = _torch_dtype(metadata["dtype"])
        start, stop = metadata["data_offsets"]
        row_numel = math.prod(shape[1:])
        element_size = torch.empty((), dtype=dtype).element_size()
        expected_bytes = math.prod(shape) * element_size
        if stop - start != expected_bytes:
            raise ValueError(
                f"invalid byte range for {name}: expected {expected_bytes}, got {stop - start}"
            )
        if row_start == row_stop:
            return torch.empty((0, *shape[1:]), dtype=dtype, device=device)
        source = torch.frombuffer(
            self._mapping,
            dtype=dtype,
            count=(row_stop - row_start) * row_numel,
            offset=self.data_start + start + row_start * row_numel * element_size,
        ).reshape(row_stop - row_start, *shape[1:])
        target = torch.device(device)
        return source.clone() if target.type == "cpu" else source.to(target)

    def close(self) -> None:
        self._mapping.close()
        self._file.close()


class ShardedSafeTensorReader:
    """Read selected tensors from an indexed safetensors checkpoint."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        with (self.folder / "model.safetensors.index.json").open(encoding="utf-8") as file:
            self.weight_map: dict[str, str] = json.load(file)["weight_map"]
        self._shards: dict[str, _SafeTensorShard] = {}
        self.bytes_read = 0

    def metadata(self, name):
        shard_name = self.weight_map[name]
        if shard_name not in self._shards:
            self._shards[shard_name] = _SafeTensorShard(self.folder / shard_name)
        return self._shards[shard_name].header[name]

    def tensor(self, name: str, device: torch.device | str) -> torch.Tensor:
        try:
            shard_name = self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"checkpoint tensor is missing: {name}") from error
        shard = self._shards.get(shard_name)
        if shard is None:
            shard = self._shards[shard_name] = _SafeTensorShard(self.folder / shard_name)
        value = shard.tensor(name, device)
        self.bytes_read += value.numel() * value.element_size()
        return value

    def tensor_rows(
        self,
        name: str,
        row_start: int,
        row_stop: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        try:
            shard_name = self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"checkpoint tensor is missing: {name}") from error
        shard = self._shards.get(shard_name)
        if shard is None:
            shard = self._shards[shard_name] = _SafeTensorShard(self.folder / shard_name)
        value = shard.tensor_rows(name, row_start, row_stop, device)
        self.bytes_read += value.numel() * value.element_size()
        return value

    def close(self) -> None:
        for shard in self._shards.values():
            shard.close()
        self._shards.clear()

    def __enter__(self) -> ShardedSafeTensorReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
