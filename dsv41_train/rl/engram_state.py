"""Immutable CPU baselines, private role views, and portable cumulative rows.

Linux stores the baseline once in sealed memfd pages. The small tempfile
fallback is only for CPU development on platforms without memfd (64 MiB cap).
Neither path uses Tensor.share_memory_ or serializes the complete table.
"""

from __future__ import annotations

import fcntl
from copy import deepcopy
import mmap
import os
from multiprocessing.reduction import DupFd
from pathlib import Path
import tempfile

import torch
from torch import nn

from dsv41_train.engram_offload import CPUEmbeddingLookup
from dsv41_train.models.dsv4.checkpoint import ShardedSafeTensorReader, dequantize_fp8_rows


def validate_delta(payload, identity, shapes, dtype):
    """Validate every table before a caller mutates any live policy state."""
    if (not isinstance(payload, dict) or set(payload) != {"format", "base", "tables"}
            or type(payload["format"]) is not int or payload["format"] != 1 or payload["base"] != identity
            or not isinstance(payload["tables"], dict) or payload["tables"].keys() != shapes.keys()):
        raise ValueError("Engram delta format or baseline identity differs")
    for name, (rows, width) in shapes.items():
        record = payload["tables"][name]
        if not isinstance(record, dict) or set(record) != {"indices", "values"}:
            raise ValueError(f"invalid Engram row record: {name}")
        indices, values = record["indices"], record["values"]
        if (not isinstance(indices, torch.Tensor) or indices.device.type != "cpu"
                or indices.dtype != torch.int64 or indices.ndim != 1
                or not isinstance(values, torch.Tensor) or values.device.type != "cpu"
                or values.dtype != dtype or tuple(values.shape) != (indices.numel(), width)
                or (indices.numel() and (indices[0] < 0 or indices[-1] >= rows
                    or bool((indices[1:] <= indices[:-1]).any())))
                or not bool(torch.isfinite(values).all())):
            raise ValueError(f"invalid Engram delta indices, dtype, shape or values: {name}")


def baseline_identity(model_dir, config, dtype, random_init, seed):
    """Identity used by portable SFT initial-row payloads; does not allocate rows."""
    from .checkpoint import run_identity
    identity = run_identity(model_dir, Path(model_dir) / "config.json", random_init)
    identity.pop("data_sha256")
    return {"source": identity, "dtype": str(dtype), "seed": seed if random_init else None,
            "tables": {str(layer): [rows, config.engram_head_dim]
                       for layer, rows in zip(config.engram_layer_ids, config.engram_num_embeddings)}}


class EngramBaseline:
    """FD-owned sealed data; spawn serializes descriptors, never Tensor storage."""
    chunk_rows = 131072

    def __init__(self, identity, shapes, dtype, fds):
        self.identity, self.shapes, self.dtype, self._fds = identity, shapes, dtype, fds

    @property
    def byte_size(self):
        return sum(rows * width for rows, width in self.shapes.values()) * torch.empty((), dtype=self.dtype).element_size()

    def __getstate__(self):
        return self.identity, self.shapes, self.dtype, {name: DupFd(fd) for name, fd in self._fds.items()}

    def __setstate__(self, state):
        identity, shapes, dtype, descriptors = state
        self.__init__(identity, shapes, dtype, {name: descriptor.detach() for name, descriptor in descriptors.items()})

    def __del__(self):
        for fd in getattr(self, "_fds", {}).values():
            try:
                os.close(fd)
            except OSError:
                pass

    def view(self, name):
        rows, width = self.shapes[name]
        size = rows * width * torch.empty((), dtype=self.dtype).element_size()
        mapping = mmap.mmap(self._fds[name], size, access=mmap.ACCESS_COPY)
        if hasattr(mapping, "madvise") and hasattr(mmap, "MADV_NOHUGEPAGE"):
            mapping.madvise(mmap.MADV_NOHUGEPAGE)
        # The Tensor retains the mapping; private writes cannot alter the sealed baseline.
        return torch.frombuffer(mapping, dtype=self.dtype).reshape(rows, width)

    @classmethod
    def create(cls, model_dir, config, dtype, random_init, seed, initial_delta=None):
        from .checkpoint import file_sha256
        from .policy import _layout, _SafeTensorShard
        root = Path(model_dir)
        identity = baseline_identity(root, config, dtype, random_init, seed)
        shapes = {name: tuple(shape) for name, shape in identity["tables"].items()}
        if not shapes or dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("trainable Engram requires nonempty FP32 or BF16 CPU tables")
        payload = None
        if initial_delta is not None:
            payload = torch.load(initial_delta, map_location="cpu", weights_only=True, mmap=True)
            validate_delta(payload, identity, shapes, dtype)
            identity = dict(identity, initial_delta_sha256=file_sha256(initial_delta))
        total = sum(rows * width for rows, width in shapes.values()) * torch.empty((), dtype=dtype).element_size()
        if not hasattr(os, "memfd_create") and total > 64 * 2**20:
            raise RuntimeError("large Engram baselines require Linux memfd; tempfile CPU fallback is limited to 64 MiB")
        fds = {}
        result = cls(identity, shapes, dtype, fds)
        released = ShardedSafeTensorReader(root) if not random_init and config.checkpoint_format == "released" else None
        native = _layout(root) if not random_init and released is None else None
        try:
            for name, (rows, width) in shapes.items():
                size = rows * width * torch.empty((), dtype=dtype).element_size()
                fallback_path = None
                if hasattr(os, "memfd_create"):
                    fd = os.memfd_create(f"engram-{name}", os.MFD_ALLOW_SEALING | os.MFD_CLOEXEC)
                else:
                    fd, fallback_path = tempfile.mkstemp(prefix="engram-baseline-")
                fds[name] = fd
                os.ftruncate(fd, size)
                mapping = mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)
                destination = torch.frombuffer(mapping, dtype=dtype).reshape(rows, width)
                reader = state = None
                try:
                    if released is not None:
                        prefix = f"layers.{name}.engram.embed"
                        for suffix, shape, storage in (("weight", [rows, width], "F8_E4M3"),
                                                       ("scale", [rows, width // 32], "F8_E8M0")):
                            meta = released.metadata(f"{prefix}.{suffix}")
                            if width % 32 or meta["shape"] != shape or meta["dtype"] != storage:
                                raise ValueError(f"invalid released Engram layout for {prefix}.{suffix}")
                    elif native is not None:
                        keys, safetensors = native
                        key = f"model.engram_tables.{name}.weight"
                        from .policy import _checkpoint_file
                        path = _checkpoint_file(root, keys[key])
                        if safetensors:
                            reader = _SafeTensorShard(path)
                            if reader.header[key]["shape"] != [rows, width]:
                                raise ValueError(f"native Engram shape differs: {name}")
                        else:
                            state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
                            if tuple(state[key].shape) != (rows, width):
                                raise ValueError(f"native Engram shape differs: {name}")
                    generator = torch.Generator().manual_seed((seed + int(name)) % (2**63 - 1))
                    for start in range(0, rows, cls.chunk_rows):
                        stop = min(rows, start + cls.chunk_rows)
                        if random_init:
                            destination[start:stop].normal_(std=config.initializer_range, generator=generator)
                        elif released is not None:
                            values = released.tensor_rows(f"{prefix}.weight", start, stop, "cpu")
                            scales = released.tensor_rows(f"{prefix}.scale", start, stop, "cpu")
                            destination[start:stop].copy_(dequantize_fp8_rows(values, scales, dtype=dtype))
                        else:
                            values = reader.tensor_rows(key, start, stop, "cpu") if reader is not None else state[key][start:stop]
                            if values.dtype not in (torch.float32, torch.bfloat16):
                                raise ValueError(f"invalid native Engram dtype: {name}")
                            destination[start:stop].copy_(values)
                    if payload is not None:
                        record = payload["tables"][name]
                        destination.index_copy_(0, record["indices"], record["values"])
                finally:
                    del destination
                    mapping.close()
                    if reader is not None:
                        reader.close()
                    if fallback_path is not None:
                        try:
                            readonly = os.open(fallback_path, os.O_RDONLY)
                            os.close(fd)
                            fds[name] = readonly
                        finally:
                            os.unlink(fallback_path)
                if hasattr(os, "memfd_create"):
                    fcntl.fcntl(fd, fcntl.F_ADD_SEALS,
                                fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SEAL)
        finally:
            if released is not None:
                released.close()
        return result


class RoleEmbedding(nn.Module):
    """Use native lookup/cache with a CPU parameter and explicit layer placement."""
    sparse_gradients = engram_offload = True

    def __init__(self, weight, trainable):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=trainable)
        self.register_buffer("placement", torch.empty(0, dtype=weight.dtype), persistent=False)
        self._cpu_lookup = CPUEmbeddingLookup()

    def _apply(self, fn, recurse=True):
        target = fn(self.placement)
        if target.dtype != self.weight.dtype:
            raise ValueError("Engram baseline dtype is fixed; construct it in the requested training dtype")
        self.synchronize()
        self.placement = target
        self.clear_cache()
        return self

    def forward(self, indices):
        return self._cpu_lookup(self.weight, indices, device=self.placement.device)

    def configure_cache(self, budget):
        self._cpu_lookup.configure_cache(budget, self.placement.device)

    def synchronize(self):
        self._cpu_lookup.synchronize()

    def clear_cache(self):
        self._cpu_lookup.clear_cache()


class EngramState:
    def __init__(self, baseline, role, trainable):
        if role not in {"learner", "reference", "actor"}:
            raise ValueError("unknown Engram policy role")
        self.baseline, self.role = baseline, role
        self.tables = {name: RoleEmbedding(baseline.view(name), trainable and role == "learner")
                       for name in baseline.shapes}
        self.dirty = {name: torch.empty(0, dtype=torch.long) for name in self.tables}

    @property
    def byte_size(self):
        return self.baseline.byte_size

    def synchronize(self):
        for table in self.tables.values():
            table.synchronize()

    def mark_touched(self):
        if self.role != "learner":
            raise RuntimeError("only the learner may update Engram")
        for name, table in self.tables.items():
            if table.weight.grad is not None:
                if not table.weight.grad.is_sparse:
                    raise ValueError("Engram gradients must be sparse")
                self.dirty[name] = torch.cat((self.dirty[name], table.weight.grad._indices()[0])).unique(sorted=True)

    def state_dict(self):
        self.synchronize()
        return {"format": 1, "base": deepcopy(self.baseline.identity), "tables": {
            name: {"indices": ids.clone(), "values": self.tables[name].weight.detach().index_select(0, ids)}
            for name, ids in self.dirty.items()}}

    def validate(self, payload):
        if self.role == "reference":
            raise ValueError("the initial reference cannot load learner Engram state")
        validate_delta(payload, self.baseline.identity, self.baseline.shapes, self.baseline.dtype)

    @torch.no_grad()
    def load_state_dict(self, payload):
        self.validate(payload)
        self.synchronize()
        for name, table in self.tables.items():
            # Undo all previous private edits before installing a cumulative
            # snapshot, including when restoring an older version.
            original = self.baseline.view(name)
            for ids in self.dirty[name].split(self.baseline.chunk_rows):
                table.weight.index_copy_(0, ids, original.index_select(0, ids))
            record = payload["tables"][name]
            table.weight.index_copy_(0, record["indices"], record["values"])
            table.weight.grad = None
            self.dirty[name] = record["indices"].clone()
            table.clear_cache()
