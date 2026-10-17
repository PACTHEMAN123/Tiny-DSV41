"""Rank-local Engram row snapshots over the original, verified SFT base.

Only cumulative SGD-touched rows are saved. Rollback rereads omitted dirty rows
from the mounted base; no second full table or collective row gather is needed.
The caller validates base identity/topology before loading and forbids concurrent
training. A base I/O failure aborts restoration; it is not a transactional undo.
"""

from copy import deepcopy
import re

import torch
from torch.distributed.tensor import DTensor

from .models.dsv4.checkpoint import ShardedSafeTensorReader, dequantize_fp8_rows


def _tables(model):
    return {f"{name}.weight" if name else "weight": module for name, module in model.named_modules()
            if getattr(module, "engram_offload", False) and getattr(module, "sparse_gradients", False)
            and module.weight.requires_grad}


def _chunk_rows(width):
    # Bound decoded output plus FP8/scale/FP32 dequantization temporaries to
    # roughly 4 MiB, or one row for unusually wide embeddings.
    return max(1, (4 * 2**20) // (24 * width))


class ReleasedEngramRows:
    """Read requested global rows in original V4.1 E4M3/E8M0 format."""

    def __init__(self, model_path, schema):
        self.model_path, self.schema = model_path, deepcopy(schema)
        self.sources = {}
        for name in schema:
            match = re.search(r"(?:^|\.)engram_tables\.(\d+)\.weight$", name)
            if match is None:
                raise ValueError(f"cannot identify released Engram layer: {name}")
            self.sources[name] = f"layers.{match[1]}.engram.embed"

    def __call__(self, name, global_ids):
        layout = self.schema[name]
        rows, width = layout["global_shape"]
        dtype = getattr(torch, layout["dtype"].removeprefix("torch."))
        if global_ids.numel() > _chunk_rows(width):
            raise ValueError("Engram base read exceeds the bounded row chunk")
        output = torch.empty((global_ids.numel(), width), dtype=dtype)
        if not global_ids.numel():
            return output
        source = self.sources[name]
        with ShardedSafeTensorReader(self.model_path) as reader:
            for suffix, shape, storage in (("weight", [rows, width], "F8_E4M3"),
                                           ("scale", [rows, width // 32], "F8_E8M0")):
                metadata = reader.metadata(f"{source}.{suffix}")
                if width % 32 or metadata["shape"] != shape or metadata["dtype"] != storage:
                    raise ValueError(f"invalid released Engram layout for {source}.{suffix}")
            # Merge only adjacent requested rows. Gaps are never loaded merely
            # because the requested IDs span a large part of the table.
            boundaries = [0, *((global_ids[1:] != global_ids[:-1] + 1).nonzero().flatten() + 1).tolist(), global_ids.numel()]
            for begin, end in zip(boundaries, boundaries[1:]):
                start, stop = int(global_ids[begin]), int(global_ids[end - 1]) + 1
                values = reader.tensor_rows(f"{source}.weight", start, stop, "cpu")
                scales = reader.tensor_rows(f"{source}.scale", start, stop, "cpu")
                output[begin:end].copy_(dequantize_fp8_rows(values, scales, dtype=dtype))
        return output


class EngramCheckpoint:
    def __init__(self, model, optimizer, read_base_rows):
        self.tables = _tables(model)
        self.parameter_names = frozenset(self.tables)
        self.read_base_rows = read_base_rows
        self._schema = {}
        for name, table in self.tables.items():
            weight = table.weight
            if (isinstance(weight, DTensor) or weight.device.type != "cpu"
                    or weight.dtype not in (torch.float32, torch.bfloat16)
                    or weight.layout != torch.strided or weight.ndim != 2
                    or not 0 <= table.row_start <= table.row_stop <= table.global_num_embeddings
                    or tuple(weight.shape) != (table.row_stop - table.row_start, table.embedding_dim)):
                raise ValueError(f"invalid CPU Engram owner layout: {name}")
            self._schema[name] = {"global_shape": [table.global_num_embeddings, table.embedding_dim],
                "local_shape": list(weight.shape), "row_start": table.row_start, "row_stop": table.row_stop,
                "dtype": str(weight.dtype)}
        self.dirty = {name: torch.empty(0, dtype=torch.long) for name in self.tables}
        if type(optimizer) is not torch.optim.SGD:
            raise ValueError("sparse Engram checkpoints require native SGD")
        groups = {id(parameter): group for group in optimizer.param_groups for parameter in group["params"]}
        for table in self.tables.values():
            group = groups.get(id(table.weight))
            if group is None or any(group.get(key) for key in (
                    "momentum", "weight_decay", "maximize", "differentiable", "foreach", "fused")):
                raise ValueError("Engram checkpoint tracking requires owned plain sparse SGD parameters")
        # Install after the standard Engram SGD hooks. CPU parameters are not
        # updated by the DTensor in-backward optimizer, so their grads remain.
        self.handle = optimizer.register_step_post_hook(self._after_step)

    @property
    def schema(self):
        return deepcopy(self._schema)

    def remove(self):
        self.handle.remove()

    def _after_step(self, _optimizer, _args, _kwargs):
        for name, table in self.tables.items():
            gradient = table.weight.grad
            if gradient is not None:
                if not gradient.is_sparse:
                    raise ValueError("Engram checkpoint tracking requires sparse gradients")
                self.dirty[name] = torch.cat((self.dirty[name], gradient._indices()[0])).unique(sorted=True)

    def _drain(self):
        for table in self.tables.values():
            table.synchronize()

    def state_dict(self):
        self._drain()
        return {"format": 1, "schema": self.schema, "tables": {
            name: {"indices": ids.clone(), "values": self.tables[name].weight.detach().index_select(0, ids)}
            for name, ids in self.dirty.items()}}

    def validate(self, payload):
        if (not isinstance(payload, dict) or set(payload) != {"format", "schema", "tables"}
                or type(payload["format"]) is not int or payload["format"] != 1
                or payload["schema"] != self._schema or not isinstance(payload["tables"], dict)
                or payload["tables"].keys() != self.tables.keys()):
            raise ValueError("Engram row snapshot schema or parameter names differ")
        for name, table in self.tables.items():
            record = payload["tables"][name]
            if not isinstance(record, dict) or set(record) != {"indices", "values"}:
                raise ValueError(f"invalid Engram row record: {name}")
            indices, values = record["indices"], record["values"]
            if (not isinstance(indices, torch.Tensor) or isinstance(indices, DTensor)
                    or indices.device.type != "cpu" or indices.layout != torch.strided
                    or indices.dtype != torch.int64 or indices.ndim != 1
                    or not isinstance(values, torch.Tensor) or isinstance(values, DTensor)
                    or values.device.type != "cpu" or values.layout != torch.strided
                    or values.dtype != table.weight.dtype or tuple(values.shape) != (indices.numel(), table.embedding_dim)
                    or (indices.numel() and (indices[0] < 0 or indices[-1] >= table.weight.shape[0]
                        or bool((indices[1:] <= indices[:-1]).any())))
                    or not bool(torch.isfinite(values).all())):
                raise ValueError(f"invalid Engram row indices, shape, dtype or values: {name}")

    @torch.no_grad()
    def load_state_dict(self, payload):
        self.validate(payload)  # All records must pass before any live mutation.
        self._drain()
        for name, table in self.tables.items():
            record, old = payload["tables"][name], self.dirty[name]
            incoming = record["indices"]
            positions = torch.searchsorted(incoming, old)
            keep = positions < incoming.numel()
            if incoming.numel():
                keep &= incoming[positions.clamp_max(incoming.numel() - 1)] == old
            for local_ids in old[~keep].split(_chunk_rows(table.embedding_dim)):
                if not local_ids.numel():
                    continue
                values = self.read_base_rows(name, local_ids + table.row_start)
                if (not isinstance(values, torch.Tensor) or values.device.type != "cpu"
                        or values.layout != torch.strided or values.dtype != table.weight.dtype
                        or tuple(values.shape) != (local_ids.numel(), table.embedding_dim)
                        or not bool(torch.isfinite(values).all())):
                    raise ValueError(f"invalid original Engram rows returned for {name}")
                table.weight.index_copy_(0, local_ids, values)
            table.weight.index_copy_(0, incoming, record["values"])
            table.weight.grad = None
            self.dirty[name] = incoming.clone()
            table.clear_cache()


def build_engram_checkpoint(model, model_path, optimizer):
    """Construct after loading/wrapping the base and installing SGD hooks."""
    if not _tables(model):
        return None
    manager = EngramCheckpoint(model, optimizer, None)
    try:
        manager.read_base_rows = ReleasedEngramRows(model_path, manager.schema)
    except Exception:
        manager.remove()
        raise
    return manager
