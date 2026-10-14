"""Benchmark-only exact update evidence over all unique sparse touched rows.

The snapshot is proportional to the sparse working set, never the full table.
All columns are checked in bounded chunks; signed-zero-only changes do not count
as a numerical optimizer update. This observes SGD, rather than predicting its
rounding behavior. Call after gradient synchronization and immediately around
optimizer.step(), and include the cost in reported validation time.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from dsv41_train.layout import local_tensor


@dataclass
class TouchedRows:
    row_ids: torch.Tensor
    before: torch.Tensor
    raw_nnz: int
    weight_pointer: int
    weight_shape: tuple


def _device(runtime):
    return torch.device("cpu") if runtime is None else runtime.device


def _reduce(value, operation, runtime):
    if dist.is_initialized():
        dist.all_reduce(value, op=operation)
    elif runtime is not None and runtime.world_size > 1:
        raise RuntimeError("distributed touched-row validation requires an initialized process group")
    return value


def _require_all(condition, runtime, message):
    flag = torch.tensor(int(condition), dtype=torch.int32, device=_device(runtime))
    if not _reduce(flag, dist.ReduceOp.MIN, runtime).item():
        raise RuntimeError(message)


def _validate_chunk(chunk_rows):
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("chunk_rows must be a positive integer")


@torch.no_grad()
def capture_touched_rows(tables, runtime=None, *, chunk_rows=8192):
    """Capture every unique local COO-gradient row in its original weight dtype."""
    _validate_chunk(chunk_rows)
    valid = bool(tables)
    for table in tables.values():
        weight = local_tensor(table.weight)
        gradient = None if table.weight.grad is None else local_tensor(table.weight.grad)
        valid &= weight.ndim == 2 and weight.is_floating_point()
        valid &= gradient is None or (gradient.is_sparse and gradient.sparse_dim() == 1
                                     and gradient.dense_dim() == 1 and gradient.shape == weight.shape)
    _require_all(valid, runtime, "full touched-row validation requires floating tables and row-sparse COO gradients")
    snapshots = {}
    for name, table in tables.items():
        weight = local_tensor(table.weight)
        gradient = None if table.weight.grad is None else local_tensor(table.weight.grad)
        rows = (torch.empty(0, dtype=torch.int64, device=weight.device) if gradient is None
                else gradient._indices()[0].unique(sorted=True))
        in_bounds = not rows.numel() or bool((rows >= 0).all() and (rows < weight.shape[0]).all())
        _require_all(in_bounds, runtime, f"table {name}: touched row ID outside parameter")
        before = weight.new_empty((rows.numel(), weight.shape[1]))
        finite = torch.ones((), dtype=torch.bool, device=weight.device)
        for start in range(0, rows.numel(), chunk_rows):
            stop = min(start + chunk_rows, rows.numel())
            before[start:stop].copy_(weight.index_select(0, rows[start:stop]))
            finite &= torch.isfinite(before[start:stop]).all()
        _require_all(bool(finite), runtime, f"table {name}: non-finite touched weight before optimizer step")
        snapshots[name] = TouchedRows(rows, before, 0 if gradient is None else gradient._nnz(),
                                      weight.data_ptr(), tuple(weight.shape))
    return snapshots


@torch.no_grad()
def verify_touched_updates(tables, snapshots, runtime=None, *, chunk_rows=8192):
    """Require a finite, actual numerical update for every table globally.

Return local and all-rank SUM/MAX evidence. Empty local gradients are permitted;
a table with no global touched-row numerical change cannot pass.
"""
    _validate_chunk(chunk_rows)
    _require_all(set(tables) == set(snapshots), runtime, "snapshot and table names differ")
    records = {}
    for name, table in tables.items():
        weight = local_tensor(table.weight)
        snapshot = snapshots[name]
        valid = (weight.data_ptr() == snapshot.weight_pointer and tuple(weight.shape) == snapshot.weight_shape
                 and weight.dtype == snapshot.before.dtype and weight.device == snapshot.before.device)
        _require_all(valid, runtime, f"table {name}: parameter storage changed across optimizer step")
        rows, before = snapshot.row_ids, snapshot.before
        maximum = torch.zeros((), dtype=torch.float32, device=weight.device)
        changed = torch.zeros((), dtype=torch.int64, device=weight.device)
        finite = torch.ones((), dtype=torch.bool, device=weight.device)
        for start in range(0, rows.numel(), chunk_rows):
            stop = min(start + chunk_rows, rows.numel())
            after = weight.index_select(0, rows[start:stop])
            old = before[start:stop]
            finite &= torch.isfinite(after).all()
            changed += after.ne(old).sum()
            # Only two chunk-sized FP32 conversions, no touched-set FP32 clone.
            delta = after.float()
            delta.sub_(old.float()).abs_()
            finite &= torch.isfinite(delta).all()
            maximum = torch.maximum(maximum, delta.max())
        _require_all(bool(finite), runtime, f"table {name}: non-finite touched weight/update after optimizer step")
        local_max, local_changed = maximum.item(), changed.item()
        value_bytes = before.untyped_storage().nbytes()
        id_bytes = rows.untyped_storage().nbytes()
        counts = torch.tensor([rows.numel(), before.numel(), local_changed, value_bytes, id_bytes, snapshot.raw_nnz],
                              dtype=torch.int64, device=_device(runtime))
        _reduce(counts, dist.ReduceOp.SUM, runtime)
        global_max = maximum.to(_device(runtime))
        _reduce(global_max, dist.ReduceOp.MAX, runtime)
        total_rows, total_elements, total_changed, total_value_bytes, total_id_bytes, total_nnz = counts.tolist()
        maximum_value = global_max.item()
        if total_changed <= 0 or maximum_value <= 0:
            raise RuntimeError(f"table {name}: no actual numerical update in ALL {total_rows} unique touched rows "
                               f"({total_elements} elements); bounded-sample ambiguity eliminated")
        records[name] = {
            "scope": "all_unique_sparse_touched_rows_all_columns", "all_unique_rows_checked": True,
            "finite_before_after": True, "local_rows_checked": rows.numel(), "global_rows_checked": total_rows,
            "global_elements_checked": total_elements, "local_changed_elements": local_changed,
            "global_changed_elements": total_changed, "local_max_abs_delta": local_max,
            "global_max_abs_delta": maximum_value, "before_snapshot_bytes_local": value_bytes,
            "before_snapshot_bytes_global": total_value_bytes, "row_ids_bytes_local": id_bytes,
            "row_ids_bytes_global": total_id_bytes, "snapshot_allocated_bytes_local": value_bytes + id_bytes,
            "snapshot_allocated_bytes_global": total_value_bytes + total_id_bytes,
            "snapshot_dtype": str(before.dtype), "chunk_rows": chunk_rows, "global_raw_gradient_nnz": total_nnz,
        }
    return records
