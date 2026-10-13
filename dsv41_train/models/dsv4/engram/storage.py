"""Local differentiable row lookup and bounded CPU-to-GPU staging."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .cache import RowCache


class _PinnedStager:
    def __init__(self, budget, metrics):
        self.budget = budget
        self.metrics = metrics
        self.buffers = [None, None]

    @property
    def allocated_bytes(self):
        return sum(buffer.numel() * buffer.element_size() for buffer in self.buffers if buffer is not None)

    def copy(self, rows, device):
        result = torch.empty(rows.shape, dtype=rows.dtype, device=device)
        if not rows.numel():
            return result
        width = rows.shape[-1]
        row_bytes = width * rows.element_size()
        if self.budget < 2 * row_bytes:
            raise ValueError("Engram staging budget must hold two complete rows")
        limit = self.budget // (2 * row_bytes)
        count = min(rows.shape[0], limit)
        capacity = min(1 << (count - 1).bit_length(), limit)
        events = [None, None]
        try:
            for number, start in enumerate(range(0, rows.shape[0], count)):
                slot = number % 2
                if events[slot] is not None:
                    events[slot].synchronize()
                buffer = self.buffers[slot]
                if buffer is None or buffer.dtype != rows.dtype or buffer.shape[1] != width or buffer.shape[0] < count:
                    buffer = self.buffers[slot] = torch.empty((capacity, width), dtype=rows.dtype, pin_memory=True)
                stop = min(start + count, rows.shape[0])
                buffer = buffer[:stop-start]
                buffer.copy_(rows[start:stop])
                result[start:stop].copy_(buffer, non_blocking=True)
                events[slot] = torch.cuda.Event()
                events[slot].record()
        finally:
            # Completion also makes reuse safe after a failed lookup.
            for event in events:
                if event is not None:
                    event.synchronize()
        self.metrics["h2d_bytes"] += rows.numel() * rows.element_size()
        return result


class _StagedCopy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, device, stager, metrics):
        ctx.source = rows.device
        ctx.metrics = metrics
        return stager.copy(rows, device)

    @staticmethod
    def backward(ctx, gradient):
        ctx.metrics["d2h_bytes"] += gradient.numel() * gradient.element_size()
        return gradient.to(ctx.source), None, None, None


@dataclass
class LookupResult:
    rows: torch.Tensor
    event: object = None


class RowStore:
    def lookup(self, weight, local_ids, *, sparse, device):
        return LookupResult(F.embedding(local_ids, weight, sparse=sparse))


class CachedRowStore(RowStore):
    """GPU-resident control using the same cache policy and coherence checks."""

    def __init__(self, budget, metrics):
        self.cache = RowCache(budget, metrics)

    def lookup(self, weight, local_ids, *, sparse, device):
        if not sparse:
            raise ValueError("Engram cache requires sparse gradients")
        return LookupResult(self.cache.lookup(weight, local_ids.cpu(), device,
                                             lambda rows, target: rows.to(target)))


class CPURowStore(RowStore):
    def __init__(self, staging_bytes, metrics, cache_bytes=0):
        self.staging_bytes = staging_bytes
        self.metrics = metrics
        self.stream = None
        self.stager = _PinnedStager(staging_bytes, metrics)
        self.cache = RowCache(cache_bytes, metrics) if cache_bytes else None

    def lookup(self, weight, local_ids, *, sparse, device):
        local_ids = local_ids.cpu()
        if device.type == "cpu":
            return LookupResult(F.embedding(local_ids, weight, sparse=sparse))
        if self.stream is None or self.stream.device != device:
            self.stream = torch.cuda.Stream(device=device)
        with torch.cuda.device(device), torch.cuda.stream(self.stream):
            if self.cache is not None:
                if not sparse:
                    raise ValueError("Engram cache requires sparse gradients")
                rows = self.cache.lookup(weight, local_ids, device, self.stager.copy)
            else:
                rows = F.embedding(local_ids, weight, sparse=sparse)
                rows = _StagedCopy.apply(rows, device, self.stager, self.metrics)
            event = torch.cuda.Event()
            event.record(self.stream)
            if self.cache is not None:
                event.synchronize()
        return LookupResult(rows, event)
