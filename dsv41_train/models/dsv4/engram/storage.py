"""Local differentiable row lookup and bounded CPU-to-GPU staging."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F


class _StagedCopy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows, device, budget, metrics):
        ctx.source = rows.device
        ctx.metrics = metrics
        result = torch.empty(rows.shape, dtype=rows.dtype, device=device)
        if not rows.numel():
            return result
        row_bytes = rows.shape[-1] * rows.element_size()
        if budget < 2 * row_bytes:
            raise ValueError("Engram staging budget must hold two complete rows")
        count = max(1, min(rows.shape[0], budget // (2 * row_bytes)))
        buffers = [torch.empty((count, rows.shape[-1]), dtype=rows.dtype, pin_memory=True) for _ in range(2)]
        events = [None, None]
        for number, start in enumerate(range(0, rows.shape[0], count)):
            slot = number % 2
            if events[slot] is not None:
                events[slot].synchronize()
            stop = min(start + count, rows.shape[0])
            buffer = buffers[slot][:stop-start]
            buffer.copy_(rows[start:stop])
            result[start:stop].copy_(buffer, non_blocking=True)
            events[slot] = torch.cuda.Event()
            events[slot].record()
        # The buffers must survive their DMA operations, including exception-free exit.
        for event in events:
            if event is not None:
                event.synchronize()
        return result

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


class CPURowStore(RowStore):
    def __init__(self, staging_bytes, metrics):
        self.staging_bytes = staging_bytes
        self.metrics = metrics
        self.stream = None

    def lookup(self, weight, local_ids, *, sparse, device):
        rows = F.embedding(local_ids.cpu(), weight, sparse=sparse)
        if device.type == "cpu":
            return LookupResult(rows)
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=device)
        with torch.cuda.device(device), torch.cuda.stream(self.stream):
            rows = _StagedCopy.apply(rows, device, self.staging_bytes, self.metrics)
            event = torch.cuda.Event()
            event.record(self.stream)
        return LookupResult(rows, event)
