"""Stable table parameter ownership with interchangeable lookup execution."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time

import torch
from torch import nn

from .policy import EngramPolicy
from .routing import RowRouter, RowRequest
from .storage import CPURowStore, CachedRowStore, LookupResult, RowStore


@dataclass
class LookupTicket:
    request: RowRequest
    result: LookupResult | Future
    device: torch.device
    resolved: bool = False


class RowShardedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, mesh=None, sparse_gradients=False, *, policy=None):
        super().__init__()
        self.global_num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mesh = mesh
        self.sparse_gradients = sparse_gradients
        self.router = RowRouter(num_embeddings, mesh)
        for key in ("size", "rank", "group", "row_start", "row_stop"):
            setattr(self, key, getattr(self.router, key))
        self.weight = nn.Parameter(torch.empty(self.row_stop - self.row_start, embedding_dim))
        self._executor = None
        self._pending = []
        self.metrics = {"lookup_rows": 0, "h2d_bytes": 0, "d2h_bytes": 0, "resolve_wait_seconds": 0.0}
        self.metrics.update({key: 0 for key in ("cache_hits", "cache_misses", "cache_fill_rows", "cache_evictions",
                                              "cache_invalidated_rows", "cache_metadata_h2d_bytes")})
        self.configure(policy or EngramPolicy())

    def configure(self, policy):
        self.synchronize()
        if policy.device == "cpu" and not self.sparse_gradients:
            raise ValueError("CPU Engram requires sparse gradients and SGD")
        self.policy = policy
        self.store = CPURowStore(policy.staging_bytes, self.metrics, policy.cache_bytes) if policy.device == "cpu" else RowStore()
        if policy.device == "cuda" and policy.cache_bytes:
            if not self.sparse_gradients:
                raise ValueError("Engram cache requires sparse gradients")
            self.store = CachedRowStore(policy.cache_bytes, self.metrics)
        if policy.device == "cpu" and not self.weight.is_meta:
            self.weight.data = self.weight.data.cpu()

    def _apply(self, fn, recurse=True):
        self.clear_cache()
        if self.policy.device == "cpu" and not self.weight.is_meta:
            probe = fn(torch.empty(0, dtype=self.weight.dtype, device="cpu"))
            if probe.device.type != "cpu":
                raise ValueError("CPU Engram placement is explicit; move the backbone before configuring offload")
        return super()._apply(fn, recurse)

    def prefetch(self, indices):
        if self._pending:
            raise RuntimeError("resolve the pending Engram lookup before prefetching another")
        request = self.router.prepare(indices, self.policy.deduplicate)
        device = indices.device
        self.metrics["lookup_rows"] += request.local_ids.numel()
        ids_ready = None
        if self.policy.prefetch == "async" and request.local_ids.is_cuda:
            # The caller enqueues D2H after routing; only the worker waits for
            # completion. The ticket/autograd retain these IDs until consumption.
            local_ids = torch.empty_like(request.local_ids, device="cpu", pin_memory=True)
            local_ids.copy_(request.local_ids, non_blocking=True)
            ids_ready = torch.cuda.Event()
            ids_ready.record(torch.cuda.current_stream(device))
            request.local_ids = local_ids
        grad_enabled = torch.is_grad_enabled()
        # Capture thread-local autograd mode; communication always remains on the caller thread.
        def lookup():
            with torch.set_grad_enabled(grad_enabled):
                if ids_ready is not None:
                    ids_ready.synchronize()
                return self.store.lookup(self.weight, request.local_ids, sparse=self.sparse_gradients, device=device)
        if self.policy.prefetch == "async":
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engram")
            result = self._executor.submit(lookup)
        else:
            result = lookup()
        ticket = LookupTicket(request, result, device)
        self._pending.append(ticket)
        return ticket

    def resolve(self, ticket):
        if ticket.resolved:
            raise RuntimeError("Engram lookup ticket was already consumed")
        if not any(pending is ticket for pending in self._pending):
            raise ValueError("Engram lookup ticket belongs to a different table")
        started = time.perf_counter()
        try:
            result = ticket.result.result() if isinstance(ticket.result, Future) else ticket.result
            if result.event is not None:
                torch.cuda.current_stream(ticket.device).wait_event(result.event)
                result.rows.record_stream(torch.cuda.current_stream(ticket.device))
            return self.router.complete(result.rows, ticket.request)
        finally:
            self.metrics["resolve_wait_seconds"] += time.perf_counter() - started
            ticket.resolved = True
            self._pending = [item for item in self._pending if item is not ticket]

    def forward(self, indices):
        return self.resolve(self.prefetch(indices))

    def clear_cache(self):
        self.synchronize()
        cache = getattr(self.store, "cache", None)
        if cache is not None:
            cache.clear()

    def _load_from_state_dict(self, *args, **kwargs):
        self.clear_cache()
        return super()._load_from_state_dict(*args, **kwargs)

    def synchronize(self):
        for ticket in self._pending:
            result = ticket.result.result() if isinstance(ticket.result, Future) else ticket.result
            if result.event is not None:
                result.event.synchronize()
        if self._pending:
            raise RuntimeError("unconsumed Engram lookups at optimizer/checkpoint boundary")

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        for ticket in self._pending:
            ticket.resolved = True
        self._pending.clear()
        self.clear_cache()
