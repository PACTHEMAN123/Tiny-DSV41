"""Stable table parameter ownership with interchangeable lookup execution."""

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time
import weakref

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
        self._sparse_accumulation_parameter = None
        self._sparse_accumulation_handle = None
        self._sparse_accumulation_state = None
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
        self._ensure_sparse_accumulation_hook()
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

    def _ensure_sparse_accumulation_hook(self):
        """Preserve raw CPU COO occurrences across separate backward calls.

        CPU sparse addition can merge equal row IDs in BF16 before our optimizer
        promotes/coalesces in FP32. CUDA sparse addition concatenates them. This
        leaf hook gives separate microbatch backward calls the same occurrence
        semantics without assigning an FP32 gradient to a BF16 parameter.

        Multiple lookups in one backward are rejected: their gradients can merge
        before this leaf hook. Decoder training performs one lookup per table per
        microbatch; checkpointing the decoder does not repeat the lookup.
        """
        weight = self.weight
        eligible = (self.sparse_gradients and weight.requires_grad and weight.device.type == "cpu"
                    and weight.dtype in (torch.float16, torch.bfloat16))
        previous = (self._sparse_accumulation_parameter()
                    if self._sparse_accumulation_parameter is not None else None)
        last_hook = next(reversed(weight._backward_hooks or {}), None)
        if eligible and previous is weight:
            if last_hook != self._sparse_accumulation_handle.id:
                raise RuntimeError("register gradient-transform hooks before the first CPU Engram forward")
            return
        if self._sparse_accumulation_handle is not None:
            self._sparse_accumulation_handle.remove()
        self._sparse_accumulation_parameter = self._sparse_accumulation_handle = None
        self._sparse_accumulation_state = None
        if not eligible:
            return
        parameter_ref = weakref.ref(weight)
        hook_id = [None]
        state = {"lookup_backwards": 0}

        def preserve_occurrences(incoming):
            parameter = parameter_ref()
            if parameter is None:
                return incoming
            lookups = state["lookup_backwards"]
            state["lookup_backwards"] = 0
            if lookups > 1:
                raise RuntimeError("CPU low-precision Engram requires one lookup per table per backward call")
            if torch.is_grad_enabled():
                raise RuntimeError("CPU Engram microbatch accumulation supports first-order backward only")
            existing = parameter.grad
            if existing is None:
                return incoming
            if next(reversed(parameter._backward_hooks or {}), None) != hook_id[0]:
                raise RuntimeError("register gradient-transform hooks before the CPU Engram forward")
            if not incoming.is_sparse or not existing.is_sparse:
                raise RuntimeError("CPU Engram accumulation requires sparse COO gradients")
            # Allocate before clearing .grad so an allocation failure retains the
            # previously accumulated gradient. Values remain in the leaf dtype.
            combined = torch.sparse_coo_tensor(
                torch.cat((existing._indices(), incoming._indices()), dim=1),
                torch.cat((existing._values(), incoming._values()), dim=0),
                incoming.shape, dtype=incoming.dtype, device=incoming.device,
                is_coalesced=False,
            )
            parameter.grad = None
            return combined

        self._sparse_accumulation_parameter = parameter_ref
        self._sparse_accumulation_state = state
        self._sparse_accumulation_handle = weight.register_hook(preserve_occurrences)
        hook_id[0] = self._sparse_accumulation_handle.id

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
            if self._sparse_accumulation_state is not None and result.rows.requires_grad:
                state = self._sparse_accumulation_state

                def count_lookup_backward(gradient):
                    state["lookup_backwards"] += 1
                    return gradient

                result.rows.register_hook(count_lookup_backward)
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
        if self._sparse_accumulation_handle is not None:
            self._sparse_accumulation_handle.remove()
        self._sparse_accumulation_parameter = self._sparse_accumulation_handle = None
        self._sparse_accumulation_state = None
