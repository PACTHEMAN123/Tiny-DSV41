"""Opt-in host-clock diagnostics, installed only after throughput measurement."""

from concurrent.futures import Future
from contextlib import ExitStack, contextmanager
import time
from unittest.mock import patch


@contextmanager
def prefetch_probe(tables, samples):
    """Measure CPU-store readiness without adding CUDA synchronizations.

    CPURowStore waits for staged DMA before returning, so store_ready is a
    conservative H2D completion timestamp. It does not include row-return NCCL.
    CPU arrival at resolve is earlier than or equal to actual GPU consumption.
    """
    with ExitStack() as stack:
        def install(name, table):
            active = None
            prefetch, resolve = table.prefetch, table.resolve
            lookup, complete = table.store.lookup, table.router.complete

            def measured_prefetch(indices):
                nonlocal active
                active = {"table": name, "request": len(samples),
                          "prefetch_start": time.perf_counter()}
                samples.append(active)
                ticket = prefetch(indices)
                active["prefetch_return"] = time.perf_counter()
                active["local_rows"] = ticket.request.local_ids.numel()
                return ticket

            def measured_lookup(*args, **kwargs):
                sample = active
                sample["store_start"] = time.perf_counter()
                result = lookup(*args, **kwargs)
                sample["store_ready"] = time.perf_counter()
                return result

            def measured_complete(*args, **kwargs):
                active["route_start"] = time.perf_counter()
                result = complete(*args, **kwargs)
                active["route_return"] = time.perf_counter()
                return result

            def measured_resolve(ticket):
                sample = active
                sample["resolve_start"] = time.perf_counter()
                sample["future_wait_ms"] = 0.0
                if isinstance(ticket.result, Future):
                    sample["future_done_at_resolve"] = ticket.result.done()
                    original_result = ticket.result.result

                    def measured_result(*args, **kwargs):
                        start = time.perf_counter()
                        sample["future_wait_start"] = start
                        try:
                            return original_result(*args, **kwargs)
                        finally:
                            sample["future_wait_return"] = time.perf_counter()
                            sample["future_wait_ms"] += (sample["future_wait_return"] - start) * 1000

                    with patch.object(ticket.result, "result", measured_result):
                        result = resolve(ticket)
                else:
                    sample["future_done_at_resolve"] = True
                    result = resolve(ticket)
                sample["resolve_return"] = time.perf_counter()
                sample["ready_before_resolve"] = sample["store_ready"] <= sample["resolve_start"]
                sample["ready_margin_ms"] = (sample["resolve_start"] - sample["store_ready"]) * 1000
                sample["prefetch_host_ms"] = (sample["prefetch_return"] - sample["prefetch_start"]) * 1000
                sample["store_ms"] = (sample["store_ready"] - sample["store_start"]) * 1000
                sample["route_host_ms"] = (sample["route_return"] - sample["route_start"]) * 1000
                return result

            for owner, method, replacement in (
                (table, "prefetch", measured_prefetch), (table, "resolve", measured_resolve),
                (table.store, "lookup", measured_lookup), (table.router, "complete", measured_complete),
            ):
                stack.enter_context(patch.object(owner, method, replacement))

        for name, table in tables.items():
            if table.policy.device != "cpu":
                raise ValueError("prefetch readiness diagnostics require CPU Engram")
            install(name, table)
        yield
