"""Logical row slices over plain, block-quantized, and fused source tensors."""

from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import chain
import math

import torch

from .quantization import dequantize_fp4_rows, dequantize_fp8_blocks, dequantize_fp8_rows


@dataclass(frozen=True)
class WeightSource:
    names: tuple[str, ...]
    encoding: str = "plain"

    def shape(self, reader, name):
        shape = tuple(reader.metadata(name if self.encoding == "plain" else name + ".weight")["shape"])
        return (shape[0], shape[1] * 2) if self.encoding == "fp4" else shape

    def _read(self, reader, name, start, stop, dtype):
        if self.encoding == "plain":
            return reader.tensor_rows(name, start, stop, "cpu").to(dtype)
        if self.encoding == "fp8_block":
            lo, hi = start // 32 * 32, (stop + 31) // 32 * 32
            weight = reader.tensor_rows(name + ".weight", lo, hi, "cpu")
            scale = reader.tensor_rows(name + ".scale", lo // 32, hi // 32, "cpu")
            return dequantize_fp8_blocks(weight, scale, dtype=dtype)[start-lo:stop-lo]
        weight = reader.tensor_rows(name + ".weight", start, stop, "cpu")
        scale = reader.tensor_rows(name + ".scale", start, stop, "cpu")
        decode = dequantize_fp4_rows if self.encoding == "fp4" else dequantize_fp8_rows
        return decode(weight, scale, dtype=dtype)

    def read_rows(self, reader, start, stop, dtype):
        parts, offset = [], 0
        for name in self.names:
            rows = self.shape(reader, name)[0]
            lo, hi = max(start, offset), min(stop, offset + rows)
            if hi > lo:
                parts.append(self._read(reader, name, lo-offset, hi-offset, dtype))
            offset += rows
        if stop > offset or start < 0:
            raise ValueError("source row slice is outside the checkpoint tensor")
        if not parts:
            return torch.empty((0, *self.shape(reader, self.names[0])[1:]), dtype=dtype)
        return parts[0] if len(parts) == 1 else torch.cat(parts)

    def fill(self, reader, target, offsets, logical_shape, policy, *, executor=None):
        if any(offsets[1:]):
            raise ValueError("direct loading currently requires dimension-zero FSDP shards")
        factor = math.prod(logical_shape[1:-1]) if len(logical_shape) > 2 else 1
        start = offsets[0] * factor
        rows = target.shape[0] * factor
        flat = target.reshape(rows, *self.shape(reader, self.names[0])[1:])
        row_bytes = max(1, math.prod(flat.shape[1:]) * target.element_size())
        alignment = 32 if self.encoding == "fp8_block" else 1
        capacity = policy.staging_bytes // (16 * row_bytes)
        if rows and capacity < alignment:
            raise ValueError("loading staging budget cannot hold one aligned quantization block")
        chunk = max(1, min(policy.chunk_rows, capacity))
        def row_ranges():
            lo = start
            while lo < start + rows:
                # Bound the decoded block, including the prefix before a shard edge.
                aligned_stop = (lo // alignment + capacity // alignment) * alignment
                hi = min(lo + chunk, start + rows, aligned_stop)
                yield lo, hi
                lo = hi
        ranges = row_ranges()
        first = next(ranges, None)
        if first is None:
            return
        pipelined = policy.pipeline and first[1] < start + rows
        def read(bounds):
            return self.read_rows(reader, *bounds, target.dtype)
        def copy(value, bounds):
            if target.is_cuda and pipelined:
                value = value.pin_memory()
            flat[bounds[0]-start:bounds[1]-start].copy_(value, non_blocking=pipelined)
        with torch.no_grad():
            if pipelined:
                context = (nullcontext(executor) if executor is not None else
                           ThreadPoolExecutor(max_workers=1, thread_name_prefix="weights"))
                with context as worker:
                    future = worker.submit(read, first)
                    bounds = first
                    try:
                        while bounds is not None:
                            value = future.result()
                            following = next(ranges, None)
                            if following is not None:
                                future = worker.submit(read, following)
                            copy(value, bounds)
                            if target.is_cuda:
                                torch.cuda.current_stream(target.device).synchronize()
                            bounds = following
                    finally:
                        # A failed copy must not leave a reader active after its mmap closes.
                        wait((future,))
            else:
                for bounds in chain((first,), ranges):
                    copy(read(bounds), bounds)
