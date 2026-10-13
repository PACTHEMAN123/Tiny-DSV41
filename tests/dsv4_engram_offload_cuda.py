"""CUDA row-store parity and allocation stability across consecutive updates."""

import argparse
import gc
import json
from pathlib import Path

import torch

from dsv41_train.models.dsv4.engram import EngramPolicy, RowShardedEmbedding
from dsv41_train.optimization import OptimizerPolicy, build_optimizer


def check(dtype, prefetch, deduplicate, cache_bytes=0):
    torch.manual_seed(42)
    reference = RowShardedEmbedding(1013, 64, sparse_gradients=True).to(device="cuda", dtype=dtype)
    actual = RowShardedEmbedding(1013, 64, sparse_gradients=True,
        policy=EngramPolicy("cpu", prefetch, deduplicate, 4096, cache_bytes)).to(dtype=dtype)
    control = RowShardedEmbedding(1013, 64, sparse_gradients=True,
        policy=EngramPolicy("cpu", prefetch, deduplicate, 4096)).to(dtype=dtype) if cache_bytes else None
    reference.weight.data.normal_(std=.1)
    actual.weight.data.copy_(reference.weight.detach().cpu())
    if control is not None:
        control.weight.data.copy_(actual.weight)
    tables = (reference, actual) if control is None else (reference, actual, control)
    optimizers = [build_optimizer(table, OptimizerPolicy("sgd", .01)) for table in tables]
    indices = [torch.tensor([[0, 1, 1, 1, 900, 1012]], device="cuda").repeat(8, 1),
               torch.empty((1, 0), device="cuda", dtype=torch.long)]
    if cache_bytes:
        indices.insert(1, indices[0])
    allocated = []
    errors = {"output": 0., "gradient": 0., "weight": 0.}
    tolerances = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=.02, atol=.002)
    placement_tolerance = tolerances if cache_bytes else dict(rtol=0, atol=0)
    try:
        for step in range(24):
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            for ids in indices:
                expected = reference(ids)
                ticket = actual.prefetch(ids)
                # Compute work may run while the worker fills the bounded staging buffers.
                expected_loss = expected.float().square().sum() * .125
                output = actual.resolve(ticket)
                torch.testing.assert_close(output, expected, **placement_tolerance)
                if control is not None:
                    control_output = control(ids)
                    torch.testing.assert_close(output, control_output, rtol=0, atol=0)
                    (control_output.float().square().sum() * .125).backward()
                    del control_output
                if output.numel():
                    errors["output"] = max(errors["output"], (output - expected).abs().max().item())
                (output.float().square().sum() * .125).backward()
                expected_loss.backward()
                del output, expected, expected_loss, ticket
            actual_gradient = actual.weight.grad.float().to_dense()
            expected_gradient = reference.weight.grad.float().to_dense().cpu()
            torch.testing.assert_close(actual_gradient, expected_gradient, **tolerances)
            if control is not None:
                torch.testing.assert_close(actual_gradient, control.weight.grad.float().to_dense(), rtol=0, atol=0)
            errors["gradient"] = max(errors["gradient"], (actual_gradient - expected_gradient).abs().max().item())
            del actual_gradient, expected_gradient
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            expected_weight = reference.weight.detach().cpu()
            torch.testing.assert_close(actual.weight, expected_weight, **placement_tolerance)
            if control is not None:
                torch.testing.assert_close(actual.weight, control.weight, rtol=0, atol=0)
            errors["weight"] = max(errors["weight"], (actual.weight - expected_weight).abs().max().item())
            del expected_weight
            assert actual.weight.device.type == "cpu" and actual.weight.grad is None
            assert not actual._pending and not optimizers[1].state
            gc.collect()
            torch.cuda.synchronize()
            if step >= 4:
                allocated.append(torch.cuda.memory_allocated())
        growth = max(allocated) - min(allocated)
        assert growth <= 1024**2, f"CUDA allocations grew by {growth} bytes"
        return {"dtype": str(dtype), "prefetch": prefetch, "deduplicate": deduplicate,
                "cache_bytes": cache_bytes,
                "cache_vs_cpu_control_exact": control is not None,
                "steps": 24, "max_errors": errors, "allocation_range_bytes": growth,
                "metrics": dict(actual.metrics)}
    finally:
        reference.close()
        actual.close()
        if control is not None:
            control.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/memory-engram-lifecycle.json")
    parser.add_argument("--cache-bytes", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    for dtype in (torch.float32, torch.bfloat16):
        for prefetch in ("sync", "async"):
            for deduplicate in (False, True):
                result = check(dtype, prefetch, deduplicate, args.cache_bytes)
                results.append(result)
                print(json.dumps(result), flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
