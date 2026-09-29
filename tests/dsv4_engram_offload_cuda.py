"""CUDA row-store parity and allocation stability across consecutive updates."""

import argparse
import gc
import json
from pathlib import Path

import torch

from dsv41_train.models.dsv4.engram import EngramPolicy, RowShardedEmbedding
from dsv41_train.optimization import OptimizerPolicy, build_optimizer


def check(dtype, prefetch, deduplicate):
    torch.manual_seed(42)
    reference = RowShardedEmbedding(1013, 64, sparse_gradients=True).to(device="cuda", dtype=dtype)
    actual = RowShardedEmbedding(1013, 64, sparse_gradients=True,
        policy=EngramPolicy("cpu", prefetch, deduplicate, 4096)).to(dtype=dtype)
    reference.weight.data.normal_(std=.1)
    actual.weight.data.copy_(reference.weight.detach().cpu())
    optimizers = [build_optimizer(table, OptimizerPolicy("sgd", .01)) for table in (reference, actual)]
    indices = [torch.tensor([[0, 1, 1, 1, 900, 1012]], device="cuda").repeat(8, 1),
               torch.empty((1, 0), device="cuda", dtype=torch.long)]
    allocated = []
    errors = {"output": 0., "gradient": 0., "weight": 0.}
    tolerances = dict(rtol=2e-5, atol=2e-6) if dtype == torch.float32 else dict(rtol=.02, atol=.002)
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
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                if output.numel():
                    errors["output"] = max(errors["output"], (output - expected).abs().max().item())
                (output.float().square().sum() * .125).backward()
                expected_loss.backward()
                del output, expected, expected_loss, ticket
            actual_gradient = actual.weight.grad.float().to_dense()
            expected_gradient = reference.weight.grad.float().to_dense().cpu()
            torch.testing.assert_close(actual_gradient, expected_gradient, **tolerances)
            errors["gradient"] = max(errors["gradient"], (actual_gradient - expected_gradient).abs().max().item())
            del actual_gradient, expected_gradient
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            expected_weight = reference.weight.detach().cpu()
            torch.testing.assert_close(actual.weight, expected_weight, rtol=0, atol=0)
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
                "steps": 24, "max_errors": errors, "allocation_range_bytes": growth,
                "metrics": dict(actual.metrics)}
    finally:
        reference.close()
        actual.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/memory-engram-lifecycle.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = []
    for dtype in (torch.float32, torch.bfloat16):
        for prefetch in ("sync", "async"):
            for deduplicate in (False, True):
                result = check(dtype, prefetch, deduplicate)
                results.append(result)
                print(json.dumps(result), flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
