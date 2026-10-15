"""Two-rank CPU CP/row-owner parity for synchronous trainable Engram offload."""

import argparse
import json
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from dsv41_train.cp import ContextParallel
from dsv41_train.engram_offload import install_engram_sgd_hooks
from dsv41_train.models.dsv4.model import RowShardedEmbedding
from dsv41_train.models.dsv4.parallel import synchronize_cp_gradients


def check(mesh, cp_size, frozen, empty_owner):
    rank, world = dist.get_rank(), dist.get_world_size()
    cp = ContextParallel(mesh if cp_size > 1 else None)
    actual, reference = nn.Module(), nn.Module()
    actual.table = RowShardedEmbedding(11, 2, mesh, sparse_gradients=True, engram_offload=True).bfloat16()
    reference.table = nn.Embedding(11, 2, sparse=True)
    actual.dense = nn.Parameter(torch.tensor(.5))
    reference.dense = nn.Parameter(torch.tensor(.5))
    with torch.no_grad():
        reference.table.weight.copy_(torch.arange(22).reshape(11, 2) / 32)
        actual.table.weight.copy_(reference.table.weight[actual.table.row_start:actual.table.row_stop])
    actual.table.requires_grad_(not frozen)
    reference.table.requires_grad_(not frozen)
    optimizer = torch.optim.SGD([p for p in actual.parameters() if p.requires_grad], lr=.125, foreach=False)
    expected_optimizer = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=.125, foreach=False)
    install_engram_sgd_hooks(actual, optimizer)
    ids = [torch.tensor([0, 0]), torch.tensor([0, 0])] if empty_owner else [torch.tensor([0, 6]), torch.tensor([1, 10])]
    maximum_error = 0.
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        expected_optimizer.zero_grad(set_to_none=True)
        for microbatch in range(2):
            coefficients = torch.tensor([[1., 2.], [2., -1.]]) * (microbatch + 1)
            expected = reference.table(torch.cat(ids)).bfloat16().float() * reference.dense
            (expected * coefficients.repeat(2, 1)).sum().div(8).backward()
            output = actual.table(ids[rank]).float() * actual.dense
            torch.testing.assert_close(output, expected.detach()[rank * 2:(rank + 1) * 2], rtol=0, atol=0)
            count = torch.tensor(2)
            loss = cp.mean_loss((output * coefficients).sum(), count)
            (loss * (world / cp_size) * cp.sum(count) / 8).backward()
        dist.all_reduce(actual.dense.grad)
        actual.dense.grad.div_(world)
        synchronize_cp_gradients(actual, cp)
        torch.testing.assert_close(actual.dense.grad, reference.dense.grad, rtol=0, atol=0)
        if frozen:
            assert actual.table.weight.grad is None
        else:
            gradient = actual.table.weight.grad.float().coalesce().to_dense()
            target = reference.table.weight.grad.to_dense()[actual.table.row_start:actual.table.row_stop]
            torch.testing.assert_close(gradient, target, rtol=0, atol=0)
            maximum_error = max(maximum_error, (gradient - target).abs().max().item())
            if empty_owner and rank == 1:
                assert actual.table.weight.grad._nnz() == 0
        optimizer.step()
        if not frozen:
            reference.table.weight.grad = reference.table.weight.grad.coalesce().bfloat16().float()
        expected_optimizer.step()
        # Keep a BF16 authoritative table in both paths: no full FP32 master is
        # used by the offloaded implementation. Only the tiny reference is FP32.
        with torch.no_grad():
            reference.table.weight.copy_(reference.table.weight.bfloat16().float())
        torch.testing.assert_close(actual.table.weight.float(),
                                   reference.table.weight[actual.table.row_start:actual.table.row_stop],
                                   rtol=0, atol=0)
    return dict(cp_size=cp_size, frozen=frozen, empty_owner=empty_owner,
                steps=3, maximum_table_gradient_error=maximum_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        if dist.get_world_size() != 2:
            raise ValueError("this test requires two ranks")
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("engram",))
        results = [check(mesh, cp, frozen, empty) for cp in (1, 2)
                   for frozen in (False, True) for empty in (False, True)]
        if dist.get_rank() == 0:
            report = json.dumps(results, indent=2) + "\n"
            if args.output:
                Path(args.output).write_text(report)
            print(report, flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
