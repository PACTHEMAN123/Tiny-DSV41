"""Two-rank output and gradient parity for row-sharded Engram lookup."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from dsv41_train.models.dsv4.model import RowShardedEmbedding


def main() -> None:
    dist.init_process_group("gloo")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            raise RuntimeError("this parity test requires exactly two ranks")

        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("ep",))
        table = RowShardedEmbedding(11, 3, mesh)
        full_weight = torch.arange(33, dtype=torch.float32).view(11, 3)
        with torch.no_grad():
            table.weight.copy_(full_weight[table.row_start : table.row_stop])

        indices = torch.tensor([[rank, 4 + rank, 10 - rank]])
        output = table(indices)
        torch.testing.assert_close(output, full_weight[indices])
        output.sum().backward()

        expected_gradient = torch.zeros_like(table.weight)
        global_indices = (0, 4, 10, 1, 5, 9)
        for index in global_indices:
            if table.row_start <= index < table.row_stop:
                expected_gradient[index - table.row_start].fill_(1.0 / world_size)
        torch.testing.assert_close(table.weight.grad, expected_gradient)
        if rank == 0:
            print("row-sharded Engram parity passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
    main()
