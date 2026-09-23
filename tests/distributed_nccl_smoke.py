"""Validate a multi-node NCCL process group before loading model weights."""

from __future__ import annotations

import argparse
import json
import os
import socket
import time

import torch
import torch.distributed as dist

from dsv41_train.runtime import initialize_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-nodes", type=int, default=2)
    parser.add_argument("--expected-local-world-size", type=int, default=8)
    parser.add_argument("--elements", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    return parser.parse_args()


def gather_hostnames(device: torch.device) -> list[str]:
    encoded = socket.gethostname().encode("utf-8")
    if len(encoded) > 255:
        raise RuntimeError("hostname exceeds the 255-byte diagnostic buffer")
    local = torch.zeros(256, dtype=torch.uint8, device=device)
    local[: len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8, device=device)
    gathered = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local)
    return [
        bytes(item.cpu().tolist()).split(b"\0", 1)[0].decode("utf-8")
        for item in gathered
    ]


def main() -> None:
    args = parse_args()
    runtime = initialize_runtime(require_distributed=True)
    try:
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        expected_world_size = args.expected_nodes * args.expected_local_world_size
        if local_world_size != args.expected_local_world_size:
            raise RuntimeError(
                f"expected {args.expected_local_world_size} local ranks, got {local_world_size}"
            )
        if runtime.world_size != expected_world_size:
            raise RuntimeError(
                f"expected {expected_world_size} total ranks, got {runtime.world_size}"
            )

        hostnames = gather_hostnames(runtime.device)
        host_counts = {hostname: hostnames.count(hostname) for hostname in sorted(set(hostnames))}
        if len(host_counts) != args.expected_nodes:
            raise RuntimeError(
                f"expected {args.expected_nodes} hosts, got {len(host_counts)}: {host_counts}"
            )
        if any(count != args.expected_local_world_size for count in host_counts.values()):
            raise RuntimeError(f"unexpected rank placement: {host_counts}")

        checksum = torch.tensor(runtime.rank, dtype=torch.int64, device=runtime.device)
        dist.all_reduce(checksum)
        expected_checksum = runtime.world_size * (runtime.world_size - 1) // 2
        if checksum.item() != expected_checksum:
            raise RuntimeError(
                f"all-reduce checksum mismatch: {checksum.item()} != {expected_checksum}"
            )

        payload = torch.empty(args.elements, dtype=torch.bfloat16, device=runtime.device)
        for _ in range(args.warmup):
            payload.fill_(runtime.rank + 1)
            dist.all_reduce(payload)
        torch.cuda.synchronize(runtime.device)
        dist.barrier()
        started = time.perf_counter()
        for _ in range(args.iterations):
            payload.fill_(runtime.rank + 1)
            dist.all_reduce(payload)
        torch.cuda.synchronize(runtime.device)
        elapsed = time.perf_counter() - started

        expected_value = runtime.world_size * (runtime.world_size + 1) // 2
        if payload[0].item() != expected_value or payload[-1].item() != expected_value:
            raise RuntimeError("payload all-reduce produced an unexpected value")

        elapsed_tensor = torch.tensor(elapsed, dtype=torch.float64, device=runtime.device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        max_elapsed = elapsed_tensor.item()
        payload_gib = payload.numel() * payload.element_size() / 1024**3
        if runtime.is_main:
            print(
                json.dumps(
                    {
                        "backend": dist.get_backend(),
                        "world_size": runtime.world_size,
                        "local_world_size": local_world_size,
                        "hosts": host_counts,
                        "checksum": checksum.item(),
                        "payload_gib": payload_gib,
                        "iterations": args.iterations,
                        "max_elapsed_seconds": max_elapsed,
                        "payload_gib_per_second": payload_gib * args.iterations / max_elapsed,
                        "torch": torch.__version__,
                        "cuda": torch.version.cuda,
                        "nccl": torch.cuda.nccl.version(),
                    }
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
