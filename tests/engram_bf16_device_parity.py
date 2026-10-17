"""Strict native BF16 sparse SGD gate, including released V41 row layout.

Run with PYTHONPATH=. python tests/engram_bf16_device_parity.py --device cpu,
or torchrun --standalone --nproc-per-node=2 ... --device cuda --output FILE.
This isolates Engram hash/routing/projection/cache; it is not a full-model test.
No FP32 master table or replacement optimizer is used in the resident path.
"""

import argparse
import json
import os
import struct
import tempfile
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from dsv41_train.engram_offload import install_engram_sgd_hooks
from dsv41_train.models.dsv4 import DeepSeekV41Config
from dsv41_train.models.dsv4.checkpoint import ShardedSafeTensorReader, _load_engram_rows
from dsv41_train.models.dsv4.model import Engram, NgramHash, RowShardedEmbedding


def exact(actual, expected, label):
    torch.testing.assert_close(actual.detach().cpu(), expected.detach().cpu(),
                               rtol=0, atol=0, msg=label)


def arithmetic(device):
    generator = torch.Generator().manual_seed(42)
    cpu = torch.randn(20_000, 2, generator=generator).bfloat16()
    resident = cpu.to(device).clone()
    fp32_round_differences = 0
    for step in range(20):
        ids = torch.randint(len(cpu), (1, 30_000), generator=generator)
        values = (torch.randn(30_000, 2, generator=generator) * .1).bfloat16()
        raw = torch.sparse_coo_tensor(ids, values, cpu.shape, check_invariants=True)
        gradients = [raw.float().coalesce().bfloat16(),
                     raw.to(device).float().coalesce().bfloat16()]
        exact(gradients[0].indices(), gradients[1].indices(), f"arithmetic indices {step}")
        exact(gradients[0].values(), gradients[1].values(), f"arithmetic gradients {step}")
        once = cpu.float().add_(gradients[0].float(), alpha=-.001).bfloat16()
        cpu.add_(gradients[0], alpha=-.001)
        resident.add_(gradients[1], alpha=-.001)
        fp32_round_differences += int((cpu != once).sum())
        exact(resident, cpu, f"native BF16 sparse SGD {step}")
    return dict(steps=20, elements=cpu.numel(), occurrences_per_step=30_000,
                cpu_device_exact=True, differences_from_fp32_single_round=fp32_round_differences)


def released_fixture(folder, rows, width):
    generator = torch.Generator().manual_seed(943)
    values = torch.randn(rows, width, generator=generator).to(torch.float8_e4m3fn)
    scales = (2. ** torch.randint(-5, 0, (rows, width // 32), generator=generator)).to(torch.float8_e8m0fnu)
    header, payload = {}, bytearray()
    for suffix, tensor, dtype in (("weight", values, "F8_E4M3"), ("scale", scales, "F8_E8M0")):
        start = len(payload)
        payload.extend(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        header[f"layers.1.engram.embed.{suffix}"] = dict(
            dtype=dtype, shape=list(tensor.shape), data_offsets=[start, len(payload)])
    encoded = json.dumps(header).encode()
    (folder / "weights.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    (folder / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {name: "weights.safetensors" for name in header}}))


def rows_gate(device, mesh, rank):
    config = replace(DeepSeekV41Config.tiny(), engram_vocab_size=17,
                     engram_num_embeddings=[97], engram_max_ngram_size=3, engram_head_dim=64)
    hashing = NgramHash(config).to(device)
    torch.manual_seed(611)
    block = Engram(config).to(device=device, dtype=torch.bfloat16).requires_grad_(False)
    models, optimizers = [], []
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        released_fixture(folder, 97, 64)
        with ShardedSafeTensorReader(folder) as reader:
            for offload, cache_rows in ((False, 0), (True, 0), (True, 3)):
                table = RowShardedEmbedding(97, 64, mesh, sparse_gradients=True,
                                           engram_offload=offload)
                table.weight = nn.Parameter(_load_engram_rows(
                    reader, "layers.1", table, device="cpu" if offload else device,
                    dtype=torch.bfloat16))
                if cache_rows:
                    table.configure_cache(cache_rows * (64 * 2 + 8), device)
                optimizer = torch.optim.SGD(table.parameters(), lr=.001, foreach=False)
                install_engram_sgd_hooks(table, optimizer)
                models.append(table)
                optimizers.append(optimizer)
    for table in models[1:]:
        exact(table.weight, models[0].weight, "released initial table")
    initial = models[0].weight.detach().cpu().clone()
    generator = torch.Generator().manual_seed(901 + rank)
    reports = []
    for step in range(1, 21):
        # Repeated ngrams, sequence boundaries and padding exercise real hash
        # indices and uneven owner routing, with one backward as in a2.
        tokens = torch.randint(3, 32, (1, 64), generator=generator)
        tokens[:, 8:32] = tokens[:, :8].repeat(1, 3)
        mask = torch.ones_like(tokens, dtype=torch.bool)
        mask[:, -3:] = False
        sequence_ids = torch.zeros_like(tokens)
        sequence_ids[:, 48:] = 1
        ids = hashing(tokens.to(device), mask.to(device), sequence_ids.to(device))[:, :, 0]
        streams = torch.randn(1, 64, config.hc_mult, config.hidden_size,
                              generator=generator).to(device=device, dtype=torch.bfloat16)
        outputs, gradients = [], []
        for table, optimizer in zip(models, optimizers):
            optimizer.zero_grad(set_to_none=True)
            output = block(streams, table(ids), mask.to(device))
            # Scale enough to cause observable BF16 table changes at lr=.001.
            output.float().square().mean().mul(128).backward()
            outputs.append(output.detach())
            gradient = table.weight.grad
            gradients.append((gradient._indices().cpu().clone(), gradient._values().cpu().clone()))
            assert torch.isfinite(gradient._values()).all()
        for index in (1, 2):
            exact(outputs[index], outputs[0], f"Engram output {step}/{index}")
            exact(gradients[index][0], gradients[0][0], f"raw indices {step}/{index}")
            exact(gradients[index][1], gradients[0][1], f"raw gradients {step}/{index}")
        for optimizer in optimizers:
            optimizer.step()
        for table in models[1:]:
            exact(table.weight.grad.indices(), models[0].weight.grad.indices(), f"coalesced indices {step}")
            exact(table.weight.grad.values(), models[0].weight.grad.values(), f"coalesced gradients {step}")
            exact(table.weight, models[0].weight, f"updated rows {step}")
        cache = models[2]._cpu_lookup._cache
        assert (cache.tags == -1).all(), "optimizer did not invalidate cache"
        assert cache._ticket is None, "optimizer left a pending transfer"
        # Owner-local probes avoid routing ambiguity, cover misses then genuine
        # hits after invalidation, and never count as natural efficiency gains.
        local_ids = torch.tensor([0, 1, 0], device=device)
        expected = models[2].weight.detach()[local_ids.cpu()]
        with torch.no_grad():
            exact(models[2]._lookup(local_ids), expected, f"updated cache miss {step}")
            before = cache.metrics.copy()
            exact(models[2]._lookup(local_ids), expected, f"updated cache hit {step}")
        assert cache.metrics["hits"] - before["hits"] == 3
        assert cache.metrics["transferred_rows"] == before["transferred_rows"]
        assert cache.metrics["resident_bytes"] <= 408
        reports.append(dict(step=step, raw_occurrences=gradients[0][0].numel(),
                            gradient_rows=models[0].weight.grad._nnz(),
                            output_gradient_update_cache_exact=True))
    changed = int((models[0].weight.detach().cpu() != initial).any(dim=1).sum())
    assert changed > 0, "fixture did not update this owner's table"
    for table in models:
        table.synchronize()
    return dict(rank=rank, steps=reports, changed_rows=changed,
                table_dtype=str(models[0].weight.dtype), cache_probe_hits=60,
                scope="V41 released row loader + hash + EP row routing + Engram projection; no FSDP/full model")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    world, rank = int(os.getenv("WORLD_SIZE", "1")), int(os.getenv("RANK", "0"))
    device = torch.device("cpu")
    if args.device == "cuda":
        device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
    mesh = None
    if world > 1:
        dist.init_process_group("nccl" if args.device == "cuda" else "gloo", timeout=timedelta(seconds=120))
        mesh = init_device_mesh(args.device, (world,), mesh_dim_names=("engram",))
    try:
        report = dict(torch=torch.__version__, device=str(device), world_size=world,
                      arithmetic=arithmetic(device), rows=rows_gate(device, mesh, rank))
        reports = [None] * world
        if world > 1:
            dist.all_gather_object(reports, report)
        else:
            reports[0] = report
        if rank == 0:
            if args.output:
                args.output.write_text(json.dumps(reports, indent=2) + "\n")
            print(json.dumps(dict(passed=True, reports=reports)), flush=True)
    finally:
        if world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
