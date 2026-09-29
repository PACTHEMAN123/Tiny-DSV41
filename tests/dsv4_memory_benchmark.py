"""Measure one memory-policy variant per fresh torchrun invocation."""

import argparse
from dataclasses import asdict
import datetime
import json
import os
from pathlib import Path
import resource
import tempfile
import time

import torch
import torch.distributed as dist

from dsv41_train.layout import local_tensor, parameter_memory
from dsv41_train.loading import LoadPolicy
from dsv41_train.models.dsv4 import load_dsv41_backbone_window
from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.models.dsv4.materialize import FSDPMaterializer
from dsv41_train.models.dsv4.parallel import build_parallelism
from dsv41_train.models.dsv4.policies import FSDPPolicy, GradientPolicy
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from post_training_helpers import small_config
from test_dsv4_loading import synthetic_checkpoint


VARIANTS = ("baseline", "direct", "cpu", "cpu_async", "cpu_dedup", "gpu_dedup",
            "dense_offload", "expert_offload", "foreach", "combined")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-path")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--table-rows", type=int, default=262147)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1:
        parser.error("steps and warmup must be positive")
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(seconds=120))
    rank, world = dist.get_rank(), dist.get_world_size()
    temporary = None
    model = None
    try:
        if args.model_path:
            folder = args.model_path
        else:
            temporary = tempfile.TemporaryDirectory(prefix="dsv41-bench-") if rank == 0 else None
            paths = [temporary.name if temporary else None]
            dist.broadcast_object_list(paths)
            folder = paths[0]
            if rank == 0:
                torch.manual_seed(42)
                config = small_config(vocab_size=1024, hidden_size=128, num_attention_heads=4,
                    q_lora_rank=64, o_lora_rank=64, moe_intermediate_size=256, n_routed_experts=6,
                    max_position_embeddings=512, engram_layer_ids=[1, 3],
                    engram_num_embeddings=[args.table_rows, args.table_rows+4],
                    engram_head_dim=128, engram_vocab_size=1023, engram_compressed_vocab_size=1024)
                synthetic_checkpoint(Path(folder), config)
        dist.barrier()
        variant = args.variant
        loading = LoadPolicy("direct_shard" if variant in {"direct", "combined"} else "layerwise",
                             staging_bytes=8 * 1024**2, pipeline=variant in {"direct", "combined"})
        engram = EngramPolicy("cpu" if variant in {"cpu", "cpu_async", "cpu_dedup", "combined"} else "cuda",
                              "async" if variant in {"cpu_async", "combined"} else "sync",
                              variant in {"gpu_dedup", "cpu_dedup", "combined"}, 8 * 1024**2)
        targets = ("dense",) if variant == "dense_offload" else ("experts",)
        if variant == "combined":
            targets = ("dense", "experts")
        fsdp = FSDPPolicy((1, 3) if variant in {"dense_offload", "expert_offload", "combined"} else (), targets)
        meshes, cp, dispatcher = build_parallelism(args.cp, args.ep)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        model = load_dsv41_backbone_window(folder, num_layers=5, device=device, dtype=torch.bfloat16,
            context_parallel=cp, token_dispatcher=dispatcher, engram_mesh=meshes.engram,
            sparse_engram_gradients=True, engram_policy=engram, load_policy=loading,
            materializer=FSDPMaterializer(meshes, fsdp))
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - started
        load_peak = torch.cuda.max_memory_allocated(device)
        optimizer = build_optimizer(model, OptimizerPolicy("sgd", .01, variant in {"foreach", "combined"}))
        gradients = GradientPolicy(model, meshes)
        generator = torch.Generator(device=device).manual_seed(123 + rank // args.cp)
        batches = [torch.randint(3, model.config.vocab_size, (1, args.seq_len),
                                 device=device, generator=generator) for _ in range(2)]

        def step():
            optimizer.zero_grad(set_to_none=True)
            gradients.begin_step()
            for tokens in batches:
                output = model(tokens, labels=tokens)
                (output.loss / len(batches)).backward()
            gradients.normalize()
            model.synchronize_engram()
            optimizer.step()
            return output.loss.detach().item()

        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize(device)
        for table in model.model.engram_tables.values():
            for key in table.metrics:
                table.metrics[key] = 0
        torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        started = time.perf_counter()
        losses = [step() for _ in range(args.steps)]
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        placement = {"cpu": 0, "cuda": 0}
        for parameter in model.parameters():
            local = local_tensor(parameter)
            placement[local.device.type] += local.numel() * local.element_size()
        metrics = {"rank": rank, "load_seconds": load_seconds, "train_seconds": elapsed,
            "load_peak_bytes": load_peak, "train_peak_bytes": torch.cuda.max_memory_allocated(device),
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.max_memory_reserved(device),
            "cpu_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "weights_by_device": placement, "parameter_bytes": parameter_memory(model, optimizer),
            "loading": model.loading_metrics, "losses": losses,
            "engram": {name: table.metrics for name, table in model.model.engram_tables.items()}}
        gathered = [None] * world
        dist.all_gather_object(gathered, metrics)
        if rank == 0:
            report = {"variant": variant, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
                "world": world, "cp": args.cp, "ep": args.ep, "steps": args.steps,
                "warmup": args.warmup, "microbatches": 2, "sequence_length": args.seq_len,
                "config": model.config.to_dict(), "load_policy": asdict(loading),
                "engram_policy": asdict(engram), "fsdp_policy": asdict(fsdp),
                "tokens_per_second": args.steps * 2 * args.seq_len * (world // args.cp)
                                     / max(item["train_seconds"] for item in gathered),
                "ranks": gathered}
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({"variant": variant, "tokens_per_second": report["tokens_per_second"],
                              "load_peak_bytes": max(item["load_peak_bytes"] for item in gathered),
                              "train_peak_bytes": max(item["train_peak_bytes"] for item in gathered)}), flush=True)
    finally:
        if model is not None:
            model.close()
        dist.destroy_process_group()
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    main()
