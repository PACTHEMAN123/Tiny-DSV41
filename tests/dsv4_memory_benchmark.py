"""Measure one memory-policy variant per fresh torchrun invocation."""

import argparse
from contextlib import nullcontext
from copy import deepcopy
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

from dsv41_train.activation import ActivationCheckpointConfig
from dsv41_train.layout import local_tensor, parameter_memory
from dsv41_train.loading import LoadPolicy
from dsv41_train.loading import source as loading_source
from dsv41_train.models.dsv4 import load_dsv41_backbone_window
from dsv41_train.models.dsv4.config import DeepSeekV41Config
from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.models.dsv4.materialize import FSDPMaterializer
from dsv41_train.models.dsv4.parallel import build_parallelism
from dsv41_train.models.dsv4.policies import FSDPPolicy, GradientPolicy
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from post_training_helpers import small_config
from test_dsv4_loading import synthetic_checkpoint


VARIANTS = ("baseline", "direct", "cpu", "cpu_async", "cpu_cache", "cpu_cache_sync", "cpu_dedup", "gpu_dedup",
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
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--input-pattern", choices=("fixed", "fresh", "repeated"), default="fixed")
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--cache-mib", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--load-mode", choices=("layerwise", "direct_shard"))
    parser.add_argument("--load-pipeline", action="store_true")
    parser.add_argument("--staging-mib", type=int, default=8)
    parser.add_argument("--activation-checkpoint", choices=("none", "full", "selective"), default="none")
    parser.add_argument("--memory-limit-gib", type=float, default=0,
                        help="per-process PyTorch allocator limit; excludes CUDA/NCCL allocations")
    parser.add_argument("--distributed-timeout", type=int, default=120)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--profile-record-shapes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trace-dir")
    parser.add_argument("--load-trace-dir", help="instrument loading in a separate diagnostic run; load_seconds then includes profiler overhead")
    parser.add_argument("--trace-steps", type=int, default=1)
    parser.add_argument("--prefetch-steps", type=int, default=0,
                        help="CPU Engram readiness diagnostics after uninstrumented timed steps")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1 or args.trace_steps < 1:
        parser.error("steps, warmup, and trace-steps must be positive")
    if args.prefetch_steps < 0:
        parser.error("prefetch-steps must be nonnegative")
    if args.microbatches < 1:
        parser.error("microbatches must be positive")
    if args.batch_size < 1 or args.staging_mib < 1 or args.memory_limit_gib < 0 or args.distributed_timeout < 1:
        parser.error("batch size, staging budget and timeout must be positive; memory limit cannot be negative")
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if args.memory_limit_gib:
        fraction = args.memory_limit_gib * 1024**3 / total_bytes
        if fraction > 1:
            parser.error("memory limit exceeds physical device capacity")
        torch.cuda.set_per_process_memory_fraction(fraction, device)
    dist.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(seconds=args.distributed_timeout))
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
        loading = LoadPolicy(args.load_mode or ("direct_shard" if variant in {"direct", "combined"} else "layerwise"),
                             staging_bytes=args.staging_mib * 1024**2,
                             pipeline=args.load_pipeline if args.load_mode else variant in {"direct", "combined"})
        engram = EngramPolicy("cpu" if variant in {"cpu", "cpu_async", "cpu_cache", "cpu_cache_sync", "cpu_dedup", "combined"} else "cuda",
                              "async" if variant in {"cpu_async", "cpu_cache", "combined"} else "sync",
                              variant in {"gpu_dedup", "cpu_dedup", "combined"}, args.staging_mib * 1024**2,
                              args.cache_mib * 1024**2 if variant in {"cpu_cache", "cpu_cache_sync"} else 0)
        targets = ("dense",) if variant == "dense_offload" else ("experts",)
        if variant == "combined":
            targets = ("dense", "experts")
        fsdp = FSDPPolicy((1, 3) if variant in {"dense_offload", "expert_offload", "combined"} else (), targets)
        meshes, cp, dispatcher = build_parallelism(args.cp, args.ep)
        torch.cuda.reset_peak_memory_stats(device)
        load_regions, load_capture = nullcontext(), nullcontext(None)
        load_summary_path = None
        if args.load_trace_dir:
            from torch.profiler import ProfilerActivity, profile
            from dsv41_train.profiling import loading_regions, summarize_profiler
            load_spans = []
            load_regions = loading_regions(load_spans)
            # Shape capture retains tensors and prevents FSDP's meta-storage swap.
            load_capture = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=True,
                experimental_config=torch._C._profiler._ExperimentalConfig(profile_all_threads=True))
        with load_regions, load_capture as load_profiler:
            started = time.perf_counter()
            if args.progress and rank == 0:
                print(json.dumps({"phase": "load_start", "variant": variant}), flush=True)
            config = DeepSeekV41Config.from_json(Path(folder) / "config.json")
            model = load_dsv41_backbone_window(folder, num_layers=config.num_hidden_layers, device=device, dtype=torch.bfloat16,
                context_parallel=cp, token_dispatcher=dispatcher, engram_mesh=meshes.engram,
                sparse_engram_gradients=True, engram_policy=engram, load_policy=loading,
                materializer=FSDPMaterializer(meshes, fsdp))
            torch.cuda.synchronize(device)
            load_seconds = time.perf_counter() - started
            load_peak = torch.cuda.max_memory_allocated(device)
        model.set_activation_checkpointing(ActivationCheckpointConfig(args.activation_checkpoint))
        if args.progress and rank == 0:
            print(json.dumps({"phase": "loaded", "seconds": load_seconds,
                              "allocated_gib": torch.cuda.memory_allocated(device) / 1024**3}), flush=True)
        if load_profiler is not None:
            load_folder = Path(args.load_trace_dir)
            load_folder.mkdir(parents=True, exist_ok=True)
            load_trace = load_folder / f"rank-{rank}.json"
            load_profiler.export_chrome_trace(str(load_trace))
            load_summary = summarize_profiler(load_profiler, 1, load_trace)
            load_summary["loading_spans"] = load_spans
            load_summary_path = str(load_folder / f"rank-{rank}-summary.json")
            Path(load_summary_path).write_text(json.dumps(load_summary, indent=2) + "\n")
        optimizer = build_optimizer(model, OptimizerPolicy("sgd", args.learning_rate, variant in {"foreach", "combined"}))
        gradients = GradientPolicy(model, meshes)
        generator = torch.Generator(device=device).manual_seed(123 + rank // args.cp)
        def new_batch():
            return torch.randint(3, model.config.vocab_size, (args.batch_size, args.seq_len), device=device, generator=generator)
        total_steps = args.warmup + args.steps + args.prefetch_steps + args.trace_steps
        if args.input_pattern == "fixed":
            batches = [new_batch() for _ in range(args.microbatches)]
            batch_schedule = [batches] * total_steps
        elif args.input_pattern == "repeated":
            batch_schedule = [[new_batch()] * args.microbatches for _ in range(total_steps)]
        else:
            batch_schedule = [[new_batch() for _ in range(args.microbatches)] for _ in range(total_steps)]
        batch_schedule = iter(batch_schedule)

        def step(tracing=False):
            from torch.profiler import record_function
            region = record_function if tracing else lambda _: nullcontext()
            with region("train/zero_grad"):
                optimizer.zero_grad(set_to_none=True)
            gradients.begin_step()
            batches = next(batch_schedule)
            for tokens in batches:
                with region("train/forward"):
                    output = model(tokens, labels=tokens)
                with region("train/backward"):
                    (output.loss / len(batches)).backward()
            with region("train/gradient_normalization"):
                gradients.normalize()
                model.synchronize_engram()
            with region("train/optimizer"):
                optimizer.step()
            torch.cuda.synchronize(device)
            return output.loss.detach().item()

        torch.cuda.reset_peak_memory_stats(device)
        for index in range(args.warmup):
            loss = step()
            if args.progress and rank == 0:
                print(json.dumps({"phase": "warmup", "step": index, "loss": loss,
                                  "peak_gib": torch.cuda.max_memory_allocated(device) / 1024**3}), flush=True)
        warmup_peak = torch.cuda.max_memory_allocated(device)
        torch.cuda.synchronize(device)
        for table in model.model.engram_tables.values():
            for key in table.metrics:
                table.metrics[key] = 0
        torch.cuda.reset_peak_memory_stats(device)
        allocator_before = torch.cuda.memory_stats(device)
        dist.barrier()
        started = time.perf_counter()
        losses, step_ms = [], []
        for index in range(args.steps):
            tick = time.perf_counter()
            losses.append(step())
            step_ms.append((time.perf_counter() - tick) * 1000)
            if args.progress and rank == 0:
                print(json.dumps({"phase": "timed", "step": index, "loss": losses[-1],
                                  "ms": step_ms[-1], "peak_gib": torch.cuda.max_memory_allocated(device) / 1024**3}), flush=True)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        allocator_after = torch.cuda.memory_stats(device)
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
            "loading": model.loading_metrics, "losses": losses, "step_ms": step_ms,
            "warmup_peak_bytes": warmup_peak, "free_bytes_at_start": free_bytes,
            "device_total_bytes": total_bytes,
            "allocator_counters_during_timing": {
                key: allocator_after.get(key, 0) - allocator_before.get(key, 0)
                for key in ("num_alloc_retries", "num_ooms", "num_sync_all_streams",
                            "num_device_alloc", "num_device_free")},
            "loading_profile": load_summary_path,
            "engram": {name: deepcopy(table.metrics) for name, table in model.model.engram_tables.items()}}
        metrics["engram_cache_bytes"] = sum(
            table.store.cache.allocated_bytes for table in model.model.engram_tables.values()
            if getattr(table.store, "cache", None) is not None)
        metrics["engram_pinned_bytes"] = sum(
            table.store.stager.allocated_bytes for table in model.model.engram_tables.values()
            if hasattr(table.store, "stager"))
        if args.trace_dir:
            from torch.profiler import ProfilerActivity, profile
            from dsv41_train.profiling import model_regions, summarize_profiler
            trace_dir = Path(args.trace_dir)
            trace_dir.mkdir(parents=True, exist_ok=True)
            spans = []
            dist.barrier()
            with model_regions(model, spans), profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=args.profile_record_shapes, profile_memory=True,
                    experimental_config=torch._C._profiler._ExperimentalConfig(profile_all_threads=True)) as profiler:
                for _ in range(args.trace_steps):
                    step(tracing=True)
            trace = trace_dir / f"rank-{rank}.json"
            profiler.export_chrome_trace(str(trace))
            summary = summarize_profiler(profiler, args.trace_steps, trace)
            summary["lookup_spans"] = spans
            summary_path = trace_dir / f"rank-{rank}-summary.json"
            summary_path.write_text(json.dumps(summary, indent=2) + "\n")
            metrics["trace"] = str(trace)
            metrics["profile_summary"] = str(summary_path)
        # Readiness probes advance CPU variants only; keep them after paired traces.
        if args.prefetch_steps and engram.device == "cpu":
            from engram_prefetch_probe import prefetch_probe
            samples, diagnostic_step_ms = [], []
            dist.barrier()
            with prefetch_probe(model.model.engram_tables, samples):
                for index in range(args.prefetch_steps):
                    first = len(samples)
                    tick = time.perf_counter()
                    step()
                    diagnostic_step_ms.append((time.perf_counter() - tick) * 1000)
                    for sample in samples[first:]:
                        sample["step"] = index
            metrics["prefetch_diagnostic"] = {"steps": args.prefetch_steps,
                "step_ms": diagnostic_step_ms, "samples": samples,
                "note": "Host-clock timestamps; CPU store return follows staged H2D completion. "
                        "Readiness excludes row-return NCCL. Diagnostic steps follow profiling, when enabled, "
                        "and are excluded from throughput."}
        gathered = [None] * world
        dist.all_gather_object(gathered, metrics)
        if rank == 0:
            report = {"variant": variant, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(device),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "model_path": str(Path(folder).resolve()),
                "loading_implementation": loading_source.__file__, "load_profiled": bool(args.load_trace_dir),
                "world": world, "cp": args.cp, "ep": args.ep, "steps": args.steps,
                "learning_rate": args.learning_rate,
                "warmup": args.warmup, "microbatches": args.microbatches, "sequence_length": args.seq_len,
                "batch_size": args.batch_size, "activation_checkpoint": args.activation_checkpoint,
                "memory_limit_gib": args.memory_limit_gib, "profile_record_shapes": args.profile_record_shapes,
                "progress_logging": args.progress,
                "input_pattern": args.input_pattern,
                "config": model.config.to_dict(), "load_policy": asdict(loading),
                "engram_policy": asdict(engram), "fsdp_policy": asdict(fsdp),
                "tokens_per_second": args.steps * args.microbatches * args.batch_size * args.seq_len * (world // args.cp)
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
