"""Full-parameter, answer-only SFT benchmark for Qwen3 MoE plus trainable Engram."""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import datetime
import json
import os
from pathlib import Path
import resource
import time

import torch
import torch.distributed as dist

from dsv41_train.layout import local_tensor, parameter_memory
from dsv41_train.models.qwen import load_qwen3_moe
from dsv41_train.models.qwen.engram import (
    QwenEngramConfig, attach_engram, shard_qwen_component, shard_qwen_root,
)
from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.models.dsv4.parallel import build_parallelism
from dsv41_train.models.dsv4.policies import GradientPolicy
from dsv41_train.optimization import OptimizerPolicy, build_optimizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--variant", choices=("gpu", "gpu_cache", "cpu", "cpu_cache"), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--ep", type=int, default=8)
    parser.add_argument("--layers", type=int, nargs="+", default=[1, 24])
    parser.add_argument("--vocab-per-head", type=int, default=900000)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument("--cache-mib", type=int, default=128, help="budget per table per rank")
    parser.add_argument("--staging-mib", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--prefetch", choices=("sync", "async"), default="async")
    parser.add_argument("--trace-dir")
    args = parser.parse_args()
    if min(args.batch_size, args.microbatches, args.steps, args.warmup) < 1:
        parser.error("batch, accumulation, warmup and measured steps must be positive")
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(seconds=900))
    rank, world = dist.get_rank(), dist.get_world_size()
    model = None
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    def progress(**record):
        if rank == 0:
            print(json.dumps({"time": time.time(), **record}), flush=True)

    try:
        data = torch.load(args.data, map_location="cpu", weights_only=True)
        global_batch = args.batch_size * args.microbatches * world
        updates = args.warmup + args.steps + bool(args.trace_dir)
        if data["input_ids"].shape[0] < global_batch * updates:
            raise ValueError("not enough distinct packed SFT sequences; prepare more packs")
        sequence_length = data["input_ids"].shape[1]
        schedule = []
        for step in range(updates):
            start = step * global_batch
            denominator = int(data["labels"][start:start+global_batch].ne(-100).sum())
            batches = []
            for micro in range(args.microbatches):
                offset = start + (micro * world + rank) * args.batch_size
                ids = data["input_ids"][offset:offset+args.batch_size].to(device)
                labels = data["labels"][offset:offset+args.batch_size].to(device)
                count = int(data["labels"][offset:offset+args.batch_size].ne(-100).sum())
                batches.append((ids, labels, count * world / denominator))
            schedule.append(batches)
        meshes, cp, dispatcher = build_parallelism(1, args.ep)
        policy = EngramPolicy("cpu" if args.variant.startswith("cpu") else "cuda",
            args.prefetch if args.variant.startswith("cpu") else "sync", True,
            args.staging_mib * 1024**2, args.cache_mib * 1024**2 if args.variant.endswith("cache") else 0)
        config = QwenEngramConfig(tuple(args.layers), args.vocab_per_head, args.head_dim)
        torch.manual_seed(42)
        torch.cuda.reset_peak_memory_stats(device)
        load_start = time.perf_counter()
        load_spans = []
        previous = load_start

        def setup(loaded):
            attach_engram(loaded, config, policy, meshes.engram, device=device)
            progress(phase="engram_initialized", table_parameters=sum(
                table.global_num_embeddings * table.embedding_dim for table in loaded.model.engram_tables.values()))

        def materialize(name, module):
            nonlocal previous
            shard_qwen_component(name, module, meshes)
            torch.cuda.synchronize(device)
            now = time.perf_counter()
            load_spans.append({"name": name, "seconds_since_previous": now-previous})
            previous = now
            if name.endswith((".0", ".12", ".24", ".36")):
                progress(phase="loading", module=name, seconds=now-load_start)

        progress(phase="load_start", variant=args.variant)
        model = load_qwen3_moe(args.model_path, device=device, context_parallel=cp,
            token_dispatcher=dispatcher, setup_callback=setup, module_callback=materialize)
        shard_qwen_root(model, meshes)
        model.gradient_checkpointing_enable()
        model.train()
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - load_start
        load_peak = torch.cuda.max_memory_allocated(device)
        optimizer = build_optimizer(model, OptimizerPolicy("sgd", args.learning_rate, True))
        gradients = GradientPolicy(model, meshes)
        tables = model.model.engram_tables
        progress(phase="loaded", seconds=load_seconds, allocated_gib=torch.cuda.memory_allocated(device)/2**30)
        initial = local_tensor(model.lm_head.weight).detach().flatten()[:4096].float().cpu().clone()
        table_updates = {}

        def run_step(index, tracing=False, verify=False):
            region = torch.profiler.record_function if tracing else lambda _: nullcontext()
            phase_ms = {}
            with region("train/zero_grad"):
                optimizer.zero_grad(set_to_none=True)
            gradients.begin_step()
            losses = []
            for ids, labels, scale in schedule[index]:
                tick = time.perf_counter()
                with region("train/forward"):
                    output = model(ids, labels=labels, output_router_logits=False)
                    loss = output.loss * scale
                phase_ms["forward_host"] = phase_ms.get("forward_host", 0) + (time.perf_counter()-tick)*1000
                tick = time.perf_counter()
                with region("train/backward"):
                    loss.backward()
                phase_ms["backward_host"] = phase_ms.get("backward_host", 0) + (time.perf_counter()-tick)*1000
                losses.append(loss.detach())
                del output, loss
            with region("train/normalize"):
                gradients.normalize()
                for table in tables.values():
                    table.synchronize()
            probes = {}
            if verify:
                for name, table in tables.items():
                    grad = table.weight.grad.float().coalesce()
                    ids = grad.indices()[0][:512]
                    probes[name] = (ids, table.weight.detach()[ids].float().cpu().clone())
            tick = time.perf_counter()
            with region("train/optimizer"):
                optimizer.step()
            phase_ms["optimizer_host"] = (time.perf_counter()-tick)*1000
            torch.cuda.synchronize(device)
            value = sum(losses).item()
            if not torch.isfinite(torch.tensor(value)):
                raise RuntimeError("non-finite SFT loss")
            for name, (ids, before) in probes.items():
                after = tables[name].weight.detach()[ids].float().cpu()
                table_updates[name] = {"max_update": float((after-before).abs().max()),
                    "nonzero_updates": int(after.ne(before).sum()),
                    "gradient_nnz": tables[name].weight.grad._nnz()}
            return value, phase_ms

        warmup_losses = []
        for index in range(args.warmup):
            tick = time.perf_counter()
            value, _ = run_step(index, verify=index == 0)
            warmup_losses.append(value)
            progress(phase="warmup", step=index, loss=value, seconds=time.perf_counter()-tick,
                     peak_gib=torch.cuda.max_memory_allocated(device)/2**30)
        torch.cuda.reset_peak_memory_stats(device)
        for table in tables.values():
            for key in table.metrics:
                table.metrics[key] = 0
        dist.barrier()
        start = time.perf_counter()
        losses, step_ms, phases = [], [], []
        for index in range(args.warmup, args.warmup+args.steps):
            tick = time.perf_counter()
            value, phase = run_step(index)
            step_ms.append((time.perf_counter()-tick)*1000)
            losses.append(value)
            phases.append(phase)
            progress(phase="timed", step=index-args.warmup, loss=value, ms=step_ms[-1],
                     peak_gib=torch.cuda.max_memory_allocated(device)/2**30)
        elapsed = time.perf_counter() - start
        parameter_samples = {}
        for name in ("lm_head.weight", f"model.layers.{args.layers[0]}.engram.proj.weight"):
            parameter_samples[name] = local_tensor(dict(model.named_parameters())[name]).detach().flatten()[:512].float().cpu().tolist()
        table_samples = {}
        for name, table in tables.items():
            grad = table.weight.grad
            ids = grad.indices()[0][:128]
            table_samples[name] = {"ids": ids.cpu().tolist(), "values": table.weight.detach()[ids, :8].float().cpu().tolist()}
        result = {"rank": rank, "variant": args.variant, "world_size": world, "ep": args.ep, "cp": 1,
            "batch_size": args.batch_size, "microbatches": args.microbatches, "steps": args.steps,
            "sequence_length": sequence_length, "global_batch": global_batch,
            "data": data["metadata"], "engram_config": asdict(config), "policy": asdict(policy),
            "engram_parameters": sum(t.global_num_embeddings*t.embedding_dim for t in tables.values()),
            "local_parameter_memory": parameter_memory(model, optimizer),
            "load_seconds": load_seconds, "load_peak_gib": load_peak/2**30,
            "logical_read_gib": model.load_bytes_read/2**30, "load_spans": load_spans,
            "elapsed_seconds": elapsed, "step_ms": step_ms, "losses": losses, "warmup_losses": warmup_losses,
            "phase_host_ms": phases, "peak_allocated_gib": torch.cuda.max_memory_allocated(device)/2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device)/2**30,
            "cpu_max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,
            "engram_metrics": {name: dict(t.metrics) for name, t in tables.items()},
            "cache_allocated_bytes": sum(getattr(t.store, "cache", None).allocated_bytes
                for t in tables.values() if getattr(t.store, "cache", None) is not None),
            "table_updates_first_step": table_updates, "parameter_samples": parameter_samples,
            "table_samples": table_samples,
            "lm_head_sample_update": float((local_tensor(model.lm_head.weight).detach().flatten()[:4096].float().cpu()-initial).abs().max())}
        (out / f"rank-{rank}.json").write_text(json.dumps(result, indent=2) + "\n")
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        if rank == 0:
            tokens = global_batch * sequence_length * args.steps
            supervised = int(data["labels"][args.warmup*global_batch:(args.warmup+args.steps)*global_batch].ne(-100).sum())
            summary = {"variant": args.variant, "tokens_per_second": tokens/max(r["elapsed_seconds"] for r in gathered),
                "supervised_tokens_per_second": supervised/max(r["elapsed_seconds"] for r in gathered),
                "peak_allocated_gib": max(r["peak_allocated_gib"] for r in gathered),
                "peak_reserved_gib": max(r["peak_reserved_gib"] for r in gathered),
                "global_losses": [sum(r["losses"][i] for r in gathered)/world for i in range(args.steps)],
                "ranks": gathered}
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            progress(phase="summary", **{key: value for key, value in summary.items() if key != "ranks"})
        if args.trace_dir:
            dist.barrier()
            trace_path = Path(args.trace_dir)
            trace_path.mkdir(parents=True, exist_ok=True)
            # Profiles are separate from the uninstrumented throughput samples.
            capture = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=True,
                experimental_config=torch._C._profiler._ExperimentalConfig(profile_all_threads=True))
                if rank == 0 else nullcontext())
            from dsv41_train.profiling import model_regions
            with model_regions(model), capture as prof:
                run_step(args.warmup+args.steps, tracing=True)
            if rank == 0:
                from dsv41_train.profiling import summarize_profiler
                trace = trace_path / "rank-0.json"
                prof.export_chrome_trace(str(trace))
                stats = summarize_profiler(prof, 1, trace)
                (trace_path / "rank-0-summary.json").write_text(json.dumps(stats, indent=2) + "\n")
            dist.barrier()
    finally:
        if model is not None:
            for table in model.model.engram_tables.values():
                table.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
