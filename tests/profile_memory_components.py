"""CPU-only fallback profiling for checkpoint loading and local Engram execution."""

import argparse
from contextlib import nullcontext
from copy import deepcopy
import gc
import json
import math
from pathlib import Path
import random
import resource
from statistics import median
import subprocess
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from dsv41_train.layout import parameter_memory
from dsv41_train.loading import LoadPolicy
from dsv41_train.loading import source as loading_source
from dsv41_train.models.dsv4 import load_dsv41_backbone_window
from dsv41_train.models.dsv4.config import DeepSeekV41Config
from dsv41_train.models.dsv4.engram import EngramPolicy
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from dsv41_train.profiling import loading_regions, model_regions, summarize_profiler
from test_dsv4_loading import synthetic_checkpoint
from test_memory_modules import LocalMaterializer


VARIANTS = ("layerwise", "direct", "direct_pipeline", "async", "dedup", "foreach")


def worker(args):
    folder = Path(args.output_dir)
    variant = args.worker
    direct = variant in {"direct", "direct_pipeline"}
    policy = LoadPolicy("direct_shard" if direct else "layerwise", staging_bytes=8 * 1024**2,
                        pipeline=variant == "direct_pipeline")
    config = DeepSeekV41Config.from_json(folder / "fixture/config.json")

    def load():
        return load_dsv41_backbone_window(folder / "fixture", num_layers=config.num_hidden_layers,
            device="cpu", dtype=torch.bfloat16, sparse_engram_gradients=True, load_policy=policy,
            materializer=LocalMaterializer() if direct else None,
            engram_policy=EngramPolicy("cpu", "async" if variant == "async" else "sync", variant == "dedup"))

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    started = time.perf_counter()
    model = load()
    first_load_ms = (time.perf_counter() - started) * 1000
    optimizer = build_optimizer(model, OptimizerPolicy("sgd", .0001, variant == "foreach"))
    generator = torch.Generator().manual_seed(123)
    batches = [torch.randint(3, config.vocab_size, (1, args.seq_len), generator=generator) for _ in range(2)]

    def step(tracing=False):
        region = record_function if tracing else lambda _: nullcontext()
        durations = {"forward": 0., "backward": 0.}
        begin = time.perf_counter()
        with region("train/zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        for tokens in batches:
            started = time.perf_counter()
            with region("train/forward"):
                output = model(tokens, labels=tokens)
            durations["forward"] += (time.perf_counter() - started) * 1000
            started = time.perf_counter()
            with region("train/backward"):
                (output.loss / len(batches)).backward()
            durations["backward"] += (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        with region("train/optimizer"):
            model.synchronize_engram()
            optimizer.step()
        durations["optimizer"] = (time.perf_counter() - started) * 1000
        durations["step"] = (time.perf_counter() - begin) * 1000
        loss = output.loss.detach().item()
        if not math.isfinite(loss):
            raise RuntimeError("nonfinite training loss")
        return durations, loss

    report = None
    try:
        for _ in range(args.warmup):
            step()
        for table in model.model.engram_tables.values():
            for key in table.metrics:
                table.metrics[key] = 0
        timings, losses = [], []
        for _ in range(args.steps):
            timing, loss = step()
            timings.append(timing)
            losses.append(loss)
        report = {"variant": variant, "repeat": args.repeat, "device": "cpu", "world": 1,
                  "loading_implementation": loading_source.__file__,
                  "torch": torch.__version__, "threads": args.threads, "config": config.to_dict(),
                  "steps": args.steps, "warmup": args.warmup, "sequence_length": args.seq_len,
                  "microbatches": 2, "learning_rate": .0001, "dtype": "bfloat16",
                  "parameters": sum(parameter.numel() for parameter in model.parameters()),
                  "first_load_ms": first_load_ms, "loading": model.loading_metrics,
                  "timing_ms": {key: median(row[key] for row in timings) for key in timings[0]},
                  "samples_ms": timings, "losses": losses,
                  "tokens_per_second": args.steps * 2 * args.seq_len / sum(row["step"] for row in timings) * 1000,
                  "engram": {name: deepcopy(table.metrics) for name, table in model.model.engram_tables.items()},
                  "parameter_bytes": parameter_memory(model, optimizer),
                  "cpu_max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                  "limitations": ["CPU only: no H2D/D2H, NCCL or FSDP measurement",
                                   "Direct loader fills full local tensors without distributed sharding",
                                   "Filesystem pages may be cached; logical reads are not disk I/O"]}
        if args.repeat == 0:
            trace_dir = folder / "traces" / variant
            trace_dir.mkdir(parents=True, exist_ok=True)
            spans = []
            with model_regions(model, spans), profile(activities=[ProfilerActivity.CPU],
                    record_shapes=True, profile_memory=True,
                    experimental_config=torch._C._profiler._ExperimentalConfig(profile_all_threads=True)) as profiler:
                step(tracing=True)
            trace = trace_dir / "training.json"
            profiler.export_chrome_trace(str(trace))
            summary = summarize_profiler(profiler, 1, trace)
            summary["lookup_spans"] = spans
            (trace_dir / "training-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            report["training_profile"] = str(trace_dir / "training-summary.json")
    finally:
        model.close()
    del model, optimizer
    gc.collect()
    started = time.perf_counter()
    model = load()
    report["warm_load_ms"] = (time.perf_counter() - started) * 1000
    model.close()
    del model
    gc.collect()
    if args.repeat == 0:
        spans = []
        with loading_regions(spans), profile(activities=[ProfilerActivity.CPU],
                record_shapes=True, profile_memory=True,
                experimental_config=torch._C._profiler._ExperimentalConfig(profile_all_threads=True)) as profiler:
            started = time.perf_counter()
            model = load()
            report["profiled_load_ms"] = (time.perf_counter() - started) * 1000
        model.close()
        trace = trace_dir / "loading.json"
        profiler.export_chrome_trace(str(trace))
        summary = summarize_profiler(profiler, 1, trace)
        summary["loading_spans"] = spans
        (trace_dir / "loading-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        report["loading_profile"] = str(trace_dir / "loading-summary.json")
    label = f"{variant}-r{args.repeat}"
    (folder / f"{label}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"variant": variant, "repeat": args.repeat, "first_load_ms": first_load_ms,
                      "warm_load_ms": report["warm_load_ms"], "timing_ms": report["timing_ms"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/profile_memory_small.json")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--worker", choices=VARIANTS)
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()
    if min(args.threads, args.steps, args.warmup, args.repeats) < 1 or args.seq_len < 2:
        parser.error("positive threads, steps, warmup, repeats and seq-len >= 2 required")
    if args.worker:
        worker(args)
        return
    variants = args.variants.split(",")
    if set(variants) - set(VARIANTS):
        parser.error("unknown variants")
    folder = Path(args.output_dir)
    fixture = folder / "fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    config = DeepSeekV41Config.from_json(args.config)
    if (fixture / "model.safetensors.index.json").exists():
        if DeepSeekV41Config.from_json(fixture / "config.json") != config:
            raise ValueError("existing fixture has a different configuration")
    else:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        synthetic_checkpoint(fixture, config)
    schedule = []
    for repeat in range(args.repeats):
        order = list(variants)
        random.Random(1729 + repeat).shuffle(order)
        schedule.extend({"case": variant, "repeat": repeat} for variant in order)
    (folder / "manifest.json").write_text(json.dumps({"arguments": vars(args), "schedule": schedule}, indent=2) + "\n")
    completed = []
    for item in schedule:
        label = f"{item['case']}-r{item['repeat']}"
        command = [sys.executable, __file__, "--worker", item["case"], "--repeat", str(item["repeat"]),
                   "--output-dir", str(folder), "--seq-len", str(args.seq_len), "--threads", str(args.threads),
                   "--steps", str(args.steps), "--warmup", str(args.warmup)]
        print(json.dumps({"start": label}), flush=True)
        with (folder / f"{label}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        completed.append({**item, "exit_code": result.returncode})
        (folder / "summary.json").write_text(json.dumps(completed, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"{label} failed; see its log")
        print(json.dumps({"finished": label}), flush=True)


if __name__ == "__main__":
    main()
