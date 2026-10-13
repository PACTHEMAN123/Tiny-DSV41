"""Run memory acceptance or benchmarks with isolated process groups per case."""

import argparse
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys

from dsv4_memory_benchmark import VARIANTS
from memory_validation_cases import validation_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("cpu", "cuda", "benchmark"), required=True)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--all-topologies", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases", help="comma-separated case IDs or benchmark variants")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--cp", type=int, default=1)
    parser.add_argument("--ep", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=.01)
    parser.add_argument("--input-pattern", choices=("fixed", "fresh", "repeated"), default="fixed")
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--cache-mib", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model-path", help="reuse a prepared checkpoint without constructing full CPU weights")
    parser.add_argument("--load-mode", choices=("layerwise", "direct_shard"))
    parser.add_argument("--load-pipeline", action="store_true")
    parser.add_argument("--staging-mib", type=int, default=8)
    parser.add_argument("--activation-checkpoint", choices=("none", "full", "selective"), default="none")
    parser.add_argument("--memory-limit-gib", type=float, default=0)
    parser.add_argument("--distributed-timeout", type=int, default=120)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--profile-record-shapes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--config", help="shared synthetic benchmark checkpoint configuration")
    parser.add_argument("--trace-variants", default="", help="comma-separated benchmark variants to profile on repeat 0")
    parser.add_argument("--load-trace-variants", default="", help="comma-separated loading variants to profile on repeat 0; load time includes profiler overhead")
    parser.add_argument("--prefetch-steps", type=int, default=0,
                        help="post-timing CPU Engram readiness diagnostic steps")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    folder = Path(args.output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if args.suite == "benchmark":
        cases = list(VARIANTS)
    else:
        cases = [str(index) for index, _ in enumerate(validation_cases(args.ranks, args.all_topologies))]
    if args.cases:
        selected = args.cases.split(",")
        if set(selected) - set(cases):
            parser.error("unknown cases")
        cases = selected
    if args.repeats < 1 or args.timeout < 1:
        parser.error("repeats and timeout must be positive")
    traces = set(filter(None, args.trace_variants.split(",")))
    load_traces = set(filter(None, args.load_trace_variants.split(",")))
    if (traces | load_traces) - set(VARIANTS):
        parser.error("unknown trace variants")
    model_path = Path(args.model_path).resolve() if args.model_path else None
    if model_path is not None:
        if not (model_path / "model.safetensors.index.json").is_file():
            parser.error("model-path must contain a prepared indexed checkpoint")
        if args.config:
            parser.error("model-path and config are mutually exclusive")
    if args.suite == "benchmark" and args.config and model_path is None:
        from dsv41_train.models.dsv4.config import DeepSeekV41Config
        from test_dsv4_loading import synthetic_checkpoint
        import torch
        model_path = folder / "fixture"
        config = DeepSeekV41Config.from_json(args.config)
        if (model_path / "model.safetensors.index.json").is_file():
            if DeepSeekV41Config.from_json(model_path / "config.json") != config:
                raise ValueError("existing benchmark fixture has a different configuration")
        else:
            model_path.mkdir(parents=True, exist_ok=True)
            torch.set_num_threads(1)
            torch.manual_seed(42)
            synthetic_checkpoint(model_path, config)
    schedule = []
    for repeat in range(args.repeats):
        order = list(cases)
        if args.suite == "benchmark" and args.repeats > 1:
            random.Random(1729 + repeat).shuffle(order)
        schedule.extend((repeat, case) for case in order)
    (folder / "manifest.json").write_text(json.dumps({
        "arguments": vars(args), "schedule": [{"repeat": repeat, "case": case} for repeat, case in schedule]
    }, indent=2) + "\n")
    summary = []
    (folder / "summary.json").write_text("[]\n")
    for repeat, case in schedule:
        label = f"{case}-r{repeat}" if args.repeats > 1 else case
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc-per-node={args.ranks}"]
        if args.suite == "benchmark":
            command += ["tests/dsv4_memory_benchmark.py", "--variant", case,
                        "--steps", str(args.steps), "--warmup", str(args.warmup),
                        "--seq-len", str(args.seq_len), "--cp", str(args.cp), "--ep", str(args.ep),
                        "--learning-rate", str(args.learning_rate), "--prefetch-steps", str(args.prefetch_steps)]
            command += ["--input-pattern", args.input_pattern, "--microbatches", str(args.microbatches)]
            command += ["--cache-mib", str(args.cache_mib)]
            command += ["--batch-size", str(args.batch_size), "--staging-mib", str(args.staging_mib),
                        "--activation-checkpoint", args.activation_checkpoint,
                        "--memory-limit-gib", str(args.memory_limit_gib),
                        "--distributed-timeout", str(args.distributed_timeout)]
            if args.load_mode:
                command += ["--load-mode", args.load_mode]
            if args.load_pipeline:
                command += ["--load-pipeline"]
            if args.progress:
                command += ["--progress"]
            if not args.profile_record_shapes:
                command += ["--no-profile-record-shapes"]
            if model_path is not None:
                command += ["--model-path", str(model_path)]
            if repeat == 0 and case in traces:
                command += ["--trace-dir", str(folder / "traces" / case)]
            if repeat == 0 and case in load_traces:
                command += ["--load-trace-dir", str(folder / "loading" / case)]
        else:
            command += ["tests/dsv4_memory_parallel.py", "--case", case]
            if args.suite == "cpu":
                command += ["--cpu"]
            if args.all_topologies:
                command += ["--all-topologies"]
            if args.bf16:
                command += ["--bf16"]
        report = folder / f"{label}.json"
        command += ["--output", str(report)]
        print(json.dumps({"start": label, "suite": args.suite}), flush=True)
        with (folder / f"{label}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=args.timeout)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        summary.append({"case": case, "repeat": repeat, "exit_code": code, "report": str(report)})
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary[-1]), flush=True)
        if code and not args.keep_going:
            raise RuntimeError(f"case {label} failed; see {folder / (label + '.log')}")
    if any(item["exit_code"] for item in summary):
        raise RuntimeError(f"some cases failed; see {folder / 'summary.json'}")


if __name__ == "__main__":
    main()
