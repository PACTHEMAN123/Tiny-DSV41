"""Collect reproducible local acceptance artifacts into a portable JSON report."""

import argparse
import datetime
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def matrix(folder, cases):
    reports = []
    for case in range(cases):
        ranks = read(folder / f"{case}.json")
        report = {key: ranks[0][key] for key in
                  ("case", "cp", "ep", "mode", "reference", "dtype", "storage", "prefetch", "offload")}
        report.update(world=len(ranks), max_gradient_error=max(rank["max_gradient_error"] for rank in ranks))
        reports.append(report)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", default="artifacts")
    parser.add_argument("--output", default="docs/results/memory-offload-h20.json")
    args = parser.parse_args()
    root = Path(args.artifacts)
    benchmark_status = read(root / "memory-bench-final/summary.json")
    if len(benchmark_status) != 10 or any(item["exit_code"] for item in benchmark_status):
        raise RuntimeError("the complete benchmark must pass before summarizing")
    benchmarks = [read(root / "memory-bench-final" / (item["case"] + ".json")) for item in benchmark_status]
    report = {
        "date": str(datetime.date.today()),
        "scope": "Synthetic quantized DSV4 checkpoints on one NVIDIA H20 host; no full-model capacity claim.",
        "checkpoint_format": 2,
        "cpu_six_rank_matrix": matrix(root / "memory-cpu-final", 16),
        "cuda_six_rank_matrix": matrix(root / "memory-gpu-topologies", 16),
        "cuda_two_rank_bf16": matrix(root / "memory-gpu-bf16-final", 5),
        "cuda_six_rank_bf16_cross": matrix(root / "memory-cross-bf16-final", 2),
        "engram_lifecycle": read(root / "memory-engram-lifecycle.json"),
        "fresh_process_resume": [read(root / f"memory-{name}-resume-final/resume-result.json")
                                 for name in ("sgd", "adamw")],
        "lora_full_fsdp_regression": read(root / "fsdp-parity.json"),
        "training_entrypoints": {},
        "benchmarks": benchmarks,
    }
    for name in ("cli-save", "cli-resume", "lora-save", "lora-resume"):
        value = read(root / f"memory-{name}.json")
        report["training_entrypoints"][name] = {key: value[key] for key in
            ("world_size", "cp_size", "ep_size", "steps", "gradient_accumulation_steps",
             "checkpoint_mode", "lora", "loss", "parameter_delta_max", "engram_policy", "fsdp_policy")}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    for value in benchmarks:
        ranks = value["ranks"]
        print(f"{value['variant']}: {value['tokens_per_second']:.2f} tokens/s; "
              f"load {max(rank['load_peak_bytes'] for rank in ranks) / 1024**2:.2f} MiB; "
              f"train {max(rank['train_peak_bytes'] for rank in ranks) / 1024**2:.2f} MiB")


if __name__ == "__main__":
    main()
