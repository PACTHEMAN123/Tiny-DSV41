"""Run memory acceptance or benchmarks with isolated process groups per case."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from dsv4_memory_benchmark import VARIANTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("cpu", "cuda", "benchmark"), required=True)
    parser.add_argument("--ranks", type=int, default=2)
    parser.add_argument("--all-topologies", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cases", help="comma-separated case IDs or benchmark variants")
    args = parser.parse_args()
    folder = Path(args.output_dir).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if args.suite == "benchmark":
        cases = list(VARIANTS)
    elif args.all_topologies:
        cases = [str(index) for index, _ in enumerate(
            [(cp, ep) for cp in range(1, args.ranks+1) for ep in range(1, args.ranks+1)
             if args.ranks % cp == args.ranks % ep == 0 and 12 % cp == 6 % ep == 0])]
    else:
        cases = [str(index) for index in range(2 if args.ranks == 6 else 5)]
    if args.cases:
        selected = args.cases.split(",")
        if set(selected) - set(cases):
            parser.error("unknown cases")
        cases = selected
    summary = []
    for case in cases:
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc-per-node={args.ranks}"]
        if args.suite == "benchmark":
            command += ["tests/dsv4_memory_benchmark.py", "--variant", case]
        else:
            command += ["tests/dsv4_memory_parallel.py", "--case", case]
            if args.suite == "cpu":
                command += ["--cpu"]
            if args.all_topologies:
                command += ["--all-topologies"]
            if args.bf16:
                command += ["--bf16"]
        command += ["--output", str(folder / f"{case}.json")]
        print(json.dumps({"start": case, "suite": args.suite}), flush=True)
        with (folder / f"{case}.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=300)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
        summary.append({"case": case, "exit_code": code})
        (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary[-1]), flush=True)
        if code:
            raise RuntimeError(f"case {case} failed; see {folder / (case + '.log')}")


if __name__ == "__main__":
    main()
