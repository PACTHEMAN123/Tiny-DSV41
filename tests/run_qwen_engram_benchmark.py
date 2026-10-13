"""Run the offload x cache matrix in fresh distributed processes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--microbatches", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--repeat-offset", type=int, default=0)
    parser.add_argument("--nproc", type=int, default=8)
    parser.add_argument("--variants", nargs="+", default=["gpu", "cpu", "cpu_cache", "gpu_cache"])
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"arguments": vars(args), "runs": []}
    for repeat in range(args.repeat_offset, args.repeat_offset+args.repeats):
        variants = args.variants if repeat % 2 == 0 else list(reversed(args.variants))
        for variant in variants:
            folder = root / f"repeat-{repeat}-{variant}"
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                f"--nproc-per-node={args.nproc}", "tests/qwen_engram_sft_benchmark.py",
                "--model-path", args.model_path, "--data", args.data, "--output", str(folder),
                "--variant", variant, "--ep", str(args.nproc), "--batch-size", str(args.batch_size),
                "--microbatches", str(args.microbatches), "--steps", str(args.steps), "--warmup", str(args.warmup)]
            if args.profile and repeat == args.repeat_offset:
                command += ["--trace-dir", str(folder / "trace")]
            print(json.dumps({"phase": "start", "repeat": repeat, "variant": variant, "command": command}), flush=True)
            tick = time.perf_counter()
            with (root/f"repeat-{repeat}-{variant}.log").open("w") as log, \
                 (root/f"repeat-{repeat}-{variant}-gpu.jsonl").open("w") as telemetry:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                    env=os.environ.copy(), start_new_session=True)
                try:
                    while process.poll() is None:
                        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                        telemetry.write(json.dumps({"time": time.time(), "gpus": gpu.stdout.strip().splitlines()}) + "\n")
                        telemetry.flush()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            if time.perf_counter()-tick > 5400:
                                raise TimeoutError("benchmark exceeded 90 minutes")
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=20)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
            entry = {"repeat": repeat, "variant": variant, "returncode": process.returncode,
                     "seconds": time.perf_counter()-tick, "output": str(folder)}
            manifest["runs"].append(entry)
            manifest_name = f"manifest-{args.repeat_offset}.json" if args.repeat_offset else "manifest.json"
            (root/manifest_name).write_text(json.dumps(manifest, indent=2) + "\n")
            print(json.dumps({"phase": "finished", **entry}), flush=True)
            if process.returncode:
                raise RuntimeError(f"benchmark failed: {folder}")


if __name__ == "__main__":
    main()
