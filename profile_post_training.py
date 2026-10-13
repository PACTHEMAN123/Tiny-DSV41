"""Run reproducible LoRA/full-training architecture comparisons in fresh processes."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/architecture-profile")
    parser.add_argument("--config", default="configs/profile_dsv41_medium.json")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--fsdp-gpus", help="optional two-GPU visibility list, e.g. 0,1")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--only", help="regular expression selecting case names")
    parser.add_argument("--no-traces", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = (root / args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / args.config).read_text(encoding="utf-8"))
    variants = {"base": config, "mhc1": {**config, "hc_sinkhorn_iters": 1},
                "experts8": {**config, "n_routed_experts": 8}}
    for name, values in variants.items():
        (output / f"config-{name}.json").write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")
    cases = []
    for length in (512, 2048, 4096):
        for lora in (False, True):
            cases.append(("base", length, lora, "none", "loop", 1, True))
    for lora in (False, True):
        cases.extend([
            ("base", 4096, lora, "selective", "loop", 1, False),
            ("base", 2048, lora, "none", "fused", 1, False),
            ("base", 2048, lora, "none", "foreach", 1, False),
            ("mhc1", 2048, lora, "none", "loop", 1, False),
            ("experts8", 2048, lora, "none", "loop", 1, False),
        ])
        if args.fsdp_gpus:
            cases.append(("base", 2048, lora, "none", "loop", 2, True))
            cases.append(("base", 2048, lora, "none", "foreach", 2, False))
    reports = []
    for variant, length, lora, ac, optimizer, world, trace in cases:
        name = f"{variant}-s{length}-{'lora' if lora else 'full'}-{ac}-{optimizer}-gpu{world}"
        if args.only and not re.search(args.only, name):
            continue
        command = [sys.executable]
        if world > 1:
            command += ["-m", "torch.distributed.run", "--standalone", f"--nproc-per-node={world}"]
        command += [str(root / "profile_dsv4.py"), "--config", str(output / f"config-{variant}.json"),
                    "--seq-len", str(length), "--warmup", str(args.warmup), "--steps", str(args.steps),
                    "--checkpoint-mode", ac, "--adamw-implementation", optimizer,
                    "--output", str(output / f"{name}.json")]
        if lora:
            command.append("--lora")
        if trace and not args.no_traces:
            command += ["--trace-dir", str(output / "traces" / name)]
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": args.fsdp_gpus if world > 1 else args.gpu}
        print(json.dumps({"starting": name, "command": command}), flush=True)
        started = time.monotonic()
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=1800)
        report = json.loads((output / f"{name}.json").read_text(encoding="utf-8"))
        report["case"] = name
        report["operator_summaries"] = [
            json.loads(Path(rank["operator_summary"]).read_text(encoding="utf-8"))
            for rank in report["per_rank"] if rank["operator_summary"]
        ]
        reports.append(report)
        (output / "results.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"completed": name, "seconds": time.monotonic() - started,
                          "timing": report["timing"],
                          "peak_mib": [rank["peak_allocated_mib"] for rank in report["per_rank"]]}), flush=True)


if __name__ == "__main__":
    main()
