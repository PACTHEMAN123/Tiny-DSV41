"""Create a complete, pinned KernelBench workload without executing references."""
import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from rollout_manager.adapters.candidates import prompt
from rollout_manager.adapters.datasets import DATASETS, load_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4.1-Flash")
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()
    manifest = json.loads((args.assets / "manifest.json").read_text())
    tasks = load_tasks(args.assets / "datasets", "kernelbench")
    counts = Counter(task.split for task in tasks)
    if counts != {"level_1": 100, "level_2": 100, "level_3": 50, "level_4": 20}:
        raise ValueError("pinned full-suite coverage changed")
    cases = []
    for task in tasks:
        item = asdict(task)
        cases.append({"task": item, "files": {}, "request": {"model": args.model,
            "messages": prompt(item, architecture=manifest["architecture"]),
            "temperature": args.temperature, "top_p": .95, "max_tokens": args.max_tokens,
            "seed": 42, "chat_template_kwargs": {"enable_thinking": False}}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(cases, stream, indent=2)
        stream.write("\n")
    metadata = {"dataset": DATASETS["kernelbench"], "counts": counts, "tasks": len(tasks),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "selection": "all pinned tasks; no correctness or resource-based exclusion",
        "sampling": {"temperature": args.temperature, "top_p": .95, "max_tokens": args.max_tokens},
        "native_correctness_trials": 5, "native_performance_trials": 10,
        "development_seed": 42, "final_seed": 1042}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
