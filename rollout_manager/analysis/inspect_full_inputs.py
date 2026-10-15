"""Estimate native input memory without materializing tensors; record unsupported tasks."""
import argparse
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from rollout_manager.adapters.datasets import load_tasks
from rollout_manager.adapters.input_memory import make_memory_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    args = parser.parse_args()
    plans, tasks = {}, []
    references = ([SimpleNamespace(**case["task"]) for case in json.loads(args.cases.read_text())]
                  if args.cases else load_tasks(args.assets / "datasets", "kernelbench"))
    for task in references:
        item = {"task_id": task.task_id, "source_sha256": task.source_sha256}
        try:
            function = next(n for n in ast.parse(task.code).body if isinstance(n, ast.FunctionDef) and n.name == "get_inputs")
            if any(isinstance(n, ast.keyword) and n.arg == "device" and
                   isinstance(n.value, ast.Constant) and str(n.value.value).startswith("cuda")
                   for n in ast.walk(function)):
                item["native_device_generator"] = True
                raise ValueError("native input generator explicitly uses CUDA RNG")
            spec = make_memory_plan(task.code)
            plans[task.source_sha256] = spec
            item.update(memory_plan_available=True, allocation_bytes=spec["allocation_bytes"],
                        input_bytes=spec["input_bytes"], operations=[x["op"] for x in spec["operations"]])
        except Exception as exc:
            item.update(memory_plan_available=False, error_type=type(exc).__name__, reason=str(exc))
        tasks.append(item)
        print(json.dumps(item), flush=True)
    with args.output.open("x") as stream:
        json.dump({"plans": plans, "tasks": tasks,
            "kind": "native input memory estimate, not correctness evidence or an OS memory limit"}, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
