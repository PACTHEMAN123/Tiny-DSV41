"""Native CUDA integration checks, separate from measured rollout performance."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import tempfile
import threading
from pathlib import Path

from rollout_manager.adapters.harnesses import HarnessRunner, prepare_workspace
from rollout_manager.common import save_json
from rollout_manager.gpu_ownership import ReviewedGpuOwners
from rollout_manager.host_memory import available_bytes, evaluator_estimates
from rollout_manager.input_cache import InputCache
from rollout_manager.runner import check_headroom
from rollout_manager.preflight import preflight


WRAPPER = "from model import Model\nclass ModelNew(Model):\n    pass\n"


def check_result(result, enabled, warm):
    if not result["process_group_stopped"] or not result["gpu_contexts_released"]:
        raise RuntimeError("CUDA integration process ownership did not drain")
    if not result["correct"]:
        raise ValueError("native self-reference CUDA check failed")
    stages = result["native"]["stage_metrics"]
    if stages["input_calls"] != 7 or stages["logical_input_indices"] != list(range(7)):
        raise ValueError("native CUDA input/trial sequence changed")
    if enabled:
        reuse = stages["input_reuse"]
        if reuse["bypasses"] or (warm and reuse["hits"] != 7):
            raise ValueError("CUDA integration did not exercise all seven warm cache hits")
        return [event["key"] for event in reuse["events"]]
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--input-memory-plan", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="+", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--cache-gib", type=int, default=128)
    args = parser.parse_args()
    root, assets = args.root.resolve(), args.assets.resolve()
    root.mkdir(parents=True, exist_ok=False)
    gpus = preflight(root)
    if len(gpus) != 8:
        raise ValueError("all eight B GPUs must be idle before integration checks")
    allowed = sorted(os.sched_getaffinity(0))
    if len(allowed) < 8:
        raise ValueError("eight allowed CPU IDs required")
    cases = {x["task"]["task_id"]: x for x in json.loads(args.cases.read_text())}
    if len(set(args.task_ids)) != len(args.task_ids) or not set(args.task_ids) <= cases.keys():
        raise ValueError("integration task IDs must be distinct members of the frozen cohort")
    selected = [cases[key] for key in args.task_ids]
    registry = json.loads(args.input_memory_plan.read_text())
    plans = registry.get("plans", registry.get("contracts", {}))
    qualified = json.loads(args.qualification.read_text())
    if qualified["passed"] is not True:
        raise ValueError("full CPU generator qualification must pass first")
    estimates = evaluator_estimates(selected, registry, json.loads(args.model_manifest.read_text()))
    host_bound = max(x["bytes"] for x in estimates.values())
    if host_bound + (args.cache_gib+32)*1024**3 > available_bytes():
        raise ValueError("integration memory allowances exceed currently available RAM")
    manifest = json.loads((assets/"manifest.json").read_text())
    os.environ["HF_HOME"] = str(args.model_manifest.resolve().parent/"hf-home")
    os.environ["HF_HUB_OFFLINE"] = "1"
    for relative, expected in manifest["upstream_files"].items():
        if hashlib.sha256((assets/relative).read_bytes()).hexdigest() != expected:
            raise ValueError("native harness source changed: " + relative)
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    owners = ReviewedGpuOwners(settle_seconds=30)
    results = []
    save_json(root/"settings.json", {"kind": "native_cuda_input_reuse_integration_not_performance",
        "task_ids": args.task_ids, "gpu_uuids": gpus, "cache_gib": args.cache_gib,
        "phases": ["off", "cold", "warm"], "seeds": [42, 1042], "candidate_wrapper": WRAPPER,
        "qualification_sha256": hashlib.sha256(args.qualification.read_bytes()).hexdigest()})
    with tempfile.TemporaryDirectory(prefix="reuse-cuda-", dir="/tmp") as temporary:
        cache = InputCache(Path(temporary)/"c", ram_bytes=args.cache_gib*1024**3)
        try:
            for index, case in enumerate(selected):
                task = case["task"]
                plan = plans.get(task["source_sha256"])
                gpu, cpu = gpus[index % 8], allowed[index % 8]
                if plan is not None and 14*plan.get("allocation_bytes", plan.get("arena_bytes")) > args.cache_gib*1024**3:
                    raise ValueError("selected smoke task cannot retain both seed namespaces in this cache")
                cache.clear()
                for phase in ("off", "cold", "warm"):
                    if stopping.is_set():
                        raise RuntimeError("integration checks cancelled")
                    workspace = root/task["task_id"].replace("/", "-")/phase
                    prepare_workspace(workspace, assets/"upstream", task, {"model_new.py": WRAPPER})
                    config = None
                    if phase != "off" and plan is not None:
                        config = {"socket": str(cache.socket_path), "plan": plan,
                            "consumer_copy_limit_bytes": estimates[task["task_id"]]["input_and_transfer_bytes"],
                            "qualifications": {"kernelbench": qualified["tasks"][task["task_id"]]}}
                    keys = []
                    for final in (False, True):
                        with owners.lease(gpu, workspace) as guard:
                            runner = HarnessRunner(assets/"upstream", gpu, manifest["architecture"], check_headroom,
                                cores=[cpu], gpu_guard=guard, preserve_integer_inputs=True, cancelled=stopping.is_set)
                            result = runner.evaluate(workspace, "kernelbench", final=final,
                                                     observe_inputs=True, input_reuse=config)
                        keys.append(check_result(result, config is not None, phase == "warm"))
                        snap = cache.snapshot()
                        if snap["consumers"] or snap["references"] or snap["generating_bytes"]:
                            raise ValueError("input cache ownership did not drain after CUDA evaluation")
                        record = {"task_id": task["task_id"], "phase": phase, "final": final,
                                  "cache_enabled": config is not None, "result": result, "cache": snap}
                        results.append(record)
                        save_json(root/"progress.json", {"completed": len(results), "results": results})
                        print(json.dumps({"task": task["task_id"], "phase": phase, "final": final,
                                          "passed": True, "completed": len(results)}), flush=True)
                    if set(keys[0]) & set(keys[1]):
                        raise ValueError("development and final-verification cache keys overlap")
            save_json(root/"summary.json", {"passed": True, "completed": len(results), "results": results,
                "cache_final": cache.snapshot(), "ownership_checks": owners.checks})
        finally:
            if not cache.snapshot()["consumers"]:
                cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
