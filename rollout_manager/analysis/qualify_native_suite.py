"""Run each pinned reference against itself to qualify the native environment.

This is explicitly not a generated-candidate experiment or performance score.
It discovers missing dependencies, unsupported references and native timeouts
before treating such failures as generated-candidate outcomes.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

from rollout_manager.adapters.harnesses import HarnessRunner, prepare_workspace
from rollout_manager.adapters.native_timing import HarnessInstrumentationError
from rollout_manager.host_memory import ByteBudget, available_bytes, evaluator_estimates, evaluator_memory
from rollout_manager.runner import Journal, check_headroom, reservation
from rollout_manager.gpu_ownership import ReviewedGpuOwners
from rollout_manager.telemetry import monitor
from rollout_manager.preflight import preflight


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--input-memory-plan", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task-ids", nargs="+", help="explicit focused qualification rerun, not cohort selection")
    parser.add_argument("--preserve-integer-inputs", action="store_true")
    parser.add_argument("--final-verify", action="store_true", help="fresh native final verification at seed 1042")
    parser.add_argument("--cores", default="8,9,10,11,12,13,14,15")
    parser.add_argument("--host-gib", type=int, default=128)
    parser.add_argument("--other-reserved-gib", type=int, default=0,
                        help="host-memory allowance reserved for other concurrent work")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.assets = args.assets.resolve()
    args.root.mkdir(parents=True, exist_ok=False)
    cpus = list(map(int, args.cores.split(",")))
    if len(set(cpus)) != len(cpus) or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("qualification CPU affinity is outside the allocation")
    gpu_ids = preflight(args.root)
    if len(gpu_ids) != len(cpus):
        raise ValueError("assign one qualified CPU and one exclusive slot per GPU")
    os.sched_setaffinity(0, cpus)
    cases = json.loads(args.cases.read_text())
    cohort_tasks = len(cases)
    if args.task_ids:
        selected = set(args.task_ids)
        if not selected <= {case["task"]["task_id"] for case in cases}:
            raise ValueError("qualification task selection is outside the frozen cohort")
        cases = [case for case in cases if case["task"]["task_id"] in selected]
    manifest = json.loads((args.assets / "manifest.json").read_text())
    for relative, expected in manifest["upstream_files"].items():
        if hashlib.sha256((args.assets / relative).read_bytes()).hexdigest() != expected:
            raise ValueError("upstream digest mismatch: " + relative)
    estimates = evaluator_estimates(cases, json.loads(args.input_memory_plan.read_text()),
                                    json.loads(args.model_manifest.read_text()))
    if max(x["bytes"] for x in estimates.values()) > args.host_gib*1024**3:
        raise ValueError("one task cannot fit evaluator host admission")
    if (args.host_gib + args.other_reserved_gib + 32)*1024**3 > available_bytes():
        raise ValueError("concurrent qualification memory allowances exceed available RAM")
    settings = {"kind": "native_reference_qualification_not_rollout_performance", "tasks": len(cases),
        "cohort_tasks": cohort_tasks, "selected_task_ids": args.task_ids,
        "preserve_integer_inputs": args.preserve_integer_inputs,
        "final_verify": args.final_verify,
        "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(), "cpu_ids": cpus,
        "gpu_uuids": gpu_ids, "host_budget_gib": args.host_gib, "other_reserved_gib": args.other_reserved_gib,
        "seed": 42, "correctness_trials": 5, "performance_trials": 10,
        "generated_candidates": False, "candidate_wrapper": "from model import Model\nclass ModelNew(Model):\n    pass\n"}
    (args.root / "settings.json").write_text(json.dumps(settings, indent=2)+"\n")
    journal = Journal(args.root)
    credits = ByteBudget(args.host_gib*1024**3, journal.event)
    gpus = queue.Queue()
    for gpu, cpu in zip(gpu_ids, cpus):
        gpus.put((gpu, cpu))
    owners = ReviewedGpuOwners(settle_seconds=30)
    fatal, stop, lock = threading.Event(), threading.Event(), threading.Lock()
    results = []
    telemetry = threading.Thread(target=monitor, args=(args.root / "telemetry-B.jsonl", stop),
        kwargs={"extra": lambda: {"queues": journal.snapshot(), "evaluator_host": credits.snapshot(),
                                  "gpu_free_slots": gpus.qsize()}}, daemon=True)
    telemetry.start()

    def evaluate_once(workspace, task, job, final=False):
        stage = "verification" if final else "evaluation"
        with evaluator_memory(credits, estimates[task["task_id"]], journal, job, stage, fatal.is_set) as host:
            with reservation(gpus, journal, job, stage, fatal.is_set) as ((gpu, core), timing):
                try:
                    with owners.lease(gpu, workspace) as guard:
                        runner = HarnessRunner(args.assets / "upstream", gpu, manifest["architecture"],
                            check_headroom, cores=[core], gpu_guard=guard,
                            preserve_integer_inputs=args.preserve_integer_inputs)
                        result = runner.evaluate(workspace, "kernelbench", observe_inputs=True, final=final)
                        if not result["process_group_stopped"] or not result["gpu_contexts_released"]:
                            raise RuntimeError("qualification cleanup is uncertain")
                    return result, gpu
                except BaseException:
                    timing["quarantined"] = host["quarantined"] = True
                    fatal.set()
                    raise

    def execute(case):
        if fatal.is_set():
            return
        task = case["task"]
        job = task["task_id"].replace("/", "-")
        workspace = args.root / "jobs" / job
        begin = time.monotonic()
        record = {"task_id": task["task_id"], "started_unix_ns": time.time_ns()}
        try:
            prepare_workspace(workspace, args.assets / "upstream", task,
                              {"model_new.py": settings["candidate_wrapper"]})
            result, gpu = evaluate_once(workspace, task, job)
            record.update(result=result, passed=result["correct"], gpu_uuid=gpu)
            if args.final_verify and result["correct"]:
                final, final_gpu = evaluate_once(workspace, task, job, final=True)
                record.update(final_result=final, final_gpu_uuid=final_gpu, passed=final["correct"])
        except Exception as exc:
            if isinstance(exc, HarnessInstrumentationError):
                fatal.set()
            record.update(passed=False, error_type=type(exc).__name__, error=str(exc))
        finally:
            record.update(wall_seconds=time.monotonic()-begin, finished_unix_ns=time.time_ns())
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "qualification.json").write_text(json.dumps(record, indent=2)+"\n")
            with lock:
                results.append(record)
                print(json.dumps({"finished": len(results), "expected": len(cases), "task": task["task_id"],
                                  "passed": record["passed"], "wall_seconds": record["wall_seconds"]}), flush=True)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpu_ids)) as executor:
            list(executor.map(execute, cases))
    finally:
        stop.set()
        telemetry.join(timeout=10)
        journal.close()
    summary = {"kind": settings["kind"], "completed": len(results), "expected": len(cases),
        "passed": sum(x["passed"] for x in results), "fatal": fatal.is_set(), "results": results,
        "host_memory_final": credits.snapshot(), "ownership_checks": owners.checks}
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps({k: summary[k] for k in ("kind", "completed", "expected", "passed", "fatal")}), flush=True)
    return 0 if len(results) == len(cases) and all(x["passed"] for x in results) and not fatal.is_set() else 1


if __name__ == "__main__":
    raise SystemExit(main())
