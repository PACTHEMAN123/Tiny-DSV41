"""Full-size equivalence of unchanged native CPU generators and cached storage."""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import hashlib
import json
import os
import random
import signal
import sys
import tempfile
import threading
import time
from pathlib import Path

from rollout_manager.common import save_json
from rollout_manager.adapters.process import run_process
from rollout_manager.host_memory import ByteBudget, MemoryAdmissionCancelled, available_bytes
from rollout_manager.input_cache import CacheClient, InputCache, canonical


def equal_inputs(torch, expected, actual):
    if len(expected) != len(actual):
        return False
    for index, (a, b) in enumerate(zip(expected, actual)):
        if isinstance(a, torch.Tensor):
            if (not isinstance(b, torch.Tensor) or a.dtype != b.dtype or a.shape != b.shape
                    or a.stride() != b.stride() or a.storage_offset() != b.storage_offset()
                    or a.requires_grad != b.requires_grad):
                return False
            for previous in range(index):
                left, right = expected[previous], actual[previous]
                if isinstance(left, torch.Tensor):
                    if ((a is left) != (b is right) or
                            (a.untyped_storage()._cdata == left.untyped_storage()._cdata) !=
                            (b.untyped_storage()._cdata == right.untyped_storage()._cdata)):
                        return False
            count = a.untyped_storage().nbytes()
            if count != b.untyped_storage().nbytes():
                return False
            left = torch.empty(0, dtype=torch.uint8, device="cpu").set_(a.untyped_storage(), 0, (count,), (1,))
            right = torch.empty(0, dtype=torch.uint8, device="cpu").set_(b.untyped_storage(), 0, (count,), (1,))
            for offset in range(0, count, 64*1024**2):
                if not torch.equal(left[offset:offset+64*1024**2], right[offset:offset+64*1024**2]):
                    return False
        elif type(a) is not type(b) or a != b:
            return False
    return True


def qualify(task_path, report):
    import numpy as np
    import torch
    from rollout_manager.adapters.input_reuse import InputReuse, rng_state, restore_rng
    data = json.loads(task_path.read_text())
    task, plan = data["task"], data["plan"]
    context = {"__name__": "native_reuse_qualification"}
    exec(compile(task["code"], "model.py", "exec"), context)
    results = []
    identity = None
    for namespace, seed in (("development", 42), ("final-verification", 1042)):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        seeds = [torch.randint(0, 2**32-1, (1,)).item() for _ in range(5)] + [seed, seed]
        for index, trial_seed in enumerate(seeds):
            torch.manual_seed(trial_seed)
            before = rng_state(torch)
            started = time.monotonic()
            native = context["get_inputs"]()
            native_seconds = time.monotonic()-started
            after = rng_state(torch)
            restore_rng(torch, before)
            adapter = InputReuse(torch, data["socket"], data["consumer"], task["code"], plan,
                namespace=namespace, seed=seed, mode="kernelbench", qualifying=True)
            cold = adapter.get_inputs(context["get_inputs"], index)
            cold_rng = canonical(rng_state(torch)) == canonical(after)
            restore_rng(torch, before)
            warm = adapter.get_inputs(context["get_inputs"], index)
            warm_rng = canonical(rng_state(torch)) == canonical(after)
            item = {"namespace": namespace, "seed": seed, "trial_seed": trial_seed,
                "trial_index": index, "native_seconds": native_seconds,
                "cold_bitwise_equal": equal_inputs(torch, native, cold),
                "warm_bitwise_equal": equal_inputs(torch, native, warm),
                "cold_rng_equal": cold_rng, "warm_rng_equal": warm_rng,
                "cache_events": adapter.events}
            if not all(item[k] for k in ("cold_bitwise_equal", "warm_bitwise_equal", "cold_rng_equal", "warm_rng_equal")):
                raise ValueError("native/cache equivalence failed: " + canonical(item))
            if adapter.events[-1]["status"] != "hit":
                raise ValueError("qualification did not exercise a cache hit")
            identity = adapter.identity
            results.append(item)
            del native, cold, warm, adapter
            gc.collect()
    if torch.cuda.is_initialized():
        raise ValueError("CPU generator qualification unexpectedly owns CUDA")
    save_json(report, {"task_id": task["task_id"], "identity": identity, "passed": True,
                       "trials": results, "cuda_initialized": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--input-memory-plan", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--cores", default="40,41;42,43;44,45;46,47;48,49;50,51;52,53;54,55")
    parser.add_argument("--cache-gib", type=int, default=256)
    parser.add_argument("--host-gib", type=int, default=128)
    args = parser.parse_args()
    if args.task:
        qualify(args.task, args.report)
        return 0
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    cores = [list(map(int, item.split(","))) for item in args.cores.split(";")]
    if not set(sum(cores, [])) <= os.sched_getaffinity(0):
        raise ValueError("qualification affinity outside allocation")
    if (args.cache_gib+args.host_gib+32)*1024**3 > available_bytes():
        raise ValueError("qualification memory allowances exceed available RAM")
    cases = json.loads(args.cases.read_text())
    registry = json.loads(args.input_memory_plan.read_text())
    plans = registry.get("plans", registry.get("contracts", {}))
    native_cuda = {t["source_sha256"] for t in registry["tasks"] if t.get("native_device_generator")}
    supported, skipped = [], []
    for case in cases:
        task = case["task"]
        if hashlib.sha256(task["code"].encode()).hexdigest() != task["source_sha256"]:
            raise ValueError("source hash mismatch")
        if task["source_sha256"] in native_cuda:
            skipped.append({"task_id": task["task_id"], "reason": "native CUDA input generator; reuse disabled"})
        elif task["source_sha256"] not in plans:
            raise ValueError("missing reviewed generator contract: " + task["task_id"])
        else:
            supported.append(task)
    credit = ByteBudget(args.host_gib*1024**3)
    fatal = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: fatal.set())
    signal.signal(signal.SIGINT, lambda *_: fatal.set())
    lock = threading.Lock()
    results = {}
    with tempfile.TemporaryDirectory(prefix="rq-", dir="/tmp") as temp:
        cache = InputCache(Path(temp)/"c", ram_bytes=args.cache_gib*1024**3)
        client = CacheClient(cache.socket_path)
        save_json(root/"settings.json", {"cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            "registry_sha256": hashlib.sha256(args.input_memory_plan.read_bytes()).hexdigest(),
            "cores": cores, "threads_per_worker": 1, "supported": len(supported), "skipped": skipped,
            "cache_gib": args.cache_gib, "host_gib": args.host_gib, "scope": "unmeasured qualification"})

        def run_worker(index):
            for task in supported[index::len(cores)]:
                if fatal.is_set():
                    break
                task_id = task["task_id"]
                plan = plans[task["source_sha256"]]
                allocation = plan.get("allocation_bytes", plan.get("arena_bytes"))
                key = task_id.replace("/", "-")
                credit.enqueue(key, 3*allocation+2*1024**3)
                credit.acquire(key, fatal.is_set)
                consumer = client.control("begin", copy_limit=2*allocation)["consumer"]
                directory = root/key
                directory.mkdir()
                task_path = directory/"input.json"
                report = directory/"result.json"
                save_json(task_path, {"task": task, "plan": plan, "socket": str(cache.socket_path), "consumer": consumer})
                env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           HF_HUB_OFFLINE="1", LITELLM_LOCAL_MODEL_COST_MAP="True")
                command = ["taskset", "-c", ",".join(map(str, cores[index])), sys.executable,
                           str(Path(__file__).resolve()), "--task", str(task_path), "--report", str(report)]
                process = run_process(command, directory, env, directory/"qualification.log",
                                      timeout=900, heartbeat=lambda: None, cancelled=fatal.is_set)
                save_json(directory/"process.json", process)
                if not process["process_group_stopped"]:
                    credit.quarantine(key)
                    fatal.set()
                    break
                client.control("end", consumer=consumer)
                credit.release(key)
                item = json.loads(report.read_text()) if report.exists() else {
                    "task_id": task_id, "passed": False, "returncode": process["returncode"]}
                if process["returncode"] != 0 or not item["passed"]:
                    fatal.set()
                with lock:
                    results[task_id] = item
                    print(json.dumps({"finished": len(results), "expected": len(supported), "task": task_id,
                                      "passed": item["passed"], "seconds": process["wall_seconds"]}), flush=True)

        def worker(index):
            try:
                run_worker(index)
            except MemoryAdmissionCancelled:
                if not fatal.is_set():
                    fatal.set()
                    raise
            except BaseException:
                fatal.set()
                raise
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(cores)) as pool:
                list(pool.map(worker, range(len(cores))))
            snap = cache.snapshot()
            passed = (len(results) == len(supported) and all(x["passed"] for x in results.values())
                      and not fatal.is_set() and not snap["consumers"] and not snap["references"])
            save_json(root/"qualification.json", {"passed": passed, "tasks": results, "skipped": skipped,
                "cache_final": snap, "host_final": credit.snapshot(), "expected": len(supported)})
            return 0 if passed else 1
        finally:
            if not cache.snapshot()["consumers"]:
                cache.close()


if __name__ == "__main__":
    raise SystemExit(main())
