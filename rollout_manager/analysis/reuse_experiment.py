"""A separate full-cohort verification comparison: off, cold, warm, warm, cold, off.

Replay exact reviewed real-vLLM responses. Keep this plan and all result roots
independent of the retired preparation experiment. Disk storage is opt-in and
must use a distinct experiment root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from rollout_manager.common import save_json
from rollout_manager.cpu_layout import controlled_layout
from rollout_manager.input_cache import InputCache
from rollout_manager.preflight import preflight
from rollout_manager.network import RpcClient
from rollout_manager.common import process_identity
from rollout_manager.workloads import model_dependency
from rollout_manager.analysis.analyze_run import analyze, distribution


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        while block := source.read(8*1024**2):
            digest.update(block)
    return digest.hexdigest()


def model_files(settings):
    manifest_path = Path(settings["model_manifest"]).resolve()
    models = {item["model"]: item for item in json.loads(manifest_path.read_text())}
    dependencies = {model_dependency(case["task"]["code"])
                    for case in json.loads(Path(settings["cases"]).read_text())} - {None}
    home = manifest_path.parent/"hf-home"
    pinned = {}
    for name in sorted(dependencies):
        model = models[name]
        snapshot = home/"hub"/("models--"+name.replace("/", "--"))/"snapshots"/model["revision"]
        for item in model["files"]:
            path = snapshot/item["name"]
            if path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
                raise ValueError("offline model asset changed: " + str(path))
            pinned[str(path)] = item["sha256"]
    return home, pinned


def build_command(settings, root, registry, qualification, socket_path, state):
    command = [sys.executable, "-u", "-m", "rollout_manager.runner", "--root", str(root)]
    fields = ("mode", "assets", "credentials", "cases", "catalog", "frozen_rollouts", "url", "rpc_timeout",
              "reference_qualification", "concurrency", "repeats", "schedule", "rollouts_per_task",
              "max_inflight_groups", "sampling_seed", "turns", "gpus", "build_slots", "cores_per_worker",
              "evaluation_cores", "evaluation_threads", "gpu_slots", "gpu_settle_seconds",
              "evaluation_host_gib", "model_manifest", "review_timeout", "ttl")
    for field in fields:
        value = settings[field]
        command += ["--"+field.replace("_", "-")] + list(map(str, value if isinstance(value, list) else [value]))
    command += ["--preserve-integer-inputs", "--observe-inputs", "--bound-cpu-budget",
                "--input-memory-plan", str(registry)]
    if state != "off":
        command += ["--input-cache-socket", str(socket_path), "--input-cache-qualification", str(qualification),
                    "--input-cache-state", state]
    return command


def validate_arm(root, expected):
    summary = json.loads((root/"summary.json").read_text())
    if summary["fatal"] or not summary["requested_levels_completed"]:
        raise ValueError("arm is incomplete or has an infrastructure failure")
    level, = summary["levels"]
    if level["jobs"] != expected:
        raise ValueError("arm coverage changed")
    host = json.loads((root/"evaluator-host-final.json").read_text())
    if host["reserved_bytes"] or host["waiting_jobs"]:
        raise ValueError("evaluator memory did not drain")
    owners = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
                                     text=True, timeout=15)
    if owners.strip():
        raise ValueError("GPU contexts remain at the arm boundary")
    for path in root.glob("jobs/*/workspace/*-kernelbench.json"):
        data = json.loads(path.read_text())
        if data.get("correctness"):
            stage = data["stage_metrics"]
            if stage["input_calls"] != 7 or stage["logical_input_indices"] != list(range(7)):
                raise ValueError("successful native trial coverage changed")
    probes = [json.loads(line) for line in (root/"probes.jsonl").read_text().splitlines()]
    if not probes or any("error" in row for row in probes):
        raise ValueError("communication probes are incomplete or failed")
    return level


def cache_measurements(root):
    initial_path, final_path = root/"input-cache-initial.json", root/"input-cache-final.json"
    if not initial_path.exists():
        return {"enabled": False}
    initial, final = json.loads(initial_path.read_text()), json.loads(final_path.read_text())
    before, after = initial["metrics"], final["metrics"]
    events = []
    for path in root.glob("jobs/*/workspace/*-kernelbench.json"):
        value = json.loads(path.read_text()).get("stage_metrics", {}).get("input_reuse", {})
        events.extend(value.get("events", []))
    return {"enabled": True, "initial": initial, "final": final,
        "delta": {k: v-before.get(k, 0) for k, v in after.items()},
        "broker_cpu_seconds": final["owner_cpu_seconds"]-initial["owner_cpu_seconds"],
        "native_generation_seconds": distribution(e.get("generation_seconds") for e in events),
        "lookup_wait_seconds": distribution(e.get("lookup_wait_seconds") for e in events),
        "mapping_seconds": distribution(e.get("mapping_seconds") for e in events),
        "publication_copy_seconds": distribution(e.get("publication_copy_seconds") for e in events),
        "cache_event_count": len(events),
        "limitations": "saved generation time is an estimate from the miss producer, not measured net wall-time savings"}


def comparison_record(root, level):
    jobs = level["results"]
    keyed = {}
    for job in jobs:
        rollout = job["rollout"]
        key = json.dumps([job["task"], rollout["pass_index"], rollout["sample_index"]])
        keyed[key] = {"verified": job["verified_correct"], "score_eligible": job["score_eligible"],
            "development_correct": job.get("development_correct"), "final_correct": job.get("final_correct"),
            "candidate_sha256": job.get("candidate_sha256"), "error_type": job.get("error_type"),
            "source_inference": job.get("source_inference"), "rollout": rollout}
    return {"root": str(root), "wall_seconds": level["wall_seconds"], "episodes": len(jobs),
        "verified": level["correct"], "verified_per_minute": level["correct"]*60/level["wall_seconds"],
        "latency_seconds": distribution(j["wall_seconds"] for j in jobs), "jobs": keyed,
        "cache": cache_measurements(root)}


def compare_jobs(baseline, actual):
    if set(baseline) != set(actual):
        raise ValueError("task/sample coverage changed")
    changed = []
    for key, job in actual.items():
        for field in ("candidate_sha256", "source_inference", "rollout", "score_eligible"):
            if job[field] != baseline[key][field]:
                raise ValueError("comparison workload changed: " + field)
        if any(job.get(field) != baseline[key].get(field) for field in (
                "verified", "development_correct", "final_correct", "error_type")):
            changed.append(key)
    return changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-settings", type=Path, required=True)
    parser.add_argument("--input-memory-plan", type=Path, required=True)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--relay-lease", type=Path, required=True)
    parser.add_argument("--cache-ram-gib", type=int, default=128)
    parser.add_argument("--cache-disk-gib", type=int, default=0)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    lock = (root/"controller.lock").open("x")
    import fcntl
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    settings = json.loads(args.baseline_settings.read_text())
    if (settings["mode"] != "tools" or settings["schedule"] != "full-suite" or settings["rollouts_per_task"] != 8
            or settings["concurrency"] != [64] or settings["max_inflight_groups"] != 8
            or settings["gpu_slots"] != 1 or settings["turns"] != 1 or settings["repeats"] != 1):
        raise ValueError("requires the frozen complete G=8, 64-episode, exclusive-GPU baseline")
    qualified = json.loads(args.qualification.read_text())
    if qualified["passed"] is not True:
        raise ValueError("all CPU generators must pass reuse equivalence before measurements")
    gpu_ids = preflight(root)
    if len(gpu_ids) != 8 or set(gpu_ids) != set(settings["gpus"].split(",")):
        raise ValueError("all eight baseline GPUs must be available")
    allowed = sorted(os.sched_getaffinity(0))
    _, _, control = controlled_layout(allowed, settings["build_slots"], settings["cores_per_worker"],
                                     8, settings["evaluation_cores"])
    package = Path(__file__).resolve().parents[1]
    pins = {str(p): sha(p) for p in package.rglob("*.py") if "__pycache__" not in p.parts}
    for key in ("cases", "catalog", "frozen_rollouts", "reference_qualification", "model_manifest"):
        pins[settings[key]] = sha(settings[key])
    hf_home, model_pins = model_files(settings)
    pins.update(model_pins)
    pins[str(args.input_memory_plan.resolve())] = sha(args.input_memory_plan)
    pins[str(args.qualification.resolve())] = sha(args.qualification)
    count = len(json.loads(Path(settings["cases"]).read_text()))*8
    order = [("off-r1", "off"), ("cold-r1", "cold"), ("warm-r1", "warm"),
             ("warm-r2", "warm"), ("cold-r2", "cold"), ("off-r2", "off")]
    stopping = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopping.set())
    signal.signal(signal.SIGINT, lambda *_: stopping.set())
    os.sched_setaffinity(0, control)
    with tempfile.TemporaryDirectory(prefix="reuse-", dir="/tmp") as temp:
        events = (root/"cache-events.jsonl").open("x")
        def event(value):
            events.write(json.dumps(value)+"\n")
            events.flush()
        cache = InputCache(Path(temp)/"c", ram_bytes=args.cache_ram_gib*1024**3,
                           disk_bytes=args.cache_disk_gib*1024**3, event=event)
        commands = {name: ["taskset", "-c", ",".join(map(str, allowed))] + build_command(
            settings, root/name, args.input_memory_plan.resolve(), args.qualification.resolve(), cache.socket_path, state)
            for name, state in order}
        plan = {"kind": "immutable_cpu_input_reuse_comparison", "created_unix": time.time(),
            "measurement": "B verification of frozen real vLLM outputs; no fresh inference",
            "baseline_settings": str(args.baseline_settings.resolve()), "baseline_settings_sha256": sha(args.baseline_settings),
            "order": order, "commands": commands, "pins": pins, "expected_episodes": count,
            "cache_ram_gib": args.cache_ram_gib, "cache_disk_gib": args.cache_disk_gib,
            "granularity": "unchanged native generator call in every arm; no asynchronous producer or trial fusion",
            "warm_definition": "bounded cache contents retained from preceding complete arm; not a guaranteed all-hit cache",
            "controller_identity": process_identity(os.getpid()), "controller_cpu_ids": control,
            "hf_home": str(hf_home), "offline_model_files": len(model_pins),
            "qualifying_data_not_reused": True}
        save_json(root/"plan.json", plan)
        results = []
        try:
            for name, state in order:
                if stopping.is_set():
                    break
                if cache.snapshot()["consumers"]:
                    raise RuntimeError("previous arm left live cache ownership")
                if state in ("off", "cold"):
                    cache.clear()
                for path, expected in pins.items():
                    if sha(path) != expected:
                        raise ValueError("pinned code/data changed: " + path)
                while not stopping.is_set():
                    save_json(root/"state.json", {"state": "waiting_relay", "next": name,
                        "finished_arms": len(results), "unix": time.time(), "cache": cache.snapshot()})
                    if args.relay_lease.exists():
                        lease = json.loads(args.relay_lease.read_text())
                        if lease["expires_unix"]-time.time() > settings["ttl"]+120:
                            break
                    stopping.wait(10)
                if stopping.is_set():
                    break
                client = RpcClient(settings["url"], Path(settings["credentials"]))
                try:
                    if client.call("/echo", {"reuse_preflight": name})["echo"] != {"reuse_preflight": name}:
                        raise ValueError("A/B echo preflight failed")
                finally:
                    client.close()
                with (root/(name+".log")).open("x") as output:
                    process = subprocess.Popen(commands[name], stdout=output, stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL, env=dict(os.environ,
                            HF_HOME=str(hf_home), HF_HUB_OFFLINE="1"), start_new_session=True)
                    save_json(root/"state.json", {"state": "running", "arm": name,
                        "identity": process_identity(process.pid), "unix": time.time(), "finished_arms": len(results)})
                    while process.poll() is None:
                        if stopping.wait(5):
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait(timeout=120)
                            raise RuntimeError("experiment stopped; partial arm retained")
                    if process.returncode:
                        raise RuntimeError(f"{name} exited with {process.returncode}")
                level = validate_arm(root/name, count)
                snap = cache.snapshot()
                if snap["consumers"] or snap["references"] or snap["generating_bytes"]:
                    raise RuntimeError("cache ownership did not drain")
                save_json(root/(name+"-analysis.json"), analyze(root/name))
                result = comparison_record(root/name, level)
                if results:
                    result["outcome_changes_vs_off_r1"] = compare_jobs(results[0]["jobs"], result["jobs"])
                results.append(result)
                save_json(root/"comparison.json", {"complete": False, "arms": results, "plan_sha256": sha(root/"plan.json")})
                if result.get("outcome_changes_vs_off_r1"):
                    raise ValueError("verification outcomes changed; inspect closed evidence before advancing")
            complete = len(results) == len(order)
            save_json(root/"comparison.json", {"complete": complete, "arms": results, "plan_sha256": sha(root/"plan.json")})
            save_json(root/"state.json", {"state": "complete" if complete else "stopped", "finished_arms": len(results),
                                          "unix": time.time()})
            return 0 if complete else 1
        except BaseException as exc:
            save_json(root/"state.json", {"state": "failed", "error_type": type(exc).__name__, "error": str(exc),
                                          "finished_arms": len(results), "unix": time.time(), "cache": cache.snapshot()})
            raise
        finally:
            if not cache.snapshot()["consumers"]:
                cache.close()
            events.close()


if __name__ == "__main__":
    raise SystemExit(main())
