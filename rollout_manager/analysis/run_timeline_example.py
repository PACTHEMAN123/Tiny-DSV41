"""Run one saved prompt or its complete G=8 group through fresh A/B execution."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from rollout_manager.common import save_json
from rollout_manager.network import RpcClient
from rollout_manager.reference_qualification import load_catalog
from rollout_manager.telemetry import snapshot


def clock_probes(url, credentials, destination):
    client = RpcClient(url, credentials, timeout=10)
    try:
        rows = [client.call("/echo", {"clock_sample": i})["communication"] for i in range(20)]
    finally:
        client.close()
    save_json(destination, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-settings", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "group"), required=True)
    parser.add_argument("--instrumentation", choices=("none", "nsys"), default="none")
    parser.add_argument("--task", default="kernelbench/level_2/30")
    parser.add_argument("--sample", type=int, choices=range(8), default=4)
    parser.add_argument('--isolate-device-inputs', action='store_true')
    parser.add_argument('--plan-only', action='store_true',
                        help='write the exact invocation without contacting A or starting any worker')
    args = parser.parse_args()
    root = args.root.resolve()
    meta = root.with_name(root.name + "-metadata")
    meta.mkdir(parents=True, exist_ok=False)
    settings = json.loads(args.baseline_settings.read_text())
    cases_path = Path(settings["cases"])
    cases = json.loads(cases_path.read_text())
    qualification = load_catalog(Path(settings["reference_qualification"]), cases_path, cases,
                                 preserve_integer_inputs=True)
    case, = (case for case in cases if case["task"]["task_id"] == args.task)
    if not qualification["tasks"][args.task]["score_eligible"]:
        raise ValueError("example requires a qualified native reference")
    level, = json.loads((args.source / "summary.json").read_text())["levels"]
    jobs = sorted((job for job in level["results"] if job["task"] == args.task),
                  key=lambda job: job["rollout"]["sample_index"])
    if [job["rollout"]["sample_index"] for job in jobs] != list(range(8)):
        raise ValueError("expected the exact original G=8 task group")
    if args.mode == "single":
        jobs = [jobs[args.sample]]
    selected, sources = [], []
    for job in jobs:
        dispatch = args.source / "jobs" / job["job"] / "dispatch.json"
        item = copy.deepcopy(case)
        item["request"] = json.loads(dispatch.read_text())["request"]
        item["rollout"] = job["rollout"]
        selected.append(item)
        sources.append({"source_job": job["job"], "sample": job["rollout"]["sample_index"],
                        "dispatch_sha256": hashlib.sha256(dispatch.read_bytes()).hexdigest()})
    selected_path = meta / "cases.json"
    save_json(selected_path, selected)
    save_json(meta / "reference-qualification.json", qualification["tasks"][args.task])
    command = [sys.executable, "-B", "-m", "rollout_manager.runner", "--mode", "coupled",
        "--root", str(root), "--cases", str(selected_path), "--catalog", str(args.catalog.resolve()),
        "--credentials", str(args.credentials.resolve()), "--schedule", "cyclic",
        "--concurrency", str(len(jobs)), "--repeats", "1", "--turns", "1",
        "--rpc-timeout", "1860", "--build-slots", "8", "--cores-per-worker", "4",
        "--evaluation-cores", "2", "--evaluation-threads", "1", "--bound-cpu-budget",
        "--evaluation-host-gib", "128", "--observe-inputs", "--preserve-integer-inputs",
        "--instrumentation", args.instrumentation, "--profile-worker", str(0 if len(jobs) == 1 else args.sample),
        "--external-instrumentation", args.instrumentation, "--review-timeout", "1800", "--ttl", "7200"]
    for flag in ("assets", "input_memory_plan", "model_manifest", "gpus", "url"):
        command += ["--" + flag.replace("_", "-"), settings[flag]]
    if args.isolate_device_inputs:
        command += ['--isolate-device-inputs']
    save_json(meta / "plan.json", {
        "fresh_inference": True, "task": args.task, "mode": args.mode, "samples": sources,
        "one_turn_per_sample": True, "input_reuse": False, "argv": command,
        'isolate_device_inputs': args.isolate_device_inputs,
        "schedule": "one finite group, one exact saved request per worker; no subsequent groups",
        "scoring_note": "profiled development scoring is diagnostic; final scoring is unprofiled",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline_settings_sha256": hashlib.sha256(args.baseline_settings.read_bytes()).hexdigest(),
        "qualification_sha256": hashlib.sha256(Path(settings["reference_qualification"]).read_bytes()).hexdigest(),
    })
    if args.plan_only:
        print(json.dumps({'plan': str(meta / 'plan.json'), 'executed': False}))
        return
    save_json(meta / "preflight.json", snapshot())
    clock_probes(settings["url"], args.credentials, meta / "clock-before.json")
    started = time.time_ns()
    result = subprocess.run(command)
    save_json(meta / "exit.json", {"started_unix_ns": started, "finished_unix_ns": time.time_ns(),
                                 "returncode": result.returncode})
    clock_probes(settings["url"], args.credentials, meta / "clock-after.json")
    save_json(meta / "postflight.json", snapshot())
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
