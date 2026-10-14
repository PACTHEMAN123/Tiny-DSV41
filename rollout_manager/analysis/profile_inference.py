"""Rerun a saved one-turn task group and retain vLLM phase and RPC timings."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

from rollout_manager.network import RpcClient


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--task", default="kernelbench/level_2/30")
    parser.add_argument("--mode", choices=("single", "group", "trace"), required=True)
    parser.add_argument("--sample", type=int, default=4)
    args = parser.parse_args()
    summary = json.loads((args.source / "summary.json").read_text())
    level, = summary["levels"]
    jobs = sorted((j for j in level["results"] if j["task"] == args.task),
                  key=lambda j: j["rollout"]["sample_index"])
    if [j["rollout"]["sample_index"] for j in jobs] != list(range(8)):
        raise ValueError("expected the exact eight-sample task group")
    if args.mode == "single":
        jobs = [jobs[args.sample]]
    requests = []
    for job in jobs:
        path = args.source / "jobs" / job["job"] / "dispatch.json"
        request = json.loads(path.read_text())["request"]
        original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if args.mode == "trace":
            request["max_tokens"] = 128
        requests.append((job, request, original_hash))
    args.output.mkdir(parents=True, exist_ok=False)
    save(args.output / "plan.json", {
        "task": args.task, "mode": args.mode, "fresh_inference": True,
        "turns_per_sample": 1, "concurrency": len(jobs),
        "source": str(args.source.resolve()), "source_jobs": [j["job"] for j in jobs],
        "trace_token_cap": 128 if args.mode == "trace" else None,
        "candidate_execution": False, "affinity": sorted(os.sched_getaffinity(0)),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "profile_note": "trace is a separate capped diagnostic, not a full rollout",
    })
    barrier = threading.Barrier(len(requests))
    run_started = time.monotonic_ns()
    run_unix = time.time_ns()

    def invoke(item):
        job, request, source_hash = item
        index = job["rollout"]["sample_index"]
        request_id = f"timeline-{args.mode}-{run_unix}-{index}"
        client = RpcClient(args.url, args.credentials, timeout=1800)
        try:
            echo = client.call("/echo", {"profile": request_id})
            barrier.wait(timeout=30)
            started = time.monotonic_ns()
            response = client.call("/generate/" + request_id, request)
            finished = time.monotonic_ns()
            save(args.output / f"sample-{index}.json", response)
            metrics = response["response"]["metrics"]
            durations = {"queue": metrics["queue_time_ms"] / 1000,
                         "prefill": metrics["time_to_first_token_ms"] / 1000,
                         "decode": metrics["generation_time_ms"] / 1000}
            if any(v < 0 for v in durations.values()):
                raise ValueError("invalid phase timing")
            record = {"sample": index, "request_id": request_id,
                      "source_job": job["job"], "source_dispatch_sha256": source_hash,
                      "start_seconds": (started - run_started) / 1e9,
                      "finish_seconds": (finished - run_started) / 1e9,
                      "phase_seconds": durations, "metrics": metrics,
                      "usage": response["response"]["usage"],
                      "finish_reason": response["response"]["choices"][0]["finish_reason"],
                      "communication": response["communication"],
                      "echo_communication": echo["communication"],
                      "response_sha256": hashlib.sha256(
                          (args.output / f"sample-{index}.json").read_bytes()).hexdigest()}
            save(args.output / f"timing-{index}.json", record)
            print(json.dumps(record), flush=True)
            return record
        except BaseException as exc:
            barrier.abort()
            save(args.output / f"error-{index}.json", {
                "error": repr(exc), "rpc_failure": getattr(exc, "rpc_failure", None)})
            raise
        finally:
            client.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as pool:
        results = list(pool.map(invoke, requests))
    save(args.output / "summary.json", {
        "mode": args.mode, "started_unix_ns": run_unix,
        "wall_seconds": (time.monotonic_ns() - run_started) / 1e9,
        "generation_span_seconds": max(r["finish_seconds"] for r in results)
                                   - min(r["start_seconds"] for r in results),
        "results": results,
    })


if __name__ == "__main__":
    main()
