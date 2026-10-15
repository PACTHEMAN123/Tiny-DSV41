"""Summarize native rollout stages, resource queues and communication without GPU libraries."""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import statistics
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def json_lines(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def distribution(values):
    values = sorted(float(x) for x in values if x is not None and math.isfinite(float(x)))
    if not values:
        return {"count": 0}
    def percentile(p):
        position = (len(values)-1)*p
        lower = int(position)
        upper = min(lower+1, len(values)-1)
        return values[lower] + (values[upper]-values[lower])*(position-lower)
    return {"count": len(values), "mean": statistics.fmean(values), "min": values[0],
            "p50": percentile(.5), "p95": percentile(.95), "p99": percentile(.99), "max": values[-1]}


def intervals(rows, start, end):
    for index, row in enumerate(rows):
        left = max(start, row["unix_ns"])
        right = min(end, rows[index+1]["unix_ns"] if index+1 < len(rows) else end)
        if right > left:
            yield row, (right-left)/1e9


def memory_kib(raw):
    return {line.split()[0].rstrip(":"): int(line.split()[1]) for line in raw.splitlines()
            if len(line.split()) >= 2 and line.split()[1].isdigit()}


def cpu_ticks(raw, cpus=None):
    selected = {"cpu"+str(cpu) for cpu in cpus} if cpus is not None else {"cpu"}
    total = 0
    for line in raw.splitlines():
        fields = line.split()
        if fields[0] in selected:
            values = list(map(int, fields[1:9]))
            total += sum(values[i] for i in (0, 1, 2, 5, 6))
    return total


def pressure_total(raw, kind="some"):
    if not isinstance(raw, str):
        return None
    for line in raw.splitlines():
        fields = line.split()
        if fields[0] == kind:
            return int(dict(field.split("=") for field in fields[1:])["total"])
    return None


def telemetry_summary(samples, settings, start, end):
    duration = (end-start)/1e9
    sums, peaks = collections.defaultdict(float), collections.defaultdict(float)
    metric_coverage = collections.defaultdict(float)
    covered = 0
    eligible = [row for row in samples if start <= row["unix_ns"] <= end]
    for row, seconds in intervals(samples, start, end):
        covered += seconds
        memory = memory_kib(row["memory"])
        fields = {"available_gib": memory.get("MemAvailable", 0)/1024**2,
                  "host_shmem_gib": memory.get("Shmem", 0)/1024**2}
        manager = row.get("manager", {})
        queues = manager.get("queues", {})
        free = manager.get("gpu_free_slots", 0)
        fields["gpu_free_slots"] = free
        host = manager.get("evaluator_host") or {}
        fields["evaluator_reserved_gib"] = host.get("reserved_bytes", 0)/1024**3
        fields["evaluator_memory_waiting_jobs"] = host.get("waiting_jobs", 0)
        cache = manager.get("input_cache") or {}
        shared = cache.get("shared_storage_bytes", {})
        for tier in ("ram", "disk"):
            fields["input_cache_"+tier+"_gib"] = shared.get(tier, 0)/1024**3
        fields["input_cache_generating_gib"] = cache.get("generating_bytes", 0)/1024**3
        fields["input_cache_consumer_cow_reserved_gib"] = cache.get("consumer_cow_reserved_bytes", 0)/1024**3
        fields["input_cache_references"] = cache.get("references", 0)
        for stage, state in queues.items():
            for kind in ("queued", "active"):
                fields[stage+"_"+kind] = state.get(kind, 0)
        for line in (row.get("gpu_csv") or "").splitlines():
            values = [x.strip() for x in line.split(",")]
            if len(values) >= 6:
                try:
                    fields["gpu_"+values[0]+"_util_percent"] = float(values[2])
                    fields["gpu_"+values[0]+"_memory_mib"] = float(values[4])
                    fields["gpu_"+values[0]+"_power_watts"] = float(values[5])
                except ValueError:
                    pass
        for key, value in fields.items():
            if not math.isfinite(value):
                continue
            sums[key] += value*seconds
            metric_coverage[key] += seconds
            peaks[key] = max(peaks[key], value)
    result = {"sample_count": len(eligible), "covered_seconds": covered,
              "time_weighted_mean": {key: value/metric_coverage[key] for key, value in sums.items()},
              "metric_covered_seconds": dict(metric_coverage),
              "gpu_query_error_samples": sum(bool(row.get("gpu_error")) for row in eligible),
              "sampled_max": dict(peaks),
              "minimum_available_gib": min((memory_kib(x["memory"])["MemAvailable"]/1024**2
                                           for x in eligible), default=None)}
    if len(eligible) > 1:
        first, last = eligible[0], eligible[-1]
        seconds = (last["unix_ns"]-first["unix_ns"])/1e9
        hz = first.get("owned_process_accounting", {}).get("clock_ticks_per_second", 100)
        result["visible_aggregate_cpu_busy_core_equivalents"] = (
            cpu_ticks(last["stat"])-cpu_ticks(first["stat"]))/hz/seconds
        visible_ids = {int(line.split()[0][3:]) for line in first["stat"].splitlines()
                       if line.split()[0].startswith("cpu") and line.split()[0][3:].isdigit()}
        affinity_ids = set(settings["cpu_affinity"])
        result["proc_stat_cpu_ids_match_affinity"] = affinity_ids <= visible_ids
        if affinity_ids <= visible_ids:
            cpus = settings["effective_driver_affinity"]
            result["allocated_cpu_busy_core_equivalents"] = (
                cpu_ticks(last["stat"], cpus)-cpu_ticks(first["stat"], cpus))/hz/seconds
        for key in ("cpu_pressure", "memory_pressure", "io_pressure"):
            before, after = pressure_total(first[key]), pressure_total(last[key])
            result[key+"_some_percent"] = (after-before)/1e4/seconds if before is not None and after is not None else None
        seen = {}
        ticks = 0
        for row in eligible:
            for process in row.get("owned_process_accounting", {}).get("processes", []):
                key = (process["pid"], process["start_ticks"])
                value = process["user_ticks"]+process["system_ticks"]
                if key in seen:
                    ticks += max(0, value-seen[key])
                seen[key] = value
        result["sampled_owned_cpu_core_equivalents_lower_bound"] = ticks/hz/seconds
    result["limits"] = ["Visible aggregate CPU and PSI scope depends on the provider; CPU IDs may be virtualized and quotas hidden.",
        "Owned CPU sampling misses startup, exit and short-lived children; shared RSS is not summed.",
        "Two-second GPU utilization samples can miss short bursts; queue occupancy is also integrated from events.",
        "Missing or invalid GPU samples are excluded from each metric's denominator; coverage is reported separately."]
    result["duration_seconds"] = duration
    return result


def queue_summary(events, start, end):
    sums, maxima = collections.defaultdict(float), collections.defaultdict(int)
    duration = (end-start)/1e9
    for event, seconds in intervals(events, start, end):
        for stage, state in event.get("queues", {}).items():
            for kind in ("queued", "active"):
                key = stage+"_"+kind
                sums[key] += state[kind]*seconds
                maxima[key] = max(maxima[key], state[kind])
    return {"mean": {k: v/duration for k, v in sums.items()} if duration else {}, "max": dict(maxima)}


def communication(rows):
    keys = ("client_total_seconds", "server_processing_seconds", "connect_tls_seconds", "residual_seconds",
            "send_seconds", "response_body_seconds", "request_body_bytes", "response_body_bytes")
    return {"all": {key: distribution(row.get(key) for row in rows) for key in keys},
            "persistent_residual_seconds": distribution(row["residual_seconds"] for row in rows
                                                        if row.get("connect_tls_seconds") == 0),
            "tcp_retransmissions": sum(row.get("tcp_retrans_delta", 0) for row in rows),
            "setup_retransmissions": sum(row.get("tcp_setup_retransmissions", 0) for row in rows),
            "tcp_rtt_us": distribution(row.get("tcp_after", {}).get("rtt_us") for row in rows),
            "total_body_bytes": sum(row.get("request_body_bytes", 0)+row.get("response_body_bytes", 0) for row in rows),
            "definition": "Residual is client interval minus server interval, including network/HTTP/setup overhead; not one-way latency."}


def native_summary(root, jobs):
    measures = collections.defaultdict(lambda: collections.defaultdict(list))
    calls = []
    def add(group, key, value):
        if value is not None:
            measures[group][key].append(value)
    for job in jobs:
        directory = root / "jobs" / job["job"]
        task = job["task"]
        timing = job.get("development_timing") or {}
        if "candidate_us" in timing:
            for key in ("candidate_us", "torch_eager_us", "torch_compile_us"):
                add(task+"/evaluation-profile", "native_"+key, timing.get(key))
        for path in (directory / "workspace").glob("*.json"):
            data = read_json(path)
            if path.name.endswith("-process.json"):
                group = task+"/"+path.name.removesuffix("-process.json")
                for key in ("wall_seconds", "spawn_seconds", "reap_seconds", "admission_check_seconds", "context_cleanup_seconds"):
                    add(group, key, data.get(key))
            if "stage_metrics" not in data:
                continue
            group = task+"/"+path.stem
            metric = data["stage_metrics"]
            for key in ("process_startup_seconds",):
                add(group, key, metric.get(key))
            calls.append({"job": job["job"], "stage": path.stem, "actual": metric["input_calls"],
                          "expected": metric["expected_input_calls"], "input_source": metric.get("input_source")})
            totals = collections.defaultdict(float)
            for event in metric["events"]:
                key = event["stage"]
                add(group, key+"_call_wall_seconds", event["wall_seconds"])
                totals[key+"_total_wall_seconds"] += event["wall_seconds"]
                if event["cuda_event_ms"] is not None:
                    add(group, key+"_call_cuda_ms", event["cuda_event_ms"])
                    totals[key+"_total_cuda_ms"] += event["cuda_event_ms"]
            for key, value in totals.items():
                add(group, key, value)
            if "runtime" in data:
                add(group, "native_candidate_ms", data["runtime"])
                add(group, "native_reference_ms", data.get("ref_runtime"))
    return {"groups": {g: {k: distribution(v) for k, v in values.items()} for g, values in measures.items()},
            "input_calls": calls,
            "all_input_counts_match": bool(calls) and all(x["actual"] == x["expected"] for x in calls)}


def analyze(root):
    summary, settings = read_json(root / "summary.json"), read_json(root / "settings.json")
    jobs = [job for level in summary["levels"] for job in level["results"]]
    events = list(json_lines(root / "events.jsonl"))
    samples = list(json_lines(root / "telemetry-B.jsonl"))
    probes = list(json_lines(root / "probes.jsonl"))
    start = min(level["started_unix_ns"] for level in summary["levels"])
    end = max(level["finished_unix_ns"] for level in summary["levels"])
    first_completions = [job["finished_unix_ns"] for job in jobs if job["job"].endswith("-r0")]
    fill_end = max(first_completions, default=end)
    drain_start = max((job["started_unix_ns"] for job in jobs), default=end)
    phases = {"full": (start, end)}
    if fill_end < drain_start:
        phases.update(fill=(start, fill_end), middle=(fill_end, drain_start), drain=(drain_start, end))
    phases_result = {}
    for name, (left, right) in phases.items():
        completed = [job for job in jobs if left < job["finished_unix_ns"] <= right
                     and job.get("verified_correct", job.get("final_correct"))]
        seconds = (right-left)/1e9
        phases_result[name] = {"started_unix_ns": left, "finished_unix_ns": right, "wall_seconds": seconds,
            "verified_completions": len(completed), "verified_jobs_per_minute": len(completed)*60/seconds,
            "telemetry": telemetry_summary(samples, settings, left, right),
            "queues": queue_summary(events, left, right)}
    stages = {}
    for stage in ("inference", "build", "evaluation", "verification"):
        stages[stage] = {key: distribution(job.get("stages", {}).get(stage, {}).get(key) for job in jobs)
                         for key in ("queue_seconds", "service_seconds")}
    critical = collections.defaultdict(list)
    for job in jobs:
        components = {"review": job.get("review_wait_seconds", 0)}
        for stage, timing in job["stages"].items():
            for kind in ("queue_seconds", "service_seconds"):
                components[stage+"_"+kind] = timing.get(kind, 0)
        components["remaining_or_unrecorded"] = job["wall_seconds"]-sum(components.values())
        for key, value in components.items():
            critical[key].append(value)
    final = read_json(root / "evaluator-host-final.json") if (root / "evaluator-host-final.json").exists() else None
    return {"name": root.name, "settings": settings, "jobs": len(jobs),
        "correct": sum(bool(job.get("verified_correct", job.get("final_correct"))) for job in jobs),
        "errors": [job for job in jobs if not job.get("completed")],
        "fatal": summary["fatal"], "requested_levels_completed": summary["requested_levels_completed"],
        "no_profiler_during_sweep": summary["no_profiler_during_sweep"],
        "task_counts": dict(collections.Counter(job["task"] for job in jobs)),
        "candidate_hash_counts": dict(collections.Counter(job.get("candidate_sha256", "fixed-tools-source") for job in jobs)),
        "latency_seconds": distribution(job["wall_seconds"] for job in jobs),
        "review_wait_seconds": distribution(job.get("review_wait_seconds") for job in jobs),
        "completion_tokens": distribution((job.get("usage") or {}).get("completion_tokens") for job in jobs),
        "stages": stages,
        "critical_path_components_seconds": {k: distribution(v) for k, v in critical.items()},
        "evaluator_memory_final": final,
        "phases": phases_result,
        "phase_definition": "Fill ends when every client's first job finishes; middle ends at the last job submission; the remainder is drain. Middle is a finite-window estimate, not proof of stationarity. If boundaries cross, only full is reported.",
        "communication": communication([job["communication"] for job in jobs if "communication" in job]),
        "echo": {"errors": [p for p in probes if "error" in p], "by_payload_bytes": {
            str(size): communication([p["communication"] for p in probes if p["bytes"] == size and "communication" in p])
            for size in sorted({p["bytes"] for p in probes})}},
        "native": native_summary(root, jobs)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"runs": [analyze(root) for root in args.roots]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    for row in result["runs"]:
        print(json.dumps({"run": row["name"], "jobs": row["jobs"], "correct": row["correct"],
            "verified_jobs_per_minute": row["phases"]["full"]["verified_jobs_per_minute"],
            "p95_latency_seconds": row["latency_seconds"].get("p95")}))


if __name__ == "__main__":
    main()
