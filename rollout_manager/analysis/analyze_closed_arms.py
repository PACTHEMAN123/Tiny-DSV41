"""Analyze the three closed input-reuse arms without extracting or changing evidence."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile


ARCHIVES = {
    "off-r1": "e994a673b647ff689ef336c8f86ec6636a3aba81645e803a0b2b1be78bde1add",
    "cold-r1": "43cc84c63b14c6f16e2da635204a5c3dec0a655c8d47c8b48d0f3b2507ff8f86",
    "warm-r1": "858b1263888309de81ff50a93652857113b73154ea281428f7f3e1b6ab12efd8",
}
DIAGNOSED = {
    ("kernelbench/level_2/98", 0, 3):
        "e0fac78ba633b688a32be68e46966ebcd637424f5849f94e7b5e3e4238cea283",
    ("kernelbench/level_2/13", 0, 5):
        "041324160a33eaa078b63c4667ebd56ff78ca707516ad8b87ad506ac87f5dc60",
}
GIB = 1024**3


def distribution(values):
    values = sorted(float(v) for v in values if v is not None and math.isfinite(v))
    if not values:
        return {"count": 0, "sum": 0.0}
    def quantile(p):
        index = (len(values) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(values) - 1)
        return values[lo] + (values[hi] - values[lo]) * (index - lo)
    return {"count": len(values), "sum": math.fsum(values), "mean": statistics.fmean(values),
            "p50": quantile(.5), "p95": quantile(.95), "p99": quantile(.99),
            "min": values[0], "max": values[-1]}


def job_key(job):
    return (job["task"], job["rollout"]["pass_index"], job["rollout"]["sample_index"])


def cpu_busy_ticks(raw):
    for line in raw.splitlines():
        parts = line.split()
        if parts[0] == "cpu":
            fields = list(map(int, parts[1:9]))
            return sum(fields[i] for i in (0, 1, 2, 5, 6))
    raise ValueError("missing aggregate CPU counters")


def memory_gib(raw):
    return {parts[0].rstrip(":"): int(parts[1]) / 1024**2
            for line in raw.splitlines() if (parts := line.split()) and len(parts) > 1
            and parts[1].isdigit()}


def pressure_total(raw):
    for line in (raw or "").splitlines():
        parts = line.split()
        if parts[0] == "some":
            return int(dict(p.split("=") for p in parts[1:])["total"])
    return None


def telemetry_timeline(archive, name, start, end, width=60):
    """Integrate each observed interval; missing GPU values have no contribution."""
    bins = []
    for index in range(math.ceil((end - start) / 1e9 / width)):
        left = start + index * width * 10**9
        right = min(end, left + width * 10**9)
        bins.append({"start_seconds": (left-start)/1e9, "end_seconds": (right-start)/1e9,
                     "sums": collections.defaultdict(float),
                     "coverage": collections.defaultdict(float)})
    previous = None
    samples = errors = truncated = 0
    offsets = []
    for line in archive.open(name + "/telemetry-B.jsonl"):
        row = json.loads(line)
        if start <= row["unix_ns"] <= end:
            samples += 1
            offsets.append(row["unix_ns"]-row["monotonic_ns"])
            errors += bool(row.get("gpu_error"))
            truncated += bool(row.get("owned_process_accounting", {}).get("truncated"))
        if previous is not None:
            left = max(start, previous["unix_ns"])
            right = min(end, row["unix_ns"])
            interval = (row["unix_ns"] - previous["unix_ns"]) / 1e9
            assert interval > 0
            if right > left:
                manager = previous["manager"]
                memory = memory_gib(previous["memory"])
                cache = manager.get("input_cache") or {}
                host = manager.get("evaluator_host") or {}
                hz = row["owned_process_accounting"]["clock_ticks_per_second"]
                seen = {(p["pid"], p["start_ticks"]): p["user_ticks"] + p["system_ticks"]
                        for p in previous["owned_process_accounting"]["processes"]}
                ticks = sum(max(0, p["user_ticks"] + p["system_ticks"] - seen[key])
                            for p in row["owned_process_accounting"]["processes"]
                            if (key := (p["pid"], p["start_ticks"])) in seen)
                fields = {
                    "visible_cpu_cores": (cpu_busy_ticks(row["stat"])-cpu_busy_ticks(previous["stat"]))/hz/interval,
                    "owned_cpu_cores_lower_bound": ticks/hz/interval,
                    "memory_total_gib": memory["MemTotal"],
                    "memory_used_estimate_gib": memory["MemTotal"]-memory["MemAvailable"],
                    "memory_available_gib": memory["MemAvailable"],
                    "evaluator_reserved_gib": host.get("reserved_bytes", 0)/GIB,
                    "cache_ram_accounted_gib": cache.get("shared_storage_bytes", {}).get("ram", 0)/GIB,
                    "memory_waiting_jobs": host.get("waiting_jobs", 0),
                    "gpu_leased_slots_percent": (8-manager["gpu_free_slots"])*100/8,
                }
                for kind in ("cpu", "memory", "io"):
                    before, after = pressure_total(previous[kind+"_pressure"]), pressure_total(row[kind+"_pressure"])
                    if before is not None and after is not None:
                        fields[kind+"_pressure_some_percent"] = max(0, after-before)/1e4/interval
                gpu = []
                for raw in (previous.get("gpu_csv") or "").splitlines():
                    parts = raw.split(",")
                    try:
                        gpu.append(float(parts[2]))
                    except (ValueError, IndexError):
                        pass
                if len(gpu) == 8:
                    fields["gpu_activity_percent"] = statistics.fmean(gpu)
                while left < right:
                    index = int((left-start)//(width*10**9))
                    edge = min(right, start + (index+1)*width*10**9)
                    seconds = (edge-left)/1e9
                    for key, value in fields.items():
                        bins[index]["sums"][key] += value*seconds
                        bins[index]["coverage"][key] += seconds
                    left = edge
        previous = row
    for row in bins:
        row["means"] = {k: v/row["coverage"][k] for k, v in row.pop("sums").items()}
        row["coverage_seconds"] = dict(row.pop("coverage"))
    return {"bin_seconds": width, "samples_in_window": samples, "gpu_error_samples": errors,
            "owned_process_truncated_samples": truncated, "bins": bins,
            "unix_minus_monotonic_ns": {"median": statistics.median(offsets),
                                         "min": min(offsets), "max": max(offsets)},
            "definition": "60-second time-weighted bins; no extrapolation past the last observation. CPU deltas include only processes present in adjacent samples. GPU gaps are excluded, not zero-filled."}


def publication_overlap(archive, name, calls, clock):
    event_file = "cache-events-through-cold-r1.jsonl" if name == "cold-r1" else "cache-events.jsonl"
    if not calls:
        return {"observed_hits": 0}
    producers, ready, hits = {}, {}, collections.defaultdict(list)
    for line in archive.open(event_file):
        event = json.loads(line)
        key = event["key"]
        if event["event"] == "miss":
            producers[key] = event["unix_ns"]
        elif event["event"] == "publish":
            ready[key] = {"start": producers[key], "end": event["unix_ns"]}
        elif event["event"] == "hit":
            hits[event["consumer"], key].append({**ready[key], "hit": event["unix_ns"]})
    grouped = collections.defaultdict(list)
    tolerance = max(10_000_000, clock["max"]-clock["min"])
    for call in calls:
        key = call["consumer"], call["key"]
        left = call["start_monotonic_ns"] + clock["median"]
        right = left + call["lookup_wait_seconds"]*1e9
        matches = [event for event in hits[key] if left-tolerance <= event["hit"] <= right+tolerance]
        if len(matches) != 1:
            grouped["unmatched"].append(call["lookup_wait_seconds"])
            continue
        event, = matches
        if event["end"] > left+tolerance:
            category = "publication_after_call_started"
        elif event["end"] < left-tolerance:
            category = "publication_before_call_started"
        else:
            category = "within_clock_tolerance"
        grouped[category].append(call["lookup_wait_seconds"])
    return {"observed_hits": len(calls), "timestamp_tolerance_seconds": tolerance/1e9,
            "lookup_seconds_by_publication_timing": {k: distribution(v) for k, v in grouped.items()},
            "definition": "Match native adapter hits to broker events by consumer/key/time. Compare publication time with the enclosing input-call start, calibrated from same-host telemetry clocks. Overlap supports producer waiting; it is not a direct measurement of broker blocking or a new experiment."}


def analyze_arm(path, name):
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    assert digest == ARCHIVES[name], (name, "archive changed")
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
        analysis = json.loads(archive.read(name + "-analysis.json"))
        summary = json.loads(archive.read(name + "/summary.json"))
        assert not summary["fatal"] and summary["requested_levels_completed"]
        level, = summary["levels"]
        jobs = level["results"]
        assert len(jobs) == 2096 and level["coverage"]["completed_groups"] == 262
        assert set(collections.Counter(j["task"] for j in jobs).values()) == {8}
        assert len({job_key(j) for j in jobs}) == len(jobs)
        assert not analysis["settings"]["fresh_inference"]
        assert analysis["settings"]["evaluation_threads"] == 1
        assert analysis["settings"]["total_declared_cpu_budget"] == 56
        verified = sum(j["verified_correct"] for j in jobs)
        assert verified == level["correct"] == analysis["correct"]
        excluded = {str(key): next(j for j in jobs if job_key(j) == key) for key in DIAGNOSED}
        for key, expected in DIAGNOSED.items():
            assert excluded[str(key)]["candidate_sha256"] == expected
        measured = collections.defaultdict(list)
        native_calls = collections.Counter()
        per_job = {}
        invocation_count = correct_count = 0
        hit_calls = []
        for job in jobs:
            job_measures = collections.defaultdict(float)
            for stage in ("evaluation", "verification"):
                timing = job["stages"].get(stage, {})
                if "host_admission" in timing:
                    wait = timing["host_admission"]["queue_seconds"]
                    measured[stage+"/host_admission_seconds"].append(wait)
                    job_measures["host_admission_seconds"] += wait
            for phase in ("evaluation", "final"):
                base = name + "/jobs/" + job["job"] + "/workspace/" + phase + "-kernelbench"
                if base+".json" in members:
                    native = json.loads(archive.read(base+".json"))
                    metrics = native.get("stage_metrics", {})
                    if not metrics:
                        continue
                    invocation_count += 1
                    if native.get("correctness"):
                        correct_count += 1
                        assert metrics["input_calls"] == 7
                        assert metrics["logical_input_indices"] == list(range(7))
                    native_calls[str(metrics.get("logical_input_indices"))] += 1
                    measured[phase+"/process_startup_seconds"].append(metrics["process_startup_seconds"])
                    totals = collections.defaultdict(float)
                    for event in metrics["events"]:
                        key = event["stage"]
                        measured[phase+"/"+key+"/call_seconds"].append(event["wall_seconds"])
                        totals[key] += event["wall_seconds"]
                        if event["cuda_event_ms"] is not None:
                            measured[phase+"/"+key+"/cuda_seconds"].append(event["cuda_event_ms"]/1000)
                    for key, value in totals.items():
                        measured[phase+"/"+key+"/invocation_seconds"].append(value)
                        job_measures[key] += value
                    reuse_events = metrics.get("input_reuse", {}).get("events", [])
                    if reuse_events:
                        config = json.loads(archive.read(base+"-input-reuse.json"))
                        input_events = [e for e in metrics["events"] if e["stage"] == "input_generation"]
                        assert len(input_events) == len(metrics["logical_input_indices"])
                        starts = dict(zip(metrics["logical_input_indices"], input_events))
                        for event in reuse_events:
                            if event["status"] == "hit":
                                hit_calls.append({"consumer": config["consumer"], "key": event["key"],
                                                  "lookup_wait_seconds": event["lookup_wait_seconds"],
                                                  "start_monotonic_ns": starts[event["trial_index"]]["start_monotonic_ns"]})
                    for event in reuse_events:
                        for key in ("lookup_wait_seconds", "mapping_seconds", "generation_seconds",
                                    "publication_copy_seconds", "wall_seconds"):
                            if key in event:
                                measured["reuse/"+event["status"]+"/"+key].append(event[key])
                if base+"-process.json" in members:
                    process = json.loads(archive.read(base+"-process.json"))
                    for key in ("wall_seconds", "spawn_seconds", "reap_seconds", "admission_check_seconds", "context_cleanup_seconds"):
                        if process.get(key) is not None:
                            measured[phase+"/process/"+key].append(process[key])
            per_job[job_key(job)] = dict(job_measures)
        for key, values in measured.items():
            if key.endswith("/invocation_seconds"):
                phase, stage, _ = key.split("/")
                archived_total = math.fsum(
                    metric["mean"] * metric["count"]
                    for group, metrics in analysis["native"]["groups"].items()
                    if group.endswith("/"+phase+"-kernelbench")
                    and (metric := metrics.get(stage+"_total_wall_seconds")) is not None)
                assert math.isclose(math.fsum(values), archived_total, rel_tol=1e-10, abs_tol=1e-8), (name, key)
        cache = {"enabled": False}
        if name+"/input-cache-initial.json" in members:
            initial = json.loads(archive.read(name+"/input-cache-initial.json"))
            final = json.loads(archive.read(name+"/input-cache-final.json"))
            assert not any(final[k] for k in ("consumers", "references", "generating_bytes",
                                            "consumer_cow_reserved_bytes", "explicit_pinned_bytes", "handoff_copy_bytes"))
            delta = {k: v-initial["metrics"].get(k, 0) for k, v in final["metrics"].items()}
            cache = {"enabled": True, "delta": delta,
                     "hit_rate": delta["hits"]/(delta["hits"]+delta["misses"]),
                     "initial_resident_gib": initial["shared_storage_bytes"]["ram"]/GIB,
                     "final_resident_gib": final["shared_storage_bytes"]["ram"]/GIB,
                     "broker_cpu_seconds": final["owner_cpu_seconds"]-initial["owner_cpu_seconds"]}
        phases = analysis["phases"]
        timeline = telemetry_timeline(archive, name, phases["full"]["started_unix_ns"],
                                      phases["full"]["finished_unix_ns"])
        cache["hit_publication_overlap"] = publication_overlap(archive, name, hit_calls,
                                                              timeline["unix_minus_monotonic_ns"])
        result = {"arm": name, "archive_sha256": digest, "episodes": len(jobs), "tasks": 262,
                  "wall_seconds": level["wall_seconds"], "verified": verified,
                  "verified_per_minute": verified*60/level["wall_seconds"],
                  "completed_native_path": level["completed"], "error_count": level["errors"],
                  "errors_by_type": dict(collections.Counter(j.get("error_type") for j in jobs if not j["completed"])),
                  "latency_seconds": analysis["latency_seconds"], "stages": analysis["stages"],
                  "native_timings": {k: distribution(v) for k, v in measured.items()},
                  "native_record_count": invocation_count, "native_correct_record_count": correct_count,
                  "native_totals_match_archived_analysis": True,
                  "native_logical_input_indices": dict(native_calls),
                  "sensitivity_excluding_task98": (verified-excluded[str(("kernelbench/level_2/98", 0, 3))]["verified_correct"])*60/level["wall_seconds"],
                  "exploratory_sensitivity_excluding_both": (verified-sum(j["verified_correct"] for j in excluded.values()))*60/level["wall_seconds"],
                  "diagnosed_outcomes": {k: {f: j.get(f) for f in ("verified_correct", "development_correct", "candidate_sha256")} for k, j in excluded.items()},
                  "cache": cache, "phases": phases, "echo": analysis["echo"],
                  "fresh_inference": False, "timeline": timeline}
        outcomes = {job_key(j): {k: j.get(k) for k in ("candidate_sha256", "verified_correct", "development_correct", "error_type", "score_eligible", "source_inference")} for j in jobs}
        return result, outcomes, per_job


def make_figure(runs, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(4, 3, figsize=(17, 12), sharex=True, sharey="row", constrained_layout=True)
    colors = ("#176aab", "#b44c3b", "#25846b")
    for column, run in enumerate(runs):
        bins = run["timeline"]["bins"]
        x = [(b["start_seconds"]+b["end_seconds"])/120 for b in bins]
        def y(key):
            return [b["means"].get(key, float("nan")) for b in bins]
        axes[0, column].plot(x, y("visible_cpu_cores"), color=colors[0], label="Visible aggregate")
        axes[0, column].plot(x, y("owned_cpu_cores_lower_bound"), color=colors[1], label="Owned processes (lower bound)")
        for key, label, color in (("memory_used_estimate_gib", "Total minus MemAvailable", colors[0]),
                                   ("evaluator_reserved_gib", "Evaluator reservation", colors[1]),
                                   ("cache_ram_accounted_gib", "Cache RAM accounting", colors[2])):
            axes[1, column].plot(x, y(key), label=label, color=color)
        axes[1, column].axhline(128, color="#6b7280", linestyle=":", linewidth=.8)
        axes[1, column].set_ylim(0, 480)
        axes[2, column].plot(x, y("memory_waiting_jobs"), color=colors[1], label="Waiting for evaluator memory")
        activity = [b["means"].get("gpu_activity_percent", float("nan"))
                    if b["coverage_seconds"].get("gpu_activity_percent", 0)/(b["end_seconds"]-b["start_seconds"]) >= .8
                    else float("nan") for b in bins]
        axes[3, column].plot(x, y("gpu_leased_slots_percent"), color=colors[1], label="GPU admission slots leased")
        axes[3, column].plot(x, activity, color=colors[0], label="Mean sampled GPU activity")
        axes[3, column].set_ylim(0, 105)
        axes[0, column].set_title(run["arm"] + " | " + f'{run["verified_per_minute"]:.3f} verified/min', fontsize=12)
        axes[3, column].set_xlabel("Minutes from arm start")
        for row in range(4):
            axis = axes[row, column]
            axis.grid(alpha=.15)
            axis.set_xlim(0, 140)
            if row != 1:
                axis.set_ylim(bottom=0)
            phase = run["phases"]
            axis.axvspan(0, phase["fill"]["wall_seconds"]/60, color="#737373", alpha=.09)
            axis.axvspan((phase["drain"]["started_unix_ns"]-phase["full"]["started_unix_ns"])/60e9,
                         phase["full"]["wall_seconds"]/60, color="#737373", alpha=.09)
    for row, label in enumerate(("CPU busy core equivalents", "Memory (GiB; total 480)", "Queued jobs", "GPU percentage")):
        axes[row, 0].set_ylabel(label)
        axes[row, 0].legend(loc="upper left", fontsize=8, framealpha=.92)
    fig.suptitle("Closed input-reuse runs on B: one run per mode, frozen inference replay", fontsize=16)
    fig.supxlabel("60-second bins. Shading: fill/drain. CPU quota unknown; CPU-ID scopes differ. GPU line gaps: <80% valid coverage.\nMemory reservations are logical budgets, not physical usage. Three remaining measured runs are incomplete.", fontsize=10)
    fig.savefig(output/"closed-arm-resources.png", dpi=150)
    fig.savefig(output/"closed-arm-resources.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results, outcomes, per_job = [], {}, {}
    for name in ARCHIVES:
        result, outcomes[name], per_job[name] = analyze_arm(args.evidence/f"input-reuse-{name}-evidence.zip", name)
        results.append(result)
        print(json.dumps({k: result[k] for k in ("arm", "wall_seconds", "verified", "verified_per_minute")}))
    paired = {}
    base = outcomes["off-r1"]
    for name in ("cold-r1", "warm-r1"):
        assert outcomes[name].keys() == base.keys()
        changes = []
        totals = collections.defaultdict(list)
        input_by_task = collections.defaultdict(float)
        for key, expected in base.items():
            actual = outcomes[name][key]
            for field in ("candidate_sha256", "error_type", "score_eligible", "source_inference"):
                assert actual[field] == expected[field], (name, key, field)
            assert actual["development_correct"] == expected["development_correct"], (name, key, "development outcome")
            if actual["verified_correct"] != expected["verified_correct"]:
                assert key == ("kernelbench/level_2/98", 0, 3)
                changes.append({"key": key, "before": expected["verified_correct"], "after": actual["verified_correct"]})
            for metric in per_job["off-r1"][key].keys() & per_job[name][key].keys():
                delta = per_job[name][key][metric]-per_job["off-r1"][key][metric]
                totals[metric].append(delta)
                if metric == "input_generation":
                    input_by_task[key[0]] += delta
        paired[name] = {"verified_outcome_changes": changes,
                        "paired_job_delta_seconds": {k: distribution(v) for k, v in totals.items()},
                        "largest_input_stage_increases": sorted(input_by_task.items(), key=lambda x: x[1], reverse=True)[:15],
                        "largest_input_stage_reductions": sorted(input_by_task.items(), key=lambda x: x[1])[:15]}
    result = {"analysis_schema": 1,
              "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "status": "partial; no new experiments or exception amendments authorized",
              "fresh_inference": False, "runs": results, "paired": paired,
              "definitions": [
                  "Raw throughput uses every verified job and the original monotonic arm wall time.",
                  "Sensitivity excludes only the named exact candidates from each numerator; all elapsed time remains. Excluding both is exploratory analysis, not an approved correctness-gate change.",
                  "Native timings are sums of observed intervals across concurrent invocations, not serial wall-time attribution. Missing/failed native records are not zero-filled.",
                  "Input-generation instrumentation encloses cache lookup/generation/mapping/publication when enabled. Cache submetrics overlap it and must not be added again.",
                  "CPU/PSI scope and quotas are provider-dependent; these data do not establish allocated-CPU saturation.",
                  "Fill/middle/drain are scheduling windows, not proof of steady-state stationarity. All three runs executed in off/cold/warm order; no repeat or order control is complete.",
              ]}
    (args.output/"closed-arm-analysis.json").write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    if not args.no_plot:
        make_figure(results, args.output)


if __name__ == "__main__":
    main()
