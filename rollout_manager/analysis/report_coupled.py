"""Extract measured A/B intervals and plot an explicitly review-free projection."""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
import zipfile

from rollout_manager.adapters.candidates import parse_candidate
from rollout_manager.common import digest

COLORS = {"queue": "#a47b12", "prefill": "#456ea8", "decode": "#16897e",
          "rpc": "#b9dcd7", "build": "#995b35",
          "evaluation": "#a65e91", "verification": "#557e30", "admission": "#d59a21",
          "input_generation": "#c45b58", "h2d": "#cf9a28", "reference_execution": "#456ea8",
          "candidate_execution": "#16897e", "validation": "#a65e91",
          "native_scoring": "#557e30", "native_cleanup": "#747b80"}


def verified_archive(path):
    archive = zipfile.ZipFile(path)
    manifest = json.loads(archive.read("SHA256-MANIFEST.json"))
    for name, identity in manifest.items():
        raw = archive.read(name)
        if len(raw) != identity["bytes"] or hashlib.sha256(raw).hexdigest() != identity["sha256"]:
            raise ValueError("archive member changed: " + name)
    if set(archive.namelist()) != set(manifest) | {"SHA256-MANIFEST.json"}:
        raise ValueError("unmanifested archive members")
    return archive


def read(archive, name):
    return json.loads(archive.read(name))


def clock_bounds(before, after):
    def best(rows):
        bounds = [(r["server_response_unix_ns"] - r["client_received_unix_ns"],
                   r["server_received_unix_ns"] - r["client_started_unix_ns"]) for r in rows]
        if any(lo > hi for lo, hi in bounds):
            raise ValueError("invalid clock probe")
        return min(bounds, key=lambda pair: pair[1] - pair[0])
    first, last = best(before), best(after)
    low, high = min(first[0], last[0]), max(first[1], last[1])
    return {"A_minus_B_ns": (low + high) // 2, "uncertainty_ns": (high - low + 1) // 2,
            "before_bounds_ns": first, "after_bounds_ns": last,
            "assumption": "offset stays within the envelope of the two tightest endpoint probes; no hidden clock step"}


def engine_match(records, response):
    matches = [r for r in records if r["response_metrics"] == response["metrics"]
               and r["completion_tokens"] == response["usage"]["completion_tokens"]]
    if len(matches) != 1:
        raise ValueError(f"engine metric join is not unique: {len(matches)}")
    return matches[0]


def extract_run(archive, name, engine, catalog):
    level, = read(archive, name + "/summary.json")["levels"]
    if read(archive, name + "-metadata/exit.json")["returncode"] != 0:
        raise ValueError("run did not close normally")
    clock = clock_bounds(read(archive, name + "-metadata/clock-before.json"),
                         read(archive, name + "-metadata/clock-after.json"))
    events = [json.loads(line) for line in archive.read(name + "/events.jsonl").splitlines()]
    origin = level["started_unix_ns"]
    rows, jobs = [], []

    def add(job, machine, stage, start, duration, scope="stage", **extra):
        row = {"session": name, "job": job["job"], "sample": job["rollout"]["sample_index"],
               "machine": machine, "scope": scope, "stage": stage,
               "start_seconds": (round(start) - origin) / 1e9, "duration_seconds": duration,
               "end_seconds": (round(start) - origin) / 1e9 + duration, **extra}
        if duration < 0:
            raise ValueError("negative interval")
        rows.append(row)

    for job in sorted(level["results"], key=lambda j: j["rollout"]["sample_index"]):
        base = name + "/jobs/" + job["job"]
        inference = read(archive, base + "/inference.json")
        response, comm = inference["response"], inference["communication"]
        candidate = parse_candidate(response["choices"][0]["message"]["content"])
        if candidate != read(archive, base + "/candidate.json") or digest(candidate) != job["candidate_sha256"]:
            raise ValueError("evaluated candidate differs from fresh inference")
        if catalog[digest(candidate)]["decision"] != "approve":
            raise ValueError("candidate lacks exact-source approval")
        for path, content in candidate.items():
            if archive.read(base + "/workspace/" + path) != content.encode():
                raise ValueError("workspace source differs from reviewed candidate")
        match = engine_match(engine, response)
        a_offset = match["unix_ns"] - (match["monotonic_before_ns"] + match["monotonic_after_ns"]) // 2
        phase = match["engine_timestamps"]
        keys = [("queue", "queued_ts", "scheduled_ts"),
                ("prefill", "scheduled_ts", "first_token_ts"),
                ("decode", "first_token_ts", "last_token_ts")]
        for label, start, end in keys:
            add(job, "A", label, round(phase[start] * 1e9) + a_offset - clock["A_minus_B_ns"],
                phase[end] - phase[start], alignment_uncertainty_ns=clock["uncertainty_ns"])
        engine_rows = rows[-3:]
        rpc_start = (comm["client_started_unix_ns"] - origin) / 1e9
        rpc_end = rpc_start + comm["client_total_seconds"]
        tolerance = clock["uncertainty_ns"] / 1e9
        if engine_rows[0]["start_seconds"] < rpc_start - tolerance or engine_rows[-1]["end_seconds"] > rpc_end + tolerance:
            raise ValueError("aligned engine phases fall outside their RPC")
        add(job, "B", "rpc", comm["client_started_unix_ns"], comm["client_total_seconds"])
        mono = comm["client_monotonic_ns"]
        b_offset = comm["client_started_unix_ns"] - mono["start"]
        for stage, start, end in [("request_send", "send_start", "sent"),
                                  ("response_body", "headers", "received")]:
            add(job, "A/B", stage, mono[start] + b_offset, (mono[end] - mono[start]) / 1e9, "communication")
        relay_offset = inference["server_received_unix_ns"] - inference["server_received_monotonic_ns"]
        for stage, start, end in [
            ("relay_dispatch", "server_received_monotonic_ns", "upstream_started_monotonic_ns"),
            ("vllm_http", "upstream_started_monotonic_ns", "upstream_finished_monotonic_ns"),
            ("relay_package", "upstream_finished_monotonic_ns", "server_response_monotonic_ns"),
        ]:
            add(job, "A", stage, inference[start] + relay_offset - clock["A_minus_B_ns"],
                (inference[end] - inference[start]) / 1e9, "communication")
        own_events = [e for e in events if e["job"] == job["job"]]
        review = next(e for e in own_events if e["stage"] == "review" and e["event"] == "queued")
        if review["unix_ns"] < comm["client_received_unix_ns"]:
            raise ValueError("candidate review precedes its response")
        add(job, "B", "review", review["unix_ns"], job["review_wait_seconds"])
        b_offset = int(statistics.median(e["unix_ns"] - e["monotonic_ns"] for e in own_events))
        memory = iter(e for e in own_events if e["stage"] == "evaluator_host_admission" and e["event"] == "queued")
        for stage, timing in job["stages"].items():
            if stage == "inference":
                continue
            if "host_admission" in timing:
                event = next(memory)
                add(job, "B", stage + "/memory_wait", event["unix_ns"], timing["host_admission"]["queue_seconds"], "admission")
            queued = next(e for e in own_events if e["stage"] == stage and e["event"] == "queued")
            add(job, "B", stage + "/queue", queued["unix_ns"], timing["queue_seconds"], "admission")
            add(job, "B", stage, timing["started_unix_ns"], timing["service_seconds"])
        native = {}
        for phase_name in ("compile", "evaluation-kernelbench", "final-kernelbench"):
            process_path = base + "/workspace/" + phase_name + "-process.json"
            if process_path not in archive.namelist():
                continue
            process = read(archive, process_path)
            if not process["process_group_stopped"] or not process["gpu_contexts_released"]:
                raise ValueError("undrained process/GPU ownership")
            add(job, "B", phase_name + "/process", process["started_unix_ns"], process["wall_seconds"], "process")
            add(job, "B", phase_name + "/context_cleanup",
                process["started_unix_ns"] + round(process["wall_seconds"] * 1e9),
                process["context_cleanup_seconds"], "process")
            if phase_name == "compile":
                native[phase_name] = {"process": process}
                continue
            result = read(archive, base + "/workspace/" + phase_name + ".json")
            metrics = result["stage_metrics"]
            expected_seed = 1042 if phase_name == "final-kernelbench" else 42
            if result["seed"] != expected_seed:
                raise ValueError("verification seed namespace changed")
            if result["correctness"] and (
                result["metadata"]["correctness_trials"] != "(5 / 5)"
                or metrics["logical_input_indices"] != list(range(7))
                or len(result["timing_samples"]) != 2
                or any(len(t["samples_ms"]) != 10 for t in result["timing_samples"])
            ):
                raise ValueError("successful native verification/scoring workload changed")
            totals = collections.defaultdict(float)
            index = -1
            for event in metrics["events"]:
                if event["stage"] == "input_generation":
                    index += 1
                logical = metrics["logical_input_indices"][index] if index >= 0 else None
                totals[event["stage"]] += event["wall_seconds"]
                add(job, "B", event["stage"], event["start_monotonic_ns"] + b_offset,
                    event["wall_seconds"], "native", phase=phase_name, input_index=logical,
                    cuda_event_ms=event["cuda_event_ms"])
            native[phase_name] = {"process": process, "totals_seconds": dict(totals),
                "startup_seconds": metrics["process_startup_seconds"], "seed": result["seed"],
                "correctness": result["correctness"], "runtime_ms": result["runtime"],
                "reference_runtime_ms": result["ref_runtime"], "metadata": result["metadata"],
                "logical_input_indices": metrics["logical_input_indices"], "timing_samples": result["timing_samples"]}
        add(job, "B", "result", job["finished_unix_ns"], 0, "boundary")
        jobs.append({k:v for k,v in job.items() if k not in ("feedback", "development_timing")}
                    | {"engine": match, "native": native, "inference_usage": response["usage"],
                       "relay_timestamps": {k: v for k, v in inference.items() if k.endswith("_ns")},
                       "inference_metrics": response["metrics"]})
    start = min(j["communication"]["client_started_unix_ns"] for j in jobs)
    finish = max(j["communication"]["client_received_unix_ns"] for j in jobs)
    return {"origin_unix_ns": origin, "wall_seconds": level["wall_seconds"], "clock": clock,
            "jobs": jobs, "intervals": rows, "completion_tokens": level["completion_tokens"],
            "inference_span_seconds": (finish - start) / 1e9,
            "inference_tokens_per_second": level["completion_tokens"] / ((finish - start) / 1e9),
            "verified": level["correct"], "profiled": "profile" in name}


def without_review(run):
    rows, jobs = [], []
    for job in run["jobs"]:
        own = [row for row in run["intervals"] if row["job"] == job["job"]]
        reviews = [row for row in own if row["stage"] == "review" and row["scope"] == "stage"]
        if len(reviews) != 1:
            raise ValueError("projection requires exactly one review interval per job")
        review, = reviews
        for row in own:
            if row is review:
                continue
            shift = 0
            if row["machine"] == "B" and row["stage"] != "rpc":
                if row["start_seconds"] < review["end_seconds"] - 1e-6:
                    raise ValueError("B tool work unexpectedly precedes completed review")
                shift = review["duration_seconds"]
            rows.append(row | {
                "recorded_start_seconds": row["start_seconds"],
                "recorded_end_seconds": row["end_seconds"],
                "start_seconds": row["start_seconds"] - shift,
                "end_seconds": row["end_seconds"] - shift,
                "shift_seconds": shift,
            })
        jobs.append({key: job[key] for key in ("job", "rollout", "verified_correct")}
                    | {"omitted_wait_seconds": review["duration_seconds"]})
    finish = max(row["end_seconds"] for row in rows if row["stage"] == "result")
    return {"kind": "projection", "clock": run["clock"], "jobs": jobs,
            "recorded_origin_unix_ns": run["origin_unix_ns"],
            "projected_span_seconds": finish, "intervals": rows,
            "method": "Remove each job's review wait and shift its subsequent B intervals by that duration.",
            "assumptions": "Measured service times, queue delays and outcomes held fixed; changed resource contention and device assignment are not simulated. Profiler overhead remains."}


def draw(run, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    jobs, rows = run["jobs"], run["intervals"]
    single = len(jobs) == 1
    fig, axes = plt.subplots(2 if not single else 3, 1, figsize=(14, 10 if not single else 9),
                             gridspec_kw={"height_ratios": [2.5, 1.2] if not single else [1.1, .8, 1.4]})
    labels = []
    for index, job in enumerate(jobs):
        sample = job["rollout"]["sample_index"]
        for machine in ("A", "B"):
            y = index * 2 + (machine == "B")
            labels.append(f"{machine} / sample {sample}")
            for row in rows:
                if row["job"] != job["job"] or row["machine"] != machine or row["scope"] != "stage":
                    continue
                axes[0].barh(y, row["duration_seconds"], left=row["start_seconds"], height=.64,
                             color=COLORS[row["stage"]])
        end, = [row["end_seconds"] for row in rows if row["job"] == job["job"] and row["stage"] == "result"]
        axes[0].plot(end, index * 2 + 1, marker="|", color="#222222", markersize=12)
        axes[0].annotate("pass" if job["verified_correct"] else "fail", (end, index * 2 + 1),
                         xytext=(5, 0), textcoords="offset points", va="center", fontsize=9)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, run["projected_span_seconds"] * 1.06)
    axes[0].set_title(title, loc="left", fontweight="bold")
    axes[0].set_xlabel("Projected seconds from B run start")
    for index, job in enumerate(jobs):
        for row in rows:
            if row["job"] == job["job"] and row["machine"] == "A" and row["scope"] == "stage":
                axes[1].barh(index, row["duration_seconds"], left=row["start_seconds"], height=.65,
                             color=COLORS[row["stage"]])
    axes[1].set_yticks(range(len(jobs)), [f"A / sample {j['rollout']['sample_index']}" for j in jobs])
    axes[1].invert_yaxis()
    last = max(r["end_seconds"] for r in rows if r["machine"] == "A" and r["scope"] == "stage")
    axes[1].set_xlim(0, last * 1.015)
    axes[1].set_title("Inference detail: queue, prefill and decode on each A lane; same time origin", loc="left")
    axes[1].set_xlabel("Seconds from B run start; measured A inference intervals")
    if single:
        stages = ["input_generation", "h2d", "reference_execution", "candidate_execution", "validation", "native_scoring", "native_cleanup"]
        nr = [r for r in rows if r["scope"] == "native"]
        for row in nr:
            axes[2].barh(stages.index(row["stage"]), row["duration_seconds"], left=row["start_seconds"],
                         height=.65, color=COLORS[row["stage"]])
        axes[2].set_yticks(range(len(stages)), [s.replace("_", " ") for s in stages])
        axes[2].invert_yaxis()
        axes[2].set_xlim(min(r["start_seconds"] for r in nr) - .2, max(r["end_seconds"] for r in nr) + .2)
        axes[2].set_title("B native intervals inside development and final tool calls (nested, not additive)", loc="left")
        axes[2].set_xlabel("Projected seconds from B run start")
    for ax in axes:
        ax.grid(axis="x", color="#e1e4e6", linewidth=.6)
        ax.set_axisbelow(True)
    legend = ["queue", "prefill", "decode", "rpc", "build", "evaluation", "verification"]
    names = {"rpc": "B waits for RPC", "build": "Compile tool",
             "evaluation": "Development tool", "verification": "Final verifier"}
    fig.legend(handles=[Patch(color=COLORS[k], label=names.get(k, k)) for k in legend],
               loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, .005))
    fig.tight_layout(rect=(0, .06, 1, .96))
    fig.text(.02, .965, "Projection: review wait removed; measured service durations and profiler overhead retained. One turn/sample.", fontsize=10)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def draw_graph_sample(analysis, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    sample = analysis["sample"]
    origin = sample["start_seconds"] * 1e9
    fig, ax = plt.subplots(figsize=(14, 5))
    labels = []
    for tid, calls in sorted(sample["host"].items(), key=lambda item: int(item[0])):
        rank = analysis["worker_ranks"][str(analysis["workers"][tid]["pid"])]
        labels.extend([f"TP rank {rank}: host graph launch", f"GPU {rank}: recorded graph"])
        for y, records, color in [(2 * rank, calls, "#456ea8"),
                                   (2 * rank + 1, sample["gpu"].get(str(rank), []), "#16897e")]:
            for event in records:
                ax.barh(y, (event["end"] - event["start"]) / 1e6,
                        left=(event["start"] - origin) / 1e6, height=.65, color=color)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, (sample["end_seconds"] - sample["start_seconds"]) * 1000)
    ax.set_xlabel(f"Milliseconds from Nsight capture time {sample['start_seconds']:.6f} s")
    ax.set_title("CUDA graph replay: an actually recorded 40 ms window of single-request decode", loc="left")
    ax.grid(axis="x", color="#e1e4e6", linewidth=.6)
    ax.set_axisbelow(True)
    fig.legend(handles=[Patch(color="#456ea8", label="CPU cudaGraphLaunch"),
                        Patch(color="#16897e", label="GPU graph activity (may include collective waiting)")],
               loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archives = {name: verified_archive(args.evidence / name) for name in ("A-evidence.zip", "B-controls.zip", "B-profiles.zip")}
    engine = [json.loads(line) for line in archives["A-evidence.zip"].read("a-r1/engine-timing.jsonl").splitlines()]
    catalog = read(archives["B-profiles.zip"], "catalog.json")
    result = {"provenance": {name: hashlib.file_digest((args.evidence / name).open("rb"), "sha256").hexdigest() for name in archives}, "runs": {}}
    for kind, file in (("control", "B-controls.zip"), ("profile", "B-profiles.zip")):
        for mode in ("single", "group"):
            name = f"{mode}-{kind}-r1"
            result["runs"][name] = extract_run(archives[file], name, engine, catalog)
    (args.output / "coupled-timelines.json").write_text(json.dumps(result, indent=2) + "\n")
    projected = {"provenance": result["provenance"],
                 "runs": {name: without_review(run) for name, run in result["runs"].items()}}
    (args.output / "projected-timelines.json").write_text(json.dumps(projected, indent=2) + "\n")
    for filename, data in (("coupled-steps.csv", result), ("projected-steps.csv", projected)):
        rows = [row for run in data["runs"].values() for row in run["intervals"]]
        with (args.output / filename).open("w") as output:
            writer = csv.DictWriter(output, fieldnames=sorted({k for row in rows for k in row}))
            writer.writeheader(); writer.writerows(rows)
    for mode in ("single", "group"):
        draw(projected["runs"][f"{mode}-profile-r1"], args.output / ("single-rollout.png" if mode == "single" else "rollout-group.png"),
             "Single rollout: L2/30 sample 4" if mode == "single" else "Rollout group: L2/30, G=8")
    profiles = {mode: read(archives["A-evidence.zip"], f"a-r1/{mode}-profile-r1/analysis.json")
                for mode in ("single", "group")}
    ranks = {pid: int(rank) for pid, rank in re.findall(
        r"\(Worker pid=(\d+)\).*world_size=4 rank=(\d+)",
        archives["A-evidence.zip"].read("a-r1/server.log").decode())}
    for profile in profiles.values():
        profile["worker_ranks"] = ranks
    (args.output / "cuda-graph-profile.json").write_text(json.dumps(profiles, indent=2) + "\n")
    draw_graph_sample(profiles["single"], args.output / "cuda-graph-profile.png")
    for archive in archives.values():
        archive.close()


if __name__ == "__main__":
    main()
