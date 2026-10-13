"""Summarize isolated memory-policy trials without treating trace time as throughput."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import mean, median


def distribution(values):
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("measurement samples must be nonempty and finite")
    return {"median": median(values), "min": min(values), "max": max(values), "samples": values}


def summarize_trial(report):
    ranks = report["ranks"]
    if sorted(row["rank"] for row in ranks) != list(range(report["world"])):
        raise ValueError("benchmark report is missing ranks")
    if any(len(row["losses"]) != report["steps"] or
           ("step_ms" in row and len(row["step_ms"]) != report["steps"]) for row in ranks):
        raise ValueError("benchmark report is missing measured steps")
    if any(not math.isfinite(loss) for row in ranks for loss in row["losses"]):
        raise ValueError("nonfinite training loss")
    result = {"tokens_per_second": report["tokens_per_second"]}
    for key in ("load_seconds", "load_peak_bytes", "train_peak_bytes", "allocated_bytes",
                "reserved_bytes", "cpu_max_rss_bytes"):
        result[key] = max(row[key] for row in ranks)
    result["logical_bytes_read"] = sum(row["loading"]["logical_bytes_read"] for row in ranks)
    for key in ("h2d_bytes", "d2h_bytes", "lookup_rows"):
        result[f"engram_{key}_per_step"] = sum(
            table[key] for row in ranks for table in row["engram"].values()) / report["steps"]
    for key in ("cache_hits", "cache_misses", "cache_fill_rows", "cache_invalidated_rows", "cache_metadata_h2d_bytes"):
        result[f"engram_{key}_per_step"] = sum(
            table.get(key, 0) for row in ranks for table in row["engram"].values()) / report["steps"]
    for key in ("engram_cache_bytes", "engram_pinned_bytes"):
        result[key] = max(row.get(key, 0) for row in ranks)
    result["engram_resolve_host_ms_per_step_max_rank"] = max(
        sum(table["resolve_wait_seconds"] for table in row["engram"].values())
        * 1000 / report["steps"] for row in ranks)
    result["engram_lookup_rows_by_rank_per_step"] = [
        sum(table["lookup_rows"] for table in row["engram"].values()) / report["steps"] for row in ranks]
    if all("step_ms" in row for row in ranks):
        result["slowest_rank_step_ms"] = [max(values) for values in zip(*(row["step_ms"] for row in ranks))]
    result["losses_by_rank"] = [row["losses"] for row in ranks]
    result["allocator_counters_by_rank"] = [row.get("allocator_counters_during_timing") for row in ranks]
    return result


def summarize_profile(paths):
    rows = [json.loads(Path(path).read_text()) for path in paths]
    regions = defaultdict(list)
    activity = defaultdict(list)
    for row in rows:
        totals = defaultdict(float)
        for span in row.get("lookup_spans", []):
            totals[span["region"]] += (span["stop"] - span["start"]) * 1000 / row["sampled_steps"]
        for key, value in totals.items():
            regions[key].append(value)
        timeline = row["cuda_activity"]
        for key in ("device_busy_ms", "h2d_compute_overlap_ms", "h2d_compute_overlap_fraction"):
            activity[key].append(timeline[key])
        for kind, values in timeline["activities"].items():
            for key, value in values.items():
                activity[f"{kind}_{key}"].append(value)
    return {"sampled_ranks": len(rows),
            "note": "Profiled samples only. Region durations are inclusive and overlap; NCCL is excluded from compute overlap.",
            "activity_by_rank": {key: {"mean": mean(values), "max": max(values), "samples": values}
                                 for key, values in activity.items()},
            "lookup_host_ms_by_rank": {key: {"mean": mean(values), "max": max(values), "samples": values}
                                       for key, values in regions.items()}}


def summarize(folder):
    folder = Path(folder)
    entries = json.loads((folder / "summary.json").read_text())
    variants, failures = defaultdict(list), []
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest is not None:
        finished = {(entry["case"], entry.get("repeat", 0)) for entry in entries}
        failures.extend({**entry, "error": "scheduled trial has not completed"}
                        for entry in manifest["schedule"] if (entry["case"], entry["repeat"]) not in finished)
    profiles = {}
    implementations = set()
    workload = None
    for entry in entries:
        if entry["exit_code"]:
            failures.append(entry)
            continue
        label = f"{entry['case']}-r{entry.get('repeat', 0)}"
        path = Path(entry.get("report", folder / f"{entry['case']}.json"))
        if not path.is_file():
            failures.append({**entry, "error": "missing report"})
            continue
        report = json.loads(path.read_text())
        implementations.add(report.get("loading_implementation", "not_recorded"))
        if report["variant"] != entry["case"]:
            raise ValueError("benchmark variant does not match the run summary")
        identity = {key: report[key] for key in ("world", "cp", "ep", "steps", "warmup",
                                                "microbatches", "sequence_length", "config", "torch", "gpu")}
        identity["learning_rate"] = report.get("learning_rate", .01)
        identity["input_pattern"] = report.get("input_pattern", "fixed")
        identity["batch_size"] = report.get("batch_size", 1)
        identity["activation_checkpoint"] = report.get("activation_checkpoint", "none")
        identity["progress_logging"] = report.get("progress_logging", False)
        identity["memory_limit_gib"] = report.get("memory_limit_gib", 0)
        identity["load_profiled"] = report.get("load_profiled", False)
        if workload is not None and workload != identity:
            raise ValueError("cannot compare different benchmark workloads")
        workload = identity
        trial = summarize_trial(report)
        trial["label"] = label
        trial["repeat"] = entry.get("repeat", 0)
        variants[entry["case"]].append(trial)
        paths = [row["profile_summary"] for row in report["ranks"] if "profile_summary" in row]
        if paths:
            if len(paths) != report["world"]:
                raise ValueError("profile summaries must cover every rank")
            profiles[label] = summarize_profile(paths)
    baseline = {trial["repeat"]: trial for trial in variants.get("baseline", [])}
    comparisons = {}
    for name, trials in variants.items():
        metrics = {key: distribution([trial[key] for trial in trials]) for key, value in trials[0].items()
                   if isinstance(value, (int, float)) and key != "repeat"}
        losses, ratios = [], []
        for trial in trials:
            reference = baseline.get(trial["repeat"])
            if reference is not None:
                ratios.append(trial["tokens_per_second"] / reference["tokens_per_second"])
                losses.append(max(abs(left - right) for actual, expected in
                                  zip(trial["losses_by_rank"], reference["losses_by_rank"])
                                  for left, right in zip(actual, expected)))
        comparisons[name] = {"completed_trials": len(trials), "metrics": metrics,
                             "throughput_ratio_to_baseline": distribution(ratios) if ratios else None,
                             "max_loss_difference_from_baseline": max(losses) if losses else None,
                             "trials": trials}
    return {"workload": workload, "loading_implementations": sorted(implementations),
            "planned_trials": len(manifest["schedule"]) if manifest else None,
            "completed_trials": sum(map(len, variants.values())),
            "failures": failures, "variants": comparisons, "profiles": profiles,
            "note": "Median and range are descriptive, not confidence intervals. Peak memory is the maximum over ranks; transfer bytes are global sums."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = summarize(args.folder)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("| Variant | Trials | Tokens/s median [min, max] | Load peak MiB | Train peak MiB |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, row in report["variants"].items():
        metrics = row["metrics"]
        speed = metrics["tokens_per_second"]
        print(f"| {name} | {row['completed_trials']} | {speed['median']:.2f} [{speed['min']:.2f}, {speed['max']:.2f}] "
              f"| {metrics['load_peak_bytes']['median']/1024**2:.2f} | {metrics['train_peak_bytes']['median']/1024**2:.2f} |")
    if report["failures"]:
        raise SystemExit("Some trials failed or are missing; see the output report.")


if __name__ == "__main__":
    main()
