"""Summarize the real Qwen/Alpaca factorial benchmark without hiding interference."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median


def flatten(values):
    for value in values:
        if isinstance(value, list):
            yield from flatten(value)
        else:
            yield value


def difference(left, right):
    left, right = list(flatten(left)), list(flatten(right))
    if len(left) != len(right):
        raise ValueError("comparison sample lengths differ")
    return max((abs(a-b) for a, b in zip(left, right)), default=0)


def summarize(root, exclude=()):
    records = []
    for path in sorted(root.glob("repeat-*/summary.json")):
        if path.parent.name in exclude:
            continue
        record = json.loads(path.read_text())
        record["repeat"] = int(path.parent.name.split("-")[1])
        record["path"] = str(path)
        records.append(record)
    if not records:
        raise ValueError("no completed benchmark runs")
    grouped = defaultdict(list)
    for record in records:
        grouped[record["variant"]].append(record)
    output = {"environment": "Eight H20; inspect gpu-processes.csv and per-run telemetry for external workload interference.",
              "excluded_runs": list(exclude),
              "variants": {}, "comparisons": []}
    for variant, runs in grouped.items():
        metrics, waits, loads, host_phases = defaultdict(list), [], [], defaultdict(list)
        for run in runs:
            ranks = run["ranks"]
            steps = ranks[0]["steps"]
            sums = defaultdict(float)
            for rank in ranks:
                loads.append(rank["load_seconds"])
                for phase in rank["phase_host_ms"]:
                    for name, value in phase.items():
                        host_phases[name].append(value)
                for table in rank["engram_metrics"].values():
                    for key, value in table.items():
                        sums[key] += value
            for key in ("h2d_bytes", "d2h_bytes", "cache_metadata_h2d_bytes", "lookup_rows"):
                metrics[key].append(sums[key] / len(ranks) / steps)
            requested = sums["cache_hits"]+sums["cache_misses"]
            metrics["cache_hit_fraction"].append(sums["cache_hits"] / requested if requested else 0)
            waits.append(sums["resolve_wait_seconds"]*1000/len(ranks)/steps)
        output["variants"][variant] = {
            "runs": len(runs), "tokens_per_second": [r["tokens_per_second"] for r in runs],
            "median_tokens_per_second": median(r["tokens_per_second"] for r in runs),
            "median_supervised_tokens_per_second": median(r["supervised_tokens_per_second"] for r in runs),
            "max_allocated_gib": max(r["peak_allocated_gib"] for r in runs),
            "max_reserved_gib": max(r["peak_reserved_gib"] for r in runs),
            "median_load_seconds": median(loads),
            "per_gpu_per_step": {key: mean(values) for key, values in metrics.items()},
            "mean_resolve_host_ms": mean(waits),
            "mean_phase_host_ms": {key: mean(values) for key, values in host_phases.items()},
            "table_first_step_updates": [{"rank": row["rank"], "tables": row["table_updates_first_step"]} for row in runs[0]["ranks"]],
            "paths": [r["path"] for r in runs],
        }
    baseline = grouped.get("gpu", [records[0]])[0]
    for run in records:
        samples, rows, same_ids = 0, 0, True
        for actual, expected in zip(run["ranks"], baseline["ranks"]):
            for name, values in actual["parameter_samples"].items():
                samples = max(samples, difference(values, expected["parameter_samples"][name]))
            for name, values in actual["table_samples"].items():
                reference = expected["table_samples"][name]
                same_ids &= values["ids"] == reference["ids"]
                if values["ids"] == reference["ids"]:
                    rows = max(rows, difference(values["values"], reference["values"]))
        output["comparisons"].append({"variant": run["variant"], "repeat": run["repeat"],
            "max_global_loss_difference": difference(run["global_losses"], baseline["global_losses"]),
            "max_parameter_sample_difference": samples, "same_table_sample_ids": same_ids,
            "max_table_sample_difference": rows})
    output["workload"] = {key: baseline["ranks"][0][key] for key in (
        "world_size", "ep", "cp", "batch_size", "microbatches", "steps", "sequence_length",
        "global_batch", "data", "engram_config", "engram_parameters")}
    return output


def timelines(root, exclude=()):
    from summarize_memory_timeline import select_events, host_totals, training_overview, detail_timeline
    for path in sorted(root.glob("repeat-*/trace/rank-0.json")):
        if path.parent.parent.name in exclude:
            continue
        summary = json.loads(path.with_name("rank-0-summary.json").read_text())
        events = select_events(json.loads(path.read_text()))
        record = {"rank": 0, "events": events}
        output = path.parent
        training_overview([record], output/"timeline.html", 5, title="Qwen3-30B + Engram")
        detail_timeline(record, "training", output/"detail.html", title="Qwen3-30B + Engram")
        report = {"scope": "rank 0 of an eight-rank run; profiled separately from throughput",
                  "host_scopes": host_totals(events, 1), "cuda_activity": summary["cuda_activity"],
                  "components": summary["components"]}
        margins = {}
        names = {event["name"].split("/")[-1] for event in events
                 if event["name"].startswith("dsv41/engram_store/")}
        for name in names:
            stores = sorted((e for e in events if e["name"] == "dsv41/engram_store/"+name),
                            key=lambda e: e["start_us"])
            resolves = sorted((e for e in events if e["name"] == "dsv41/engram_resolve/"+name),
                              key=lambda e: e["start_us"])
            margins[name] = [(resolve["start_us"]-store["start_us"]-store["duration_us"])/1000
                             for store, resolve in zip(stores, resolves)]
        report["profiled_store_ready_margin_ms"] = margins
        report["readiness_note"] = "Profiler-affected host observations; positive is ready before resolve. Row-return NCCL is excluded."
        (output/"breakdown.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timelines", action="store_true")
    parser.add_argument("--exclude-run", nargs="*", default=[])
    args = parser.parse_args()
    result = summarize(args.root, args.exclude_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.timelines:
        timelines(args.root, args.exclude_run)
    for variant, row in result["variants"].items():
        print(json.dumps({"variant": variant, "tokens_per_second": row["median_tokens_per_second"],
                          "peak_gib": row["max_allocated_gib"], "metrics": row["per_gpu_per_step"]}), flush=True)
    print(json.dumps({"comparisons": result["comparisons"]}), flush=True)


if __name__ == "__main__":
    main()
