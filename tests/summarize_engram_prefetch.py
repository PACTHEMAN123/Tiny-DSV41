"""Compare uninstrumented throughput with separately sampled prefetch deadlines."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from summarize_memory_benchmark import distribution, summarize


def percentiles(values):
    values = sorted(values)

    def percentile(fraction):
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {"mean": mean(values), "p50": percentile(.5), "p95": percentile(.95),
            "p99": percentile(.99), "min": values[0], "max": values[-1]}


def diagnostics(reports):
    by_table = defaultdict(list)
    wait_steps, route_steps, prefetch_steps, diagnostic_steps = [], [], [], []
    for report in reports:
        rank_waits, rank_routes, rank_prefetches, rank_steps = [], [], [], []
        for rank in report["ranks"]:
            diagnostic = rank["prefetch_diagnostic"]
            steps = diagnostic["steps"]
            samples = diagnostic["samples"]
            tables = len(report["config"]["engram_layer_ids"])
            expected = steps * report["microbatches"] * tables
            if len(samples) != expected or any("ready_margin_ms" not in sample for sample in samples):
                raise ValueError("incomplete prefetch diagnostic")
            for sample in samples:
                dispatch_end = max(row["prefetch_return"] for row in samples
                                   if row["request"] // tables == sample["request"] // tables)
                sample["ready_before_decoder_dispatch"] = sample["store_ready"] <= dispatch_end
                by_table[sample["table"]].append(sample)
            for target, key in ((rank_waits, "future_wait_ms"), (rank_routes, "route_host_ms"),
                                (rank_prefetches, "prefetch_host_ms")):
                target.append([sum(sample[key] for sample in samples if sample["step"] == step)
                               for step in range(steps)])
            rank_steps.append(diagnostic["step_ms"])
        for target, rows in ((wait_steps, rank_waits), (route_steps, rank_routes),
                             (prefetch_steps, rank_prefetches), (diagnostic_steps, rank_steps)):
            target.extend(max(values) for values in zip(*rows))
    return {
        "tables": {name: {"lookups": len(rows),
                    "ready_fraction": sum(row["ready_before_resolve"] for row in rows) / len(rows),
                    "future_done_fraction": sum(row["future_done_at_resolve"] for row in rows) / len(rows),
                    "ready_before_decoder_dispatch_fraction": sum(
                        row["ready_before_decoder_dispatch"] for row in rows) / len(rows),
                    **{key: percentiles([row[key] for row in rows]) for key in (
                        "ready_margin_ms", "future_wait_ms", "prefetch_host_ms", "store_ms", "route_host_ms")}}
                   for name, rows in by_table.items()},
        "max_rank_future_wait_ms_per_step": percentiles(wait_steps),
        "max_rank_route_host_ms_per_step": percentiles(route_steps),
        "max_rank_prefetch_host_ms_per_step": percentiles(prefetch_steps),
        "max_rank_diagnostic_step_ms": percentiles(diagnostic_steps),
    }


def export_timeline(report, path):
    samples = [(rank["rank"], row) for rank in report["ranks"]
               for row in rank["prefetch_diagnostic"]["samples"] if row["step"] == 0]
    origin = min(row["prefetch_start"] for _, row in samples)
    events = []
    for rank, row in samples:
        for label, start, stop, thread in (
            ("prefetch", "prefetch_start", "prefetch_return", 0),
            ("CPU lookup + staged H2D", "store_start", "store_ready", int(row["table"]) + 1),
            ("resolve", "resolve_start", "resolve_return", 0),
            ("Future.result wait", "future_wait_start", "future_wait_return", 0),
            ("row-return routing", "route_start", "route_return", 0),
        ):
            if start not in row:
                continue
            events.append({"name": f"layer {row['table']}: {label}", "cat": "engram",
                           "ph": "X", "pid": rank, "tid": thread,
                           "ts": (row[start] - origin) * 1e6, "dur": (row[stop] - row[start]) * 1e6,
                           "args": {"ready_before_resolve": row["ready_before_resolve"],
                                    "ready_margin_ms": row["ready_margin_ms"], "local_rows": row["local_rows"]}})
    path.write_text(json.dumps({"traceEvents": events, "displayTimeUnit": "ms",
                               "note": "Host spans on one host; store return follows H2D DMA completion. "
                                       "This trace contains no GPU kernel intervals."}) + "\n")


def summarize_prefetch(folder):
    folder = Path(folder)
    result = summarize(folder)
    if result["failures"]:
        raise ValueError("cannot summarize an incomplete benchmark")
    reports = defaultdict(dict)
    for entry in json.loads((folder / "summary.json").read_text()):
        reports[entry["case"]][entry["repeat"]] = json.loads(Path(entry["report"]).read_text())
    synchronous, asynchronous = reports["cpu"], reports["cpu_async"]
    if not synchronous or set(synchronous) != set(asynchronous):
        raise ValueError("need matching sync and async repeats")
    result["async_to_sync_throughput_ratio"] = distribution([
        asynchronous[key]["tokens_per_second"] / synchronous[key]["tokens_per_second"]
        for key in sorted(synchronous)])
    result["async_sync_max_loss_difference"] = max(
        abs(actual - expected) for key in synchronous
        for left, right in zip(asynchronous[key]["ranks"], synchronous[key]["ranks"])
        for actual, expected in zip(left["losses"], right["losses"]))
    result["prefetch_diagnostics"] = {name: diagnostics(list(rows.values()))
                                      for name, rows in reports.items() if name in {"cpu", "cpu_async"}}
    timeline = folder / "prefetch-host-timeline.json"
    export_timeline(asynchronous[min(asynchronous)], timeline)
    result["prefetch_host_timeline"] = str(timeline)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {Path(folder).name: summarize_prefetch(folder) for folder in args.folders}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    for name, report in result.items():
        print(json.dumps({"workload": name, "async_sync_ratio": report["async_to_sync_throughput_ratio"],
                          "max_loss_difference": report["async_sync_max_loss_difference"],
                          "async_diagnostics": report["prefetch_diagnostics"]["cpu_async"]}, indent=2))


if __name__ == "__main__":
    main()
