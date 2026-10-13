"""Cache hit rates, total H2D traffic, paired throughput, and route synchronization."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from summarize_engram_prefetch import diagnostics
from summarize_memory_benchmark import distribution, summarize


def cache_workload(folder):
    folder = Path(folder)
    result = summarize(folder)
    if result["failures"]:
        raise ValueError("cannot summarize incomplete cache trials")
    variants = result["variants"]
    if "cpu_cache" in variants and "cpu_async" in variants:
        reference = {row["repeat"]: row for row in variants["cpu_async"]["trials"]}
        trials = variants["cpu_cache"]["trials"]
        if set(reference) != {row["repeat"] for row in trials}:
            raise ValueError("cache comparisons require matching repeats")
        result["cache_to_async_throughput_ratio"] = distribution([
            row["tokens_per_second"] / reference[row["repeat"]]["tokens_per_second"] for row in trials])
    reports = defaultdict(list)
    for entry in json.loads((folder / "summary.json").read_text()):
        report = json.loads(Path(entry["report"]).read_text())
        reports[entry["case"]].append(report)
    result["cache_accounting"] = {}
    for name, trials in variants.items():
        ratios, totals = [], []
        for trial in trials["trials"]:
            hit = trial["engram_cache_hits_per_step"]
            miss = trial["engram_cache_misses_per_step"]
            ratios.append(hit / (hit + miss) if hit + miss else 0.)
            totals.append(trial["engram_h2d_bytes_per_step"] + trial["engram_cache_metadata_h2d_bytes_per_step"])
        result["cache_accounting"][name] = {"hit_fraction": distribution(ratios),
            "row_and_cache_metadata_h2d_bytes_per_step": distribution(totals)}
    result["prefetch_diagnostics"] = {name: diagnostics(rows) for name, rows in reports.items()
        if all("prefetch_diagnostic" in row["ranks"][0] for row in rows)}
    return result


def route_accounting(folder):
    records = []
    for path in sorted(Path(folder).glob("rank-*.json")):
        if path.name.endswith("-summary.json"):
            continue
        events = json.loads(path.read_text())["traceEvents"]
        spans = [row for row in events if row.get("cat") == "user_annotation"
                 and row.get("name", "").startswith("dsv41/engram_id_route/")]
        samples = defaultdict(lambda: {"calls": 0, "host_ms": 0.})
        for row in events:
            if row.get("cat") not in {"cpu_op", "cuda_runtime"}:
                continue
            if row.get("name") not in {"aten::_local_scalar_dense", "cudaStreamSynchronize", "cudaMemcpyAsync"}:
                continue
            if any(row.get("tid") == span.get("tid") and span["ts"] <= row.get("ts", -1)
                   and row.get("ts", -1) + row.get("dur", 0) <= span["ts"] + span["dur"] for span in spans):
                samples[row["name"]]["calls"] += 1
                samples[row["name"]]["host_ms"] += row.get("dur", 0) / 1000
        records.append({"rank": int(path.stem.split("-")[1]), "trace": str(path),
                        "route_calls": len(spans), "route_host_ms": sum(row["dur"] for row in spans) / 1000,
                        "operations": dict(samples)})
    if not records:
        raise ValueError("no route traces found")
    return {"ranks": records, "route_host_ms": distribution([row["route_host_ms"] for row in records]),
            "note": "Raw profiled capture totals, inclusive host time. Scalar reads, memcpy, and synchronization overlap; do not add them."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", nargs="+")
    parser.add_argument("--route-traces", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = {"workloads": {Path(folder).name: cache_workload(folder) for folder in args.folders},
              "route_profiles": {folder: route_accounting(folder) for folder in args.route_traces}}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    for name, workload in result["workloads"].items():
        print(json.dumps({"workload": name,
            "cache_to_async": workload.get("cache_to_async_throughput_ratio"),
            "accounting": workload["cache_accounting"]}, indent=2))


if __name__ == "__main__":
    main()
