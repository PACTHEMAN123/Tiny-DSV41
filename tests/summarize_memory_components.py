"""Portable summary of the CPU memory-component profiling fallback."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


def samples(values):
    return {"median": median(values), "min": min(values), "max": max(values), "samples": values}


def span_totals(spans):
    result = defaultdict(lambda: {"calls": 0, "inclusive_ms": 0., "tensor_bytes": 0})
    for span in spans:
        key = span["region"] + "/" + span.get("domain", "all")
        row = result[key]
        row["calls"] += 1
        row["inclusive_ms"] += (span["stop"] - span["start"]) * 1000
        row["tensor_bytes"] += span.get("tensor_bytes", 0)
    return dict(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder")
    parser.add_argument("--output", required=True)
    parser.add_argument("--historical-cuda", help="optional earlier profile_dsv4 JSON; explicitly labelled historical")
    args = parser.parse_args()
    folder = Path(args.folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    completed = json.loads((folder / "summary.json").read_text())
    if {(item["case"], item["repeat"]) for item in completed if item["exit_code"] == 0} != {
            (item["case"], item["repeat"]) for item in manifest["schedule"]}:
        raise ValueError("profiling schedule is incomplete or contains failures")
    trials = defaultdict(list)
    for item in completed:
        trials[item["case"]].append(json.loads((folder / f"{item['case']}-r{item['repeat']}.json").read_text()))
    baselines = {row["repeat"]: row for row in trials["layerwise"]}
    result = {"device": "cpu", "current_cuda_measurement": False,
              "loading_implementations": sorted({report.get("loading_implementation", "not_recorded")
                                                 for reports in trials.values() for report in reports}),
              "configuration": manifest["arguments"], "config": trials["layerwise"][0]["config"],
              "parameters": trials["layerwise"][0]["parameters"],
              "torch": trials["layerwise"][0]["torch"], "variants": {},
              "limitations": trials["layerwise"][0]["limitations"] + [
                  "Profiled region times are inclusive and may overlap across threads",
                  "Ranges are descriptive; three trials do not establish a confidence interval"]}
    for name, reports in trials.items():
        row = {"completed_trials": len(reports)}
        for key in ("first_load_ms", "warm_load_ms", "tokens_per_second", "cpu_max_rss_bytes"):
            row[key] = samples([report[key] for report in reports])
        row["timing_ms"] = {stage: samples([report["timing_ms"][stage] for report in reports])
                            for stage in reports[0]["timing_ms"]}
        row["max_loss_difference_from_baseline"] = max(abs(left - right) for report in reports
            for left, right in zip(report["losses"], baselines[report["repeat"]]["losses"]))
        row["parameter_bytes"] = reports[0]["parameter_bytes"]
        row["logical_bytes_read"] = reports[0]["loading"]["logical_bytes_read"]
        row["engram_lookup_rows_per_step"] = sum(table["lookup_rows"] for table in reports[0]["engram"].values()) / reports[0]["steps"]
        row["engram_resolve_host_ms_per_step"] = samples([
            sum(table["resolve_wait_seconds"] for table in report["engram"].values()) * 1000 / report["steps"]
            for report in reports])
        profiled = next(report for report in reports if report["repeat"] == 0)
        row["profiled_load_ms"] = profiled["profiled_load_ms"]
        for phase, span_key in (("training", "lookup_spans"), ("loading", "loading_spans")):
            profile = json.loads(Path(profiled[phase + "_profile"]).read_text())
            row[phase + "_profile"] = {"components": profile["components"], "regions": profile["regions"],
                "spans": span_totals(profile[span_key]), "top_operators_cpu": profile["top_operators_cpu"][:12]}
            if phase == "loading":
                row[phase + "_profile"]["threadpool_constructions"] = sum(
                    span["region"] == "load_executor" and span["name"] == "__init__" for span in profile[span_key])
        result["variants"][name] = row
    if args.historical_cuda:
        from dsv41_train.profiling import summarize_cuda_activity
        old = json.loads(Path(args.historical_cuda).read_text())
        profile = json.loads(Path(old["per_rank"][0]["operator_summary"]).read_text())
        trace = json.loads(Path(old["per_rank"][0]["traces"][0]).read_text())
        result["historical_cuda_context"] = {"source": args.historical_cuda,
            "note": "Earlier code and a different, larger GPU-resident workload; not this run or an offload measurement.",
            "environment": old["environment"], "model": old["configuration"]["model"],
            "timing": old["timing"], "kernel_calls": profile["kernel_calls"],
            "kernel_total_ms": profile["kernel_total_ms"], "components": profile["components"],
            "cuda_activity": summarize_cuda_activity(trace["traceEvents"], profile["sampled_steps"])}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("| Variant | First load ms | Warm load ms | Forward ms | Backward ms | Optimizer ms | Step ms | Tokens/s |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, row in result["variants"].items():
        timing = row["timing_ms"]
        values = [row["first_load_ms"]["median"], row["warm_load_ms"]["median"],
                  *(timing[stage]["median"] for stage in ("forward", "backward", "optimizer", "step")),
                  row["tokens_per_second"]["median"]]
        print(f"| {name} | " + " | ".join(f"{value:.2f}" for value in values) + " |")


if __name__ == "__main__":
    main()
