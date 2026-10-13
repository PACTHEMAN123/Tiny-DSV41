"""Export offline timelines and time accounting from a memory benchmark report.

Only rendering needs Plotly. Parsing and accounting use the standard library;
the original Chrome traces remain the source for operator-level investigation.
"""

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean


COLORS = {"compute": "#147d64", "nccl": "#bd4249", "h2d": "#2076bb",
          "d2h": "#94651f", "d2d": "#7d58a0", "forward": "#147d64",
          "backward": "#2076bb", "optimizer": "#bd4249",
          "gradient_normalization": "#7d58a0", "zero_grad": "#777777"}


def union(intervals):
    result = []
    for start, stop in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(stop, result[-1][1]))
        else:
            result.append((start, stop))
    return result


def occupancy(intervals, origin, width, bins):
    """Percent busy in each bin, unioning streams before accumulating time."""
    if width <= 0 or bins < 1:
        raise ValueError("occupancy requires a positive width and bin count")
    values = [0.0] * bins
    for start, stop in union(intervals):
        lo = max(0, math.floor((start - origin) / width))
        hi = min(bins, math.ceil((stop - origin) / width))
        for index in range(lo, hi):
            left = origin + index * width
            values[index] += max(0.0, min(stop, left + width) - max(start, left))
    return [round(min(100.0, value * 100 / width), 3) for value in values]


def select_events(trace):
    selected = []
    for event in trace["traceEvents"]:
        if event.get("ph") != "X" or event.get("dur", 0) <= 0:
            continue
        name, category = event.get("name", ""), event.get("cat")
        kind = None
        if category == "user_annotation" and name.startswith(("train/", "dsv41/")):
            kind = "host"
        elif category == "kernel":
            kind = "nccl" if "nccl" in name.lower() else "compute"
        elif category == "gpu_memcpy":
            kind = "h2d" if "HtoD" in name else "d2h" if "DtoH" in name else "d2d"
        if kind is not None:
            selected.append({"name": name, "kind": kind, "start_us": event["ts"],
                             "duration_us": event["dur"], "thread": str(event.get("tid", "")),
                             "device": event.get("args", {}).get("device"),
                             "stream": event.get("args", {}).get("stream"),
                             "bytes": event.get("args", {}).get("bytes", 0)})
    return selected


def host_totals(events, steps):
    totals = defaultdict(lambda: {"calls_per_step": 0.0, "inclusive_ms_per_step": 0.0})
    for event in events:
        if event["kind"] != "host":
            continue
        parts = event["name"].split("/")
        key = "/".join(parts[:3]) if parts[1].startswith("load_") else "/".join(parts[:2])
        totals[key]["calls_per_step"] += 1 / steps
        totals[key]["inclusive_ms_per_step"] += event["duration_us"] / 1000 / steps
    return dict(totals)


def component_group(label):
    category = label.split("/", 1)[1]
    if category.startswith("engram"):
        return "engram"
    if category.startswith("attention") or category in {"compressor", "indexer"}:
        return "attention_and_indexer"
    if category in {"routed_experts", "shared_expert", "router", "dispatch_and_route"}:
        return "moe_and_dispatch"
    if category in {"embedding", "lm_head", "loss_and_aux"}:
        return "embedding_head_loss"
    if category in {"backbone_other", "residual_and_norm"}:
        return "backbone_norm_fsdp"
    if category == "unattributed":
        return "unattributed"
    return category


def distribution(values):
    return {"mean": mean(values), "min": min(values), "max": max(values), "samples": values}


def aggregate(rows, section):
    keys = sorted({key for row in rows for key in row[section]})
    return {key: {metric: distribution([row[section].get(key, {}).get(metric, 0.0) for row in rows])
                  for metric in sorted({metric for row in rows for metric in row[section].get(key, {})})}
            for key in keys}


def read_phase(report, report_path, phase):
    records = []
    reference_base = None
    for row in report["ranks"]:
        summary_name = row.get("profile_summary" if phase == "training" else "loading_profile")
        if not summary_name:
            raise ValueError(f"rank {row['rank']} has no {phase} profile")
        summary_path = Path(summary_name)
        if not summary_path.is_file():
            summary_path = report_path.parent / phase / f"rank-{row['rank']}-summary.json"
        summary = json.loads(summary_path.read_text())
        trace_path = summary_path.with_name(summary_path.name.replace("-summary.json", ".json"))
        trace = json.loads(trace_path.read_text())
        if "baseTimeNanoseconds" not in trace:
            raise ValueError("multi-rank timeline requires the trace clock base")
        base = trace["baseTimeNanoseconds"]
        if reference_base is None:
            reference_base = base
        selected = select_events(trace)
        del trace
        if not selected:
            raise ValueError(f"empty {phase} trace for rank {row['rank']}")
        for event in selected:
            event["start_us"] += (base - reference_base) / 1000
        groups = defaultdict(lambda: {"self_cpu_ms": 0.0, "linked_cuda_ms": 0.0})
        for label, values in summary["components"].items():
            groups[component_group(label)]["self_cpu_ms"] += values["self_cpu_ms"]
            groups[component_group(label)]["linked_cuda_ms"] += values["cuda_ms"]
        host = host_totals(selected, summary["sampled_steps"])
        stages = [event for event in selected if event["kind"] == "host" and event["name"].startswith("train/")]
        records.append({"rank": row["rank"], "events": selected,
                        "source_trace": str(trace_path), "sampled_steps": summary["sampled_steps"],
                        "host_regions": host, "components": summary["components"],
                        "component_groups": dict(groups), "cuda_activity": summary["cuda_activity"],
                        "profiled_stage_window_ms": ((max(e["start_us"] + e["duration_us"] for e in stages)
                            - min(e["start_us"] for e in stages)) / 1000 / summary["sampled_steps"]) if stages else None})
    return records


def training_overview(records, output, width_ms, title="DSV4"):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    origin = min(event["start_us"] for row in records for event in row["events"])
    stop = max(event["start_us"] + event["duration_us"] for row in records for event in row["events"])
    bins = math.ceil((stop - origin) / (width_ms * 1000))
    if bins > 20000:
        raise ValueError("overview has too many bins; increase --bin-ms")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[.25, .75], vertical_spacing=.10,
                        subplot_titles=("CPU training scopes", f"GPU busy fraction per {width_ms:g} ms bin"))
    for stage in ("zero_grad", "forward", "backward", "gradient_normalization", "optimizer"):
        selected = [(row["rank"], event) for row in records for event in row["events"]
                    if event["kind"] == "host" and event["name"] == "train/" + stage]
        fig.add_trace(go.Bar(name=stage, orientation="h", y=[f"Rank {rank}" for rank, _ in selected],
            base=[(e["start_us"]-origin)/1000 for _, e in selected],
            x=[e["duration_us"]/1000 for _, e in selected], marker_color=COLORS[stage],
            hovertemplate="%{y}<br>Start %{base:.3f} ms<br>Host scope %{x:.3f} ms<extra>%{fullData.name}</extra>"), row=1, col=1)
    lanes, matrix = [], []
    for row in records:
        for kind in ("compute", "nccl", "h2d", "d2h", "d2d"):
            intervals = [(e["start_us"], e["start_us"]+e["duration_us"]) for e in row["events"] if e["kind"] == kind]
            lanes.append(f"Rank {row['rank']} / {kind}")
            matrix.append(occupancy(intervals, origin, width_ms * 1000, bins))
    fig.add_trace(go.Heatmap(x=[(i+.5)*width_ms for i in range(bins)], y=lanes, z=matrix,
        colorscale=[[0, "#f2f3f4"], [.01, "#c0dcd5"], [.5, "#438f80"], [1, "#125b4f"]],
        zmin=0, zmax=100, colorbar=dict(title="Busy %", len=.65, y=.35),
        hovertemplate="%{y}<br>%{x:.3f} ms<br>Busy %{z:.1f}%<extra></extra>"), row=2, col=1)
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Time from shared trace origin (ms)", row=2, col=1)
    fig.update_layout(title=dict(text=f"{title} training timeline / {len(records)}-rank capture", x=.02, y=.99, yanchor="top"),
                      template="plotly_white", barmode="overlay", height=1400, margin=dict(l=180, t=160),
                      legend=dict(orientation="h", y=1.07, yanchor="bottom"), font=dict(family="Arial", size=12))
    fig.write_html(output, include_plotlyjs=True, full_html=True, config={"responsive": True})


def detail_timeline(record, phase, output, title="DSV4"):
    import plotly.graph_objects as go

    events = record["events"]
    origin = min(event["start_us"] for event in events)
    fig = go.Figure()
    categories = defaultdict(list)
    for event in events:
        if event["kind"] == "host":
            parts = event["name"].split("/")
            if phase == "training" and not (parts[0] == "train" or parts[1].startswith("engram")):
                continue
            lane = "/".join(parts[:3]) if phase == "loading" else "/".join(parts[:2])
            lane = f"CPU {lane} / thread {event['thread']}"
        else:
            lane = f"GPU {event['device']} / {event['kind']} / stream {event['stream']}"
        categories[lane].append(event)
    for lane, values in sorted(categories.items()):
        kind = values[0]["kind"]
        if kind == "host":
            fig.add_trace(go.Bar(name=lane, orientation="h", y=[lane]*len(values),
                base=[(e["start_us"]-origin)/1000 for e in values], x=[e["duration_us"]/1000 for e in values],
                hovertext=[e["name"] for e in values], textposition="none",
                marker_color=COLORS.get(values[0]["name"].split("/")[1], "#528c87"),
                hovertemplate="%{hovertext}<br>Start %{base:.6f} ms<br>Duration %{x:.6f} ms<extra></extra>"))
        else:
            # Exact event endpoints; GPU lanes use WebGL to preserve short kernels.
            x, y, text = [], [], []
            for event in values:
                start, duration = (event["start_us"]-origin)/1000, event["duration_us"]/1000
                label = f"{event['name'][:180]}<br>{duration:.6f} ms / {event['bytes']} bytes"
                x.extend((start, start+duration, None))
                y.extend((lane, lane, None))
                text.extend((label, label, None))
            fig.add_trace(go.Scattergl(name=lane, x=x, y=y, text=text, mode="lines",
                line=dict(color=COLORS[kind], width=8), hovertemplate="%{text}<extra></extra>"))
    fig.update_layout(title=f"{title} {phase} / rank {record['rank']} / exact event intervals",
        template="plotly_white", barmode="overlay", showlegend=False, height=max(650, len(categories)*30),
        margin=dict(l=350), font=dict(family="Arial", size=12),
        xaxis_title="Time from rank trace origin (ms)", yaxis=dict(autorange="reversed"))
    fig.write_html(output, include_plotlyjs=True, full_html=True, config={"responsive": True})


def write_report(report, phases, output):
    result = {"source_workload": {key: report[key] for key in ("variant", "world", "cp", "ep", "steps", "warmup", "sequence_length")},
              "note": "Existing trace analysis; host inclusive scopes overlap, self CPU sums span threads, linked CUDA includes communication/copies. No values are a critical-path decomposition.",
              "unprofiled_step_ms_by_rank": {row["rank"]: row["step_ms"] for row in report["ranks"]}, "phases": {}}
    result["source_workload"].update({key: report.get(key, default) for key, default in (
        ("batch_size", 1), ("microbatches", 2), ("activation_checkpoint", "none"),
        ("memory_limit_gib", 0), ("load_profiled", False))})
    result["source_workload"].update({key: report[key] for key in (
        "config", "load_policy", "engram_policy", "input_pattern", "model_path", "cuda_visible_devices")
        if key in report})
    csv_rows = []
    for phase, records in phases.items():
        item = {"ranks": [{key: value for key, value in row.items() if key != "events"} for row in records]}
        for section in ("host_regions", "components", "component_groups"):
            item[section] = aggregate(records, section)
            for row in records:
                for label, values in row[section].items():
                    for metric, value in values.items():
                        csv_rows.append((phase, row["rank"], section, label, metric, value))
        for row in records:
            for kind, values in row["cuda_activity"]["activities"].items():
                for metric, value in values.items():
                    csv_rows.append((phase, row["rank"], "cuda_activity", kind, metric, value))
        item["gpu_activity_ms"] = {kind: distribution([row["cuda_activity"]["activities"].get(kind, {}).get("duration_ms", 0)
                                                         for row in records]) for kind in ("compute", "nccl", "h2d", "d2h", "d2d")}
        result["phases"][phase] = item
    (output / "time-breakdown.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "time-breakdown.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("phase", "rank", "section", "component", "metric", "value_per_profiled_step"))
        writer.writerows(csv_rows)
    links = (["[Training overview](timeline.html)"] if "training" in phases else [])
    links.extend(f"[{phase.title()} rank 0]({phase}-rank-0.html)" for phase in phases)
    lines = ["# DSV4 timeline and time breakdown", "",
             f"Analysis of an existing {report['world']}-rank {report['variant']} capture, CP{report['cp']}/EP{report['ep']}; no GPU rerun.", "",
             " | ".join(links), "",
             "Charts are standalone offline Plotly HTML. Exact operator/flow details remain in the original Chrome JSON traces for Perfetto.", "",
             f"CPU scopes below are inclusive, summed per profiled optimizer step ({report.get('microbatches', 2)} microbatches, batch size {report.get('batch_size', 1)}, sequence {report['sequence_length']}). Values are mean [min, max] across ranks, in ms.", ""]
    for phase, item in result["phases"].items():
        lines.extend((f"## {phase.title()} host scopes", "", "| Scope | Mean ms | Min ms | Max ms |", "| --- | ---: | ---: | ---: |"))
        for label, values in item["host_regions"].items():
            if phase == "training" and not label.startswith(("train/", "dsv41/engram")):
                continue
            value = values["inclusive_ms_per_step"]
            lines.append(f"| {label} | {value['mean']:.3f} | {value['min']:.3f} | {value['max']:.3f} |")
        lines.extend(("", f"## {phase.title()} GPU activities", "", "| Activity | Mean ms | Min ms | Max ms |", "| --- | ---: | ---: | ---: |"))
        for label, value in item["gpu_activity_ms"].items():
            lines.append(f"| {label} | {value['mean']:.3f} | {value['min']:.3f} | {value['max']:.3f} |")
        lines.extend(("", f"## {phase.title()} module attribution", "", "| Group | Self CPU mean ms | Linked CUDA mean ms |", "| --- | ---: | ---: |"))
        for label, values in item["component_groups"].items():
            lines.append(f"| {label} | {values['self_cpu_ms']['mean']:.3f} | {values['linked_cuda_ms']['mean']:.3f} |")
        lines.append("")
    lines.extend(("## Interpretation", "", "- Profiler overhead is substantial. Use the unprofiled step samples in the JSON for throughput; do not replace them with traced durations.",
        "- Inclusive scopes nest and worker threads overlap. Self CPU is attributed thread time, not a single wall-clock budget.",
        "- Linked CUDA module totals include NCCL, memory copies and waits. Raw activity separates compute from communication and DMA.",
        "- NCCL kernels include waiting and rank skew; their durations do not establish bandwidth saturation.",
        "- The overview unions concurrent stream intervals within each category before binning; overlapping categories still must not be added.",
        "- All ranks were on one host. Clock bases are retained for alignment; loading and training have separate time origins.",
        "- The eight-rank BF16 parity failures remain open. A successful profiling capture is not correctness acceptance.", ""))
    (output / "README.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bin-ms", type=float, default=.25)
    parser.add_argument("--phases", nargs="+", choices=("training", "loading"),
                        default=("training", "loading"))
    args = parser.parse_args()
    if not math.isfinite(args.bin_ms) or args.bin_ms <= 0:
        parser.error("bin-ms must be finite and positive")
    report = json.loads(args.report.read_text())
    if sorted(row["rank"] for row in report["ranks"]) != list(range(report["world"])):
        raise ValueError("report must contain each rank exactly once")
    phases = {phase: read_phase(report, args.report, phase) for phase in args.phases}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_report(report, phases, args.output_dir)
    if "training" in phases:
        training_overview(phases["training"], args.output_dir / "timeline.html", args.bin_ms)
    for phase, records in phases.items():
        detail_timeline(next(row for row in records if row["rank"] == 0), phase,
                        args.output_dir / f"{phase}-rank-0.html")
    print(json.dumps({"output": str(args.output_dir), "ranks": report["world"], "phases": list(phases)}))


if __name__ == "__main__":
    main()
