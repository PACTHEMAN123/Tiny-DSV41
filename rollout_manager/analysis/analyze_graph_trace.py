"""Audit CUDA graph replay and CPU submission directly from Nsight SQLite."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics


def sha_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coverage(rows):
    end, total = None, 0
    for start, stop in sorted(rows):
        total += max(0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def quantiles(values):
    values = sorted(values)
    return {"n": len(values), "p50": statistics.median(values),
            "p95": values[min(len(values) - 1, int(.95 * len(values)))],
            "mean": statistics.mean(values)} if values else None


def analyze(capture, engine_path):
    db_path = capture / "trace.sqlite"
    db = sqlite3.connect("file:" + str(db_path) + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    begin = json.loads((capture / "annotations-start.json").read_text())["finished_unix_ns"]
    end = json.loads((capture / "annotations-stop.json").read_text())["started_unix_ns"]
    requests = [json.loads(line) for line in engine_path.read_text().splitlines()]
    requests = [r for r in requests if begin <= r["unix_ns"] <= end]
    if not requests:
        raise ValueError("capture contains no engine response records")
    session = dict(db.execute("select * from TARGET_INFO_SESSION_START_TIME").fetchone())
    mono = session["systemClockNs"]
    low = round(max(r["engine_timestamps"]["first_token_ts"] for r in requests) * 1e9) - mono
    high = round(min(r["engine_timestamps"]["last_token_ts"] for r in requests) * 1e9) - mono
    if low >= high:
        raise ValueError("no shared decode interval")
    names = dict(db.execute("select id,value from StringIds"))
    graph_id, = [i for i, name in names.items() if name.startswith("cudaGraphLaunch_")]
    workers = {}
    for tid, count in db.execute("select globalTid,count(*) from CUPTI_ACTIVITY_KIND_RUNTIME where nameId=? group by globalTid", (graph_id,)):
        calls = list(db.execute("select start,end,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and start>=? and end<=? order by start", (tid, low, high)))
        launches = [(r["start"], r["end"]) for r in calls if r["nameId"] == graph_id]
        os_calls = list(db.execute("select start,end from OSRT_API where globalTid=? and start>=? and end<=?", (tid, low, high)))
        workers[str(tid)] = {"pid": (tid >> 24) & 0xffffff, "tid": tid & 0xffffff,
            "graph_launch_count_entire_capture": count, "cuda_calls_in_common_decode": len(calls),
            "graph_launch_count_common_decode": len(launches),
            "calls_per_graph_launch": len(calls) / len(launches),
            "graph_launch_api_ms": quantiles([(b - a) / 1e6 for a, b in launches]),
            "host_graph_start_interval_ms": quantiles([(b[0] - a[0]) / 1e6 for a, b in zip(launches, launches[1:])]),
            "cuda_api_union_seconds": coverage([(r["start"], r["end"]) for r in calls]) / 1e9,
            "osrt_union_seconds": coverage([(r["start"], r["end"]) for r in os_calls]) / 1e9}
    tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
    graphs = {}
    devices = list(db.execute("select distinct deviceId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE order by deviceId")) if "CUPTI_ACTIVITY_KIND_GRAPH_TRACE" in tables else []
    for gpu, in devices:
        values = list(db.execute("select start,end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE where deviceId=? order by start", (gpu,)))
        graphs[str(gpu)] = {"count": len(values), "start_seconds": values[0][0] / 1e9,
            "end_seconds": max(r[1] for r in values) / 1e9,
            "duration_ms": quantiles([(r[1] - r[0]) / 1e6 for r in values])}
    # Use an early window with graph records on every GPU, not the unrecorded tail.
    window_start = round((max((v["start_seconds"] for v in graphs.values()), default=low / 1e9) + .3) * 1e9)
    window_end = window_start + 40_000_000
    sample = {"start_seconds": window_start / 1e9, "end_seconds": window_end / 1e9, "gpu": {}, "host": {}}
    for gpu in graphs:
        values = [dict(r) for r in db.execute("select start,end,graphId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE where deviceId=? and end>? and start<? order by start", (int(gpu), window_start, window_end))]
        sample["gpu"][gpu] = values
    for tid in workers:
        sample["host"][tid] = [dict(r) for r in db.execute("select start,end from CUPTI_ACTIVITY_KIND_RUNTIME where globalTid=? and nameId=? and end>? and start<? order by start", (int(tid), graph_id, window_start, window_end))]
    counts = {table: db.execute("select count(*) from " + table).fetchone()[0]
              for table in ("COMPOSITE_EVENTS", "SAMPLING_CALLCHAINS", "SCHED_EVENTS", "NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL")}
    warnings = [dict(r) for r in db.execute("select * from DIAGNOSTIC_EVENT where severity>1")]
    result = {"sqlite_sha256": sha_file(db_path),
        "session": session, "request_count": len(requests), "completion_tokens": sum(r["completion_tokens"] for r in requests),
        "common_decode_interval_seconds": [low / 1e9, high / 1e9],
        "workers": workers, "gpu_graph_records": graphs, "trace_counts": counts,
        "warnings": warnings, "sample": sample,
        "coverage_note": "CPU launch counts are distinct from recorded GPU graph activities. GPU graph records do not cover the entire decode; do not treat absent graph intervals as idle."}
    db.close()
    return result


def analyze_native(path):
    db = sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    names = dict(db.execute("select id,value from StringIds"))
    kernels = [dict(r) for r in db.execute(
        "select demangledName, count(*) as calls, sum(end-start) as total_ns, "
        "avg(end-start) as mean_ns, min(start) as first_ns, max(end) as last_ns "
        "from CUPTI_ACTIVITY_KIND_KERNEL group by demangledName order by total_ns desc")]
    for row in kernels:
        row["name"] = names[row.pop("demangledName")]
    copies = [dict(r) for r in db.execute(
        "select copyKind, count(*) as calls, sum(bytes) as bytes, sum(end-start) as total_ns "
        "from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind")]
    ranges = [dict(r) for r in db.execute(
        "select text,textId,count(*) as calls,sum(end-start) as total_ns "
        "from NVTX_EVENTS where end>start group by text,textId order by total_ns desc")]
    for row in ranges:
        row["name"] = row.pop("text") or names.get(row.pop("textId"), "")
        row.pop("textId", None)
    result = {"sqlite_sha256": sha_file(path), "kernels": kernels, "copies": copies,
              "nvtx_ranges": ranges,
              "warnings": [dict(r) for r in db.execute("select * from DIAGNOSTIC_EVENT where severity>1")]}
    db.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--capture", type=Path)
    source.add_argument("--native-trace", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.capture and not args.engine:
        parser.error("--engine is required with --capture")
    result = analyze_native(args.native_trace) if args.native_trace else analyze(args.capture, args.engine)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k in (
        "request_count", "workers", "gpu_graph_records", "trace_counts", "kernels", "copies")}))


if __name__ == "__main__":
    main()
