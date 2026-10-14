"""Summarize the closed, selected A trace without treating trace gaps as idle."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import statistics
import tempfile
import zipfile

ARCHIVE_SHA = "8b3b595f2b9752feb26475d2500efb8238995a104d8f99f0079d58a9dc6ff89e"
DATABASE = "a-r2/profiles/trace-r1.sqlite"


def union_ns(intervals):
    total, right = 0, None
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("negative interval")
        if right is None or start > right:
            total += end-start
        else:
            total += max(0, end-right)
        right = end if right is None else max(end, right)
    return total


def distribution(values):
    ordered = sorted(values)
    return {"min": ordered[0], "p50": statistics.median(ordered),
            "p95": ordered[math.ceil(.95 * len(ordered)) - 1], "max": ordered[-1]}


def collective(name):
    return any(token in name for token in ("Allreduce", "LamportMHC", "ncclDevKernel"))


def host_audit(connection, first, last):
    workers = []
    for device, process, pid in connection.execute("""
        SELECT DISTINCT k.deviceId, k.globalPid, p.pid
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN PROCESSES p USING(globalPid)
        ORDER BY k.deviceId
    """):
        thread = process + pid
        api = connection.execute("""
            SELECT start,end FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE globalTid=? AND start<? AND end>?
        """, (thread, last, first)).fetchall()
        osrt = connection.execute("""
            SELECT start,end FROM OSRT_API
            WHERE globalTid=? AND start<? AND end>?
        """, (thread, last, first)).fetchall()
        polls = connection.execute("""
            SELECT o.start,o.end FROM OSRT_API o JOIN StringIds s ON s.id=o.nameId
            WHERE o.globalTid=? AND s.value='poll'
        """, (thread,)).fetchall()
        clipped = lambda rows: [(max(s, first), min(e, last)) for s, e in rows
                                if s < last and e > first]
        workers.append({
            "device": device, "pid": pid, "main_global_tid": thread,
            "poll_calls_full_capture": len(polls),
            "poll_union_full_capture_seconds": union_ns(polls)/1e9,
            "poll_union_kernel_window_seconds": union_ns(clipped(polls))/1e9,
            "cuda_api_union_kernel_window_seconds": union_ns(clipped(api))/1e9,
            "osrt_union_kernel_window_seconds": union_ns(clipped(osrt))/1e9,
            "outside_recorded_cuda_api_and_osrt_seconds":
                (last-first-union_ns(clipped(api+osrt)))/1e9,
        })
    # This is the exact poll event selected in the user's Nsight window.
    poll = connection.execute("""
        SELECT o.start,o.end,o.globalTid,o.returnValue,o.callchainId
        FROM OSRT_API o JOIN StringIds s ON s.id=o.nameId
        WHERE o.globalTid=? AND o.start=? AND s.value='poll'
    """, (workers[0]["main_global_tid"], 7032457808)).fetchone()
    stack = connection.execute("""
        SELECT cc.stackDepth,s.value,m.value FROM OSRT_CALLCHAINS cc
        LEFT JOIN StringIds s ON s.id=cc.symbol
        LEFT JOIN StringIds m ON m.id=cc.module
        WHERE cc.id=? ORDER BY cc.stackDepth
    """, (poll[4],)).fetchall()
    overhead = [dict(zip(("pid", "thread_id", "kind", "calls", "summed_seconds"), row))
                for row in connection.execute("""
                    SELECT p.pid,o.globalTid & 16777215,e.label,count(*),sum(o.end-o.start)/1e9
                    FROM CUPTI_ACTIVITY_KIND_OVERHEAD o
                    JOIN ENUM_CUPTI_OVERHEAD_TYPE e ON e.id=o.overheadType
                    LEFT JOIN PROCESSES p ON p.globalPid=(o.globalTid & -16777216)
                    GROUP BY o.globalTid,o.overheadType ORDER BY sum(o.end-o.start) DESC
                """)]
    sample_tables = [row[0] for row in connection.execute("""
        SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'ENUM_%'
            AND (name LIKE '%SAMPLE%' OR name LIKE '%SCHED%' OR name LIKE '%CONTEXT_SWITCH%')
    """)]
    return {
        "capture_duration_seconds": connection.execute(
            "SELECT duration/1e9 FROM ANALYSIS_DETAILS").fetchone()[0],
        "cpu_sample_or_scheduling_tables": sample_tables,
        "workers": workers,
        "selected_poll": dict(zip(("start_ns", "end_ns", "global_tid", "return_value",
                                    "callchain_id"), poll)),
        "selected_poll_stack": [dict(zip(("depth", "symbol", "module"), row)) for row in stack],
        "cuda_overhead": overhead,
        "interpretation": "Unattributed host wall time is not measured CPU execution or CPU utilization. "
                          "OS-call backtraces exist, but CPU execution sampling and scheduling data do not.",
    }


def analyze(connection):
    kernels = connection.execute("""
        SELECT k.deviceId, k.start, k.end, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
        ORDER BY k.start
    """).fetchall()
    first, last = min(k[1] for k in kernels), max(k[2] for k in kernels)
    by_gpu, names = collections.defaultdict(list), collections.defaultdict(lambda: [0, 0])
    for device, start, end, name in kernels:
        by_gpu[device].append((start, end, name))
        names[name][0] += 1
        names[name][1] += end-start
    gpus = []
    for device, rows in sorted(by_gpu.items()):
        all_ns = union_ns((s, e) for s, e, _ in rows)
        comm_ns = union_ns((s, e) for s, e, n in rows if collective(n))
        gpus.append({"device": device, "kernel_count": len(rows),
                     "kernel_union_seconds": all_ns/1e9,
                     "collective_union_seconds": comm_ns/1e9,
                     "other_kernel_only_seconds": (all_ns-comm_ns)/1e9,
                     "no_recorded_kernel_seconds": (last-first-all_ns)/1e9,
                     "kernel_sum_seconds": sum(e-s for s, e, _ in rows)/1e9})
    matched = collections.defaultdict(list)
    # Nsight encodes process identity in the upper bits of globalTid.
    for row in connection.execute("""
        SELECT k.deviceId, k.start, k.end, k.correlationId, r.start, r.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName
        LEFT JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON k.correlationId=r.correlationId
            AND k.globalPid=(r.globalTid & -16777216)
        WHERE s.value='oneshotAllreduceFusionKernel' ORDER BY k.deviceId, k.start
    """):
        device, start, end, correlation, cpu_start, cpu_end = row
        matched[device].append({"device": device, "start_ns": start, "end_ns": end,
                               "correlation_id": correlation,
                               "launch_start_ns": cpu_start, "launch_end_ns": cpu_end})
    if sorted(matched) != [0, 1, 2, 3] or {len(rows) for rows in matched.values()} != {5588}:
        raise ValueError("collective sequence coverage changed")
    end_spread, arrival_spread, launch_lead = [], [], []
    last_arrivals, last_launches = collections.Counter(), collections.Counter()
    groups = list(zip(*(matched[g] for g in sorted(matched))))
    for group in groups:
        if any(row["launch_start_ns"] is None for row in group):
            raise ValueError("unmatched CPU launch")
        starts = [row["start_ns"] for row in group]
        ends = [row["end_ns"] for row in group]
        launches = [row["launch_start_ns"] for row in group]
        last_arrivals[group[starts.index(max(starts))]["device"]] += 1
        last_launches[group[launches.index(max(launches))]["device"]] += 1
        arrival_spread.append((max(starts)-min(starts))/1e6)
        end_spread.append((max(ends)-min(ends))/1e3)
        launch_lead.append((group[1]["start_ns"]-group[1]["launch_start_ns"])/1e3)
    copies = [dict(zip(("kind", "calls", "bytes", "summed_seconds"), row))
              for row in connection.execute("""
                  SELECT e.label, count(*), sum(m.bytes), sum(m.end-m.start)/1e9
                  FROM CUPTI_ACTIVITY_KIND_MEMCPY m JOIN ENUM_CUDA_MEMCPY_OPER e ON e.id=m.copyKind
                  GROUP BY m.copyKind
              """)]
    warnings = [dict(zip(("text", "count"), row)) for row in connection.execute(
        "SELECT text,count(*) FROM DIAGNOSTIC_EVENT WHERE severity>=2 GROUP BY text ORDER BY text")]
    return {"window": {"origin": "Nsight capture-relative nanoseconds",
                       "first_kernel_ns": first, "last_kernel_ns": last,
                       "seconds": (last-first)/1e9},
            "gpus": gpus,
            "top_kernels": [{"name": name, "calls": count, "summed_seconds": ns/1e9}
                            for name, (count, ns) in sorted(names.items(), key=lambda item: -item[1][1])[:15]],
            "collective_pairing": {
                "method": "same kernel name and per-device occurrence order; not a cross-rank correlation ID",
                "groups": len(groups), "last_gpu_start_counts": dict(last_arrivals),
                "last_cpu_launch_counts": dict(last_launches),
                "arrival_spread_ms": distribution(arrival_spread),
                "end_spread_us": distribution(end_spread),
                "gpu1_launch_to_kernel_start_us": distribution(launch_lead),
                "examples_zero_based": {str(i): groups[i] for i in (100, 1000, 2000, 4000)}},
            "explicit_memcpy": copies, "warnings": warnings,
            "host_audit": host_audit(connection, first, last),
            "limitations": [
                "The 8 x 128-token diagnostic includes prefill and decode; it is not a full rollout.",
                "Trace warnings permit missing events; no recorded kernel is not proof of GPU idleness.",
                "Collective intervals can include waiting and fused computation, not just data transfer.",
                "No CPU execution samples, scheduling trace, or hardware link-bandwidth counters were collected; OS-call backtraces exist.",
                "CPU launch times show submission order, not why a host thread submitted late.",
                "Observed union coverage and overlapping kernel-duration sums are different quantities."]}


def plot(data, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    colors = ("#b88624", "#178a7a", "#e4e6e8")
    keys = ("collective_union_seconds", "other_kernel_only_seconds", "no_recorded_kernel_seconds")
    for row in data["gpus"]:
        left = 0
        for key, color in zip(keys, colors):
            axes[0].barh(row["device"], row[key], left=left, height=.6, color=color)
            left += row[key]
        axes[0].text(data["window"]["seconds"]+.2, row["device"],
                     f"{row['kernel_union_seconds']:.3f} s recorded", va="center")
    axes[0].set_xlim(0, data["window"]["seconds"]+4.7)
    axes[0].set_yticks(range(4), [f"GPU {i}" for i in range(4)])
    axes[0].invert_yaxis()
    axes[0].set_title("Recorded kernel coverage: interval unions across streams, not summed durations", loc="left")
    axes[0].set_xlabel(f"Seconds within the {data['window']['seconds']:.6f} s observed kernel window; aggregate coverage, not chronology")
    axes[0].legend(handles=[Patch(color=c, label=n) for c,n in zip(colors,
                   ("Collective kernels (can wait)", "Other kernels outside collectives", "No recorded kernel"))],
                   loc="lower center", bbox_to_anchor=(.5, 1.05), ncol=3, frameon=False)
    group = data["collective_pairing"]["examples_zero_based"]["1000"]
    origin = min(row["start_ns"] for row in group)
    for row in group:
        y = row["device"]
        left, width = (row["start_ns"]-origin)/1e6, (row["end_ns"]-row["start_ns"])/1e6
        axes[1].barh(y, width, left=left, height=.6, color=colors[0] if y != 1 else colors[1])
        axes[1].text(1.92, y, f"kernel {width*1000:.1f} us; CPU launch {(row['launch_start_ns']-origin)/1e6:+.3f} ms", va="center")
    axes[1].axvline((group[1]["start_ns"]-origin)/1e6, color="#bc4b51", linestyle="--", linewidth=1)
    axes[1].set_xlim(-.04, 3.75)
    axes[1].set_yticks(range(4), [f"GPU {i}" for i in range(4)])
    axes[1].invert_yaxis()
    axes[1].set_title("One matched all-reduce: GPU 1 enters last; all four finish within 3.4 us", loc="left")
    axes[1].set_xlabel("Milliseconds since the earliest GPU starts this collective (capture t = 41.403402 s)")
    fig.suptitle("A decode diagnostic: DeepSeek-V4.1-Flash, TP4, C8, 128 output tokens/request\n"
                 "Late rank-1 host submission and collective waiting; cause of host delay remains unresolved", fontsize=13)
    fig.supxlabel("Nsight warns of potentially missing events. No recorded kernel does not establish idle time or useful utilization.", fontsize=10)
    fig.savefig(output/"a-decode-profile.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ARCHIVE_SHA:
            raise ValueError("closed A evidence archive changed")
    with zipfile.ZipFile(args.archive) as archive, tempfile.TemporaryDirectory() as temp:
        manifest = json.loads(archive.read("manifest.json"))
        db = Path(temp)/"trace.sqlite"
        with archive.open(DATABASE) as source, db.open("wb") as target:
            shutil.copyfileobj(source, target)
        with db.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != manifest[DATABASE]["sha256"] or db.stat().st_size != manifest[DATABASE]["bytes"]:
            raise ValueError("trace member identity differs")
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
            data = analyze(connection)
        connection.close()
    data["evidence"] = {"archive": args.archive.name, "archive_sha256": ARCHIVE_SHA,
                        "member": DATABASE, "member_sha256": digest}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"inference-trace.json").write_text(json.dumps(data, indent=2)+"\n")
    plot(data, args.output)
    print(json.dumps({"gpus": data["gpus"], "pairing": data["collective_pairing"]}, indent=2))


if __name__ == "__main__":
    main()
