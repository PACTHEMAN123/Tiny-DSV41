"""Low-frequency host, container, GPU, vLLM and network counters."""
from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


def read(path):
    try:
        return Path(path).read_text()
    except OSError as exc:
        return {"unavailable": type(exc).__name__}


def snapshot(vllm=False):
    start = time.monotonic()
    result = {"unix_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns(),
              "host": socket.gethostname(), "cpu_affinity": sorted(os.sched_getaffinity(0))}
    paths = {"stat": "/proc/stat", "memory": "/proc/meminfo", "load": "/proc/loadavg",
        "netdev": "/proc/net/dev", "snmp": "/proc/net/snmp", "netstat": "/proc/net/netstat",
        "softnet": "/proc/net/softnet_stat", "cpu_pressure": "/proc/pressure/cpu",
        "memory_pressure": "/proc/pressure/memory", "io_pressure": "/proc/pressure/io",
        "cgroup_cpu": "/sys/fs/cgroup/cpu.stat", "cgroup_cpu_max": "/sys/fs/cgroup/cpu.max",
        "cgroup_memory": "/sys/fs/cgroup/memory.current", "cgroup_memory_max": "/sys/fs/cgroup/memory.max",
        "cgroup_memory_events": "/sys/fs/cgroup/memory.events", "cgroup_io": "/sys/fs/cgroup/io.stat"}
    result.update({key: read(path) for key, path in paths.items()})
    try:
        fields = "index,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,clocks.sm,clocks.mem,temperature.gpu"
        result["gpu_fields"] = fields.split(",")
        result["gpu_csv"] = subprocess.check_output(["nvidia-smi", "--query-gpu=" + fields,
            "--format=csv,noheader,nounits"], text=True, timeout=3).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        result["gpu_error"] = str(exc)
    if vllm:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=2) as response:
                raw = response.read(2 * 1024 * 1024).decode()
            result["vllm_metrics"] = "\n".join(line for line in raw.splitlines() if not line.startswith("#"))
        except OSError as exc:
            result["vllm_error"] = str(exc)
    usage = shutil.disk_usage(".")
    result["disk_free_bytes"] = usage.free
    result["sample_seconds"] = time.monotonic() - start
    return result


def process_accounting(root_pid):
    """Sample the owned tree, including children launched by non-main threads."""
    pending, seen, rows = [root_pid], set(), []
    while pending and len(seen) < 4096:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        root = Path("/proc") / str(pid)
        try:
            stat = (root / "stat").read_text().rsplit(")", 1)[1].split()
            status = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                      for line in (root / "status").read_text().splitlines() if ":" in line}
            for task in (root / "task").iterdir():
                try:
                    pending.extend(map(int, (task / "children").read_text().split()))
                except OSError:
                    pass
            rows.append({"pid": pid, "start_ticks": stat[19], "user_ticks": int(stat[11]),
                         "system_ticks": int(stat[12]), "minor_faults": int(stat[7]), "major_faults": int(stat[9]),
                         "status": {k: status.get(k) for k in ("Name", "VmRSS", "RssAnon", "RssShmem",
                                    "VmPin", "VmLck", "Threads", "Cpus_allowed_list")}})
        except (OSError, IndexError, ValueError):
            pass
    return {"clock_ticks_per_second": os.sysconf("SC_CLK_TCK"), "processes": rows,
            "truncated": bool(pending), "rss_shared_pages_must_not_be_summed": True}


def monitor(path: Path, stop: threading.Event, *, vllm=False, interval=2, ttl=21600, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    end = time.monotonic() + ttl
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "xt") as output:
        while not stop.is_set() and time.monotonic() < end:
            start = time.monotonic()
            sample = snapshot(vllm)
            if extra is not None:
                sample["manager"] = extra()
                sample["owned_process_accounting"] = process_accounting(os.getpid())
            output.write(json.dumps(sample) + "\n")
            output.flush()
            limit = 1536 if path.suffix == ".gz" else 512
            if output.tell() > limit * 1024 * 1024:
                raise RuntimeError(f"telemetry exceeded {limit} MiB of uncompressed data")
            stop.wait(max(0, interval - (time.monotonic() - start)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vllm", action="store_true")
    parser.add_argument("--ttl", type=int, default=21600)
    args = parser.parse_args()
    monitor(args.output, threading.Event(), vllm=args.vllm, ttl=min(args.ttl, 21600))


if __name__ == "__main__":
    main()
