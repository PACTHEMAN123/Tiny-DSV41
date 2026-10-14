"""Record current resource visibility and require an idle GPU inventory."""
import csv
import json
import os
import subprocess
from pathlib import Path


def preflight(root):
    from .telemetry import snapshot, read
    record = snapshot()
    record["cgroup_membership"] = read("/proc/self/cgroup")
    record["cgroup_mounts"] = [x for x in Path("/proc/self/mountinfo").read_text().splitlines() if "cgroup" in x]
    topology = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        directory = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        topology[str(cpu)] = {name: read(directory / name) for name in (
            "physical_package_id", "core_id", "thread_siblings_list")}
    record["cpu_topology"] = topology
    record["gpu_owners"] = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader"],
        text=True, timeout=15)
    record["storage"] = subprocess.check_output(["df", "-B1", ".", "/dev/shm"], text=True)
    (root / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    if record["gpu_owners"].strip():
        raise RuntimeError("GPU owners must be reconciled before a new experiment")
    # Required inventory has a separate timeout from optional telemetry sampling.
    try:
        inventory = subprocess.check_output([
            "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
            text=True, timeout=15)
        rows = list(csv.reader(inventory.splitlines()))
        if not rows or any(len(row) != 2 or not row[1].strip().startswith("GPU-") for row in rows):
            raise ValueError("invalid GPU inventory")
        gpu_ids = [row[1].strip() for row in rows]
        if len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("duplicate GPU UUIDs in inventory")
        record["gpu_inventory_csv"] = inventory
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        record["gpu_inventory_error"] = str(exc)
        (root / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
        raise RuntimeError("required GPU inventory unavailable; experiment not admitted") from exc
    (root / "preflight.json").write_text(json.dumps(record, indent=2) + "\n")
    return gpu_ids
