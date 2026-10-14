"""Disjoint compiler, evaluator and control CPUs, preserving physical-core isolation."""
from pathlib import Path


def controlled_layout(cpus, builds, build_cores, gpu_count, evaluation_cores):
    if min(builds, build_cores, evaluation_cores) < 1 or gpu_count < 0 or len(set(cpus)) != len(cpus):
        raise ValueError("positive worker dimensions and unique CPU IDs required")
    groups = {}
    for cpu in sorted(cpus):
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        key = ((root / "physical_package_id").read_text().strip(), (root / "core_id").read_text().strip())
        groups.setdefault(key, []).append(cpu)
    groups = list(groups.values())
    needed = 8 + builds * build_cores + evaluation_cores * gpu_count
    if len(groups) < needed:
        raise ValueError("insufficient distinct physical cores for controlled worker layout")
    control = [group[0] for group in groups[:8]]
    available = groups[8:]
    nbuild = builds * build_cores
    build_ids = [group[0] for group in available[:nbuild]]
    build_chunks = [build_ids[i:i+build_cores] for i in range(0, nbuild, build_cores)]
    evaluation_ids = [group[0] for group in available[nbuild:nbuild+evaluation_cores*gpu_count]]
    gpu_chunks = [evaluation_ids[i:i+evaluation_cores]
                  for i in range(0, len(evaluation_ids), evaluation_cores)]
    return build_chunks, gpu_chunks, control
