"""Estimated evaluator host-memory admission, held through process cleanup."""
from __future__ import annotations

import hashlib
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .workloads import model_dependency

GIB = 1024 ** 3


class MemoryAdmissionCancelled(RuntimeError):
    pass


class ByteBudget:
    """FIFO byte admission; uncertain cleanup retains its reservation."""

    def __init__(self, capacity, event=None):
        if capacity <= 0:
            raise ValueError("positive memory budget required")
        self.capacity = capacity
        self.condition = threading.Condition()
        self.waiting = []
        self.allocations = {}
        self.event = event or (lambda *a, **k: None)

    def enqueue(self, key, size):
        with self.condition:
            if not 0 < size <= self.capacity:
                raise ValueError("one allocation cannot fit the memory budget")
            if key in self.allocations or any(k == key for k, _ in self.waiting):
                raise ValueError("duplicate memory reservation key")
            self.waiting.append((key, size))

    def acquire(self, key, cancelled):
        with self.condition:
            while True:
                if cancelled():
                    self.cancel(key)
                    raise MemoryAdmissionCancelled(key)
                entry = next(((k, n) for k, n in self.waiting if k == key), None)
                if entry is None:
                    raise MemoryAdmissionCancelled(key)
                if self.waiting[0] == entry and self.used + entry[1] <= self.capacity:
                    self.waiting.pop(0)
                    self.allocations[key] = {"bytes": entry[1], "state": "admitted"}
                    self._emit(key)
                    self.condition.notify_all()
                    return
                self.condition.wait(.2)

    @property
    def used(self):
        return sum(item["bytes"] for item in self.allocations.values())

    def snapshot(self):
        with self.condition:
            return {"capacity_bytes": self.capacity, "reserved_bytes": self.used,
                    "waiting_jobs": len(self.waiting), "state_bytes": {
                        state: sum(item["bytes"] for item in self.allocations.values() if item["state"] == state)
                        for state in ("admitted", "quarantined")}}

    def _emit(self, key):
        self.event(key, "evaluator_host_memory", "snapshot", **self.snapshot())

    def quarantine(self, key):
        with self.condition:
            self.allocations[key]["state"] = "quarantined"
            self._emit(key)

    def cancel(self, key):
        with self.condition:
            self.waiting = [(k, n) for k, n in self.waiting if k != key]
            self.condition.notify_all()

    def release(self, key):
        with self.condition:
            self.allocations.pop(key, None)
            self.cancel(key)
            self._emit(key)


def available_bytes():
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                     if line.startswith("MemAvailable:"))
    # Check the cgroup v2 mount and the common v1 memory-controller mount.
    for limit_path, current_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        try:
            limit = Path(limit_path).read_text().strip()
            if limit != "max":
                available = min(available, max(0, int(limit) - int(Path(current_path).read_text())))
        except (OSError, ValueError):
            pass
    return available


def evaluator_estimates(cases, registry, models):
    weights = {model["model"]: sum(item["bytes"] for item in model["files"]
               if item["name"].endswith((".bin", ".safetensors"))) for model in models}
    native = {task["source_sha256"] for task in registry["tasks"] if task.get("native_device_generator")}
    estimates = {}
    for case in cases:
        task = case["task"]
        sha = hashlib.sha256(task["code"].encode()).hexdigest()
        if task["source_sha256"] != sha:
            raise ValueError("task source digest mismatch")
        # Older measured cohorts store the same per-call estimate in contracts.
        spec = registry.get("plans", registry.get("contracts", {})).get(sha)
        if spec is None and sha not in native:
            raise ValueError("no evaluator memory plan for " + task["task_id"])
        if spec is not None and spec["source_sha256"] != sha:
            raise ValueError("input memory plan digest mismatch")
        dependency = model_dependency(task["code"])
        if dependency is not None and dependency not in weights:
            raise ValueError("no pinned model memory estimate for " + dependency)
        allocation = (spec.get("allocation_bytes", spec.get("arena_bytes")) if spec is not None else 0)
        if not isinstance(allocation, int) or allocation < 0:
            raise ValueError("invalid input memory estimate")
        checkpoint = weights.get(dependency, 0)
        # One native call with intermediates and transfer staging, two models,
        # a loader copy, and the evaluator runtime.
        estimates[task["task_id"]] = {"bytes": 8*GIB + 2*allocation + 3*checkpoint,
            "runtime_bytes": 8*GIB, "input_and_transfer_bytes": 2*allocation,
            "model_loading_bytes": 3*checkpoint, "model": dependency,
            "kind": "admission estimate, not an OS-enforced process memory limit"}
    return estimates


@contextmanager
def evaluator_memory(budget, estimate, journal, job, stage, cancelled):
    key = job + "/" + stage
    started = time.monotonic()
    journal.event(job, "evaluator_host_admission", "queued", bytes=estimate["bytes"])
    try:
        budget.enqueue(key, estimate["bytes"])
        budget.acquire(key, lambda: cancelled() or time.monotonic() - started > 1800)
    except BaseException:
        budget.cancel(key)
        journal.event(job, "evaluator_host_admission", "cancel_queued")
        raise
    admitted = time.monotonic()
    timing = {"queue_seconds": admitted-started, "bytes": estimate["bytes"]}
    journal.event(job, "evaluator_host_admission", "start", **timing)
    try:
        yield timing
    finally:
        if timing.get("quarantined"):
            budget.quarantine(key)
        else:
            budget.release(key)
        journal.event(job, "evaluator_host_admission", "finish", **timing,
                      service_seconds=time.monotonic()-admitted)
