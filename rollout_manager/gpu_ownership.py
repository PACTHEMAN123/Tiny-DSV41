"""Ownership checks for explicitly shared GPUs in the reviewed load experiment."""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .adapters.process import gpu_processes


class GpuOwnershipError(RuntimeError):
    pass


class ReviewedGpuOwners:
    def __init__(self, settle_seconds=5.0):
        self.lock = threading.Lock()
        self.gpu_locks = {}
        self.workspaces = {}
        self.proc = Path("/proc")
        self.settle_seconds = settle_seconds
        self.observations = []
        self.checks = []

    def gpu_lock(self, gpu):
        with self.lock:
            return self.gpu_locks.setdefault(gpu, threading.Lock())

    def await_absent(self, gpu, pid):
        started = time.monotonic()
        while True:
            present = any(int(row.split(",")[1].strip()) == pid for row in gpu_processes(gpu))
            elapsed = time.monotonic() - started
            if not present or elapsed >= self.settle_seconds:
                self.observations.append({"unix_ns": time.time_ns(), "gpu": gpu, "pid": pid,
                    "settle_seconds": elapsed, "context_disappeared": not present})
                return not present
            time.sleep(min(.1, self.settle_seconds - elapsed))

    @contextmanager
    def lease(self, gpu, workspace):
        workspace = workspace.resolve()
        lock = self.gpu_lock(gpu)
        with lock:
            active = self.workspaces.setdefault(gpu, set())
            if workspace in active:
                raise GpuOwnershipError("workspace already owns a GPU lease")
            active.add(workspace)
        try:
            yield lambda: self.verify(gpu, workspace)
        finally:
            with lock:
                self.workspaces[gpu].remove(workspace)

    def verify(self, gpu, workspace):
        """Require this workspace to be idle and every peer to be a registered harness."""
        queued = time.monotonic()
        record = {"gpu": gpu, "workspace": str(workspace), "accepted": False}
        try:
            with self.gpu_lock(gpu):
                record["lock_wait_seconds"] = time.monotonic() - queued
                peers = self.workspaces.get(gpu, set()) - {workspace}
                for row in gpu_processes(gpu):
                    try:
                        pid = int(row.split(",")[1].strip())
                        process = self.proc / str(pid)
                        before = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
                        cwd = (process / "cwd").resolve(strict=True)
                        argv = (process / "cmdline").read_bytes().split(bytes([0]))
                        environment = (process / "environ").read_bytes().split(bytes([0]))
                        after = (process / "stat").read_text().rsplit(")", 1)[1].split()[19]
                    except FileNotFoundError:
                        # NVML may retain a context after /proc has removed its owner.
                        if not self.await_absent(gpu, pid):
                            raise GpuOwnershipError(f"GPU {gpu} context PID {pid} remains without inspectable identity")
                        continue
                    except (OSError, ValueError, IndexError) as exc:
                        raise GpuOwnershipError("cannot establish GPU peer ownership") from exc
                    if (before != after or cwd not in peers or b"rollout_manager.adapters.harness_child" not in argv
                            or ("CUDA_VISIBLE_DEVICES=" + gpu).encode() not in environment):
                        raise GpuOwnershipError("GPU context is not a verified peer harness")
                record["accepted"] = True
            return True
        finally:
            record.update(unix_ns=time.time_ns(), wall_seconds=time.monotonic() - queued)
            with self.lock:
                self.checks.append(record)
