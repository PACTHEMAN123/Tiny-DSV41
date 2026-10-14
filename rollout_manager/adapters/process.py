from __future__ import annotations

import os
import selectors
import signal
import subprocess
import time
from pathlib import Path
from typing import Callable


def run_process(argv: list[str], cwd: Path, env: dict[str, str], log: Path, *,
                timeout: float, heartbeat: Callable[[], None], max_log: int = 4 * 1024 * 1024,
                pass_fds: tuple[int, ...] = (), cancelled: Callable[[], bool] | None = None) -> dict:
    """Bound a reviewed child and its process group. This is not a sandbox."""
    start = time.monotonic()
    started_unix_ns = time.time_ns()
    env = dict(env, ROLLOUT_LAUNCH_MONOTONIC_NS=str(time.monotonic_ns()))
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True, pass_fds=pass_fds)
    spawn_seconds = time.monotonic() - start
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    count = 0
    stopped_for = None
    next_heartbeat = start
    try:
        with log.open("wb") as output:
            while selector.get_map() or process.poll() is None:
                now = time.monotonic()
                if cancelled is not None and cancelled():
                    stopped_for = "cancelled"
                    break
                if now >= next_heartbeat:
                    heartbeat()
                    next_heartbeat = now + 5
                if now - start > timeout or count > max_log:
                    stopped_for = "deadline" if now - start > timeout else "log_bound"
                    break
                for key, _ in selector.select(.2):
                    block = os.read(key.fileobj.fileno(), 65536)
                    if block:
                        output.write(block[:max(0, max_log - count)])
                        count += len(block)
                    else:
                        selector.unregister(key.fileobj)
            output.flush()
            os.fsync(output.fileno())
    finally:
        reap_start = time.monotonic()
        # Also reap lingering group members after an apparently successful parent exit.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)
        selector.close()
        process.stdout.close()
    group_stopped = False
    for _ in range(20):
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            group_stopped = True
            break
        time.sleep(.1)
    return {"returncode": process.returncode, "wall_seconds": time.monotonic() - start,
            "started_unix_ns": started_unix_ns, "spawn_seconds": spawn_seconds,
            "reap_seconds": time.monotonic() - reap_start,
            "stopped_for": stopped_for, "process_group_stopped": group_stopped,
            "log": str(log), "log_bytes": min(count, max_log), "pid": process.pid}


def gpu_processes(gpu_uuid: str) -> list[str]:
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                    "--format=csv,noheader"], text=True, timeout=10)
    return [line for line in text.splitlines() if line.split(",", 1)[0].strip() == gpu_uuid]
