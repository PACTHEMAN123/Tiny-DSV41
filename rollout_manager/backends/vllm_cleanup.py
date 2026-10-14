"""Last-resort cleanup of the actor owned by one failed manager instance."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import time


def live_owned_process(receipt):
    root = Path("/proc") / str(receipt["pid"])
    try:
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        command = (root / "cmdline").read_bytes()
    except FileNotFoundError:
        return False
    if int(fields[19]) != receipt["start_ticks"]:
        raise RuntimeError("owned actor identity changed; refusing process cleanup")
    if fields[0] in {"Z", "X"}:
        return False
    if hashlib.sha256(command).hexdigest() != receipt["cmdline_sha256"]:
        raise RuntimeError("owned actor identity changed; refusing process cleanup")
    return True


def stop_owned_actor(directory, *, term_seconds=5., kill_seconds=5.):
    """Return true only for forced termination; never claim a sampler drain.

    The caller supplies its fresh manager evidence path, not a discovered PID.
    Strong process identities and process-group membership are checked before
    each signal; unrelated processes are never selected by name or GPU usage.
    """
    directory = Path(directory)
    launch = directory / "vllm-actor" / "actor-launch.json"
    if not launch.exists():
        return False
    for path in (launch, launch.parent, directory, *directory.parents):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("owned actor cleanup refuses symlink evidence")
    if directory.stat().st_uid != os.getuid() or directory.stat().st_mode & 0o077:
        raise ValueError("owned actor cleanup requires a same-owner private directory")
    actor = json.loads(launch.read_text())["actor_process"]
    ownership = directory / "actor-ownership.json"
    if ownership.is_symlink():
        raise ValueError("owned actor cleanup refuses symlink ownership evidence")
    workers = json.loads(ownership.read_text())["workers"] if ownership.exists() else []
    receipts = [actor, *workers]

    def alive():
        live = [receipt for receipt in receipts if live_owned_process(receipt)]
        if any(os.getpgid(receipt["pid"]) != actor["pid"] for receipt in live):
            raise RuntimeError("owned actor process group changed; refusing process cleanup")
        return live

    if not alive():
        return False
    for sig, duration in ((signal.SIGTERM, term_seconds), (signal.SIGKILL, kill_seconds)):
        # Recheck all stored identities immediately before signalling.
        if not alive():
            break
        os.killpg(actor["pid"], sig)
        deadline = time.monotonic() + duration
        while alive() and time.monotonic() < deadline:
            time.sleep(.05)
    if alive():
        raise RuntimeError("owned vLLM process group did not exit after bounded cleanup")
    from ..terminal_bench.controller import write_new
    write_new(directory / "manager-forced-actor-exit.json", {
        "actor_process": actor, "drain_complete": False, "forced_termination": True,
        "automatic_replay": False, "finished_unix": time.time()})
    return True
