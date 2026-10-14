#!/usr/bin/env python3
"""Run GPU tests serially, keeping an immutable memory baseline and an audit trail.

Linux only, Python >= 3.9, standard library plus nvidia-smi. All invocations for a
matrix MUST use the same state directory on a local filesystem (not an OSS/FUSE
mount). This is a cooperative gate, not a sandbox or a host-wide GPU reservation.
It cannot attribute allocations hidden by the container's PID namespace.

Example:
  python scripts/gpu_test_gate.py run --state-dir artifacts/gpu-gate --gpus 0,1 \
    --name qwen --timeout-seconds 1800 -- python -m torch.distributed.run ...
  python scripts/gpu_test_gate.py status --state-dir artifacts/gpu-gate

A failed, timed-out, interrupted, or unclean invocation latches the gate closed.
There is deliberately no automatic reset/rebaseline. Inspect the evidence and
resolve the cause before starting a separately reviewed matrix/state directory.
Unknown processes are never signalled. The gate uses SIGTERM, never SIGKILL or
GPU reset; an uncooperative owned child can remain alive and is reported as such.
Python builds lacking pidfd bindings use the same kernel operations through
ctypes on native 64-bit Linux x86_64/aarch64; unavailable/denied operations block.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import datetime
import errno
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


class GateError(RuntimeError):
    pass


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def machine_identity():
    return {"hostname": socket.gethostname(), "uid": os.getuid(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "pid_namespace": os.readlink("/proc/self/ns/pid")}


def parse_proc_stat(text):
    # comm can itself contain spaces and ')' characters.
    left, right = text.index("("), text.rindex(")")
    fields = text[right + 2:].split()
    return {"pid": int(text[:left].strip()), "comm": text[left + 1:right],
            "state": fields[0], "ppid": int(fields[1]),
            "pgid": int(fields[2]), "sid": int(fields[3]),
            "start_ticks": int(fields[19])}


def read_process(pid):
    path = Path("/proc") / str(pid)
    try:
        record = parse_proc_stat((path / "stat").read_text())
        record["uid"] = path.stat().st_uid
        try:
            record["argv"] = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        except OSError:
            record["argv"] = None
        return record
    except (OSError, ValueError, IndexError):
        return None


def process_inventory():
    processes, unreadable = [], []
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            record = read_process(int(path.name))
            if record is None:
                unreadable.append(int(path.name))
            else:
                processes.append(record)
    return {"processes": sorted(processes, key=lambda item: item["pid"]),
            "unreadable_or_disappeared_pids": sorted(unreadable),
            "scope": "current PID namespace only; absence does not prove host process exit"}


def identity(process):
    return {key: process[key] for key in ("pid", "start_ticks", "uid")}


def same_process(expected, actual):
    return actual is not None and identity(actual) == identity(expected)


def alive(process):
    return process is not None and process["state"] not in ("Z", "X", "x")


def owned_descendants(supervisor, processes):
    """Require the supervisor identity, then prove ancestry in this snapshot."""
    records = {process["pid"]: process for process in processes}
    if not same_process(supervisor, records.get(supervisor["pid"])):
        raise GateError("supervisor identity changed; refusing to infer ownership")
    owned = {supervisor["pid"]}
    while True:
        added = {pid for pid, process in records.items()
                 if process["ppid"] in owned and pid not in owned}
        if not added:
            break
        owned.update(added)
    return [records[pid] for pid in sorted(owned - {supervisor["pid"]})]


def _pidfd_syscall(number, *arguments):
    # Native 64-bit x86 and AArch64 both use 424/434. Do not guess other ABIs.
    # https://github.com/torvalds/linux/blob/v6.6/arch/x86/entry/syscalls/syscall_64.tbl
    # https://github.com/torvalds/linux/blob/v6.6/include/uapi/asm-generic/unistd.h
    if (sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "aarch64")
            or ctypes.sizeof(ctypes.c_void_p) != 8 or ctypes.sizeof(ctypes.c_long) != 8):
        raise OSError(errno.ENOSYS, "pidfd syscall fallback requires native Linux x86_64/aarch64")
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    syscall.argtypes = [ctypes.c_long]  # remaining arguments are explicitly typed varargs
    result = syscall(ctypes.c_long(number), *arguments)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, f"pidfd syscall {number}: {os.strerror(error)}")
    return int(result)


def pidfd_open(pid, flags=0):
    native = getattr(os, "pidfd_open", None)
    if callable(native):
        return native(pid, flags)
    return _pidfd_syscall(434, ctypes.c_int(pid), ctypes.c_uint(flags))


def pidfd_send_signal(fd, sig):
    native = getattr(signal, "pidfd_send_signal", None)
    if callable(native):
        return native(fd, sig)
    return _pidfd_syscall(424, ctypes.c_int(fd), ctypes.c_int(sig), ctypes.c_void_p(), ctypes.c_uint(0))


def pidfd_backend_status():
    """Signal 0 probes permissions/support using only this process's own pidfd."""
    backend = {"open": "stdlib" if callable(getattr(os, "pidfd_open", None)) else "ctypes-syscall",
               "send_signal": "stdlib" if callable(getattr(signal, "pidfd_send_signal", None)) else "ctypes-syscall"}
    if sys.platform != "linux":
        return {"available": False, "backend": backend, "error": "Linux required"}
    try:
        fd = pidfd_open(os.getpid(), 0)
        try:
            pidfd_send_signal(fd, 0)
        finally:
            os.close(fd)
    except (OSError, AttributeError) as exc:
        return {"available": False, "backend": backend, "error": str(exc), "errno": getattr(exc, "errno", None)}
    return {"available": True, "backend": backend}


def signal_verified(process, sig=signal.SIGTERM):
    """pidfd closes the PID-reuse race between verification and signalling."""
    if process["uid"] != os.getuid() or not same_process(process, read_process(process["pid"])):
        return {"identity": identity(process), "sent": False, "reason": "identity/uid mismatch or exited"}
    try:
        fd = pidfd_open(process["pid"], 0)
        try:
            if not same_process(process, read_process(process["pid"])):
                return {"identity": identity(process), "sent": False, "reason": "identity changed after pidfd_open"}
            pidfd_send_signal(fd, sig)
        finally:
            os.close(fd)
        return {"identity": identity(process), "sent": True, "signal": int(sig)}
    except OSError as exc:
        return {"identity": identity(process), "sent": False, "reason": str(exc)}


def require_linux():
    if sys.platform != "linux" or not Path("/proc/self/stat").is_file():
        raise GateError("execution requires Linux /proc; CPU parsing/policy tests can run elsewhere")
    status = pidfd_backend_status()
    if not status["available"]:
        raise GateError(f"safe pidfd backend unavailable; no unchecked kill fallback: {status}")
    return status


def enable_subreaper():
    # Orphaned/double-forked descendants reparent here, not to container init.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise GateError(f"cannot enable child subreaper: errno {ctypes.get_errno()}")


def supervise(payload_path):
    """Internal child supervisor. It stays alive as the ancestry/session anchor."""
    pidfd_status = require_linux()
    payload = json.loads(Path(payload_path).read_text())
    directory = Path(payload["case_dir"])
    enable_subreaper()
    supervisor = read_process(os.getpid())
    if supervisor is None or supervisor["sid"] != os.getpid() or supervisor["pgid"] != os.getpid():
        raise GateError("supervisor must have its own session and process group")
    stopped = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda number, _frame: stopped.append(number))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(payload["gpu_uuids"])
    child = subprocess.Popen(payload["command"], cwd=payload["cwd"], env=env,
                             stdin=subprocess.DEVNULL)
    started = now()
    write_json(directory / "process.json", {"supervisor": supervisor,
               "command_pid": child.pid, "pgid": supervisor["pgid"],
               "machine": machine_identity(), "pidfd_backend": pidfd_status, "started": started})
    observed, signals = {}, []
    deadline = time.monotonic() + payload["timeout_seconds"]
    reason = None

    def collect():
        descendants = owned_descendants(supervisor, process_inventory()["processes"])
        for process in descendants:
            observed[(process["pid"], process["start_ticks"])] = process
        write_json(directory / "observed-processes.json", list(observed.values()))
        return [process for process in descendants if alive(process)]

    while True:
        collect()
        rc = child.poll()
        if stopped:
            reason = "interrupted"
            break
        if time.monotonic() >= deadline:
            reason = "timeout"
            break
        if rc is not None:
            reason = "command_failed" if rc else None
            break
        time.sleep(0.2)

    # Give normal exits a short interval for worker teardown before classifying
    # background descendants as a leak. The wrapper remains their subreaper.
    grace_deadline = time.monotonic() + (0.5 if reason is None else 0)
    remaining = collect()
    while remaining and time.monotonic() < grace_deadline:
        time.sleep(0.1)
        remaining = collect()
    if remaining and reason is None:
        reason = "residual_processes_after_success"
    cleanup_deadline = time.monotonic() + payload["terminate_wait_seconds"]
    signalled = set()
    while remaining:
        for process in remaining:
            key = (process["pid"], process["start_ticks"])
            if key not in signalled:
                signals.append(signal_verified(process))
                signalled.add(key)
        if time.monotonic() >= cleanup_deadline:
            break
        time.sleep(0.1)
        child.poll()
        remaining = collect()
    rc = child.poll()
    # Only after Popen has reaped the direct command may we reap adopted children.
    if rc is not None:
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG)[0] == 0:
                    break
            except ChildProcessError:
                break
    # A setuid child or restricted /proc must not be mistaken for successful
    # cleanup just because it disappeared from the readable process inventory.
    try:
        unresolved_children = [int(value) for value in
                               Path(f"/proc/self/task/{os.getpid()}/children").read_text().split()]
    except OSError as exc:
        unresolved_children = [f"cannot verify child list: {exc}"]
    if unresolved_children and reason is None:
        reason = "unresolved_child_processes"
    result = {"started": started, "finished": now(), "supervisor": supervisor,
              "command_pid": child.pid, "command_exit_code": rc, "reason": reason,
              "observed_processes": list(observed.values()), "signals": signals,
              "remaining_processes": remaining, "unresolved_children": unresolved_children,
              "clean": not remaining and not unresolved_children,
              "passed": reason is None and rc == 0 and not remaining and not unresolved_children}
    write_json(directory / "process-result.json", result)
    return 0 if result["passed"] else 1


def run_query(command):
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GateError(f"GPU query failed: {exc}") from exc
    if proc.returncode:
        raise GateError(f"GPU query exit {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout


def parse_gpu_snapshot(csv_text, xml_text):
    xml_gpus = {gpu.findtext("uuid"): gpu for gpu in ET.fromstring(xml_text).findall("gpu")}
    result = []
    for row in csv.reader(csv_text.splitlines()):
        if not row:
            continue
        if len(row) != 7:
            raise GateError("unexpected nvidia-smi GPU CSV columns")
        index, uuid, pci, total, used, free, util = [value.strip() for value in row]
        xml = xml_gpus.get(uuid)
        if xml is None:
            raise GateError(f"GPU {uuid} missing from nvidia-smi XML")
        try:
            metrics = {"total_mib": float(total), "used_mib": float(used),
                       "free_mib": float(free), "utilization_percent": float(util)}
        except ValueError as exc:
            raise GateError(f"unavailable GPU metrics for {uuid}") from exc
        if not all(math.isfinite(value) and value >= 0 for value in metrics.values()):
            raise GateError(f"invalid GPU metrics for {uuid}")
        processes = []
        process_node = xml.find("processes")
        if process_node is None or (process_node.text or "").strip():
            raise GateError(f"GPU process inventory unavailable for {uuid}")
        for proc in xml.findall("./processes/process_info"):
            processes.append({child.tag: child.text for child in proc})
        result.append({"index": int(index), "uuid": uuid, "pci_bus_id": pci,
                       **metrics, "processes": processes,
                       "mig_mode": xml.findtext("./mig_mode/current_mig"),
                       "reserved": xml.findtext("./fb_memory_usage/reserved")})
    if not result or len({gpu["uuid"] for gpu in result}) != len(result):
        raise GateError("empty or duplicate GPU inventory")
    return result


def take_snapshot():
    csv_text = run_query(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,memory.total,memory.used,memory.free,utilization.gpu",
                          "--format=csv,noheader,nounits"])
    xml_text = run_query(["nvidia-smi", "-q", "-x"])
    return {"time": now(), "machine": machine_identity(),
            "gpus": parse_gpu_snapshot(csv_text, xml_text),
            "visible_processes": process_inventory(),
            "raw_gpu_csv": csv_text, "raw_gpu_xml": xml_text}


def select_gpus(snapshot, selection):
    selected = []
    for value in selection.split(","):
        matches = [gpu for gpu in snapshot["gpus"] if value in (str(gpu["index"]), gpu["uuid"])]
        if len(matches) != 1:
            raise GateError(f"GPU selection must name an exact physical index or UUID: {value!r}")
        if matches[0]["uuid"] in [gpu["uuid"] for gpu in selected]:
            raise GateError("duplicate GPU selection")
        selected.append(matches[0])
    return selected


def memory_issues(current, baseline, policy, initial=False):
    indexed = {gpu["uuid"]: gpu for gpu in current}
    issues = []
    for original in baseline:
        uuid = original["uuid"]
        gpu = indexed.get(uuid)
        if gpu is None:
            issues.append(f"{uuid}: GPU disappeared")
            continue
        if gpu["mig_mode"] not in ("Disabled", "N/A", "[N/A]"):
            issues.append(f"{uuid}: physical non-MIG GPU is required")
        if gpu["processes"]:
            issues.append(f"{uuid}: NVML reports active GPU processes")
        if gpu["utilization_percent"] > policy["max_utilization_percent"]:
            issues.append(f"{uuid}: utilization {gpu['utilization_percent']}% exceeds policy")
        if gpu["free_mib"] < policy["min_free_mib"]:
            issues.append(f"{uuid}: free {gpu['free_mib']} MiB below required minimum")
        if initial and gpu["used_mib"] > policy["max_baseline_used_mib"]:
            issues.append(f"{uuid}: baseline used {gpu['used_mib']} MiB exceeds policy, even if PID list is empty")
        if gpu["pci_bus_id"] != original["pci_bus_id"] or gpu["total_mib"] != original["total_mib"]:
            issues.append(f"{uuid}: GPU identity/capacity changed")
        if gpu["used_mib"] > original["used_mib"] + policy["tolerance_mib"]:
            issues.append(f"{uuid}: used memory has not returned to baseline")
        if gpu["free_mib"] < original["free_mib"] - policy["tolerance_mib"]:
            issues.append(f"{uuid}: free memory has not returned to baseline")
    return issues


@contextlib.contextmanager
def exclusive_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "gate.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise GateError("another invocation holds this gate; no command launched") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def wait_for_baseline(directory, phase, baseline, policy, wait_seconds, poll_seconds,
                      snapshot_fn=take_snapshot, initial=False):
    deadline = time.monotonic() + wait_seconds
    consecutive, attempt = 0, 0
    while True:
        snapshot = snapshot_fn()
        write_json(directory / f"{phase}-{attempt:03d}.json", snapshot)
        issues = memory_issues(snapshot["gpus"], baseline, policy, initial=initial)
        consecutive = consecutive + 1 if not issues else 0
        if consecutive >= 2:
            write_json(directory / f"{phase}.json", snapshot)
            return snapshot
        if time.monotonic() >= deadline:
            raise GateError("; ".join(issues) if issues else "insufficient consecutive stable snapshots")
        attempt += 1
        time.sleep(poll_seconds)


def launch_supervisor(payload_path, directory):
    with (directory / "command.log").open("ab", buffering=0) as log:
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_supervise", str(payload_path)],
                                stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True)
        write_json(directory / "launcher.json", {"pid": proc.pid, "pgid": proc.pid, "started": now()})
        interrupted = False
        old_handlers = {}

        def interrupt(_number, _frame):
            nonlocal interrupted
            interrupted = True
            path = directory / "process.json"
            if path.exists():
                record = json.loads(path.read_text())["supervisor"]
                signal_verified(record)
            # Before process.json exists, no identity is known: do not guess.

        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.signal(sig, interrupt)
        try:
            while proc.poll() is None:
                if interrupted:
                    interrupt(None, None)
                time.sleep(0.2)
            return {"supervisor_exit_code": proc.returncode, "interrupted": interrupted}
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def run_gate(args, snapshot_fn=take_snapshot, supervisor_fn=launch_supervisor,
             machine_fn=machine_identity):
    directory = args.state_dir.resolve()
    policy = {key: getattr(args, key) for key in
              ("tolerance_mib", "max_baseline_used_mib", "min_free_mib", "max_utilization_percent")}
    with exclusive_lock(directory):
        state_path = directory / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        machine = machine_fn()
        if state:
            if state["machine"] != machine:
                raise GateError("state belongs to another host/boot/UID/PID namespace")
            if state["status"] != "passed":
                raise GateError(f"gate latched {state['status']}; inspect {state.get('last_case')}; no command launched")
            if state["policy"] != policy:
                raise GateError("policy must remain unchanged across this matrix")
            previous = Path(state["last_case"]) / "process-result.json"
            report = json.loads(previous.read_text())
            for process in [report["supervisor"], *report["observed_processes"]]:
                current = read_process(process["pid"])
                if same_process(process, current) and alive(current):
                    raise GateError(f"previous owned PID {process['pid']} remains alive; no command launched")
        sequence = 1 if state is None else state["sequence"] + 1
        case = directory / f"{sequence:04d}-{args.name}"
        case.mkdir()
        initial = state is None
        state = state or {"schema": 1, "machine": machine, "policy": policy, "baseline": None}
        state.update(status="checking", sequence=sequence, last_case=str(case), updated=now())
        write_json(state_path, state)
        result = {"name": args.name, "command": args.command, "started": now(), "passed": False}
        try:
            first = snapshot_fn()
            write_json(case / "before-initial.json", first)
            selected = select_gpus(first, args.gpus)
            if initial:
                state["baseline"] = selected
                issues = memory_issues(first["gpus"], selected, policy, initial=True)
                if issues:
                    raise GateError("; ".join(issues))
            elif [gpu["uuid"] for gpu in selected] != [gpu["uuid"] for gpu in state["baseline"]]:
                raise GateError("GPU selection/order must remain unchanged across this matrix")
            write_json(state_path, state)
            wait_for_baseline(case, "before", state["baseline"], policy, args.recovery_seconds,
                              args.poll_seconds, snapshot_fn, initial=initial)
            payload = {"command": args.command, "cwd": str(Path.cwd()), "case_dir": str(case),
                       "gpu_uuids": [gpu["uuid"] for gpu in state["baseline"]],
                       "timeout_seconds": args.timeout_seconds,
                       "terminate_wait_seconds": args.terminate_wait_seconds}
            write_json(case / "invocation.json", payload)
            state.update(status="running", updated=now())
            write_json(state_path, state)
            launcher = supervisor_fn(case / "invocation.json", case)
            result.update(launcher)
            report_path = case / "process-result.json"
            report = json.loads(report_path.read_text()) if report_path.exists() else None
            result["process_result"] = report
            # Even a failed/aborted command gets an after snapshot and a bounded
            # recovery check. Passing the memory check never clears a failure.
            try:
                wait_for_baseline(case, "after", state["baseline"], policy, args.recovery_seconds,
                                  args.poll_seconds, snapshot_fn)
                result["memory_recovered"] = True
            except Exception as exc:
                result["memory_recovered"] = False
                result["recovery_error"] = str(exc)
            result["passed"] = (launcher["supervisor_exit_code"] == 0 and not launcher["interrupted"]
                                and report is not None and report["passed"]
                                and result["memory_recovered"])
        except BaseException as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["finished"] = now()
        write_json(case / "result.json", result)
        state.update(status="passed" if result["passed"] else "failed", updated=now())
        write_json(state_path, state)
        return result


def parser():
    root = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = root.add_subparsers(dest="action", required=True)
    status = sub.add_parser("status", help="read persisted state without running GPU commands")
    status.add_argument("--state-dir", type=Path, required=True)
    run = sub.add_parser("run", help="gate and run exactly one command; invoke serially for a matrix")
    run.add_argument("--state-dir", type=Path, required=True)
    run.add_argument("--gpus", required=True, help="physical nvidia-smi indices or full GPU UUIDs")
    run.add_argument("--name", required=True)
    run.add_argument("--timeout-seconds", type=float, default=3600)
    run.add_argument("--terminate-wait-seconds", type=float, default=15)
    run.add_argument("--recovery-seconds", type=float, default=60)
    run.add_argument("--poll-seconds", type=float, default=2)
    run.add_argument("--tolerance-mib", type=float, default=256)
    run.add_argument("--max-baseline-used-mib", type=float, default=1024,
                     help="first-run upper bound, not waived by an empty GPU PID list")
    run.add_argument("--min-free-mib", type=float, default=1024)
    run.add_argument("--max-utilization-percent", type=float, default=5)
    run.add_argument("command", nargs=argparse.REMAINDER)
    internal = sub.add_parser("_supervise", help=argparse.SUPPRESS)
    internal.add_argument("payload")
    return root


def main(argv=None):
    root = parser()
    args = root.parse_args(argv)
    try:
        if args.action == "status":
            print((args.state_dir / "state.json").read_text())
            return 0
        if args.action == "_supervise":
            return supervise(args.payload)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.name):
            raise GateError("name must contain 1..64 letters, numbers, '.', '_' or '-'")
        if args.command[:1] == ["--"]:
            args.command = args.command[1:]
        if not args.command:
            raise GateError("a command after -- is required")
        for key in ("timeout_seconds", "terminate_wait_seconds", "recovery_seconds", "poll_seconds",
                    "min_free_mib", "max_baseline_used_mib", "tolerance_mib", "max_utilization_percent"):
            value = getattr(args, key)
            if not math.isfinite(value) or value < 0 or (key.endswith("seconds") and value == 0):
                raise GateError(f"{key} must be finite and positive (memory/utilization thresholds may be zero)")
        if args.recovery_seconds < args.poll_seconds or args.max_utilization_percent > 100:
            raise GateError("recovery must allow two snapshots; utilization must be <= 100")
        require_linux()
        result = run_gate(args)
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 1
    except (GateError, OSError, ValueError, KeyError) as exc:
        print(f"GPU_GATE_BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
