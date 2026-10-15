"""Bounded, owned graph-enabled vLLM launch using an archived serving recipe."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def identity(pid):
    proc = Path("/proc") / str(pid)
    command = (proc / "cmdline").read_bytes()
    return {"pid": pid, "start_ticks": (proc / "stat").read_text().rsplit(")", 1)[1].split()[19],
            "command_sha256": hashlib.sha256(command).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--nsys", type=Path, required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--ttl", type=int, default=14400)
    args = parser.parse_args()
    if not 60 <= args.ttl <= 21600 or not args.session.replace("-", "").isalnum():
        parser.error("bounded lifetime and alphanumeric session name required")
    owners = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], text=True)
    if owners.strip():
        raise RuntimeError("GPU ownership must be reconciled before launch: " + owners)
    for port in (8000, 8765):
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError("port is already owned: " + str(port))
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    recipe = json.loads(args.recipe.read_text())
    argv = list(recipe["identity"]["argv"])
    if "--enforce-eager" not in argv or "--compilation-config" in argv:
        raise ValueError("expected the inspected eager recipe")
    argv.remove("--enforce-eager")
    argv += ["--compilation-config", json.dumps({"cudagraph_capture_sizes": [1, 2, 4, 8]})]
    hook = Path(__file__).with_name("vllm_timeline_entry.py").resolve()
    argv.insert(1, str(hook))
    env = dict(os.environ, **recipe["environment"])
    temp = tempfile.mkdtemp(prefix="graph-v31-", dir="/tmp")
    env.update(VLLM_RPC_BASE_PATH=temp, TMPDIR=temp, VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
               ROLLOUT_ENGINE_TIMELINE=str(root / "engine-timing.jsonl"))
    cores = sorted(os.sched_getaffinity(0))[:-4]
    command = [str(args.nsys), "launch", "--session-new=" + args.session,
               "--trace=cuda,nvtx,osrt", "--cuda-graph-trace=graph", "--wait=all",
               "taskset", "-c", ",".join(map(str, cores))] + argv
    save(root / "launch.json", {
        "supervisor": identity(os.getpid()), "started_unix_ns": time.time_ns(),
        "argv": command, "cpu_affinity": cores, "ttl": args.ttl, "session": args.session,
        "nsys_tmpdir": temp, "recipe": str(args.recipe.resolve()),
        "recipe_sha256": hashlib.sha256(args.recipe.read_bytes()).hexdigest(),
        "hook_sha256": hashlib.sha256(hook.read_bytes()).hexdigest(),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "changes": ["remove enforce-eager", "capture batch sizes 1,2,4,8", "record engine timestamps"],
        "environment": {key: env[key] for key in recipe["environment"]},
    })
    reason, child, gateway = None, None, None

    def stop(signum, frame):
        nonlocal reason
        reason = "signal"

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, stop)
    started = time.monotonic()
    with (root / "server.log").open("xb") as log, (root / "gateway.log").open("xb") as gateway_log:
        try:
            gateway = subprocess.Popen(recipe["gateway"]["argv"], cwd=recipe["cwd"], env=env,
                stdin=subprocess.DEVNULL, stdout=gateway_log, stderr=subprocess.STDOUT, start_new_session=True)
            save(root / "gateway-identity.json", identity(gateway.pid))
            for _ in range(30):
                if gateway.poll() is not None:
                    raise RuntimeError("cached-kernel gateway failed")
                try:
                    with urllib.request.urlopen("http://127.0.0.1:8765/health", timeout=1):
                        break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError("kernel gateway readiness")
            child = subprocess.Popen(command, cwd=recipe["cwd"], env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            save(root / "child-identity.json", identity(child.pid))
            while child.poll() is None and reason is None:
                memory = {p[0].rstrip(":"): int(p[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                          if (p := line.split()) and p[0] in ("MemTotal:", "MemAvailable:")}
                if memory["MemAvailable"] < .18 * memory["MemTotal"]:
                    reason = "host_memory_guard"
                if time.monotonic() - started > args.ttl:
                    reason = "duration_guard"
                if log.tell() > 64 * 1024**2:
                    reason = "log_guard"
                time.sleep(.2)
        finally:
            if child is not None and child.poll() is None:
                stopped = subprocess.run([str(args.nsys), "shutdown", "--session=" + args.session,
                    "--kill=sigterm"], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=60)
                save(root / "nsys-shutdown.json", {"returncode": stopped.returncode})
                child.wait(timeout=60)
            if gateway is not None and gateway.poll() is None:
                gateway.terminate()
                gateway.wait(timeout=15)
            save(root / "exit.json", {"reason": reason, "finished_unix_ns": time.time_ns(),
                "returncode": child.returncode if child else None,
                "wall_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
