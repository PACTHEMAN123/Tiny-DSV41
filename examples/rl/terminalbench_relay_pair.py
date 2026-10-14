"""Launch a fresh owned training browser relay or snapshot its actual CPU readiness.

The operator pairs the private pages in Chrome: open A's page first, then open
the CPU companion from that page. This helper prints private file paths only;
it never prints a token, launch URL fragment, or creates browser readiness.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

PORTS = {"gpu": 18093, "cpu": 18092}
PROFILE = "native-pytorch-v1"
PROFILES = (PROFILE, "vllm-training-v1")


def relay_port(args, role=None):
    role = role or args.role
    value = getattr(args, "port", None)
    value = PORTS[role] if value is None else value
    if type(value) is not int or not 1024 <= value <= 65535:
        raise ValueError("relay port must be an integer within 1024..65535")
    return value


def serving_profile(args):
    profile = getattr(args, "serving_profile", PROFILE)
    if profile not in PROFILES:
        raise ValueError("An explicit supported owned training profile is required")
    return profile


def configure_source(value):
    """Allow the CPU wheel target and copied entrypoint to live independently."""
    source = real_path(value, directory=True)
    package = source / "rollout_manager/__init__.py"
    if not package.is_file():
        raise ValueError("explicit source-root must contain the selected shared package")
    sys.path.insert(0, str(source))
    import rollout_manager
    if Path(rollout_manager.__file__).absolute() != package:
        raise ValueError("imported shared package differs from the explicit source-root")
    return source


def real_path(value, *, directory=False):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("explicit absolute unlinked project path required")
    if directory and not path.is_dir():
        raise ValueError("existing real project directory required")
    return path


def scoped(project, value):
    path = real_path(value)
    if project not in path.parents:
        raise ValueError("new relay files must remain below the selected existing project")
    return path


def private_json(path, limit=65536):
    from rollout_manager.terminal_bench.browser_relay.relay_server import private_controller_file
    return private_controller_file(path, limit)


def local_api(root, port, route):
    from rollout_manager.terminal_bench.browser_relay.relay_server import native_controller_token
    token = native_controller_token(root / "api-token.txt")
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("GET", route, headers={"X-Relay-Token": token})
        response = connection.getresponse()
        raw = response.read(65537)
        if response.status != 200 or len(raw) > 65536 or response.length not in (0, None):
            raise RuntimeError("actual local relay API is unavailable")
        from rollout_manager.terminal_bench.controller import unique
        return json.loads(raw, object_pairs_hook=unique)
    finally:
        connection.close()


def launch_plan(args, project, root):
    profile = serving_profile(args)
    port = relay_port(args)
    source = real_path(args.source_root, directory=True)
    if not (source / "rollout_manager/terminal_bench/browser_relay/relay_server.py").is_file():
        raise ValueError("selected source must include the matched owned training browser relay")
    python = Path(args.python)
    # Preserve the supplied venv executable. Its symlink is intentionally not resolved.
    if not python.is_absolute() or not python.is_file() or not python.name.startswith("python"):
        raise ValueError("invoke the verified absolute venv Python executable directly")
    public, peer = urlsplit(args.public_base), urlsplit(args.peer_origin)
    if (public.scheme != "https" or not public.hostname or public.username or public.password or
            public.query or public.fragment or not public.path.endswith("/proxy/%d/" % port)):
        raise ValueError("public-base must be the current HTTPS WebIDE proxy base for the selected relay port")
    if (peer.scheme != "https" or not peer.hostname or peer.username or peer.password or
            peer.path not in ("", "/") or peer.query or peer.fragment):
        raise ValueError("peer-origin must be the exact other browser HTTPS origin")
    if args.proxy_origin:
        proxy = urlsplit(args.proxy_origin)
        if (proxy.scheme not in ("http", "https") or not proxy.hostname or proxy.username or proxy.password or
                proxy.path or proxy.query or proxy.fragment):
            raise ValueError("proxy-origin must be an exact explicitly inspected HTTP or HTTPS gateway origin")
    if not root.parent.is_dir():
        raise ValueError("fresh relay state needs an existing project parent")
    context = scoped(project, root.with_name(root.name + "-context"))
    log = scoped(project, root.with_name(root.name + "-launch.log"))
    receipt = scoped(project, root.with_name(root.name + "-launch.json"))
    for path in (root, context, log, receipt):
        if path.exists() or path.is_symlink():
            raise ValueError("existing or uncertain launch path requires inspection, never automatic replay")
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
    finally:
        probe.close()
    command = [str(python), "-B", "-m", "rollout_manager.terminal_bench.browser_relay.relay_server",
        "--role", args.role, "--root", str(root), "--listen", "127.0.0.1", "--port", str(port),
        "--public-base", args.public_base, "--peer-origin", args.peer_origin.rstrip("/"),
        "--serving-profile", profile, "--upstream", "http://127.0.0.1:8002", "--model", args.model,
        "--max-inflight", "1", "--generation-seconds", "360", "--max-seconds", str(args.max_seconds)]
    if args.proxy_origin:
        command += ["--proxy-origin", args.proxy_origin]
    if args.role == "gpu":
        from rollout_manager.terminal_bench.browser_relay.relay_server import (
            owned_native_controller, owned_training_controller)
        native = profile == PROFILE
        prefix = "native" if native else "training"
        other = "training" if native else "native"
        if any(getattr(args, other + "_controller_" + name, None) for name in ("receipt", "token_file")):
            raise ValueError("Controller inputs must use the selected serving profile's distinct flags")
        startup = scoped(project, getattr(args, prefix + "_controller_receipt", None))
        token = scoped(project, getattr(args, prefix + "_controller_token_file", None))
        if native:
            owned_native_controller(startup, token, args.model)
        else:
            owned_training_controller(startup, token, args.model, serving_profile=profile)
        command += ["--" + prefix + "-controller-receipt", str(startup),
                    "--" + prefix + "-controller-token-file", str(token)]
    return source, context, log, receipt, command


def launch(args, project, root):
    profile = serving_profile(args)
    source, context, log, receipt, command = launch_plan(args, project, root)
    summary = {"action": "launch", "role": args.role, "port": relay_port(args), "state_root": str(root),
        "serving_profile": profile, "generation_slots": 1, "generation_seconds": 360,
        "lifetime_seconds": args.max_seconds, "private_launch_file": str(root / "launch-url.txt"),
        "browser_pair_qualified": False, "executed": args.execute}
    if not args.execute:
        return summary
    os.umask(0o077)
    context.mkdir(mode=0o700)
    for name in ("home", "cache", "tmp"):
        (context / name).mkdir(mode=0o700)
    environment = {key: value for key, value in os.environ.items()
        if not key.startswith(("PYTHON", "PIP_", "HARBOR_"))}
    environment.update(HOME=str(context / "home"), XDG_CACHE_HOME=str(context / "cache"),
        TMPDIR=str(context / "tmp"), TMP=str(context / "tmp"), TEMP=str(context / "tmp"),
        PYTHONPATH=str(source), PYTHONDONTWRITEBYTECODE="1", HARBOR_TELEMETRY="0")
    from rollout_manager.terminal_bench.controller import write_new
    with log.open("xb") as stream:
        child = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
    write_new(receipt, {**summary, "submitted_pid": child.pid,
        "log": str(log), "source_relay_sha256": hashlib.sha256(
            (source / "rollout_manager/terminal_bench/browser_relay/relay_server.py").read_bytes()).hexdigest(),
        "source_page_sha256": hashlib.sha256(
            (source / "rollout_manager/terminal_bench/browser_relay/relay.html").read_bytes()).hexdigest(),
        "submission_is_not_readiness": True})
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if child.poll() is not None:
            raise RuntimeError("relay child ended; inspect preserved log/state/receipt before further action")
        try:
            config = local_api(root, relay_port(args), "/api/config")
        except (FileNotFoundError, ConnectionError, OSError, RuntimeError):
            time.sleep(.1)
            continue
        if (config.get("role") != args.role or config.get("serving_profile") != profile or
                config.get("model") != args.model or config.get("max_inflight") != 1):
            raise RuntimeError("actual relay startup differs from submitted profile; inspect owned state")
        from rollout_manager.terminal_bench.admission import proc_identity
        start, state = proc_identity(child.pid)
        if state in {"Z", "X"}:
            raise RuntimeError("submitted relay process is no longer live")
        verified = {**summary, "relay_pid": child.pid, "relay_pid_start_ticks": start,
            "boot_id": config["boot_id"], "launch_receipt": str(receipt), "log": str(log)}
        write_new(root / "verified-startup.json", verified)
        return verified
    raise TimeoutError("startup waiter expired; preserve and inspect the actual submitted child, never relaunch")


def snapshot_guard(args, project, root):
    profile = serving_profile(args)
    key = "native_controller" if profile == PROFILE else "training_controller"
    from rollout_manager.terminal_bench import admission, controller
    output = scoped(project, args.output)
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise ValueError("guard snapshot requires a new file below an existing project parent")
    lease = private_json(scoped(project, args.lease_file), 1024**2)
    if lease.get("kind") != "terminalbench-policy-lease" or lease.get("served_model") != args.model:
        raise ValueError("CPU guard needs A's exact pre-admission policy lease")
    config = local_api(root, relay_port(args, "cpu"), "/api/config")
    ready = admission.check_readiness(root, args.model, 1, 3060,
        required_generation_seconds=360, serving_profile=profile, expected_policy=lease["policy"])
    if (config.get("role") != "cpu" or config.get("serving_profile") != profile or
            config.get("model") != args.model or config.get("boot_id") != ready["cpu"]["boot_id"]):
        raise ValueError("actual local CPU relay differs from the paired readiness identity")
    evidence = {"schema_version": 2, "serving_profile": profile, "model": args.model,
        key: ready["gpu"][key],
        "cpu_relay_boot_id": ready["cpu"]["boot_id"], "gpu_relay_boot_id": ready["gpu"]["boot_id"],
        "source": "actual authenticated browser readiness; never an operator-supplied proof"}
    controller.guard_gate(evidence, ready, 3060, time.time(), serving_profile=profile,
        model=args.model, expected_policy=lease["policy"])
    if args.execute:
        controller.write_new(output, evidence)
    return {"action": "snapshot-guard", "output": str(output), "executed": args.execute,
        "policy_ack": evidence[key]["policy_ack"], "required_seconds": 3060,
        "actual_browser_roundtrip_validated": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "snapshot-guard"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--expected-hostname", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--role", choices=("cpu", "gpu"))
    parser.add_argument("--port", type=int, help="inspected relay port; defaults to CPU18092 or GPU18093")
    parser.add_argument("--source-root", required=True,
                        help="A checkout root or CPU isolated wheel target; no parent-path assumption")
    parser.add_argument("--python")
    parser.add_argument("--public-base")
    parser.add_argument("--peer-origin")
    parser.add_argument("--proxy-origin")
    parser.add_argument("--max-seconds", type=int, default=30000)
    parser.add_argument("--serving-profile", choices=PROFILES, default=PROFILE)
    parser.add_argument("--native-controller-receipt")
    parser.add_argument("--native-controller-token-file")
    parser.add_argument("--training-controller-receipt")
    parser.add_argument("--training-controller-token-file")
    parser.add_argument("--lease-file")
    parser.add_argument("--output")
    args = parser.parse_args()
    if sys.platform != "linux" or socket.gethostname() != args.expected_hostname:
        parser.error("use the inspected existing Linux machine with its current hostname")
    project = real_path(args.project_root, directory=True)
    root = scoped(project, args.state_root)
    configure_source(args.source_root)
    if args.action == "launch":
        if not all((args.role, args.source_root, args.python, args.public_base, args.peer_origin)):
            parser.error("launch requires role/source-root/python/public-base/peer-origin")
        if not 3660 <= args.max_seconds <= 43200:
            parser.error("relay lifetime must cover the unchanged task window within 43200 seconds")
        if args.role == "gpu":
            prefix = "native" if args.serving_profile == PROFILE else "training"
            if not all(getattr(args, prefix + "_controller_" + key) for key in ("receipt", "token_file")):
                parser.error("GPU launch requires the selected profile actual controller startup receipt and private token file")
        result = launch(args, project, root)
    else:
        if not all((args.lease_file, args.output)):
            parser.error("snapshot-guard requires the real A lease-file and new output")
        result = snapshot_guard(args, project, root)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
