"""Private child environment for Harbor execution; no services are started here."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile

MARKER = "ROLLOUT_WORKER_ROOT"
SCHEMA_VERSION = 1


def _layout(runtime, configuration_sha256):
    if not isinstance(configuration_sha256, str) or re.fullmatch(r"[a-f0-9]{64}", configuration_sha256) is None:
        raise ValueError("Worker isolation requires the exact configuration SHA256")
    executor = Path(runtime["executor_root"])
    output = Path(runtime["output_dir"])
    # The backend output is nested below the not-yet-created unified run. Keep
    # isolation beside that run so EventStore can still create it exclusively.
    run_output = output.parent
    root = Path(runtime.get("worker_root", str(run_output.with_name(run_output.name + "-worker"))))
    for path in (executor, output, root):
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("Worker paths must be explicit absolute paths")
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("Linked worker execution path")
    if executor not in root.parents or executor not in output.parents:
        raise ValueError("Worker and output must remain below the explicit CPU executor root")
    paths = {"root": root, "home": root / "home", "cache": root / "cache", "tmp": root / "tmp"}
    binding = {"schema_version": SCHEMA_VERSION, "configuration_sha256": configuration_sha256,
               "executor_root": str(executor), "output_dir": str(output), "worker_root": str(root),
               "owner_uid": os.getuid()}
    return paths, binding


def _private_directory(path):
    info = path.lstat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or
            stat.S_IMODE(info.st_mode) & 0o077):
        raise ValueError("Worker directory must be private and owned by this CPU controller")


def prepare_worker_environment(runtime, configuration_sha256):
    """Create a fresh isolation tree and return child-only env; never mutate HOME."""
    paths, binding = _layout(runtime, configuration_sha256)
    if not paths["root"].parent.is_dir():
        raise ValueError("Worker isolation needs an existing CPU output parent")
    # No resume or fallback path: an uncertain earlier launch must be inspected.
    paths["root"].mkdir(mode=0o700)
    for name in ("home", "cache", "tmp"):
        paths[name].mkdir(mode=0o700)
    receipt = paths["root"] / "worker-environment.json"
    fd = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump(binding, stream, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("PYTHON", "PIP_", "HARBOR_")) and key != MARKER}
    env.update({MARKER: str(paths["root"]), "HOME": str(paths["home"]),
                "XDG_CACHE_HOME": str(paths["cache"]), "TMPDIR": str(paths["tmp"]),
                "TMP": str(paths["tmp"]), "TEMP": str(paths["tmp"]),
                "HARBOR_TELEMETRY": "0", "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def validate_worker_environment(runtime, configuration_sha256):
    """Read-only guard before importing Harbor or admitting a child/container."""
    paths, expected = _layout(runtime, configuration_sha256)
    required = {MARKER: str(paths["root"]), "HOME": str(paths["home"]),
                "XDG_CACHE_HOME": str(paths["cache"]), "TMPDIR": str(paths["tmp"]),
                "TMP": str(paths["tmp"]), "TEMP": str(paths["tmp"]),
                "HARBOR_TELEMETRY": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    if any(os.environ.get(key) != value for key, value in required.items()):
        raise ValueError("Harbor execution requires the prepared private child environment")
    allowed = {"HARBOR_TELEMETRY", "PYTHONDONTWRITEBYTECODE"}
    if any(key.startswith(("PYTHON", "PIP_", "HARBOR_")) and key not in allowed for key in os.environ):
        raise ValueError("Inherited Python/PIP/Harbor configuration was not filtered")
    if not sys.dont_write_bytecode:
        raise ValueError("The Harbor child must start with -B or disabled bytecode")
    if tempfile.tempdir is not None and Path(tempfile.tempdir) != paths["tmp"]:
        raise ValueError("Previously cached temporary directory is outside the Harbor child")
    for path in paths.values():
        _private_directory(path)
    receipt = paths["root"] / "worker-environment.json"
    fd = os.open(receipt, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid() or
                stat.S_IMODE(before.st_mode) & 0o077 or not 0 < before.st_size <= 65536):
            raise ValueError("Invalid private worker binding receipt")
        raw = stream.read(65537)
        after = os.fstat(stream.fileno())
    signature = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
    if len(raw) != before.st_size or signature(before) != signature(after):
        raise ValueError("Worker binding receipt changed while reading")
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("Duplicate worker binding key")
            value[key] = item
        return value
    if json.loads(raw, object_pairs_hook=unique) != expected:
        raise ValueError("Worker receipt is not bound to this configuration and output")
