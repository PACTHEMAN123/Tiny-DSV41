"""Optional short IPC paths for the owned LoRA actor and its spawned workers."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
import stat
import tempfile

# Actor-lifetime references: workers preserve this live owner alias after fork
# or spawn. A repeated helper call never opens another descriptor for that path.
_COMPILER_DIRECTORIES = {}


def lora_ipc_directory():
    """Validate an explicitly selected private directory beneath the owned cwd.

    Return its relative spelling: an absolute task path can exceed AF_UNIX's
    pathname limit even when the final directory name is short.
    """
    value = os.environ.get("DSV41_LORA_IPC_DIR")
    if value is None:
        return None
    path = Path(value)
    if not value or path.is_absolute() or ".." in path.parts or path == Path("."):
        raise ValueError("DSV41_LORA_IPC_DIR must be a nonempty relative owned directory")
    root = Path.cwd().resolve()
    resolved = path.resolve(strict=True)
    if path.is_symlink() or root not in resolved.parents:
        raise ValueError("DSV41_LORA_IPC_DIR must remain inside the owned cwd without a symlink")
    info = resolved.stat()
    if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
            or info.st_mode & 0o022 or not os.access(resolved, os.W_OK | os.X_OK)):
        raise ValueError("DSV41_LORA_IPC_DIR must be an existing private UID-owned writable directory")
    relative = str(resolved.relative_to(root))
    if len(os.fsencode(Path(relative) / "rl-vllm-xxxxxxxx" / "actor.sock")) >= 104:
        raise ValueError("DSV41_LORA_IPC_DIR is too long for a portable Unix socket")
    return relative


def _native_socket_fits(directory):
    # Torch v2.13 POSIX symmetric-memory IPC formats TMPDIR/symm_mem-<pid>.
    # Linux sun_path holds 108 bytes; reserve 20 PID digits and a terminating NUL.
    return len(os.fsencode(str(directory) + "/symm_mem-" + "9" * 20)) + 1 <= 108


def _compiler_directory(directory):
    value = os.environ.get("TMPDIR", directory)
    path = Path(value)
    resolved = path.resolve(strict=True)
    root = Path.cwd().resolve()
    info = resolved.stat()
    if (root not in resolved.parents or not stat.S_ISDIR(info.st_mode)
            or info.st_uid != os.getuid() or info.st_mode & 0o077
            or not os.access(resolved, os.W_OK | os.X_OK)):
        raise ValueError("LoRA compiler TMPDIR must resolve to a private UID-owned directory beneath the owned cwd")
    # Reuse an externally retained or inherited owner alias; opening another FD
    # in every spawned worker would create unnecessary descriptor lifetimes.
    if re.fullmatch(r"/proc/[1-9][0-9]*/fd/[0-9]+", value) and _native_socket_fits(value):
        return value
    if _native_socket_fits(resolved):
        return str(resolved)
    if sys.platform != "linux":
        raise ValueError("long LoRA compiler TMPDIR requires Linux proc directory aliases")
    key = str(resolved)
    retained = _COMPILER_DIRECTORIES.get(key)
    opened = retained is None
    if opened:
        fd = os.open(resolved, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        actual = os.fstat(fd)
        if (actual.st_dev, actual.st_ino) != (info.st_dev, info.st_ino):
            os.close(fd)
            raise ValueError("LoRA compiler directory changed while opening its retained descriptor")
        retained = (os.getpid(), fd)
        _COMPILER_DIRECTORIES[key] = retained
    alias = f"/proc/{retained[0]}/fd/{retained[1]}"
    try:
        valid = _native_socket_fits(alias) and Path(alias).resolve(strict=True) == resolved
    except OSError:
        valid = False
    if not valid:
        if opened:
            os.close(_COMPILER_DIRECTORIES.pop(key)[1])
        raise ValueError("retained LoRA compiler alias is unavailable or too long")
    return alias


def configure_lora_ipc():
    """Configure Python IPC and native/compiler paths only for explicit LoRA.

    Python temporary IPC keeps a short relative name. Native Torch IPC needs a
    short TMPDIR while nvcc needs an absolute one despite its changing cwd.
    Long compiler directories use a live actor-owned Linux proc FD alias.
    """
    directory = lora_ipc_directory()
    if directory is not None:
        os.environ["TMPDIR"] = _compiler_directory(directory)
        tempfile.tempdir = directory
        os.environ["VLLM_RPC_BASE_PATH"] = directory
        # Multiprocessing spawn reimports the actor as __mp_main__ before its
        # vLLM target runs. This marker is emitted only by the LoRA path.
        os.environ["_DSV41_LORA_IPC_ACTIVE"] = "1"
    return directory


def create_lora_socket_directory():
    """Return a fresh actor socket directory and the optional validated base."""
    directory = lora_ipc_directory()
    path = Path(tempfile.mkdtemp(prefix="rl-vllm-", dir=directory))
    if directory is not None and path.is_absolute():
        # Python 3.12+ returns an absolute mkdtemp result even for relative dir.
        path = path.relative_to(Path.cwd())
    return path, directory
