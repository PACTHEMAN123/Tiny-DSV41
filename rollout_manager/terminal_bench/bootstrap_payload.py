"""Assemble and validate generic offline dependencies, never task test sources."""

import argparse
from email.parser import BytesParser
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
import zipfile


TASKS = ("fix-git", "large-scale-text-editing", "openssl-selfsigned-cert", "sqlite-db-truncate")
PROFILE = "tb2-original-verifier-offline-four-v1"
TARGET = "/opt/harbor-bootstrap"
MAX_PAYLOAD = 64 * 1024**2
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ENV = {"UV_OFFLINE": "1", "UV_FIND_LINKS": TARGET + "/wheels",
       "UV_PYTHON_DOWNLOADS": "never", "UV_PYTHON_PREFERENCE": "only-system",
       "UV_CACHE_DIR": "/root/.cache/harbor-uv"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def read(path, maximum=MAX_PAYLOAD, *, allow_empty=False):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or not (0 if allow_empty else 1) <= info.st_size <= maximum:
            raise ValueError("Expected bounded regular payload file")
        data = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    if len(data) != info.st_size or (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError("Payload file changed while reading")
    return data


def wheel_closure(directory):
    # Parse distribution metadata, including target-specific environment markers.
    from packaging.markers import default_environment
    from packaging.requirements import Requirement
    from packaging.specifiers import SpecifierSet
    from packaging.utils import canonicalize_name
    import taskspec_runner
    environment = default_environment()
    environment.update(python_version="3.13", python_full_version="3.13.7",
                       sys_platform="linux", platform_system="Linux", os_name="posix",
                       platform_machine="x86_64", platform_python_implementation="CPython",
                       implementation_name="cpython", implementation_version="3.13.7", extra="")
    distributions = {}
    for name, expected in taskspec_runner.WHEELS.items():
        data = read(directory / name, 4 * 1024**2)
        if digest(data) != expected:
            raise ValueError("Pinned pytest wheel differs")
        with zipfile.ZipFile(directory / name) as archive:
            matches = [item for item in archive.infolist() if item.filename.endswith(".dist-info/METADATA")]
            if len(matches) != 1 or matches[0].file_size > 1024**2:
                raise ValueError("Unexpected wheel metadata")
            metadata = BytesParser().parsebytes(archive.read(matches[0]))
        if "3.13.7" not in SpecifierSet(metadata.get("Requires-Python", "")):
            raise ValueError("Wheel excludes the pinned Python version")
        distributions[canonicalize_name(metadata["Name"])] = metadata
    for metadata in distributions.values():
        for raw in metadata.get_all("Requires-Dist", []):
            dependency = Requirement(raw)
            if dependency.marker and not dependency.marker.evaluate(environment):
                continue
            found = distributions.get(canonicalize_name(dependency.name))
            if found is None or found["Version"] not in dependency.specifier:
                raise ValueError("Incomplete Python 3.13/Linux wheel closure")
    for name, version in (("pytest", "8.4.1"), ("pytest-json-ctrf", "0.3.5")):
        if distributions[name]["Version"] != version:
            raise ValueError("Verifier root package version differs")
    return {name: data["Version"] for name, data in sorted(distributions.items())}


def verify(directory, manifest_sha256, *, profile=PROFILE, tasks=TASKS):
    directory = Path(directory)
    if not directory.is_absolute() or directory.is_symlink() or not directory.is_dir():
        raise ValueError("Payload must be an absolute real directory")
    raw = read(directory / "manifest.json", 1024**2)
    if not isinstance(manifest_sha256, str) or not SHA256.fullmatch(manifest_sha256) or digest(raw) != manifest_sha256:
        raise ValueError("Pinned bootstrap manifest differs")
    manifest = json.loads(raw, object_pairs_hook=strict_object)
    if manifest.get("profile") != profile or manifest.get("environment") != ENV:
        raise ValueError("Unexpected bootstrap profile or environment")
    if set(manifest.get("tasks", {})) != set(tasks) or not 1 <= len(manifest.get("files", [])) <= 64:
        raise ValueError("Unexpected task or file count")
    seen, total = {"manifest.json"}, 0
    for row in manifest["files"]:
        name = row["path"]
        relative = PurePosixPath(name)
        if (not isinstance(name, str) or relative.is_absolute() or str(relative) != name or
                not relative.parts or any(part in (".", "..") for part in relative.parts) or name in seen):
            raise ValueError("Unsafe or duplicate payload path")
        if any((directory / Path(*relative.parts[:index])).is_symlink() for index in range(1, len(relative.parts))):
            raise ValueError("Linked payload parent")
        data = read(directory / name)
        total += len(data)
        if type(row["bytes"]) is not int or len(data) != row["bytes"] or digest(data) != row["sha256"] or total > MAX_PAYLOAD:
            raise ValueError("Payload hash, length or total differs")
        seen.add(name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if not path.is_dir()}
    if actual != seen or any(path.is_symlink() for path in directory.rglob("*")):
        raise ValueError("Unexpected or linked payload member")
    if total != manifest["payload_bytes"]:
        raise ValueError("Payload total differs")
    return manifest
