"""Durable bounded artifacts over the authenticated browser relay.

Artifact retries repeat file delivery only, never a model or sandbox request.
Receiving bytes does not validate a training group or prove GPU drainage.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
import time

CHUNK_BYTES = 128 * 1024
MAX_BYTES = 512 * 1024**2
MAX_ARTIFACTS = 1024
MAX_QUEUE_BYTES = 4 * 1024**3
HASH = re.compile(r"[a-f0-9]{64}\Z")
KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}\Z")
KINDS = {"policy_lease", "sealed_group", "public_capsule"}


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def checked_path(path, *, directory=False):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("artifact paths cannot contain symlinks")
    info = path.lstat()
    if info.st_uid != os.getuid() or not (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)):
        raise ValueError("artifact must be a same-owner regular file or directory")
    return path


def pin(path):
    path = checked_path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("artifact changed while reading")
    return {"bytes": before.st_size, "sha256": digest.hexdigest()}


def write_new(path, raw):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    fsync_directory(Path(path).parent)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def read_json(path):
    path = checked_path(path)
    if path.stat().st_size > 1024**2:
        raise ValueError("artifact metadata is too large")
    return json.loads(path.read_bytes())


def validate_manifest(value):
    if (not isinstance(value, dict) or set(value) != {"schema_version", "id", "kind", "key", "bytes", "sha256", "chunk_bytes"}
            or type(value["schema_version"]) is not int or value["schema_version"] != 1
            or not isinstance(value["kind"], str) or value["kind"] not in KINDS
            or not KEY.fullmatch(str(value["key"])) or not HASH.fullmatch(str(value["id"]))
            or not HASH.fullmatch(str(value["sha256"])) or type(value["bytes"]) is not int
            or not 0 < value["bytes"] <= MAX_BYTES or type(value["chunk_bytes"]) is not int or value["chunk_bytes"] != CHUNK_BYTES):
        raise ValueError("invalid bounded artifact manifest")
    body = {k: v for k, v in value.items() if k != "id"}
    if hashlib.sha256(encoded(body)).hexdigest() != value["id"]:
        raise ValueError("artifact identity does not bind metadata and content")
    if value["kind"] in {"policy_lease", "public_capsule"} and value["bytes"] > 1024**2:
        raise ValueError("policy/public capsule exceeds 1 MiB")
    return value


class ArtifactQueue:
    def __init__(self, relay_root, role=None):
        root = checked_path(relay_root, directory=True)
        if root.stat().st_mode & 0o077:
            raise ValueError("artifact relay root must be private")
        self.root, self.role = root / "artifacts", role
        self.root.mkdir(mode=0o700, exist_ok=True)
        for path in (self.root, self.root / "outgoing", self.root / "incoming"):
            path.mkdir(mode=0o700, exist_ok=True)
            checked_path(path, directory=True)
            if path.stat().st_mode & 0o077:
                raise ValueError("artifact directories must be private")

    @contextmanager
    def locked(self):
        fd = os.open(self.root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def entry(self, direction, identity):
        if not HASH.fullmatch(str(identity)):
            raise ValueError("invalid artifact id")
        return self.root / direction / identity

    def bounded(self, extra):
        entries = [p for name in ("incoming", "outgoing") for p in (self.root / name).iterdir()]
        total = 0
        for entry in entries:
            checked_path(entry, directory=True)
            if (entry / "manifest.json").exists():
                total += validate_manifest(read_json(entry / "manifest.json"))["bytes"]
        if len(entries) >= MAX_ARTIFACTS or total + extra > MAX_QUEUE_BYTES:
            raise ValueError("artifact evidence storage bound reached; no deletion or replay")

    def publish(self, kind, key, source):
        source = checked_path(source)
        body = {"schema_version": 1, "kind": kind, "key": key, **pin(source), "chunk_bytes": CHUNK_BYTES}
        manifest = validate_manifest({**body, "id": hashlib.sha256(encoded(body)).hexdigest()})
        with self.locked():
            entries = list((self.root / "outgoing").iterdir())
            for old in entries:
                previous = read_json(old / "manifest.json")
                if (previous["kind"], previous["key"]) == (kind, key):
                    if previous != manifest:
                        raise ValueError("artifact key already binds different immutable content")
                    return manifest
            self.bounded(manifest["bytes"])
            temporary = Path(tempfile.mkdtemp(prefix=".publish-", dir=self.root))
            try:
                target = temporary / "blob"
                with source.open("rb") as stream, os.fdopen(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as destination:
                    shutil.copyfileobj(stream, destination, CHUNK_BYTES)
                    destination.flush()
                    os.fsync(destination.fileno())
                if pin(target) != {k: manifest[k] for k in ("bytes", "sha256")}:
                    raise ValueError("source changed before durable artifact publication")
                write_new(temporary / "manifest.json", encoded(manifest))
                os.rename(temporary, self.entry("outgoing", manifest["id"]))
                fsync_directory(self.root / "outgoing")
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        return manifest

    def next(self):
        with self.locked():
            entries = sorted((self.root / "outgoing").iterdir())
            for entry in entries:
                if not (entry / "delivery.json").exists():
                    return validate_manifest(read_json(entry / "manifest.json"))
        return None

    def status(self):
        with self.locked():
            return {"artifact_pending": sum(not (p / "delivery.json").exists() for p in (self.root / "outgoing").iterdir()),
                    "artifact_incomplete": sum(not (p / "receipt.json").exists() for p in (self.root / "incoming").iterdir())}

    def chunk(self, identity, index):
        with self.locked():
            entry = self.entry("outgoing", identity)
            manifest = validate_manifest(read_json(entry / "manifest.json"))
            if type(index) is not int or not 0 <= index < (manifest["bytes"] + CHUNK_BYTES - 1) // CHUNK_BYTES:
                raise ValueError("invalid artifact chunk index")
            with checked_path(entry / "blob").open("rb") as stream:
                stream.seek(index * CHUNK_BYTES)
                raw = stream.read(CHUNK_BYTES)
            return {"id": identity, "index": index, "data": base64.b64encode(raw).decode(),
                    "sha256": hashlib.sha256(raw).hexdigest()}

    def start(self, manifest):
        manifest = validate_manifest(manifest)
        if self.role == "cpu" and manifest["kind"] != "policy_lease" or self.role == "gpu" and manifest["kind"] == "policy_lease":
            raise ValueError("artifact kind is not admitted for this relay role")
        with self.locked():
            parent = self.root / "incoming"
            for old in parent.iterdir():
                previous = read_json(old / "manifest.json")
                if (previous["kind"], previous["key"]) == (manifest["kind"], manifest["key"]) and previous != manifest:
                    raise ValueError("received artifact key already has different content")
            entry = self.entry("incoming", manifest["id"])
            if not entry.exists():
                self.bounded(manifest["bytes"])
                temporary = Path(tempfile.mkdtemp(prefix=".receive-", dir=self.root))
                try:
                    write_new(temporary / "manifest.json", encoded(manifest))
                    os.rename(temporary, entry)
                    fsync_directory(parent)
                finally:
                    if temporary.exists():
                        shutil.rmtree(temporary)
            elif read_json(entry / "manifest.json") != manifest:
                raise ValueError("artifact manifest conflict")
            index = 0
            while (entry / ("chunk-%05d" % index)).exists():
                index += 1
            return {"id": manifest["id"], "next_index": index, "committed": (entry / "receipt.json").exists()}

    def receive_chunk(self, value):
        if set(value) != {"id", "index", "data", "sha256"} or type(value["index"]) is not int:
            raise ValueError("invalid artifact chunk")
        if not isinstance(value["data"], str) or len(value["data"]) > (CHUNK_BYTES + 2) // 3 * 4:
            raise ValueError("artifact chunk exceeds bound")
        raw = base64.b64decode(value["data"], validate=True)
        if hashlib.sha256(raw).hexdigest() != value["sha256"]:
            raise ValueError("artifact chunk hash mismatch")
        with self.locked():
            entry = self.entry("incoming", value["id"])
            manifest = validate_manifest(read_json(entry / "manifest.json"))
            index = value["index"]
            remaining = manifest["bytes"] - index * CHUNK_BYTES
            if index < 0 or remaining <= 0 or len(raw) != min(CHUNK_BYTES, remaining):
                raise ValueError("artifact chunk length or index differs from manifest")
            if index and not (entry / ("chunk-%05d" % (index - 1))).exists():
                raise ValueError("artifact chunk is out of order")
            path = entry / ("chunk-%05d" % index)
            if path.exists():
                if checked_path(path).read_bytes() != raw:
                    raise ValueError("artifact chunk conflicts with durable bytes")
            else:
                write_new(path, raw)
            return {"id": value["id"], "index": index, "sha256": value["sha256"], "durable": True}

    def commit(self, identity):
        with self.locked():
            entry = self.entry("incoming", identity)
            manifest = validate_manifest(read_json(entry / "manifest.json"))
            receipt = {"schema_version": 1, "id": identity, "bytes": manifest["bytes"], "sha256": manifest["sha256"], "durable": True}
            if (entry / "receipt.json").exists():
                if read_json(entry / "receipt.json") != receipt:
                    raise ValueError("artifact commit conflicts")
                return receipt
            temporary = entry / "blob.partial"
            if temporary.exists():
                temporary.unlink()  # only our incomplete reconstruction; durable chunks remain
            with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
                for index in range((manifest["bytes"] + CHUNK_BYTES - 1) // CHUNK_BYTES):
                    stream.write(checked_path(entry / ("chunk-%05d" % index)).read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            if pin(temporary) != {k: manifest[k] for k in ("bytes", "sha256")}:
                raise ValueError("artifact final hash mismatch")
            os.replace(temporary, entry / "blob")
            write_new(entry / "receipt.json", encoded(receipt))
            return receipt

    def acknowledge(self, receipt):
        with self.locked():
            entry = self.entry("outgoing", receipt.get("id"))
            manifest = validate_manifest(read_json(entry / "manifest.json"))
            expected = {"schema_version": 1, "id": manifest["id"], "bytes": manifest["bytes"], "sha256": manifest["sha256"], "durable": True}
            if receipt != expected:
                raise ValueError("artifact delivery acknowledgement is not content-bound")
            path = entry / "delivery.json"
            if not path.exists():
                write_new(path, encoded(receipt))
            elif read_json(path) != receipt:
                raise ValueError("artifact delivery acknowledgement changed")
            return receipt

    def received(self, kind, key):
        if kind not in KINDS or not KEY.fullmatch(str(key)):
            raise ValueError("invalid artifact lookup")
        with self.locked():
            for entry in (self.root / "incoming").iterdir():
                manifest = validate_manifest(read_json(entry / "manifest.json"))
                if (manifest["kind"], manifest["key"]) == (kind, key) and (entry / "receipt.json").exists():
                    if pin(entry / "blob") != {k: manifest[k] for k in ("bytes", "sha256")}:
                        raise ValueError("committed artifact bytes changed")
                    return manifest, entry / "blob"
        return None


def publish_file(relay_root, kind, key, path):
    return ArtifactQueue(relay_root).publish(kind, key, path)


def artifact_api(queue, route, value):
    if route == "/api/artifact-next" and not value:
        return queue.next()
    if route == "/api/artifact-read" and set(value) == {"id", "index"}:
        return queue.chunk(value["id"], value["index"])
    if route == "/api/artifact-start":
        return queue.start(value)
    if route == "/api/artifact-chunk":
        return queue.receive_chunk(value)
    if route == "/api/artifact-commit" and set(value) == {"id"}:
        return queue.commit(value["id"])
    if route == "/api/artifact-ack":
        return queue.acknowledge(value)
    raise ValueError("invalid artifact operation")


def wait_received(relay_root, kind, key, timeout):
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or not 0 < timeout <= 86400:
        raise ValueError("artifact wait must be bounded within one day")
    queue, deadline = ArtifactQueue(relay_root), time.monotonic() + timeout
    while True:
        value = queue.received(kind, key)
        if value is not None:
            return value
        if time.monotonic() >= deadline:
            raise TimeoutError("automatic artifact did not arrive; no workload replay")
        time.sleep(min(.2, max(0, deadline - time.monotonic())))
