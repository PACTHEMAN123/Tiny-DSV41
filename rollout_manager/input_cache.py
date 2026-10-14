"""Bounded CPU-storage cache. Only descriptors and JSON cross the Unix socket.

The owner of a harness process registers its consumer and retires it only after
the process group is reaped. A disconnected child cannot silently free credits.
"""
from __future__ import annotations

import array
import ctypes
import fcntl
import hashlib
import json
import os
import platform
import resource
import socket
import socketserver
import struct
import threading
import time
import uuid
from collections import Counter, OrderedDict
from pathlib import Path


MAX_MESSAGE = 4 * 1024 * 1024


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def cache_key(value):
    required = {"task_sha256", "generator_sha256", "configuration", "namespace", "seed",
                "trial_index", "input_contract", "adapter_version", "runtime", "rng_before"}
    if set(value) != required or value["namespace"] not in ("development", "final-verification"):
        raise ValueError("incomplete cache identity or invalid verification namespace")
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def memory_file(size):
    if hasattr(os, "memfd_create"):
        fd = os.memfd_create("rollout-input-cache", flags=3)
    elif platform.system() == "Linux" and platform.machine() == "x86_64":
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.syscall(319, b"rollout-input-cache", 3)
        if fd < 0:
            raise OSError(ctypes.get_errno(), "memfd_create")
    else:
        raise OSError("sealed RAM input reuse requires Linux memfd")
    try:
        os.ftruncate(fd, max(1, size))
        return fd
    except BaseException:
        os.close(fd)
        raise


def receive(sock):
    header, ancillary, flags, _ = sock.recvmsg(4, socket.CMSG_SPACE(array.array("i").itemsize))
    fds = array.array("i")
    for level, kind, data in ancillary:
        if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
            fds.frombytes(data[:len(data) - len(data) % fds.itemsize])
    try:
        while len(header) < 4:
            block = sock.recv(4-len(header))
            if not block:
                raise EOFError("cache protocol closed")
            header += block
        size, = struct.unpack("!I", header)
        if flags & socket.MSG_CTRUNC or size > MAX_MESSAGE or len(fds) > 1:
            raise ValueError("invalid cache message")
        body = bytearray()
        while len(body) < size:
            block = sock.recv(size-len(body))
            if not block:
                raise EOFError("cache message truncated")
            body.extend(block)
        result = json.loads(body)
        for fd in fds:
            os.set_inheritable(fd, False)
        return result, (fds[0] if fds else None)
    except BaseException:
        for fd in fds:
            os.close(fd)
        raise


def send(sock, value, fd=None):
    body = canonical(value).encode()
    if len(body) > MAX_MESSAGE:
        raise ValueError("cache metadata is too large")
    packet = struct.pack("!I", len(body)) + body
    ancillary = [] if fd is None else [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))]
    sent = sock.sendmsg([packet], ancillary)
    if sent < len(packet):
        sock.sendall(packet[sent:])


class CacheClient:
    def __init__(self, path, timeout=250):
        self.path, self.timeout = str(path), timeout

    def call(self, op, **args):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.timeout)
            sock.connect(self.path)
            send(sock, {"op": op, **args})
            value, fd = receive(sock)
        if "error" in value:
            if fd is not None:
                os.close(fd)
            raise RuntimeError("input cache: " + value["error"])
        return value, fd

    def control(self, op, **args):
        value, fd = self.call(op, **args)
        if fd is not None:
            os.close(fd)
            raise RuntimeError("unexpected cache descriptor")
        return value


class InputCache:
    def __init__(self, root, *, ram_bytes, disk_bytes=0, event=None, max_entries=4096):
        if min(ram_bytes, disk_bytes) < 0 or ram_bytes + disk_bytes <= 0:
            raise ValueError("cache capacity must be positive")
        if not isinstance(max_entries, int) or max_entries < 1:
            raise ValueError("positive cache entry limit required")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.capacity = {"ram": int(ram_bytes), "disk": int(disk_bytes)}
        self.entries = OrderedDict()
        self.consumers = {}
        self.cv = threading.Condition(threading.RLock())
        self.metrics = Counter()
        self.event = event
        soft_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        self.max_entries = min(max_entries, max(1, soft_limit-128)) if soft_limit > 0 else max_entries
        self.socket_path = self.root / "cache.sock"
        # sockaddr_un is short; output roots can be longer than this.
        if len(os.fsencode(self.socket_path)) >= 104:
            raise ValueError("use a short cache root for the Unix socket")
        owner = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                fd = None
                try:
                    self.request.settimeout(260)
                    request, received = receive(self.request)
                    if received is not None:
                        os.close(received)
                        raise ValueError("clients may not send cache descriptors")
                    result, fd = owner.dispatch(request)
                    send(self.request, result, fd)
                except Exception as exc:
                    try:
                        send(self.request, {"error": str(exc)})
                    except OSError:
                        pass
                finally:
                    if fd is not None:
                        os.close(fd)

        class Server(socketserver.ThreadingUnixStreamServer):
            daemon_threads = True
            block_on_close = True

        self.server = Server(str(self.socket_path), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def _record(self, event, **details):
        if self.event:
            self.event({"event": event, "unix_ns": time.time_ns(), **details})

    def _used(self, tier):
        return sum(e["bytes"] for e in self.entries.values() if e["tier"] == tier)

    def _remove(self, key, reason):
        item = self.entries.pop(key)
        os.close(item["fd"])
        if item.get("path"):
            Path(item["path"]).unlink()
        if reason == "evict":
            self.metrics["evictions"] += 1
            self.metrics["evicted_bytes"] += item["bytes"]
        self._record(reason, key=key, bytes=item["bytes"], tier=item["tier"])

    def _space(self, tier, size):
        if size > self.capacity[tier]:
            return False
        for key, item in list(self.entries.items()):
            if self._used(tier)+size <= self.capacity[tier] and len(self.entries) < self.max_entries:
                return True
            if item["tier"] == tier and item["state"] == "ready" and not item["leases"]:
                self._remove(key, "evict")
        return self._used(tier)+size <= self.capacity[tier] and len(self.entries) < self.max_entries

    def begin(self, copy_limit):
        if not isinstance(copy_limit, int) or copy_limit < 0:
            raise ValueError("nonnegative consumer copy budget required")
        token = uuid.uuid4().hex
        self.consumers[token] = {"copy_limit": int(copy_limit), "leases": {}}
        return {"consumer": token}

    def release(self, consumer, lease):
        record = self.consumers[consumer]
        key = record["leases"].pop(lease, None)
        if key in self.entries:
            self.entries[key]["leases"].discard(lease)
        self.cv.notify_all()

    def end(self, consumer):
        if consumer not in self.consumers:
            return {"released": False}
        for lease in list(self.consumers[consumer]["leases"]):
            self.release(consumer, lease)
        for key, item in list(self.entries.items()):
            if item["state"] == "generating" and item["producer"] == consumer:
                self._remove(key, "aborted")
        del self.consumers[consumer]
        self.cv.notify_all()
        return {"released": True}

    def acquire(self, consumer, identity, size, wait_seconds=240):
        key = cache_key(identity)
        if not isinstance(size, int) or size < 0:
            raise ValueError("invalid cache storage reservation")
        start = time.monotonic()
        while key in self.entries and self.entries[key]["state"] == "generating":
            if consumer not in self.consumers:
                raise RuntimeError("consumer cancelled")
            left = min(wait_seconds, 240) - (time.monotonic()-start)
            if left <= 0:
                self.metrics["bypasses"] += 1
                return {"status": "bypass", "reason": "generation_wait_timeout"}, None
            self.cv.wait(min(.2, left))
        record = self.consumers[consumer]
        if key in self.entries:
            item = self.entries[key]
            private = sum(self.entries[k]["bytes"] for k in record["leases"].values())
            if private + item["bytes"] > record["copy_limit"]:
                self.metrics["bypasses"] += 1
                return {"status": "bypass", "reason": "consumer_copy_budget"}, None
            lease = uuid.uuid4().hex
            record["leases"][lease] = key
            item["leases"].add(lease)
            self.entries.move_to_end(key)
            self.metrics["hits"] += 1
            self.metrics["generation_bytes_saved"] += item["bytes"]
            self.metrics["estimated_generation_seconds_saved"] += item["metadata"]["generation_seconds"]
            self._record("hit", key=key, consumer=consumer, lease=lease, bytes=item["bytes"])
            return {"status": "hit", "key": key, "lease": lease, "metadata": item["metadata"],
                    "bytes": item["bytes"], "tier": item["tier"],
                    "wait_seconds": time.monotonic()-start}, os.dup(item["fd"])
        self.metrics["misses"] += 1
        tier = next((t for t in ("ram", "disk") if self.capacity[t] and self._space(t, size)), None)
        if tier is None:
            self.metrics["bypasses"] += 1
            return {"status": "bypass", "reason": "capacity_or_pinned_entries"}, None
        path = None
        if tier == "ram":
            fd = memory_file(size)
        else:
            path = str(self.root / (key + ".storage"))
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                if hasattr(os, "posix_fallocate"):
                    os.posix_fallocate(fd, 0, max(1, size))
                else:
                    os.ftruncate(fd, max(1, size))
            except BaseException:
                os.close(fd)
                Path(path).unlink()
                raise
        self.entries[key] = {"fd": fd, "path": path, "bytes": size, "tier": tier,
                             "state": "generating", "producer": consumer, "leases": set()}
        self._record("miss", key=key, consumer=consumer, bytes=size, tier=tier)
        return {"status": "miss", "key": key, "bytes": size, "tier": tier,
                "wait_seconds": time.monotonic()-start}, os.dup(fd)

    def publish(self, consumer, key, metadata, size):
        item = self.entries[key]
        if item["state"] != "generating" or item["producer"] != consumer or not 0 <= size <= item["bytes"]:
            raise ValueError("invalid cache publication")
        metadata_bytes = len(canonical(metadata).encode())
        if metadata_bytes > 128*1024:
            raise ValueError("cache metadata exceeds its per-entry bound")
        os.ftruncate(item["fd"], max(1, size))
        if item["tier"] == "ram":
            fcntl.fcntl(item["fd"], 1033, 15)  # F_ADD_SEALS: seal, shrink, grow, write.
            read_fd = os.open("/proc/self/fd/" + str(item["fd"]), os.O_RDONLY | os.O_CLOEXEC)
        else:
            os.fsync(item["fd"])
            os.fchmod(item["fd"], 0o400)
            read_fd = os.open(item["path"], os.O_RDONLY | os.O_CLOEXEC)
        os.close(item["fd"])
        item.update(fd=read_fd, bytes=size, metadata=metadata, metadata_bytes=metadata_bytes, state="ready")
        self.metrics["publications"] += 1
        self.metrics["publication_copy_bytes"] += metadata["publication_copy_bytes"]
        self.metrics["publication_copy_seconds"] += metadata["publication_copy_seconds"]
        self._record("publish", key=key, bytes=size, tier=item["tier"])
        self.cv.notify_all()
        return {"published": True}

    def snapshot(self):
        with self.cv:
            return {"capacity_bytes": dict(self.capacity), "shared_storage_bytes": {
                tier: self._used(tier) for tier in self.capacity},
                "generating_bytes": sum(e["bytes"] for e in self.entries.values() if e["state"] == "generating"),
                "entries": len(self.entries), "consumers": len(self.consumers),
                "max_entries": self.max_entries,
                "metadata_bytes": sum(e.get("metadata_bytes", 0) for e in self.entries.values()),
                "metadata_bound_bytes": self.max_entries*128*1024,
                "references": sum(len(c["leases"]) for c in self.consumers.values()),
                "consumer_cow_reserved_bytes": sum(self.entries[k]["bytes"] for c in self.consumers.values()
                                                   for k in c["leases"].values()),
                "explicit_pinned_bytes": 0, "handoff_copy_bytes": 0,
                "owner_pid": os.getpid(), "owner_cpu_seconds": time.process_time(), "metrics": dict(self.metrics)}

    def dispatch(self, request):
        request = dict(request)
        op = request.pop("op")
        with self.cv:
            if op == "snapshot":
                return self.snapshot(), None
            if op == "begin":
                return self.begin(**request), None
            consumer = request.get("consumer")
            if consumer not in self.consumers:
                raise ValueError("unknown or retired cache consumer")
            if op == "acquire":
                return self.acquire(**request)
            if op == "publish":
                return self.publish(**request), None
            if op == "release":
                self.release(**request)
                return {"released": True}, None
            if op == "end":
                return self.end(**request), None
            if op == "abort":
                item = self.entries.get(request["key"])
                if item and item["state"] == "generating" and item["producer"] == consumer:
                    self._remove(request["key"], "aborted")
                    self.cv.notify_all()
                return {"aborted": True}, None
            raise ValueError("unknown cache operation")

    def clear(self):
        with self.cv:
            if self.consumers:
                raise RuntimeError("cannot clear a cache with live consumers")
            for key in list(self.entries):
                self._remove(key, "clear")

    def close(self):
        self.clear()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.socket_path.unlink(missing_ok=True)
