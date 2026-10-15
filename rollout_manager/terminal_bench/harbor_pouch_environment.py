"""Harbor environment plugin for already-staged Pouch images.

Import path: harbor_pouch_environment:PouchEnvironment. No image pulling,
building, verifier substitution, command retry, or host bind mounts occur here.
Requires the real Harbor package (the audited checkout is 0.23.0, Python 3.12+).
"""

import asyncio
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import struct
import tarfile
from urllib.parse import quote, urlencode
import uuid

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import (
    EnvironmentCapabilities, EnvironmentResourceCapabilities,
)
from harbor.models.task.config import NetworkMode
from harbor.models.trial.config import ResourceMode


OWNER_LABEL = "io.harbor.pouch.owner"
OBJECT_ID = re.compile(r"[a-f0-9]{64}\Z")
IMAGE_ID = re.compile(r"sha256:[a-f0-9]{64}\Z")
MAX_JSON = 2 * 1024**2
MAX_EXEC = 16 * 1024**2
EXEC_GRACE_SEC = 5


class PouchError(RuntimeError):
    pass


class PouchHTTPError(PouchError):
    def __init__(self, status):
        self.status = status
        # Daemon error bodies can contain task content or environment secrets.
        super().__init__(f"Pouch API returned HTTP {status}")


def absolute_path(value):
    value = str(value)
    if not value.startswith("/") or "\0" in value:
        raise ValueError("Container path must be absolute and contain no NUL")
    return value


def seconds(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be finite and positive")
    return float(value)


def image_mapping(mapping, path):
    if mapping is not None and path is not None:
        raise ValueError("Supply image_id_by_reference or image_id_map_path, not both")
    if path is not None:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 1024**2:
                raise ValueError("Image map must be a bounded regular JSON file")
            raw = handle.read(1024**2 + 1)
            if len(raw) != info.st_size:
                raise ValueError("Image map changed while reading")
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("Duplicate image reference")
                result[key] = value
            return result
        mapping = json.loads(raw, object_pairs_hook=unique)
    if mapping is None:
        return None
    if not isinstance(mapping, dict) or not 0 < len(mapping) <= 10000:
        raise ValueError("Image map must be a nonempty bounded object")
    for reference, image_id in mapping.items():
        if not isinstance(reference, str) or not 0 < len(reference) <= 1024 or "\0" in reference or (
                not isinstance(image_id, str) or not IMAGE_ID.fullmatch(image_id)):
            raise ValueError("Image mapping requires original references and full sha256 configuration IDs")
    return dict(mapping)


def demux(raw):
    streams = {1: bytearray(), 2: bytearray()}
    offset = 0
    while offset < len(raw):
        if len(raw) - offset < 8:
            raise PouchError("Truncated exec frame header")
        stream, padding, size = raw[offset], raw[offset + 1:offset + 4], struct.unpack(">I", raw[offset + 4:offset + 8])[0]
        offset += 8
        if stream not in streams or padding != b"\0\0\0" or size > len(raw) - offset:
            raise PouchError("Malformed exec frame")
        streams[stream].extend(raw[offset:offset + size])
        offset += size
    return tuple(streams[k].decode("utf-8", errors="replace") for k in (1, 2))


class PouchAPI:
    """Cancellation-aware, one-request-per-connection Unix HTTP transport."""

    def __init__(self, socket_path):
        self.socket_path = str(socket_path)

    async def request(self, method, path, payload=None, *, timeout=30, max_bytes=MAX_JSON,
                      content_type="application/json"):
        async def exchange():
            reader, writer = await asyncio.open_unix_connection(self.socket_path)
            try:
                body = payload if isinstance(payload, bytes) else (
                    json.dumps(payload, allow_nan=False).encode() if payload is not None else b"")
                if any(char in path for char in "\r\n "):
                    raise ValueError("Invalid API path")
                header = (f"{method} /v1.24{path} HTTP/1.1\r\nHost: localhost\r\n"
                          f"Connection: close\r\nContent-Type: {content_type}\r\n"
                          f"Content-Length: {len(body)}\r\n\r\n").encode("ascii")
                writer.write(header)
                writer.write(body)
                await writer.drain()
                status_line = await reader.readline()
                fields = status_line.split()
                if len(fields) < 2 or fields[0] not in (b"HTTP/1.0", b"HTTP/1.1"):
                    raise PouchError("Malformed HTTP status")
                status = int(fields[1])
                headers, header_bytes = {}, len(status_line)
                while True:
                    line = await reader.readline()
                    header_bytes += len(line)
                    if not line or header_bytes > 65536:
                        raise PouchError("Missing or oversized HTTP headers")
                    if line == b"\r\n":
                        break
                    key, separator, value = line.partition(b":")
                    if not separator:
                        raise PouchError("Malformed HTTP header")
                    key = key.strip().lower()
                    if key in headers and key in (b"content-length", b"transfer-encoding"):
                        raise PouchError("Ambiguous HTTP framing")
                    headers[key] = value.strip().lower()
                data = bytearray()
                async def take(size):
                    if size < 0 or len(data) + size > max_bytes:
                        raise PouchError("Response exceeds configured byte bound")
                    data.extend(await reader.readexactly(size))
                transfer = headers.get(b"transfer-encoding")
                if transfer is not None:
                    if transfer != b"chunked" or b"content-length" in headers:
                        raise PouchError("Unsupported HTTP framing")
                    while True:
                        line = await reader.readline()
                        if len(line) > 1024 or not line.endswith(b"\r\n"):
                            raise PouchError("Malformed chunk header")
                        size = int(line.split(b";", 1)[0], 16)
                        if size == 0:
                            trailer_bytes = 0
                            while True:
                                line = await reader.readline()
                                trailer_bytes += len(line)
                                if not line or trailer_bytes > 65536:
                                    raise PouchError("Malformed HTTP trailers")
                                if line == b"\r\n":
                                    break
                            break
                        await take(size)
                        if await reader.readexactly(2) != b"\r\n":
                            raise PouchError("Malformed chunk terminator")
                elif b"content-length" in headers:
                    await take(int(headers[b"content-length"]))
                elif status not in (204, 304):
                    # Exec can use a hijacked or close-delimited HTTP response.
                    while True:
                        chunk = await reader.read(min(65536, max_bytes + 1 - len(data)))
                        if not chunk:
                            break
                        data.extend(chunk)
                        if len(data) > max_bytes:
                            raise PouchError("Response exceeds configured byte bound")
                if status != 101 and not 200 <= status < 300:
                    raise PouchHTTPError(status)
                return bytes(data)
            finally:
                writer.close()
                # Closing does not wait for a peer that keeps a hijack open.
                try:
                    await asyncio.wait_for(writer.wait_closed(), 1)
                except (OSError, asyncio.TimeoutError):
                    pass
        return await asyncio.wait_for(exchange(), seconds(timeout, "request timeout"))

    async def json(self, method, path, payload=None, *, timeout=30):
        raw = await self.request(method, path, payload, timeout=timeout)
        result = json.loads(raw) if raw else {}
        if not isinstance(result, dict):
            raise PouchError("Expected JSON object from Pouch")
        return result


class PouchEnvironment(BaseEnvironment):
    def __init__(self, *args, socket_path="/run/pouchd.sock", runtime=None,
                 api_timeout_sec=30, transfer_timeout_sec=120,
                 max_transfer_bytes=256 * 1024**2, allow_public_network=False,
                 image_id_by_reference=None, image_id_map_path=None, **kwargs):
        self.api_timeout_sec = seconds(api_timeout_sec, "API timeout")
        self.transfer_timeout_sec = seconds(transfer_timeout_sec, "transfer timeout")
        if type(max_transfer_bytes) is not int or not 1024 <= max_transfer_bytes <= 1024**3:
            raise ValueError("Transfer byte bound must be between 1 KiB and 1 GiB")
        self.max_transfer_bytes = max_transfer_bytes
        self.api = PouchAPI(socket_path)
        self.runtime = runtime
        if type(allow_public_network) is not bool:
            raise ValueError("allow_public_network must be an explicit boolean")
        self.allow_public_network = allow_public_network
        self.image_id_by_reference = image_mapping(image_id_by_reference, image_id_map_path)
        self.owner = uuid.uuid4().hex
        self.container_name = "harbor-pouch-" + self.owner
        self.container_id = None
        self._create_attempted = False
        self._creation_task = None
        self._creation_uncertain = False
        self._broken = False
        self._cleanup_lock = asyncio.Lock()
        self._workdir = "/"
        self._receipt = {"provider": "pouch", "owner": self.owner,
                         "storage_enforced": False, "image_pull_performed": False}
        super().__init__(*args, **kwargs)

    @staticmethod
    def type():
        return "pouch"

    @property
    def capabilities(self):
        return EnvironmentCapabilities(disable_internet=True, mounted=False)

    @classmethod
    def resource_capabilities(cls):
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    def _validate_definition(self):
        if not self.task_env_config.docker_image:
            raise ValueError("Pouch requires an already-staged docker_image")
        if any((self.environment_dir / name).exists() for name in
               ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml")):
            raise ValueError("This single-container Pouch plugin does not implement Compose")
        if self.network_policy.network_mode == NetworkMode.PUBLIC and not self.allow_public_network:
            raise ValueError("Public networking requires explicit allow_public_network=True")
        if self.image_id_by_reference is not None and self.task_env_config.docker_image not in self.image_id_by_reference:
            raise ValueError("Original task image is absent from the supplied pinned image map")

    def _record(self, **values):
        self._receipt.update(values)
        directory = self.trial_paths.trial_dir
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "pouch-environment.json"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "w") as handle:
            json.dump(self._receipt, handle, indent=2, allow_nan=False)
            handle.write("\n")

    async def _owned(self, missing_ok=False):
        reference = self.container_id or self.container_name
        try:
            info = await self.api.json("GET", "/containers/" + reference + "/json",
                                       timeout=self.api_timeout_sec)
        except PouchHTTPError as error:
            if missing_ok and error.status == 404:
                return None
            raise
        found = info.get("Id")
        if not isinstance(found, str) or not OBJECT_ID.fullmatch(found) or (
                self.container_id is not None and found != self.container_id) or (
                info.get("Config", {}).get("Labels", {}).get(OWNER_LABEL) != self.owner):
            raise PouchError("Container ownership mismatch; refusing operation")
        self.container_id = found
        return info

    async def start(self, force_build=False):
        if force_build:
            raise ValueError("Pouch plugin never builds or pulls images; stage the exact image first")
        if self._create_attempted:
            raise PouchError("Container creation is single-attempt")
        try:
            reference = self.task_env_config.docker_image
            pinned_id = self.image_id_by_reference.get(reference) if self.image_id_by_reference is not None else None
            image = await self.api.json("GET", "/images/" + quote(pinned_id or reference, safe="") + "/json",
                                        timeout=self.api_timeout_sec)
            image_id = image.get("Id")
            if not isinstance(image_id, str) or not IMAGE_ID.fullmatch(image_id):
                raise PouchError("Local image did not resolve to an immutable ID")
            if pinned_id is not None and image_id != pinned_id:
                raise PouchError("Local image identity differs from the supplied pinned configuration ID")
            if image.get("Os", "linux").lower() != "linux":
                raise PouchError("Pouch plugin requires a Linux image")
            self._workdir = absolute_path(self.task_env_config.workdir or
                                         image.get("Config", {}).get("WorkingDir") or "/")
            cpu = self._resource_limit_value("cpu", auto_mode=ResourceMode.LIMIT)
            memory = self._resource_limit_value("memory", auto_mode=ResourceMode.LIMIT)
            for name, value in (("CPU", cpu), ("memory", memory)):
                if value is not None and (type(value) is not int or value <= 0):
                    raise ValueError(name + " limit must be a positive integer")
            host = {"NetworkMode": "none" if self.network_policy.network_mode == NetworkMode.NO_NETWORK else "bridge",
                    "Privileged": False}
            if cpu is not None:
                host.update(CpuPeriod=100000, CpuQuota=int(cpu) * 100000)
            if memory is not None:
                host.update(Memory=int(memory) * 1024**2)
            if self.runtime is not None:
                host["Runtime"] = self.runtime
            payload = {"Image": image_id, "Cmd": ["sh", "-c", "sleep infinity"],
                       "WorkingDir": self._workdir, "Labels": {OWNER_LABEL: self.owner},
                       "HostConfig": host}
            startup_env = self._startup_env()
            if startup_env:
                payload["Env"] = [key + "=" + value for key, value in startup_env.items()]
            self._record(image_id=image_id, original_image_reference=reference,
                         image_resolution="pinned-config-map" if pinned_id else "local-reference",
                         workdir=self._workdir, cpu_limit=cpu,
                         memory_limit_mb=memory, storage_requested_mb=self._effective_storage_mb,
                         network_mode=host["NetworkMode"], container_name=self.container_name,
                         public_network_explicitly_allowed=self.allow_public_network)
            self._create_attempted = True
            self._creation_task = asyncio.create_task(self.api.json(
                "POST", "/containers/create?" + urlencode({"name": self.container_name}),
                payload, timeout=self.api_timeout_sec))
            # Do not abandon an in-flight create on Harbor cancellation: wait
            # for its bounded response before resolving the owned name.
            created = await asyncio.shield(self._creation_task)
            container_id = created.get("Id")
            if not isinstance(container_id, str) or not OBJECT_ID.fullmatch(container_id):
                raise PouchError("Container create did not return a full ID")
            self.container_id = container_id
            info = await self._owned()
            observed = info.get("HostConfig", {})
            for key, value in host.items():
                if observed.get(key) != value:
                    raise PouchError("Runtime changed requested container setting: " + key)
            await self.api.request("POST", "/containers/" + self.container_id + "/start", timeout=self.api_timeout_sec)
            info = await self._owned()
            if info.get("State", {}).get("Running") is not True:
                raise PouchError("Owned container did not reach running state")
            self._record(container_id=self.container_id, started=True,
                         create_warning_count=len(created.get("Warnings") or []))
            await self.ensure_dirs(self._mount_targets(writable_only=True))
            await self._upload_environment_dir_after_start()
        except BaseException:
            self._broken = True
            if self._create_attempted:
                await self._cleanup_after_failure()
            raise

    async def _cleanup_after_failure(self):
        try:
            await asyncio.wait_for(self.stop(delete=True), 6 * self.api_timeout_sec + 5)
        except BaseException as error:
            try:
                self._record(cleanup_verified=False, cleanup_error_type=type(error).__name__)
            except OSError:
                pass

    async def stop(self, delete=True):
        async with self._cleanup_lock:
            if not self._create_attempted:
                return
            if self._creation_task is not None:
                try:
                    await asyncio.shield(self._creation_task)
                except Exception:
                    self._creation_uncertain = True
            info = await self._owned(missing_ok=True)
            if info is None:
                if self._creation_uncertain:
                    self._record(cleanup_verified=False, create_outcome_unknown=True)
                    raise PouchError("Create outcome unknown; absence now cannot prove no late creation")
                self._record(cleanup_verified=True)
                return
            if info.get("State", {}).get("Running") is True:
                await self.api.request("POST", "/containers/" + self.container_id + "/stop?t=1",
                                       timeout=self.api_timeout_sec)
                info = await self._owned()
                if info.get("State", {}).get("Running") is not False:
                    raise PouchError("Owned container is still running after stop")
            if delete:
                await self.api.request("DELETE", "/containers/" + self.container_id + "?force=false&v=true",
                                       timeout=self.api_timeout_sec)
                if await self._owned(missing_ok=True) is not None:
                    raise PouchError("Owned container is still present after deletion")
            self._record(stopped=True, deleted=delete, cleanup_verified=delete)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if self._broken:
            raise PouchError("Environment has an uncertain operation; no further commands admitted")
        if not isinstance(command, str) or "\0" in command:
            raise ValueError("Command must be a string without NUL")
        budget = seconds(timeout_sec, "exec timeout") if timeout_sec is not None else None
        argv = ["bash", "-c", command]
        if budget is not None:
            argv = ["timeout", "--signal=TERM", "--kill-after=2s", str(budget) + "s"] + argv
        # Pouch 1.24 does not reliably implement the exec WorkingDir field.
        argv = ["sh", "-c", 'cd "$1" && shift && exec "$@"', "harbor",
                absolute_path(cwd or self._workdir)] + argv
        payload = {"AttachStdin": False, "AttachStdout": True, "AttachStderr": True,
                   "Tty": False, "Privileged": False, "Cmd": argv}
        merged_env = self._merge_env(env)
        if merged_env:
            payload["Env"] = [key + "=" + value for key, value in merged_env.items()]
        resolved_user = self._resolve_user(user)
        if resolved_user is not None:
            payload["User"] = str(resolved_user)
        terminal_proven = False
        completion = None
        try:
            await self._owned()
            created = await self.api.json("POST", "/containers/" + self.container_id + "/exec", payload,
                                         timeout=self.api_timeout_sec)
            exec_id = created.get("Id")
            if not isinstance(exec_id, str) or not OBJECT_ID.fullmatch(exec_id):
                raise PouchError("Exec creation did not return a full ID")
            # Harbor's outer agent/verifier deadline remains authoritative when
            # no per-command timeout is supplied (notably stock Verifier.exec).
            transport_budget = budget + EXEC_GRACE_SEC if budget is not None else 24 * 3600
            loop = asyncio.get_running_loop()
            deadline = loop.time() + transport_budget
            async def collect():
                raw = await self.api.request("POST", "/exec/" + exec_id + "/start",
                                             {"Detach": False, "Tty": False}, timeout=transport_budget, max_bytes=MAX_EXEC)
                stdout, stderr = demux(raw)
                status = await self.api.json("GET", "/exec/" + exec_id + "/json", timeout=self.api_timeout_sec)
                if status.get("Running") is not False or type(status.get("ExitCode")) is not int:
                    raise PouchError("Exec completion is uncertain; command will not be replayed")
                return ExecResult(stdout=stdout, stderr=stderr, return_code=status["ExitCode"])
            completion = asyncio.create_task(asyncio.wait_for(collect(), transport_budget))
            try:
                result = await asyncio.shield(completion)
                terminal_proven = True
            except asyncio.CancelledError:
                drain_started = loop.time()
                # Harbor grades after an agent timeout. Preserve saved state only
                # when this already-started, bounded command yields terminal proof.
                # There is no fresh command admission or reset of its deadline.
                try:
                    if budget is not None and deadline > loop.time():
                        await asyncio.wait_for(asyncio.shield(completion), deadline - loop.time())
                        terminal_proven = True
                except (Exception, asyncio.CancelledError):
                    pass
                elapsed = loop.time() - drain_started
                self._record(cancellation_drain_seconds=elapsed, deadline_overrun_seconds=elapsed,
                             deadline_overrun_reference="elapsed after caller cancellation; source deadline unavailable",
                             cancellation_terminal_proven=terminal_proven,
                             strict_wallclock_parity=False)
                raise
            stdout, stderr = result.stdout, result.stderr
            callback = self._output_callback()
            if callback is not None:
                if stdout:
                    await callback(stdout, "stdout")
                if stderr:
                    await callback(stderr, "stderr")
            return result
        except BaseException as error:
            if completion is not None and not completion.done():
                completion.cancel()
                try:
                    await completion
                except (Exception, asyncio.CancelledError):
                    pass
            if isinstance(error, asyncio.CancelledError) and terminal_proven:
                raise
            self._broken = True
            await self._cleanup_after_failure()
            if type(error) is asyncio.TimeoutError:
                raise PouchError("Exec provider operation timed out; command will not be replayed") from error
            raise

    def _archive(self, source, arcname):
        buffer = io.BytesIO()
        total, count = 0, 0
        def check(member):
            nonlocal total, count
            total += member.size
            count += 1
            if count > 100000 or total > self.max_transfer_bytes:
                raise PouchError("Upload exceeds configured transfer bound")
            return member
        with tarfile.open(fileobj=buffer, mode="w", dereference=False) as archive:
            archive.add(source, arcname=arcname, filter=check)
        data = buffer.getvalue()
        if len(data) > self.max_transfer_bytes:
            raise PouchError("Upload tar exceeds configured transfer bound")
        return data

    async def _upload(self, source, target, directory):
        target = absolute_path(target)
        source = Path(source)
        if directory and not source.is_dir() or not directory and not source.is_file():
            raise ValueError("Upload source has the wrong kind")
        parent = target if directory else str(PurePosixPath(target).parent)
        result = await self.exec("mkdir -p -- " + shlex.quote(parent), user="root", timeout_sec=30)
        if result.return_code:
            raise PouchError("Cannot create upload destination")
        data = self._archive(source, "." if directory else PurePosixPath(target).name)
        await self._owned()
        await self.api.request("PUT", "/containers/" + self.container_id + "/archive?" + urlencode({"path": parent}),
                               data, timeout=self.transfer_timeout_sec, content_type="application/x-tar")

    async def upload_file(self, source_path, target_path):
        await self._upload(source_path, target_path, False)

    async def upload_dir(self, source_dir, target_dir):
        await self._upload(source_dir, target_dir, True)

    async def _download(self, source, target, directory):
        source = absolute_path(source)
        await self._owned()
        data = await self.api.request("GET", "/containers/" + self.container_id + "/archive?" + urlencode({"path": source}),
                                      timeout=self.transfer_timeout_sec, max_bytes=self.max_transfer_bytes)
        target = Path(target)
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as archive:
            members, total = [], 0
            for member in archive:
                total += member.size
                if len(members) >= 100000 or total > self.max_transfer_bytes:
                    raise PouchError("Expanded download exceeds configured transfer bound")
                if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                    raise PouchError("Download contains a special file")
                members.append(member)
            if not directory:
                files = [member for member in members if member.isfile()]
                if len(files) != 1 or PurePosixPath(files[0].name).name != PurePosixPath(source).name:
                    raise PouchError("File download must contain exactly the requested regular file")
                target.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(archive.extractfile(files[0]).read())
                return
            target.mkdir(parents=True, exist_ok=True)
            root = PurePosixPath(source.rstrip("/")).name
            seen = set()
            for member in members:
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != root:
                    raise PouchError("Unexpected archive root or unsafe path")
                relative = PurePosixPath(*path.parts[1:])
                if str(relative) in seen:
                    raise PouchError("Duplicate download archive path")
                seen.add(str(relative))
                member.name = str(relative)
                if member.islnk():
                    link = PurePosixPath(member.linkname)
                    if not link.parts or link.parts[0] != root:
                        raise PouchError("Unexpected hardlink archive root")
                    member.linkname = str(PurePosixPath(*link.parts[1:]))
            # Python's data filter rejects links escaping target and special files.
            archive.extractall(target, members=members, filter="data")

    async def download_file(self, source_path, target_path):
        await self._download(source_path, target_path, False)

    async def download_dir(self, source_dir, target_dir):
        await self._download(source_dir, target_dir, True)
