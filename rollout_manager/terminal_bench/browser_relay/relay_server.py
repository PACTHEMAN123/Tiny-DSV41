#!/usr/bin/env python3
"""Bounded, same-origin transport for a laptop-mediated CPU/GPU pilot."""
import argparse
import fcntl
import hashlib
import hmac
import http.client
import json
import math
import os
import re
import secrets
import select
import signal
import socket
import stat
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MAX_BODY = 512 * 1024
MAX_RESPONSE = 2 * 1024 * 1024
MAX_INFLIGHT = 64
MAX_MODEL_SEQUENCES = 1024
MAX_TRACKED = 10000
MAX_GENERATION_SECONDS = 600
PULSE_INTERVAL = 5.0
PULSE_TTL = 15.0
CLAIM_RECEIPT_TTL = 30.0
ID = re.compile(r"^[0-9a-f]{32}$")


class CapacityError(Exception):
    pass


class ConflictError(ValueError):
    pass


class GoneError(ValueError):
    pass


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def acquire_runtime_lock(root):
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(root / ".relay.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077:
            raise ValueError("Runtime lock must be a private regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def encoded(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def open_upstream_connection(timeout, port=8000):
    return http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)


def private_controller_file(path, maximum=65536):
    path = Path(path)
    if not path.is_absolute() or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Training controller input must be an absolute unlinked private file")
    from rollout_manager.terminal_bench.evidence_io import read_json
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Training controller input must be a same-owner private regular file")
    return read_json(path, maximum)


def native_controller_token(path):
    path = Path(path)
    if not path.is_absolute() or any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError("Training controller token path must be absolute and unlinked")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077 or
                not 0 < info.st_size <= 128):
            raise ValueError("Training controller token must be a same-owner private bounded file")
        token = stream.read(129).decode().strip()
    if not re.fullmatch(r"[0-9a-f]{64}", token):
        raise ValueError("Invalid training controller token")
    return token


def owned_training_controller(receipt_path, token_path, model, *, serving_profile, proc_root=Path("/proc")):
    """Authenticate current loaded policy, then independently prove the local owner/socket."""
    from rollout_manager.terminal_bench.evidence_io import unique
    from rollout_manager.terminal_bench.serving_contracts import (
        training_controller_identity, validate_training_controller)
    receipt = private_controller_file(receipt_path)
    token = native_controller_token(token_path)
    connection = open_upstream_connection(5, 8002)
    try:
        connection.request("GET", "/rollout/profile", headers={"Authorization": "Bearer " + token})
        response = connection.getresponse()
        raw = response.read(65537)
        if response.status != 200 or len(raw) > 65536 or response.length not in (0, None):
            raise ValueError("Training controller proof unavailable or exceeds bound")
        proof = json.loads(raw, object_pairs_hook=unique)
    finally:
        connection.close()
    validate_training_controller(proof, model, serving_profile=serving_profile, require_running=False)
    if training_controller_identity(receipt, include_policy=False) != training_controller_identity(proof, include_policy=False):
        raise ValueError("Live training controller differs from private startup identity")
    root = proc_root / str(proof["model_pid"])
    if root.stat().st_uid != os.getuid():
        raise ValueError("Training controller is not owned by the relay user")
    net = os.readlink(root / "ns" / "net")
    if (net != os.readlink(proc_root / "self" / "ns" / "net") or
            net != "net:[%d]" % proof["netns_inode"]):
        raise ValueError("Training controller network namespace differs")
    def identity():
        fields = read_proc_bytes(root / "stat", 8192).decode().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X"):
            raise ValueError("Training controller process is not live")
        return int(fields[19])
    if identity() != proof["model_pid_start_ticks"]:
        raise ValueError("Training controller process start identity changed")
    command = read_proc_bytes(root / "cmdline", 65536)
    if (hashlib.sha256(command).hexdigest() != proof["model_cmdline_sha256"] or
            not command.endswith(b"\0") or command.split(b"\0")[0].decode() != proof["python_executable"]):
        raise ValueError("Training controller Python command identity differs")
    listeners = set()
    for line in read_proc_bytes(root / "net" / "tcp", 1024 * 1024).decode().splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 10 and fields[1] == "0100007F:1F42" and fields[3] == "0A":
            listeners.add(fields[9])
    owned = set()
    with os.scandir(root / "fd") as entries:
        for count, entry in enumerate(entries):
            if count >= 4096:
                raise ValueError("Training controller descriptor inventory exceeds bound")
            try:
                target = os.readlink(entry.path)
            except FileNotFoundError:
                continue
            match = re.fullmatch(r"socket:\[([0-9]+)\]", target)
            if match:
                owned.add(match[1])
    if not listeners.intersection(owned) or identity() != proof["model_pid_start_ticks"]:
        raise ValueError("Training controller does not own the verified loopback8002 listener")
    return proof


def owned_native_controller(receipt_path, token_path, model, proc_root=Path("/proc")):
    """Preserve the native route and its distinct backend identity."""
    return owned_training_controller(receipt_path, token_path, model,
        serving_profile="native-pytorch-v1", proc_root=proc_root)

def process_start_ticks(pid):
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except FileNotFoundError:
        return None


def read_proc_bytes(path, maximum):
    with open(path, "rb") as handle:
        value = handle.read(maximum + 1)
    if len(value) > maximum:
        raise ValueError("Owned model process accounting exceeds read bound")
    return value


def owned_model_capacity(pid, model, proc_root=Path("/proc")):
    """Bind configured capacity to a live local listener, not an operator cap alone."""
    if type(pid) is not int or pid <= 0:
        raise ValueError("A positive owned model PID is required")
    root = proc_root / str(pid)
    model_net = os.readlink(root / "ns" / "net")
    if not re.fullmatch(r"net:\[[0-9]+\]", model_net) or model_net != os.readlink(proc_root / "self" / "ns" / "net"):
        raise ValueError("Owned model is not in the relay network namespace")

    def identity():
        fields = read_proc_bytes(root / "stat", 8192).decode().rsplit(")", 1)[1].split()
        if fields[0] in ("Z", "X"):
            raise ValueError("Owned model process is not live")
        return int(fields[19])

    start = identity()
    raw = read_proc_bytes(root / "cmdline", 65536)
    if not raw.endswith(b"\0"):
        raise ValueError("Incomplete owned model command line")
    argv = raw[:-1].decode().split("\0")
    if (len(argv) < 4 or Path(argv[1]).name != "vllm" or argv[2] != "serve"
            or not Path(argv[0]).name.startswith("python") or not Path(argv[3]).is_absolute()
            or Path(argv[0]).parent != Path(argv[1]).parent):
        raise ValueError("Owned model must use the qualified vllm serve command")

    def option(name):
        locations = [index for index, item in enumerate(argv) if item == name or item.startswith(name + "=")]
        if len(locations) != 1:
            raise ValueError("Missing or ambiguous owned model option")
        index = locations[0]
        if argv[index] != name or index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            raise ValueError("Owned model requires explicit separate option values")
        return argv[index + 1]

    if option("--served-model-name") != model or option("--host") != "127.0.0.1" or option("--port") != "8000":
        raise ValueError("Owned model identity or listener differs from fixed upstream")
    maximum = option("--max-num-seqs")
    if not re.fullmatch(r"[1-9][0-9]{0,3}", maximum) or not 1 <= int(maximum) <= MAX_MODEL_SEQUENCES:
        raise ValueError("Owned model max-num-seqs must be between 1 and 1024")
    listeners = set()
    for line in read_proc_bytes(root / "net" / "tcp", 1024 * 1024).decode().splitlines()[1:]:
        fields = line.split()
        if len(fields) >= 10 and fields[1] == "0100007F:1F40" and fields[3] == "0A":
            listeners.add(fields[9])
    owned = set()
    with os.scandir(root / "fd") as entries:
        for count, entry in enumerate(entries):
            if count >= 4096:
                raise ValueError("Owned model descriptor count exceeds read bound")
            try:
                target = os.readlink(entry.path)
            except FileNotFoundError:
                continue
            match = re.fullmatch(r"socket:\[([0-9]+)\]", target)
            if match:
                owned.add(match[1])
    if not listeners.intersection(owned) or identity() != start:
        raise ValueError("Owned model process does not own the fixed listening socket")
    return {"model_max_num_seqs": int(maximum), "model_capacity_source": "owned_vllm_process",
            "model_pid": pid, "model_pid_start_ticks": start,
            "model_cmdline_sha256": hashlib.sha256(raw).hexdigest()}


def origin(url):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("An explicit HTTP(S) origin is required")
    return f"{parsed.scheme}://{parsed.netloc}"


def atomic_write(path, value):
    data = encoded(value)
    temporary = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def read_json(path, limit=MAX_BODY):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Invalid or oversized spool file")
    return json.loads(path.read_bytes())


def check_job(job, model, allow_expired=False):
    if not isinstance(job, dict) or set(job) != {"id", "operation", "payload", "deadline_unix"}:
        raise ValueError("Invalid job fields")
    if not isinstance(job["id"], str) or not ID.fullmatch(job["id"]):
        raise ValueError("Invalid job ID")
    deadline = job["deadline_unix"]
    if (isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline)
            or deadline <= 0 or deadline > time.time() + 3600 or (not allow_expired and deadline <= time.time())):
        raise ValueError("Expired or excessive job deadline")
    if job["operation"] == "echo":
        if job["payload"] != {"chunks": 3}:
            raise ValueError("Invalid qualification payload")
    elif job["operation"] == "generate":
        check_generation(job["payload"], model)
    else:
        raise ValueError("Unsupported operation")


def check_generation(payload, model):
    allowed = {"model", "messages", "max_tokens", "temperature", "top_p", "stop", "stream", "seed", "tools", "tool_choice", "parallel_tool_calls", "chat_template_kwargs"}
    if not isinstance(payload, dict) or set(payload) - allowed:
        raise ValueError("Unsupported generation fields")
    if payload.get("model") != model or payload.get("stream", False) is not False:
        raise ValueError("The fixed model and buffered generation are required")
    if payload.get("parallel_tool_calls", False) is not False:
        raise ValueError("Parallel tool calls are disabled for this pilot")
    if "chat_template_kwargs" in payload and payload["chat_template_kwargs"] != {"enable_thinking": False}:
        raise ValueError("Only disabling thinking is supported by the pilot")
    maximum = payload.get("max_tokens")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or not 1 <= maximum <= 2048:
        raise ValueError("max_tokens must be between 1 and 2048")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not 1 <= len(messages) <= 256:
        raise ValueError("Invalid messages")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError("Invalid message role")
        if set(message) - {"role", "content", "name", "tool_calls", "tool_call_id", "reasoning_content"}:
            raise ValueError("Unsupported message field")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ValueError("Only text messages are supported")
        if "reasoning_content" in message and (message["role"] != "assistant" or not isinstance(message["reasoning_content"], str)):
            raise ValueError("Reasoning content must be an assistant string")
    if len(encoded(payload)) > MAX_BODY:
        raise ValueError("Generation payload too large")


class RelayState:
    def __init__(self, root, role, public_base, peer_origin, upstream, model, proxy_origin=None, max_inflight=1, max_seconds=1800, generation_seconds=120, model_pid=None,
                 serving_profile="vllm-v1", native_controller_receipt=None, native_controller_token_file=None,
                 training_controller_receipt=None, training_controller_token_file=None):
        if type(max_inflight) is not int or not 1 <= max_inflight <= MAX_INFLIGHT:
            raise ValueError("max-inflight must be between 1 and 64")
        if type(max_seconds) is not int or not 1 <= max_seconds <= 43200:
            raise ValueError("max-seconds must be between 1 and 43200")
        if type(generation_seconds) is not int or not 1 <= generation_seconds <= MAX_GENERATION_SECONDS:
            raise ValueError("generation-seconds must be an integer between 1 and 600")
        self.root = Path(root).resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self.root.stat().st_mode & 0o077:
            raise ValueError("Relay state root must have mode 0700")
        self.role, self.model = role, model
        if serving_profile not in {"vllm-v1", "native-pytorch-v1", "vllm-training-v1"}:
            raise ValueError("Unsupported explicit serving profile")
        self.serving_profile = serving_profile
        native = serving_profile == "native-pytorch-v1"
        vllm_training = serving_profile == "vllm-training-v1"
        controlled = native or vllm_training
        if controlled and max_inflight != 1:
            raise ValueError("Training controller preserves actual serial capacity1")
        if native and (training_controller_receipt or training_controller_token_file):
            raise ValueError("Native profile requires its original native controller inputs")
        if not native and (native_controller_receipt or native_controller_token_file):
            raise ValueError("Native controller inputs require the explicit native profile")
        if not vllm_training and (training_controller_receipt or training_controller_token_file):
            raise ValueError("Training controller inputs require the explicit vLLM training profile")
        self.native_controller_receipt = native_controller_receipt
        self.native_controller_token_file = native_controller_token_file
        self.training_controller_receipt = training_controller_receipt
        self.training_controller_token_file = training_controller_token_file
        self.controller_receipt = native_controller_receipt if native else training_controller_receipt
        self.controller_token_file = native_controller_token_file if native else training_controller_token_file
        self.controller_proof_key = "native_controller" if native else "training_controller"
        if controlled and role == "gpu" and (not self.controller_receipt or not self.controller_token_file):
            raise ValueError("Training GPU relay requires private startup identity and authentication token")
        if model_pid is not None and role != "gpu":
            raise ValueError("Owned model PID is GPU-only")
        self.model_pid = model_pid
        self.model_identity = owned_model_capacity(model_pid, model) if model_pid is not None and not controlled else None
        self.native_identity = None
        self.training_identity = None
        if controlled and role == "gpu":
            from rollout_manager.terminal_bench.serving_contracts import training_controller_identity
            proof = (owned_native_controller(self.controller_receipt, self.controller_token_file, model)
                if native else owned_training_controller(self.controller_receipt, self.controller_token_file,
                    model, serving_profile=serving_profile))
            if model_pid is not None and model_pid != proof["model_pid"]:
                raise ValueError("Configured training controller PID differs from live owner")
            identity = training_controller_identity(proof, include_policy=False)
            if native:
                self.native_identity = identity
            else:
                self.training_identity = identity
        self.public_base = public_base.rstrip("/") + "/"
        self.origin = origin(public_base)
        self.peer_origin = origin(peer_origin)
        if peer_origin.rstrip("/") != self.peer_origin:
            raise ValueError("peer-origin must not contain a path")
        self.proxy_origin = origin(proxy_origin) if proxy_origin is not None else None
        if proxy_origin is not None and proxy_origin != self.proxy_origin:
            raise ValueError("proxy-origin must be an exact HTTP(S) origin without a trailing slash")
        parsed = urllib.parse.urlsplit(self.public_base)
        if parsed.query or parsed.fragment:
            raise ValueError("public-base must not contain query or fragment")
        self.prefix = parsed.path.rstrip("/")
        self.upstream_port = 8002 if controlled else 8000
        if upstream != "http://127.0.0.1:%d" % self.upstream_port:
            raise ValueError("Only the profile's verified fixed loopback upstream is permitted")
        self.upstream = upstream + "/v1/chat/completions"
        self.max_inflight = max_inflight
        self.generation_seconds = generation_seconds
        self.boot_id = uuid.uuid4().hex
        self.lock = threading.RLock()
        self.trace_lock = threading.Lock()
        self.active_generations = {}
        self.completions = {}
        self.completion_changed = threading.Condition(self.lock)
        self.completion_batch_gate = threading.BoundedSemaphore(1)
        self.gpu_journal = {}
        self.claim_receipts = {}
        from .artifacts import ArtifactQueue
        self.artifacts = ArtifactQueue(self.root, role)
        self.created = time.monotonic()
        self.started_unix = time.time()
        self.expires_unix = self.started_unix + max_seconds
        self.deadline = self.created + max_seconds
        self.pid_start_ticks = process_start_ticks(os.getpid())
        self.stop_event = threading.Event()
        self.pulse_gate = threading.BoundedSemaphore(1)
        self.claim_poll_gate = threading.BoundedSemaphore(1)
        self.claim_receipt_gate = threading.BoundedSemaphore(1)
        self.next_pulse_due = self.created
        self.pair_challenge = None
        for name in ("inbox", "inflight", "outbox", "requests"):
            directory = self.root / name
            directory.mkdir(mode=0o700, exist_ok=True)
            if directory.is_symlink() or directory.stat().st_mode & 0o077:
                raise ValueError("Spool directories must be private and not symlinks")
        token_path = self.root / "api-token.txt"
        if token_path.exists():
            if token_path.is_symlink() or token_path.stat().st_mode & 0o077:
                raise ValueError("Existing token file is not private")
            self.token = token_path.read_text().strip()
            if not re.fullmatch(r"[0-9a-f]{64}", self.token):
                raise ValueError("Invalid existing token")
        else:
            self.token = secrets.token_hex(32)
            with os.fdopen(os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as handle:
                handle.write(self.token + "\n")
        launch = self.root / "launch-url.txt"
        with os.fdopen(os.open(launch, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as handle:
            handle.write(self.public_base + "relay.html#token=" + self.token + "\n")
        os.chmod(launch, 0o600)
        self.recover()
        if role == "cpu":
            atomic_write(self.root / "readiness.json", {"schema_version": 1, "protocol": 2, "valid_until_unix": 0})
        self.trace("server_start", role=role, max_inflight=max_inflight, boot_id=self.boot_id)

    def configuration(self):
        capacity_fields = {}
        controller_identity = self.native_identity or self.training_identity
        if controller_identity is not None:
            from rollout_manager.terminal_bench.serving_contracts import training_controller_identity
            proof = (owned_native_controller(self.controller_receipt, self.controller_token_file, self.model)
                if self.serving_profile == "native-pytorch-v1" else owned_training_controller(
                    self.controller_receipt, self.controller_token_file, self.model,
                    serving_profile=self.serving_profile))
            if training_controller_identity(proof, include_policy=False) != controller_identity:
                raise ValueError("Training controller process changed; requalification required")
            capacity_fields = {self.controller_proof_key: proof}
        if self.model_identity is not None:
            capacity_fields = owned_model_capacity(self.model_pid, self.model)
            if capacity_fields != self.model_identity:
                raise ValueError("Owned model process identity changed; requalification required")
        with self.lock:
            return {"role": self.role, "peer_origin": self.peer_origin, "public_base": self.public_base,
                    "model": self.model, "protocol": 2, "max_inflight": self.max_inflight, "boot_id": self.boot_id,
                    "generation_transport": "submit-poll-v1",
                    "completion_transport": "shared-poll-v1",
                    "generation_seconds": self.generation_seconds,
                    "uncertain": sum(record["status"] == "uncertain" for record in self.gpu_journal.values()),
                    "readiness_transport": "server-challenge-v1",
                    "claim_transport": "receipt-v1",
                    "artifact_transport": "durable-chunks-v1",
                    "lifetime_remaining_ms": max(0, (self.deadline - time.monotonic()) * 1000),
                    "serving_profile": self.serving_profile, **capacity_fields}

    def pair_receipt(self, value):
        expected = {"protocol", "boot_id", "model", "max_inflight", "lifetime_remaining_ms", "uncertain", "challenge_id"}
        capacity_keys = {"model_max_num_seqs", "model_capacity_source", "model_pid", "model_pid_start_ticks", "model_cmdline_sha256"}
        if set(value) - {"generation_seconds", "serving_profile", "native_controller", "training_controller"} - capacity_keys != expected or value["protocol"] != 2 or value["model"] != self.model:
            raise ValueError("Invalid paired configuration")
        if value.get("serving_profile", "vllm-v1") != self.serving_profile:
            raise ValueError("Paired serving profile differs")
        controller_fields = {}
        if self.serving_profile in {"native-pytorch-v1", "vllm-training-v1"}:
            from rollout_manager.terminal_bench.serving_contracts import validate_training_controller
            key = "native_controller" if self.serving_profile == "native-pytorch-v1" else "training_controller"
            other = "training_controller" if key == "native_controller" else "native_controller"
            proof = validate_training_controller(value.get(key), self.model,
                serving_profile=self.serving_profile, require_running=False)
            if value["max_inflight"] != 1 or any(name in value for name in capacity_keys) or other in value:
                raise ValueError("Training proof cannot claim legacy vLLM CLI capacity or a different controller profile")
            controller_fields = {key: proof}
        elif {"native_controller", "training_controller"} & set(value):
            raise ValueError("Unexpected training proof under the legacy serving profile")
        model_proof = {key: value[key] for key in capacity_keys if key in value}
        if model_proof:
            if (set(model_proof) != capacity_keys or model_proof["model_capacity_source"] != "owned_vllm_process"
                    or type(model_proof["model_max_num_seqs"]) is not int or not 1 <= model_proof["model_max_num_seqs"] <= MAX_MODEL_SEQUENCES
                    or any(type(model_proof[key]) is not int or model_proof[key] <= 0 for key in ("model_pid", "model_pid_start_ticks"))
                    or not isinstance(model_proof["model_cmdline_sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", model_proof["model_cmdline_sha256"])):
                raise ValueError("Invalid owned model capacity proof")
        generation_seconds = value.get("generation_seconds", 120)
        if type(generation_seconds) is not int or not 1 <= generation_seconds <= MAX_GENERATION_SECONDS:
            raise ValueError("Invalid paired generation budget")
        if not isinstance(value["boot_id"], str) or not ID.fullmatch(value["boot_id"]):
            raise ValueError("Invalid paired boot identity")
        if type(value["max_inflight"]) is not int or not 1 <= value["max_inflight"] <= MAX_INFLIGHT:
            raise ValueError("Invalid paired capacity")
        if type(value["uncertain"]) is not int or not 0 <= value["uncertain"] <= value["max_inflight"]:
            raise ValueError("Invalid unresolved reservation count")
        remaining = value["lifetime_remaining_ms"]
        if type(remaining) not in (int, float) or not math.isfinite(remaining) or not 0 < remaining <= 43200000:
            raise ValueError("Invalid paired lifetime")
        now = time.time()
        with self.lock:
            challenge = self.pair_challenge
            if (challenge is None or value["challenge_id"] != challenge["id"]
                    or time.monotonic() >= challenge["expires"] or self.stop_event.is_set()):
                raise ConflictError("Missing, consumed or expired browser-roundtrip challenge")
            self.pair_challenge = None
            age_ms = (time.monotonic() - challenge["issued"]) * 1000
            receipt = {"schema_version": 1, "protocol": 2,
                       "cpu": {"boot_id": self.boot_id, "pid": os.getpid(), "pid_start_ticks": self.pid_start_ticks,
                               "model": self.model, "max_inflight": self.max_inflight,
                               "generation_seconds": self.generation_seconds,
                               "started_unix": self.started_unix, "expires_unix": self.expires_unix,
                               "serving_profile": self.serving_profile},
                       "gpu": {"boot_id": value["boot_id"], "model": value["model"],
                               "max_inflight": value["max_inflight"], "uncertain": value["uncertain"],
                               "generation_seconds": generation_seconds,
                               "expires_unix": now + remaining / 1000, "serving_profile": self.serving_profile,
                               **model_proof, **controller_fields},
                       "paired_unix": now, "valid_until_unix": min(now + 15, self.expires_unix, now + remaining / 1000)}
            if receipt["valid_until_unix"] <= now:
                raise ValueError("Relay lifetime expired")
            atomic_write(self.root / "readiness.json", receipt)
            self.trace("pair_verified", challenge_id=value["challenge_id"], challenge_age_ms=age_ms)
            return {"recorded": True, "valid_until_unix": receipt["valid_until_unix"]}

    def trace(self, name, **fields):
        event = {"name": name, "ph": "i", "s": "t", "ts": time.monotonic_ns() / 1000,
                 "pid": os.getpid(), "tid": threading.get_ident(),
                 "args": {"wall_unix": time.time(), **fields}}
        with self.trace_lock:
            fd = os.open(self.root / "trace.perfetto.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(fd, "ab") as handle:
                handle.write(encoded(event) + b"\n")

    def recover(self):
        if self.role == "cpu":
            for pending in (self.root / "inflight").glob("*.json"):
                if not (self.root / "outbox" / pending.name).exists():
                    job = read_json(pending)
                    check_job(job, self.model, allow_expired=True)
                    if pending.name != job["id"] + ".json":
                        raise ValueError("Invalid recovered claim identity")
                    self._fail_claim(job["id"], "CPU relay restarted; outcome uncertain, no replay")
        else:
            for path in (self.root / "requests").glob("*.json"):
                record = read_json(path)
                if (not ID.fullmatch(path.stem) or not isinstance(record, dict) or record.get("id") != path.stem
                        or record.get("status") not in ("running", "uncertain", "finished", "failed", "cancelled")):
                    raise ValueError("Invalid GPU dispatch journal")
                if record["status"] == "running":
                    record["status"] = "uncertain"
                    atomic_write(path, record)
                self.gpu_journal[record["id"]] = record
                if len(self.gpu_journal) > MAX_TRACKED:
                    raise ValueError("GPU journal requires offline reconciliation")

    def _fail_claim(self, request_id, message):
        path = self.root / "outbox" / (request_id + ".json")
        if not path.exists():
            atomic_write(path, {"id": request_id, "ok": False, "error": message})
            self.trace("job_result", request_id=request_id, ok=False)

    def _unfinished_claims(self):
        active = []
        for path in (self.root / "inflight").glob("*.json"):
            if (self.root / "outbox" / path.name).exists():
                continue
            job = read_json(path)
            check_job(job, self.model, allow_expired=True)
            if path.name != job["id"] + ".json":
                raise ValueError("Invalid claimed job identity")
            if job["deadline_unix"] <= time.time():
                self._fail_claim(job["id"], "CPU deadline expired; no replay")
            else:
                active.append(job["id"])
        return active

    def claim(self, requested=None):
        if requested is not None and (not isinstance(requested, str) or not ID.fullmatch(requested)):
            raise ValueError("Invalid requested ID")
        with self.lock:
            if len(self._unfinished_claims()) >= self.max_inflight:
                return None
            paths = [self.root / "inbox" / (requested + ".json")] if requested else sorted((self.root / "inbox").glob("*.json"))
            for path in paths:
                if not path.exists():
                    continue
                job = read_json(path)
                check_job(job, self.model, allow_expired=True)
                if path.name != job["id"] + ".json":
                    raise ValueError("Spool filename does not match ID")
                destination = self.root / "inflight" / path.name
                if destination.exists() or (self.root / "outbox" / path.name).exists():
                    raise ValueError("Duplicate job ID")
                os.rename(path, destination)
                fsync_directory(path.parent)
                fsync_directory(destination.parent)
                if job["deadline_unix"] <= time.time():
                    self._fail_claim(job["id"], "CPU deadline expired before dispatch; no replay")
                    continue
                self.trace("job_claim", request_id=job["id"], operation=job["operation"])
                return job
        return None

    def begin_claim_receipt(self, value):
        claim_id, boot_id = value.get("claim_id"), value.get("boot_id")
        if not isinstance(claim_id, str) or not ID.fullmatch(claim_id):
            raise ValueError("Invalid claim operation ID")
        if boot_id != self.boot_id:
            raise ConflictError("CPU relay boot changed; no claim recovery")
        parameters = (value.get("id"), value.get("wait_ms", 0))
        with self.lock:
            if claim_id in self.claim_receipts:
                entry = self.claim_receipts[claim_id]
                if entry["parameters"] != parameters:
                    raise ConflictError("Claim operation parameters changed")
                self.claim_receipt(claim_id, boot_id)
                return entry, False
            if len(self.claim_receipts) >= MAX_TRACKED:
                raise CapacityError("Claim receipt limit reached; drain before restarting")
            entry = {"parameters": parameters, "status": "pending", "job_id": None,
                     "expires": min(time.monotonic() + CLAIM_RECEIPT_TTL, self.deadline)}
            self.claim_receipts[claim_id] = entry
            return entry, True

    def finish_claim_receipt(self, claim_id, job):
        with self.lock:
            entry = self.claim_receipts[claim_id]
            entry["job_id"] = job["id"] if job else None
            maximum = self.generation_seconds if job and job["operation"] == "generate" else 120
            entry["dispatch_deadline"] = min(time.monotonic() + maximum, self.deadline,
                time.monotonic() + job["deadline_unix"] - time.time() if job else self.deadline)
            entry["status"] = "complete"
            self.trace("claim_receipt_recorded", claim_id=claim_id, request_id=entry["job_id"])

    def claim_receipt(self, claim_id, boot_id):
        if not isinstance(claim_id, str) or not ID.fullmatch(claim_id):
            raise ValueError("Invalid claim operation ID")
        with self.lock:
            if boot_id != self.boot_id:
                raise ConflictError("CPU relay boot changed; no claim recovery")
            entry = self.claim_receipts[claim_id]
            if self.stop_event.is_set() or time.monotonic() >= entry["expires"]:
                raise GoneError("Claim receipt expired; no replay")
            if entry["status"] == "failed":
                raise GoneError("Claim outcome unavailable; reconcile without replay")
            result = {"claim_id": claim_id, "boot_id": self.boot_id,
                      "pending": entry["status"] == "pending", "job": None, "remaining_ms": 0}
            if result["pending"] or entry["job_id"] is None:
                return result
            request_id = entry["job_id"]
            if (self.root / "outbox" / (request_id + ".json")).exists():
                raise GoneError("Claim already completed or cancelled; no replay")
            remaining = min(entry["dispatch_deadline"], self.deadline) - time.monotonic()
            if remaining <= 0:
                raise GoneError("Claim dispatch deadline expired; no replay")
            job = read_json(self.root / "inflight" / (request_id + ".json"))
            check_job(job, self.model)
            if job["id"] != request_id:
                raise ValueError("Claim receipt job identity changed")
            result.update(job=job, remaining_ms=min(remaining, job["deadline_unix"] - time.time()) * 1000)
            return result

    def result(self, value):
        if not isinstance(value, dict) or not isinstance(value.get("id"), str) or not ID.fullmatch(value["id"]):
            raise ValueError("Invalid result ID")
        if type(value.get("ok")) is not bool:
            raise ValueError("Invalid result status")
        required = {"id", "ok", "response"} if value["ok"] else {"id", "ok", "error"}
        if not value["ok"] and "generation_failure" in value:
            required.add("generation_failure")
        if set(value) != required:
            raise ValueError("Invalid result fields")
        if value["ok"] and not isinstance(value["response"], dict):
            raise ValueError("Response must be an object")
        if not value["ok"] and (not isinstance(value["error"], str) or len(value["error"]) > 1000):
            raise ValueError("Invalid error")
        with self.lock:
            self._unfinished_claims()
            path = self.root / "outbox" / (value["id"] + ".json")
            pending = self.root / "inflight" / path.name
            if "generation_failure" in value:
                from rollout_manager.generation_failure import validate_generation_failure
                job = read_json(pending)
                check_job(job, self.model, allow_expired=True)
                if job["id"] != value["id"] or job["operation"] != "generate":
                    raise ValueError("Failure receipt has no matching generation claim")
                validate_generation_failure(value["generation_failure"], job["payload"], value["id"])
            if path.exists():
                if read_json(path, MAX_RESPONSE + MAX_BODY) != value:
                    raise ValueError("Conflicting result already committed")
                return {"committed": True, "duplicate": True}
            if not pending.exists():
                raise ValueError("No matching claimed job")
            atomic_write(path, value)
            self.trace("job_result", request_id=value["id"], ok=value["ok"])
            return {"committed": True, "duplicate": False}

    def cancel_claim(self, request_id):
        with self.lock:
            path = self.root / "inflight" / (request_id + ".json")
            inbox = self.root / "inbox" / path.name
            if not path.exists() and inbox.exists():
                job = read_json(inbox)
                check_job(job, self.model, allow_expired=True)
                if job["id"] != request_id:
                    raise ValueError("Invalid queued identity")
                os.rename(inbox, path)
                fsync_directory(inbox.parent)
                fsync_directory(path.parent)
            if not path.exists():
                raise ValueError("Unknown claim")
            self._fail_claim(request_id, "Request cancelled; no replay")
            return {"cancelled": True}

    def admit_generation(self, request_id, payload, timeout_ms, is_echo=False):
        maximum = 20000 if is_echo else self.generation_seconds * 1000
        if (type(timeout_ms) not in (int, float) or not math.isfinite(timeout_ms)
                or not 0 < timeout_ms <= maximum):
            raise ValueError("timeout_ms must be finite and within " + str(maximum))
        if self.stop_event.is_set() or time.monotonic() >= self.deadline:
            raise ValueError("Relay service lifetime ended")
        deadline = min(time.monotonic() + timeout_ms / 1000, self.deadline)
        with self.lock:
            if request_id in self.gpu_journal:
                record = self.gpu_journal[request_id]
                self.trace("generation_rejected", request_id=request_id, status=record["status"],
                           payload_matches=record.get("payload_sha256") == hashlib.sha256(encoded(payload)).hexdigest())
                raise ConflictError("GPU request ID already admitted; no replay")
            occupied = {key for key, record in self.gpu_journal.items() if record["status"] in ("running", "uncertain")}
            if len(occupied | self.completions.keys()) >= self.max_inflight:
                raise CapacityError("GPU relay capacity occupied; uncertain requests retain reservations")
            if len(self.gpu_journal) >= MAX_TRACKED:
                raise CapacityError("GPU journal limit reached; offline reconciliation required")
            record = {"id": request_id, "status": "running", "boot_id": self.boot_id,
                      "payload_sha256": hashlib.sha256(encoded(payload)).hexdigest()}
            atomic_write(self.root / "requests" / (request_id + ".json"), record)
            self.gpu_journal[request_id] = record
            context = {"cancel": threading.Event(), "connection": None, "dispatched": False,
                       "deadline": deadline}
            self.active_generations[request_id] = context
            self.trace("generation_admitted", request_id=request_id, timeout_ms=timeout_ms)
            return context

    def submit_generation(self, value):
        with self.lock:
            context = self.admit_generation(value["id"], value["payload"], value["timeout_ms"])
            entry = {"event": threading.Event(), "poll_lock": threading.Lock(), "value": None}
            self.completions[value["id"]] = entry

        def worker():
            try:
                code, result = execute_generation(self, value, context)
            except Exception:
                with self.lock:
                    if self.active_generations.get(value["id"]) is context:
                        self.finish_generation(value["id"], context, "uncertain" if context["dispatched"] else "failed")
                code, result = 500, {"error": "Generation worker failed; no replay"}
            with self.lock:
                entry["value"] = {"id": value["id"], "pending": False, "ok": code == 200,
                                  "response" if code == 200 else "error": result if code == 200 else result["error"]}
                if code != 200 and "generation_failure" in result:
                    entry["value"]["generation_failure"] = result["generation_failure"]
                entry["event"].set()
                self.completion_changed.notify_all()

        try:
            threading.Thread(target=worker, name="generation-" + value["id"], daemon=True).start()
        except Exception:
            with self.lock:
                self.finish_generation(value["id"], context, "failed")
                del self.completions[value["id"]]
            raise
        return {"id": value["id"], "admitted": True}

    def completion_entry(self, request_id):
        with self.lock:
            entry = self.completions.get(request_id)
            if entry is None:
                if request_id not in self.gpu_journal:
                    raise KeyError("Unknown request ID")
                raise GoneError("Completion unavailable or acknowledged; no replay")
            return entry

    def acknowledge_delivery(self, request_id):
        with self.lock:
            record = self.gpu_journal.get(request_id)
            if record is not None and record.get("delivery_acknowledged"):
                return {"acknowledged": True}
            entry = self.completion_entry(request_id)
            if not entry["event"].is_set():
                raise ConflictError("Generation has not completed")
            record = {**record, "delivery_acknowledged": True}
            atomic_write(self.root / "requests" / (request_id + ".json"), record)
            self.gpu_journal[request_id] = record
            del self.completions[request_id]
            self.trace("generation_delivery_ack", request_id=request_id)
            return {"acknowledged": True}

    def cancel_generation(self, request_id):
        with self.lock:
            record = self.gpu_journal.get(request_id)
            if record is None:
                # Cancellation may overtake dispatch. This tombstone prevents a
                # delayed browser request from starting the cancelled generation.
                if len(self.gpu_journal) >= MAX_TRACKED:
                    raise CapacityError("GPU journal limit reached")
                record = {"id": request_id, "status": "cancelled", "boot_id": self.boot_id}
                atomic_write(self.root / "requests" / (request_id + ".json"), record)
                self.gpu_journal[request_id] = record
            context = self.active_generations.get(request_id)
            connection = None
            if context:
                context["cancel"].set()
                connection = context["connection"]
            self.trace("generation_cancel", request_id=request_id, status=record["status"], active=context is not None)
        if connection is not None:
            try:
                if connection.sock is not None:
                    connection.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            connection.close()
        return {"cancel_requested": True, "upstream_stop_confirmed": False}

    def finish_generation(self, request_id, context, status):
        with self.lock:
            if self.active_generations.get(request_id) is not context:
                raise ConflictError("Generation ownership changed")
            record = {**self.gpu_journal[request_id], "status": status}
            atomic_write(self.root / "requests" / (request_id + ".json"), record)
            self.gpu_journal[request_id] = record
            del self.active_generations[request_id]
            self.trace("generation_terminal", request_id=request_id, status=status)

    def status(self):
        with self.lock:
            active = len(self._unfinished_claims()) if self.role == "cpu" else len(self.active_generations)
            uncertain = sum(record["status"] == "uncertain" for record in self.gpu_journal.values())
            counts = {name: len(list((self.root / name).glob("*.json"))) for name in ("inbox", "inflight", "outbox")}
            return {"role": self.role, "counts": counts, "max_inflight": self.max_inflight,
                    **self.artifacts.status(),
                    "active": active, "active_generations": len(self.active_generations),
                    "uncertain": uncertain, "boot_id": self.boot_id,
                    "retained_unacknowledged": len(self.completions),
                    "acknowledged_ids": sorted(key for key, record in self.gpu_journal.items()
                        if record.get("delivery_acknowledged") is True)}


class BoundedHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, max_threads=16):
        if type(max_threads) is not int or not 1 <= max_threads <= MAX_INFLIGHT + 8:
            raise ValueError("Invalid HTTP handler thread limit")
        self.worker_slots = threading.BoundedSemaphore(max_threads)
        super().__init__(address, handler)

    def get_request(self):
        connection, address = super().get_request()
        connection.settimeout(10)
        return connection, address

    def process_request(self, request, client_address):
        if not self.worker_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.worker_slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.worker_slots.release()


class Handler(BaseHTTPRequestHandler):
    server_version = "RelayPilot/1"

    def log_message(self, *args):
        pass

    @property
    def state(self):
        return self.server.state

    def route(self):
        path = urllib.parse.urlsplit(self.path).path
        if "%" in path or ".." in path or "//" in path:
            return None
        prefix = self.state.prefix
        if prefix and (path == prefix or path.startswith(prefix + "/")):
            path = path[len(prefix):] or "/"
        return path

    def send_json(self, code, value):
        data = encoded(value)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def client_alive(self):
        readable, _, _ = select.select([self.connection], [], [], 0)
        return not readable or bool(self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT))

    def wait_until(self, deadline):
        while time.monotonic() < deadline:
            if self.state.stop_event.is_set() or time.monotonic() >= self.state.deadline:
                raise ValueError("Relay service lifetime ended")
            if not self.client_alive():
                raise BrokenPipeError("Poll client disconnected")
            self.state.stop_event.wait(min(0.1, max(0, deadline - time.monotonic())))

    def pulse(self):
        if not self.state.pulse_gate.acquire(blocking=False):
            raise CapacityError("One readiness poll is already outstanding")
        try:
            self.wait_until(self.state.next_pulse_due)
            if not self.client_alive():
                raise BrokenPipeError("Poll client disconnected")
            with self.state.lock:
                now = time.monotonic()
                if self.state.stop_event.is_set() or now >= self.state.deadline:
                    raise ValueError("Relay service lifetime ended")
                challenge = {"id": uuid.uuid4().hex, "issued": now, "expires": min(now + PULSE_TTL, self.state.deadline)}
                self.state.pair_challenge = challenge
                self.state.next_pulse_due = now + PULSE_INTERVAL
            # Issuing a challenge does not create or refresh readiness.json.
            self.send_json(200, {"challenge_id": challenge["id"], "ttl_ms": (challenge["expires"] - now) * 1000})
        finally:
            self.state.pulse_gate.release()

    def claim_response(self, value):
        if set(value) - {"id", "wait_ms", "claim_id", "boot_id"}:
            raise ValueError("Unexpected claim fields")
        wait_ms = value.get("wait_ms", 0)
        if type(wait_ms) is not int or not 0 <= wait_ms <= 5000:
            raise ValueError("Claim wait_ms must be an integer from 0 through 5000")
        if "id" in value and (not isinstance(value["id"], str) or not ID.fullmatch(value["id"])):
            raise ValueError("Invalid requested ID")
        entry, claim_id = None, value.get("claim_id")
        if "claim_id" in value or "boot_id" in value:
            entry, owner = self.state.begin_claim_receipt(value)
            if not owner:
                self.claim_receipt_response(claim_id, value["boot_id"], wait_ms)
                return
        acquired = False
        started, job = time.monotonic(), None
        try:
            if wait_ms:
                acquired = self.state.claim_poll_gate.acquire(blocking=False)
                if not acquired:
                    raise CapacityError("One claim poll is already outstanding")
            deadline = min(started + wait_ms / 1000, self.state.deadline)
            while True:
                if self.state.stop_event.is_set() or time.monotonic() >= self.state.deadline:
                    raise ValueError("Relay service lifetime ended")
                if not self.client_alive():
                    raise BrokenPipeError("Poll client disconnected")
                with self.state.lock:
                    job = self.state.claim(value.get("id"))
                    complete = job is not None or time.monotonic() >= deadline
                    if complete and entry is not None:
                        self.state.finish_claim_receipt(claim_id, job)
                if complete:
                    break
                self.wait_until(min(deadline, time.monotonic() + 0.1))
            if entry is not None:
                result = self.state.claim_receipt(claim_id, value["boot_id"])
                result["server_wait_ms"] = max(0, (time.monotonic() - started) * 1000)
                self.send_json(200, result)
                return
            maximum = self.state.generation_seconds * 1000 if job and job["operation"] == "generate" else 120000
            self.send_json(200, {"job": job, "remaining_ms": max(0, min(maximum, (job["deadline_unix"] - time.time()) * 1000, (self.state.deadline - time.monotonic()) * 1000)) if job else 0,
                                 "server_wait_ms": max(0, (time.monotonic() - started) * 1000)})
        finally:
            if entry is not None:
                with self.state.lock:
                    if entry["status"] == "pending":
                        entry["status"] = "failed"
            if acquired:
                self.state.claim_poll_gate.release()

    def claim_receipt_response(self, claim_id, boot_id, wait_ms):
        if not self.state.claim_receipt_gate.acquire(blocking=False):
            raise CapacityError("One claim receipt lookup is already outstanding")
        started = time.monotonic()
        try:
            deadline = min(started + wait_ms / 1000, self.state.deadline)
            while True:
                value = self.state.claim_receipt(claim_id, boot_id)
                if not value["pending"] or time.monotonic() >= deadline:
                    break
                self.wait_until(min(deadline, time.monotonic() + 0.1))
            value["server_wait_ms"] = max(0, (time.monotonic() - started) * 1000)
            self.state.trace("claim_receipt_lookup", claim_id=claim_id, pending=value["pending"],
                             request_id=value["job"]["id"] if value["job"] else None)
            self.send_json(200, value)
        finally:
            self.state.claim_receipt_gate.release()

    def authenticate(self, write=False):
        supplied_headers = self.headers.get_all("X-Relay-Token", [])
        supplied = supplied_headers[0] if len(supplied_headers) == 1 else ""
        if not re.fullmatch(r"[0-9a-f]{64}", supplied) or not hmac.compare_digest(supplied, self.state.token):
            self.state.trace("authentication_rejected", relay_token_present=bool(supplied_headers),
                             relay_token_length=len(supplied_headers[0]) if supplied_headers else 0,
                             duplicate_relay_token_headers=len(supplied_headers) > 1,
                             authorization_present="Authorization" in self.headers)
            self.send_json(401, {"error": "Authentication required"})
            return False
        origins = self.headers.get_all("Origin", [])
        relay_origins = self.headers.get_all("X-Relay-Origin", [])
        provided = origins[0] if len(origins) == 1 else None
        asserted = relay_origins[0] if len(relay_origins) == 1 else None
        exact_public = provided == self.state.origin
        exact_proxy = self.state.proxy_origin is not None and provided == self.state.proxy_origin
        original_assertion = asserted == self.state.origin
        # The custom header is a client assertion, trusted only alongside the
        # secret token and an explicitly configured private gateway Origin.
        valid_direct = exact_public and (not relay_origins or original_assertion)
        valid_proxy = exact_proxy and original_assertion
        valid_originless_read = not write and not origins and (not relay_origins or original_assertion)
        if len(origins) > 1 or len(relay_origins) > 1 or not (valid_direct or valid_proxy or valid_originless_read):
            self.state.trace("origin_rejected", origin_present=bool(origins),
                             origin_matches_public=exact_public, origin_matches_proxy=exact_proxy,
                             relay_origin_present=bool(relay_origins), relay_origin_matches_public=original_assertion,
                             duplicate_origin_headers=len(origins) > 1,
                             duplicate_relay_origin_headers=len(relay_origins) > 1, write_request=write)
            self.send_json(403, {"error": "Origin rejected"})
            return False
        return True

    def do_GET(self):
        route = self.route()
        if route in ("/", "/relay.html"):
            data = Path(__file__).with_name("relay.html").read_bytes()
            self.send_response(200)
            for key, value in {"Content-Type": "text/html; charset=utf-8", "Content-Length": str(len(data)),
                               "Cache-Control": "no-store", "Referrer-Policy": "no-referrer",
                               "X-Content-Type-Options": "nosniff",
                               "Content-Security-Policy": "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"}.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)
        elif route == "/health":
            self.send_json(200, {"alive": True})
        elif ((route in ("/api/completion", "/api/completions") and self.state.role == "gpu")
              or (route == "/api/claim-receipt" and self.state.role == "cpu")):
            if not self.authenticate():
                return
            try:
                if route == "/api/completion":
                    self.completion_response()
                elif route == "/api/completions":
                    self.completions_response()
                else:
                    query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query,
                        keep_blank_values=True, strict_parsing=True, max_num_fields=3)
                    if (set(query) != {"claim_id", "boot_id", "wait_ms"}
                            or any(len(values) != 1 for values in query.values())
                            or not re.fullmatch(r"[0-9]{1,4}", query["wait_ms"][0])
                            or int(query["wait_ms"][0]) > 5000):
                        raise ValueError("Invalid claim receipt query")
                    self.claim_receipt_response(query["claim_id"][0], query["boot_id"][0], int(query["wait_ms"][0]))
            except CapacityError as exc:
                self.send_json(429, {"error": str(exc)})
            except ConflictError as exc:
                self.send_json(409, {"error": str(exc)})
            except GoneError as exc:
                self.send_json(410, {"error": str(exc)})
            except KeyError:
                self.send_json(404, {"error": "Unknown request ID"})
            except ValueError as exc:
                self.send_json(400, {"error": str(exc)[:200]})
            except (BrokenPipeError, ConnectionResetError):
                pass
        elif route in ("/api/config", "/api/status"):
            if not self.authenticate():
                return
            if route == "/api/config":
                try:
                    self.send_json(200, self.state.configuration())
                except (OSError, ValueError, IndexError):
                    self.send_json(503, {"error": "Owned model capacity unavailable; requalification required"})
            else:
                self.send_json(200, self.state.status())
        else:
            self.send_json(404, {"error": "Not found"})

    def completion_response(self):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query, keep_blank_values=True,
                                      strict_parsing=True, max_num_fields=2)
        if (set(query) - {"id", "wait_ms"} or "id" not in query
                or any(len(values) != 1 for values in query.values())
                or not ID.fullmatch(query["id"][0])):
            raise ValueError("Invalid completion query")
        raw_wait = query.get("wait_ms", ["0"])[0]
        if not re.fullmatch(r"[0-9]{1,4}", raw_wait) or int(raw_wait) > 5000:
            raise ValueError("wait_ms must be an integer within 5000")
        request_id = query["id"][0]
        entry = self.state.completion_entry(request_id)
        if not entry["poll_lock"].acquire(blocking=False):
            raise CapacityError("One completion poll per request is permitted")
        started = time.monotonic()
        try:
            deadline = min(time.monotonic() + int(raw_wait) / 1000, self.state.deadline)
            while not entry["event"].is_set() and time.monotonic() < deadline:
                if self.state.stop_event.is_set():
                    raise ValueError("Relay service lifetime ended")
                if not self.client_alive():
                    raise BrokenPipeError("Completion client disconnected")
                entry["event"].wait(min(0.1, max(0, deadline - time.monotonic())))
            with self.state.lock:
                result = entry["value"] if entry["event"].is_set() else {"id": request_id, "pending": True}
            self.state.trace("generation_poll", request_id=request_id, pending=result["pending"],
                             server_wait_ms=(time.monotonic() - started) * 1000)
            self.send_json(200, result)
        finally:
            entry["poll_lock"].release()

    def completions_response(self):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query, keep_blank_values=True,
                                      strict_parsing=True, max_num_fields=3)
        if set(query) != {"ids", "boot_id", "wait_ms"} or any(len(value) != 1 for value in query.values()):
            raise ValueError("Invalid shared completion query")
        ids = query["ids"][0].split(",")
        if not 1 <= len(ids) <= self.state.max_inflight or len(set(ids)) != len(ids) or any(not ID.fullmatch(value) for value in ids):
            raise ValueError("Invalid shared completion IDs")
        if query["boot_id"][0] != self.state.boot_id:
            raise ConflictError("GPU boot identity changed; no replay")
        raw_wait = query["wait_ms"][0]
        if not re.fullmatch(r"[0-9]{1,4}", raw_wait) or int(raw_wait) > 5000:
            raise ValueError("wait_ms must be an integer within 5000")
        if not self.state.completion_batch_gate.acquire(blocking=False):
            raise CapacityError("Only one shared completion poll is permitted")
        started = time.monotonic()
        deadline = min(started + int(raw_wait) / 1000, self.state.deadline)
        try:
            result = None
            while True:
                if self.state.stop_event.is_set() or time.monotonic() >= self.state.deadline:
                    raise GoneError("Relay service lifetime ended")
                if not self.client_alive():
                    raise BrokenPipeError("Completion client disconnected")
                with self.state.completion_changed:
                    for request_id in ids:
                        try:
                            entry = self.state.completion_entry(request_id)
                        except (KeyError, GoneError):
                            result = {"id": request_id, "pending": False, "ok": False,
                                      "error": "Completion unknown or unavailable; no replay"}
                            break
                        if entry["event"].is_set():
                            result = entry["value"]
                            break
                    if result is not None or time.monotonic() >= deadline:
                        break
                    self.state.completion_changed.wait(min(0.1, max(0, deadline - time.monotonic())))
            # One terminal body per response keeps the existing 2 MiB payload bound.
            self.state.trace("generation_batch_poll", request_ids=ids,
                             ready_id=result["id"] if result else None,
                             server_wait_ms=(time.monotonic() - started) * 1000)
            self.send_json(200, {"boot_id": self.state.boot_id, "polled_ids": ids, "result": result})
        finally:
            self.state.completion_batch_gate.release()

    def do_POST(self):
        if not self.authenticate(write=True):
            return
        route = self.route()
        limit = MAX_RESPONSE + MAX_BODY if route == "/api/result" else MAX_BODY
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if not 0 <= length <= limit or self.headers.get("Transfer-Encoding"):
                self.send_json(413, {"error": "Request size rejected"})
                return
            if self.headers.get_content_type() != "application/json":
                raise ValueError("JSON content type required")
            self.connection.settimeout(10)
            value = json.loads(self.rfile.read(length))
            if not isinstance(value, dict):
                raise ValueError("JSON object required")
            cpu = self.state.role == "cpu"
            if route.startswith("/api/artifact-"):
                from .artifacts import artifact_api
                self.send_json(200, artifact_api(self.state.artifacts, route, value))
            elif route == "/api/claim" and cpu:
                self.claim_response(value)
            elif route == "/api/pulse" and cpu:
                if value:
                    raise ValueError("Empty pulse request required")
                self.pulse()
            elif route == "/api/pair" and cpu:
                self.send_json(200, self.state.pair_receipt(value))
            elif route == "/api/result" and cpu:
                self.send_json(200, self.state.result(value))
            elif route == "/api/cancel":
                if set(value) != {"id"} or not isinstance(value["id"], str) or not ID.fullmatch(value["id"]):
                    raise ValueError("Invalid cancellation ID")
                self.send_json(200, self.state.cancel_claim(value["id"]) if cpu else self.state.cancel_generation(value["id"]))
            elif route == "/api/qualify" and cpu:
                if value:
                    raise ValueError("Empty object required")
                job = {"id": uuid.uuid4().hex, "operation": "echo", "payload": {"chunks": 3}, "deadline_unix": time.time() + 120}
                atomic_write(self.state.root / "inbox" / (job["id"] + ".json"), job)
                self.state.trace("qualification_enqueue", request_id=job["id"])
                self.send_json(200, {"id": job["id"]})
            elif route == "/api/telemetry":
                if set(value) != {"request_id", "event", "duration_ms"} or not isinstance(value["request_id"], str) or not ID.fullmatch(value["request_id"]):
                    raise ValueError("Invalid telemetry")
                if value["event"] not in ("browser_dispatch", "browser_complete", "browser_error", "echo_chunk"):
                    raise ValueError("Unknown telemetry event")
                duration = value["duration_ms"]
                if isinstance(duration, bool) or not isinstance(duration, (float, int)) or not 0 <= duration <= 3600000:
                    raise ValueError("Invalid duration")
                self.state.trace(value["event"], request_id=value["request_id"], duration_ms=duration)
                self.send_json(200, {"recorded": True})
            elif route == "/api/echo" and not cpu:
                self.echo(value)
            elif route == "/api/generate" and not cpu:
                self.generate(value)
            elif route == "/api/submit" and not cpu:
                self.validate_generation(value)
                self.send_json(202, self.state.submit_generation(value))
            elif route == "/api/complete-ack" and not cpu:
                if set(value) != {"id"} or not isinstance(value["id"], str) or not ID.fullmatch(value["id"]):
                    raise ValueError("Invalid delivery acknowledgment")
                self.send_json(200, self.state.acknowledge_delivery(value["id"]))
            else:
                self.send_json(404, {"error": "Not found"})
        except CapacityError as exc:
            self.send_json(429, {"error": str(exc)})
        except ConflictError as exc:
            self.send_json(409, {"error": str(exc)})
        except GoneError as exc:
            self.send_json(410, {"error": str(exc)})
        except KeyError:
            self.send_json(404, {"error": "Unknown request ID"})
        except (ValueError, json.JSONDecodeError) as exc:
            self.send_json(400, {"error": str(exc)[:200]})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            self.send_json(500, {"error": "Internal relay error; inspect local state"})

    def echo(self, value):
        if set(value) != {"id"} or not isinstance(value["id"], str) or not ID.fullmatch(value["id"]):
            raise ValueError("Invalid qualification ID")
        context = self.state.admit_generation(value["id"], {"operation": "echo"}, 20000, is_echo=True)
        terminal = "failed"
        try:
            self.state.trace("echo_begin", request_id=value["id"])
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Connection", "close")
            self.end_headers()
            for sequence in range(3):
                if context["cancel"].is_set() or time.monotonic() >= context["deadline"]:
                    return
                self.wfile.write(encoded({"id": value["id"], "sequence": sequence, "echo": "relay-qualification"}) + b"\n")
                self.wfile.flush()
                time.sleep(0.2)
            self.close_connection = True
            terminal = "finished"
            self.state.trace("echo_end", request_id=value["id"])
        finally:
            self.state.finish_generation(value["id"], context, terminal)

    def validate_generation(self, value):
        if set(value) != {"id", "payload", "timeout_ms"} or not isinstance(value["id"], str) or not ID.fullmatch(value["id"]):
            raise ValueError("Invalid generation request")
        check_generation(value["payload"], self.state.model)

    def generate(self, value):
        self.validate_generation(value)
        context = self.state.admit_generation(value["id"], value["payload"], value["timeout_ms"])
        code, result = execute_generation(self.state, value, context)
        self.send_json(code, result)


def execute_generation(state, value, context):
    request_id = value["id"]
    connection = None
    timer = threading.Timer(max(0, context["deadline"] - time.monotonic()), state.cancel_generation, args=(request_id,))
    timer.daemon = True
    terminal = "failed"
    state.trace("generation_begin", request_id=request_id)
    try:
        timer.start()
        timeout = max(0.001, context["deadline"] - time.monotonic())
        controlled = state.serving_profile in {"native-pytorch-v1", "vllm-training-v1"}
        connection = open_upstream_connection(timeout, 8002) if controlled else open_upstream_connection(timeout)
        with state.lock:
            context["connection"] = connection
        # Fixed HTTPConnection has no environment proxy or redirect behavior.
        connection.connect()
        connection.auto_open = 0
        with state.lock:
            if context["cancel"].is_set() or time.monotonic() >= context["deadline"]:
                raise TimeoutError("Request cancelled before dispatch")
            context["dispatched"] = True
        terminal = "uncertain"
        headers = {"Content-Type": "application/json"}
        if controlled:
            headers["X-Rollout-Request-Id"] = request_id
            headers["Authorization"] = "Bearer " + native_controller_token(state.controller_token_file)
        connection.request("POST", "/v1/chat/completions", encoded(value["payload"]), headers)
        response = connection.getresponse()
        data = response.read(MAX_RESPONSE + 1)
        if len(data) > MAX_RESPONSE:
            raise ValueError("Upstream response too large")
        if context["cancel"].is_set() or time.monotonic() >= context["deadline"]:
            raise TimeoutError("Request completed after cancellation or deadline")
        if response.length not in (0, None):
            raise http.client.HTTPException("Incomplete upstream response")
        if response.status != 200:
            if controlled:
                from rollout_manager.generation_failure import validate_generation_failure, failure_is_terminal
                body = json.loads(data)
                if isinstance(body, dict) and "generation_failure" in body:
                    receipt = validate_generation_failure(body["generation_failure"], value["payload"], request_id)
                    terminal = "failed" if failure_is_terminal(receipt) else "uncertain"
                    state.trace("generation_failure_receipt", request_id=request_id,
                                sampler_outcome=receipt["sampler_outcome"], terminal_proven=failure_is_terminal(receipt))
                    return 502, {"error": "Loopback model service rejected generation; no replay",
                                 "generation_failure": receipt}
            else:
                terminal = "failed"
            raise ValueError("Fixed loopback upstream rejected the request")
        result = json.loads(data)
        if not isinstance(result, dict) or not isinstance(result.get("choices"), list):
            raise ValueError("Invalid model response")
        if len(encoded(result)) > MAX_RESPONSE:
            raise ValueError("Encoded model response too large")
        terminal = "finished"
        state.trace("generation_end", request_id=request_id, response_bytes=len(data))
        return 200, result
    except (OSError, http.client.HTTPException, ValueError) as exc:
        state.trace("generation_error", request_id=request_id, error_type=type(exc).__name__,
                    cancelled=context["cancel"].is_set(), deadline_expired=time.monotonic() >= context["deadline"],
                    dispatched=context["dispatched"])
        return 502, {"error": "Loopback model service failed, cancelled, or timed out; no replay"}
    finally:
        timer.cancel()
        if connection is not None:
            connection.close()
        state.finish_generation(request_id, context, terminal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--listen", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--public-base", required=True)
    parser.add_argument("--peer-origin", required=True)
    parser.add_argument("--proxy-origin", help="Opt-in exact Origin set by a verified private gateway")
    parser.add_argument("--upstream", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash")
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--max-inflight", type=int, default=1)
    parser.add_argument("--generation-seconds", type=int, default=120)
    parser.add_argument("--model-pid", type=int, help="GPU-only exact API process owning the qualified loopback listener")
    parser.add_argument("--serving-profile", choices=("vllm-v1", "native-pytorch-v1", "vllm-training-v1"), default="vllm-v1")
    parser.add_argument("--native-controller-receipt", type=Path)
    parser.add_argument("--native-controller-token-file", type=Path)
    parser.add_argument("--training-controller-receipt", type=Path)
    parser.add_argument("--training-controller-token-file", type=Path)
    args = parser.parse_args()
    if not 1 <= args.max_seconds <= 43200 or not 1024 <= args.port <= 65535:
        parser.error("Use an unprivileged port and lifetime of 1..43200 seconds")
    if not 1 <= args.generation_seconds <= MAX_GENERATION_SECONDS:
        parser.error("generation-seconds must be between 1 and 600")
    lease = acquire_runtime_lock(args.root)
    state = RelayState(args.root, args.role, args.public_base, args.peer_origin, args.upstream, args.model, args.proxy_origin, args.max_inflight, args.max_seconds, args.generation_seconds, model_pid=args.model_pid,
        serving_profile=args.serving_profile, native_controller_receipt=args.native_controller_receipt,
        native_controller_token_file=args.native_controller_token_file,
        training_controller_receipt=args.training_controller_receipt,
        training_controller_token_file=args.training_controller_token_file)
    server = BoundedHTTPServer((args.listen, args.port), Handler, max_threads=args.max_inflight + 8)
    server.state = state
    def stop(*unused):
        state.stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    timer = threading.Timer(args.max_seconds, stop)
    timer.daemon = True
    timer.start()
    print(f"{args.role} relay listening on {args.listen}:{args.port}; private launch URL: {args.root / 'launch-url.txt'}", flush=True)
    try:
        server.serve_forever(poll_interval=0.2)
    finally:
        timer.cancel()
        state.trace("server_stop")
        server.server_close()
        os.close(lease)


if __name__ == "__main__":
    main()
