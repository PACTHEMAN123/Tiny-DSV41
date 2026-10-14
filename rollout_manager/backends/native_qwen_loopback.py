"""Owned loopback HTTP shim for the SAME native benchmark policy controller.

No model is constructed here. The browser FileRelay still owns durable dispatch,
CPU result commits and completion acknowledgements. Only an actual local relay
drain plus actual native sampler drain permits the next immutable activation.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import secrets
import stat
import threading
import time
import traceback
import sys

ID = re.compile(r"[a-f0-9]{32}\Z")
TOKEN = re.compile(r"[a-f0-9]{64}\Z")
MAX_REQUEST = 512 * 1024
MAX_RESPONSE = 1024 * 1024
STATIC_PROOF_FIELDS = ("schema_version", "serving_profile", "served_model", "model_pid",
    "model_pid_start_ticks", "model_cmdline_sha256", "generation_slots", "started_unix",
    "expires_unix", "model_dir", "python_executable", "netns_inode")


def encoded(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def strict_json(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    def invalid(_):
        raise ValueError("nonfinite JSON constant")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    encoded(value)
    return value


def read_private(path, maximum):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_mode & 0o077 or info.st_size > maximum:
            raise ValueError("expected a bounded private regular file")
        value = stream.read(maximum + 1)
    if len(value) > maximum:
        raise ValueError("private file exceeded byte budget")
    return value


def write_new(path, value):
    data = encoded(value)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def linux_process_identity():
    """Actual calling Linux process; no platform fallback or supplied identity."""
    pid = os.getpid()
    fields = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] in {"Z", "X"}:
        raise RuntimeError("native controller process is not live")
    command = Path("/proc/self/cmdline").read_bytes()
    network = re.fullmatch(r"net:\[([0-9]+)\]", os.readlink("/proc/self/ns/net"))
    if not command.endswith(b"\0") or network is None:
        raise RuntimeError("native process identity is incomplete")
    return {"model_pid": pid, "model_pid_start_ticks": int(fields[19]),
            "model_cmdline_sha256": hashlib.sha256(command).hexdigest(),
            "python_executable": sys.executable, "netns_inode": int(network[1])}


class _Server(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # No payloads, bearer values or private URLs in HTTP logs.

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    @property
    def owner(self):
        return self.server.owner

    def send(self, code, value):
        data = encoded(value)
        if len(data) > MAX_RESPONSE:
            code, data = 500, encoded({"error": "native response exceeds byte budget"})
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def authenticated(self):
        values = self.headers.get_all("Authorization", [])
        expected = "Bearer " + self.owner.token
        if len(values) != 1 or not hmac.compare_digest(values[0], expected):
            self.send(401, {"error": "private loopback authentication required"})
            return False
        return True

    def do_GET(self):
        if not self.authenticated():
            return
        if self.path not in {"/rollout/profile", "/owned-controller-proof"}:
            self.send(404, {"error": "unknown native endpoint"})
            return
        try:
            self.send(200, self.owner.profile())
        except (RuntimeError, ValueError, OSError):
            self.send(503, {"error": "native policy or process proof unavailable"})

    def do_POST(self):
        if not self.authenticated():
            return
        if self.path != "/v1/chat/completions":
            self.send(404, {"error": "unknown native endpoint"})
            return
        identities = self.headers.get_all("X-Rollout-Request-Id", [])
        if len(identities) != 1 or not ID.fullmatch(identities[0]):
            self.send(400, {"error": "one durable 32-hex request header required"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if (not 0 < length <= MAX_REQUEST or self.headers.get("Transfer-Encoding")
                    or self.headers.get_content_type() != "application/json"):
                raise ValueError("invalid native request framing")
            payload = strict_json(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("native payload must be an object")
            if not self.owner.reserve(identities[0], payload):
                self.send(429, {"error": "native admission paused, expired or occupied"})
                return
        except FileExistsError:
            self.send(409, {"error": "durable native identity already claimed; no replay"})
            return
        except (ValueError, OSError):
            self.send(400, {"error": "native request rejected before sampling"})
            return
        identity, delivered, stage = identities[0], False, "native_generation"
        sampler_drained, failure_receipt = False, None
        try:
            completion = self.owner.controller.generate(payload, identity)
            # This synchronous successful controller return includes the final
            # actual sampled-token/probability CPU completion barrier.
            sampler_drained = True
            stage = "completion_persistence"
            write_new(self.owner.output / (identity + ".completion.json"), completion)
            from rollout_manager.generation_training import TrainingGenerationContract
            stage = "training_contract"
            trace = TrainingGenerationContract(self.owner.controller.policy, self.owner.controller.capabilities()).validate(
                payload, completion, identity, 1)
            write_new(self.owner.output / (identity + ".training.json"), trace.to_dict())
            stage = "response_byte_budget"
            if len(encoded(completion)) > MAX_RESPONSE:
                raise ValueError("native completion exceeds byte budget")
            stage = "http_delivery"
            self.send(200, completion)
            delivered = True
        except Exception as error:
            stage = getattr(error, "native_generation_stage", stage)
            failure = getattr(self.owner.controller, "generation_failure_receipt", None)
            if failure is not None:
                try:
                    failure_receipt = failure(payload, identity, stage)
                    if failure_receipt is not None:
                        sampler_drained = failure_receipt["sampler_drained"]
                except Exception:
                    sampler_drained, failure_receipt = False, None
            # Private diagnostics include no locals, headers or bearer values.
            write_new(self.owner.output / (identity + ".error.json"), {
                "request_id": identity, "stage": stage,
                "error_type": type(error).__name__,
                "error_message": str(error).replace(self.owner.token, "[redacted]"),
                "traceback": traceback.format_exc().replace(self.owner.token, "[redacted]"),
                "automatic_replay": False})
            try:
                body = {"error": "native completion failed; no replay"}
                if failure_receipt is not None:
                    body["generation_failure"] = failure_receipt
                self.send(500, body)
            except OSError:
                pass
        finally:
            try:
                write_new(self.owner.output / (identity + ".http-state.json"),
                          {"id": identity, "http_response_written": delivered, "sampler_drained": sampler_drained,
                           "generation_failure": failure_receipt,
                           "browser_delivery_acknowledged": False, "automatic_replay": False})
            finally:
                self.owner.release()


class NativeControllerHTTPShim:
    def __init__(self, controller, output, *, port=8002, lifetime_seconds=30000):
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError("an explicit unprivileged owned port is required")
        if type(lifetime_seconds) is not int or not 1 <= lifetime_seconds <= 43200:
            raise ValueError("native lifetime must be bounded by 43200 seconds")
        self.controller, self.output = controller, Path(output)
        self.output.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.controller.configure_diagnostics(self.output)
        self.token_file = self.output / "controller-token.txt"
        self.token = secrets.token_hex(32)
        fd = os.open(self.token_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(self.token + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.started_unix = time.time()
        self.expires_unix = self.started_unix + lifetime_seconds
        self._expires = time.monotonic() + lifetime_seconds
        self._condition, self._active = threading.Condition(), 0
        self._phase, self._requests = "new", set()
        self._server = _Server(("127.0.0.1", port), _Handler)
        self._server.owner = self
        self.port = self._server.server_port
        self._thread = None

    def start(self):
        with self._condition:
            if self._phase != "new":
                raise RuntimeError("shim start is single-use")
            self._phase = "running"
        proof = self.profile()
        if proof["serving_profile"] == "vllm-training-v1":
            from ..terminal_bench.controller import training_controller_identity
            startup = training_controller_identity(proof, include_policy=False)
        else:
            startup = {key: proof[key] for key in STATIC_PROOF_FIELDS}
        write_new(self.output / "owned-controller-startup.json", startup)
        self._thread = threading.Thread(target=self._server.serve_forever, name="native-qwen-loopback", daemon=False)
        self._thread.start()

    def profile(self):
        with self._condition:
            if self._phase not in {"running", "paused"} or time.monotonic() >= self._expires:
                raise RuntimeError("native shim is unavailable")
            with self.controller._condition:
                state = self.controller._state
                if state not in {"running", "paused"} or state != self._phase:
                    raise RuntimeError("native policy is not ready at this control boundary")
                policy = self.controller.policy.to_dict()
                ack = dict(self.controller.ack)
                capabilities = self.controller.capabilities()
                active = max(self._active, self.controller._active)
            proof = {"schema_version": 1,
                "serving_profile": getattr(self.controller, "serving_profile", "native-pytorch-v1"),
                "served_model": self.controller.served_model, **linux_process_identity(),
                "model_dir": str(self.controller.model_dir), "generation_slots": 1,
                "started_unix": self.started_unix, "expires_unix": self.expires_unix,
                "observed_unix": time.time(), "policy": policy, "policy_ack": ack,
                "capabilities": capabilities, "controller_state": state, "active_generations": active}
            if proof["serving_profile"] == "vllm-training-v1":
                proof["vllm_engine"] = self.controller.engine_receipt()
            return proof

    def reserve(self, identity, payload):
        with self._condition:
            if self._phase != "running" or self._active or time.monotonic() >= self._expires:
                return False
            write_new(self.output / (identity + ".request.json"),
                      {"id": identity, "payload": copy.deepcopy(payload), "automatic_replay": False})
            self._requests.add(identity)
            self._active += 1
            return True

    def release(self):
        with self._condition:
            self._active -= 1
            self._condition.notify_all()

    def quiesce(self):
        with self._condition:
            self._phase = "quiescing"
            while self._active:
                self._condition.wait()
            return tuple(sorted(self._requests))

    def phase(self, value):
        with self._condition:
            if self._active or value not in {"paused", "loading", "running"}:
                raise RuntimeError("invalid native shim control transition")
            self._phase = value

    def close(self):
        self.quiesce()
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join()
        self._server.server_close()  # Own all handler threads until actual exit.
        with self._condition:
            self._phase = "closed"


class LocalGPURelayDeliveryProbe:
    """Read actual local GPU relay API and durable ACK records, never fabricate drain."""
    def __init__(self, relay_root, *, relay_port, expected_model):
        self.root, self.port, self.model = Path(relay_root), relay_port, expected_model
        if type(relay_port) is not int or not 1024 <= relay_port <= 65535:
            raise ValueError("expected an owned loopback GPU relay port")
        token = read_private(self.root / "api-token.txt", 256).decode().strip()
        if not TOKEN.fullmatch(token):
            raise ValueError("invalid private GPU relay token")
        self._token, self.boot_id = token, None

    def _get(self, route):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        try:
            connection.request("GET", route, headers={"X-Relay-Token": self._token})
            response = connection.getresponse()
            data = response.read(MAX_RESPONSE + 1)
            if response.status != 200 or len(data) > MAX_RESPONSE:
                raise RuntimeError("local GPU relay receipt unavailable")
            return strict_json(data)
        finally:
            connection.close()

    def bind(self):
        value = self._get("/api/config")
        if (value.get("role") != "gpu" or value.get("model") != self.model
                or not isinstance(value.get("boot_id"), str) or not ID.fullmatch(value["boot_id"])):
            raise RuntimeError("local relay process/model cannot be bound")
        self.boot_id = value["boot_id"]
        return value

    def wait(self, identities, timeout=30):
        deadline = time.monotonic() + timeout
        while True:
            value = self._get("/api/status")
            if value.get("role") != "gpu" or value.get("boot_id") != self.boot_id:
                raise RuntimeError("GPU relay boot changed; no activation")
            if type(value.get("uncertain")) is not int or value["uncertain"] != 0:
                raise RuntimeError("GPU relay has uncertain requests; no activation")
            ready = all(type(value.get(key)) is int and value[key] == 0
                        for key in ("active", "active_generations", "retained_unacknowledged"))
            acknowledged = value.get("acknowledged_ids")
            if not isinstance(acknowledged, list) or any(not isinstance(item, str) or not ID.fullmatch(item) for item in acknowledged):
                raise RuntimeError("GPU relay lacks durable delivery acknowledgement IDs")
            if ready and set(identities) <= set(acknowledged):
                for identity in identities:
                    record = strict_json(read_private(self.root / "requests" / (identity + ".json"), 65536))
                    if (record.get("id") != identity or record.get("boot_id") != self.boot_id
                            or record.get("status") != "finished" or record.get("delivery_acknowledged") is not True):
                        raise RuntimeError("CPU-commit/browser-completion ACK chain is unproven")
                return {"delivery_drain_complete": True, "relay_status": value,
                        "acknowledged_request_ids": list(identities)}
            if time.monotonic() >= deadline:
                raise TimeoutError("actual browser delivery drain timed out; no activation")
            time.sleep(.05)


async def _owned_call(function, *args):
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
        task.result()
        raise


class BrowserRelayPolicyController:
    def __init__(self, controller, shim, delivery_probe):
        if shim.controller is not controller:
            raise ValueError("loopback must share the actual policy controller")
        self.controller, self.shim, self.delivery = controller, shim, delivery_probe
        self._paused = False
        self._error = None

    def policy_factory(self, artifact):
        return self.controller.policy_factory(artifact)

    def capabilities(self):
        if self._error:
            raise RuntimeError("browser delivery is unresolved") from self._error
        return self.controller.capabilities()

    async def validate_group_delivery(self, group):
        """CPU sealing cannot substitute for the local generation/ACK chain."""
        identities = [turn.request_id for sample in group.samples for turn in sample.turns]
        if len(identities) != len(set(identities)):
            raise ValueError("sealed group repeats a sampled request identity")
        with self.shim._condition:
            if not set(identities) <= self.shim._requests:
                raise ValueError("sealed group contains requests absent from the owned native shim")
        for sample in group.samples:
            for turn in sample.turns:
                observed = strict_json(read_private(self.shim.output / (turn.request_id + ".training.json"), MAX_RESPONSE))
                # Local serving knows request identity; CPU owns episode turn order.
                expected = turn.to_dict()
                observed.pop("turn_id", None)
                expected.pop("turn_id", None)
                if observed != expected:
                    raise ValueError("sealed turn differs from the actual local native training evidence")
        if self.delivery.boot_id is None:
            await _owned_call(self.delivery.bind)
        return await _owned_call(self.delivery.wait, identities)

    async def start(self, artifact):
        ack = await self.controller.start(artifact)
        try:
            await _owned_call(self.shim.start)
        except BaseException:
            await _owned_call(self.shim.close)
            await self.controller.close()
            raise
        return ack

    async def pause(self):
        self._paused = False
        try:
            identities = await _owned_call(self.shim.quiesce)
            ack = await self.controller.pause()
            if self.delivery.boot_id is None:
                await _owned_call(self.delivery.bind)
            proof = await _owned_call(self.delivery.wait, identities)
            write_new(self.shim.output / ("delivery-drain-" + secrets.token_hex(8) + ".json"), proof)
            self.shim.phase("paused")
            self._paused = True
            return {**ack, **proof}
        except BaseException as error:
            self._error = error
            raise

    async def activate(self, artifact):
        if not self._paused or self._error:
            raise RuntimeError("policy activation requires sampler and browser delivery drain")
        self._paused = False
        self.shim.phase("loading")
        try:
            ack = await self.controller.activate(artifact)
            self.shim.phase("paused")
            self._paused = True
            return ack
        except BaseException as error:
            self._error = error
            raise

    async def resume(self):
        if not self._paused or self._error:
            raise RuntimeError("resume requires an unfailed paused browser route")
        ack = await self.controller.resume()
        self.shim.phase("running")
        self._paused = False
        return ack

    async def close(self):
        await _owned_call(self.shim.close)
        return await self.controller.close()
