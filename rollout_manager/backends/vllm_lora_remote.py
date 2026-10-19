"""Private-network LoRA RPC: authenticated JSON and raw tensors, never pickle.

One controller and event loop live for the server lifetime. A client uses one
connection per call and never retries an unknown outcome. Run ``sample_group``
in a background learner thread; join it before pause/activate/resume controls.
"""

from __future__ import annotations
import asyncio
import copy
import hmac
import ipaddress
import json
import math
import re
import socket
import socketserver
import struct
import sys
import threading
from .vllm_lora_transport import _read_exact, receive_update, send_update

MAX_JSON_BYTES = 32 * 1024 * 1024
_SPEC_FIELDS = {"prompt_ids", "max_tokens", "temperature", "request_id", "sampling_seed"}
_OPERATIONS = {"initialize", "status", "activate", "sample_group", "pause", "resume", "close"}


def _endpoint(address):
    host, port = address
    ip = ipaddress.IPv4Address(host)
    private = any(
        (
            ip in ipaddress.ip_network(net)
            for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8")
        )
    )
    if not private or type(port) is not int or (not 0 <= port <= 65535):
        raise ValueError("RPC requires an explicit private or loopback IPv4 address and valid port")
    return (str(ip), port)


def _token(value):
    if not isinstance(value, str) or not re.fullmatch("[A-Za-z0-9_-]{32,256}", value):
        raise ValueError("RPC token requires 32..256 ASCII letters, digits, underscores or hyphens")
    return value


def _send_json(stream, value):
    data = json.dumps(value, separators=(",", ":"), allow_nan=False).encode()
    if not 0 < len(data) <= MAX_JSON_BYTES:
        raise ValueError("RPC JSON exceeds byte budget")
    stream.sendall(struct.pack("!I", len(data)) + data)


def _receive_json(stream, limit=MAX_JSON_BYTES):
    (size,) = struct.unpack("!I", _read_exact(stream, 4))
    if not 0 < size <= limit:
        raise ValueError("RPC JSON exceeds byte budget")

    def reject_constant(value):
        raise ValueError("RPC JSON requires finite values")

    data = _read_exact(stream, size)
    return json.loads(data, parse_constant=reject_constant)


def _specs(values):
    if not isinstance(values, list) or not 1 <= len(values) <= 32:
        raise ValueError("sample_group requires 1..32 request specifications")
    seen = set()
    for spec in values:
        if not isinstance(spec, dict) or set(spec) != _SPEC_FIELDS:
            raise ValueError("invalid sample request fields")
        ids, request = (spec["prompt_ids"], spec["request_id"])
        if (
            not isinstance(ids, list)
            or not 1 <= len(ids) <= 4096
            or any((type(token) is not int or not 0 <= token < 2**31 for token in ids))
            or (type(spec["max_tokens"]) is not int)
            or (not 1 <= spec["max_tokens"] <= 4096)
            or (type(spec["temperature"]) not in (int, float))
            or (not math.isfinite(spec["temperature"]))
            or (not 0 < spec["temperature"] <= 10)
            or (not isinstance(request, str))
            or (not re.fullmatch("[0-9a-f]{32}", request))
            or (request in seen)
            or (type(spec["sampling_seed"]) is not int)
            or (not 0 <= spec["sampling_seed"] < 2**63 - 1)
        ):
            raise ValueError("invalid bounded sample request or duplicate request ID")
        seen.add(request)
    return values


class RemoteLoRAClient:
    """Synchronous client; errors latch failure, with only status/close permitted."""

    def __init__(self, host, port, token, *, timeout=7200):
        self.address, self._token = (_endpoint((host, port)), _token(token))
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or (not 0 < timeout <= 7200)
        ):
            raise ValueError("RPC timeout must be finite and at most 7200 seconds")
        self.timeout, self.failed, self.closed = (timeout, False, False)
        self._lock = threading.Lock()
        self.ack = self.activation_receipts = self.base_identity = None

    def _call(self, operation, arguments=None, update=None):
        if operation not in _OPERATIONS:
            raise ValueError("unknown RPC operation")
        with self._lock:
            if self.closed or (self.failed and operation not in {"status", "close"}):
                raise RuntimeError(
                    "remote client has a closed or unknown prior outcome; no retry is allowed"
                )
            try:
                with socket.create_connection(self.address, timeout=self.timeout) as stream:
                    _send_json(stream, {"token": self._token})
                    if _receive_json(stream, 4096) != {"authenticated": True}:
                        raise PermissionError("remote authentication rejected")
                    _send_json(stream, {"operation": operation, "arguments": arguments or {}})
                    if update is not None:
                        send_update(stream, *update)
                    response = _receive_json(stream)
                if (
                    not isinstance(response, dict)
                    or response.get("ok") is not True
                    or "result" not in response
                ):
                    detail = (
                        "{}: {}".format(
                            response.get("error_type", "UnknownError"),
                            response.get("reason", "outcome unknown"),
                        )
                        if isinstance(response, dict)
                        else "invalid response; outcome unknown"
                    )
                    raise RuntimeError(f"remote {operation} failed: {detail}")
                return response["result"]
            except BaseException:
                self.failed = True
                raise

    def initialize(self):
        result = self._call("initialize")
        self.base_identity = result["base_identity"]
        return result

    def status(self):
        return self._call("status")

    def activate(self, manifest, payload):
        result = self._call("activate", update=(manifest, payload))
        self.ack, self.activation_receipts = (result["ack"], result["activation_receipts"])
        return result

    def sample_group(self, specs):
        return self._call("sample_group", {"specs": _specs(specs)})

    def pause(self):
        return self._call("pause")

    def resume(self):
        return self._call("resume")

    def close(self):
        if self.closed:
            return {"closed": True, "drain_complete": True}
        result = self._call("close")
        self.closed = True
        return result


class _Handler(socketserver.BaseRequestHandler):
    def handle(self):
        authenticated, operation = False, "authentication"
        try:
            self.request.settimeout(5)
            auth = _receive_json(self.request, 4096)
            if (
                not isinstance(auth, dict)
                or set(auth) != {"token"}
                or not isinstance(auth["token"], str)
                or not hmac.compare_digest(auth["token"].encode(), self.server.token.encode())
            ):
                _send_json(self.request, {"authenticated": False})
                return
            _send_json(self.request, {"authenticated": True})
            authenticated = True
            self.request.settimeout(self.server.io_timeout)
            request = _receive_json(self.request)
            if (
                not isinstance(request, dict)
                or set(request) != {"operation", "arguments"}
                or request["operation"] not in _OPERATIONS
                or not isinstance(request["arguments"], dict)
            ):
                raise ValueError("invalid RPC request envelope")
            operation = request["operation"]
            # Never read or deserialize adapter bytes before authentication.
            update = receive_update(self.request) if operation == "activate" else None
            future = asyncio.run_coroutine_threadsafe(
                self.server.dispatch(operation, request["arguments"], update), self.server.loop
            )
            result = future.result(timeout=self.server.io_timeout)
            _send_json(self.request, {"ok": True, "result": result})
        except Exception as error:
            response = {"ok": False, "error": "remote operation rejected or outcome unknown"}
            if authenticated:
                self.server.failed = True
                reason = (str(error) or "RPC timeout or unknown outcome").replace(
                    self.server.token, "<redacted>"
                )[:2048]
                response.update(error_type=type(error).__name__, reason=reason)
                print(
                    f"LoRA RPC {operation}: {type(error).__name__}: {reason}",
                    file=sys.stderr,
                    flush=True,
                )
            try:
                _send_json(self.request, response)
            except OSError:
                pass


class RemoteLoRAServer(socketserver.ThreadingTCPServer):
    """One persistent controller, bounded connections, and one owned event loop."""

    daemon_threads = True
    request_queue_size = 8

    def __init__(self, address, token, controller, *, io_timeout=7200):
        address, self.token = (_endpoint(address), _token(token))
        if (
            type(io_timeout) not in (int, float)
            or not math.isfinite(io_timeout)
            or (not 0 < io_timeout <= 7200)
        ):
            raise ValueError("server RPC timeout must be finite and at most 7200 seconds")
        self.controller, self.io_timeout, self.failed = (controller, io_timeout, False)
        self._slots, self._busy = (threading.BoundedSemaphore(8), False)
        super().__init__(address, _Handler)
        self.loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self._thread.start()

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._slots.release()

    def _status(self):
        state = self.controller._state
        return {
            "state": state,
            "failed": self.failed or state == "failed",
            "policy_ack": copy.deepcopy(self.controller.ack),
            "base_identity": copy.deepcopy(self.controller.base_identity),
            "activation_receipts": copy.deepcopy(self.controller.activation_receipts),
            "engine_receipt": (
                self.controller.engine_receipt()
                if not self.failed and state in {"paused", "pausing", "running"}
                else None
            ),
        }

    async def dispatch(self, operation, arguments, update):
        if operation == "status" and (not arguments):
            return self._status()
        if operation == "close" and (not arguments):
            self.failed = True
            return await self.controller.close()
        if self._busy:
            raise RuntimeError("join the active group or control before issuing another operation")
        if self.failed and operation != "close":
            raise RuntimeError("a previous remote outcome failed; further admission is closed")
        self._busy = True
        try:
            if operation != "close":
                self.controller._timeout()
            if operation == "sample_group" and set(arguments) == {"specs"}:
                return await asyncio.to_thread(
                    self.controller.sample_batch, _specs(arguments["specs"])
                )
            if arguments:
                raise ValueError("this RPC operation accepts no JSON arguments")
            if operation == "initialize":
                await self.controller.start()
                return self._status()
            if operation == "activate" and update is not None:
                ack = await self.controller.activate(*update)
                return {
                    "ack": ack,
                    "activation_receipts": copy.deepcopy(self.controller.activation_receipts),
                    "engine_receipt": self.controller.engine_receipt(),
                }
            if operation in {"pause", "resume", "close"}:
                return await getattr(self.controller, operation)()
            raise ValueError("unknown RPC operation")
        except BaseException:
            self.failed = True
            raise
        finally:
            self._busy = False

    def server_close(self):
        super().server_close()
        if hasattr(self, "loop") and self.loop.is_running():
            self.failed = True
            try:
                asyncio.run_coroutine_threadsafe(self.controller.close(), self.loop).result(
                    timeout=self.io_timeout
                )
            finally:
                self.loop.call_soon_threadsafe(self.loop.stop)
                self._thread.join(timeout=5)
                self.loop.close()
