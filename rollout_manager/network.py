"""Bounded, authenticated experiment relay and interval-based RPC measurements."""
from __future__ import annotations

import argparse
import hmac
import hashlib
import http.client
import http.server
import json
import re
import shutil
import socket
import ssl
import struct
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .common import MAX_BODY, MODEL, save_json
from .common import digest


def generation_completion(result):
    """Return the untouched model envelope, before native candidate parsing.

    The existing relay wraps completions in ``response``. Training evidence must
    originate inside that completion; this client does not synthesize receipts.
    """
    if not isinstance(result, dict) or not isinstance(result.get("response"), dict):
        raise ValueError("Native relay returned no raw model completion")
    return result["response"]


def tcp_info(sock):
    if not hasattr(socket, "TCP_INFO") or sock is None:
        return {}
    data = sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_INFO, 104)
    if len(data) < 104:
        return {}
    values = struct.unpack_from("=24I", data, 8)
    return {"rtt_us": values[15], "rttvar_us": values[16], "snd_cwnd": values[18],
            "total_retrans": values[23], "unacked": values[4], "snd_mss": values[2]}


class RpcClient:
    """One client per thread; persistent TLS connections, no automatic replay."""
    def __init__(self, url: str, credentials: Path, timeout=900, max_idle_seconds=600):
        if not 0 < timeout <= 7200:
            raise ValueError("RPC timeout must be positive and at most two hours")
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme != "https" or parsed.path not in ("", "/"):
            raise ValueError("a TLS relay origin is required")
        self.host, self.port = parsed.hostname, parsed.port or 443
        self.context = ssl.create_default_context(cafile=str(credentials / "cert.pem"))
        self.token = (credentials / "token").read_text().strip()
        self.timeout = timeout
        if not 0 < max_idle_seconds < 910:
            raise ValueError("client idle limit must be below the relay's 910-second idle timeout")
        self.max_idle_seconds = max_idle_seconds
        self.connection = None
        self.last_response_ns = None

    def close(self):
        if self.connection:
            self.connection.close()
        self.connection = None
        self.last_response_ns = None

    def call(self, path: str, payload: dict) -> dict:
        serialization_start = time.monotonic_ns()
        encoded = json.dumps(payload, separators=(",", ":")).encode()
        serialization_seconds = (time.monotonic_ns() - serialization_start) / 1e9
        if len(encoded) > MAX_BODY:
            raise ValueError("RPC request exceeds bound")
        start = time.monotonic_ns()
        unix_start = time.time_ns()
        setup = 0
        idle = (start - self.last_response_ns) / 1e9 if self.last_response_ns is not None else 0
        refreshed = self.connection is not None and idle >= self.max_idle_seconds
        if refreshed or (self.connection is not None and self.connection.sock is None):
            self.close()
        phase, dispatch_possible, http_status = "connect", False, None
        try:
            if self.connection is None:
                self.connection = http.client.HTTPSConnection(self.host, self.port,
                    context=self.context, timeout=self.timeout)
                self.connection.connect()
                self.connection.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                setup = time.monotonic_ns() - start
            before = tcp_info(self.connection.sock)
            send_start = time.monotonic_ns()
            phase, dispatch_possible = "send", True
            self.connection.request("POST", path, body=encoded,
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + self.token})
            sent = time.monotonic_ns()
            phase = "response_headers"
            response = self.connection.getresponse()
            http_status = response.status
            headers = time.monotonic_ns()
            phase = "response_body"
            body = response.read(MAX_BODY * 3 + 1)
            received = time.monotonic_ns()
            received_unix = time.time_ns()
            after = tcp_info(self.connection.sock)
            if len(body) > MAX_BODY * 3 or response.status != 200:
                raise RuntimeError(f"relay HTTP {response.status}, bytes={len(body)}")
            phase = "decode"
            result = json.loads(body)
            server_ns = int(response.getheader("X-Server-Processing-Ns", "0"))
            result["communication"] = {
                "client_total_seconds": (received - start) / 1e9,
                "client_started_unix_ns": unix_start,
                "client_finished_unix_ns": time.time_ns(),
                "client_received_unix_ns": received_unix,
                "client_monotonic_ns": {"start": start, "send_start": send_start,
                    "sent": sent, "headers": headers, "received": received},
                "serialization_seconds": serialization_seconds,
                "server_received_unix_ns": result.get("server_received_unix_ns"),
                "server_response_unix_ns": result.get("server_response_unix_ns"),
                "connect_tls_seconds": setup / 1e9,
                "connection_idle_seconds": idle,
                "idle_connection_refreshed": refreshed,
                "max_idle_seconds": self.max_idle_seconds,
                "send_seconds": (sent - send_start) / 1e9,
                "response_headers_seconds": (headers - start) / 1e9,
                "response_body_seconds": (received - headers) / 1e9,
                "server_processing_seconds": server_ns / 1e9,
                "residual_seconds": (received - start - server_ns) / 1e9,
                "residual_definition": "client interval minus server interval; includes connection setup, network, dispatch and HTTP overhead; not one-way latency",
                "request_body_bytes": len(encoded), "response_body_bytes": len(body),
                "tcp_before": before, "tcp_after": after,
                "tcp_setup_retransmissions": before.get("total_retrans", 0) if setup else 0,
                "tcp_retrans_delta": after.get("total_retrans", 0) - before.get("total_retrans", 0),
            }
            self.last_response_ns = received
            return result
        except BaseException as exc:
            try:
                counters = tcp_info(self.connection.sock if self.connection else None)
            except OSError:
                counters = {}
            exc.rpc_failure = {"phase": phase, "dispatch_may_have_occurred": dispatch_possible,
                "automatic_replay": False, "client_started_unix_ns": unix_start,
                "client_finished_unix_ns": time.time_ns(),
                "client_elapsed_seconds": (time.monotonic_ns() - start) / 1e9,
                "http_status": http_status,
                "connect_tls_seconds": setup / 1e9, "connection_idle_seconds": idle,
                "idle_connection_refreshed": refreshed, "tcp_at_failure": counters}
            self.close()
            raise


def validate_generation(payload, workload="pilot"):
    fields = {"model", "messages", "temperature", "max_tokens", "seed", "chat_template_kwargs"}
    if workload == "grpo":
        fields.add("top_p")
    if not isinstance(payload, dict) or set(payload) - fields or payload.get("model") != MODEL:
        raise ValueError("request outside experiment policy")
    temperature = payload.get("temperature")
    tokens = payload.get("max_tokens")
    if type(tokens) is not int or not 1 <= tokens <= (8192 if workload == "grpo" else 4096):
        raise ValueError("generation token budget outside experiment policy")
    if workload == "grpo":
        if (type(temperature) not in (int, float) or not 0 <= temperature <= 1
                or type(payload.get("top_p", 1)) not in (int, float)
                or not 0 < payload.get("top_p", 1) <= 1
                or type(payload.get("seed")) is not int or not 0 <= payload["seed"] < 2**32):
            raise ValueError("GRPO sampling outside experiment policy")
    elif temperature != 0:
        raise ValueError("pilot policy requires greedy sampling")


def serve(args):
    token = (args.credentials / "token").read_text().strip()
    args.results.mkdir(parents=True, exist_ok=False)
    record_lock = threading.Lock()
    connections = threading.BoundedSemaphore(args.max_connections)
    asset = getattr(args, "asset", None)
    asset_sha = hashlib.sha256(asset.read_bytes()).hexdigest() if asset else None
    handshake_mode = getattr(args, "tls_handshake", "worker")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(args.credentials / "cert.pem", args.credentials / "key.pem")

    class Server(http.server.ThreadingHTTPServer):
        daemon_threads = True
        request_queue_size = 256

        def process_request(self, request, address):
            if address[0] != args.peer:
                request.close()
                return
            if not connections.acquire(blocking=False):
                request.close()
                return
            try:
                super().process_request(request, address)
            except BaseException:
                connections.release()
                raise

        def process_request_thread(self, request, address):
            try:
                if handshake_mode == "worker":
                    # A slow TLS client must not serialize the listener's accept loop.
                    request.settimeout(10)
                    request = context.wrap_socket(request, server_side=True)
                super().process_request_thread(request, address)
            except (OSError, ssl.SSLError):
                request.close()
            finally:
                connections.release()

    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *unused):
            pass

        def do_POST(self):
            start = time.monotonic_ns()
            received_unix = time.time_ns()
            record, result = None, None
            self.connection.settimeout(910)
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if self.client_address[0] != args.peer or not hmac.compare_digest(
                    self.headers.get("Authorization", ""), "Bearer " + token):
                self.send_error(403)
                return
            match = re.fullmatch(r"/generate/([a-zA-Z0-9_-]{1,100})", self.path)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_BODY or self.headers.get("Transfer-Encoding"):
                    raise ValueError("invalid body")
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("incomplete body")
                payload = json.loads(body)
                if self.path == "/asset" and asset is not None and payload == {"sha256": asset_sha}:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(asset.stat().st_size))
                    self.end_headers()
                    with asset.open("rb") as source:
                        shutil.copyfileobj(source, self.wfile, 1024 * 1024)
                    return
                elif self.path == "/echo":
                    result = {"echo": payload}
                elif match:
                    validate_generation(payload, getattr(args, "workload", "pilot"))
                    record = args.results / (match[1] + ".json")
                    with record_lock:
                        if record.exists():
                            self.send_error(409, "Request ID already used; no implicit replay")
                            return
                        result = {"request": payload, "request_sha256": digest(payload),
                                  "started_unix": time.time(), "status": "uncertain",
                                  "generation_timeout_seconds": getattr(args, "generation_timeout", 900)}
                        save_json(record, result)
                    upstream_start = time.monotonic_ns()
                    result["upstream_started_unix_ns"] = time.time_ns()
                    result["upstream_started_monotonic_ns"] = upstream_start
                    request = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions",
                        body, {"Content-Type": "application/json"})
                    with urllib.request.urlopen(request, timeout=getattr(args, "generation_timeout", 900)) as response:
                        result["upstream_headers_monotonic_ns"] = time.monotonic_ns()
                        content = response.read(MAX_BODY + 1)
                    result["upstream_finished_monotonic_ns"] = time.monotonic_ns()
                    result["upstream_finished_unix_ns"] = time.time_ns()
                    if len(content) > MAX_BODY:
                        raise ValueError("model response exceeds bound")
                    result.update(response=json.loads(content), status="finished",
                        wall_seconds=(time.monotonic_ns() - upstream_start) / 1e9)
                    save_json(record, result)
                else:
                    self.send_error(404)
                    return
                result["server_response_unix_ns"] = time.time_ns()
                result["server_received_unix_ns"] = received_unix
                result["server_received_monotonic_ns"] = start
                result["server_response_monotonic_ns"] = time.monotonic_ns()
                encoded = json.dumps(result, separators=(",", ":")).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.send_header("X-Server-Processing-Ns", str(time.monotonic_ns() - start))
                self.end_headers()
                self.wfile.write(encoded)
            except Exception as exc:
                if record is not None and result is not None and result.get("status") == "uncertain":
                    result.update(error_type=type(exc).__name__, failed_unix=time.time(),
                                  automatic_replay=False, upstream_completion_unknown=True)
                    save_json(record, result)
                print(json.dumps({"error_type": type(exc).__name__, "path": self.path}), flush=True)
                self.send_error(502, "Experiment request failed; no automatic replay")

    server = Server((args.bind, args.port), Handler)
    if handshake_mode == "acceptor":
        server.socket = context.wrap_socket(server.socket, server_side=True)
    server.timeout = 1
    print(json.dumps({"pid": __import__("os").getpid(), "bind": args.bind, "port": args.port,
                      "peer": args.peer, "ttl": args.ttl, "tls_handshake": handshake_mode,
                      "generation_timeout_seconds": getattr(args, "generation_timeout", 900),
                      "workload": getattr(args, "workload", "pilot")}), flush=True)
    deadline = time.monotonic() + args.ttl
    try:
        while time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", required=True)
    parser.add_argument("--peer", required=True)
    parser.add_argument("--port", type=int, default=18443)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--max-connections", type=int, default=128)
    parser.add_argument("--ttl", type=int, default=21600)
    parser.add_argument("--generation-timeout", type=int, default=900,
                        help="upstream generation deadline; the client deadline should be longer")
    parser.add_argument("--workload", choices=["pilot", "grpo"], default="pilot",
                        help="grpo admits bounded stochastic sampling and up to 8192 output tokens")
    parser.add_argument("--tls-handshake", choices=["acceptor", "worker"], default="worker",
                        help="acceptor reproduces the initial experiment; worker prevents listener blocking")
    parser.add_argument("--asset", type=Path, help="optional fixed setup archive; same peer and token restrictions")
    args = parser.parse_args()
    if not 1 <= args.max_connections <= 2048 or not 1 <= args.ttl <= 21600:
        parser.error("connections <=2048 and duration <=6h required")
    if not 1 <= args.generation_timeout <= 3600:
        parser.error("generation timeout must be between one second and one hour")
    serve(args)


if __name__ == "__main__":
    main()
