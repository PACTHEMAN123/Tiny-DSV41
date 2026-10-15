import argparse
import concurrent.futures
import http.client
import json
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from rollout_manager.network import RpcClient, serve, validate_generation
from rollout_manager.common import MODEL


class RelayIntegrationTests(unittest.TestCase):
    def test_grpo_sampling_is_explicit_and_bounded(self):
        request = {"model": MODEL, "messages": [], "temperature": .8,
                   "top_p": .95, "seed": 42, "max_tokens": 8192}
        validate_generation(request, "grpo")
        with self.assertRaises(ValueError):
            validate_generation(request)
        for change in ({"temperature": float("nan")}, {"top_p": 0}, {"max_tokens": 8193},
                       {"seed": -1}, {"seed": True}, {"model": "other"}, {"n": 8}):
            with self.assertRaises(ValueError):
                validate_generation(request | change, "grpo")

    def test_tls_auth_concurrency_measurements_and_duplicate_fencing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "token").write_text("local-integration-test")
            subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                "-keyout", str(root / "key.pem"), "-out", str(root / "cert.pem"),
                "-subj", "/CN=localhost", "-addext", "subjectAltName=IP:127.0.0.1"],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            args = argparse.Namespace(credentials=root, results=root / "records", bind="127.0.0.1",
                port=port, peer="127.0.0.1", max_connections=16, ttl=4, generation_timeout=17)
            server = threading.Thread(target=serve, args=(args,), daemon=True)
            server.start()
            for _ in range(100):
                if args.results.exists():
                    break
                time.sleep(.01)
            stalled = None
            client = RpcClient(f"https://127.0.0.1:{port}", root, timeout=2)
            try:
                for _ in range(100):
                    try:
                        echoed = client.call("/echo", {"data": "payload"})
                        break
                    except ConnectionRefusedError:
                        time.sleep(.01)
                self.assertEqual(echoed["echo"], {"data": "payload"})
                self.assertGreater(echoed["communication"]["client_total_seconds"], 0)
                timing = echoed["communication"]
                self.assertLessEqual(timing["client_started_unix_ns"], timing["server_received_unix_ns"])
                self.assertLessEqual(timing["server_received_unix_ns"], timing["server_response_unix_ns"])
                self.assertLessEqual(timing["server_response_unix_ns"], timing["client_received_unix_ns"])
                bounds = list(timing["client_monotonic_ns"].values())
                self.assertEqual(bounds, sorted(bounds))
                self.assertAlmostEqual((bounds[-1]-bounds[0])/1e9, timing["client_total_seconds"])
                reused = client.call("/echo", {"data": "again"})
                self.assertEqual(reused["communication"]["connect_tls_seconds"], 0)
                client.last_response_ns -= 601 * 10 ** 9
                renewed = client.call("/echo", {"data": "after-tool-queue"})
                self.assertEqual(renewed["echo"], {"data": "after-tool-queue"})
                self.assertTrue(renewed["communication"]["idle_connection_refreshed"])
                self.assertGreater(renewed["communication"]["connect_tls_seconds"], 0)
                with patch.object(client.connection, "getresponse", side_effect=http.client.RemoteDisconnected), \
                        patch.object(client.connection, "request", wraps=client.connection.request) as dispatch:
                    with self.assertRaises(http.client.RemoteDisconnected) as failure:
                        client.call("/echo", {"uncertain-response": True})
                    self.assertEqual(dispatch.call_count, 1)
                self.assertEqual(failure.exception.rpc_failure["phase"], "response_headers")
                self.assertTrue(failure.exception.rpc_failure["dispatch_may_have_occurred"])
                self.assertFalse(failure.exception.rpc_failure["automatic_replay"])
                self.assertIsNone(client.connection)
                self.assertIsNone(client.last_response_ns)
                stalled = socket.create_connection(("127.0.0.1", port), timeout=2)
                another = RpcClient(f"https://127.0.0.1:{port}", root, timeout=2)
                try:
                    self.assertEqual(another.call("/echo", {"slow_peer": True})["echo"], {"slow_peer": True})
                finally:
                    another.close()

                class Upstream:
                    def __enter__(self):
                        return self
                    def __exit__(self, *args):
                        pass
                    def read(self, limit):
                        time.sleep(.15)
                        return json.dumps({"choices": [{"message": {"content": "test"}}]}).encode()

                payload = {"model": MODEL, "temperature": 0, "max_tokens": 8, "messages": []}
                def request(index):
                    other = RpcClient(f"https://127.0.0.1:{port}", root)
                    try:
                        return other.call(f"/generate/parallel-{index}", payload)
                    finally:
                        other.close()

                with patch("rollout_manager.network.urllib.request.urlopen", return_value=Upstream()) as upstream:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        values = list(executor.map(request, range(4)))
                    self.assertTrue(all(call.kwargs["timeout"] == 17 for call in upstream.call_args_list))
                intervals = sorted((v["started_unix"], v["started_unix"] + v["wall_seconds"]) for v in values)
                self.assertLess(intervals[-1][0], intervals[0][1])
                with self.assertRaisesRegex(RuntimeError, "409"):
                    client.call("/generate/parallel-0", payload)
                with patch("rollout_manager.network.urllib.request.urlopen", side_effect=TimeoutError):
                    with self.assertRaisesRegex(RuntimeError, "502"):
                        client.call("/generate/timeout", payload)
                uncertain = json.loads((args.results / "timeout.json").read_text())
                self.assertEqual(uncertain["status"], "uncertain")
                self.assertEqual(uncertain["error_type"], "TimeoutError")
                self.assertEqual(uncertain["generation_timeout_seconds"], 17)
                self.assertFalse(uncertain["automatic_replay"])
                with self.assertRaisesRegex(RuntimeError, "409"):
                    client.call("/generate/timeout", payload)
                client.token = "wrong-token"
                with self.assertRaisesRegex(RuntimeError, "403"):
                    client.call("/echo", {})
            finally:
                client.close()
                if stalled is not None:
                    stalled.close()
                server.join(timeout=6)
            self.assertFalse(server.is_alive())
