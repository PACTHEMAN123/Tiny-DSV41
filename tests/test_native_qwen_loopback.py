"""Portable local tests; macOS process metadata is explicitly a test fixture."""
import asyncio
import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import socket
import threading
import time
import unittest
from unittest.mock import patch

from rollout_manager.backends.native_qwen_loopback import (BrowserRelayPolicyController, LocalGPURelayDeliveryProbe,
                                  NativeControllerHTTPShim, encoded, write_new)
from rollout_manager.generation_training import TrainingGenerationContract
from dsv41_train.rl.rollout import RolloutGroup, Sample
import test_pytorch_controller as native_tests


class DeliveryFixture:
    boot_id = "a" * 32
    def __init__(self):
        self.acknowledged = threading.Event()
    def wait(self, identities):
        if not self.acknowledged.wait(5):
            raise TimeoutError("fixture delivery acknowledgement missing")
        return {"delivery_drain_complete": True, "acknowledged_request_ids": list(identities)}


def unused_port():
    with socket.socket() as value:
        value.bind(("127.0.0.1", 0))
        return value.getsockname()[1]


class LoopbackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await native_tests.PyTorchControllerTests.asyncSetUp(self)
        self.native = self.controller
        if not Path("/proc/self/stat").is_file():
            identity = {"model_pid": os.getpid(), "model_pid_start_ticks": 1,
                        "model_cmdline_sha256": hashlib.sha256(b"LOCAL MACOS TEST FIXTURE").hexdigest(),
                        "python_executable": "LOCAL MACOS TEST FIXTURE", "netns_inode": 1}
            self.identity_patch = patch("rollout_manager.backends.native_qwen_loopback.linux_process_identity", return_value=identity)
            self.identity_patch.start()
            self.addCleanup(self.identity_patch.stop)
        self.shim = NativeControllerHTTPShim(self.native, self.root.resolve() / "shim", port=unused_port())
        self.delivery = DeliveryFixture()
        self.wrapper = BrowserRelayPolicyController(self.native, self.shim, self.delivery)
        self.addAsyncCleanup(self.wrapper.close)

    def artifact(self, version):
        return native_tests.PyTorchControllerTests.artifact(self, version)

    def call(self, method, path, body=None, identity=None, token=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.shim.port, timeout=10)
        headers = {"Authorization": "Bearer " + (self.shim.token if token is None else token)}
        if identity is not None:
            headers["X-Rollout-Request-Id"] = identity
        data = None if body is None else encoded(body)
        if data is not None:
            headers["Content-Type"] = "application/json"
        try:
            connection.request(method, path, body=data, headers=headers)
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    async def test_real_native_completion_proof_auth_and_replay(self):
        await self.wrapper.start(self.artifact(0))
        code, profile = await asyncio.to_thread(self.call, "GET", "/rollout/profile")
        self.assertEqual(code, 200)
        self.assertEqual(profile["model_pid"], os.getpid())
        self.assertEqual(profile["policy"], self.native.policy.to_dict())
        self.assertEqual(profile["policy_ack"], self.native.ack)
        self.assertEqual(profile["capabilities"], self.native.capabilities())
        self.assertEqual(profile["generation_slots"], 1)
        self.assertEqual(profile["controller_state"], "running")
        self.assertEqual((await asyncio.to_thread(self.call, "GET", "/rollout/profile", token="wrong"))[0], 401)
        code, completion = await asyncio.to_thread(self.call, "POST", "/v1/chat/completions", self.payload, "b" * 32)
        self.assertEqual(code, 200)
        trace = TrainingGenerationContract(self.native.policy, self.native.capabilities()).validate(
            self.payload, completion, "b" * 32, 1)
        self.assertEqual(trace.policy.version, 0)
        self.assertEqual(self.native._seen, {"b" * 32})
        self.assertEqual((await asyncio.to_thread(self.call, "POST", "/v1/chat/completions", self.payload, "b" * 32))[0], 409)
        self.delivery.acknowledged.set()
        proof = await self.wrapper.pause()
        self.assertTrue(proof["drain_complete"] and proof["delivery_drain_complete"])
        self.assertEqual((await asyncio.to_thread(self.call, "GET", "/rollout/profile"))[1]["controller_state"], "paused")
        await self.wrapper.activate(self.artifact(1))
        await self.wrapper.resume()
        self.assertEqual((await asyncio.to_thread(self.call, "GET", "/rollout/profile"))[1]["policy_ack"]["version"], 1)

    async def test_sampler_and_delivery_both_hold_activation_barrier(self):
        await self.wrapper.start(self.artifact(0))
        entered, release = threading.Event(), threading.Event()
        forward = self.native.model.forward
        def block(*args, **kwargs):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("fixture sampler release missing")
            return forward(*args, **kwargs)
        self.native.model.forward = block
        completion = asyncio.create_task(asyncio.to_thread(self.call, "POST", "/v1/chat/completions", self.payload, "c" * 32))
        self.assertTrue(await asyncio.to_thread(entered.wait, 5))
        self.assertEqual((await asyncio.to_thread(self.call, "GET", "/rollout/profile"))[1]["active_generations"], 1)
        pausing = asyncio.create_task(self.wrapper.pause())
        await asyncio.sleep(.03)
        self.assertFalse(pausing.done())
        release.set()
        self.assertEqual((await completion)[0], 200)
        await asyncio.sleep(.03)
        self.assertFalse(pausing.done())  # Native completion alone is insufficient.
        with self.assertRaises(RuntimeError):
            await self.wrapper.activate(self.artifact(1))
        self.delivery.acknowledged.set()
        self.assertTrue((await pausing)["delivery_drain_complete"])
        self.assertEqual(self.native._active, 0)

    async def test_multicall_contract_keeps_sampled_tokens_and_malformed_diagnostics(self):
        await self.wrapper.start(self.artifact(0))
        # Controlled sampler/text fixtures isolate transport, parsing and raw
        # receipt binding; the real mounted-model probe qualifies generation.
        actions, behavior = [3, 4, 2], [-.8, -.9, -1.]
        def fixture_sample(model, record, *args, **kwargs):
            return RolloutGroup([Sample(record.prompt_ids, actions, behavior, 0., 0, "eos", 1)], 0, 0)
        valid = ''.join('<tool_call>' + json.dumps({"name": "shell", "arguments": {"cmd": command}})
                        + '</tool_call>' for command in ("pwd", "printf READY"))
        identity = "d" * 32
        with patch("dsv41_train.rl.rollout.generate_group", side_effect=fixture_sample), patch.object(
                self.native.tokenizer, "decode", return_value=valid):
            code, completion = await asyncio.to_thread(self.call, "POST", "/v1/chat/completions", self.payload, identity)
        self.assertEqual(code, 200)
        trace = TrainingGenerationContract(self.native.policy, self.native.capabilities()).validate(
            self.payload, completion, identity, 1)
        self.assertEqual(list(trace.action_token_ids), actions)
        self.assertEqual(list(trace.behavior_log_probs), behavior)
        self.assertEqual(trace.finish_reason, "tool_calls")
        self.assertEqual(len({call["id"] for call in completion["choices"][0]["message"]["tool_calls"]}), 2)
        raw = json.loads((self.shim.output / (identity + ".sampled.json")).read_bytes())
        self.assertEqual((raw["decoded_action_text"], raw["action_token_ids"], raw["behavior_log_probs"]),
                         (valid, actions, behavior))
        self.assertFalse(raw["completion_parsed"])
        malformed, bad_id = '<tool_call>{"name":"shell","arguments":{"cmd":"pwd","extra":1}}</tool_call>', "e" * 32
        with patch("dsv41_train.rl.rollout.generate_group", side_effect=fixture_sample), patch.object(
                self.native.tokenizer, "decode", return_value=malformed):
            code, failure = await asyncio.to_thread(self.call, "POST", "/v1/chat/completions", self.payload, bad_id)
        self.assertEqual((code, failure), (500, {"error": "native completion failed; no replay"}))
        preserved = json.loads((self.shim.output / (bad_id + ".sampled.json")).read_bytes())
        self.assertEqual(preserved["decoded_action_text"], malformed)
        diagnostic_path = self.shim.output / (bad_id + ".error.json")
        diagnostic = json.loads(diagnostic_path.read_bytes())
        self.assertEqual(diagnostic["stage"], "shell_completion")
        self.assertEqual(diagnostic["error_type"], "ModelCompletionFormatError")
        self.assertIn("only cmd", diagnostic["error_message"])
        self.assertIn("shell_completion", diagnostic["traceback"])
        self.assertNotIn(self.shim.token, diagnostic_path.read_text())
        self.assertEqual(diagnostic_path.stat().st_mode & 0o777, 0o600)
        self.assertFalse((self.shim.output / (bad_id + ".completion.json")).exists())
        self.assertFalse((self.shim.output / (bad_id + ".training.json")).exists())
        http_state = json.loads((self.shim.output / (bad_id + ".http-state.json")).read_text())
        self.assertFalse(http_state["sampler_drained"])
        self.assertIsNone(http_state["generation_failure"])
        self.assertEqual(self.native._active, 0)
        self.assertEqual((await asyncio.to_thread(self.call, "POST", "/v1/chat/completions", self.payload, bad_id))[0], 409)


class RelayAckProbeTests(unittest.TestCase):
    def test_actual_http_status_and_private_journal_both_required(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            root, identity, boot = Path(temporary), "d" * 32, "e" * 32
            (root / "requests").mkdir(mode=0o700)
            (root / "api-token.txt").write_text("f" * 64 + "\n")
            (root / "api-token.txt").chmod(0o600)
            status = {"role": "gpu", "boot_id": boot, "active": 0, "active_generations": 0,
                      "uncertain": 0, "retained_unacknowledged": 1, "acknowledged_ids": []}
            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *args): pass
                def do_GET(self):
                    value = ({"role": "gpu", "model": "fixture/qwen", "boot_id": boot}
                             if self.path == "/api/config" else status)
                    data = encoded(value)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                probe = LocalGPURelayDeliveryProbe(root, relay_port=server.server_port, expected_model="fixture/qwen")
                probe.bind()
                with self.assertRaises(TimeoutError):
                    probe.wait([identity], timeout=.01)
                status.update(retained_unacknowledged=0, acknowledged_ids=[identity])
                with self.assertRaises(FileNotFoundError):
                    probe.wait([identity], timeout=.01)
                write_new(root / "requests" / (identity + ".json"),
                          {"id": identity, "boot_id": boot, "status": "finished", "delivery_acknowledged": True})
                self.assertTrue(probe.wait([identity])["delivery_drain_complete"])
                status["uncertain"] = 1
                with self.assertRaises(RuntimeError):
                    probe.wait([identity], timeout=.01)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == "__main__":
    unittest.main()
