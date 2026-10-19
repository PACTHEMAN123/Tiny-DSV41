from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr
import io
import socket
import struct
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from dsv41_train.rl.lora_config import DistributedLoRAConfig
from rollout_manager.backends import vllm_lora_remote as rpc
from rollout_manager.backends.vllm_lora_transport import policy_ack
from test_deepseek_vllm_transport import update


class FakeController:
    def __init__(self):
        self.config = SimpleNamespace(lora_config=DistributedLoRAConfig())
        self._state, self.ack, self.activation_receipts = "new", None, None
        self.base_identity = {"config_sha256": "a" * 64, "index_sha256": "b" * 64}
        self.calls, self.updates, self.timeout_checks = [], [], 0
        self.entered, self.release = threading.Event(), threading.Event()
        self.finished, self.close_entered = threading.Event(), threading.Event()
        self.finished.set()
        self.release.set()
        self.fail_sample = False

    def _timeout(self):
        self.timeout_checks += 1
        return 600

    async def start(self):
        self.calls.append("start")
        self._state = "paused"
        return self.engine_receipt()

    def engine_receipt(self):
        return {"workers": [{"device_uuid": "GPU-unit"}], "loaded_policy_ack": self.ack}

    async def activate(self, manifest, payload):
        self.updates.append((manifest, payload))
        self.ack = policy_ack(manifest)
        self.activation_receipts = [{"policy_ack": self.ack, "readback_digest": manifest["digest"]}]
        return self.ack

    async def resume(self):
        self._state = "running"
        return self.ack

    async def pause(self):
        self._state = "paused"
        return {"drain_complete": True, "policy_ack": self.ack}

    def sample_batch(self, specs):
        self.calls.append([spec["request_id"] for spec in specs])
        self.entered.set()
        self.finished.clear()
        try:
            if not self.release.wait(timeout=3):
                raise TimeoutError("fixture release timed out")
            if self.fail_sample:
                raise ValueError("actual sampled token lacks old logprob")
            return [{"context_token_ids": spec["prompt_ids"], "action_token_ids": [3, 1],
                    "behavior_log_probs": [-.125, -.25], "stop_token_count": 1, "finish_reason": "stop",
                    "logprob_convention": "sampled_distribution", "request_id": spec["request_id"],
                    "generation_metrics": {"policy_ack": self.ack, "engine_receipt": self.engine_receipt()}}
                    for spec in specs]
        finally:
            self.finished.set()

    async def close(self):
        self._state = "closing"
        self.close_entered.set()
        if not await asyncio.to_thread(self.finished.wait, 3):
            raise TimeoutError("fixture actor did not drain")
        self._state = "closed"
        return {"closed": True, "drain_complete": True}


def specs(count=1):
    return [{"prompt_ids": [0, 4], "max_tokens": 2, "temperature": .8,
             "request_id": f"{index:032x}", "sampling_seed": index} for index in range(count)]


class RemoteLoRATests(unittest.TestCase):
    def setUp(self):
        self.controller, self.token = FakeController(), "unit-secret-" + "a" * 32
        self.server = rpc.RemoteLoRAServer(("127.0.0.1", 0), self.token, self.controller, io_timeout=3)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
        self.thread.start()
        self.client = self.new_client()
        self.errors = io.StringIO()
        self.redirect = redirect_stderr(self.errors)
        self.redirect.__enter__()

    def new_client(self, **kwargs):
        return rpc.RemoteLoRAClient(*self.server.server_address, self.token, timeout=kwargs.get("timeout", 3))

    def tearDown(self):
        self.controller.release.set()
        self.server.shutdown()
        self.thread.join(timeout=3)
        self.server.server_close()
        self.redirect.__exit__(None, None, None)

    def activate(self):
        initialized = self.client.initialize()
        self.assertEqual(initialized["lora_config"], {"rank": 8, "alpha": 8.0, "layers": None})
        manifest, payload, _ = update(base_identity=self.controller.base_identity)
        receipt = self.client.activate(manifest, payload)
        self.assertEqual(receipt["ack"], policy_ack(manifest))
        self.assertEqual(receipt["activation_receipts"], self.controller.activation_receipts)
        self.assertEqual(self.controller.updates, [(manifest, payload)])
        self.client.resume()

    def test_matching_custom_configuration_is_acknowledged(self):
        profile = DistributedLoRAConfig(rank=16, alpha=32, layers=(0, 2))
        self.controller.config.lora_config = profile
        initialized = self.client.initialize(lora_config=profile.to_dict())
        self.assertEqual(initialized["lora_config"], profile.to_dict())
        self.assertEqual(self.controller.calls, ["start"])

    def test_configuration_mismatch_is_rejected_before_actor_start(self):
        profile = DistributedLoRAConfig(rank=16, alpha=32, layers=(0, 2))
        with self.assertRaisesRegex(RuntimeError, "LoRA configurations differ"):
            self.client.initialize(lora_config=profile.to_dict())
        self.assertEqual(self.controller.calls, [])
        self.assertEqual(self.controller._state, "new")
        self.assertTrue(self.client.failed)
        self.assertTrue(self.server.failed)

    def test_missing_or_inconsistent_initialize_acknowledgement_latches_failure(self):
        for value in (None, {"rank": 16, "alpha": 8, "layers": None},
                      {"rank": 8, "alpha": 8}):
            with self.subTest(config=value):
                result = {"base_identity": self.controller.base_identity}
                if value is not None:
                    result["lora_config"] = value
                client = self.new_client()
                with patch.object(client, "_call", return_value=result):
                    with self.assertRaisesRegex(RuntimeError, "invalid LoRA configuration"):
                        client.initialize()
                self.assertTrue(client.failed)
                self.assertIsNone(client.base_identity)

    def test_real_socket_tensor_activation_group32_and_concurrent_status(self):
        self.activate()
        self.controller.release.clear()
        with ThreadPoolExecutor(max_workers=1) as pool:
            group = pool.submit(self.client.sample_group, specs(32))
            self.assertTrue(self.controller.entered.wait(timeout=2))
            status = self.new_client().status()
            self.assertEqual(status["state"], "running")
            self.assertFalse(group.done())
            self.controller.release.set()
            results = group.result(timeout=2)
        self.assertEqual([item["request_id"] for item in results], [item["request_id"] for item in specs(32)])
        self.assertEqual(results[0]["behavior_log_probs"], [-.125, -.25])
        self.assertEqual(self.controller.calls, ["start", [spec["request_id"] for spec in specs(32)]])
        self.assertEqual(results[0]["stop_token_count"], 1)
        self.assertEqual(results[0]["generation_metrics"]["policy_ack"], self.client.ack)
        self.assertTrue(self.client.pause()["drain_complete"])
        self.assertTrue(self.client.close()["closed"])

    def test_bad_auth_cannot_receive_tensor_or_mutate_actor(self):
        other = rpc.RemoteLoRAClient(*self.server.server_address, "b" * 32, timeout=2)
        manifest, payload, _ = update()
        with patch.object(rpc, "receive_update", side_effect=AssertionError("read unauthenticated tensor")) as receive:
            with self.assertRaises(PermissionError):
                other.activate(manifest, payload)
        receive.assert_not_called()
        self.assertEqual(self.controller.calls, [])
        self.assertFalse(self.server.failed)
        self.assertEqual(self.errors.getvalue(), "")

    def test_actual_sample_error_is_reported_and_latches_admission(self):
        self.activate()
        self.controller.fail_sample = True
        with self.assertRaisesRegex(RuntimeError, "ValueError: actual sampled token lacks old logprob"):
            self.client.sample_group(specs())
        with self.assertRaisesRegex(RuntimeError, "no retry"):
            self.client.resume()
        self.assertTrue(self.client.status()["failed"])
        with self.assertRaisesRegex(RuntimeError, "admission is closed"):
            self.new_client().resume()
        record = self.errors.getvalue()
        self.assertIn('LoRA RPC sample_group:', record)
        self.assertIn("ValueError: actual sampled token lacks old logprob", record)
        self.assertNotIn(self.token, record)
        self.assertTrue(self.client.close()["closed"])

    def test_timeout_is_unknown_and_cannot_automatically_replay(self):
        self.activate()
        impatient = self.new_client(timeout=.05)
        self.controller.release.clear()
        with self.assertRaises(TimeoutError):
            impatient.sample_group(specs())
        with self.assertRaisesRegex(RuntimeError, "no retry"):
            impatient.sample_group(specs())
        self.assertEqual(self.controller.calls.count(["0" * 32]), 1)
        self.controller.release.set()

    def test_close_after_socket_timeout_drains_active_request_and_stops_remaining_group(self):
        self.activate()
        impatient = self.new_client(timeout=.1)
        self.controller.release.clear()
        with self.assertRaises(TimeoutError):
            impatient.sample_group(specs(32))
        with self.assertRaisesRegex(RuntimeError, "no retry"):
            impatient.resume()
        with ThreadPoolExecutor(max_workers=1) as pool:
            cleanup = pool.submit(impatient.close)
            self.assertTrue(self.controller.close_entered.wait(timeout=1))
            self.assertEqual(self.controller._state, "closing")
            self.assertFalse(cleanup.done())
            self.controller.release.set()
            self.assertTrue(cleanup.result(timeout=1)["drain_complete"])
        self.assertEqual(self.controller._state, "closed")
        self.assertTrue(impatient.closed)
        self.assertTrue(self.server.failed)
        self.assertEqual([call for call in self.controller.calls if call != "start"], [[spec["request_id"] for spec in specs(32)]])
        self.assertEqual(self.new_client().status()["state"], "closed")

    def test_private_bind_group_budget_and_json_frame_limits(self):
        for host in ("0.0.0.0", "8.8.8.8", "::1"):
            with self.assertRaises(ValueError):
                rpc.RemoteLoRAClient(host, 1234, self.token)
        with self.assertRaises(ValueError):
            self.client.sample_group(specs(33))
        with self.assertRaises(ValueError):
            self.client.sample_group(specs(1) * 2)
        left, right = socket.socketpair()
        with left, right:
            left.sendall(struct.pack("!I", rpc.MAX_JSON_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "byte budget"):
                rpc._receive_json(right)


if __name__ == "__main__":
    unittest.main()
import asyncio
