"""Bound native relay ownership and transport evidence; never claims live browser qualification."""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from rollout_manager.terminal_bench import controller
from rollout_manager.terminal_bench.browser_relay import relay_server as relay
from .test_native_profile import MODEL, proof


def live_proof():
    now = time.time()
    value = proof(now)
    value.update(started_unix=now - 10, expires_unix=now + 12000)
    return value


class OwnedNativeTests(unittest.TestCase):
    def test_authenticated_profile_is_bound_to_exact_owned_python_listener(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            value = live_proof()
            command = (value["python_executable"] + "\0/workspace/native_helper.py\0").encode()
            value["model_cmdline_sha256"] = hashlib.sha256(command).hexdigest()
            receipt, token = root / "receipt.json", root / "token.txt"
            controller.write_new(receipt, controller.native_controller_identity(value, include_policy=False))
            token.write_text("f" * 64 + "\n")
            token.chmod(0o600)
            proc = root / "proc"
            owner = proc / str(value["model_pid"])
            for path in (owner / "ns", owner / "fd", owner / "net", proc / "self/ns"):
                path.mkdir(parents=True)
            (owner / "ns/net").symlink_to("net:[77]")
            (proc / "self/ns/net").symlink_to("net:[77]")
            (owner / "stat").write_text("123 (python) " + " ".join(["S"] + ["0"] * 18 + ["456"]))
            (owner / "cmdline").write_bytes(command)
            (owner / "net/tcp").write_text("header\n0: 0100007F:1F42 00000000:0000 0A 0 0 0 0 0 12345\n")
            (owner / "fd/3").symlink_to("socket:[12345]")
            connection = Mock()
            response = connection.getresponse.return_value
            response.status, response.length = 200, 0
            response.read.return_value = json.dumps(value).encode()
            with patch.object(relay, "open_upstream_connection", return_value=connection) as opener:
                actual = relay.owned_native_controller(receipt, token, MODEL, proc_root=proc)
            self.assertEqual(actual, value)
            opener.assert_called_once_with(5, 8002)
            self.assertEqual(connection.request.call_args.args, ("GET", "/rollout/profile"))
            self.assertEqual(connection.request.call_args.kwargs["headers"]["Authorization"], "Bearer " + "f" * 64)
            (owner / "fd/3").unlink()
            with patch.object(relay, "open_upstream_connection", return_value=connection), self.assertRaisesRegex(ValueError, "listener"):
                relay.owned_native_controller(receipt, token, MODEL, proc_root=proc)
            token.chmod(0o644)
            with self.assertRaisesRegex(ValueError, "private"):
                relay.native_controller_token(token)

    def test_native_startup_receipt_change_stops_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = live_proof()
            with patch.object(relay, "owned_native_controller", return_value=value):
                state = relay.RelayState(Path(tmp) / "relay", "gpu", "https://gpu.test/",
                    "https://cpu.test", "http://127.0.0.1:8002", MODEL, serving_profile=controller.NATIVE_PROFILE,
                    native_controller_receipt="/private/receipt.json", native_controller_token_file="/private/token.txt")
                self.assertEqual(state.configuration()["native_controller"]["policy_ack"], value["policy_ack"])
            changed = copy.deepcopy(value)
            changed["model_pid_start_ticks"] += 1
            with patch.object(relay, "owned_native_controller", return_value=changed), self.assertRaisesRegex(ValueError, "process changed"):
                state.configuration()


class NativeTransportTests(unittest.TestCase):
    def test_payload_and_original_request_id_survive_forwarding_until_durable_ack(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            token = root / "token.txt"
            token.write_text("f" * 64 + "\n")
            token.chmod(0o600)
            with patch.object(relay, "owned_native_controller", return_value=live_proof()):
                state = relay.RelayState(root / "relay", "gpu", "https://gpu.test/", "https://cpu.test",
                    "http://127.0.0.1:8002", MODEL, serving_profile=controller.NATIVE_PROFILE,
                    native_controller_receipt=root / "receipt.json", native_controller_token_file=token)
            request_id = "1" * 32
            payload = {"model": MODEL, "messages": [{"role": "user", "content": "original prompt"}],
                "temperature": 1, "max_tokens": 32}
            value = {"id": request_id, "payload": payload, "timeout_ms": 10000}
            context = state.admit_generation(request_id, payload, 10000)
            event = threading.Event()
            state.completions[request_id] = {"event": event}
            completion = {"model": MODEL, "choices": [], "training_evidence": {"request_id": request_id}}
            connection = Mock()
            response = connection.getresponse.return_value
            response.status, response.length = 200, 0
            response.read.return_value = json.dumps(completion).encode()
            with patch.object(relay, "open_upstream_connection", return_value=connection) as opener:
                code, actual = relay.execute_generation(state, value, context)
            self.assertEqual((code, actual), (200, completion))
            self.assertEqual(opener.call_args.args[1], 8002)
            sent = connection.request.call_args.args
            self.assertEqual(json.loads(sent[2]), payload)
            self.assertEqual(sent[3]["X-Rollout-Request-Id"], request_id)
            self.assertEqual(state.status()["retained_unacknowledged"], 1)
            self.assertEqual(state.status()["active_generations"], 0)
            event.set()
            state.acknowledge_delivery(request_id)
            self.assertEqual(state.status()["retained_unacknowledged"], 0)
            self.assertEqual(state.status()["acknowledged_ids"], [request_id])
            record = json.loads((state.root / "requests" / (request_id + ".json")).read_text())
            self.assertEqual(record["status"], "finished")
            self.assertTrue(record["delivery_acknowledged"])
            self.assertEqual(record["boot_id"], state.boot_id)
            with self.assertRaises(relay.ConflictError):
                state.admit_generation(request_id, payload, 10000)

    def test_echo_and_pre_dispatch_tombstones_preserve_journal_without_claiming_retained_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(relay, "owned_native_controller", return_value=live_proof()):
                state = relay.RelayState(Path(tmp).resolve() / "relay", "gpu", "https://gpu.test/",
                    "https://cpu.test", "http://127.0.0.1:8002", MODEL,
                    serving_profile=controller.NATIVE_PROFILE,
                    native_controller_receipt="/private/receipt.json", native_controller_token_file="/private/token.txt")
            echo_id, cancelled_id = "2" * 32, "3" * 32
            context = state.admit_generation(echo_id, {"operation": "echo"}, 20000, is_echo=True)
            state.finish_generation(echo_id, context, "finished")
            state.cancel_generation(cancelled_id)
            self.assertEqual(state.status()["retained_unacknowledged"], 0)
            self.assertEqual(state.status()["active_generations"], 0)
            self.assertEqual(state.status()["acknowledged_ids"], [])
            self.assertEqual(state.gpu_journal[echo_id]["status"], "finished")
            self.assertEqual(state.gpu_journal[cancelled_id]["status"], "cancelled")
            self.assertTrue(all((state.root / "requests" / (identity + ".json")).is_file()
                                for identity in (echo_id, cancelled_id)))

    def test_cpu_readiness_requires_native_proof_from_consumable_browser_challenge(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = relay.RelayState(Path(tmp) / "relay", "cpu", "https://cpu.test/", "https://gpu.test",
                "http://127.0.0.1:8002", MODEL, serving_profile=controller.NATIVE_PROFILE,
                generation_seconds=360, max_seconds=12000)
            value = {"protocol": 2, "serving_profile": controller.NATIVE_PROFILE, "boot_id": "b" * 32,
                "model": MODEL, "max_inflight": 1, "generation_seconds": 360, "uncertain": 0,
                "lifetime_remaining_ms": 12000000, "challenge_id": "a" * 32,
                "native_controller": live_proof()}
            with self.assertRaises(relay.ConflictError):
                state.pair_receipt(value)
            state.pair_challenge = {"id": value["challenge_id"], "issued": time.monotonic(), "expires": time.monotonic() + 15}
            self.assertTrue(state.pair_receipt(value)["recorded"])
            saved = json.loads((state.root / "readiness.json").read_text())
            self.assertEqual(saved["gpu"]["native_controller"], value["native_controller"])
            self.assertEqual(saved["cpu"]["serving_profile"], controller.NATIVE_PROFILE)
            with self.assertRaises(relay.ConflictError):
                state.pair_receipt(value)


if __name__ == "__main__":
    unittest.main()
