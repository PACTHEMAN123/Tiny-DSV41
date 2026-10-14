"""vLLM controller routing fixtures never claim a live GPU qualification."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

from rollout_manager.terminal_bench.browser_relay import relay_server as relay
from .test_browser_relay_native import live_proof
from .test_native_profile import MODEL

PROFILE = "vllm-training-v1"


def vllm_proof():
    p = live_proof()
    p["serving_profile"] = PROFILE
    p["policy"]["identities"].update({"rollout_engine": "vllm", "vllm_version": "0.31.0",
        "vllm_logprobs_mode": "processed_logprobs", "qwen_learner_replay_path": "request_kv_cache_v1", "vllm_max_model_len": "32768",
        "vllm_kv_cache_memory_bytes": "34359738368", "vllm_moe_backend": "triton",
        "vllm_enable_jit_warmup": "false", "vllm_enable_cutedsl_warmup": "false",
        "vllm_enable_flashinfer_autotune": "false", "vllm_startup_adapter": "qwen3-moe-minimax-import-guard-v1",
        "vllm_worker_rpc": "rollout_manager_owned_call", "vllm_insecure_serialization": "false"})
    p["capabilities"]["policy"] = copy.deepcopy(p["policy"])
    p["capabilities"]["serving_profile"] = PROFILE
    adapter = {"profile": "qwen3-moe-minimax-import-guard-v1", "applied_count": 1,
        "stock_kernel_warmup_sha256": "69d9f633e353d02898b0e0054613f10596747341ea23e4369c1cdfd88ec2cac5",
        "model_type": "qwen3_moe", "preserves_original_warmup_body": True, "bypassed_unrelated_minimax_import": True}
    p["vllm_engine"] = {"schema_version": 1, "version": "0.31.0", "serving_profile": PROFILE,
        "tensor_parallel_size": 1, "logprobs_mode": "processed_logprobs", "prefix_caching": False,
        "enforce_eager": False, "compilation_mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [1],
        "loaded_policy_ack": copy.deepcopy(p["policy_ack"]),
        "loaded_trainable_tensors": {"sha256": "e" * 64, "shape": [128, 32],
            "dtype": "torch.bfloat16", "parameter_names": ["backend.lm_head.weight"]},
        "max_model_len": 32768, "kv_cache_memory_bytes": 34359738368,
        "gpu_memory_utilization": .70, "kv_bytes_per_token": 98304, "full_context_raw_kv_bytes": 3221225472,
        "kernel_config": {"moe_backend": "triton", "enable_jit_warmup": False,
            "enable_cutedsl_warmup": False, "enable_flashinfer_autotune": False},
        "worker_rpc_method": "rollout_manager_owned_call", "insecure_serialization": False,
        "startup_adapter": copy.deepcopy(adapter),
        "actor_process": {"pid": 320, "start_ticks": 321, "cmdline_sha256": "f" * 64},
        "workers": [{"pid": 321, "start_ticks": 322, "cmdline_sha256": "f" * 64, "device": "cuda:0", "device_uuid": "GPU-test",
            "kernel_config": {"moe_backend": "triton", "enable_jit_warmup": False,
                "enable_cutedsl_warmup": False, "enable_flashinfer_autotune": False},
            "worker_rpc_method": "rollout_manager_owned_call", "insecure_serialization": False,
        "startup_adapter": copy.deepcopy(adapter),
            "moe_implementations": [{"module": "model.layers.0.mlp.experts", "backend": "TRITON",
                "selected_experts_class": "vllm.model_executor.layers.fused_moe.experts.triton_moe.TritonExperts",
                "experts_class": "vllm.model_executor.layers.fused_moe.experts.triton_moe.TritonExperts"}],
            "graph": {"captured_graph_count": 1, "replay_calls": 12}}], "engine_startup_seconds": 1.5}
    return p


class VLLMTrainingRelayTests(unittest.TestCase):
    def test_profile_forwards_request_identity_auth_and_raw_training_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            token = root / "token.txt"
            token.write_text("f" * 64 + "\n")
            token.chmod(0o600)
            with patch.object(relay, "owned_training_controller", return_value=vllm_proof()), \
                    patch.object(relay, "owned_model_capacity", side_effect=AssertionError("No CLI impersonation")):
                state = relay.RelayState(root / "relay", "gpu", "https://gpu.test/", "https://cpu.test",
                    "http://127.0.0.1:8002", MODEL, serving_profile=PROFILE,
                    training_controller_receipt=root / "receipt.json", training_controller_token_file=token)
                config = state.configuration()
            self.assertIn("training_controller", config)
            self.assertNotIn("native_controller", config)
            self.assertNotIn("model_capacity_source", config)
            rid = "1" * 32
            payload = {"model": MODEL, "messages": [{"role": "user", "content": "original prompt"}],
                "temperature": 1, "max_tokens": 32}
            context = state.admit_generation(rid, payload, 10000)
            state.completions[rid] = {"event": threading.Event()}
            completion = {"model": MODEL, "choices": [], "training_evidence": {"request_id": rid}}
            connection = Mock()
            response = connection.getresponse.return_value
            response.status, response.length = 200, 0
            response.read.return_value = json.dumps(completion).encode()
            with patch.object(relay, "open_upstream_connection", return_value=connection) as opener:
                result = relay.execute_generation(state, {"id": rid, "payload": payload}, context)
            self.assertEqual(result, (200, completion))
            self.assertEqual(opener.call_args.args[1], 8002)
            sent = connection.request.call_args.args
            self.assertEqual(json.loads(sent[2]), payload)
            self.assertEqual(sent[3]["X-Rollout-Request-Id"], rid)
            self.assertEqual(sent[3]["Authorization"], "Bearer " + "f" * 64)
            self.assertEqual(state.status()["retained_unacknowledged"], 1)

    def test_changed_engine_worker_stops_configuration_and_profiles_do_not_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = vllm_proof()
            with patch.object(relay, "owned_training_controller", return_value=value):
                state = relay.RelayState(Path(tmp) / "relay", "gpu", "https://gpu.test/",
                    "https://cpu.test", "http://127.0.0.1:8002", MODEL, serving_profile=PROFILE,
                    training_controller_receipt="/private/receipt.json", training_controller_token_file="/private/token.txt")
            changed = copy.deepcopy(value)
            changed["vllm_engine"]["workers"][0]["pid"] += 1
            with patch.object(relay, "owned_training_controller", return_value=changed), self.assertRaisesRegex(ValueError, "process changed"):
                state.configuration()
            with self.assertRaises(ValueError):
                relay.RelayState(Path(tmp) / "wrong", "gpu", "https://gpu.test/", "https://cpu.test",
                    "http://127.0.0.1:8002", MODEL, serving_profile=PROFILE,
                    native_controller_receipt="/private/receipt.json", native_controller_token_file="/private/token.txt")

    def test_pairing_rejects_cross_profile_and_stale_loaded_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = relay.RelayState(Path(tmp) / "relay", "cpu", "https://cpu.test/", "https://gpu.test",
                "http://127.0.0.1:8002", MODEL, serving_profile=PROFILE, generation_seconds=360, max_seconds=12000)
            value = {"protocol": 2, "serving_profile": PROFILE, "boot_id": "b" * 32, "model": MODEL,
                "max_inflight": 1, "generation_seconds": 360, "uncertain": 0,
                "lifetime_remaining_ms": 12000000, "challenge_id": "a" * 32, "training_controller": vllm_proof()}
            for mutation in ("native_key", "legacy_capacity", "stale_loaded_ack"):
                changed = copy.deepcopy(value)
                if mutation == "native_key":
                    changed["native_controller"] = changed.pop("training_controller")
                elif mutation == "legacy_capacity":
                    changed["model_capacity_source"] = "owned_vllm_process"
                else:
                    changed["training_controller"]["vllm_engine"]["loaded_policy_ack"]["version"] = 1
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    state.pair_receipt(changed)
            state.pair_challenge = {"id": value["challenge_id"], "issued": time.monotonic(), "expires": time.monotonic() + 15}
            self.assertTrue(state.pair_receipt(value)["recorded"])
            saved = json.loads((state.root / "readiness.json").read_text())
            self.assertEqual(saved["gpu"]["training_controller"], value["training_controller"])
            self.assertNotIn("native_controller", saved["gpu"])


if __name__ == "__main__":
    unittest.main()
