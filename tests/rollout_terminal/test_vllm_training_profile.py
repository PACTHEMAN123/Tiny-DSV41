"""Contract fixtures only: no CUDA graph, engine or benchmark is claimed live."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rollout_manager.backends.vllm_profile import (VLLM_KV_CACHE_MEMORY_BYTES, VLLM_KERNEL_CONFIG,
    VLLM_KV_BYTES_PER_TOKEN, TRITON_EXPERT_CLASS, VLLM_WORKER_RPC, vllm_policy_identities)
from rollout_manager.backends.vllm_worker import VLLM_STARTUP_ADAPTER, STOCK_KERNEL_WARMUP_SHA256
from rollout_manager.generation_training import TrainingGenerationContract
from rollout_manager.training import PolicyRef
from rollout_manager.terminal_bench import admission, controller
from rollout_manager.terminal_bench.vllm_workflow import require_actual_graph_sample
from .test_native_profile import proof as native_proof, MODEL


def proof(now=1000):
    value = native_proof(now)
    policy = PolicyRef(0, "a" * 64, "vllm-contract-fixture", {
        "tokenizer_sha256": "b" * 64, "chat_template_sha256": "c" * 64,
        "qwen_logits_path": "last_position_head_v1", **vllm_policy_identities(262144)},
        {"temperature": .8, "top_p": 1, "top_k": 0})
    ack = {key: getattr(policy, key) for key in ("version", "digest", "lineage")}
    adapter = {"profile": VLLM_STARTUP_ADAPTER, "applied_count": 1,
        "stock_kernel_warmup_sha256": STOCK_KERNEL_WARMUP_SHA256, "model_type": "qwen3_moe",
        "preserves_original_warmup_body": True, "bypassed_unrelated_minimax_import": True}
    value.update(serving_profile=controller.VLLM_TRAINING_PROFILE,
        policy=policy.to_dict(), policy_ack=ack,
        capabilities={"schema_version": 1, "policy": policy.to_dict(),
            "serving_profile": controller.VLLM_TRAINING_PROFILE,
            "logprob_convention": "sampled_distribution",
            **{key: True for key in TrainingGenerationContract.REQUIRED}})
    value["vllm_engine"] = {"schema_version": 1, "version": "0.31.0",
        "serving_profile": controller.VLLM_TRAINING_PROFILE, "tensor_parallel_size": 1,
        "logprobs_mode": "processed_logprobs", "prefix_caching": False, "enforce_eager": False,
        "cudagraph_capture_sizes": [1], "compilation_mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
        "max_model_len": 262144, "kv_cache_memory_bytes": VLLM_KV_CACHE_MEMORY_BYTES,
        "gpu_memory_utilization": .70, "kv_bytes_per_token": VLLM_KV_BYTES_PER_TOKEN,
        "full_context_raw_kv_bytes": 262144 * VLLM_KV_BYTES_PER_TOKEN,
        "kernel_config": dict(VLLM_KERNEL_CONFIG),
        "startup_adapter": copy.deepcopy(adapter),
        "worker_rpc_method": VLLM_WORKER_RPC, "insecure_serialization": False,
        "actor_process": {"pid": 876, "start_ticks": 71, "cmdline_sha256": "f" * 64},
        "loaded_policy_ack": ack,
        "loaded_trainable_tensors": {"sha256": "e" * 64, "parameter_names": ["backend.lm_head.weight"],
                                    "shape": [16, 8], "dtype": "torch.bfloat16"},
        "workers": [{"pid": 987, "start_ticks": 72, "cmdline_sha256": "d" * 64,
                     "device": "cuda:0", "device_uuid": "GPU-FIXTURE-NO-CUDA-EXECUTED",
                     "kernel_config": dict(VLLM_KERNEL_CONFIG),
                     "startup_adapter": copy.deepcopy(adapter),
                     "worker_rpc_method": VLLM_WORKER_RPC, "insecure_serialization": False,
                     "moe_implementations": [{"module": "model.layers.0.mlp.experts", "backend": "TRITON",
                         "selected_experts_class": TRITON_EXPERT_CLASS, "experts_class": TRITON_EXPERT_CLASS}],
                     "graph": {"captured_graph_count": 1, "replay_calls": 127}}]}
    return value


def readiness(value):
    base = {"serving_profile": controller.VLLM_TRAINING_PROFILE, "model": MODEL, "boot_id": "boot",
        "max_inflight": 1, "expires_unix": 12000, "generation_seconds": 360}
    return {"schema_version": 1, "protocol": 2, "paired_unix": 999, "valid_until_unix": 1010,
        "cpu": {**base, "pid": 1, "pid_start_ticks": 17},
        "gpu": {**base, "uncertain": 0, "training_controller": value}}


class VLLMTrainingProofTests(unittest.TestCase):
    def validate(self, value):
        return controller.validate_training_controller(value, MODEL,
            serving_profile=controller.VLLM_TRAINING_PROFILE, now=1000, required_seconds=3060)

    def test_engine_identity_rejects_raw_probabilities_and_unproven_weights(self):
        value = proof()
        self.assertIs(self.validate(value), value)
        for key, bad in (("logprobs_mode", "raw_logprobs"), ("enforce_eager", True),
                ("prefix_caching", True), ("tensor_parallel_size", 8), ("compilation_mode", 1),
                ("cudagraph_mode", "NONE"), ("loaded_trainable_tensors", {}),
                ("loaded_policy_ack", {"version": 99}), ("workers", [])):
            with self.subTest(field=key):
                changed = copy.deepcopy(value)
                changed["vllm_engine"][key] = bad
                with self.assertRaises(ValueError):
                    self.validate(changed)
        with self.assertRaises(ValueError):
            controller.validate_native_controller(value, MODEL, now=1000)

    def test_static_pairing_keeps_worker_ownership_but_excludes_moving_counters(self):
        value = proof()
        identity = controller.training_controller_identity(value, include_policy=False)
        changed = copy.deepcopy(value)
        changed["vllm_engine"]["workers"][0]["graph"]["replay_calls"] += 1
        self.assertEqual(identity, controller.training_controller_identity(changed, include_policy=False))
        changed["vllm_engine"]["workers"][0]["pid"] += 1
        self.assertNotEqual(identity, controller.training_controller_identity(changed, include_policy=False))

    def test_engine_and_worker_kernel_receipts_require_actual_triton_implementation(self):
        value = proof()
        mutations = (
            lambda e: e["kernel_config"].update(moe_backend="auto"),
            lambda e: e["kernel_config"].update(enable_jit_warmup=0),
            lambda e: e["workers"][0]["kernel_config"].update(enable_cutedsl_warmup=True),
            lambda e: e["workers"][0].update(moe_implementations=[]),
            lambda e: e["workers"][0]["moe_implementations"][0].update(experts_class="uninstantiated"),
            lambda e: e.update(full_context_raw_kv_bytes=1),
            lambda e: e.update(gpu_memory_utilization=.60),
            lambda e: e["workers"][0].update(start_ticks=0),
            lambda e: e["startup_adapter"].update(applied_count=0),
            lambda e: e["workers"][0]["startup_adapter"].update(stock_kernel_warmup_sha256="0" * 64),
            lambda e: e["workers"][0]["startup_adapter"].update(preserves_original_warmup_body=1),
            lambda e: e.update(worker_rpc_method="callable"),
            lambda e: e.update(insecure_serialization=True),
            lambda e: e["workers"][0].update(worker_rpc_method="another_owned_call"),
            lambda e: e["workers"][0].update(insecure_serialization=0),
        )
        for index, mutate in enumerate(mutations):
            with self.subTest(mutation=index):
                changed = copy.deepcopy(value)
                mutate(changed["vllm_engine"])
                self.assertNotEqual(json.dumps(controller.training_controller_identity(value, include_policy=False)),
                    json.dumps(controller.training_controller_identity(changed, include_policy=False)))
                with self.assertRaises(ValueError):
                    self.validate(changed)

    def test_guard_and_readiness_bind_vllm_loaded_policy_and_retained_capacity(self):
        live = proof()
        evidence = {"schema_version": 2, "serving_profile": controller.VLLM_TRAINING_PROFILE,
                    "model": MODEL, "training_controller": live}
        ready = readiness(live)
        guard = controller.guard_gate(evidence, ready, 3060, 1000,
            serving_profile=controller.VLLM_TRAINING_PROFILE, model=MODEL, expected_policy=live["policy"])
        self.assertEqual(guard["policy_ack"], live["policy_ack"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            controller.write_new(root / "readiness.json", ready)
            def check():
                return admission.check_readiness(root, MODEL, 1, 3060, now=1000,
                    identity_reader=lambda pid: (17, "S"), serving_profile=controller.VLLM_TRAINING_PROFILE,
                    expected_policy=live["policy"])
            self.assertEqual(check()["gpu"]["training_controller"], live)
            ready["gpu"]["uncertain"] = 1
            (root / "readiness.json").write_text(json.dumps(ready))
            with self.assertRaises(admission.episode.EpisodeError):
                check()

    def test_qualification_requires_actual_completed_graph_replays(self):
        value = proof()
        sample = {"prompt_ids": [1, 2], "response_ids": [3] * 128,
            "behavior_log_probs": [-.5] * 128, "finish_reason": "length", "stop_token_count": 0,
            "engine_receipt": value["vllm_engine"], "vllm_generation": {"elapsed_seconds": 2.,
                "graph_replay_calls_before": 0, "graph_replay_calls_after": 127,
                "graph_replay_calls_delta": 127, "graph_captured_count": 1}}
        policy = PolicyRef.from_dict(value["policy"])
        self.assertEqual(len(require_actual_graph_sample(sample, policy).action_token_ids), 128)
        for key, replacement in (("graph_replay_calls_delta", 0), ("graph_captured_count", 0),
                ("graph_replay_calls_after", 128), ("elapsed_seconds", float("nan"))):
            with self.subTest(field=key):
                changed = copy.deepcopy(sample)
                changed["vllm_generation"][key] = replacement
                with self.assertRaises(ValueError):
                    require_actual_graph_sample(changed, policy)


if __name__ == "__main__":
    unittest.main()
