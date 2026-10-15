"""Bound the vLLM actor separately from the native learner's replay kernels."""
VLLM_TRAINING_PROFILE = "vllm-training-v1"
VLLM_ROLLOUT_PROFILE = "sealed-original-terminal-vllm-cpu-browser-v1"
VLLM_VERSION = "0.31.0"
VLLM_KV_CACHE_MEMORY_BYTES = 32 * 1024**3
VLLM_KV_BYTES_PER_TOKEN = 98304
VLLM_GPU_MEMORY_UTILIZATION = .70
VLLM_KERNEL_CONFIG = {"moe_backend": "triton", "enable_jit_warmup": False,
                      "enable_cutedsl_warmup": False, "enable_flashinfer_autotune": False}
TRITON_EXPERT_CLASS = "vllm.model_executor.layers.fused_moe.experts.triton_moe.TritonExperts"
VLLM_WORKER_RPC = "rollout_manager_owned_call"
VLLM_POLICY_IDENTITIES = {
    "rollout_engine": "vllm",
    "vllm_version": VLLM_VERSION,
    "vllm_logprobs_mode": "processed_logprobs",
    "qwen_learner_replay_path": "request_kv_cache_v1",
    "vllm_kv_cache_memory_bytes": str(VLLM_KV_CACHE_MEMORY_BYTES),
    "vllm_moe_backend": "triton",
    "vllm_enable_jit_warmup": "false",
    "vllm_enable_cutedsl_warmup": "false",
    "vllm_enable_flashinfer_autotune": "false",
    "vllm_startup_adapter": "qwen3-moe-minimax-import-guard-v1",
    "vllm_worker_rpc": VLLM_WORKER_RPC,
    "vllm_insecure_serialization": "false",
}


def vllm_policy_identities(max_position_embeddings):
    if type(max_position_embeddings) is not int or max_position_embeddings < 1:
        raise ValueError("vLLM profile requires the mounted model's original context size")
    return {**VLLM_POLICY_IDENTITIES, "vllm_max_model_len": str(max_position_embeddings)}


def require_vllm_learner(config):
    if (config.trainable_scope != "lm_head" or config.dtype != "bfloat16" or
            not config.qwen_kv_cache or not config.qwen_last_position_only or
            config.backprop_chunk_tokens != 0):
        raise ValueError("vLLM training requires BF16 head-only native cached replay without chunking")


def validate_vllm_policy_profile(policy, config, max_position_embeddings):
    require_vllm_learner(config)
    if (any(policy.identities.get(key) != value for key, value in
            vllm_policy_identities(max_position_embeddings).items()) or
            policy.identities.get("qwen_sampling_path") is not None or
            policy.identities.get("qwen_logits_path") != "last_position_head_v1"):
        raise ValueError("vLLM behavior identity differs from the negotiated actor/learner profiles")
