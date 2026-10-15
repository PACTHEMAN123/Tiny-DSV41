"""vLLM owns sampled actions; the native learner owns GRPO and checkpoints."""
from __future__ import annotations

import math
from pathlib import Path
import secrets
import time

from .sealed_provider import marker, trainable_tensor_digest
from .controller import validate_vllm_engine
from ..model_completion import SHELL_PARAMETERS
from ..generation_training import TrainingGenerationContract
from ..backends.native_qwen_loopback import write_new


def require_actual_graph_sample(sample, policy, *, expected_tokens=128):
    """No graph flag, native fallback or raw pre-temperature LP can qualify a run."""
    from ..training import TurnTrace
    if not isinstance(sample, dict):
        raise ValueError("vLLM qualification requires an actual engine sample receipt")
    trace = TurnTrace(1, "public-vllm-qualification", sample["prompt_ids"], sample["response_ids"],
        sample["behavior_log_probs"], policy, sample["finish_reason"], sample["stop_token_count"], "none")
    if len(trace.action_token_ids) != expected_tokens:
        raise ValueError("vLLM diagnostic did not complete its explicit token budget")
    engine = validate_vllm_engine(sample["engine_receipt"], policy)
    metrics = sample["vllm_generation"]
    elapsed = metrics.get("elapsed_seconds")
    before, after, delta = (metrics.get("graph_replay_calls_" + key) for key in ("before", "after", "delta"))
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or not 0 < elapsed <= 360 or
            any(type(value) is not int or value < 0 for value in (before, after, delta)) or
            after - before != delta or delta < 1 or
            type(metrics.get("graph_captured_count")) is not int or metrics["graph_captured_count"] < 1 or
            engine["workers"][0]["graph"]["replay_calls"] != after or
            engine["workers"][0]["graph"]["captured_graph_count"] < 1):
        raise ValueError("actual completed vLLM CUDA graph replay is unproven")
    return trace


def public_probe_ids(tokenizer):
    words = "public sample words for cache timing only ".split() * 128
    low, high, probe_ids = 1, len(words), None
    while low <= high:
        count = (low + high) // 2
        ids = tokenizer.encode_conversation([{"role": "user", "content":
            "Continue this public performance diagnostic briefly: " + " ".join(words[:count])}],
            chat_template_kwargs={"enable_thinking": False})
        if len(ids) <= 512:
            probe_ids, low = ids, count + 1
        else:
            high = count - 1
    if probe_ids is None:
        raise RuntimeError("public vLLM diagnostic prompt exceeds its context budget")
    return probe_ids


def qualify_vllm_model(controller, output, *, learner_model):
    """Actual engine tokens/probabilities and honest native replay/gradient diagnostics.

    Cross-engine BF16 rounding is measured. The GRPO denominator remains the
    actual processed vLLM probabilities, never reconstructed native probabilities.
    No optimizer or benchmark verifier is run in this qualification.
    """
    import torch
    from dsv41_train.rl.algorithm import response_log_probs

    if learner_model is None:
        raise ValueError("vLLM qualification requires the actual live native learner")
    from ..training import token_digest
    output = Path(output)
    initial_head = trainable_tensor_digest(learner_model)
    actor_head = controller.loaded_trainable_tensor_digest()
    if initial_head["sha256"] != actor_head["sha256"]:
        raise RuntimeError("initial vLLM worker head differs from the actual learner weights")
    versions = {name: value._version for name, value in learner_model.named_parameters()}
    frozen = learner_model.frozen_parameter_fingerprint()
    policy = controller.policy
    probe = controller.sample_tokens(public_probe_ids(controller.tokenizer), 128,
        policy.sampling["temperature"], secrets.token_hex(16), ignore_eos=True, diagnostic_seed=731)
    write_new(output / "mounted-vllm-probe-sample.json", {
        "benchmark_task": False, "benchmark_reward": None, "tool_executed": False,
        "public_prompt_only": True, "sampling_profile": "vllm-training-v1", "sample": probe})
    trace = require_actual_graph_sample(probe, policy)
    replay_started = time.monotonic()
    learner_model.zero_grad(set_to_none=True)
    try:
        current = response_log_probs(learner_model, trace.context_token_ids,
            trace.action_token_ids, policy.sampling["temperature"])
        behavior = current.new_tensor(trace.behavior_log_probs)
        difference = current.detach() - behavior
        ratios = difference.exp()
        if not torch.isfinite(ratios).all():
            raise RuntimeError("native learner/vLLM sampled-action importance ratios are nonfinite")
        cross_engine = {
            "action_tokens": len(trace.action_token_ids),
            "max_abs_logprob_error": float(difference.abs().max()),
            "mean_abs_logprob_error": float(difference.abs().mean()),
            "importance_ratio_min": float(ratios.min()), "importance_ratio_max": float(ratios.max()),
            "importance_ratio_mean": float(ratios.mean()),
            "outside_grpo_clip_fraction": float(((ratios - 1).abs() > controller.config.clip_eps).float().mean()),
            "exact_cross_engine_parity_claimed": False,
            "behavior_denominator": "actual_vllm_processed_sampled_distribution",
            "learner_replay": "request_kv_cache_v1", "optimizer_step": False}
        (-current.mean()).backward()
        gradients = [value.grad for value in learner_model.parameters() if value.requires_grad]
        if not gradients or any(value is None or not torch.isfinite(value).all() for value in gradients):
            raise RuntimeError("live native learner head gradient is missing or nonfinite")
        gradient_norm = sum(float(value.float().norm()) for value in gradients)
        if gradient_norm <= 0:
            raise RuntimeError("live native learner head gradient has zero norm")
    finally:
        learner_model.zero_grad(set_to_none=True)
    if (initial_head != trainable_tensor_digest(learner_model) or
            versions != {name: value._version for name, value in learner_model.named_parameters()} or
            frozen != learner_model.frozen_parameter_fingerprint() or
            actor_head["sha256"] != controller.loaded_trainable_tensor_digest()["sha256"]):
        raise RuntimeError("vLLM/native learner qualification mutated weights")
    receipt = {"kind": "mounted-vllm-model-qualification", "benchmark_task": False,
        "benchmark_reward": None, "tool_executed": False, "public_prompt_only": True,
        "policy": policy.to_dict(), "sample": probe, "cross_engine_replay": cross_engine,
        "context_token_ids_sha256": token_digest(trace.context_token_ids), "sampling_seed": 731,
        "head_gradient_norm": gradient_norm, "replay_backward_seconds": time.monotonic() - replay_started,
        "parameters_unchanged": True, "actual_actor_trainable_tensors": actor_head,
        "actual_learner_trainable_tensors": initial_head,
        "generation_seconds": probe["vllm_generation"]["elapsed_seconds"],
        "tokens_per_second": 128 / probe["vllm_generation"]["elapsed_seconds"],
        "full_action_budget_throughput_qualified": False, "graph_replay_qualified": True}
    write_new(output / "mounted-vllm-qualification.json", receipt)
    # Independently exercise ordinary tool serialization without executing it.
    payload = {"model": controller.served_model, "messages": [
        {"role": "system", "content": "Use the shell tool when asked to run a command. Be brief."},
        {"role": "user", "content": "Call shell once with the command printf READY. Do not give a final answer yet."}],
        "tools": [{"type": "function", "function": {"name": "shell", "description": "Run shell.",
            "parameters": SHELL_PARAMETERS}}], "tool_choice": "auto", "parallel_tool_calls": False,
        "chat_template_kwargs": {"enable_thinking": False}, "stream": False,
        "temperature": policy.sampling["temperature"], "max_tokens": controller.config.max_new_tokens}
    request_id = secrets.token_hex(16)
    started = time.monotonic()
    completion = controller.generate(payload, request_id)
    elapsed = time.monotonic() - started
    TrainingGenerationContract(policy, controller.capabilities()).validate(payload, completion, request_id, 1)
    write_new(output / "mounted-vllm-tool-qualification.json", {"benchmark_task": False,
        "tool_executed": False, "payload": payload, "completion": completion, "generation_seconds": elapsed})
    if completion["choices"][0]["finish_reason"] != "tool_calls" or elapsed > 360:
        raise RuntimeError("mounted vLLM qualification requires a real shell-tool completion within 360 seconds")
    marker("A_MOUNTED_VLLM_QUALIFIED", generation_seconds=receipt["generation_seconds"],
        tokens_per_second=receipt["tokens_per_second"], graph_replay_calls=probe["vllm_generation"]["graph_replay_calls_delta"],
        head_gradient_norm=gradient_norm, max_cross_engine_logprob_error=cross_engine["max_abs_logprob_error"],
        exact_cross_engine_parity_claimed=False, benchmark_tasks_executed=0)


def main():
    from .native_workflow import main as workflow_main
    from ..backends.vllm_controller import VLLMPolicyController
    workflow_main(controller_factory=VLLMPolicyController, qualification=qualify_vllm_model, vllm=True)


if __name__ == "__main__":
    main()
