#!/usr/bin/env python3
"""A owns Qwen/GRPO; the existing CPU executor owns original Pouch/Harbor tasks.

No Docker, Pouch, task sandbox or task verifier is started inside A. The browser
FileRelay is the only cross-machine generation route. CPU groups must be sealed
after original verification/cleanup and independently match A's actual receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.runtime import run_training
from rollout_manager.backends.native_qwen_loopback import strict_json, write_new
from rollout_manager.backends.pytorch_controller import PyTorchPolicyController
from rollout_manager.model_completion import SHELL_PARAMETERS
from .sealed_provider import SealedCPUGroupProvider, marker, trainable_tensor_digest


def external_verifier_reward(record, response_ids):
    raise RuntimeError("Terminal-Bench rewards must come from sealed original verifiers, never token/text scoring")


def qualify_native_model(controller, output):
    """Bounded actual mounted sampling/replay/head-gradient check; no RL reward."""
    import torch
    from dsv41_train.rl.algorithm import response_log_probs
    from rollout_manager.generation_training import TrainingGenerationContract
    import secrets

    payload = {"model": controller.served_model, "messages": [
        {"role": "system", "content": "Use the shell tool when asked to run a command. Be brief."},
        {"role": "user", "content": "Call shell once with the command printf READY. Do not give a final answer yet."}],
        "tools": [{"type": "function", "function": {"name": "shell", "description": "Run shell.",
            "parameters": SHELL_PARAMETERS}}], "tool_choice": "auto", "parallel_tool_calls": False,
        "chat_template_kwargs": {"enable_thinking": False}, "stream": False,
        "temperature": controller.config.temperature, "max_tokens": controller.config.max_new_tokens}
    identity = secrets.token_hex(16)
    model = controller.model
    before = model.frozen_parameter_fingerprint()
    versions = {name: parameter._version for name, parameter in model.named_parameters()}
    if controller.config.qwen_kv_cache:
        from rollout_manager.backends.qwen_kv_cache import qualify_qwen_cache
        words = ("public sample words for cache timing only ".split() * 128)
        low, high, probe_ids = 1, len(words), None
        while low <= high:
            count = (low + high) // 2
            public_messages = [{"role": "user", "content": (
                "Continue this public performance diagnostic briefly: " + " ".join(words[:count]))}]
            ids = controller.tokenizer.encode_conversation(public_messages,
                chat_template_kwargs={"enable_thinking": False})
            if len(ids) <= 512:
                probe_ids, low = ids, count + 1
            else:
                high = count - 1
        if probe_ids is None:
            raise RuntimeError("public cache diagnostic prompt exceeds its context budget")
        cache_receipt = qualify_qwen_cache(model, probe_ids, decode_tokens=128,
            temperature=controller.config.temperature, seed=731)
        cache_receipt["public_prompt_only"] = True
        write_new(Path(output) / "mounted-cache-qualification.json", cache_receipt)
        if not cache_receipt["cached_replay_qualified"]:
            raise RuntimeError("actual mounted cached sampling/replay probability parity failed")
        marker("A_MOUNTED_CACHE_QUALIFIED", action_tokens=cache_receipt["action_tokens"],
               context_tokens=cache_receipt["context_tokens"],
               generation_seconds=cache_receipt["generation_seconds"],
               tokens_per_second=cache_receipt["tokens_per_second"],
               max_logprob_error=cache_receipt["cached_sampling_vs_cached_replay_max_logprob_error"],
               full_action_budget_throughput_qualified=False)
    started = time.monotonic()
    completion = controller.generate(payload, identity)
    generation_seconds = time.monotonic() - started
    write_new(Path(output) / "mounted-model-probe-completion.json",
              {"benchmark_task": False, "tool_executed": False, "payload": payload,
               "completion": completion, "generation_seconds": generation_seconds})
    trace = TrainingGenerationContract(controller.policy, controller.capabilities()).validate(payload, completion, identity, 1)
    if completion["choices"][0]["finish_reason"] != "tool_calls" or generation_seconds > 360:
        write_new(Path(output) / "mounted-model-qualification-failed.json",
                  {"benchmark_task": False, "payload": payload, "completion": completion,
                   "generation_seconds": generation_seconds, "generation_deadline_seconds": 360})
        raise RuntimeError("mounted model qualification requires a real shell-tool completion within 360 seconds")
    replay_started = time.monotonic()
    model.zero_grad(set_to_none=True)
    try:
        probabilities = response_log_probs(model, trace.context_token_ids, trace.action_token_ids, controller.config.temperature)
        behavior = torch.tensor(trace.behavior_log_probs, device=probabilities.device, dtype=probabilities.dtype)
        maximum_error = float((probabilities.detach() - behavior).abs().max())
        if not torch.isfinite(probabilities).all() or maximum_error > 1e-5:
            raise RuntimeError("actual mounted native sampling/replay probability parity failed")
        (-probabilities.mean()).backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        if not gradients or any(value is None or not torch.isfinite(value).all() for value in gradients):
            raise RuntimeError("actual mounted head gradient is missing or nonfinite")
        gradient_norm = sum(float(value.float().norm()) for value in gradients)
        if gradient_norm <= 0:
            raise RuntimeError("actual mounted head gradient has zero norm")
    finally:
        model.zero_grad(set_to_none=True)
    model.assert_engram_invariants()
    after = model.frozen_parameter_fingerprint()
    if before != after or versions != {name: parameter._version for name, parameter in model.named_parameters()}:
        raise RuntimeError("mounted model qualification changed parameters")
    receipt = {"kind": "mounted-native-model-qualification", "benchmark_task": False,
        "policy": controller.policy.to_dict(), "payload": payload, "completion": completion,
        "tool_executed": False, "action_budget": payload["max_tokens"],
        "generation_deadline_seconds": 360, "full_action_budget_throughput_qualified": False,
        "generation_seconds": generation_seconds, "replay_backward_seconds": time.monotonic() - replay_started,
        "max_logprob_error": maximum_error, "head_gradient_norm": gradient_norm,
        "parameters_unchanged": True, "frozen_parameter_report": after,
        "cpu_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_bytes": {str(device): {"allocated": torch.cuda.max_memory_allocated(device),
            "reserved": torch.cuda.max_memory_reserved(device)} for device in
            sorted({parameter.device.index for parameter in model.parameters() if parameter.device.type == "cuda"})}}
    write_new(Path(output) / "mounted-model-qualification.json", receipt)
    marker("A_MOUNTED_MODEL_QUALIFIED", generation_seconds=generation_seconds,
           replay_backward_seconds=receipt["replay_backward_seconds"], max_logprob_error=maximum_error,
           head_gradient_norm=gradient_norm, benchmark_tasks_executed=0)


def turn_budget_declaration(requested, initial_policy=None):
    """Separate an inherited policy identity from the actual CPU budget gate."""
    if requested is not None and (type(requested) is not int or not 1 <= requested <= 1024):
        raise ValueError("agent_max_turns must be an integer in 1..1024")
    if initial_policy is None:
        declared = 128 if requested is None else requested
        return declared, declared
    declared = initial_policy["identity"]["rollout_profile"].get("max_turns", 128)
    if type(declared) is not int or not 1 <= declared <= 1024:
        raise ValueError("original policy max_turns is invalid")
    return declared, requested


def main(*, controller_factory=PyTorchPolicyController, qualification=qualify_native_model, vllm=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/mnt/fuse/Qwen/Qwen3-30B-A3B-Instruct-2507")
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--group-manifests", type=Path, nargs="+", required=True)
    from rollout_manager.terminal_bench.sealed import TASKS
    parser.add_argument("--task-ids", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--agent-max-turns", type=int,
                        help="expected CPU agent turn budget (fresh run default: 128)")
    parser.add_argument("--post-update-manifest", type=Path)
    parser.add_argument("--qwen-kv-cache", action="store_true")
    parser.add_argument("--gpu-relay-root", type=Path, required=True)
    parser.add_argument("--gpu-relay-port", type=int, default=18093)
    parser.add_argument("--shim-port", type=int, default=8002)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--learner-devices", nargs="+", default=["cuda:0"])
    parser.add_argument("--rollout-devices", nargs="+", default=["cuda:1"])
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--worker-seconds", type=float, default=30000)
    parser.add_argument("--backprop-chunk-tokens", type=int, default=0 if vllm else 16)
    parser.add_argument("--qualify-model", action="store_true", default=vllm)
    parser.add_argument("--automatic-artifacts", action="store_true", default=vllm)
    parser.add_argument("--manual-artifacts", action="store_false", dest="automatic_artifacts")
    parser.add_argument("--recover-pre-update-from", type=Path)
    parser.add_argument("--original-workflow-receipt", type=Path)
    parser.add_argument("--original-policy-sha256")
    parser.add_argument("--original-actor-exit-sha256")
    parser.add_argument("--original-group-seal-sha256")
    parser.add_argument("--historical-gpu-relay-root", type=Path)
    parser.add_argument("--historical-gpu-relay-port", type=int)
    parser.add_argument("--historical-gpu-relay-startup-sha256")
    parser.add_argument("--historical-gpu-relay-cmdline-sha256")
    args = parser.parse_args()
    if (not 2 <= args.group_size <= 64 or len(args.task_ids) != len(args.group_manifests)
            or len(set(args.task_ids)) != len(args.task_ids)):
        parser.error("select distinct tasks, one manifest per task, and a training group size in 2..64")
    if args.agent_max_turns is not None and not 1 <= args.agent_max_turns <= 1024:
        parser.error("--agent-max-turns must be in 1..1024")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    output = args.output_dir.absolute()
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    capsule_bytes = args.capsule.read_bytes()
    capsule = strict_json(capsule_bytes)
    initial_policy, historical_delivery, initial_consumer = None, None, None
    if args.recover_pre_update_from is not None:
        if not vllm or not all((args.original_workflow_receipt, args.original_policy_sha256,
                args.original_actor_exit_sha256, args.original_group_seal_sha256,
                args.historical_gpu_relay_root, args.historical_gpu_relay_port,
                args.historical_gpu_relay_startup_sha256, args.historical_gpu_relay_cmdline_sha256)):
            parser.error("pre-update recovery requires vLLM and every original artifact/owner/relay pin")
        from .preupdate_recovery import original_initial_policy, HistoricalGroupDelivery
        from ..training import PolicyRef
        initial_policy, original_lease, recovery_proof, initial_consumer = original_initial_policy(args.recover_pre_update_from,
            args.original_workflow_receipt, expected_digest=args.original_policy_sha256,
            expected_actor_exit_sha256=args.original_actor_exit_sha256, consumer_output=output,
            expected_group_seal_sha256=args.original_group_seal_sha256)
        historical_delivery = HistoricalGroupDelivery(args.recover_pre_update_from,
            args.historical_gpu_relay_root, args.historical_gpu_relay_port,
            PolicyRef.from_dict(original_lease["policy"]), original_lease["served_model"],
            args.historical_gpu_relay_startup_sha256, args.historical_gpu_relay_cmdline_sha256)
        write_new(output / "original-pre-update-recovery.json", recovery_proof)
    elif any((args.original_workflow_receipt, args.original_policy_sha256,
            args.original_actor_exit_sha256, args.original_group_seal_sha256, args.historical_gpu_relay_root,
            args.historical_gpu_relay_port, args.historical_gpu_relay_startup_sha256,
            args.historical_gpu_relay_cmdline_sha256)):
        parser.error("original recovery pins require --recover-pre-update-from")
    declared_max_turns, expected_cpu_max_turns = turn_budget_declaration(args.agent_max_turns, initial_policy)
    data = output / "external-cpu-dataset.jsonl"
    # This placeholder is never sampled or rewarded. Actual actions/rewards are
    # the imported original benchmark TrainingGroup records exclusively.
    write_new(data, {"prompt_ids": [0], "metadata": {"dataset": "external-sealed-terminal-bench",
              "public_capsule_sha256": hashlib.sha256(capsule_bytes).hexdigest()}})
    config = RLConfig(max_steps=1, group_size=args.group_size, groups_per_step=len(args.task_ids), max_new_tokens=2048,
        rollout_engine="vllm" if vllm else "native-pytorch",
        temperature=.8, learning_rate=1e-4, kl_beta=0, weight_decay=0,
        queue_capacity=1, max_policy_lag=0, checkpoint_interval=1,
        reward=__name__ + ":external_verifier_reward", trainable_scope="lm_head",
        qwen_last_position_only=True, qwen_kv_cache=args.qwen_kv_cache or vllm,
        backprop_chunk_tokens=args.backprop_chunk_tokens,
        learner_devices=tuple(args.learner_devices), rollout_devices=tuple(args.rollout_devices),
        dtype=args.dtype, output_dir=str(output), worker_timeout_seconds=args.worker_seconds,
        log_samples_per_step=0, length_policy="error")
    config.validate()
    write_new(output / "experiment-config.json", config.to_dict())
    def provider(*positional, **kwargs):
        selected_qualification = qualification
        if vllm:
            from functools import partial
            selected_qualification = partial(qualification, learner_model=kwargs.pop("learner_model"))
        return SealedCPUGroupProvider(*positional, **kwargs, capsule=args.capsule,
            manifests=args.group_manifests, relay_root=args.gpu_relay_root,
            relay_port=args.gpu_relay_port, shim_port=args.shim_port, qualify_model=args.qualify_model,
            task_ids=args.task_ids, post_update_manifest=args.post_update_manifest,
            controller_factory=controller_factory, qualification=selected_qualification,
            historical_delivery=historical_delivery,
            expected_original_seal_sha256=args.original_group_seal_sha256,
            automatic_artifacts=args.automatic_artifacts,
            expected_max_turns=expected_cpu_max_turns)
    marker("A_VLLM_LOADING" if vllm else "A_NATIVE_LOADING", output_dir=str(output), model_dir=args.model_dir,
           learner_devices=config.learner_devices, rollout_devices=config.rollout_devices,
           cpu_executor_owns_pouch=True, max_tokens=2048, max_turns=declared_max_turns,
           expected_cpu_max_turns=expected_cpu_max_turns,
           turn_budget_source="original-policy-provenance" if initial_policy is not None else "workflow-declaration",
           generation_seconds=360)
    try:
        checkpoint = run_training(model_dir=args.model_dir, data_path=data, config=config,
            random_init=False, rollout_factory=provider,
            rollout_identity={"schema_version": 1, "profile": (
                "sealed-original-terminal-vllm-cpu-browser-v1" if vllm else "sealed-original-terminal-cpu-browser-v1"),
                "public_capsule_sha256": hashlib.sha256(capsule_bytes).hexdigest(),
                "reward_mapping": "terminal-validated-verifier-v1", "tasks": args.task_ids,
                "samples_per_task": args.group_size, "max_tokens": 2048, "max_turns": declared_max_turns,
                "generation_seconds": 360, "post_update_episode_required": args.post_update_manifest is not None},
            initial_policy=initial_policy, initial_policy_consumer=initial_consumer)
    finally:
        if initial_consumer is not None:
            initial_consumer.close()
    marker("A_TERMINAL_RL_COMPLETED", checkpoint=str(checkpoint),
           checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
