"""Prompt-response groups through the same owned vLLM actor as Terminal-Bench."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

from ..contracts import EpisodeResult
from ..training import TrainingGroup, TrainingTrajectory, TurnTrace, token_digest
from .vllm_profile import VLLM_KV_CACHE_MEMORY_BYTES, require_vllm_learner


def require_vllm_rollout(model_config, config, *, random_init=False):
    """Check the implemented weight adapter before allocating learner memory."""
    if getattr(model_config, "model_type", None) != "qwen3_moe":
        raise ValueError("vLLM RL weight activation currently supports flat Qwen3 MoE only; "
                         "DeepSeek requires a vLLM weight/checkpoint adapter. "
                         "Select native-pytorch explicitly for legacy compatibility.")
    require_vllm_learner(config)
    if random_init:
        raise ValueError("vLLM RL requires mounted pretrained weights; random initialization is unsupported")
    if len(config.rollout_devices) != 1 or not config.rollout_devices[0].startswith("cuda:"):
        raise ValueError("vLLM RL requires exactly one dedicated CUDA rollout device")
    if any(not device.startswith("cuda:") for device in config.learner_devices):
        raise ValueError("vLLM RL requires CUDA learner devices and a separate CUDA actor; "
                         "configure --trainer-num-gpus 1 --rollout-num-gpus 1 or explicit device lists")
    actor = int(config.rollout_devices[0].split(":")[1])
    learner = {int(device.split(":")[1]) for device in config.learner_devices}
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    labels = [value.strip().lower() for value in visible.split(",")] if visible else []
    if actor in learner or labels and actor < len(labels) and any(
            index < len(labels) and labels[index] == labels[actor] for index in learner):
        raise ValueError("vLLM actor and native learner require disjoint physical GPUs")
    if config.stop_token_sequences:
        raise ValueError("vLLM RL cannot approximate token stop sequences with text stops")
    size = 2 * model_config.num_hidden_layers * model_config.num_key_value_heads * model_config.head_dim * 2
    if size * model_config.max_position_embeddings > VLLM_KV_CACHE_MEMORY_BYTES:
        raise ValueError("vLLM KV allocation cannot preserve this mounted model's full context")


def validate_vllm_device_ownership(config):
    """Resolve UUIDs after device availability checks, before model allocation."""
    import torch
    identities = {}
    for device in (*config.learner_devices, *config.rollout_devices):
        uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
        if uuid is None or not str(uuid).strip():
            raise ValueError("vLLM GPU ownership requires actual CUDA device UUIDs")
        identities[device] = str(uuid).lower()
    if identities[config.rollout_devices[0]] in {identities[device] for device in config.learner_devices}:
        raise ValueError("vLLM actor and native learner require disjoint physical GPUs")


def evaluation_seed(base_seed, prompt_ids, sample_index):
    # Fixed evaluation seeds are independent of training requests and policy
    # versions. They belong to the vLLM engine, not a CPU torch generator.
    raw = json.dumps([base_seed, prompt_ids, sample_index], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63 - 1)


class VLLMRLBackend:
    def __init__(self, model_dir, model_config, config, random_init, *,
                 engram_weights=None, initial_policy, rng_state=None, evidence_dir):
        require_vllm_rollout(model_config, config, random_init=random_init)
        if engram_weights or rng_state is not None:
            raise ValueError("vLLM actor cannot restore native Engram or CPU sampler state")
        self.model_dir, self.model_config, self.config = model_dir, model_config, config
        self.initial_policy, self.evidence_dir = initial_policy, Path(evidence_dir).resolve()
        self._packets, self._busy = {}, False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vllm-policy")
        self.controller = None

    @property
    def ack(self):
        return self.controller.ack

    @property
    def policy(self):
        return self.controller.policy

    @property
    def active_artifact(self):
        return self.controller.active_artifact

    async def start(self):
        from dsv41_train.rl.data import build_reward
        from .vllm_controller import VLLMPolicyController
        self.evidence_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.controller = VLLMPolicyController(self.model_dir, self.config, served_model="rl-policy")
        self.controller.configure_diagnostics(self.evidence_dir)
        self.tokenizer = self.controller.tokenizer
        self.reward = build_reward(self.config.reward, tokenizer=self.tokenizer)
        try:
            await self.controller.start(self.initial_policy)
            from ..terminal_bench.controller import write_new
            engine = self.controller.engine_receipt()
            write_new(self.evidence_dir / "actor-ownership.json", {
                "actor_process": engine["actor_process"], "workers": engine["workers"]})
        except BaseException:
            await self.controller.close()
            raise

    def rng_state(self):
        # Each request seed is derived from its never-reused identity/policy.
        # Exact sampled seeds live in the durable generation receipts.
        return None

    async def pause(self):
        if self._busy:
            raise RuntimeError("vLLM pause requires a completed group boundary")
        return await self.controller.pause()

    async def activate(self, artifact):
        if self._busy:
            raise RuntimeError("vLLM activation requires a completed group boundary")
        return await self.controller.activate(artifact)

    async def resume(self):
        return await self.controller.resume()

    async def _run(self, function):
        if self._busy:
            raise RuntimeError("vLLM admits one group or evaluation at a time")
        self._busy = True
        future = asyncio.get_running_loop().run_in_executor(self._executor, function)
        try:
            try:
                return await asyncio.shield(future)
            except asyncio.CancelledError:
                # Cancellation of the waiter is not a sampler drain or retry.
                while not future.done():
                    try:
                        await asyncio.shield(future)
                    except asyncio.CancelledError:
                        pass
                future.result()
                raise
        finally:
            self._busy = False

    def _sample(self, record, request_id, *, sampling_seed=None):
        from dsv41_train.rl.rollout import Sample
        from ..terminal_bench.controller import write_new
        result = self.controller.sample_tokens(record.prompt_ids, self.config.max_new_tokens,
            self.config.temperature, request_id, sampling_seed=sampling_seed)
        sample = Sample(result["prompt_ids"], result["response_ids"], result["behavior_log_probs"],
                        0., self.policy.version, result["finish_reason"], result["stop_token_count"])
        write_new(self.evidence_dir / (request_id + ".sampled.json"), {
            "schema_version": 1, "kind": "vllm-actual-prompt-response", "request_id": request_id,
            "policy": self.policy.to_dict(), "context_token_ids": sample.prompt_ids,
            "action_token_ids": sample.response_ids, "behavior_log_probs": sample.behavior_log_probs,
            "finish_reason": sample.finish_reason, "stop_token_count": sample.stop_token_count,
            "reward": None,
            "logprob_convention": "sampled_distribution", "automatic_replay": False,
            "vllm_generation": result["vllm_generation"]})
        scoring = sample.content_ids if getattr(self.reward, "strip_stop_tokens", False) else sample.response_ids
        raw = float(self.reward(record, scoring))
        reward = self.config.truncated_reward if (sample.finish_reason == "length" and
                    self.config.truncated_reward is not None) else raw
        sample = Sample(sample.prompt_ids, sample.response_ids, sample.behavior_log_probs, reward,
                        sample.policy_version, sample.finish_reason, sample.stop_token_count, raw)
        write_new(self.evidence_dir / (request_id + ".reward.json"), {
            "request_id": request_id, "policy_ack": dict(self.ack),
            "reward": sample.reward, "raw_reward": sample.raw_reward})
        return sample, result

    async def execute(self, plan, cancellation):
        if cancellation.cancelled():
            return EpisodeResult(plan.identity, "not_admitted", "not_performed", False,
                                 "not_required", {"reason": "cancelled"}, {})
        if plan.task.benchmark != "prompt-response":
            raise ValueError("vLLM prompt-response backend received another benchmark")
        from dsv41_train.rl.data import PromptRecord
        from dsv41_train.rl.rollout import RolloutGroup
        payload = plan.payload or plan.task.payload["prompt_response"]
        if payload["policy"] != self.ack:
            raise ValueError("admitted group differs from the actual loaded vLLM policy")
        record, group_id = PromptRecord(**payload["record"]), payload["group_id"]

        def generate():
            started, samples, trajectories = time.monotonic(), [], []
            for index in range(self.config.group_size):
                request_id = hashlib.sha256(f"{plan.identity.trial_id}:sample:{index}".encode()).hexdigest()[:32]
                sample, result = self._sample(record, request_id)
                samples.append(sample)
                trace = TurnTrace(1, request_id, sample.prompt_ids, sample.response_ids,
                    sample.behavior_log_probs, self.policy, sample.finish_reason, sample.stop_token_count)
                trajectories.append(TrainingTrajectory(group_id, plan.task.task_id,
                    token_digest(sample.prompt_ids), plan.task.source_sha256, index, [trace], sample.reward,
                    raw_reward=sample.raw_reward,
                    sampling_seed=result["vllm_generation"]["sampling_seed"],
                    evidence={"vllm_generation": result["vllm_generation"]}))
            training = TrainingGroup(group_id, plan.task.task_id, token_digest(record.prompt_ids),
                plan.task.source_sha256, self.policy, self.config.group_size, trajectories)
            return {"group": RolloutGroup(samples, group_id, self.policy.version), "training_group": training,
                "rng_state": None, "started_at": started, "finished_at": time.monotonic(),
                "worker_pid": os.getpid(), "engram_shared": False, "policy_ack": dict(self.ack)}

        packet = await self._run(generate)
        self._packets[group_id] = packet
        return EpisodeResult(plan.identity, "completed", "native_reward", True, "verified",
            {"kind": "prompt-response", "training_group": packet["training_group"].to_dict()}, {},
            timings={"clock_domain": "rollout_process_monotonic",
                     "wall_seconds": packet["finished_at"] - packet["started_at"]},
            model_identity=self.policy.digest, source_sha256=plan.task.source_sha256,
            request_state={"state": "response_received", "policy_ack": dict(self.ack)})

    async def evaluate(self, records, *, step, phase, evaluation_id):
        from dsv41_train.rl.data import PromptRecord
        from dsv41_train.rl.evaluation import evaluation_metrics, sample_record
        if step != self.policy.version:
            raise ValueError("evaluation policy version differs from the actual actor")
        records = [PromptRecord(**record) for record in records]

        def generate():
            samples, rows = [], []
            for index, record in enumerate(records):
                for sample_index in range(self.config.eval_samples_per_prompt):
                    request_id = hashlib.sha256(f"{evaluation_id}:{index}:{sample_index}".encode()).hexdigest()[:32]
                    seed = evaluation_seed(self.config.eval_seed, record.prompt_ids, sample_index)
                    sample, result = self._sample(record, request_id, sampling_seed=seed)
                    samples.append(sample)
                    rows.append({**sample_record(record, sample, step=step, group_id=index,
                        tokenizer=self.tokenizer, phase=phase), "request_id": request_id,
                        "vllm_generation": result["vllm_generation"]})
            return {"metrics": {**evaluation_metrics(samples, records, step=step, phase=phase),
                    "rollout_engine": "vllm", "policy_ack": dict(self.ack)}, "rows": rows}

        # Manager admission stays paused while this explicit evaluation owns
        # the actor, then the actual sampler is drained before returning.
        await self.controller.resume()
        try:
            return await self._run(generate)
        finally:
            await self.controller.pause()

    def take_packet(self, group_id):
        return self._packets.pop(group_id)

    async def close(self):
        if self._busy:
            raise RuntimeError("vLLM backend cannot close while a sampler outcome is pending")
        try:
            if self.controller is not None:
                return await self.controller.close()
        finally:
            self._executor.shutdown(wait=True)
