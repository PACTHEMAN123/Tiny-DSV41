"""Persistent native PyTorch sampling with an exact policy activation barrier."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

from ..contracts import EpisodeResult


class PyTorchRLBackend:
    """One model and CPU sampling generator, owned by its spawned process.

    A whole native GRPO group is one atomic episode. Keeping the existing serial
    sampler preserves its categorical draws and RNG progression exactly.
    """

    def __init__(self, model_dir, model_config, config, random_init, *,
                 engram_weights=None, initial_policy, rng_state=None):
        if config.rollout_engine != "native-pytorch":
            raise ValueError("native sampling requires explicit rollout_engine=native-pytorch")
        self.model_dir, self.model_config, self.config = model_dir, model_config, config
        self.random_init, self.engram_weights = random_init, engram_weights
        self.initial_policy, self.initial_rng = initial_policy, rng_state
        self._packets, self._busy = {}, False
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pytorch-policy")

    async def start(self):
        import torch
        from dsv41_train.rl.family import build_policy, policy_lora_config
        from dsv41_train.rl.data import build_reward
        from dsv41_train.rl.rollout import generation_options
        from dsv41_train.rl.tokenizer import ModelTokenizer

        torch.set_num_threads(self.config.num_threads)
        torch.manual_seed(self.config.seed)
        if self.config.trainable_scope == "lora":
            from dsv41_train.rl.checkpoint import add_lora_identity, run_identity, validate_identity
            actual = add_lora_identity(run_identity(self.model_dir,
                Path(self.model_dir) / "config.json", self.random_init), self.config, self.model_config)
            actual.pop("data_sha256")
            validate_identity({key: self.initial_policy["identity"].get(key) for key in actual}, actual)
        self.model = build_policy(self.model_dir, self.model_config,
            random_init=self.random_init, devices=self.config.rollout_devices,
            dtype=getattr(torch, self.config.dtype), engram_weights=self.engram_weights,
            engram_offload=self.config.engram_offload, trainable_scope=self.config.trainable_scope,
            lora_config=policy_lora_config(self.config, self.model_config),
            initial_adapter=self.config.initial_adapter)
        if self.config.qwen_last_position_only:
            if getattr(self.model_config, "model_type", None) != "qwen3_moe":
                raise ValueError("last-position logits require native Qwen")
            self.model.qwen_last_position_only = True
        if self.config.qwen_kv_cache:
            if getattr(self.model_config, "model_type", None) != "qwen3_moe":
                raise ValueError("request KV-cache sampling requires native Qwen")
            self.model.qwen_kv_cache = True
        self.model.eval()
        tokenizer = ModelTokenizer(self.model_dir) if (Path(self.model_dir) / "tokenizer.json").is_file() else None
        self.reward = build_reward(self.config.reward, tokenizer=tokenizer)
        self.generation = generation_options(self.model_dir, self.model_config, self.config)
        self.generator = torch.Generator(device="cpu").manual_seed(self.config.seed + 1)
        if self.initial_rng is not None:
            self.generator.set_state(torch.tensor(list(self.initial_rng), dtype=torch.uint8))
        self.activate(self.initial_policy)

    def activate(self, artifact):
        if self._busy:
            raise RuntimeError("policy activation requires a drained group boundary")
        from dsv41_train.rl.checkpoint import load_policy_artifact
        from ..training import PolicyRef

        if artifact["sampling"] != {"temperature": self.config.temperature, "top_p": 1, "top_k": 0}:
            raise ValueError("policy sampling identity differs from native categorical sampling")
        from dsv41_train.rl.qwen_policy import LAST_POSITION_LOGITS_PATH
        profile = LAST_POSITION_LOGITS_PATH if self.config.qwen_last_position_only else None
        if artifact["identity"].get("qwen_logits_path") != profile:
            raise ValueError("artifact differs from the selected Qwen logits profile")
        from .qwen_kv_cache import QWEN_KV_SAMPLING_PATH
        sampling_profile = QWEN_KV_SAMPLING_PATH if self.config.qwen_kv_cache else None
        if artifact["identity"].get("qwen_sampling_path") != sampling_profile:
            raise ValueError("artifact differs from the selected Qwen sampling profile")
        if hasattr(self, "active_artifact"):
            if any(artifact[key] != self.active_artifact[key] for key in ("lineage", "identity")):
                raise ValueError("activation cannot change run lineage or base identity")
            if (artifact["version"] < self.policy.version or
                    (artifact["version"] == self.policy.version and artifact["digest"] != self.policy.digest)):
                raise ValueError("activation cannot roll back or replace a policy version")
        ack = load_policy_artifact(self.model, artifact)
        expected = {key: artifact[key] for key in ("version", "digest", "lineage")}
        if ack != expected:
            raise ValueError("native policy acknowledgement differs from requested artifact")
        identity = hashlib.sha256(json.dumps(artifact["identity"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        identities = {"run_identity_sha256": identity}
        if profile:
            identities["qwen_logits_path"] = profile
        if sampling_profile:
            identities["qwen_sampling_path"] = sampling_profile
        self.policy = PolicyRef(artifact["version"], artifact["digest"], artifact["lineage"],
                                identities, artifact["sampling"])
        self.active_artifact, self.ack = dict(artifact), ack
        self.model.assert_engram_invariants()
        return ack

    def rng_state(self):
        return bytes(self.generator.get_state().tolist())

    async def execute(self, plan, cancellation):
        if cancellation.cancelled():
            return EpisodeResult(plan.identity, "not_admitted", "not_performed", False,
                                 "not_required", {"reason": "cancelled"}, {})
        if plan.task.benchmark != "prompt-response" or self._busy:
            raise ValueError("native PyTorch admits one prompt-response group at a time")
        from dsv41_train.rl.data import PromptRecord
        from dsv41_train.rl.rollout import generate_group
        from ..training import from_rollout_group

        payload = plan.payload or plan.task.payload["prompt_response"]
        record = PromptRecord(**payload["record"])
        group_id = payload["group_id"]
        expected = payload["policy"]
        if expected != self.ack:
            raise ValueError("admitted group policy differs from the loaded native acknowledgement")
        self._busy = True
        loop = asyncio.get_running_loop()

        def sample():
            started = time.monotonic()
            group = generate_group(self.model, record, self.config.group_size,
                self.config.max_new_tokens, self.config.temperature, **self.generation,
                policy_version=self.policy.version, group_id=group_id,
                generator=self.generator, reward_fn=self.reward)
            training = from_rollout_group(group, self.policy, plan.task.task_id,
                reset_sha256=plan.task.source_sha256,
                request_prefix=plan.identity.trial_id)
            packet = {"group": group, "training_group": training,
                      "rng_state": self.rng_state(), "started_at": started,
                      "finished_at": time.monotonic(), "worker_pid": os.getpid(),
                      "engram_shared": self.model.engram_shared,
                      "policy_ack": dict(self.ack)}
            return packet

        future = loop.run_in_executor(self._executor, sample)
        try:
            try:
                packet = await asyncio.shield(future)
            except asyncio.CancelledError:
                cancellation.cancel("pytorch_waiter_cancelled")
                while not future.done():
                    try:
                        await asyncio.shield(future)
                    except asyncio.CancelledError:
                        pass
                packet = future.result()
            self._packets[group_id] = packet
            return EpisodeResult(plan.identity, "completed", "native_reward", True, "verified",
                {"kind": "prompt-response", "training_group": packet["training_group"].to_dict()}, {},
                timings={"clock_domain": "rollout_process_monotonic",
                         "wall_seconds": packet["finished_at"] - packet["started_at"]},
                model_identity=self.policy.digest, source_sha256=plan.task.source_sha256,
                request_state={"state": "response_received", "policy_ack": dict(self.ack)})
        finally:
            self._busy = False

    def take_packet(self, group_id):
        return self._packets.pop(group_id)

    async def close(self):
        if self._busy:
            raise RuntimeError("native backend cannot close before its actual sampler drains")
        self._executor.shutdown(wait=True)
