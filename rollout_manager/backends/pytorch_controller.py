"""In-process Qwen serving for benchmark training; no network listener.

Generation owns a real serial categorical sample, never probabilities inferred
from decoded text. A cancelled caller must still wait for ``pause`` to prove
actual sampler drain before loading weights. The caller owns durable transport.
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time

from ..benchmark_training import artifact_policy
from ..generation_training import TrainingGenerationContract, evidence_digest
# Retain legacy parser imports while both serving backends use the shared grammar.
from ..model_completion import SHELL_PARAMETERS, _TOOL_CALL, _strict_object, shell_completion


_REQUEST_ID = re.compile(r"[a-f0-9]{32}\Z")


def _tokenizer_identities(directory, tokenizer):
    template = tokenizer._template_source
    if not isinstance(template, str) or not template.strip():
        raise ValueError("benchmark serving requires the mounted chat template")
    return {"tokenizer_sha256": hashlib.sha256((directory / "tokenizer.json").read_bytes()).hexdigest(),
            "chat_template_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest()}


async def _owned_thread(function, *args, **kwargs):
    """Waiter cancellation cannot outlive an owned load or drain operation."""
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                pass
        worker.result()
        raise


class PyTorchPolicyController:
    """One mounted Qwen policy, with a thread-safe admission and load barrier.

    ``generate(payload, request_id)`` is synchronous and may be dispatched by a
    caller's existing transport thread. Control methods are asynchronous for
    BenchmarkTrainingProvider. The model is built once; newer immutable policy
    artifacts load only after a successful pause receipt. No retries or replay.
    """

    def __init__(self, model_dir, config, *, served_model):
        from dsv41_train.rl.family import load_model_config
        from dsv41_train.rl.tokenizer import ModelTokenizer

        config.validate()
        if not isinstance(served_model, str) or not served_model.strip():
            raise ValueError("an explicit served model name is required")
        self.model_dir, self.config, self.served_model = Path(model_dir), config, served_model
        self.model_config = load_model_config(self.model_dir)
        if self.model_config.model_type != "qwen3_moe":
            raise ValueError("benchmark policy controller currently supports only native Qwen3 MoE")
        self.tokenizer = ModelTokenizer(self.model_dir)
        self.identities = _tokenizer_identities(self.model_dir, self.tokenizer)
        if config.qwen_last_position_only:
            from dsv41_train.rl.qwen_policy import LAST_POSITION_LOGITS_PATH
            self.identities["qwen_logits_path"] = LAST_POSITION_LOGITS_PATH
        if config.qwen_kv_cache:
            from .qwen_kv_cache import QWEN_KV_SAMPLING_PATH
            self.identities["qwen_sampling_path"] = QWEN_KV_SAMPLING_PATH
        self._condition, self._sampler_lock = threading.Condition(), threading.Lock()
        self._state, self._active, self._drained = "new", 0, False
        self._seen = set()
        self.model, self.policy, self.ack = None, None, None
        self._control_lock = asyncio.Lock()
        self.diagnostics_dir = None

    def configure_diagnostics(self, directory):
        """The owning shim selects one existing private directory before start."""
        path = Path(directory)
        info = path.lstat()
        if (self._state != "new" or self.diagnostics_dir is not None or not path.is_absolute()
                or ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents))
                or not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077):
            raise ValueError("native diagnostics require one same-owner private directory before start")
        self.diagnostics_dir = path

    def _record_sample(self, request_id, payload, policy, sample, text, generation_metrics):
        if self.diagnostics_dir is None:
            return
        from ..terminal_bench.controller import write_new
        write_new(self.diagnostics_dir / (request_id + ".sampled.json"), {
            "schema_version": 1, "kind": "native-actual-sampled-action", "request_id": request_id,
            "request_payload_sha256": evidence_digest(payload), "policy": policy.to_dict(),
            "decoded_action_text": text, "context_token_ids": sample.prompt_ids,
            "action_token_ids": sample.response_ids, "behavior_log_probs": sample.behavior_log_probs,
            "sampling_finish_reason": sample.finish_reason, "stop_token_count": sample.stop_token_count,
            "logprob_convention": "sampled_distribution", "completion_parsed": False,
            "benchmark_reward": None, "automatic_replay": False,
            "native_generation": generation_metrics})
        descriptor = os.open(self.diagnostics_dir, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def policy_factory(self, artifact):
        return artifact_policy(artifact, identities=self.identities)

    def capabilities(self):
        with self._condition:
            if self.policy is None or self._state not in {"running", "paused"}:
                raise RuntimeError("serving capabilities require a successfully loaded policy")
            return self._capabilities(self.policy)

    @staticmethod
    def _capabilities(policy):
        return {"schema_version": 1, "policy": policy.to_dict(),
            **{name: True for name in TrainingGenerationContract.REQUIRED},
            "logprob_convention": "sampled_distribution"}

    def _check_artifact(self, artifact):
        from dsv41_train.rl.checkpoint import file_sha256
        from dsv41_train.rl.tokenizer import ModelTokenizer

        candidate = self.policy_factory(artifact)
        if dict(candidate.sampling) != {"temperature": self.config.temperature, "top_p": 1, "top_k": 0}:
            raise ValueError("artifact sampling differs from the controller")
        identity = artifact["identity"]
        if identity.get("qwen_logits_path") != self.identities.get("qwen_logits_path"):
            raise ValueError("artifact differs from the selected Qwen logits profile")
        if identity.get("qwen_sampling_path") != self.identities.get("qwen_sampling_path"):
            raise ValueError("artifact differs from the selected Qwen sampling profile")
        if (identity.get("random_init") is not False or
                identity.get("trainable_scope") != self.config.trainable_scope or
                identity.get("model_config_sha256") != file_sha256(self.model_dir / "config.json")):
            raise ValueError("artifact differs from the mounted pretrained configuration or trainable scope")
        files = []
        for path in sorted(self.model_dir.rglob("*")):
            if path.suffix in (".safetensors", ".bin") or path.name.endswith(".index.json"):
                info = path.stat()
                files.append([str(path.relative_to(self.model_dir)), info.st_size, info.st_mtime_ns])
        if not files or identity.get("base_files") != files:
            raise ValueError("artifact mounted base-file inventory differs")
        text_assets = {name: file_sha256(self.model_dir / name) for name in
            ("tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json")
            if (self.model_dir / name).is_file()}
        if (identity.get("engram_mapping_sha256", {}).get("tokenizer.json") != self.identities["tokenizer_sha256"] or
                identity.get("text_assets_sha256", {}) != text_assets):
            raise ValueError("artifact tokenizer, chat template or generation metadata differs")
        if any(self.identities[key] != value for key, value in
               _tokenizer_identities(self.model_dir, ModelTokenizer(self.model_dir)).items()):
            raise ValueError("mounted tokenizer or selected chat template changed")
        if self.policy is not None and (candidate.lineage != self.policy.lineage or
                candidate.version <= self.policy.version or identity != self.active_artifact["identity"] or
                dict(candidate.identities) != dict(self.policy.identities)):
            raise ValueError("activation requires a newer same-lineage policy with identical base identity")
        return candidate

    def _load(self, artifact, *, initial=False):
        import torch
        from dsv41_train.rl.checkpoint import load_policy_artifact
        from dsv41_train.rl.family import build_policy
        from dsv41_train.rl.rollout import generation_options

        artifact = copy.deepcopy(artifact)
        candidate = self._check_artifact(artifact)
        try:
            if initial:
                self.model = build_policy(self.model_dir, self.model_config, random_init=False,
                    devices=self.config.rollout_devices, dtype=getattr(torch, self.config.dtype),
                    trainable_scope=self.config.trainable_scope)
                self.model.qwen_last_position_only = self.config.qwen_last_position_only
                self.model.qwen_kv_cache = self.config.qwen_kv_cache
                self.model.eval()
                self.generation = generation_options(self.model_dir, self.model_config, self.config)
                self.generator = torch.Generator(device="cpu").manual_seed(self.config.seed + 1)
            ack = load_policy_artifact(self.model, artifact)
            if ack != {key: artifact[key] for key in ("version", "digest", "lineage")}:
                raise ValueError("loaded policy acknowledgement differs from the immutable artifact")
            self.model.assert_engram_invariants()
        except BaseException:
            with self._condition:
                self._state = "failed"  # A partial weight copy must never retain an old serving receipt.
            raise
        with self._condition:
            self.policy, self.ack, self.active_artifact = candidate, dict(ack), artifact
            self._state = "running" if initial else "paused"
        return dict(ack)

    async def start(self, artifact):
        async with self._control_lock:
            with self._condition:
                if self._state != "new":
                    raise RuntimeError("controller start is single-use")
                self._state = "loading"
            try:
                return await _owned_thread(self._load, artifact, initial=True)
            except BaseException:
                with self._condition:
                    self._state = "failed"
                raise

    def _wait_drained(self):
        with self._condition:
            while self._active:
                self._condition.wait()

    async def pause(self):
        async with self._control_lock:
            with self._condition:
                if self._state not in {"running", "paused"}:
                    raise RuntimeError("cannot pause an unloaded or failed policy")
                self._state, self._drained = "paused", False
            await _owned_thread(self._wait_drained)
            with self._condition:
                self._drained = True
                return {**self.ack, "drain_complete": True, "active_requests": 0}

    async def activate(self, artifact):
        async with self._control_lock:
            with self._condition:
                if self._state != "paused" or not self._drained or self._active:
                    raise RuntimeError("activation requires a proven paused sampler drain")
                self._drained = False
            try:
                return await _owned_thread(self._load, artifact)
            except asyncio.CancelledError:
                with self._condition:
                    self._state = "failed"
                raise

    async def resume(self):
        async with self._control_lock:
            with self._condition:
                if self._state != "paused" or self._active:
                    raise RuntimeError("resume requires an unfailed paused policy")
                self._state, self._drained = "running", False
                return dict(self.ack)

    async def close(self):
        async with self._control_lock:
            with self._condition:
                self._state, self._drained = "closed", False
            try:
                await _owned_thread(self._wait_drained)
            finally:
                self.model = None
            return {**(self.ack or {}), "drain_complete": True, "active_requests": 0}

    def _context(self, payload, policy):
        TrainingGenerationContract(policy, self.capabilities()).check_payload(payload)
        allowed = {"model", "messages", "tools", "tool_choice", "parallel_tool_calls", "chat_template_kwargs",
                   "stream", "temperature", "max_tokens", "top_p", "top_k", "n"}
        if (set(payload) - allowed or payload.get("model") != self.served_model or
                payload.get("tool_choice") != "auto" or payload.get("parallel_tool_calls") is not False or
                payload.get("chat_template_kwargs") != {"enable_thinking": False}):
            raise ValueError("unsupported benchmark generation payload")
        tools = payload.get("tools")
        if not isinstance(tools, list) or len(tools) != 1:
            raise ValueError("exactly one shell tool schema is required")
        tool = tools[0]
        if (not isinstance(tool, dict) or set(tool) != {"type", "function"} or tool["type"] != "function"
                or not isinstance(tool["function"], dict) or tool["function"].get("name") != "shell"
                or tool["function"].get("parameters") != SHELL_PARAMETERS):
            raise ValueError("unsupported shell tool schema")
        prompt = self.tokenizer.encode_conversation(payload.get("messages"), tools=tools,
                                                    chat_template_kwargs=payload["chat_template_kwargs"])
        budget = payload["max_tokens"]
        if (budget > self.config.max_new_tokens or
                self.config.max_prompt_tokens is not None and len(prompt) > self.config.max_prompt_tokens or
                len(prompt) + budget > self.model_config.max_position_embeddings):
            raise ValueError("request exceeds the configured action or untruncated context budget")
        return prompt

    def generate(self, payload, request_id):
        """Return one actual completion plus bound training evidence, never replay."""
        from dsv41_train.rl.data import PromptRecord
        from dsv41_train.rl.rollout import generate_group

        if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
            raise ValueError("generation requires a durable 32-hex request identity")
        payload = copy.deepcopy(payload)
        with self._condition:
            if self._state != "running":
                raise RuntimeError("controller admission is paused, closed or unqualified")
            if request_id in self._seen:
                raise ValueError("generation request identity was already admitted; no replay")
            policy = self.policy
            try:
                prompt = self._context(payload, policy)
            except Exception as error:
                error.native_generation_stage = "payload_validation"
                raise
            self._seen.add(request_id)
            self._active += 1
        stage = "sampler_admission"
        try:
            with self._sampler_lock:
                stage = "model_invariants_before_sampling"
                self.model.assert_engram_invariants()
                stage = "sampling"
                sampler = self.model
                if self.config.qwen_kv_cache:
                    from .qwen_kv_cache import QwenCachedSampler
                    sampler = QwenCachedSampler(self.model)
                sampling_started = time.monotonic()
                sample = generate_group(sampler, PromptRecord(prompt), 1, payload["max_tokens"],
                    policy.sampling["temperature"], policy_version=policy.version, group_id=0,
                    generator=self.generator, reward_fn=lambda record, ids: 0., **self.generation).samples[0]
                generation_metrics = {"elapsed_seconds": time.monotonic() - sampling_started,
                    "clock_domain": "native_process_monotonic", "cache_enabled": self.config.qwen_kv_cache,
                    "completion_barrier": "sampled_token_probabilities_copied_to_cpu",
                    "context_tokens": len(prompt), "action_tokens": len(sample.response_ids)}
                if self.config.qwen_kv_cache:
                    generation_metrics.update(sampler.cache.timings)
                stage = "sample_decode"
                text = self.tokenizer.decode(sample.content_ids, skip_special_tokens=False)
                stage = "sample_persistence"
                self._record_sample(request_id, payload, policy, sample, text, generation_metrics)
                stage = "model_invariants_after_sampling"
                self.model.assert_engram_invariants()
                stage = "shell_completion"
                message, finish = shell_completion(text, request_id, truncated=sample.finish_reason == "length")
                completion = {"id": "chatcmpl-" + request_id, "object": "chat.completion",
                    "model": self.served_model, "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(sample.response_ids),
                              "total_tokens": len(prompt) + len(sample.response_ids)}}
                evidence_finish = "tool_calls" if finish == "tool_calls" else sample.finish_reason
                completion["training_evidence"] = {"schema_version": 1, "request_id": request_id,
                    "request_payload_sha256": evidence_digest(payload), "response_sha256": evidence_digest(completion),
                    "policy": policy.to_dict(), "context_token_ids": sample.prompt_ids,
                    "action_token_ids": sample.response_ids, "behavior_log_probs": sample.behavior_log_probs,
                    "finish_reason": evidence_finish, "stop_token_count": sample.stop_token_count,
                    "context_truncation": "none", "logprob_convention": "sampled_distribution",
                    "sampling_finish_reason": sample.finish_reason}
                # Validate the exact object that will leave the controller, before a tool can execute.
                stage = "training_evidence_validation"
                TrainingGenerationContract(policy, self._capabilities(policy)).validate(payload, completion, request_id, 1)
                return completion
        except Exception as error:
            error.native_generation_stage = stage
            raise
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()
