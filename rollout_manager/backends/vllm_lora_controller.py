"""Owned DeepSeek LoRA rollout control; no trainer or checkpoint publication."""

from __future__ import annotations
import asyncio
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import threading
import time
from .pytorch_controller import _owned_thread
from .vllm_controller import OwnedVLLMActor
from .vllm_lora_transport import policy_ack, validate_update


@dataclass(frozen=True)
class LoRARolloutConfig:
    deadline_unix: float | None = None
    operation_timeout_seconds: int = 7200
    rollout_devices: tuple[str, ...] = tuple(f"cuda:{index}" for index in range(8))
    max_model_len: int = 512
    max_num_seqs: int = 1
    gpu_memory_utilization: float = 0.85
    seed: int = 42
    dtype: str = "bfloat16"

    def validate(self):
        if self.deadline_unix is not None and (
            type(self.deadline_unix) not in (int, float)
            or not math.isfinite(self.deadline_unix)
            or self.deadline_unix <= time.time()
        ):
            raise ValueError("LoRA deadline must be a future finite timestamp or None")
        if (
            type(self.operation_timeout_seconds) is not int
            or not 1 <= self.operation_timeout_seconds <= 7200
        ):
            raise ValueError("each LoRA operation requires a finite timeout in 1..7200 seconds")
        if (
            not self.rollout_devices
            or len(self.rollout_devices) > 8
            or len(set(self.rollout_devices)) != len(self.rollout_devices)
            or any(
                (
                    not isinstance(device, str) or not re.fullmatch("cuda:\\d+", device)
                    for device in self.rollout_devices
                )
            )
        ):
            raise ValueError("LoRA rollout requires distinct explicit CUDA devices")
        if (
            type(self.max_model_len) is not int
            or not 2 <= self.max_model_len <= 4096
            or type(self.max_num_seqs) is not int
            or (not 1 <= self.max_num_seqs <= 32)
            or (type(self.seed) is not int)
            or (not 0 <= self.seed < 2**63 - 1)
            or (self.dtype != "bfloat16")
            or (type(self.gpu_memory_utilization) not in (int, float))
            or (not math.isfinite(self.gpu_memory_utilization))
            or (not 0 < self.gpu_memory_utilization <= 0.95)
        ):
            raise ValueError("invalid bounded DeepSeek LoRA rollout configuration")


class VLLMLoRAController:
    """One existing actor with a direct tensor activation and admission barrier.

    ``start`` loads the frozen mounted model and stays paused. Activate version
    zero, then resume and sample. Later activations require a completed pause.
    This rollout-only interface does not claim adapter backward or GRPO support.
    """

    def __init__(self, model_dir, config, *, work_dir):
        config.validate()
        self.model_dir, self.config = (Path(model_dir).resolve(), config)
        values = json.loads((self.model_dir / "config.json").read_text())
        if values.get("model_type") != "deepseek_v41":
            raise ValueError("direct LoRA rollout requires the mounted DeepSeek V4.1 model")
        self.base_identity = {
            key: hashlib.sha256((self.model_dir / name).read_bytes()).hexdigest()
            for key, name in (
                ("config_sha256", "config.json"),
                ("index_sha256", "model.safetensors.index.json"),
            )
        }
        generation_path = self.model_dir / "generation_config.json"
        generation = json.loads(generation_path.read_text()) if generation_path.is_file() else {}
        eos = generation.get("eos_token_id", values.get("eos_token_id"))
        self.eos_token_ids = [eos] if type(eos) is int else eos
        if (
            not isinstance(self.eos_token_ids, list)
            or not self.eos_token_ids
            or any((type(token) is not int or token < 0 for token in self.eos_token_ids))
        ):
            raise ValueError("DeepSeek LoRA rollout requires explicit mounted EOS token IDs")
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
        self._condition, self._sampler_lock = (threading.Condition(), threading.Lock())
        self._control_lock = asyncio.Lock()
        self._state, self._active = ("new", 0)
        self._seen, self._used_adapter_ids = (set(), set())
        self.actor = self.ack = self.active_manifest = self.activation_receipts = None
        self._engine_receipt = None

    def _timeout(self):
        if self.config.deadline_unix is None:
            return self.config.operation_timeout_seconds
        remaining = self.config.deadline_unix - time.time()
        if remaining <= 0:
            raise TimeoutError("LoRA rollout deadline expired")
        return min(remaining, self.config.operation_timeout_seconds)

    def _check_engine(self, engine, manifest=None):
        if (
            engine.get("serving_profile") != "deepseek-lora-v1"
            or engine.get("base_identity") != self.base_identity
            or engine.get("tensor_parallel_size") != len(self.config.rollout_devices)
        ):
            raise RuntimeError("LoRA actor receipt differs from the requested mounted deployment")
        workers = engine.get("workers", [])
        uuids = [worker.get("device_uuid") for worker in workers]
        if (
            len(workers) != len(self.config.rollout_devices)
            or any((not value for value in uuids))
            or len(set(uuids)) != len(uuids)
        ):
            raise RuntimeError(
                "LoRA actor receipt lacks distinct complete TP worker acknowledgements"
            )
        if manifest is not None:
            expected = policy_ack(manifest)
            if (
                engine.get("loaded_policy_ack") != expected
                or engine.get("adapter_id") != manifest["adapter_id"]
            ):
                raise RuntimeError(
                    "LoRA engine receipt does not acknowledge the requested immutable adapter"
                )
            for worker in workers:
                if (
                    worker.get("base_identity") != self.base_identity
                    or worker.get("policy_ack") != expected
                    or worker.get("adapter_id") != manifest["adapter_id"]
                    or (worker.get("readback_digest") != manifest["digest"])
                ):
                    raise RuntimeError(
                        "LoRA worker readback acknowledgement differs from the trainer payload"
                    )

    def _start(self):
        ttl = max(1, int(self._timeout()))
        actor_deadline = self.config.deadline_unix
        bootstrap = {
            "serving_profile": "deepseek-lora-v1",
            "model_dir": str(self.model_dir),
            "dtype": "bfloat16",
            "seed": self.config.seed,
            "max_model_len": self.config.max_model_len,
            "max_num_seqs": self.config.max_num_seqs,
            "eos_token_ids": self.eos_token_ids,
            "ttl_seconds": ttl,
            "deadline_unix": actor_deadline,
            "tensor_parallel_size": len(self.config.rollout_devices),
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "base_identity": self.base_identity,
        }
        try:
            self.actor = OwnedVLLMActor(
                self.model_dir,
                self.config,
                self.work_dir / "actor",
                max_model_len=self.config.max_model_len,
                bootstrap_override=bootstrap,
            )
            self._check_engine(self.actor.receipt)
            self._engine_receipt = copy.deepcopy(self.actor.receipt)
            self._timeout()
            self._state = "paused"
            return self.engine_receipt()
        except BaseException:
            self._state = "failed"
            raise

    async def start(self):
        async with self._control_lock:
            if self._state != "new":
                raise RuntimeError("LoRA actor start is single-use")
            self._state = "loading"
            return await _owned_thread(self._start)

    def _activate(self, manifest, payload):
        try:
            result = self.actor.call(
                "load_lora", tensor_update=(manifest, payload), timeout=min(600, self._timeout())
            )
            self._check_engine(result["engine"], manifest)
            expected = policy_ack(manifest)
            receipts = result.get("activation_receipts", [])
            if (
                result.get("ack") != expected
                or result.get("adapter_id") != manifest["adapter_id"]
                or len(receipts) != len(self.config.rollout_devices)
                or any(
                    (
                        item.get("policy_ack") != expected
                        or item.get("readback_digest") != manifest["digest"]
                        for item in receipts
                    )
                )
            ):
                raise RuntimeError(
                    "LoRA activation did not establish every actual worker tensor readback"
                )
            self.ack, self.active_manifest = (expected, manifest)
            self.activation_receipts = copy.deepcopy(receipts)
            self._engine_receipt = copy.deepcopy(result["engine"])
            self._used_adapter_ids.add(manifest["adapter_id"])
            return dict(expected)
        except BaseException:
            self._state, self.ack = ("failed", None)
            raise

    async def activate(self, manifest, payload):
        manifest = copy.deepcopy(manifest)
        validate_update(manifest, payload)
        async with self._control_lock:
            with self._condition:
                if self._state != "paused" or self._active:
                    raise RuntimeError("LoRA activation requires a proven paused and drained actor")
                expected_version = 0 if self.ack is None else self.ack["version"] + 1
                if (
                    manifest["version"] != expected_version
                    or manifest["adapter_id"] in self._used_adapter_ids
                    or (self.ack is not None and manifest["lineage"] != self.ack["lineage"])
                    or (manifest["base_identity"] != self.base_identity)
                    or (manifest["peft"] != {"r": 8, "lora_alpha": 8, "target_modules": ["wq_a"]})
                ):
                    raise ValueError(
                        "LoRA policy requires the next version, unused ID, and unchanged base/profile"
                    )
            self._timeout()
            return await _owned_thread(self._activate, manifest, payload)

    def _wait_drained(self):
        with self._condition:
            while self._active:
                self._condition.wait(timeout=0.2)

    async def pause(self):
        async with self._control_lock:
            with self._condition:
                if self._state != "running":
                    raise RuntimeError("only a running healthy LoRA actor can be paused")
                self._state = "pausing"
            await _owned_thread(self._wait_drained)
            with self._condition:
                if self._state == "failed":
                    raise RuntimeError(
                        "unknown LoRA inference outcome cannot establish a successful pause"
                    )
                self._state = "paused"
            return {"drain_complete": True, "policy_ack": dict(self.ack)}

    async def resume(self):
        async with self._control_lock:
            self._timeout()
            with self._condition:
                if self._state != "paused" or self.ack is None:
                    raise RuntimeError("LoRA resume requires an acknowledged activated policy")
                self._state = "running"
            return dict(self.ack)

    @contextmanager
    def _healthy_sampler(self):
        with self._sampler_lock:
            with self._condition:
                if self._state == "failed":
                    raise RuntimeError(
                        "a previous LoRA inference failed; queued requests cannot dispatch"
                    )
            try:
                yield
            except BaseException:
                with self._condition:
                    self._state = "failed"
                raise

    def sample_tokens(self, prompt_ids, max_tokens, temperature, request_id, *, sampling_seed):
        return self.sample_batch(
            [
                {
                    "prompt_ids": list(prompt_ids),
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "request_id": request_id,
                    "sampling_seed": sampling_seed,
                }
            ]
        )[0]

    def sample_batch(self, specs):
        """Dispatch one immutable group; reject stale or replayed requests."""
        from .vllm_actor import lora_batch_id
        from .vllm_lora_remote import _specs

        specs = copy.deepcopy(_specs(specs))
        if len(specs) > self.config.max_num_seqs or any(
            len(spec["prompt_ids"]) + spec["max_tokens"] > self.config.max_model_len
            for spec in specs
        ):
            raise ValueError(
                "LoRA batch exceeds configured sequence/context capacity; no serial fallback"
            )
        self._timeout()
        request_ids = [spec["request_id"] for spec in specs]
        with self._condition:
            if self._state != "running" or any(
                request_id in self._seen for request_id in request_ids
            ):
                raise RuntimeError(
                    "LoRA sampling admission is paused, failed, or would replay a request"
                )
            manifest = copy.deepcopy(self.active_manifest)
            self._seen.update(request_ids)
            self._active += len(specs)
        try:
            with self._healthy_sampler():
                expected = policy_ack(manifest)
                batch_id = lora_batch_id(specs, expected)
                reply = self.actor.call(
                    "generate_batch",
                    specs=specs,
                    batch_id=batch_id,
                    policy_ack=expected,
                    adapter_id=manifest["adapter_id"],
                    timeout=self._timeout(),
                )
                results, engine = reply["results"], reply["engine_receipt"]
                self._check_engine(engine, manifest)
                if (
                    reply.get("batch_id") != batch_id
                    or not isinstance(results, list)
                    or len(results) != len(specs)
                ):
                    raise RuntimeError("LoRA batch membership differs")
                engine_ids = []
                for index, (spec, result) in enumerate(zip(specs, results)):
                    metrics = result["generation_metrics"]
                    engine_id = metrics.get("engine_request_id")
                    if (
                        metrics.get("policy_ack") != expected
                        or metrics.get("adapter_request_id") != manifest["adapter_id"]
                        or result.get("logprob_convention") != "sampled_distribution"
                        or result.get("context_token_ids") != spec["prompt_ids"]
                        or metrics.get("request_id") != spec["request_id"]
                        or metrics.get("sampling_seed") != spec["sampling_seed"]
                        or metrics.get("batch_request_index") != index
                        or not isinstance(engine_id, str)
                        or not engine_id.isdecimal()
                    ):
                        raise RuntimeError(
                            "LoRA inference is not bound to its admitted ordered policy batch"
                        )
                    engine_ids.append(engine_id)
                if len(set(engine_ids)) != len(engine_ids) or list(map(int, engine_ids)) != sorted(
                    map(int, engine_ids)
                ):
                    raise RuntimeError("LoRA batch output request identity/order differs")
                self._engine_receipt = copy.deepcopy(engine)
                return results
        except BaseException:
            with self._condition:
                self._state = "failed"
            raise
        finally:
            with self._condition:
                self._active -= len(specs)
                self._condition.notify_all()

    def engine_receipt(self):
        if self._state not in {"running", "paused", "pausing"} or self._engine_receipt is None:
            raise RuntimeError("LoRA engine receipt requires actual healthy startup/activation")
        return copy.deepcopy(self._engine_receipt)

    async def close(self):
        async with self._control_lock:
            if self._state == "closed":
                return {"closed": True, "drain_complete": True}
            with self._condition:
                self._state = "closing"
            await _owned_thread(self._wait_drained)
            try:
                if self.actor is not None:
                    await _owned_thread(self.actor.close)
                self._state = "closed"
                return {"closed": True, "drain_complete": True}
            except BaseException:
                self._state = "failed"
                raise
