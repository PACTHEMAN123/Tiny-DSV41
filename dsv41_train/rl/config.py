from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RLConfig:
    max_steps: int = 3
    group_size: int = 4
    groups_per_step: int = 1
    max_new_tokens: int = 4
    temperature: float = 1.0
    clip_eps: float = 0.2
    kl_beta: float = 0.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    max_policy_lag: int = 2
    queue_capacity: int = 2
    reward: str = "token_accuracy"
    learner_devices: tuple[str, ...] = ("cpu",)
    rollout_devices: tuple[str, ...] = ("cpu",)
    rollout_engine: str = "vllm"
    engram_offload: bool = True
    trainable_scope: str = "all"
    qwen_last_position_only: bool = False
    qwen_kv_cache: bool = False
    backprop_chunk_tokens: int = 0
    gradient_checkpointing: bool = False
    dtype: str = "float32"
    seed: int = 42
    num_threads: int = 1
    checkpoint_interval: int = 1
    output_dir: str = "runs/default"
    worker_timeout_seconds: float = 600.0
    max_prompt_tokens: int | None = None
    length_policy: str = "error"
    eos_token_ids: tuple[int, ...] | None = None
    stop_token_sequences: tuple[tuple[int, ...], ...] = ()
    truncated_reward: float | None = None
    eval_interval: int = 0
    eval_samples_per_prompt: int = 1
    eval_seed: int = 1234
    log_samples_per_step: int = 1

    @classmethod
    def from_dict(cls, values: dict) -> "RLConfig":
        values = dict(values)
        counts = ("trainer_num_gpus", "rollout_num_gpus")
        if any(name in values for name in counts):
            if not all(name in values for name in counts):
                raise ValueError("specify both trainer_num_gpus and rollout_num_gpus")
            if any(name in values for name in ("learner_devices", "rollout_devices")):
                raise ValueError("specify GPU counts or explicit device lists, not both")
            trainer, rollout = (values.pop(name) for name in counts)
            if any(type(count) is not int or count < 0 for count in (trainer, rollout)):
                raise ValueError("GPU counts must be nonnegative integers; zero selects CPU")
            values["learner_devices"] = tuple(f"cuda:{i}" for i in range(trainer)) or ("cpu",)
            values["rollout_devices"] = tuple(f"cuda:{i}" for i in range(trainer, trainer + rollout)) or ("cpu",)
        for name in ("learner_devices", "rollout_devices"):
            if name in values:
                values[name] = tuple(values[name])
        if "stop_token_sequences" in values:
            values["stop_token_sequences"] = tuple(tuple(ids) for ids in values["stop_token_sequences"])
        if values.get("eos_token_ids") is not None:
            values["eos_token_ids"] = tuple(values["eos_token_ids"])
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> "RLConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        if self.rollout_engine not in {"vllm", "native-pytorch"}:
            raise ValueError("rollout_engine must be vllm or native-pytorch")
        for name in ("max_steps", "groups_per_step", "max_new_tokens", "queue_capacity",
                     "num_threads", "checkpoint_interval", "eval_samples_per_prompt"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.group_size) is not int or self.group_size < 2:
            raise ValueError("group_size must be at least two")
        if type(self.max_policy_lag) is not int or self.max_policy_lag < 0:
            raise ValueError("max_policy_lag must be a nonnegative integer")
        for name in ("eval_interval", "log_samples_per_step", "eval_seed"):
            if type(getattr(self, name)) is not int or not 0 <= getattr(self, name) < 2**63 - 1:
                raise ValueError(f"{name} must be a nonnegative integer below 2**63 - 1")
        if self.max_prompt_tokens is not None and (type(self.max_prompt_tokens) is not int or self.max_prompt_tokens < 1):
            raise ValueError("max_prompt_tokens must be a positive integer or null")
        if self.length_policy not in {"error", "skip"}:
            raise ValueError("length_policy must be error or skip")
        if self.eos_token_ids is not None and any(type(token) is not int or token < 0
                                                for token in self.eos_token_ids):
            raise ValueError("eos_token_ids must contain nonnegative integer IDs or be null")
        for ids in self.stop_token_sequences:
            if not ids or any(type(token) is not int or token < 0 for token in ids):
                raise ValueError("stop_token_sequences must contain nonempty lists of nonnegative integer IDs")
        if self.truncated_reward is not None and not math.isfinite(self.truncated_reward):
            raise ValueError("truncated_reward must be finite or null")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63 - 1:
            raise ValueError("seed must be an integer in [0, 2**63 - 1)")
        for name in ("temperature", "learning_rate", "max_grad_norm", "worker_timeout_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("kl_beta", "weight_decay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.clip_eps) or not 0 < self.clip_eps < 1:
            raise ValueError("clip_eps must be in (0, 1)")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("dtype must be float32 or bfloat16")
        if self.trainable_scope not in {"all", "lm_head"}:
            raise ValueError("trainable_scope must be all or lm_head")
        if type(self.qwen_last_position_only) is not bool:
            raise ValueError("qwen_last_position_only must be a boolean")
        if type(self.qwen_kv_cache) is not bool:
            raise ValueError("qwen_kv_cache must be a boolean")
        if type(self.backprop_chunk_tokens) is not int or self.backprop_chunk_tokens < 0:
            raise ValueError("backprop_chunk_tokens must be a nonnegative integer")
        if self.qwen_kv_cache and (self.trainable_scope != "lm_head" or not self.qwen_last_position_only
                                   or self.backprop_chunk_tokens):
            raise ValueError("qwen_kv_cache requires lm_head, last-position logits and unchunked turn replay")
        if not self.reward or not self.output_dir:
            raise ValueError("reward and output_dir must not be empty")
        for name in ("engram_offload", "gradient_checkpointing"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        for devices in (self.learner_devices, self.rollout_devices):
            if not devices or len(set(devices)) != len(devices):
                raise ValueError("device lists must be nonempty with no duplicates")
            if devices == ("cpu",):
                continue
            if any(not isinstance(d, str) or not d.startswith("cuda:") or not d[5:].isdigit()
                   for d in devices):
                raise ValueError("use ['cpu'] or explicit CUDA devices such as ['cuda:0', 'cuda:1']")
