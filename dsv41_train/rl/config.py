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
    dtype: str = "float32"
    seed: int = 42
    num_threads: int = 1
    checkpoint_interval: int = 1
    output_dir: str = "runs/default"
    worker_timeout_seconds: float = 600.0

    @classmethod
    def from_dict(cls, values: dict) -> "RLConfig":
        values = dict(values)
        for name in ("learner_devices", "rollout_devices"):
            if name in values:
                values[name] = tuple(values[name])
        result = cls(**values)
        result.validate()
        return result

    @classmethod
    def from_json(cls, path: str | Path) -> "RLConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        for name in ("max_steps", "groups_per_step", "max_new_tokens", "queue_capacity",
                     "num_threads", "checkpoint_interval"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.group_size) is not int or self.group_size < 2:
            raise ValueError("group_size must be at least two")
        if type(self.max_policy_lag) is not int or self.max_policy_lag < 0:
            raise ValueError("max_policy_lag must be a nonnegative integer")
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
        if not self.reward or not self.output_dir:
            raise ValueError("reward and output_dir must not be empty")
        for devices in (self.learner_devices, self.rollout_devices):
            if len(devices) != 1:
                raise ValueError("RL requires exactly one device per policy process")
            if devices == ("cpu",):
                continue
            if any(not isinstance(d, str) or not d.startswith("cuda:") or not d[5:].isdigit()
                   for d in devices):
                raise ValueError("use ['cpu'] or one explicit CUDA device such as ['cuda:0']")
