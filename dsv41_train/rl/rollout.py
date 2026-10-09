"""Serial autoregressive rollout; each complete GRPO group has one policy version."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from torch import nn

from .algorithm import extract_logits, input_device, validate_temperature
from .data import PromptRecord, RewardFn, build_reward, validate_token_ids


@dataclass
class Sample:
    prompt_ids: list[int]
    response_ids: list[int]
    behavior_log_probs: list[float]
    reward: float
    policy_version: int

    def __post_init__(self) -> None:
        self.prompt_ids = validate_token_ids(self.prompt_ids, "prompt_ids")
        self.response_ids = validate_token_ids(self.response_ids, "response_ids")
        self.behavior_log_probs = [float(value) for value in self.behavior_log_probs]
        if len(self.response_ids) != len(self.behavior_log_probs):
            raise ValueError("each response token requires one behavior log probability")
        if any(not math.isfinite(value) or value > 1e-6 for value in self.behavior_log_probs):
            raise ValueError("behavior log probabilities must be finite and nonpositive")
        self.reward = float(self.reward)
        if not math.isfinite(self.reward):
            raise ValueError("reward must be finite")
        if isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int) or self.policy_version < 0:
            raise ValueError("policy_version must be a nonnegative integer")


@dataclass
class RolloutGroup:
    samples: list[Sample]
    group_id: int
    policy_version: int

    def __post_init__(self) -> None:
        if not self.samples:
            raise ValueError("a rollout group must not be empty")
        if isinstance(self.group_id, bool) or not isinstance(self.group_id, int) or self.group_id < 0:
            raise ValueError("group_id must be a nonnegative integer")
        if isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int) or self.policy_version < 0:
            raise ValueError("policy_version must be a nonnegative integer")
        if any(sample.policy_version != self.policy_version for sample in self.samples):
            raise ValueError("all group samples must have the group policy_version")
        if any(sample.prompt_ids != self.samples[0].prompt_ids for sample in self.samples):
            raise ValueError("all group samples must have the same prompt")


def generate_group(
    model: nn.Module,
    record: PromptRecord | Mapping[str, Any],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    eos_token_id: Optional[int],
    policy_version: int,
    group_id: int,
    generator: Optional[torch.Generator] = None,
    reward_fn: Optional[RewardFn] = None,
) -> RolloutGroup:
    """Generate untruncated categorical samples and their actual log probabilities.

    Sampling always uses a CPU generator, including when logits are on a GPU.
    EOS is included in response_ids, behavior_log_probs, and reward evaluation.
    The caller must keep model weights fixed until this complete group returns.
    """
    validate_temperature(temperature)
    for name, value in (("group_size", group_size), ("max_new_tokens", max_new_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("policy_version", policy_version), ("group_id", group_id)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if eos_token_id is not None and (isinstance(eos_token_id, bool) or not isinstance(eos_token_id, int) or eos_token_id < 0):
        raise ValueError("eos_token_id must be a nonnegative integer or None")
    if generator is not None and generator.device.type != "cpu":
        raise ValueError("generation requires a CPU torch.Generator")
    if not isinstance(record, PromptRecord):
        record = PromptRecord.from_mapping(record)
    if reward_fn is None:
        reward_fn = build_reward("token_accuracy")
    samples = []
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(group_size):
                response, behavior = [], []
                for _ in range(max_new_tokens):
                    tokens = torch.tensor([record.prompt_ids + response], dtype=torch.long, device=input_device(model))
                    logits = extract_logits(model(tokens, attention_mask=torch.ones_like(tokens)))
                    if logits.shape[1] != tokens.shape[1]:
                        raise ValueError("model must return one logits vector per input token")
                    if eos_token_id is not None and eos_token_id >= logits.shape[-1]:
                        raise ValueError("eos_token_id is outside the model vocabulary")
                    log_probs = torch.log_softmax(logits[0, -1].float() / temperature, dim=-1)
                    if not torch.isfinite(log_probs).all().item():
                        raise ValueError("generation log probabilities contain non-finite values")
                    probabilities = log_probs.exp().cpu()
                    token = int(torch.multinomial(probabilities, 1, generator=generator).item())
                    response.append(token)
                    behavior.append(float(log_probs[token].item()))
                    if token == eos_token_id:
                        break
                reward = float(reward_fn(record, response))
                samples.append(Sample(record.prompt_ids, response, behavior, reward, policy_version))
    finally:
        model.train(was_training)
    return RolloutGroup(samples, group_id, policy_version)
