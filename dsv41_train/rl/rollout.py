"""Serial autoregressive rollout; each complete GRPO group has one policy version."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

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
    finish_reason: str = "length"
    stop_token_count: int = 0
    raw_reward: Optional[float] = None

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
        self.raw_reward = self.reward if self.raw_reward is None else float(self.raw_reward)
        if not math.isfinite(self.raw_reward):
            raise ValueError("raw reward must be finite")
        if self.finish_reason not in {"eos", "stop", "length"}:
            raise ValueError("finish_reason must be eos, stop or length")
        if type(self.stop_token_count) is not int or not 0 <= self.stop_token_count <= len(self.response_ids):
            raise ValueError("invalid stop_token_count")
        if isinstance(self.policy_version, bool) or not isinstance(self.policy_version, int) or self.policy_version < 0:
            raise ValueError("policy_version must be a nonnegative integer")

    @property
    def content_ids(self):
        return self.response_ids[:-self.stop_token_count] if self.stop_token_count else self.response_ids


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
    eos_token_id: Optional[int | Sequence[int]],
    policy_version: int,
    group_id: int,
    generator: Optional[torch.Generator] = None,
    reward_fn: Optional[RewardFn] = None,
    stop_token_sequences: Sequence[Sequence[int]] = (),
    truncated_reward: Optional[float] = None,
) -> RolloutGroup:
    """Generate untruncated categorical samples and their actual log probabilities.

    Sampling always uses a CPU generator, including when logits are on a GPU.
    Stop tokens remain in response_ids and behavior_log_probs for training.
    Text rewards can request content only; legacy rewards still see every token.
    The caller must keep model weights fixed until this complete group returns.
    """
    validate_temperature(temperature)
    for name, value in (("group_size", group_size), ("max_new_tokens", max_new_tokens)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name, value in (("policy_version", policy_version), ("group_id", group_id)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    eos_ids = [] if eos_token_id is None else validate_token_ids(
        [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id,
        "eos_token_id", allow_empty=True)
    stops = sorted((validate_token_ids(ids, "stop_token_sequences") for ids in stop_token_sequences),
                   key=len, reverse=True)
    if truncated_reward is not None and not math.isfinite(truncated_reward):
        raise ValueError("truncated_reward must be finite or None")
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
                finish_reason, stop_count = "length", 0
                for _ in range(max_new_tokens):
                    tokens = torch.tensor([record.prompt_ids + response], dtype=torch.long, device=input_device(model))
                    logits = extract_logits(model(tokens, attention_mask=torch.ones_like(tokens)))
                    if logits.shape[1] != tokens.shape[1]:
                        raise ValueError("model must return one logits vector per input token")
                    if any(token >= logits.shape[-1] for token in eos_ids + [t for ids in stops for t in ids]):
                        raise ValueError("stop token ID is outside the model vocabulary")
                    log_probs = torch.log_softmax(logits[0, -1].float() / temperature, dim=-1)
                    if not torch.isfinite(log_probs).all().item():
                        raise ValueError("generation log probabilities contain non-finite values")
                    probabilities = log_probs.exp().cpu()
                    token = int(torch.multinomial(probabilities, 1, generator=generator).item())
                    response.append(token)
                    behavior.append(float(log_probs[token].item()))
                    matched = next((ids for ids in stops if response[-len(ids):] == ids), None)
                    if matched is not None:
                        finish_reason, stop_count = "stop", len(matched)
                        break
                    if token in eos_ids:
                        finish_reason, stop_count = "eos", 1
                        break
                scoring_ids = response[:-stop_count] if stop_count and getattr(
                    reward_fn, "strip_stop_tokens", False) else response
                raw_reward = float(reward_fn(record, scoring_ids))
                reward = truncated_reward if finish_reason == "length" and truncated_reward is not None else raw_reward
                samples.append(Sample(record.prompt_ids, response, behavior, reward, policy_version,
                                      finish_reason, stop_count, raw_reward))
    finally:
        model.train(was_training)
    return RolloutGroup(samples, group_id, policy_version)


def generation_options(model_dir, model_config, config):
    from .tokenizer import read_eos_token_ids

    eos_ids = (tuple(config.eos_token_ids) if config.eos_token_ids is not None
               else read_eos_token_ids(model_dir, model_config.eos_token_id))
    if any(token >= model_config.vocab_size for token in eos_ids + tuple(
        token for ids in config.stop_token_sequences for token in ids
    )):
        raise ValueError("stop token ID is outside the model vocabulary")
    return dict(eos_token_id=eos_ids, stop_token_sequences=config.stop_token_sequences,
                truncated_reward=config.truncated_reward)
