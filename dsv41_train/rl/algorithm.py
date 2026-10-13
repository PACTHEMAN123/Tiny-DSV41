"""Response-only GRPO with the actual asynchronous sampling policy as behavior."""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import torch
from torch import Tensor, nn

from .data import validate_token_ids


def _finite(tensor: Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} contains non-finite values")


def validate_temperature(temperature: float) -> None:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")


def input_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def extract_logits(output: Any) -> Tensor:
    logits = output if isinstance(output, Tensor) else output.logits
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] == 0 or logits.shape[2] == 0:
        raise ValueError("model logits must have shape [1, sequence_length, vocabulary_size]")
    return logits


def response_log_probs(
    model: nn.Module,
    prompt_ids: Sequence[int],
    response_ids: Sequence[int],
    temperature: float = 1.0,
) -> Tensor:
    """Teacher-forced probabilities, with the same temperature as generation.

    There is no padding: token ID zero is a real token and the attention mask
    therefore contains only ones. The returned tensor includes terminal EOS.
    """
    validate_temperature(temperature)
    prompt = validate_token_ids(prompt_ids, "prompt_ids")
    response = validate_token_ids(response_ids, "response_ids")
    tokens = torch.tensor([prompt + response[:-1]], dtype=torch.long, device=input_device(model))
    if getattr(model, "supports_token_log_probs", False):
        targets = torch.full_like(tokens, -100)
        targets[0, len(prompt) - 1:] = torch.tensor(response, device=tokens.device)
        scores = model(tokens, attention_mask=torch.ones_like(tokens), target_ids=targets,
                       target_mask=targets != -100, temperature=temperature)
        if scores.shape != tokens.shape:
            raise ValueError("model must return one log probability per input token")
        log_probs = scores[0, len(prompt) - 1:]
        _finite(log_probs, "response log probabilities")
        return log_probs
    logits = extract_logits(model(tokens, attention_mask=torch.ones_like(tokens)))
    if logits.shape[1] != tokens.shape[1]:
        raise ValueError("model must return one logits vector per input token")
    logits = logits[0, len(prompt) - 1 :].float() / temperature
    targets = torch.tensor(response, dtype=torch.long, device=logits.device)
    if targets.max().item() >= logits.shape[-1]:
        raise ValueError("response token ID is outside the model vocabulary")
    log_probs = torch.log_softmax(logits, dim=-1).gather(-1, targets[:, None]).squeeze(-1)
    _finite(log_probs, "response log probabilities")
    return log_probs


def group_advantages(rewards: Sequence[float] | Tensor, eps: float = 1e-6) -> Tensor:
    """Normalize within one complete prompt group using population std.

    Constant-reward groups, including singletons, have zero advantages.
    """
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("advantage epsilon must be finite and positive")
    values = torch.as_tensor(rewards, dtype=torch.float32).detach()
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("rewards must be a nonempty vector")
    _finite(values, "rewards")
    return (values - values.mean()) / (values.std(unbiased=False) + eps)


def grpo_loss(
    policy_log_probs: Sequence[Tensor],
    behavior_log_probs: Sequence[Sequence[float] | Tensor],
    advantages: Sequence[float] | Tensor,
    clip_eps: float = 0.2,
    ref_log_probs: Optional[Sequence[Tensor]] = None,
    kl_beta: float = 0.0,
) -> tuple[Tensor, dict[str, float]]:
    """Mean of per-response means of the clipped GRPO objective.

    The behavior probabilities must come from generation, even after several
    learner updates. There is no top-k/top-p truncation in either policy.

    Optional KL uses (pi / behavior) * k3(pi, reference), with gradients through
    the importance weight. At each observed prefix its expectation and gradient
    equal KL(pi || reference). Prefixes remain distributed under behavior;
    this is not a full-trajectory occupancy correction. Reference and behavior
    probabilities and advantages are treated as constants.
    """
    if not math.isfinite(clip_eps) or not 0 <= clip_eps < 1:
        raise ValueError("clip_eps must be finite and in [0, 1)")
    if not math.isfinite(kl_beta) or kl_beta < 0:
        raise ValueError("kl_beta must be finite and nonnegative")
    size = len(policy_log_probs)
    if size == 0 or len(behavior_log_probs) != size:
        raise ValueError("policy and behavior must contain the same nonzero number of responses")
    advantage_values = torch.as_tensor(advantages, dtype=torch.float32).detach()
    if advantage_values.ndim != 1 or len(advantage_values) != size:
        raise ValueError("one scalar advantage is required per response")
    _finite(advantage_values, "advantages")
    if ref_log_probs is not None and len(ref_log_probs) != size:
        raise ValueError("one reference log-probability vector is required per response")
    if kl_beta and ref_log_probs is None:
        raise ValueError("positive kl_beta requires reference log probabilities")

    policy_terms, kl_terms, clip_fractions, ratios, approx_kls = [], [], [], [], []
    for index, current in enumerate(policy_log_probs):
        if current.ndim != 1 or current.numel() == 0:
            raise ValueError("each policy log-probability tensor must be a nonempty vector")
        current = current.float()
        _finite(current, "policy log probabilities")
        behavior = torch.as_tensor(behavior_log_probs[index], dtype=torch.float32, device=current.device).detach()
        if behavior.shape != current.shape:
            raise ValueError("behavior and policy response lengths must match")
        _finite(behavior, "behavior log probabilities")
        log_ratio = current - behavior
        ratio = torch.exp(log_ratio)
        _finite(ratio, "importance ratios (reduce policy lag or learning rate)")
        advantage = advantage_values[index].to(current.device)
        unclipped = ratio * advantage
        clipped = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage
        policy_terms.append(-torch.minimum(unclipped, clipped).mean())
        if kl_beta:
            reference = ref_log_probs[index].to(device=current.device, dtype=torch.float32).detach()
            if reference.shape != current.shape:
                raise ValueError("reference and policy response lengths must match")
            _finite(reference, "reference log probabilities")
            delta = reference - current
            weighted_k3 = ratio * (torch.expm1(delta) - delta)
            _finite(weighted_k3, "importance-weighted reference KL")
            kl_terms.append(weighted_k3.mean())
        else:
            kl_terms.append(current.new_zeros(()))
        clip_fractions.append(((ratio - 1).abs() > clip_eps).float().mean().detach())
        ratios.append(ratio.mean().detach())
        approx_kls.append((-log_ratio).mean().detach())
    policy_loss = torch.stack(policy_terms).mean()
    kl = torch.stack(kl_terms).mean()
    loss = policy_loss + kl_beta * kl
    _finite(loss, "GRPO loss")
    return loss, {
        "loss": loss.detach().item(),
        "policy_loss": policy_loss.detach().item(),
        "kl": kl.detach().item(),
        "clip_fraction": torch.stack(clip_fractions).mean().item(),
        "importance_ratio": torch.stack(ratios).mean().item(),
        "approx_kl": torch.stack(approx_kls).mean().item(),
    }
