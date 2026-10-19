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
    """Action probabilities, with the same temperature as generation.

    There is no padding: token ID zero is a real token and the attention mask
    therefore contains only ones. Models may replay sampling prefixes through
    a differentiable ``response_log_probs`` hook to avoid shape-dependent
    rounding. Other models use one teacher-forced pass. Terminal EOS is included.
    """
    validate_temperature(temperature)
    prompt = validate_token_ids(prompt_ids, "prompt_ids")
    response = validate_token_ids(response_ids, "response_ids")
    hook = getattr(model, "response_log_probs", None)
    if hook is not None:
        if not callable(hook):
            raise ValueError("model response_log_probs hook must be callable")
        log_probs = hook(prompt, response, temperature)
        if not isinstance(log_probs, Tensor) or log_probs.shape != (len(response),):
            raise ValueError("model response_log_probs hook must return one tensor value per action")
        log_probs = log_probs.float()
        _finite(log_probs, "response log probabilities")
        if bool((log_probs > 1e-6).any()):
            raise ValueError("response log probabilities must be nonpositive")
        return log_probs
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


def response_log_probs_batch(
    model: nn.Module,
    prompt_ids_batch: Sequence[Sequence[int]],
    response_ids_batch: Sequence[Sequence[int]],
    temperature: float = 1.0,
    *,
    return_metadata: bool = False,
) -> list[Tensor] | tuple[list[Tensor], dict[str, Any]]:
    """Score ordered responses in one physical, right-padded model forward.

    Each row contains prompt + response[:-1]. Explicit masks preserve token zero
    and select only the aligned response targets, including terminal EOS. Rows
    remain separate examples; there is no sequence packing or serial fallback.
    Models with a sampling-prefix replay hook require their own explicit batch
    implementation rather than silently changing that hook's scoring semantics.
    """
    validate_temperature(temperature)
    size = len(prompt_ids_batch)
    if not size or len(response_ids_batch) != size:
        raise ValueError("prompt and response batches require the same nonzero size")
    prompts = [validate_token_ids(ids, "prompt_ids") for ids in prompt_ids_batch]
    responses = [validate_token_ids(ids, "response_ids") for ids in response_ids_batch]
    if getattr(model, "response_log_probs", None) is not None:
        raise ValueError("a sampling-prefix response hook cannot use the teacher-forced batch path")
    lengths = [len(prompt) + len(response) - 1 for prompt, response in zip(prompts, responses)]
    tokens = torch.zeros((size, max(lengths)), dtype=torch.long, device=input_device(model))
    token_mask = torch.zeros_like(tokens, dtype=torch.bool)
    targets = torch.full_like(tokens, -100)
    for row, (prompt, response, length) in enumerate(zip(prompts, responses, lengths)):
        tokens[row, :length] = torch.tensor(prompt + response[:-1], device=tokens.device)
        token_mask[row, :length] = True
        targets[row, len(prompt) - 1:length] = torch.tensor(response, device=tokens.device)
    target_mask = targets != -100
    if getattr(model, "supports_token_log_probs", False):
        scores = model(tokens, attention_mask=token_mask, target_ids=targets,
                       target_mask=target_mask, temperature=temperature)
        if not isinstance(scores, Tensor) or scores.shape != tokens.shape:
            raise ValueError("model must return one log probability per input token")
        selected = scores[target_mask].float()
    else:
        output = model(tokens, attention_mask=token_mask)
        logits = output if isinstance(output, Tensor) else output.logits
        if logits.ndim != 3 or logits.shape[:2] != tokens.shape or not logits.shape[-1]:
            raise ValueError("model logits must have shape [batch, sequence_length, vocabulary_size]")
        selected_targets = targets[target_mask]
        if selected_targets.max().item() >= logits.shape[-1]:
            raise ValueError("response token ID is outside the model vocabulary")
        selected = (logits[target_mask].float() / temperature).log_softmax(-1).gather(
            -1, selected_targets[:, None]).squeeze(-1)
    _finite(selected, "response log probabilities")
    if bool((selected > 1e-6).any()):
        raise ValueError("response log probabilities must be nonpositive")
    values = list(selected.split([len(response) for response in responses]))
    if not return_metadata:
        return values
    metadata = {"source": "one_padded_model_forward", "model_forward_calls": 1,
        "batch_size": int(tokens.shape[0]), "input_shape": list(tokens.shape),
        "input_lengths": lengths, "response_lengths": [len(response) for response in responses],
        "valid_input_tokens": sum(lengths),
        "padded_input_tokens": tokens.numel() - sum(lengths),
        "allocated_input_tokens": tokens.numel(), "response_tokens": sum(map(len, responses)),
        "padding_side": "right"}
    return values, metadata


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


def decoupled_token_tis_loss(
    policy_log_probs: Sequence[Tensor],
    native_old_log_probs: Sequence[Sequence[float] | Tensor],
    behavior_log_probs: Sequence[Sequence[float] | Tensor],
    advantages: Sequence[float] | Tensor,
    clip_eps: float = 0.2,
    is_cap: float = 2.0,
) -> tuple[Tensor, dict[str, float]]:
    """Optional three-policy Token-TIS surrogate, without a reference KL term.

    ``old`` must be frozen native scores at the exact generating adapter version;
    ``behavior`` remains the actual sampler distribution, at the same temperature.
    Detached weights min(cap, old/behavior) multiply PPO(current/old), so numerical
    backend mismatch does not itself consume PPO's update clipping interval.
    Average tokens within each response, then average responses. There is no lower
    clipping, weight normalization or rejection. Upper truncation introduces bias;
    token weighting does not correct the distribution of preceding prefixes.
    Weight ESS is an algebraic token-weight diagnostic, not independent sample size.
    The existing ``grpo_loss`` and its strict admission route remain unchanged.
    """
    if not math.isfinite(clip_eps) or not 0 <= clip_eps < 1:
        raise ValueError("clip_eps must be finite and in [0, 1)")
    if not math.isfinite(is_cap) or is_cap <= 0:
        raise ValueError("is_cap must be finite and positive")
    size = len(policy_log_probs)
    if size == 0 or len(native_old_log_probs) != size or len(behavior_log_probs) != size:
        raise ValueError("current, native-old and behavior require the same nonzero response count")
    advantage_values = torch.as_tensor(advantages, dtype=torch.float32).detach()
    if advantage_values.ndim != 1 or len(advantage_values) != size:
        raise ValueError("one scalar advantage is required per response")
    _finite(advantage_values, "advantages")
    terms, weights_all, raw_weights_all, ratios_all, deltas_all = [], [], [], [], []
    for index, current in enumerate(policy_log_probs):
        if not isinstance(current, Tensor) or current.ndim != 1 or current.numel() == 0:
            raise ValueError("each policy log-probability tensor must be a nonempty vector")
        current = current.float()
        old = torch.as_tensor(native_old_log_probs[index], dtype=torch.float32, device=current.device).detach()
        behavior = torch.as_tensor(behavior_log_probs[index], dtype=torch.float32, device=current.device).detach()
        if old.shape != current.shape or behavior.shape != current.shape:
            raise ValueError("native-old, behavior and policy response lengths must match")
        for name, values in (("policy", current), ("native-old", old), ("behavior", behavior)):
            _finite(values, name + " log probabilities")
            if bool((values > 1e-6).any()):
                raise ValueError(name + " log probabilities must be nonpositive")
        delta = old - behavior
        # Float64 diagnostics retain raw tail mass; training weights stay FP32.
        raw = delta.double().exp()
        _finite(raw, "raw rollout importance weights")
        if bool((raw == 0).any()):
            raise ValueError("raw rollout importance weights underflowed")
        weights = raw.clamp(max=is_cap).float().detach()
        _finite(weights, "training rollout importance weights")
        if bool((weights == 0).any()):
            raise ValueError("training rollout importance weights underflowed")
        ratio = (current - old).exp()
        _finite(ratio, "native PPO ratios")
        if bool((ratio == 0).any()):
            raise ValueError("native PPO ratios underflowed")
        advantage = advantage_values[index].to(current.device)
        surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage)
        terms.append(-(weights * surrogate).mean())
        weights_all.append(raw.clamp(max=is_cap))
        raw_weights_all.append(raw)
        ratios_all.append(ratio.detach().double())
        deltas_all.append(delta.double())
    loss = torch.stack(terms).mean()
    _finite(loss, "Token-TIS loss")
    weights, raw = torch.cat(weights_all), torch.cat(raw_weights_all)
    ratios, deltas = torch.cat(ratios_all), torch.cat(deltas_all)
    total, squared, raw_total = weights.sum(), weights.square().sum(), raw.sum()
    _finite(raw_total, "summed raw rollout importance weights")
    ess = total.square() / squared
    _finite(ess, "token weight ESS")
    return loss, {
        "loss": loss.detach().item(),
        "policy_loss": loss.detach().item(),
        "response_count": float(size),
        "token_count": float(weights.numel()),
        "is_cap": float(is_cap),
        "is_weight_sum": total.item(),
        "is_weight_square_sum": squared.item(),
        "is_weight_mean": weights.mean().item(),
        "is_weight_min": weights.min().item(),
        "is_weight_max": weights.max().item(),
        "is_raw_weight_sum": raw_total.item(),
        "is_raw_weight_min": raw.min().item(),
        "is_raw_weight_max": raw.max().item(),
        "is_clipped_tokens": float((raw > is_cap).sum().item()),
        "is_clipped_fraction": (raw > is_cap).double().mean().item(),
        "is_dropped_weight_mass": (raw_total - total).clamp(min=0).item(),
        "is_dropped_weight_mass_fraction": ((raw_total - total).clamp(min=0) / raw_total).item(),
        "is_token_weight_ess": ess.item(),
        "is_token_weight_ess_fraction": ess.item() / weights.numel(),
        "native_old_minus_behavior_mean": deltas.mean().item(),
        "native_old_minus_behavior_max_abs": deltas.abs().max().item(),
        "native_ppo_ratio_mean": ratios.mean().item(),
        "native_ppo_clip_fraction": ((ratios - 1).abs() > clip_eps).double().mean().item(),
    }
