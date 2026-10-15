"""Request-local native Qwen decode and matching frozen-backbone replay.

Cached BF16 decode is a distinct numerical sampling profile from repeatedly
forwarding the full prefix. Both sampling and replay use this implementation;
the recorded probabilities always come from the actual sampled distribution.
"""
from __future__ import annotations

from contextlib import nullcontext
import time

import torch
import torch.nn.functional as F
from torch import nn

from dsv41_train.models.qwen.model import apply_rotary_pos_emb, repeat_kv


QWEN_KV_SAMPLING_PATH = "request_kv_cache_v1"


class QwenRequestKVCache:
    """One unpadded conversation turn; never reused across requests or weights."""

    def __init__(self, policy):
        if (policy.config.model_type != "qwen3_moe" or policy.trainable_scope != "lm_head"
                or not policy.qwen_last_position_only or policy.backend.context_parallel.enabled):
            raise ValueError("cached Qwen decode requires local frozen-backbone last-position head training")
        policy.assert_engram_invariants()
        self.policy = policy
        self._head_version = policy.backend.lm_head.weight._version
        self._tokens = None
        self._keys_values = [None] * len(policy.backend.model.layers)
        self.timings = {"decoder_calls": 0, "prefill_tokens": 0, "decode_tokens": 0,
                        "prefill_host_submission_seconds": 0., "decode_host_submission_seconds": 0.}

    @torch.no_grad()
    def next_hidden(self, input_ids, attention_mask=None):
        """Prefill once, then decode exactly one appended token at its position."""
        started = time.monotonic()
        if (input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.dtype != torch.long
                or not 0 < input_ids.shape[1] <= self.policy.config.max_position_embeddings):
            raise ValueError("cached Qwen input must be one unpadded token sequence within its context")
        if attention_mask is not None and (attention_mask.shape != input_ids.shape
                or not bool(attention_mask.bool().all())):
            raise ValueError("cached Qwen decode does not accept padding")
        if self.policy.backend.lm_head.weight._version != self._head_version:
            raise RuntimeError("Qwen request cache cannot cross a policy weight update")
        previous = 0 if self._tokens is None else self._tokens.shape[1]
        if previous and (input_ids.shape[1] != previous + 1
                or not torch.equal(input_ids[:, :previous], self._tokens)):
            raise ValueError("Qwen request cache requires exactly one unchanged-prefix append")
        model = self.policy.backend.model
        hidden = model.embed_tokens(input_ids[:, previous:].to(model.embed_tokens.weight.device))
        length = hidden.shape[1]
        positions = torch.arange(previous, previous + length, device=hidden.device).unsqueeze(0)
        rotary = {}
        for index, layer in enumerate(model.layers):
            device = next(layer.parameters()).device
            with torch.cuda.device(device) if device.type == "cuda" else nullcontext():
                hidden = hidden.to(device)
                if device not in rotary:
                    rotary[device] = model.rotary_emb(hidden, positions.to(device))
                cos, sin = rotary[device]
                residual = hidden
                normalized = layer.input_layernorm(hidden)
                attention = layer.self_attn
                query = attention.q_proj(normalized).view(1, length, attention.num_heads, attention.head_dim)
                key = attention.k_proj(normalized).view(1, length, attention.num_key_value_heads, attention.head_dim)
                value = attention.v_proj(normalized).view(1, length, attention.num_key_value_heads, attention.head_dim)
                query = attention.q_norm(query).transpose(1, 2)
                key = attention.k_norm(key).transpose(1, 2)
                value = value.transpose(1, 2)
                query, key = apply_rotary_pos_emb(query, key, cos, sin)
                cached = self._keys_values[index]
                if cached is not None:
                    key = torch.cat((cached[0], key), dim=2)
                    value = torch.cat((cached[1], value), dim=2)
                self._keys_values[index] = (key, value)
                key = repeat_kv(key, attention.num_key_value_groups)
                value = repeat_kv(value, attention.num_key_value_groups)
                # Native prefill uses an explicit causal mask even without
                # padding. A single decode query must attend EVERY cached key;
                # SDPA is_causal=True would align it to the first key instead.
                mask = (torch.ones(length, length, dtype=torch.bool, device=device).tril()
                        .view(1, 1, length, length)) if not previous else None
                attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask,
                                                          dropout_p=0., is_causal=False)
                attended = attended.transpose(1, 2).reshape(1, length, -1)
                hidden = residual + attention.o_proj(attended)
                residual = hidden
                output = layer.mlp(layer.post_attention_layernorm(hidden))
                hidden = residual + (output[0] if isinstance(output, tuple) else output)
        self._tokens = input_ids.detach().clone()
        result = model.norm(hidden)[:, -1:].clone()
        self.timings["decoder_calls"] += 1
        self.timings["decode_tokens" if previous else "prefill_tokens"] += length
        self.timings["decode_host_submission_seconds" if previous else "prefill_host_submission_seconds"] += (
            time.monotonic() - started)
        return result


class QwenCachedSampler(nn.Module):
    """Adapt a fresh cache to the unchanged categorical sampler and CPU RNG."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.cache = QwenRequestKVCache(policy)
        self.train(policy.training)

    def next_token_logits(self, input_ids, attention_mask=None):
        hidden = self.cache.next_hidden(input_ids, attention_mask)
        return self.policy.backend.lm_head(hidden)[:, -1]


def cached_response_log_probs(policy, prompt_ids, response_ids, temperature):
    """Replay actual decode shapes, retaining gradients only through the head."""
    from dsv41_train.rl.algorithm import input_device

    cache = QwenRequestKVCache(policy)
    probabilities = []
    for index, token in enumerate(response_ids):
        tokens = torch.tensor([prompt_ids + response_ids[:index]], dtype=torch.long,
                              device=input_device(policy))
        with torch.no_grad():
            hidden = cache.next_hidden(tokens)
        logits = policy.backend.lm_head(hidden)[0, -1]
        if token >= logits.shape[-1]:
            raise ValueError("response token ID is outside the model vocabulary")
        probabilities.append((logits.float() / temperature).log_softmax(-1)[token])
    return torch.stack(probabilities)


def generate_cached_qwen_group(policy, record, group_size, **kwargs):
    """Keep independent sample caches while preserving the shared RNG stream."""
    from dsv41_train.rl.rollout import RolloutGroup, generate_group

    samples = []
    for _ in range(group_size):
        samples.extend(generate_group(QwenCachedSampler(policy), record, 1, **kwargs).samples)
    return RolloutGroup(samples, kwargs["group_id"], kwargs["policy_version"])


def qualify_qwen_cache(policy, prompt_ids, *, decode_tokens=128, temperature=.8, seed=731):
    """Bounded local diagnostic, separate from benchmark commands or rewards."""
    from dsv41_train.rl.algorithm import input_device, response_log_probs
    from dsv41_train.rl.data import PromptRecord
    from dsv41_train.rl.rollout import generate_group

    if not policy.qwen_kv_cache:
        raise ValueError("cache qualification requires the explicitly selected sampling profile")
    versions = {name: parameter._version for name, parameter in policy.named_parameters()}
    sampler = QwenCachedSampler(policy)
    started = time.monotonic()
    sample = generate_group(sampler, PromptRecord(prompt_ids), 1, decode_tokens, temperature,
                            None, 0, 0, torch.Generator().manual_seed(seed),
                            reward_fn=lambda record, ids: 0.).samples[0]
    generation_seconds = time.monotonic() - started
    with torch.no_grad():
        probabilities = response_log_probs(policy, prompt_ids, sample.response_ids, temperature)
        behavior = probabilities.new_tensor(sample.behavior_log_probs)
        replay_error = float((probabilities - behavior).abs().max())
        # Shape-rounding comparison is diagnostic, never substituted for the
        # actual cached behavior probabilities or its matching replay gate.
        cache = QwenRequestKVCache(policy)
        legacy_logit_error = legacy_action_logprob_error = 0.
        for index, token in enumerate(sample.response_ids[:8]):
            tokens = torch.tensor([prompt_ids + sample.response_ids[:index]], dtype=torch.long,
                                  device=input_device(policy))
            cached = policy.backend.lm_head(cache.next_hidden(tokens))[0, -1].float()
            legacy = policy.next_token_logits(tokens, torch.ones_like(tokens))[0].float()
            legacy_logit_error = max(legacy_logit_error, float((cached - legacy).abs().max()))
            legacy_action_logprob_error = max(legacy_action_logprob_error, float((
                (cached / temperature).log_softmax(-1)[token] -
                (legacy / temperature).log_softmax(-1)[token]).abs()))
    policy.assert_engram_invariants()
    if versions != {name: parameter._version for name, parameter in policy.named_parameters()}:
        raise RuntimeError("cache qualification changed model parameters")
    return {"kind": "native-qwen-cache-diagnostic", "benchmark_task": False,
            "benchmark_reward": None, "tool_executed": False,
            "sampling_path": QWEN_KV_SAMPLING_PATH, "sampling_seed": seed,
            "context_tokens": len(prompt_ids), "action_tokens": len(sample.response_ids),
            "generation_seconds": generation_seconds,
            "tokens_per_second": len(sample.response_ids) / generation_seconds,
            "completion_barrier": "sampled_token_probabilities_copied_to_cpu",
            "cached_sampling_vs_cached_replay_max_logprob_error": replay_error,
            "cached_replay_qualified": replay_error <= 1e-5,
            "cached_vs_full_prefix_compared_positions": min(8, len(sample.response_ids)),
            "cached_vs_full_prefix_max_logit_error": legacy_logit_error,
            "cached_vs_full_prefix_max_action_logprob_error": legacy_action_logprob_error,
            "full_action_budget_throughput_qualified": False,
            "parameters_unchanged": True, "decoder_timings": dict(sampler.cache.timings)}
