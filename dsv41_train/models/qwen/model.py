# Portions Copyright 2025 The Qwen team, Alibaba Group and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0.
"""Pure PyTorch Qwen3 MoE model used for native training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ...dispatch import TokenDispatcher
from ...parallel import ContextParallel
from .config import Qwen3MoeConfig


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_rotary_pos_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + rotate_half(query) * sin,
        key * cos + rotate_half(key) * sin,
    )


def repeat_kv(hidden: torch.Tensor, repetitions: int) -> torch.Tensor:
    if repetitions == 1:
        return hidden
    batch, heads, length, head_dim = hidden.shape
    hidden = hidden[:, :, None].expand(batch, heads, repetitions, length, head_dim)
    return hidden.reshape(batch, heads * repetitions, length, head_dim)


class Qwen3MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        dtype = hidden.dtype
        hidden = hidden.float()
        hidden = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * hidden.to(dtype)


class Qwen3MoeRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

    def forward(
        self,
        hidden: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.head_dim, 2, device=hidden.device, dtype=torch.float32)
                / self.head_dim
            )
        )
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            frequencies = position_ids.float().unsqueeze(-1) * inv_freq
            embedding = torch.cat((frequencies, frequencies), dim=-1)
            cos, sin = embedding.cos(), embedding.sin()
        return cos.to(hidden.dtype), sin.to(hidden.dtype)


class Qwen3MoeAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_id: int,
        context_parallel: ContextParallel,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.context_parallel = context_parallel
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = hidden.shape
        query = self.q_proj(hidden).view(batch, length, self.num_heads, self.head_dim)
        key = self.k_proj(hidden).view(batch, length, self.num_key_value_heads, self.head_dim)
        value = self.v_proj(hidden).view(
            batch,
            length,
            self.num_key_value_heads,
            self.head_dim,
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)
        key = self.context_parallel.gather(key, dim=2)
        value = self.context_parallel.gather(value, dim=2)
        key = repeat_kv(key, self.num_key_value_groups)
        value = repeat_kv(value, self.num_key_value_groups)

        sdpa_mask = None
        is_causal = attention_mask is None and not self.context_parallel.enabled
        if self.context_parallel.enabled:
            key_positions = self.context_parallel.gather(position_ids, dim=1)
            sdpa_mask = key_positions[:, None, None, :] <= position_ids[:, None, :, None]
            if attention_mask is not None:
                valid = self.context_parallel.gather(attention_mask, dim=1).bool()
                sdpa_mask = sdpa_mask & valid[:, None, None, :]
        elif attention_mask is not None:
            valid = attention_mask.bool()
            causal = torch.ones(length, length, dtype=torch.bool, device=hidden.device).tril()
            sdpa_mask = causal.view(1, 1, length, length)
            sdpa_mask = sdpa_mask & valid[:, None, None, :]

        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=sdpa_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        output = output.transpose(1, 2).reshape(batch, length, -1)
        return self.o_proj(output)


class Qwen3MoeMLP(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, intermediate_size: int | None = None) -> None:
        super().__init__()
        size = config.intermediate_size if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, size, bias=False)
        self.down_proj = nn.Linear(size, config.hidden_size, bias=False)
        if config.hidden_act != "silu":
            raise ValueError(f"unsupported hidden_act: {config.hidden_act}")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden)) * self.up_proj(hidden))


class Qwen3MoeExperts(nn.Module):
    """Stack the experts owned by this EP rank for deterministic FSDP collectives."""

    def __init__(self, config: Qwen3MoeConfig, dispatcher: TokenDispatcher) -> None:
        super().__init__()
        self.dispatcher = dispatcher
        self.global_num_experts = config.num_experts
        self.expert_start, expert_stop = dispatcher.expert_range(config.num_experts)
        self.num_experts = expert_stop - self.expert_start
        self.intermediate_size = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                2 * config.moe_intermediate_size,
                config.hidden_size,
            )
        )
        self.down_proj = nn.Parameter(
            torch.empty(
                self.num_experts,
                config.hidden_size,
                config.moe_intermediate_size,
            )
        )

    def forward(
        self,
        hidden: torch.Tensor,
        selected_experts: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        hidden, selected_experts, weights, metadata = self.dispatcher.dispatch(
            hidden,
            selected_experts,
            weights,
        )
        output = torch.zeros_like(hidden)
        for expert_id in selected_experts.unique():
            token_id = torch.where(selected_experts == expert_id)[0]
            current = hidden[token_id]
            gate, up = F.linear(current, self.gate_up_proj[expert_id]).chunk(2, dim=-1)
            current = F.silu(gate) * up
            current = F.linear(current, self.down_proj[expert_id])
            current = current * weights[token_id, None]
            output.index_add_(0, token_id, current.to(hidden.dtype))
        return self.dispatcher.combine(output, metadata)


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, dispatcher: TokenDispatcher) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = Qwen3MoeExperts(config, dispatcher)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, hidden_size = hidden.shape
        flat = hidden.reshape(-1, hidden_size)
        router_logits = self.gate(flat)
        weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        weights, selected_experts = torch.topk(weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(flat.dtype)

        output = self.experts(flat, selected_experts, weights)
        return output.reshape(batch, length, hidden_size), router_logits


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        layer_id: int,
        context_parallel: ContextParallel,
        dispatcher: TokenDispatcher,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3MoeAttention(config, layer_id, context_parallel)
        is_sparse = (
            layer_id not in config.mlp_only_layers
            and config.num_experts > 0
            and (layer_id + 1) % config.decoder_sparse_step == 0
        )
        self.mlp = (
            Qwen3MoeSparseMoeBlock(config, dispatcher)
            if is_sparse
            else Qwen3MoeMLP(config, config.intermediate_size)
        )
        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden
        hidden = self.self_attn(
            self.input_layernorm(hidden),
            cos,
            sin,
            attention_mask,
            position_ids,
        )
        hidden = residual + hidden

        residual = hidden
        mlp_output = self.mlp(self.post_attention_layernorm(hidden))
        router_logits = None
        if isinstance(mlp_output, tuple):
            mlp_output, router_logits = mlp_output
        return residual + mlp_output, router_logits


@dataclass
class Qwen3MoeModelOutput:
    last_hidden_state: torch.Tensor
    router_logits: tuple[torch.Tensor, ...] | None


class Qwen3MoeModel(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        context_parallel: ContextParallel | None = None,
        token_dispatcher: TokenDispatcher | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.context_parallel = context_parallel or ContextParallel()
        token_dispatcher = token_dispatcher or TokenDispatcher()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3MoeDecoderLayer(
                    config,
                    layer_id,
                    self.context_parallel,
                    token_dispatcher,
                )
                for layer_id in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config)
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        *,
        output_router_logits: bool = False,
    ) -> Qwen3MoeModelOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("sequence is longer than max_position_embeddings")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")

        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        input_ids = self.context_parallel.shard(input_ids)
        position_ids = self.context_parallel.shard(position_ids)
        if attention_mask is not None:
            attention_mask = self.context_parallel.shard(attention_mask)
        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(hidden, position_ids)

        collected_logits = []
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden, router_logits = checkpoint(
                    layer,
                    hidden,
                    cos,
                    sin,
                    attention_mask,
                    position_ids,
                    use_reentrant=False,
                )
            else:
                hidden, router_logits = layer(
                    hidden,
                    cos,
                    sin,
                    attention_mask,
                    position_ids,
                )
            if output_router_logits and router_logits is not None:
                collected_logits.append(router_logits)

        return Qwen3MoeModelOutput(
            last_hidden_state=self.norm(hidden),
            router_logits=tuple(collected_logits) if output_router_logits else None,
        )


def load_balancing_loss(
    router_logits: tuple[torch.Tensor, ...] | None,
    num_experts: int,
    top_k: int,
    attention_mask: torch.Tensor | None,
    context_parallel: ContextParallel | None = None,
) -> torch.Tensor | int:
    if not router_logits:
        return 0
    logits = torch.cat(router_logits, dim=0)
    probabilities = F.softmax(logits, dim=-1)
    selected = probabilities.topk(top_k, dim=-1).indices
    expert_mask = F.one_hot(selected, num_classes=num_experts)

    cp = context_parallel or ContextParallel()
    if attention_mask is None and not cp.enabled:
        tokens_per_expert = expert_mask.float().mean(dim=0)
        router_prob_per_expert = probabilities.mean(dim=0)
    else:
        if attention_mask is None:
            assignments = expert_mask.float().sum(dim=0)
            probability_sum = probabilities.sum(dim=0)
            total = logits.new_tensor(logits.shape[0], dtype=torch.float32)
        else:
            # Router logits are concatenated layer-major, then batch/sequence.
            layers = logits.shape[0] // attention_mask.numel()
            mask = attention_mask.to(logits.device).reshape(-1).repeat(layers)
            if cp.enabled:
                mask = mask.float()
            assignments = (expert_mask.float() * mask[:, None, None]).sum(dim=0)
            probability_sum = (probabilities * mask[:, None]).sum(dim=0)
            total = mask.sum()
        total = cp.sum(total).clamp_min(1)
        tokens_per_expert = cp.sum(assignments) / total
        router_prob_per_expert = cp.sum(probability_sum, autograd=True) / total
    return num_experts * (tokens_per_expert * router_prob_per_expert.unsqueeze(0)).sum()


@dataclass
class Qwen3MoeCausalLMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    aux_loss: torch.Tensor | int | None
    router_logits: tuple[torch.Tensor, ...] | None


class Qwen3MoeForCausalLM(nn.Module):
    def __init__(
        self,
        config: Qwen3MoeConfig,
        context_parallel: ContextParallel | None = None,
        token_dispatcher: TokenDispatcher | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.context_parallel = context_parallel or ContextParallel()
        self.model = Qwen3MoeModel(config, self.context_parallel, token_dispatcher)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Qwen3MoeExperts):
            nn.init.normal_(module.gate_up_proj, mean=0.0, std=self.config.initializer_range)
            nn.init.normal_(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, Qwen3MoeRMSNorm):
            nn.init.ones_(module.weight)

    def gradient_checkpointing_enable(self) -> None:
        self.model.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        output_router_logits: bool | None = None,
    ) -> Qwen3MoeCausalLMOutput:
        output_router_logits = (
            self.config.output_router_logits
            if output_router_logits is None
            else output_router_logits
        )
        output = self.model(
            input_ids,
            attention_mask,
            position_ids,
            output_router_logits=output_router_logits,
        )
        logits = self.lm_head(output.last_hidden_state)
        loss = None
        if labels is not None:
            if self.context_parallel.enabled:
                targets = torch.full_like(labels, -100)
                targets[:, :-1] = labels[:, 1:]
                if attention_mask is not None:
                    targets[:, :-1].masked_fill_(~attention_mask[:, 1:].bool(), -100)
                targets = self.context_parallel.shard(targets)
                loss_sum = F.cross_entropy(
                    logits.float().reshape(-1, self.config.vocab_size),
                    targets.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                loss = self.context_parallel.mean_loss(
                    loss_sum,
                    (targets != -100).sum(),
                )
            else:
                targets = labels[:, 1:].clone()
                loss = F.cross_entropy(
                    logits[:, :-1].float().reshape(-1, self.config.vocab_size),
                    targets.reshape(-1),
                    ignore_index=-100,
                )

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss(
                output.router_logits,
                self.config.num_experts,
                self.config.num_experts_per_tok,
                self.context_parallel.shard(attention_mask)
                if attention_mask is not None
                else None,
                self.context_parallel,
            )
            if loss is not None and isinstance(aux_loss, torch.Tensor):
                loss = loss + self.config.router_aux_loss_coef * aux_loss
        return Qwen3MoeCausalLMOutput(loss, logits, aux_loss, output.router_logits)


__all__ = [
    "Qwen3MoeAttention",
    "Qwen3MoeCausalLMOutput",
    "Qwen3MoeDecoderLayer",
    "Qwen3MoeExperts",
    "Qwen3MoeForCausalLM",
    "Qwen3MoeMLP",
    "Qwen3MoeModel",
    "Qwen3MoeModelOutput",
    "Qwen3MoeRMSNorm",
    "Qwen3MoeSparseMoeBlock",
    "load_balancing_loss",
]
