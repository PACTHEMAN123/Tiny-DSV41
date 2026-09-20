# Portions Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0. See https://www.apache.org/licenses/LICENSE-2.0
"""A compact, training-only DeepSeek V4.1 model implemented with PyTorch alone.

This keeps the architecture used by the tiny trainer: mHC residual streams, CSA2
compressed attention, sparse MoE layers, and optional Engram memory. Generation
caches, multimodal modules, distributed kernels, and checkpoint conversion are
deliberately outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .config import DeepSeekV41Config
from .moe import RoutedExperts, SparseMoE, TokenDispatcher, TopKRouter
from .parallel import ContextParallel


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class HyperConnection(nn.Module):
    """Compute the input, output, and residual mixing weights for one mHC site."""

    def __init__(self, config: DeepSeekV41Config) -> None:
        super().__init__()
        self.hc_mult = config.hc_mult
        self.sinkhorn_iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        mix_size = (2 + config.hc_mult) * config.hc_mult
        self.fn = nn.Parameter(torch.empty(mix_size, config.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.zeros(mix_size))
        self.scale = nn.Parameter(torch.ones(3))
        self.norm_eps = config.rms_norm_eps

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flat = streams.flatten(2).float()
        flat = flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + self.norm_eps)
        hc = self.hc_mult
        pre, post, comb = F.linear(flat, self.fn.float()).split([hc, hc, hc * hc], dim=-1)
        pre_bias, post_bias, comb_bias = self.base.float().split([hc, hc, hc * hc])
        pre_scale, post_scale, comb_scale = self.scale.float()

        pre = torch.sigmoid(pre * pre_scale + pre_bias) + self.eps
        post = 2 * torch.sigmoid(post * post_scale + post_bias)
        comb = comb.view(*comb.shape[:-1], hc, hc) * comb_scale + comb_bias.view(hc, hc)
        comb = torch.softmax(comb, dim=-1) + self.eps
        for _ in range(self.sinkhorn_iters):
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
        return pre, post, comb


class RotaryEmbedding(nn.Module):
    def __init__(self, config: DeepSeekV41Config) -> None:
        super().__init__()
        dim = config.qk_rope_head_dim
        main = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        compressed = 1.0 / (config.compress_rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("main", main, persistent=False)
        self.register_buffer("compressed", compressed, persistent=False)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, *, compressed: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.compressed if compressed else self.main
        frequencies = positions.float().unsqueeze(-1) * inv_freq.float()
        return frequencies.cos().to(x.dtype), frequencies.sin().to(x.dtype)


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *, inverse: bool = False
) -> torch.Tensor:
    if inverse:
        sin = -sin
    rope_dim = cos.shape[-1] * 2
    plain, rotary = x[..., :-rope_dim], x[..., -rope_dim:]
    if x.ndim == 4:
        cos, sin = cos.unsqueeze(2), sin.unsqueeze(2)
    even, odd = rotary[..., 0::2].float(), rotary[..., 1::2].float()
    rotary = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
    return torch.cat((plain, rotary.to(x.dtype)), dim=-1)


class GroupedLinear(nn.Module):
    def __init__(self, input_per_group: int, output_per_group: int, groups: int) -> None:
        super().__init__()
        self.groups = groups
        self.weight = nn.Parameter(torch.empty(groups, output_per_group, input_per_group))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...gi,goi->...go", x, self.weight)


class Compressor(nn.Module):
    """Pool complete groups of live tokens into shared KV latents."""

    def __init__(self, config: DeepSeekV41Config, layer_id: int) -> None:
        super().__init__()
        self.ratio = config.compress_ratios[layer_id]
        self.kv_proj = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.gate_proj = (
            nn.Linear(config.hidden_size, config.head_dim, bias=False) if self.ratio > 1 else None
        )
        self.norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(
        self, x: torch.Tensor, positions: torch.Tensor, token_mask: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        live = token_mask.bool()
        counts = live.long().sum(-1)
        group_counts = counts // self.ratio
        query_lengths = live.long().cumsum(-1)

        kv = self.kv_proj(x)
        gate = self.gate_proj(x).float() if self.gate_proj is not None else None
        destinations = torch.where(live, live.long().cumsum(-1) - 1, length)
        order = positions.new_zeros(batch, length + 1)
        order.scatter_(1, destinations, torch.arange(length, device=x.device).expand(batch, -1))

        max_groups = length // self.ratio
        indices = order[:, : max_groups * self.ratio]
        group_positions = positions.gather(1, indices[:, :: self.ratio])
        if max_groups == 0:
            return None, group_positions, query_lengths

        gathered = kv.gather(1, indices.unsqueeze(-1).expand(-1, -1, kv.shape[-1]))
        if gate is None:
            latent = gathered
        else:
            gathered_gate = gate.gather(1, indices.unsqueeze(-1).expand(-1, -1, gate.shape[-1]))
            gathered = gathered.view(batch, max_groups, self.ratio, -1).float()
            gathered_gate = gathered_gate.view(batch, max_groups, self.ratio, -1)
            latent = (gathered * gathered_gate.softmax(dim=2)).sum(dim=2)
        valid = torch.arange(max_groups, device=x.device) < group_counts.unsqueeze(-1)
        latent = latent.masked_fill(~valid.unsqueeze(-1), 0)
        return self.norm(latent.to(x.dtype)), group_positions, query_lengths


def select_candidate_blocks(
    scores: torch.Tensor, visible: torch.Tensor, topk_blocks: int, block_size: int
) -> torch.Tensor:
    width = scores.shape[-1]
    block_scores = F.pad(scores, (0, -width % block_size), value=float("-inf"))
    block_scores = block_scores.unflatten(-1, (-1, block_size)).amax(-1)
    last = (visible - 1) // block_size
    block_ids = torch.arange(block_scores.shape[-1], device=scores.device)
    block_scores = block_scores.masked_fill(block_ids == last, torch.inf)
    picked = block_scores.topk(min(topk_blocks, block_scores.shape[-1]), dim=-1)
    keep = torch.zeros_like(block_scores, dtype=torch.bool)
    keep.scatter_(-1, picked.indices, picked.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class SparseIndexer(nn.Module):
    def __init__(self, config: DeepSeekV41Config, layer_id: int, rotary: RotaryEmbedding) -> None:
        super().__init__()
        self.owns_keys = layer_id in config.kv_source_layer_ids
        self.is_candidate_source = layer_id == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_id
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk
        self.topk_blocks = config.candidate_topk_blocks
        self.block_size = config.candidate_block_size
        self.rotary = rotary
        self.q_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.weight_proj = nn.Linear(config.hidden_size, self.num_heads, bias=False)
        if self.owns_keys:
            self.k_proj = nn.Linear(config.head_dim, self.head_dim, bias=False)
            self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        q_residual: torch.Tensor,
        latent: torch.Tensor | None,
        group_positions: torch.Tensor,
        positions: torch.Tensor,
        shared: dict[str, torch.Tensor | None],
    ) -> None:
        batch, length, _ = x.shape
        if self.owns_keys:
            shared["index_keys"] = None
            if latent is not None:
                keys = self.k_norm(self.k_proj(latent))
                cos, sin = self.rotary(keys, group_positions, compressed=True)
                shared["index_keys"] = apply_rope(keys, cos, sin).unsqueeze(1)

        index_keys = shared.get("index_keys")
        if index_keys is None:
            shared["topk_indices"] = None
            if self.is_candidate_source:
                shared["candidates"] = None
            return

        keys = index_keys[:, 0].float()
        query = self.q_proj(q_residual).view(batch, length, self.num_heads, self.head_dim)
        cos, sin = self.rotary(query, positions, compressed=True)
        query = apply_rope(query, cos, sin).float()
        head_weights = self.weight_proj(x).float() * self.num_heads**-0.5
        scores = torch.einsum("bshd,btd->bsht", query, keys).relu_() * self.head_dim**-0.5
        scores = (scores * head_weights.unsqueeze(-1)).sum(2)

        visible = shared["compress_lengths"].unsqueeze(-1)
        key_ids = torch.arange(keys.shape[1], device=x.device)
        scores = scores.masked_fill(key_ids >= visible, float("-inf"))
        previous_candidates = shared.get("candidates") if self.uses_candidates else None
        if self.is_candidate_source:
            shared["candidates"] = select_candidate_blocks(
                scores, visible, self.topk_blocks, self.block_size
            )
        elif previous_candidates is not None:
            scores = scores.masked_fill(~previous_candidates, float("-inf"))

        picked = scores.topk(min(self.topk, keys.shape[1]), dim=-1, sorted=False)
        shared["topk_indices"] = torch.where(
            picked.values > float("-inf"), picked.indices, torch.full_like(picked.indices, -1)
        )


class CompressedAttention(nn.Module):
    def __init__(
        self,
        config: DeepSeekV41Config,
        layer_id: int,
        rotary: RotaryEmbedding,
        context_parallel: ContextParallel,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.ratio = config.compress_ratios[layer_id]
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout
        self.rotary = rotary
        self.context_parallel = context_parallel
        self.q_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        group_input = self.num_heads * self.head_dim // config.o_groups
        self.o_a = GroupedLinear(group_input, config.o_lora_rank, config.o_groups)
        self.o_b = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))
        self.compressor = Compressor(config, layer_id) if layer_id in config.kv_source_layer_ids else None
        self.indexer = (
            SparseIndexer(config, layer_id, rotary) if layer_id in config.index_source_layer_ids else None
        )

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        token_mask: torch.Tensor,
        shared: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        compressed = self.ratio > 0
        cos, sin = self.rotary(x, positions, compressed=compressed)
        q_residual = self.q_norm(self.q_a(x))
        query = self.q_b(q_residual).view(batch, length, self.num_heads, self.head_dim)
        query = apply_rope(query, cos, sin).transpose(1, 2)
        kv = apply_rope(self.kv_norm(self.kv_proj(x)), cos, sin).unsqueeze(1)
        kv = self.context_parallel.gather(kv, dim=2)

        selected = selected_valid = None
        if compressed:
            latent = None
            group_positions = positions[:, :0]
            if self.compressor is not None:
                full_x = self.context_parallel.gather(x)
                full_positions = self.context_parallel.gather(positions)
                full_token_mask = self.context_parallel.gather(token_mask)
                latent, group_positions, query_lengths = self.compressor(
                    full_x, full_positions, full_token_mask
                )
                shared["compress_lengths"] = self.context_parallel.shard(query_lengths) // self.ratio
                shared["compressed_kv"] = None
                shared["index_keys"] = None
                shared["topk_indices"] = None
                shared["candidates"] = None

            if self.indexer is not None:
                self.indexer(x, q_residual, latent, group_positions, positions, shared)

            if latent is not None:
                latent_cos, latent_sin = self.rotary(latent, group_positions, compressed=True)
                shared["compressed_kv"] = apply_rope(latent, latent_cos, latent_sin).unsqueeze(1)

            compressed_kv = shared.get("compressed_kv")
            topk_indices = shared.get("topk_indices")
            if compressed_kv is not None and topk_indices is not None and topk_indices.shape[-1]:
                entries = compressed_kv[:, 0]
                selected_valid = topk_indices >= 0
                rows = torch.arange(batch, device=x.device).view(batch, 1, 1)
                selected = entries[rows, topk_indices.clamp_min(0)]

        output = self._attend(query, kv, attention_mask, selected, selected_valid)
        output = apply_rope(output, cos, sin, inverse=True)
        grouped = output.reshape(batch, length, self.config.o_groups, -1)
        return self.o_b(self.o_a(grouped).flatten(2))

    def _attend(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        mask: torch.Tensor,
        selected: torch.Tensor | None,
        selected_valid: torch.Tensor | None,
    ) -> torch.Tensor:
        logits = torch.matmul(query, kv.transpose(2, 3)) * self.head_dim**-0.5
        logits = logits + mask
        window = logits.shape[-1]
        if selected is not None:
            picked = torch.einsum("bhsd,bskd->bhsk", query, selected.to(query.dtype))
            picked = picked * self.head_dim**-0.5
            picked = picked.masked_fill(~selected_valid.unsqueeze(1), float("-inf"))
            logits = torch.cat((logits, picked), dim=-1)

        sinks = self.sinks.view(1, -1, 1, 1).expand(query.shape[0], -1, query.shape[-2], -1)
        probabilities = torch.softmax(torch.cat((logits.float(), sinks), dim=-1), dim=-1)[..., :-1]
        probabilities = F.dropout(probabilities, self.dropout, self.training).to(kv.dtype)
        output = torch.matmul(probabilities[..., :window], kv)
        if selected is not None:
            output = output + torch.einsum(
                "bhsk,bskd->bhsd", probabilities[..., window:], selected.to(probabilities.dtype)
            )
        return output.transpose(1, 2).contiguous()


def _is_prime(value: int) -> bool:
    if value < 2 or value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _next_prime(value: int, used: set[int]) -> int:
    value += 1
    while value in used or not _is_prime(value):
        value += 1
    return value


class NgramHash(nn.Module):
    def __init__(self, config: DeepSeekV41Config) -> None:
        super().__init__()
        self.max_ngram = config.engram_max_ngram_size
        self.pad_id = config.engram_pad_id % config.engram_compressed_vocab_size
        layer_primes, layer_offsets, used = [], [], set()
        for table_size in config.engram_num_embeddings:
            primes, offsets, offset = [], [], 0
            current = config.engram_vocab_size - 1
            for _ in range((self.max_ngram - 1) * config.engram_n_heads):
                current = _next_prime(current, used)
                used.add(current)
                primes.append(current)
                offsets.append(offset)
                offset += current
            if offset > table_size:
                raise ValueError(f"engram table needs {offset} rows, got {table_size}")
            layer_primes.append(primes)
            layer_offsets.append(offsets)

        primes = torch.tensor(layer_primes).view(
            len(config.engram_layer_ids), self.max_ngram - 1, config.engram_n_heads
        )
        offsets = torch.tensor(layer_offsets)
        bound = max(2, ((2**63 - 1) // config.engram_compressed_vocab_size) // 2)
        multipliers = []
        for layer_id in config.engram_layer_ids:
            values = [((layer_id + 1) * 1000003 + (i + 1) * 9176) % bound for i in range(self.max_ngram)]
            multipliers.append([2 * value + 1 for value in values])
        self.compressed_vocab_size = config.engram_compressed_vocab_size
        self.register_buffer("primes", primes, persistent=False)
        self.register_buffer("offsets", offsets, persistent=False)
        self.register_buffer("multipliers", torch.tensor(multipliers), persistent=False)

    def forward(self, input_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        tokens = input_ids.remainder(self.compressed_vocab_size)
        dead = -1
        tokens = tokens.masked_fill(~token_mask, dead)
        context = self.max_ngram - 1
        history = F.pad(tokens, (context, 0), value=dead)
        shifted, blocked = [], torch.zeros_like(tokens, dtype=torch.bool)
        for distance in range(self.max_ngram):
            source = history[:, context - distance : context - distance + tokens.shape[1]]
            blocked |= source == dead
            shifted.append(torch.where(blocked, self.pad_id, source))
        windows = torch.stack(shifted, dim=-1)

        products = windows.unsqueeze(2) * self.multipliers
        rolling, hashes = products[..., 0], []
        for index in range(1, self.max_ngram):
            rolling = torch.bitwise_xor(rolling, products[..., index])
            hashes.append(rolling.unsqueeze(-1) % self.primes[:, index - 1])
        return torch.cat(hashes, dim=-1) + self.offsets


class Engram(nn.Module):
    def __init__(self, config: DeepSeekV41Config) -> None:
        super().__init__()
        columns = (config.engram_max_ngram_size - 1) * config.engram_n_heads
        self.hidden_size = config.hidden_size
        self.hc_mult = config.hc_mult
        self.eps = config.rms_norm_eps
        self.proj = nn.Linear(
            columns * config.engram_head_dim,
            config.hidden_size * (config.hc_mult + 1),
            bias=False,
        )
        self.q_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))
        self.k_weight = nn.Parameter(torch.ones(config.hc_mult, config.hidden_size))

    def forward(self, streams: torch.Tensor, rows: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        key, value = self.proj(rows.flatten(-2).to(streams.dtype)).split(
            [self.hc_mult * self.hidden_size, self.hidden_size], dim=-1
        )
        key = key.float().unflatten(-1, (self.hc_mult, self.hidden_size))
        hidden = streams.float()
        scale = torch.rsqrt(hidden.square().mean(-1) + self.eps)
        scale = scale * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (hidden * self.q_weight.float() * self.k_weight.float() * key).sum(-1)
        dot = dot * scale * self.hidden_size**-0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        gate = gate.masked_fill(~mask.unsqueeze(-1), 0)
        return (hidden + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(streams.dtype)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        config: DeepSeekV41Config,
        layer_id: int,
        rotary: RotaryEmbedding,
        context_parallel: ContextParallel,
        token_dispatcher: TokenDispatcher,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attention = CompressedAttention(config, layer_id, rotary, context_parallel)
        self.moe = SparseMoE(config, token_dispatcher)
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.engram = Engram(config) if layer_id in config.engram_layer_ids else None
        self.attention_hc = HyperConnection(config)
        self.moe_hc = HyperConnection(config)

    @staticmethod
    def collapse(streams: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return (weights.unsqueeze(-1) * streams.float()).sum(2).to(streams.dtype)

    @staticmethod
    def expand(
        value: torch.Tensor,
        residual: torch.Tensor,
        output_weights: torch.Tensor,
        residual_weights: torch.Tensor,
    ) -> torch.Tensor:
        mixed = torch.einsum("bsjk,bsjd->bskd", residual_weights.float(), residual.float())
        return (output_weights.unsqueeze(-1) * value.float().unsqueeze(-2) + mixed).to(residual.dtype)

    def forward(
        self,
        streams: torch.Tensor,
        pre_mix: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor,
        token_mask: torch.Tensor,
        shared: dict[str, torch.Tensor | None],
        engram_rows: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.engram is not None:
            streams = self.engram(streams, engram_rows, token_mask)

        residual = streams
        next_pre, post, combine = self.attention_hc(streams)
        collapsed = self.collapse(streams, pre_mix)
        value = self.attention(
            self.input_norm(collapsed), positions, attention_mask, token_mask, shared
        )
        streams = self.expand(value, residual, post, combine)

        residual = streams
        final_pre, post, combine = self.moe_hc(streams)
        collapsed = self.collapse(streams, next_pre)
        value, router_logits = self.moe(self.post_attention_norm(collapsed))
        streams = self.expand(value, residual, post, combine)
        return streams, final_pre, router_logits


class DeepSeekV41Model(nn.Module):
    def __init__(
        self,
        config: DeepSeekV41Config,
        context_parallel: ContextParallel | None = None,
        token_dispatcher: TokenDispatcher | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.context_parallel = context_parallel or ContextParallel()
        token_dispatcher = token_dispatcher or TokenDispatcher()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary = RotaryEmbedding(config)
        self.layers = nn.ModuleList(
            DecoderLayer(
                config,
                layer_id,
                self.rotary,
                self.context_parallel,
                token_dispatcher,
            )
            for layer_id in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hash = NgramHash(config) if config.engram_layer_ids else None
        self.engram_tables = nn.ModuleDict(
            {
                str(layer_id): nn.Embedding(config.engram_num_embeddings[index], config.engram_head_dim)
                for index, layer_id in enumerate(config.engram_layer_ids)
            }
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, global_length = input_ids.shape
        if global_length > self.config.max_position_embeddings:
            raise ValueError("sequence is longer than max_position_embeddings")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        global_positions = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
        causal_mask = self._causal_mask(attention_mask, self.embedding.weight.dtype)
        hashes = self.hash(input_ids, attention_mask) if self.hash is not None else None

        input_ids = self.context_parallel.shard(input_ids)
        positions = self.context_parallel.shard(global_positions)
        attention_mask = self.context_parallel.shard(attention_mask)
        hidden = self.embedding(input_ids)
        length = hidden.shape[1]
        streams = hidden.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        causal_mask = causal_mask.to(hidden.dtype)

        engram_rows: dict[int, torch.Tensor] = {}
        if hashes is not None:
            hashes = self.context_parallel.shard(hashes)
            for index, layer_id in enumerate(self.config.engram_layer_ids):
                engram_rows[layer_id] = self.engram_tables[str(layer_id)](hashes[:, :, index])

        shared: dict[str, torch.Tensor | None] = {}
        pre_mix = hidden.new_zeros(batch, length, self.config.hc_mult, dtype=torch.float32)
        pre_mix[..., 0] = 1
        router_logits = []
        for layer in self.layers:
            streams, pre_mix, logits = layer(
                streams,
                pre_mix,
                positions,
                causal_mask,
                attention_mask,
                shared,
                engram_rows.get(layer.layer_id),
            )
            router_logits.append(logits)

        hidden = DecoderLayer.collapse(streams, pre_mix)
        return self.norm(hidden), tuple(router_logits)

    def _causal_mask(self, token_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        length = token_mask.shape[1]
        key_positions = torch.arange(length, device=token_mask.device)
        query_positions = self.context_parallel.shard(key_positions, dim=0)
        query_mask = self.context_parallel.shard(token_mask)
        distance = query_positions[:, None] - key_positions[None, :]
        allowed = (distance >= 0) & (distance < self.config.sliding_window)
        allowed = allowed.view(1, 1, query_positions.numel(), length)
        allowed = allowed & token_mask[:, None, None, :] & query_mask[:, None, :, None]
        return torch.zeros((), dtype=dtype, device=token_mask.device).expand_as(allowed).masked_fill(
            ~allowed, torch.finfo(dtype).min
        )


def load_balancing_loss(
    router_logits: tuple[torch.Tensor, ...],
    num_experts: int,
    topk: int,
    attention_mask: torch.Tensor | None,
    context_parallel: ContextParallel | None = None,
) -> torch.Tensor:
    device = router_logits[0].device
    assignments = torch.zeros(num_experts, device=device)
    probabilities = torch.zeros(num_experts, device=device)
    total = torch.zeros((), device=device)
    mask = None if attention_mask is None else attention_mask.flatten().float().to(device)
    for logits in router_logits:
        probs = torch.softmax(logits.float(), dim=-1)
        selected = probs.topk(topk, dim=-1).indices
        if mask is None:
            assignments += torch.bincount(selected.flatten(), minlength=num_experts)
            probabilities += probs.sum(0)
            total += probs.shape[0]
        else:
            assignments.scatter_add_(0, selected.flatten(), mask.repeat_interleave(topk))
            probabilities += (probs * mask.unsqueeze(-1)).sum(0)
            total += mask.sum()
    if context_parallel is not None:
        assignments = context_parallel.sum(assignments)
        probabilities = context_parallel.sum(probabilities, autograd=True)
        total = context_parallel.sum(total)
    return num_experts * ((assignments / total) * (probabilities / total)).sum()


@dataclass
class CausalLMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    aux_loss: torch.Tensor | None
    router_logits: tuple[torch.Tensor, ...]


class DeepSeekV41ForCausalLM(nn.Module):
    def __init__(
        self,
        config: DeepSeekV41Config,
        context_parallel: ContextParallel | None = None,
        token_dispatcher: TokenDispatcher | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.context_parallel = context_parallel or ContextParallel()
        self.model = DeepSeekV41Model(config, self.context_parallel, token_dispatcher)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=std)
        if isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
        elif isinstance(module, HyperConnection):
            nn.init.normal_(module.fn, std=std)
            nn.init.zeros_(module.base)
            nn.init.ones_(module.scale)
        elif isinstance(module, GroupedLinear):
            nn.init.normal_(module.weight, std=std)
        elif isinstance(module, TopKRouter):
            nn.init.normal_(module.weight, std=std)
            nn.init.zeros_(module.selection_bias)
        elif isinstance(module, RoutedExperts):
            if module.num_experts == module.global_num_experts:
                nn.init.normal_(module.gate_up, std=std)
                nn.init.normal_(module.down, std=std)
            else:
                generator = torch.Generator(device=module.gate_up.device)
                generator.manual_seed(torch.initial_seed() + module.expert_start)
                nn.init.normal_(module.gate_up, std=std, generator=generator)
                nn.init.normal_(module.down, std=std, generator=generator)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        token_mask = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.bool()
        )
        hidden, router_logits = self.model(input_ids, token_mask)
        logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            targets = torch.full_like(labels, -100)
            targets[:, :-1] = labels[:, 1:]
            targets[:, :-1].masked_fill_(~token_mask[:, 1:], -100)
            targets = self.context_parallel.shard(targets)
            loss_sum = F.cross_entropy(
                logits.float().reshape(-1, self.config.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            loss = self.context_parallel.mean_loss(loss_sum, (targets != -100).sum())

        local_token_mask = self.context_parallel.shard(token_mask)
        aux_loss = load_balancing_loss(
            router_logits,
            self.config.n_routed_experts,
            self.config.num_experts_per_tok,
            local_token_mask,
            self.context_parallel,
        )
        if loss is not None:
            loss = loss + self.config.router_aux_loss_coef * aux_loss
        return CausalLMOutput(loss, logits, aux_loss, router_logits)


# Keep the former spelling available for callers migrating from Transformers.
DeepseekV41ForCausalLM = DeepSeekV41ForCausalLM


__all__ = [
    "CausalLMOutput",
    "DeepSeekV41Config",
    "DeepSeekV41ForCausalLM",
    "DeepSeekV41Model",
    "DeepseekV41ForCausalLM",
]
