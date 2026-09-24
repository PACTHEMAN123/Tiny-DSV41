# Portions Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0. See https://www.apache.org/licenses/LICENSE-2.0
"""A compact, training-only DeepSeek V4.1 model implemented with PyTorch alone.

This keeps the architecture used by the tiny trainer: mHC residual streams, CSA2
compressed attention, sparse MoE layers, and optional Engram memory. Generation
caches, multimodal modules, distributed kernels, and checkpoint conversion are
deliberately outside this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch import nn

from .config import DeepSeekV41Config
from .moe import RoutedExperts, SparseMoE, TopKRouter
from ...dispatch import TokenDispatcher
from ...parallel import ContextParallel

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


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
        comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        for _ in range(self.sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + self.eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + self.eps)
        return pre, post, comb


class RotaryEmbedding(nn.Module):
    def __init__(self, config: DeepSeekV41Config) -> None:
        super().__init__()
        dim = config.qk_rope_head_dim
        main = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        compressed = 1.0 / (config.compress_rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        scaling = config.rope_scaling or {}
        if scaling.get("rope_type", scaling.get("type")) == "yarn":
            factor = scaling["factor"]
            original = scaling["original_max_position_embeddings"]
            beta_fast = scaling.get("beta_fast", 32)
            beta_slow = scaling.get("beta_slow", 1)

            def corrected_dim(rotations: float) -> float:
                return dim * math.log(original / (rotations * 2 * math.pi)) / (
                    2 * math.log(config.compress_rope_theta)
                )

            low = max(math.floor(corrected_dim(beta_fast)), 0)
            high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
            ramp = (
                (torch.arange(dim // 2, dtype=torch.float32) - low)
                / max(high - low, 1e-3)
            ).clamp(0, 1)
            smooth = 1 - ramp
            compressed = compressed / factor * (1 - smooth) + compressed * smooth
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
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        weight = self.weight.transpose(1, 2)
        grouped = x.reshape(-1, self.groups, hidden_dim).transpose(0, 1)
        output = torch.bmm(grouped, weight).transpose(0, 1)
        return output.reshape(*input_shape, self.groups, -1)


_FP4_MAX = 6.0
_FP4_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)
_FP8_MAX = 448.0


def _pow2_ceil_scale(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int32)
    exponent = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    power = exponent - 127 + (mantissa != 0).to(torch.int32)
    return torch.exp2(power.to(torch.float32))


def _e2m1_codes(values: torch.Tensor) -> torch.Tensor:
    magnitude = values.abs()
    negative = torch.signbit(values)
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=torch.float32,
        device=values.device,
    )
    ties_up = torch.tensor(
        [False, True, False, True, False, True, False], device=values.device
    )
    thresholds = torch.where(
        ties_up,
        boundaries,
        torch.nextafter(boundaries, torch.full_like(boundaries, float("inf"))),
    )
    codes = (magnitude.unsqueeze(-1) >= thresholds).sum(-1).to(torch.uint8)
    return codes | (negative.to(torch.uint8) << 3)


def _fake_quant_fp4_block(
    tensor: torch.Tensor, block_size: int, *, e4m3_scales: bool = False
) -> torch.Tensor:
    width = tensor.shape[-1]
    if width % block_size:
        return tensor
    blocks = tensor.float().view(*tensor.shape[:-1], width // block_size, block_size)
    maximum = blocks.abs().amax(-1)
    if e4m3_scales:
        scale = (maximum.clamp_min(_FP4_MAX * 2.0**-9) / _FP4_MAX).to(
            torch.float8_e4m3fn
        ).float()
    else:
        scale = _pow2_ceil_scale(
            maximum.clamp_min(_FP4_MAX * 2.0**-126) / _FP4_MAX
        )
    quantized = (blocks / scale.unsqueeze(-1)).clamp(-_FP4_MAX, _FP4_MAX)
    lookup = torch.tensor(_FP4_VALUES, device=tensor.device, dtype=torch.float32)
    values = lookup[_e2m1_codes(quantized).long()]
    return (values * scale.unsqueeze(-1)).view_as(tensor).to(tensor.dtype)


def _fake_quant_fp8_block(tensor: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    width = tensor.shape[-1]
    if width % block_size:
        return tensor
    blocks = tensor.float().view(*tensor.shape[:-1], width // block_size, block_size)
    maximum = blocks.abs().amax(-1).clamp_min(1e-4)
    scale = _pow2_ceil_scale(maximum / _FP8_MAX)
    quantized = (blocks / scale.unsqueeze(-1)).clamp(-_FP8_MAX, _FP8_MAX)
    values = quantized.to(torch.float8_e4m3fn).float() * scale.unsqueeze(-1)
    return values.view_as(tensor).to(tensor.dtype)


class AttentionSinks(nn.Module):
    """Keep trainable FP32 attention sinks in their own precision domain."""

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_heads))

    def forward(self) -> torch.Tensor:
        # The result outlives this nested FSDP module's forward. Materialize it
        # before FSDP reshards and releases the unsharded parameter storage.
        return self.weight.clone()


class _ScaleGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return tensor

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return gradient * ctx.scale, None


class RowShardedEmbedding(nn.Module):
    """Shard embedding rows over a mesh and route lookups to their owner."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        mesh: DeviceMesh | None = None,
        sparse_gradients: bool = False,
    ) -> None:
        super().__init__()
        if mesh is not None and mesh.ndim != 1:
            raise ValueError("the embedding mesh must be one-dimensional")
        self.global_num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.mesh = mesh
        self.sparse_gradients = sparse_gradients
        self.size = 1 if mesh is None else mesh.size()
        self.rank = 0 if mesh is None else mesh.get_local_rank()
        self.group = None if mesh is None else mesh.get_group()
        if num_embeddings < self.size:
            raise ValueError("embedding rows must be at least the embedding mesh size")

        base, remainder = divmod(num_embeddings, self.size)
        self.row_start = self.rank * base + min(self.rank, remainder)
        self.row_stop = self.row_start + base + int(self.rank < remainder)
        self.weight = nn.Parameter(torch.empty(self.row_stop - self.row_start, embedding_dim))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        if self.size == 1:
            return F.embedding(indices, self.weight, sparse=self.sparse_gradients)
        if indices.numel() and (indices.min() < 0 or indices.max() >= self.global_num_embeddings):
            raise IndexError("embedding index is outside the configured row range")

        shape = indices.shape
        flat = indices.reshape(-1)
        base, remainder = divmod(self.global_num_embeddings, self.size)
        large_rows = (base + 1) * remainder
        owners = torch.where(
            flat < large_rows,
            torch.div(flat, base + 1, rounding_mode="floor"),
            remainder + torch.div(flat - large_rows, base, rounding_mode="floor"),
        )
        order = torch.argsort(owners, stable=True)
        send_counts = torch.bincount(owners, minlength=self.size).to(torch.int64)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=self.group)
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        routed = self._exchange(
            flat[order].contiguous(), recv_splits, send_splits, autograd=False
        )
        rows = F.embedding(
            routed - self.row_start,
            self.weight,
            sparse=self.sparse_gradients,
        )
        rows = _ScaleGradient.apply(rows, 1.0 / self.size)
        returned = self._exchange(
            rows.contiguous(), send_splits, recv_splits, autograd=True
        )
        return returned[torch.argsort(order)].view(*shape, self.embedding_dim)

    def _exchange(
        self,
        tensor: torch.Tensor,
        output_splits: list[int],
        input_splits: list[int],
        *,
        autograd: bool,
    ) -> torch.Tensor:
        output = tensor.new_empty((sum(output_splits), *tensor.shape[1:]))
        if autograd:
            return dist_nn.all_to_all_single(
                output,
                tensor,
                output_split_sizes=output_splits,
                input_split_sizes=input_splits,
                group=self.group,
            )
        dist.all_to_all_single(
            output,
            tensor,
            output_split_sizes=output_splits,
            input_split_sizes=input_splits,
            group=self.group,
        )
        return output


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

        if self.gate_proj is None:
            kv = self.kv_proj(x)
            gate = None
        else:
            kv = F.linear(x.float(), self.kv_proj.weight.float())
            gate = F.linear(x.float(), self.gate_proj.weight.float())
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
                keys = apply_rope(keys, cos, sin)
                shared["index_keys"] = _fake_quant_fp4_block(
                    keys, 32
                ).unsqueeze(1)

        index_keys = shared.get("index_keys")
        if index_keys is None:
            shared["topk_indices"] = None
            if self.is_candidate_source:
                shared["candidates"] = None
            return

        keys = index_keys[:, 0].float()
        query = self.q_proj(q_residual).view(batch, length, self.num_heads, self.head_dim)
        cos, sin = self.rotary(query, positions, compressed=True)
        query = _fake_quant_fp4_block(
            apply_rope(query, cos, sin), 32
        ).float()
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

        picked = scores.topk(min(self.topk, keys.shape[1]), dim=-1)
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
        self.sinks = AttentionSinks(self.num_heads)
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
        kv = _fake_quant_fp8_block(kv)
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
                rotated = apply_rope(latent, latent_cos, latent_sin)
                shared["compressed_kv"] = _fake_quant_fp4_block(
                    rotated, 16, e4m3_scales=True
                ).unsqueeze(1)

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

        sinks = self.sinks().view(1, -1, 1, 1).expand(
            query.shape[0], -1, query.shape[-2], -1
        )
        combined = torch.cat((logits.float(), sinks), dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probabilities = torch.softmax(combined, dim=-1, dtype=combined.dtype)[..., :-1]
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


def build_compressed_token_map(tokenizer_file: str | Path) -> tuple[torch.Tensor, int]:
    try:
        from tokenizers import Regex, Tokenizer, normalizers
    except ImportError as error:
        raise RuntimeError(
            "loading Engram checkpoint weights requires the tokenizers package"
        ) from error

    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    compressed: dict[str, int] = {}
    mapping = []
    for token_id in range(tokenizer.get_vocab_size(with_added_tokens=True)):
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            key = tokenizer.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized or text
        new_id = compressed.get(key)
        if new_id is None:
            new_id = len(compressed)
            compressed[key] = new_id
        mapping.append(new_id)
    return torch.tensor(mapping, dtype=torch.long), len(compressed)


class NgramHash(nn.Module):
    def __init__(
        self,
        config: DeepSeekV41Config,
        layer_ids: list[int] | None = None,
        token_map: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.max_ngram = config.engram_max_ngram_size
        if token_map is None:
            token_map = torch.arange(config.vocab_size).remainder(
                config.engram_compressed_vocab_size
            )
        if token_map.ndim != 1 or token_map.shape[0] != config.vocab_size:
            raise ValueError("Engram token map must contain one entry per vocabulary id")
        if not token_map.is_meta and token_map.numel() and (
            token_map.min() < 0
            or token_map.max() >= config.engram_compressed_vocab_size
        ):
            raise ValueError("Engram token map contains an out-of-range compressed id")
        self.pad_id = (
            config.engram_pad_id % config.engram_compressed_vocab_size
            if token_map.is_meta
            else int(token_map[config.engram_pad_id])
        )
        layer_ids = list(config.engram_layer_ids if layer_ids is None else layer_ids)
        if not set(layer_ids).issubset(config.engram_layer_ids):
            raise ValueError("Engram hash layers must be configured Engram layers")
        selected = set(layer_ids)
        layer_primes, layer_offsets, used = [], [], set()
        for layer_id, table_size in zip(
            config.engram_layer_ids, config.engram_num_embeddings
        ):
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
            if layer_id in selected:
                layer_primes.append(primes)
                layer_offsets.append(offsets)

        primes = torch.tensor(layer_primes).view(
            len(layer_ids), self.max_ngram - 1, config.engram_n_heads
        )
        offsets = torch.tensor(layer_offsets)
        bound = max(
            1,
            (np.iinfo(np.int64).max // config.engram_compressed_vocab_size) // 2,
        )
        multipliers = []
        for layer_id in layer_ids:
            generator = np.random.default_rng(10007 * layer_id)
            values = generator.integers(
                low=0,
                high=bound,
                size=(self.max_ngram,),
                dtype=np.int64,
            )
            multipliers.append(torch.from_numpy(values * 2 + 1))
        self.compressed_vocab_size = config.engram_compressed_vocab_size
        self.register_buffer("token_map", token_map.to(torch.long), persistent=False)
        self.register_buffer("primes", primes, persistent=False)
        self.register_buffer("offsets", offsets, persistent=False)
        self.register_buffer("multipliers", torch.stack(multipliers), persistent=False)

    def forward(self, input_ids: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        tokens = self.token_map[input_ids]
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
        engram_mesh: DeviceMesh | None = None,
        layer_ids: list[int] | None = None,
        sparse_engram_gradients: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.context_parallel = context_parallel or ContextParallel()
        token_dispatcher = token_dispatcher or TokenDispatcher()
        layer_ids = (
            list(range(config.num_hidden_layers)) if layer_ids is None else layer_ids
        )
        if not layer_ids or any(
            not 0 <= layer_id < config.num_hidden_layers for layer_id in layer_ids
        ):
            raise ValueError("layer_ids must select at least one configured layer")
        self.layer_ids = list(layer_ids)
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
            for layer_id in self.layer_ids
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        table_sizes = dict(zip(config.engram_layer_ids, config.engram_num_embeddings))
        self.engram_layer_ids = [
            layer_id for layer_id in config.engram_layer_ids if layer_id in self.layer_ids
        ]
        self.hash = (
            NgramHash(config, self.engram_layer_ids) if self.engram_layer_ids else None
        )
        self.engram_tables = nn.ModuleDict(
            {
                str(layer_id): RowShardedEmbedding(
                    table_sizes[layer_id],
                    config.engram_head_dim,
                    engram_mesh,
                    sparse_gradients=sparse_engram_gradients,
                )
                for layer_id in self.engram_layer_ids
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
            for index, layer_id in enumerate(self.engram_layer_ids):
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
        engram_mesh: DeviceMesh | None = None,
        layer_ids: list[int] | None = None,
        sparse_engram_gradients: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.context_parallel = context_parallel or ContextParallel()
        self.model = DeepSeekV41Model(
            config,
            self.context_parallel,
            token_dispatcher,
            engram_mesh,
            layer_ids,
            sparse_engram_gradients,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if any(parameter.is_meta for parameter in module.parameters(recurse=False)):
            return
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Embedding, RowShardedEmbedding)):
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
            if not module.gate_up or module.gate_up[0].is_meta:
                return
            if module.num_experts == module.global_num_experts:
                for parameter in module.gate_up:
                    nn.init.normal_(parameter, std=std)
                for parameter in module.down:
                    nn.init.normal_(parameter, std=std)
            else:
                generator = torch.Generator(device=module.gate_up[0].device)
                generator.manual_seed(torch.initial_seed() + module.expert_start)
                for parameter in module.gate_up:
                    nn.init.normal_(parameter, std=std, generator=generator)
                for parameter in module.down:
                    nn.init.normal_(parameter, std=std, generator=generator)

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
    "RowShardedEmbedding",
]
