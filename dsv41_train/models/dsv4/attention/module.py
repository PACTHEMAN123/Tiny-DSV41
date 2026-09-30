"""DeepSeek V4 CSA2 attention modules and cross-layer reuse state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn

from ....cp import ModelContext
from ..config import DeepSeekV41Config
from .sparse_attention import is_available as triton_attention_available
from .sparse_attention import sparse_attention
from .sparse_indexer import fused_index_scores


class RMSNorm(nn.Module):
    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().square().mean(-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, config: DeepSeekV41Config) -> None:
        super().__init__()
        dim = config.qk_rope_head_dim
        main = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
        compressed = 1.0 / (
            config.compress_rope_theta ** (torch.arange(0, dim, 2).float() / dim)
        )
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
    rotary = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    ).flatten(-2)
    return torch.cat((plain, rotary.to(x.dtype)), dim=-1)


class GroupedLinear(nn.Module):
    def __init__(self, input_per_group: int, output_per_group: int, groups: int) -> None:
        super().__init__()
        self.groups = groups
        self.weight = nn.Parameter(
            torch.empty(groups, output_per_group, input_per_group)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        weight = self.weight.transpose(1, 2)
        grouped = x.reshape(-1, self.groups, hidden_dim).transpose(0, 1)
        output = torch.bmm(grouped, weight).transpose(0, 1)
        return output.reshape(*input_shape, self.groups, -1)


class AttentionSinks(nn.Module):
    """Keep trainable FP32 attention sinks in their own precision domain."""

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_heads))

    def forward(self) -> torch.Tensor:
        # Materialize the parameter before nested FSDP releases its storage.
        return self.weight.clone()


class CSA2Mode(str, Enum):
    FULL = "full"
    REINDEX = "reindex"
    REUSE = "reuse"


class Compressor(nn.Module):
    """Pool complete groups of live tokens into shared global KV latents."""

    def __init__(self, config: DeepSeekV41Config, layer_id: int) -> None:
        super().__init__()
        self.ratio = config.compress_ratios[layer_id]
        self.kv_proj = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.gate_proj = (
            nn.Linear(config.hidden_size, config.head_dim, bias=False)
            if self.ratio > 1
            else None
        )
        self.norm = RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        token_mask: torch.Tensor,
        sequence_ids: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, length, _ = x.shape
        live = token_mask.bool()
        end_indices = torch.arange(length, device=x.device) + self.ratio - 1
        safe_ends = end_indices.clamp_max(length - 1).expand(batch, -1)
        complete = live & positions.remainder(self.ratio).eq(0) & end_indices.lt(length)
        complete &= live.gather(1, safe_ends) & sequence_ids.eq(
            sequence_ids.gather(1, safe_ends)
        )
        group_counts = complete.long().sum(-1)

        if self.gate_proj is None:
            kv = self.kv_proj(x)
            gate = None
        else:
            kv = F.linear(x.float(), self.kv_proj.weight.float())
            gate = F.linear(x.float(), self.gate_proj.weight.float())
        destinations = torch.where(
            complete, complete.long().cumsum(-1) - 1, length
        )
        order = positions.new_zeros(batch, length + 1)
        order.scatter_(
            1,
            destinations,
            torch.arange(length, device=x.device).expand(batch, -1),
        )

        max_groups = length // self.ratio
        starts = order[:, :max_groups]
        indices = starts.unsqueeze(-1) + torch.arange(self.ratio, device=x.device)
        indices = indices.flatten(1)
        group_positions = positions.gather(1, starts)
        group_sequence_ids = sequence_ids.gather(1, starts)
        valid = torch.arange(max_groups, device=x.device) < group_counts.unsqueeze(-1)
        group_positions = group_positions.masked_fill(~valid, 0)
        group_sequence_ids = group_sequence_ids.masked_fill(~valid, -1)
        if max_groups == 0:
            return None, group_positions, group_sequence_ids, valid

        gathered = kv.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, kv.shape[-1])
        )
        if gate is None:
            latent = gathered
        else:
            gathered_gate = gate.gather(
                1, indices.unsqueeze(-1).expand(-1, -1, gate.shape[-1])
            )
            gathered = gathered.view(batch, max_groups, self.ratio, -1).float()
            gathered_gate = gathered_gate.view(batch, max_groups, self.ratio, -1)
            latent = (gathered * gathered_gate.softmax(dim=2)).sum(dim=2)
        latent = latent.masked_fill(~valid.unsqueeze(-1), 0)
        return self.norm(latent.to(x.dtype)), group_positions, group_sequence_ids, valid


@dataclass
class ShadowIndexers:
    """Global KV and index tensors reused across CSA2 layers."""

    compressed_sequence_ids: torch.Tensor | None = None
    compressed_kv: torch.Tensor | None = None
    index_keys: torch.Tensor | None = None
    topk_indices: torch.Tensor | None = None
    candidates: torch.Tensor | None = None

    @classmethod
    def load(cls, tensors: tuple[torch.Tensor, ...]) -> "ShadowIndexers":
        return cls(*(None if tensor.numel() == 0 else tensor for tensor in tensors))

    def dump(self, empty: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(
            value if value is not None else empty
            for value in (
                self.compressed_sequence_ids,
                self.compressed_kv,
                self.index_keys,
                self.topk_indices,
                self.candidates,
            )
        )


def select_candidate_blocks(
    scores: torch.Tensor, visible: torch.Tensor, topk_blocks: int, block_size: int
) -> torch.Tensor:
    width = scores.shape[-1]
    block_scores = F.pad(scores, (0, -width % block_size), value=float("-inf"))
    block_scores = block_scores.unflatten(-1, (-1, block_size)).amax(-1)
    last = (visible - 1) // block_size
    block_ids = torch.arange(block_scores.shape[-1], device=scores.device)
    block_scores = block_scores.masked_fill(block_ids == last.unsqueeze(-1), torch.inf)
    picked = block_scores.topk(min(topk_blocks, block_scores.shape[-1]), dim=-1)
    keep = torch.zeros_like(block_scores, dtype=torch.bool)
    keep.scatter_(-1, picked.indices, picked.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class SparseIndexer(nn.Module):
    def __init__(
        self, config: DeepSeekV41Config, layer_id: int, rotary: RotaryEmbedding
    ) -> None:
        super().__init__()
        self.owns_keys = layer_id in config.kv_source_layer_ids
        self.is_candidate_source = layer_id == config.candidate_source_layer_id
        self.uses_candidates = 0 <= config.candidate_source_layer_id < layer_id
        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.topk = config.index_topk
        self.topk_blocks = config.candidate_topk_blocks
        self.block_size = config.candidate_block_size
        self.ratio = config.compress_ratios[layer_id]
        self.rotary = rotary
        self.q_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.head_dim, bias=False
        )
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
        context: ModelContext,
        shadow: ShadowIndexers,
    ) -> None:
        batch, length, _ = x.shape
        if self.owns_keys:
            shadow.index_keys = None
            if latent is not None:
                keys = self.k_norm(self.k_proj(latent))
                cos, sin = self.rotary(keys, group_positions, compressed=True)
                shadow.index_keys = apply_rope(keys, cos, sin).unsqueeze(1)

        index_keys = shadow.index_keys
        if index_keys is None:
            shadow.topk_indices = None
            if self.is_candidate_source:
                shadow.candidates = None
            return

        keys = index_keys[:, 0]
        key_sequence_ids = shadow.compressed_sequence_ids
        assert key_sequence_ids is not None
        sentinel = torch.iinfo(key_sequence_ids.dtype).max
        searchable_ids = key_sequence_ids.masked_fill(
            key_sequence_ids < 0, sentinel
        ).contiguous()
        query_ids = context.sequence_ids.contiguous()
        key_starts = torch.searchsorted(searchable_ids, query_ids, right=False)
        sequence_ends = torch.searchsorted(searchable_ids, query_ids, right=True)
        visible_counts = (context.positions + 1) // self.ratio
        key_ends = torch.minimum(sequence_ends, key_starts + visible_counts)
        valid_queries = (context.sequence_ids >= 0) & (key_starts < key_ends)
        key_ends = torch.where(valid_queries, key_ends, key_starts)

        query = self.q_proj(q_residual).view(
            batch, length, self.num_heads, self.head_dim
        )
        cos, sin = self.rotary(query, context.positions, compressed=True)
        query = apply_rope(query, cos, sin)
        head_weights = self.weight_proj(x).float() * self.num_heads**-0.5
        max_sequence_width = int((sequence_ends - key_starts).max().item())
        previous_candidates = shadow.candidates if self.uses_candidates else None
        scores = fused_index_scores(
            query,
            keys,
            head_weights,
            key_starts,
            key_ends,
            max_sequence_width,
            previous_candidates,
        )
        if self.is_candidate_source:
            shadow.candidates = select_candidate_blocks(
                scores,
                key_ends - key_starts,
                self.topk_blocks,
                self.block_size,
            )
        picked = scores.topk(min(self.topk, scores.shape[-1]), dim=-1, sorted=False)
        picked_indices = key_starts.unsqueeze(-1) + picked.indices
        shadow.topk_indices = torch.where(
            picked.values > float("-inf"),
            picked_indices,
            torch.full_like(picked_indices, -1),
        )


class CSA2Attention(nn.Module):
    def __init__(
        self,
        config: DeepSeekV41Config,
        layer_id: int,
        rotary: RotaryEmbedding,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.ratio = config.compress_ratios[layer_id]
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.dropout = config.attention_dropout
        self.rotary = rotary
        self.q_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank, config.rms_norm_eps)
        self.q_b = nn.Linear(
            config.q_lora_rank, self.num_heads * self.head_dim, bias=False
        )
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        group_input = self.num_heads * self.head_dim // config.o_groups
        self.o_a = GroupedLinear(
            group_input, config.o_lora_rank, config.o_groups
        )
        self.o_b = nn.Linear(
            config.o_groups * config.o_lora_rank, config.hidden_size, bias=False
        )
        self.sinks = AttentionSinks(self.num_heads)
        self.compressor = (
            Compressor(config, layer_id)
            if layer_id in config.kv_source_layer_ids
            else None
        )
        self.indexer = (
            SparseIndexer(config, layer_id, rotary)
            if layer_id in config.index_source_layer_ids
            else None
        )
        self.mode = (
            None
            if self.ratio <= 0
            else CSA2Mode.FULL
            if self.compressor is not None
            else CSA2Mode.REINDEX
            if self.indexer is not None
            else CSA2Mode.REUSE
        )

    def forward(
        self,
        x: torch.Tensor,
        context: ModelContext,
        shadow: ShadowIndexers,
        source_x: torch.Tensor | None = None,
        source_context: ModelContext | None = None,
    ) -> torch.Tensor:
        batch, length, _ = x.shape
        compressed = self.ratio > 0
        cos, sin = self.rotary(x, context.positions, compressed=compressed)
        q_residual = self.q_norm(self.q_a(x))
        query = self.q_b(q_residual).view(
            batch, length, self.num_heads, self.head_dim
        )
        query = apply_rope(query, cos, sin).transpose(1, 2)
        kv = apply_rope(self.kv_norm(self.kv_proj(x)), cos, sin).unsqueeze(1)
        kv = context.gather_attention(kv, dim=2)[:, 0]

        compressed_entries = topk_indices = None
        if compressed:
            latent = None
            group_positions = context.positions[:, :0]
            if self.compressor is not None:
                compressor_x = x if source_x is None else source_x
                compressor_context = context if source_context is None else source_context
                full_x = compressor_context.gather(compressor_x)
                latent, group_positions, group_sequence_ids, _ = self.compressor(
                    full_x,
                    compressor_context.gather(compressor_context.positions),
                    compressor_context.gather(compressor_context.token_mask),
                    compressor_context.gather(compressor_context.sequence_ids),
                )
                shadow.compressed_sequence_ids = group_sequence_ids
                shadow.compressed_kv = None
                shadow.index_keys = None
                shadow.topk_indices = None
                shadow.candidates = None

            if self.indexer is not None:
                # Top-k is discrete; indexer parameters have no loss gradient.
                with torch.no_grad():
                    self.indexer(
                        x,
                        q_residual,
                        latent,
                        group_positions,
                        context,
                        shadow,
                    )

            if latent is not None:
                latent_cos, latent_sin = self.rotary(
                    latent, group_positions, compressed=True
                )
                shadow.compressed_kv = apply_rope(
                    latent, latent_cos, latent_sin
                ).unsqueeze(1)

            compressed_kv = shadow.compressed_kv
            topk_indices = shadow.topk_indices
            if compressed_kv is not None and topk_indices is not None:
                compressed_entries = compressed_kv[:, 0]

        output = self._attend(
            query,
            kv,
            context.attention_indices,
            context.attention_mask,
            compressed_entries,
            topk_indices,
        )
        output = apply_rope(output, cos, sin, inverse=True)
        grouped = output.reshape(batch, length, self.config.o_groups, -1)
        return self.o_b(self.o_a(grouped).flatten(2))

    def _attend(
        self,
        query: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        mask: torch.Tensor,
        compressed_kv: torch.Tensor | None,
        compressed_indices: torch.Tensor | None,
    ) -> torch.Tensor:
        sinks = self.sinks()
        if triton_attention_available(query) and not (self.training and self.dropout):
            return sparse_attention(
                query,
                kv,
                indices,
                mask[:, 0],
                compressed_kv,
                compressed_indices,
                sinks,
            ).transpose(1, 2).contiguous()

        rows = torch.arange(query.shape[0], device=query.device).view(-1, 1, 1)
        attention_kv = kv[rows, indices]
        selected = selected_valid = None
        if compressed_kv is not None and compressed_indices is not None:
            selected_valid = compressed_indices >= 0
            selected = compressed_kv[rows, compressed_indices.clamp_min(0)]
        logits = torch.einsum(
            "bhld,blwd->bhlw", query, attention_kv
        ) * self.head_dim**-0.5
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        window = logits.shape[-1]
        if selected is not None:
            picked = torch.einsum(
                "bhsd,bskd->bhsk", query, selected.to(query.dtype)
            )
            picked = picked * self.head_dim**-0.5
            picked = picked.masked_fill(
                ~selected_valid.unsqueeze(1), float("-inf")
            )
            logits = torch.cat((logits, picked), dim=-1)

        sink_logits = sinks.view(1, -1, 1, 1).expand(
            query.shape[0], -1, query.shape[-2], -1
        )
        combined = torch.cat((logits.float(), sink_logits), dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probabilities = torch.softmax(combined, dim=-1, dtype=combined.dtype)[..., :-1]
        probabilities = F.dropout(
            probabilities, self.dropout, self.training
        ).to(kv.dtype)
        output = torch.einsum(
            "bhlw,blwd->bhld", probabilities[..., :window], attention_kv
        )
        if selected is not None:
            output = output + torch.einsum(
                "bhsk,bskd->bhsd",
                probabilities[..., window:],
                selected.to(probabilities.dtype),
            )
        return output.transpose(1, 2).contiguous()

__all__ = [
    "AttentionSinks",
    "CSA2Attention",
    "CSA2Mode",
    "GroupedLinear",
    "RMSNorm",
    "RotaryEmbedding",
    "ShadowIndexers",
    "apply_rope",
]
