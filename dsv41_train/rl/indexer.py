# SparseIndexer.forward follows the native DSV4 implementation with stable top-k.
# Portions Copyright 2026 The HuggingFace Inc. team. Apache-2.0.
"""Portable sparse indexing with prefix-stable tie handling for RL probabilities."""

import torch
import torch.nn.functional as F

from dsv41_train.models.dsv4.attention.module import SparseIndexer, apply_rope


def index_scores(query, keys, weights, starts, ends, width, candidates=None):
    if width == 0:
        return query.new_empty((*query.shape[:2], 0), dtype=torch.float32)
    indices = starts.unsqueeze(-1) + torch.arange(width, device=query.device)
    valid = indices < ends.unsqueeze(-1)
    if candidates is not None:
        valid = valid & candidates
    rows = torch.arange(query.shape[0], device=query.device)[:, None, None]
    gathered = keys[rows, indices.clamp_max(keys.shape[1] - 1)]
    dots = torch.einsum("blhd,blkd->blkh", query.float(), gathered.float())
    scores = (dots.relu() * query.shape[-1] ** -0.5 * weights.float().unsqueeze(2)).sum(-1)
    return scores.masked_fill(~valid, float("-inf"))


def _select(scores, count):
    indices = torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :count]
    return indices, scores.gather(-1, indices)


def _candidate_blocks(scores, visible, count, block_size):
    width = scores.shape[-1]
    blocks = F.pad(scores, (0, -width % block_size), value=float("-inf"))
    blocks = blocks.unflatten(-1, (-1, block_size)).amax(-1)
    last = (visible - 1) // block_size
    blocks = blocks.masked_fill(torch.arange(blocks.shape[-1], device=scores.device) == last.unsqueeze(-1),
                                torch.inf)
    indices, values = _select(blocks, min(count, blocks.shape[-1]))
    keep = torch.zeros_like(blocks, dtype=torch.bool)
    keep.scatter_(-1, indices, values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


class PortableSparseIndexer(SparseIndexer):
    def forward(self, x, q_residual, latent, group_positions, context, shadow):
        batch, length, _ = x.shape
        if self.owns_keys:
            shadow.index_keys = None
            if latent is not None:
                keys = self.k_norm(self.k_proj(latent))
                cos, sin = self.rotary(keys, group_positions, compressed=True)
                shadow.index_keys = apply_rope(keys, cos, sin).unsqueeze(1)
        if shadow.index_keys is None:
            shadow.topk_indices = None
            if self.is_candidate_source:
                shadow.candidates = None
            return
        keys = shadow.index_keys[:, 0]
        sequence_ids = shadow.compressed_sequence_ids
        searchable = sequence_ids.masked_fill(sequence_ids < 0, torch.iinfo(sequence_ids.dtype).max).contiguous()
        starts = torch.searchsorted(searchable, context.sequence_ids.contiguous(), right=False)
        sequence_ends = torch.searchsorted(searchable, context.sequence_ids.contiguous(), right=True)
        ends = torch.minimum(sequence_ends, starts + (context.positions + 1) // self.ratio)
        ends = torch.where((context.sequence_ids >= 0) & (starts < ends), ends, starts)
        query = self.q_proj(q_residual).view(batch, length, self.num_heads, self.head_dim)
        cos, sin = self.rotary(query, context.positions, compressed=True)
        query = apply_rope(query, cos, sin)
        weights = self.weight_proj(x).float() * self.num_heads ** -0.5
        scores = index_scores(query, keys, weights, starts, ends,
                              int((sequence_ends - starts).max().item()),
                              shadow.candidates if self.uses_candidates else None)
        if self.is_candidate_source:
            shadow.candidates = _candidate_blocks(scores, ends - starts, self.topk_blocks, self.block_size)
        indices, values = _select(scores, min(self.topk, scores.shape[-1]))
        shadow.topk_indices = torch.where(values > float("-inf"), starts.unsqueeze(-1) + indices, -1)
