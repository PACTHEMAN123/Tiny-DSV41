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
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .attention import (
    CSA2Attention,
    GroupedLinear,
    RMSNorm,
    RotaryEmbedding,
    ShadowIndexers,
)
from .config import DeepSeekV41Config
from ...cp import ContextParallel, ModelContext
from .moe import RoutedExperts, SparseMoE, TokenDispatcher, TopKRouter

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


def _direct(function, *args, **_kwargs):
    return function(*args)


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


def _build_hash_multipliers(
    layer_ids: list[int], max_ngram: int, compressed_vocab_size: int
) -> torch.Tensor:
    """Rebuild the fixed hash constants used to address pretrained Engram rows."""
    bound = max(1, (np.iinfo(np.int64).max // compressed_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(0, bound, size=max_ngram, dtype=np.int64)
        rows.append(torch.from_numpy(values * 2 + 1))
    return torch.stack(rows)


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
        multipliers = _build_hash_multipliers(
            layer_ids, self.max_ngram, config.engram_compressed_vocab_size
        )
        self.compressed_vocab_size = config.engram_compressed_vocab_size
        self.register_buffer("token_map", token_map.to(torch.long), persistent=False)
        self.register_buffer("primes", primes, persistent=False)
        self.register_buffer("offsets", offsets, persistent=False)
        self.register_buffer("multipliers", multipliers, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_mask: torch.Tensor,
        sequence_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sequence_ids is None:
            sequence_ids = torch.zeros_like(input_ids)
        tokens = self.token_map[input_ids]
        dead = -1
        tokens = tokens.masked_fill(~token_mask, dead)
        sequence_ids = sequence_ids.masked_fill(~token_mask, dead)
        context = self.max_ngram - 1
        history = F.pad(tokens, (context, 0), value=dead)
        sequence_history = F.pad(sequence_ids, (context, 0), value=dead)
        shifted, blocked = [], torch.zeros_like(tokens, dtype=torch.bool)
        for distance in range(self.max_ngram):
            source = history[:, context - distance : context - distance + tokens.shape[1]]
            source_sequence_ids = sequence_history[
                :, context - distance : context - distance + tokens.shape[1]
            ]
            blocked |= (source == dead) | (source_sequence_ids != sequence_ids)
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
        token_dispatcher: TokenDispatcher,
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attention = CSA2Attention(config, layer_id, rotary)
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
        context: ModelContext,
        shadow: ShadowIndexers,
        engram_rows: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.engram is not None:
            streams = self.engram(streams, engram_rows, context.token_mask)

        residual = streams
        next_pre, post, combine = self.attention_hc(streams)
        collapsed = self.collapse(streams, pre_mix)
        value = self.attention(
            self.input_norm(collapsed),
            context,
            shadow,
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
        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        sequence_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, global_length = input_ids.shape
        if global_length > self.config.max_position_embeddings:
            raise ValueError("sequence is longer than max_position_embeddings")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if sequence_ids is None:
            sequence_ids = torch.zeros_like(input_ids)
        elif sequence_ids.shape != input_ids.shape:
            raise ValueError("sequence_ids must have the same shape as input_ids")
        sequence_ids = sequence_ids.masked_fill(~attention_mask, -1)

        context = ModelContext.build(
            self.context_parallel,
            input_ids,
            attention_mask,
            sequence_ids,
            self.config.sliding_window,
        )
        hashes = self.hash(input_ids, attention_mask, sequence_ids) if self.hash else None

        hidden = self.embedding(context.input_ids)
        batch, length = context.input_ids.shape
        streams = hidden.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()

        engram_rows: dict[int, torch.Tensor] = {}
        if hashes is not None:
            hashes = self.context_parallel.shard(hashes)
            for index, layer_id in enumerate(self.engram_layer_ids):
                engram_rows[layer_id] = self.engram_tables[str(layer_id)](hashes[:, :, index])

        pre_mix = hidden.new_zeros(batch, length, self.config.hc_mult, dtype=torch.float32)
        pre_mix[..., 0] = 1
        empty = hidden.new_empty(0)
        shadow = ShadowIndexers().dump(empty)
        loader = checkpoint if self.gradient_checkpointing and self.training else _direct
        router_logits = []
        for layer in self.layers:
            streams, pre_mix, logits, *shadow = loader(
                lambda *args, current=layer: self._forward_layer(current, context, *args),
                streams,
                pre_mix,
                engram_rows.get(layer.layer_id, empty),
                *shadow,
                use_reentrant=False,
            )
            router_logits.append(logits)

        hidden = DecoderLayer.collapse(streams, pre_mix)
        return self.norm(hidden), tuple(router_logits)

    def _forward_layer(
        self,
        layer: DecoderLayer,
        context: ModelContext,
        streams: torch.Tensor,
        pre_mix: torch.Tensor,
        engram_row: torch.Tensor,
        *shadow_tensors: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        shadow = ShadowIndexers.load(shadow_tensors)
        streams, pre_mix, logits = layer(
            streams,
            pre_mix,
            context,
            shadow,
            None if engram_row.numel() == 0 else engram_row,
        )
        return streams, pre_mix, logits, *shadow.dump(streams.new_empty(0))


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
    logits: torch.Tensor | None
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
            if module.gate_up.is_meta:
                return
            if module.num_experts == module.global_num_experts:
                nn.init.normal_(module.gate_up, std=std)
                nn.init.normal_(module.down, std=std)
            else:
                generator = torch.Generator(device=module.gate_up.device)
                generator.manual_seed(torch.initial_seed() + module.expert_start)
                nn.init.normal_(module.gate_up, std=std, generator=generator)
                nn.init.normal_(module.down, std=std, generator=generator)

    def gradient_checkpointing_enable(self) -> None:
        self.model.gradient_checkpointing = True

    def _lm_loss(self, hidden: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden)
        return F.cross_entropy(
            logits.float().reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        sequence_ids: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        token_mask = (
            torch.ones_like(input_ids, dtype=torch.bool)
            if attention_mask is None
            else attention_mask.bool()
        )
        hidden, router_logits = self.model(input_ids, token_mask, sequence_ids)
        logits = None
        loss = None
        if labels is not None:
            targets = torch.full_like(labels, -100)
            targets[:, :-1] = labels[:, 1:]
            targets[:, :-1].masked_fill_(~token_mask[:, 1:], -100)
            if sequence_ids is not None:
                targets[:, :-1].masked_fill_(
                    sequence_ids[:, :-1] != sequence_ids[:, 1:], -100
                )
            targets = self.context_parallel.shard(targets)
            if self.training and torch.is_grad_enabled():
                loss_sum = checkpoint(
                    self._lm_loss,
                    hidden,
                    targets,
                    use_reentrant=False,
                )
            else:
                logits = self.lm_head(hidden)
                loss_sum = F.cross_entropy(
                    logits.float().reshape(-1, self.config.vocab_size),
                    targets.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
            loss = self.context_parallel.mean_loss(loss_sum, (targets != -100).sum())
        else:
            logits = self.lm_head(hidden)

        local_token_mask = self.context_parallel.shard(token_mask)
        auxiliary_logits = tuple(logits.detach().requires_grad_() for logits in router_logits)
        aux_loss = load_balancing_loss(
            auxiliary_logits,
            self.config.n_routed_experts,
            self.config.num_experts_per_tok,
            local_token_mask,
            self.context_parallel,
        )
        if loss is not None:
            coefficient = self.config.router_aux_loss_coef
            if torch.is_grad_enabled() and coefficient:
                gradients = torch.autograd.grad(aux_loss, auxiliary_logits)
                for layer, gradient in zip(self.model.layers, gradients):
                    layer.moe.stage_router_aux_gradient(gradient * coefficient)
            loss = loss + coefficient * aux_loss.detach()
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
