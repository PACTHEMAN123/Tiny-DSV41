# Portions Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0. See https://www.apache.org/licenses/LICENSE-2.0
"""Configuration for the small, training-only DeepSeek V4.1 implementation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class DeepSeekV41Config:
    vocab_size: int = 129280
    hidden_size: int = 5120
    num_hidden_layers: int = 40
    num_attention_heads: int = 64
    head_dim: int = 512
    q_lora_rank: int = 1280
    qk_rope_head_dim: int = 64
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_scaling: dict | None = None
    max_position_embeddings: int = 1048576
    attention_dropout: float = 0.0

    sliding_window: int = 128
    compress_ratios: list[int] | None = None
    kv_source_layer_ids: list[int] | None = None
    index_source_layer_ids: list[int] | None = None
    candidate_source_layer_id: int | None = None
    candidate_topk_blocks: int = 2048
    candidate_block_size: int = 8
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 512

    o_groups: int = 8
    o_lora_rank: int = 1024

    moe_intermediate_size: int = 2304
    n_routed_experts: int = 384
    num_experts_per_tok: int = 6
    scoring_func: str = "sqrtsoftplus"
    gate_temp: float = 1.0
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    swiglu_limit: float = 10.0

    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1.0e-6

    engram_layer_ids: list[int] | None = None
    engram_vocab_size: int = 16000000
    engram_num_embeddings: list[int] | None = None
    engram_max_ngram_size: int = 4
    engram_n_heads: int = 8
    engram_head_dim: int = 256
    engram_pad_id: int = 2
    engram_compressed_vocab_size: int = 99092

    initializer_range: float = 0.02
    rms_norm_eps: float = 1.0e-20
    router_aux_loss_coef: float = 0.001
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def __post_init__(self) -> None:
        n = self.num_hidden_layers
        if self.compress_ratios is None:
            if n == 40:
                self.compress_ratios = [0, 0] + [2] * 18 + [1] * 20
            else:
                n_slide = min(2, n)
                n_encoder = max(0, (n - n_slide + 1) // 2)
                self.compress_ratios = [0] * n_slide + [2] * n_encoder + [1] * (n - n_slide - n_encoder)
        self.compress_ratios = list(self.compress_ratios[:n])
        if len(self.compress_ratios) != n:
            raise ValueError("compress_ratios must contain one value per layer")

        if self.kv_source_layer_ids is None:
            if n == 40 and self.compress_ratios == [0, 0] + [2] * 18 + [1] * 20:
                self.kv_source_layer_ids = [2, 8, 14, 20]
            else:
                self.kv_source_layer_ids = [
                    i
                    for i, ratio in enumerate(self.compress_ratios)
                    if ratio > 0 and (i == 0 or self.compress_ratios[i - 1] != ratio)
                ]
        else:
            self.kv_source_layer_ids = list(self.kv_source_layer_ids)

        if self.index_source_layer_ids is None:
            if self.kv_source_layer_ids == [2, 8, 14, 20]:
                self.index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
            else:
                self.index_source_layer_ids = list(self.kv_source_layer_ids)
        else:
            self.index_source_layer_ids = list(self.index_source_layer_ids)

        if self.candidate_source_layer_id is None:
            self.candidate_source_layer_id = self.kv_source_layer_ids[-1] if self.kv_source_layer_ids else -1

        self.engram_layer_ids = list(self.engram_layer_ids or [])
        self.engram_num_embeddings = list(self.engram_num_embeddings or [0] * len(self.engram_layer_ids))
        self._validate()

    def _validate(self) -> None:
        if self.hidden_size <= 0 or self.num_hidden_layers <= 0:
            raise ValueError("hidden_size and num_hidden_layers must be positive")
        if self.head_dim < self.qk_rope_head_dim or self.qk_rope_head_dim % 2:
            raise ValueError("qk_rope_head_dim must be even and no larger than head_dim")
        if self.num_attention_heads * self.head_dim % self.o_groups:
            raise ValueError("num_attention_heads * head_dim must be divisible by o_groups")
        if not 1 <= self.num_experts_per_tok <= self.n_routed_experts:
            raise ValueError("num_experts_per_tok must be in [1, n_routed_experts]")
        if self.scoring_func not in {"sqrtsoftplus", "softmax", "sigmoid"}:
            raise ValueError(f"unsupported scoring_func: {self.scoring_func}")
        if len(self.engram_num_embeddings) != len(self.engram_layer_ids):
            raise ValueError("engram_num_embeddings must match engram_layer_ids")

        for layer_id in self.kv_source_layer_ids:
            if not 0 <= layer_id < self.num_hidden_layers or self.compress_ratios[layer_id] <= 0:
                raise ValueError(f"invalid KV source layer: {layer_id}")
        for layer_id in self.index_source_layer_ids:
            if not 0 <= layer_id < self.num_hidden_layers or self.compress_ratios[layer_id] <= 0:
                raise ValueError(f"invalid index source layer: {layer_id}")
            if not any(source <= layer_id for source in self.kv_source_layer_ids):
                raise ValueError(f"index source {layer_id} has no preceding KV source")
        for layer_id, ratio in enumerate(self.compress_ratios):
            if ratio > 0 and not any(source <= layer_id for source in self.kv_source_layer_ids):
                raise ValueError(f"compressed layer {layer_id} has no preceding KV source")
        if self.candidate_source_layer_id >= 0 and self.candidate_source_layer_id not in self.index_source_layer_ids:
            raise ValueError("candidate_source_layer_id must be an index source")

    @classmethod
    def tiny(cls) -> "DeepSeekV41Config":
        return cls(
            vocab_size=32,
            hidden_size=256,
            num_hidden_layers=6,
            num_attention_heads=8,
            head_dim=64,
            q_lora_rank=128,
            qk_rope_head_dim=32,
            max_position_embeddings=128,
            sliding_window=32,
            compress_ratios=[0, 2, 2, 1, 1, 1],
            kv_source_layer_ids=[1, 3],
            index_source_layer_ids=[1, 3, 4],
            candidate_source_layer_id=3,
            candidate_topk_blocks=4,
            candidate_block_size=4,
            index_n_heads=4,
            index_head_dim=64,
            index_topk=16,
            o_groups=4,
            o_lora_rank=128,
            moe_intermediate_size=512,
            n_routed_experts=8,
            num_experts_per_tok=2,
            routed_scaling_factor=1.0,
            engram_layer_ids=[1],
            engram_num_embeddings=[4096],
            engram_vocab_size=251,
            engram_n_heads=2,
            engram_head_dim=32,
            engram_pad_id=0,
            engram_compressed_vocab_size=32,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "DeepSeekV41Config":
        with Path(path).open(encoding="utf-8") as file:
            values = json.load(file)
        values = values.get("text_config", values)
        if "engram_pad_token_id" in values and "engram_pad_id" not in values:
            values["engram_pad_id"] = values["engram_pad_token_id"]
        known = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})

    def to_dict(self) -> dict:
        return asdict(self)
