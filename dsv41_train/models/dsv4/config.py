# Portions Copyright 2026 The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0. See https://www.apache.org/licenses/LICENSE-2.0
"""DeepSeek V4.1 architecture supplied by a checkpoint config."""

from __future__ import annotations

import json
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path


@dataclass(kw_only=True)
class DeepSeekV41Config:
    vocab_size: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    head_dim: int
    q_lora_rank: int
    qk_rope_head_dim: int
    rope_theta: float
    compress_rope_theta: float
    rope_scaling: dict | None = None
    max_position_embeddings: int
    attention_dropout: float

    sliding_window: int
    compress_ratios: list[int]
    kv_source_layer_ids: list[int]
    index_source_layer_ids: list[int]
    candidate_source_layer_id: int
    candidate_topk_blocks: int
    candidate_block_size: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int

    o_groups: int
    o_lora_rank: int

    moe_intermediate_size: int
    n_routed_experts: int
    n_shared_experts: int
    num_experts_per_tok: int
    scoring_func: str
    gate_temp: float = 1.0
    norm_topk_prob: bool
    routed_scaling_factor: float
    swiglu_limit: float

    hc_mult: int
    hc_sinkhorn_iters: int
    hc_eps: float

    engram_layer_ids: list[int]
    engram_vocab_size: int
    engram_num_embeddings: list[int]
    engram_max_ngram_size: int
    engram_n_heads: int
    engram_head_dim: int
    engram_pad_id: int
    engram_compressed_vocab_size: int

    initializer_range: float
    rms_norm_eps: float
    router_aux_loss_coef: float = 0.001
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int

    def __post_init__(self) -> None:
        for name in ("compress_ratios", "kv_source_layer_ids", "index_source_layer_ids",
                     "engram_layer_ids", "engram_num_embeddings"):
            setattr(self, name, list(getattr(self, name)))
        self._validate()

    def _validate(self) -> None:
        for name in ("vocab_size", "hidden_size", "num_hidden_layers", "num_attention_heads",
                     "head_dim", "q_lora_rank", "qk_rope_head_dim", "max_position_embeddings",
                     "sliding_window", "candidate_topk_blocks", "candidate_block_size",
                     "index_n_heads", "index_head_dim", "index_topk", "o_groups", "o_lora_rank",
                     "moe_intermediate_size", "n_routed_experts", "n_shared_experts", "hc_mult",
                     "hc_sinkhorn_iters", "engram_vocab_size", "engram_n_heads",
                     "engram_head_dim", "engram_compressed_vocab_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.compress_ratios) != self.num_hidden_layers or any(
            type(ratio) is not int or ratio < 0 for ratio in self.compress_ratios
        ):
            raise ValueError("compress_ratios must contain one nonnegative integer per decoder layer")
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
        """Load the native model test fixture; runtime loaders never select it implicitly."""
        return cls.from_json(Path(__file__).with_name("tiny_config.json"))

    @classmethod
    def from_json(cls, path: str | Path) -> "DeepSeekV41Config":
        with Path(path).open(encoding="utf-8") as file:
            return cls.from_dict(json.load(file))

    @classmethod
    def from_dict(cls, root: dict) -> "DeepSeekV41Config":
        values = dict(root.get("text_config", root))
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if root.get(name) is not None:
                values[name] = root[name]
        if "engram_pad_token_id" in values and "engram_pad_id" not in values:
            values["engram_pad_id"] = values["engram_pad_token_id"]
        known = {field.name: field for field in fields(cls)}
        missing = [name for name, field in known.items()
                   if field.default is MISSING and field.default_factory is MISSING
                   and values.get(name) is None]
        if missing:
            raise ValueError(f"DeepSeek config is missing required fields: {', '.join(missing)}")
        # Released configs may append MTP entries after the decoder backbone.
        n = values["num_hidden_layers"]
        mtp = values.get("num_nextn_predict_layers", 0)
        if type(mtp) is not int or mtp < 0:
            raise ValueError("num_nextn_predict_layers must be a nonnegative integer")
        if len(values["compress_ratios"]) != n:
            if len(values["compress_ratios"]) != n + mtp:
                raise ValueError("compress_ratios must match decoder layers plus any declared MTP layers")
            values["compress_ratios"] = values["compress_ratios"][:n]
        return cls(**{key: value for key, value in values.items() if key in known})

    def to_dict(self) -> dict:
        return asdict(self)
