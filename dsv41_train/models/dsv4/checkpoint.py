"""Load the released DeepSeek V4.1 checkpoint with PyTorch and the stdlib."""

from __future__ import annotations

import json
import math
import mmap
import struct
from pathlib import Path

import torch

from ...dispatch import TokenDispatcher
from ...parallel import ContextParallel
from .config import DeepSeekV41Config
from .model import DeepSeekV41ForCausalLM


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


def _torch_dtype(name: str) -> torch.dtype:
    if name == "BF16":
        return torch.bfloat16
    if name == "F32":
        return torch.float32
    if name == "I8":
        return torch.int8
    attribute = {
        "F8_E4M3": "float8_e4m3fn",
        "F8_E8M0": "float8_e8m0fnu",
    }.get(name)
    dtype = None if attribute is None else getattr(torch, attribute, None)
    if dtype is None:
        raise RuntimeError(f"this PyTorch build does not support safetensors dtype {name}")
    return dtype


class _SafeTensorShard:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = path.open("rb")
        header_size_bytes = self._file.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError(f"invalid safetensors header: {path}")
        self.header_size = struct.unpack("<Q", header_size_bytes)[0]
        self.header = json.loads(self._file.read(self.header_size))
        self.data_start = 8 + self.header_size
        self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_COPY)

    def tensor(self, name: str, device: torch.device | str) -> torch.Tensor:
        metadata = self.header[name]
        dtype = _torch_dtype(metadata["dtype"])
        shape = tuple(metadata["shape"])
        start, stop = metadata["data_offsets"]
        numel = math.prod(shape)
        expected_bytes = numel * torch.empty((), dtype=dtype).element_size()
        if stop - start != expected_bytes:
            raise ValueError(
                f"invalid byte range for {name}: expected {expected_bytes}, got {stop - start}"
            )
        source = torch.frombuffer(
            self._mapping,
            dtype=dtype,
            count=numel,
            offset=self.data_start + start,
        ).reshape(shape)
        target = torch.device(device)
        return source.clone() if target.type == "cpu" else source.to(target)

    def close(self) -> None:
        self._mapping.close()
        self._file.close()


class ShardedSafeTensorReader:
    """Read selected tensors from an indexed safetensors checkpoint."""

    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        with (self.folder / "model.safetensors.index.json").open(encoding="utf-8") as file:
            self.weight_map: dict[str, str] = json.load(file)["weight_map"]
        self._shards: dict[str, _SafeTensorShard] = {}

    def tensor(self, name: str, device: torch.device | str) -> torch.Tensor:
        try:
            shard_name = self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"checkpoint tensor is missing: {name}") from error
        shard = self._shards.get(shard_name)
        if shard is None:
            shard = self._shards[shard_name] = _SafeTensorShard(self.folder / shard_name)
        return shard.tensor(name, device)

    def close(self) -> None:
        for shard in self._shards.values():
            shard.close()
        self._shards.clear()

    def __enter__(self) -> ShardedSafeTensorReader:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@torch.no_grad()
def dequantize_fp8_blocks(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a 32x32 E4M3 weight using its E8M0 block scales."""

    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("FP8 weight and scale must both be matrices")
    out_features, in_features = weight.shape
    out_blocks, in_blocks = scale.shape
    if out_features != out_blocks * 32 or in_features != in_blocks * 32:
        raise ValueError(
            f"FP8 shape mismatch: weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
        )
    blocks = weight.view(out_blocks, 32, in_blocks, 32).float()
    return (blocks * scale.float()[:, None, :, None]).reshape(weight.shape).to(dtype)


@torch.no_grad()
def dequantize_fp4_rows(
    packed: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Unpack E2M1 pairs and apply one E8M0 scale per 32 input values."""

    if packed.dtype != torch.int8 or packed.ndim != 2 or scale.ndim != 2:
        raise ValueError("FP4 weight must be an int8 matrix with a matrix of scales")
    out_features, packed_features = packed.shape
    if scale.shape[0] != out_features or packed_features * 2 != scale.shape[1] * 32:
        raise ValueError(
            f"FP4 shape mismatch: weight={tuple(packed.shape)}, scale={tuple(scale.shape)}"
        )
    values = torch.tensor(_FP4_VALUES, device=packed.device, dtype=torch.float32)
    raw = packed.view(torch.uint8)
    low = values[(raw & 0x0F).long()]
    high = values[((raw >> 4) & 0x0F).long()]
    unpacked = torch.stack((low, high), dim=-1).flatten(1)
    blocks = unpacked.view(out_features, scale.shape[1], 32)
    return (blocks * scale.float().unsqueeze(-1)).flatten(1).to(dtype)


def _prefix_config(folder: Path, num_layers: int) -> DeepSeekV41Config:
    full = DeepSeekV41Config.from_json(folder / "config.json")
    if num_layers != 1:
        raise NotImplementedError(
            "checkpoint-backed training currently supports the real layer-0 prefix only"
        )
    values = full.to_dict()
    values["num_hidden_layers"] = num_layers
    values["compress_ratios"] = full.compress_ratios[:num_layers]
    values["kv_source_layer_ids"] = [x for x in full.kv_source_layer_ids if x < num_layers]
    values["index_source_layer_ids"] = [
        x for x in full.index_source_layer_ids if x < num_layers
    ]
    values["candidate_source_layer_id"] = -1
    engram = [
        (layer_id, rows)
        for layer_id, rows in zip(full.engram_layer_ids, full.engram_num_embeddings)
        if layer_id < num_layers
    ]
    values["engram_layer_ids"] = [layer_id for layer_id, _ in engram]
    values["engram_num_embeddings"] = [rows for _, rows in engram]
    return DeepSeekV41Config(**values)


def load_dsv41_backbone_prefix(
    folder: str | Path,
    *,
    num_layers: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    context_parallel: ContextParallel | None = None,
    token_dispatcher: TokenDispatcher | None = None,
) -> DeepSeekV41ForCausalLM:
    """Load a trainable prefix directly from the released quantized checkpoint."""

    folder = Path(folder)
    config = _prefix_config(folder, num_layers)
    dispatcher = token_dispatcher or TokenDispatcher()
    with torch.device("meta"):
        model = DeepSeekV41ForCausalLM(config, context_parallel, dispatcher)

    layer = model.model.layers[0]
    routed = layer.moe.routed
    target_device = torch.device(device)
    state: dict[str, torch.Tensor] = {}

    with ShardedSafeTensorReader(folder) as checkpoint:

        def read(name: str) -> torch.Tensor:
            return checkpoint.tensor(name, target_device)

        def fp8(name: str) -> torch.Tensor:
            return dequantize_fp8_blocks(
                read(f"{name}.weight"), read(f"{name}.scale"), dtype=dtype
            )

        state.update(
            {
                "model.embedding.weight": read("embed.weight").to(dtype),
                "model.norm.weight": read("norm.weight").to(dtype),
                "lm_head.weight": read("head.weight").to(dtype),
                "model.layers.0.attention_hc.base": read("layers.0.hc_attn_base"),
                "model.layers.0.attention_hc.fn": read("layers.0.hc_attn_fn"),
                "model.layers.0.attention_hc.scale": read("layers.0.hc_attn_scale"),
                "model.layers.0.moe_hc.base": read("layers.0.hc_ffn_base"),
                "model.layers.0.moe_hc.fn": read("layers.0.hc_ffn_fn"),
                "model.layers.0.moe_hc.scale": read("layers.0.hc_ffn_scale"),
                "model.layers.0.attention.sinks.weight": read("layers.0.attn.attn_sink"),
                "model.layers.0.attention.q_a.weight": fp8("layers.0.attn.wq_a"),
                "model.layers.0.attention.q_norm.weight": read(
                    "layers.0.attn.q_norm.weight"
                ).to(dtype),
                "model.layers.0.attention.q_b.weight": fp8("layers.0.attn.wq_b"),
                "model.layers.0.attention.kv_proj.weight": fp8("layers.0.attn.wkv"),
                "model.layers.0.attention.kv_norm.weight": read(
                    "layers.0.attn.kv_norm.weight"
                ).to(dtype),
                "model.layers.0.attention.o_a.weight": fp8("layers.0.attn.wo_a").view(
                    config.o_groups, config.o_lora_rank, -1
                ),
                "model.layers.0.attention.o_b.weight": fp8("layers.0.attn.wo_b"),
                "model.layers.0.input_norm.weight": read(
                    "layers.0.attn_norm.weight"
                ).to(dtype),
                "model.layers.0.post_attention_norm.weight": read(
                    "layers.0.ffn_norm.weight"
                ).to(dtype),
                "model.layers.0.moe.router.weight": read(
                    "layers.0.ffn.gate.weight"
                ).to(dtype),
                "model.layers.0.moe.router.selection_bias": read(
                    "layers.0.ffn.gate.bias"
                ),
                "model.layers.0.moe.shared.gate.weight": fp8(
                    "layers.0.ffn.shared_experts.w1"
                ),
                "model.layers.0.moe.shared.up.weight": fp8(
                    "layers.0.ffn.shared_experts.w3"
                ),
                "model.layers.0.moe.shared.down.weight": fp8(
                    "layers.0.ffn.shared_experts.w2"
                ),
            }
        )

        gate_up = torch.empty(
            routed.num_experts,
            2 * config.moe_intermediate_size,
            config.hidden_size,
            device=target_device,
            dtype=dtype,
        )
        down = torch.empty(
            routed.num_experts,
            config.hidden_size,
            config.moe_intermediate_size,
            device=target_device,
            dtype=dtype,
        )
        for local_id, global_id in enumerate(
            range(routed.expert_start, routed.expert_start + routed.num_experts)
        ):
            prefix = f"layers.0.ffn.experts.{global_id}"
            gate = dequantize_fp4_rows(
                read(f"{prefix}.w1.weight"), read(f"{prefix}.w1.scale"), dtype=dtype
            )
            up = dequantize_fp4_rows(
                read(f"{prefix}.w3.weight"), read(f"{prefix}.w3.scale"), dtype=dtype
            )
            gate_up[local_id, : config.moe_intermediate_size].copy_(gate)
            gate_up[local_id, config.moe_intermediate_size :].copy_(up)
            down[local_id].copy_(
                dequantize_fp4_rows(
                    read(f"{prefix}.w2.weight"),
                    read(f"{prefix}.w2.scale"),
                    dtype=dtype,
                )
            )
        state["model.layers.0.moe.routed.gate_up"] = gate_up
        state["model.layers.0.moe.routed.down"] = down

    result = model.load_state_dict(state, strict=True, assign=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={result.missing_keys[:5]}, "
            f"unexpected={result.unexpected_keys[:5]}"
        )
    rotary = model.model.rotary
    rotary.main = (
        1.0
        / (
            config.rope_theta
            ** (torch.arange(0, config.qk_rope_head_dim, 2, device=target_device).float()
                / config.qk_rope_head_dim)
        )
    )
    rotary.compressed = (
        1.0
        / (
            config.compress_rope_theta
            ** (torch.arange(0, config.qk_rope_head_dim, 2, device=target_device).float()
                / config.qk_rope_head_dim)
        )
    )
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("checkpoint loading left meta parameters in the model")
    return model


__all__ = [
    "ShardedSafeTensorReader",
    "dequantize_fp4_rows",
    "dequantize_fp8_blocks",
    "load_dsv41_backbone_prefix",
]
