"""Load the released DeepSeek V4.1 checkpoint with PyTorch and the stdlib."""

from __future__ import annotations

import json
import math
import mmap
import struct
from collections.abc import Callable
from pathlib import Path

import torch

from ...dispatch import TokenDispatcher
from ...parallel import ContextParallel
from .config import DeepSeekV41Config
from .model import (
    DecoderLayer,
    DeepSeekV41ForCausalLM,
    NgramHash,
    RotaryEmbedding,
    build_compressed_token_map,
)

try:
    from torch.distributed.device_mesh import DeviceMesh
except ImportError:  # pragma: no cover - only needed for older PyTorch type checking
    DeviceMesh = object


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

    def tensor_rows(
        self,
        name: str,
        row_start: int,
        row_stop: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        metadata = self.header[name]
        shape = tuple(metadata["shape"])
        if not shape:
            raise ValueError(f"cannot slice rows from scalar tensor {name}")
        if not 0 <= row_start <= row_stop <= shape[0]:
            raise IndexError(
                f"invalid row range for {name}: [{row_start}, {row_stop}) of {shape[0]}"
            )
        dtype = _torch_dtype(metadata["dtype"])
        start, stop = metadata["data_offsets"]
        row_numel = math.prod(shape[1:])
        element_size = torch.empty((), dtype=dtype).element_size()
        expected_bytes = math.prod(shape) * element_size
        if stop - start != expected_bytes:
            raise ValueError(
                f"invalid byte range for {name}: expected {expected_bytes}, got {stop - start}"
            )
        source = torch.frombuffer(
            self._mapping,
            dtype=dtype,
            count=(row_stop - row_start) * row_numel,
            offset=self.data_start + start + row_start * row_numel * element_size,
        ).reshape(row_stop - row_start, *shape[1:])
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

    def tensor_rows(
        self,
        name: str,
        row_start: int,
        row_stop: int,
        device: torch.device | str,
    ) -> torch.Tensor:
        try:
            shard_name = self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"checkpoint tensor is missing: {name}") from error
        shard = self._shards.get(shard_name)
        if shard is None:
            shard = self._shards[shard_name] = _SafeTensorShard(self.folder / shard_name)
        return shard.tensor_rows(name, row_start, row_stop, device)

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
def dequantize_fp8_rows(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize E4M3 rows with one E8M0 scale per 32 columns."""

    if weight.ndim != 2 or scale.ndim != 2:
        raise ValueError("FP8 weight and scale must both be matrices")
    rows, features = weight.shape
    if scale.shape[0] != rows or features != scale.shape[1] * 32:
        raise ValueError(
            f"row-scaled FP8 shape mismatch: weight={tuple(weight.shape)}, "
            f"scale={tuple(scale.shape)}"
        )
    blocks = weight.view(rows, scale.shape[1], 32).float()
    return (blocks * scale.float().unsqueeze(-1)).flatten(1).to(dtype)


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
    if not 1 <= num_layers <= full.num_hidden_layers:
        raise NotImplementedError(
            f"a full checkpoint prefix supports one to {full.num_hidden_layers} real layers"
        )
    values = full.to_dict()
    values["num_hidden_layers"] = num_layers
    values["compress_ratios"] = full.compress_ratios[:num_layers]
    values["kv_source_layer_ids"] = [x for x in full.kv_source_layer_ids if x < num_layers]
    values["index_source_layer_ids"] = [
        x for x in full.index_source_layer_ids if x < num_layers
    ]
    candidate_source = full.candidate_source_layer_id
    values["candidate_source_layer_id"] = (
        candidate_source if 0 <= candidate_source < num_layers else -1
    )
    engram = [
        (layer_id, rows)
        for layer_id, rows in zip(full.engram_layer_ids, full.engram_num_embeddings)
        if layer_id < num_layers
    ]
    values["engram_layer_ids"] = [layer_id for layer_id, _ in engram]
    values["engram_num_embeddings"] = [rows for _, rows in engram]
    return DeepSeekV41Config(**values)


def _materialize_rotary(
    rotary: RotaryEmbedding,
    config: DeepSeekV41Config,
    device: torch.device,
) -> None:
    materialized = RotaryEmbedding(config).to(device)
    rotary.main = materialized.main
    rotary.compressed = materialized.compressed


def load_dsv41_backbone_window(
    folder: str | Path,
    *,
    start_layer: int = 0,
    num_layers: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    context_parallel: ContextParallel | None = None,
    token_dispatcher: TokenDispatcher | None = None,
    engram_mesh: DeviceMesh | None = None,
    sparse_engram_gradients: bool = False,
    offload_engram: bool = False,
    layer_loaded: Callable[[DecoderLayer], None] | None = None,
) -> DeepSeekV41ForCausalLM:
    """Load a trainable prefix or source-anchored layer window."""

    if offload_engram and not sparse_engram_gradients:
        raise ValueError("Engram CPU offload requires sparse Engram gradients")

    folder = Path(folder)
    if start_layer == 0:
        config = _prefix_config(folder, num_layers)
        layer_ids = list(range(num_layers))
    else:
        config = DeepSeekV41Config.from_json(folder / "config.json")
        stop_layer = start_layer + num_layers
        if num_layers < 1 or not 0 < start_layer < stop_layer <= config.num_hidden_layers:
            raise ValueError("the layer window must be inside the checkpoint backbone")
        if start_layer not in config.kv_source_layer_ids:
            raise ValueError(
                "a non-prefix layer window must start at a KV source layer"
            )
        layer_ids = list(range(start_layer, stop_layer))
    dispatcher = token_dispatcher or TokenDispatcher()
    with torch.device("meta"):
        model = DeepSeekV41ForCausalLM(
            config,
            context_parallel,
            dispatcher,
            engram_mesh,
            layer_ids,
            sparse_engram_gradients,
        )

    target_device = torch.device(device)
    rotary = model.model.rotary
    _materialize_rotary(rotary, config, target_device)
    with ShardedSafeTensorReader(folder) as checkpoint:

        def read(name: str) -> torch.Tensor:
            return checkpoint.tensor(name, target_device)

        def fp8(name: str) -> torch.Tensor:
            return dequantize_fp8_blocks(
                read(f"{name}.weight"), read(f"{name}.scale"), dtype=dtype
            )

        def assign(module: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
            result = module.load_state_dict(state, strict=True, assign=True)
            if result.missing_keys or result.unexpected_keys:
                raise RuntimeError(
                    f"checkpoint mismatch: missing={result.missing_keys[:5]}, "
                    f"unexpected={result.unexpected_keys[:5]}"
                )

        assign(model.model.embedding, {"weight": read("embed.weight").to(dtype)})
        assign(model.model.norm, {"weight": read("norm.weight").to(dtype)})
        assign(model.lm_head, {"weight": read("head.weight").to(dtype)})

        for layer in model.model.layers:
            layer_id = layer.layer_id
            source = f"layers.{layer_id}"
            state: dict[str, torch.Tensor] = {
                "attention_hc.base": read(f"{source}.hc_attn_base"),
                "attention_hc.fn": read(f"{source}.hc_attn_fn"),
                "attention_hc.scale": read(f"{source}.hc_attn_scale"),
                "moe_hc.base": read(f"{source}.hc_ffn_base"),
                "moe_hc.fn": read(f"{source}.hc_ffn_fn"),
                "moe_hc.scale": read(f"{source}.hc_ffn_scale"),
                "attention.sinks.weight": read(f"{source}.attn.attn_sink"),
                "attention.q_a.weight": fp8(f"{source}.attn.wq_a"),
                "attention.q_norm.weight": read(
                    f"{source}.attn.q_norm.weight"
                ).to(dtype),
                "attention.q_b.weight": fp8(f"{source}.attn.wq_b"),
                "attention.kv_proj.weight": fp8(f"{source}.attn.wkv"),
                "attention.kv_norm.weight": read(
                    f"{source}.attn.kv_norm.weight"
                ).to(dtype),
                "attention.o_a.weight": fp8(f"{source}.attn.wo_a").view(
                    config.o_groups, config.o_lora_rank, -1
                ),
                "attention.o_b.weight": fp8(f"{source}.attn.wo_b"),
                "input_norm.weight": read(f"{source}.attn_norm.weight").to(dtype),
                "post_attention_norm.weight": read(f"{source}.ffn_norm.weight").to(
                    dtype
                ),
                "moe.router.weight": read(f"{source}.ffn.gate.weight").to(dtype),
                "moe.router.selection_bias": read(f"{source}.ffn.gate.bias"),
                "moe.shared.gate.weight": fp8(f"{source}.ffn.shared_experts.w1"),
                "moe.shared.up.weight": fp8(f"{source}.ffn.shared_experts.w3"),
                "moe.shared.down.weight": fp8(f"{source}.ffn.shared_experts.w2"),
            }

            routed = layer.moe.routed
            for local_id, global_id in enumerate(
                range(routed.expert_start, routed.expert_start + routed.num_experts)
            ):
                prefix = f"{source}.ffn.experts.{global_id}"
                gate = dequantize_fp4_rows(
                    read(f"{prefix}.w1.weight"),
                    read(f"{prefix}.w1.scale"),
                    dtype=dtype,
                )
                up = dequantize_fp4_rows(
                    read(f"{prefix}.w3.weight"),
                    read(f"{prefix}.w3.scale"),
                    dtype=dtype,
                )
                state[f"moe.routed.gate_up.{local_id}"] = torch.cat(
                    (gate, up), dim=0
                )
                state[f"moe.routed.down.{local_id}"] = dequantize_fp4_rows(
                    read(f"{prefix}.w2.weight"),
                    read(f"{prefix}.w2.scale"),
                    dtype=dtype,
                )

            compressor = layer.attention.compressor
            if compressor is not None:
                state["attention.compressor.kv_proj.weight"] = read(
                    f"{source}.attn.compressor.wkv.weight"
                ).to(dtype)
                if compressor.gate_proj is not None:
                    state["attention.compressor.gate_proj.weight"] = read(
                        f"{source}.attn.compressor.wgate.weight"
                    ).to(dtype)
                state["attention.compressor.norm.weight"] = read(
                    f"{source}.attn.compressor.norm.weight"
                ).to(dtype)

            indexer = layer.attention.indexer
            if indexer is not None:
                state["attention.indexer.q_proj.weight"] = fp8(
                    f"{source}.attn.indexer.wq_b"
                )
                state["attention.indexer.weight_proj.weight"] = read(
                    f"{source}.attn.indexer.weights_proj.weight"
                ).to(dtype)
                if indexer.owns_keys:
                    state["attention.indexer.k_proj.weight"] = read(
                        f"{source}.attn.indexer.wk.weight"
                    ).to(dtype)
                    state["attention.indexer.k_norm.weight"] = read(
                        f"{source}.attn.indexer.k_norm.weight"
                    ).to(dtype)

            if layer.engram is not None:
                table = model.model.engram_tables[str(layer_id)]
                table_device = torch.device("cpu") if offload_engram else target_device
                table_weight = torch.empty(
                    table.weight.shape, device=table_device, dtype=dtype
                )
                chunk_rows = 131072
                for chunk_start in range(table.row_start, table.row_stop, chunk_rows):
                    chunk_stop = min(chunk_start + chunk_rows, table.row_stop)
                    weight = checkpoint.tensor_rows(
                        f"{source}.engram.embed.weight",
                        chunk_start,
                        chunk_stop,
                        target_device,
                    )
                    scale = checkpoint.tensor_rows(
                        f"{source}.engram.embed.scale",
                        chunk_start,
                        chunk_stop,
                        target_device,
                    )
                    table_weight[chunk_start - table.row_start : chunk_stop - table.row_start].copy_(
                        dequantize_fp8_rows(weight, scale, dtype=dtype).to(table_device)
                    )
                assign(table, {"weight": table_weight})
                state["engram.q_weight"] = read(f"{source}.engram.q_weight").to(dtype)
                state["engram.k_weight"] = read(f"{source}.engram.k_weight").to(dtype)
                state["engram.proj.weight"] = fp8(f"{source}.engram.wkv")

            assign(layer, state)
            if layer_loaded is not None:
                layer_loaded(layer)

    if model.model.hash is not None:
        token_map, compressed_vocab_size = build_compressed_token_map(
            folder / "tokenizer.json"
        )
        if compressed_vocab_size != config.engram_compressed_vocab_size:
            raise ValueError(
                "tokenizer-derived compressed vocabulary size "
                f"({compressed_vocab_size}) does not match the model config "
                f"({config.engram_compressed_vocab_size})"
            )
        model.model.hash = NgramHash(
            config,
            model.model.engram_layer_ids,
            token_map=token_map,
        ).to(target_device)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError("checkpoint loading left meta parameters in the model")
    return model


def load_dsv41_backbone_prefix(
    folder: str | Path,
    *,
    num_layers: int = 1,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    context_parallel: ContextParallel | None = None,
    token_dispatcher: TokenDispatcher | None = None,
    engram_mesh: DeviceMesh | None = None,
    sparse_engram_gradients: bool = False,
    offload_engram: bool = False,
    layer_loaded: Callable[[DecoderLayer], None] | None = None,
) -> DeepSeekV41ForCausalLM:
    """Load a trainable prefix directly from the released checkpoint."""

    return load_dsv41_backbone_window(
        folder,
        start_layer=0,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
        context_parallel=context_parallel,
        token_dispatcher=token_dispatcher,
        engram_mesh=engram_mesh,
        sparse_engram_gradients=sparse_engram_gradients,
        offload_engram=offload_engram,
        layer_loaded=layer_loaded,
    )


__all__ = [
    "ShardedSafeTensorReader",
    "dequantize_fp4_rows",
    "dequantize_fp8_blocks",
    "dequantize_fp8_rows",
    "load_dsv41_backbone_prefix",
    "load_dsv41_backbone_window",
]
