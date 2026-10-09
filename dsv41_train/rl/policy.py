"""DSV4 RL policy with frozen CPU tables and prefix-stable sparse indexing."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from dsv41_train.models.dsv4.attention import AttentionSinks, RotaryEmbedding
from dsv41_train.models.dsv4.config import DeepSeekV41Config
from dsv41_train.models.dsv4.checkpoint import _SafeTensorShard, load_dsv41_backbone_prefix
from dsv41_train.models.dsv4.mhc import HyperConnection
from dsv41_train.models.dsv4.model import DeepSeekV41ForCausalLM, NgramHash, build_compressed_token_map

from .indexer import PortableSparseIndexer


@dataclass
class DSV41Config(DeepSeekV41Config):
    checkpoint_format: str = "tiny_dsv41"
    checkpoint_dtype: str = "float32"

    @classmethod
    def from_model_dir(cls, model_dir):
        with (Path(model_dir) / "config.json").open(encoding="utf-8") as handle:
            root = json.load(handle)
        text = dict(root.get("text_config", root))
        for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
            if root.get(name) is not None:
                text[name] = root[name]
        if "engram_pad_token_id" in text:
            text["engram_pad_id"] = text["engram_pad_token_id"]
        quantized = bool(root.get("quantization_config") or text.get("quantization_config"))
        text["checkpoint_format"] = root.get(
            "checkpoint_format", text.get("checkpoint_format", "released" if quantized else "tiny_dsv41"))
        text["checkpoint_dtype"] = root.get("dtype", root.get("torch_dtype")) or text.get(
            "dtype", text.get("torch_dtype", "bfloat16" if quantized else "float32"))
        config = cls(**{key: value for key, value in text.items() if key in {f.name for f in fields(cls)}})
        if config.checkpoint_format not in {"tiny_dsv41", "released"}:
            raise ValueError("DeepSeek checkpoint_format must be tiny_dsv41 or released")
        if quantized and config.checkpoint_format != "released":
            raise ValueError("quantized DeepSeek checkpoints require the released loader")
        if config.checkpoint_dtype not in {"float32", "bfloat16"}:
            raise ValueError("DeepSeek checkpoint dtype must be float32 or bfloat16")
        if config.attention_dropout:
            raise ValueError("RL policies require attention_dropout=0 for behavior probability consistency")
        if any(not 0 <= token < config.vocab_size for token in (
            config.bos_token_id, config.eos_token_id, config.pad_token_id, config.engram_pad_id)):
            raise ValueError("special token IDs must be inside the model vocabulary")
        if len(set(config.engram_layer_ids)) != len(config.engram_layer_ids) or any(
            layer < 0 or layer >= config.num_hidden_layers for layer in config.engram_layer_ids
        ):
            raise ValueError("Engram layer IDs must be distinct zero-based decoder indices")
        return config

    def validate_random_init(self):
        table_elements = sum(self.engram_num_embeddings) * self.engram_head_dim
        if (self.hidden_size > 256 or self.num_hidden_layers > 8 or self.n_routed_experts > 16
                or self.vocab_size > 4096 or table_elements > 20_000_000):
            raise ValueError("random initialization is restricted to tiny test configurations")
        with torch.device("meta"):
            model = DeepSeekV41ForCausalLM(self)
        if sum(parameter.numel() for parameter in model.parameters()) > 50_000_000:
            raise ValueError("random initialization is restricted to tiny test configurations")


class _CPUHash(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.source = source.cpu()

    def _apply(self, fn, recurse=True):
        return self

    def forward(self, input_ids, token_mask, sequence_ids=None):
        return self.source(input_ids.cpu(), token_mask.cpu(),
                           None if sequence_ids is None else sequence_ids.cpu())


class _CPUEmbedding(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight.detach().cpu())
        self.register_buffer("placement", torch.empty(0, dtype=weight.dtype), persistent=False)

    def _apply(self, fn, recurse=True):
        weight = self._buffers.pop("weight")
        try:
            super()._apply(fn, recurse=recurse)
        finally:
            self.register_buffer("weight", weight)
        return self

    @torch.no_grad()
    def forward(self, indices):
        values = F.embedding(indices.to(device="cpu", dtype=torch.long), self.weight)
        return values.to(device=self.placement.device, dtype=self.placement.dtype)


class DSV41Policy(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend
        self.config = backend.config
        for name, table in list(backend.model.engram_tables.items()):
            backend.model.engram_tables[name] = _CPUEmbedding(table.weight)
        if backend.model.hash is not None:
            backend.model.hash = _CPUHash(backend.model.hash)
        for layer in backend.model.layers:
            if layer.attention.indexer is not None:
                with torch.device("meta"):
                    indexer = PortableSparseIndexer(self.config, layer.layer_id, backend.model.rotary)
                indexer.load_state_dict(layer.attention.indexer.state_dict(), assign=True)
                layer.attention.indexer = indexer.requires_grad_(False)
        self.assert_engram_invariants()

    @property
    def engram_device(self):
        return "cpu" if self.backend.model.engram_tables else "none"

    def assert_engram_invariants(self):
        for table in self.backend.model.engram_tables.values():
            if table.weight.device.type != "cpu" or table.weight.requires_grad or list(table.parameters()):
                raise RuntimeError("Engram tables must remain frozen CPU buffers")
        if self.backend.model.hash is not None and any(
            value.device.type != "cpu" for value in self.backend.model.hash.buffers()
        ):
            raise RuntimeError("Engram hash metadata must remain on CPU")

    def _apply(self, fn, recurse=True):
        # Preserve upstream FP32 precision islands before any lossy dtype cast.
        parameters = [(p, p.data, p.grad) for module in self.backend.modules()
                      if isinstance(module, (HyperConnection, AttentionSinks))
                      for p in module.parameters(recurse=False)]
        rotary = self.backend.model.rotary
        frequencies = (rotary.main, rotary.compressed)
        routing_biases = [(layer.moe.router, layer.moe.router.selection_bias)
                          for layer in self.backend.model.layers]
        super()._apply(fn, recurse=recurse)
        for parameter, value, gradient in parameters:
            parameter.data = value.to(device=parameter.device, dtype=torch.float32)
            if gradient is not None:
                parameter.grad = gradient.to(device=parameter.device, dtype=torch.float32)
        rotary.main = frequencies[0].to(device=rotary.main.device, dtype=torch.float32)
        rotary.compressed = frequencies[1].to(device=rotary.compressed.device, dtype=torch.float32)
        for router, value in routing_biases:
            router.selection_bias = value.to(device=router.selection_bias.device, dtype=torch.float32)
        self.assert_engram_invariants()
        return self

    def to_training_devices(self, devices, dtype):
        if len(devices) != 1:
            raise ValueError("Tiny-DSV41 RL currently requires exactly one device per policy process")
        return self.to(device=devices[0], dtype=dtype)

    def _partition(self, trainable, include_engram=False):
        state = {name: value for name, value in self.named_parameters() if value.requires_grad == trainable}
        if not trainable:
            state.update({name: value for name, value in self.named_buffers()
                          if not name.endswith(".placement") and (include_engram or ".engram_tables." not in name)})
        return state

    def trainable_state_dict(self):
        return {name: value.detach().cpu().clone() for name, value in self._partition(True).items()}

    def frozen_state_dict(self, *, include_engram=False):
        return {name: value.detach().cpu().clone()
                for name, value in self._partition(False, include_engram).items()}

    @torch.no_grad()
    def _load_partition(self, state, destinations):
        if state.keys() != destinations.keys():
            raise ValueError("DeepSeek policy state keys do not match")
        for name, value in destinations.items():
            if value.shape != state[name].shape:
                raise ValueError(f"DeepSeek policy tensor shape mismatch: {name}")
        for name, value in destinations.items():
            value.copy_(state[name].to(device=value.device, dtype=value.dtype))
        self.assert_engram_invariants()

    def load_trainable_state_dict(self, state):
        self._load_partition(state, self._partition(True))

    def load_frozen_state_dict(self, state, *, include_engram=False):
        self._load_partition(state, self._partition(False, include_engram))

    def forward(self, input_ids, attention_mask=None):
        # The RL objective consumes logits; the SFT wrapper's auxiliary loss is unused.
        hidden, _ = self.backend.model(input_ids, attention_mask)
        return self.backend.lm_head(hidden)


def _load_token_map(root, config):
    path = root / "engram_token_map.pt"
    if path.is_file():
        token_map = torch.load(path, weights_only=True, map_location="cpu")
        if not isinstance(token_map, torch.Tensor) or token_map.dtype != torch.int64:
            raise ValueError("engram_token_map.pt must contain an int64 tensor")
        return token_map
    if not (root / "tokenizer.json").is_file():
        raise FileNotFoundError("pretrained Engram requires tokenizer.json or engram_token_map.pt")
    token_map, size = build_compressed_token_map(root / "tokenizer.json")
    if size != config.engram_compressed_vocab_size:
        raise ValueError("tokenizer compressed vocabulary does not match the Engram config")
    return token_map


def _checkpoint_file(root, filename):
    path = (root / filename).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"checkpoint shard must be inside model directory: {filename}")
    return path


def _layout(root):
    for stem, extension in (("model", "safetensors"), ("pytorch_model", "bin")):
        index = root / f"{stem}.{extension}.index.json"
        single = root / f"{stem}.{extension}"
        safetensors = extension == "safetensors"
        if index.is_file():
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("checkpoint index must have a nonempty weight_map")
            return weight_map, safetensors
        if single.is_file():
            if safetensors:
                shard = _SafeTensorShard(single)
                try:
                    names = [name for name in shard.header if name != "__metadata__"]
                finally:
                    shard.close()
            else:
                names = torch.load(single, map_location="cpu", weights_only=True, mmap=True).keys()
            return {name: single.name for name in names}, safetensors
    raise FileNotFoundError(f"no model.safetensors or pytorch_model.bin checkpoint found in {root}")


def _load_native(root, config):
    weight_map, safetensors = _layout(root)
    # Materialize checkpoint tensors directly, avoiding a second random full model.
    with torch.device("meta"):
        model = DeepSeekV41ForCausalLM(config)
    expected = model.state_dict()
    if expected.keys() != weight_map.keys():
        raise ValueError(f"DeepSeek checkpoint keys: missing={sorted(expected.keys() - weight_map.keys())[:8]}, "
                         f"unexpected={sorted(weight_map.keys() - expected.keys())[:8]}")
    by_file = defaultdict(list)
    for name, filename in weight_map.items():
        _checkpoint_file(root, filename)
        by_file[filename].append(name)
    for filename, names in by_file.items():
        path = _checkpoint_file(root, filename)
        if safetensors:
            reader = _SafeTensorShard(path)
            try:
                state = {name: reader.tensor(name, "cpu") for name in names}
            finally:
                reader.close()
        else:
            state = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        shard = {}
        for name in names:
            tensor = state[name]
            if tensor.shape != expected[name].shape or tensor.dtype not in {torch.float32, torch.bfloat16}:
                raise ValueError(f"invalid native DeepSeek tensor shape or dtype: {name}")
            shard[name] = tensor
        model.load_state_dict(shard, strict=False, assign=True)
    rotary = RotaryEmbedding(config)
    model.model.rotary.main = rotary.main
    model.model.rotary.compressed = rotary.compressed
    if config.engram_layer_ids:
        model.model.hash = NgramHash(config, token_map=_load_token_map(root, config))
    if any(value.is_meta for value in model.parameters()):
        raise RuntimeError("DeepSeek checkpoint left uninitialized parameters")
    return model


def build_dsv41_policy(model_dir, config, *, random_init, devices, dtype):
    if len(devices) != 1:
        raise ValueError("Tiny-DSV41 RL currently requires exactly one device per policy process")
    if random_init:
        config.validate_random_init()
        backend = DeepSeekV41ForCausalLM(config)
        for table in backend.model.engram_tables.values():
            table.to(dtype=getattr(torch, config.checkpoint_dtype))
    elif config.checkpoint_format == "released":
        backend = load_dsv41_backbone_prefix(
            model_dir, num_layers=config.num_hidden_layers, device="cpu",
            dtype=getattr(torch, config.checkpoint_dtype))
        backend.config = config
    else:
        backend = _load_native(Path(model_dir), config)
    return DSV41Policy(backend).to_training_devices(devices, dtype)
