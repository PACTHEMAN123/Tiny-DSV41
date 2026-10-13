"""Write a quantized, randomly addressed DSV4 fixture with bounded CPU memory."""

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import time

import torch

from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.weights import layer_sources


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple
    dtype: str
    fill: str
    scale: float = 1.0

    @property
    def size(self):
        return math.prod(self.shape) * (4 if self.dtype == "F32" else 1)


def plain_spec(name, tensor, logical_name):
    if logical_name.endswith((".base", "selection_bias", "sinks.weight")):
        fill = "zero"
    elif logical_name.endswith(("norm.weight", "q_weight", "k_weight", ".scale")):
        fill = "one"
    else:
        fill = "normal"
    return TensorSpec(name, tuple(tensor.shape), "F32", fill)


def encoded_specs(name, shape, encoding, std):
    rows, cols = math.prod(shape[:-1]), shape[-1]
    if cols % 32:
        raise ValueError("quantized fixture widths must be divisible by 32")
    if encoding == "fp4":
        return [TensorSpec(name + ".weight", (rows, cols // 2), "I8", "fp4"),
                TensorSpec(name + ".scale", (rows, cols // 32), "F32", "constant", std / math.sqrt(8.5625))]
    scale_shape = ((rows + 31) // 32, (cols + 31) // 32) if encoding == "fp8_block" else (rows, cols // 32)
    return [TensorSpec(name + ".weight", (rows, cols), "F8_E4M3", "fp8"),
            TensorSpec(name + ".scale", scale_shape, "F32", "one")]


def checkpoint_specs(config):
    with torch.device("meta"):
        model = DeepSeekV41ForCausalLM(config)
    counts = {"engram": 0, "experts": 0, "other": 0}
    for name, parameter in model.named_parameters():
        category = "engram" if "engram_tables." in name else "experts" if ".moe.routed." in name else "other"
        counts[category] += parameter.numel()
    specs = [plain_spec("embed.weight", model.model.embedding.weight, "embedding.weight"),
             plain_spec("norm.weight", model.model.norm.weight, "norm.weight"),
             plain_spec("head.weight", model.lm_head.weight, "lm_head.weight")]
    for layer in model.model.layers:
        state = layer.state_dict()
        for logical_name, source in layer_sources(layer).items():
            tensor = state[logical_name]
            if source.encoding == "plain":
                specs.append(plain_spec(source.names[0], tensor, logical_name))
            else:
                shape = tuple(tensor.shape)
                if len(source.names) > 1:
                    shape = (shape[0] // len(source.names), *shape[1:])
                for name in source.names:
                    specs.extend(encoded_specs(name, shape, source.encoding, config.initializer_range))
        if layer.engram is not None:
            table = model.model.engram_tables[str(layer.layer_id)]
            specs.extend(encoded_specs(f"layers.{layer.layer_id}.engram.embed", tuple(table.weight.shape),
                                       "fp8_rows", config.initializer_range))
    model.close()
    return specs, counts


def blocks(spec, chunk_bytes, seed, std):
    element_size = 4 if spec.dtype == "F32" else 1
    capacity = max(1, chunk_bytes // element_size)
    digest = hashlib.sha256(spec.name.encode()).digest()
    generator = torch.Generator().manual_seed((seed + int.from_bytes(digest[:8], "little")) % (2**63))
    remaining = math.prod(spec.shape)
    while remaining:
        size = min(capacity, remaining)
        if spec.fill == "fp4":
            value = torch.randint(-128, 128, (size,), dtype=torch.int8, generator=generator)
        elif spec.fill == "fp8":
            # Signed finite E4M3 values near the model's initialization scale.
            value = torch.randint(0, 256, (size,), dtype=torch.uint8, generator=generator)
            value.bitwise_and_(0x8F).bitwise_or_(0x08)
        elif spec.fill == "normal":
            value = torch.randn(size, generator=generator).mul_(std)
        else:
            constant = {"zero": 0.0, "one": 1.0, "constant": spec.scale}[spec.fill]
            value = torch.full((size,), constant, dtype=torch.float32)
        yield value
        remaining -= size


def write_shard(path, specs, chunk_bytes, seed, std):
    header, offset = {}, 0
    for spec in specs:
        header[spec.name] = {"dtype": spec.dtype, "shape": list(spec.shape),
                             "data_offsets": [offset, offset + spec.size]}
        offset += spec.size
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        for spec in specs:
            for value in blocks(spec, chunk_bytes, seed, std):
                stream.write(memoryview(value.view(torch.uint8).numpy()))
    temporary.replace(path)


def create_checkpoint(folder, config, *, chunk_bytes=16 * 1024**2, shard_bytes=1024**3, seed=42):
    if chunk_bytes < 4 or shard_bytes < 1:
        raise ValueError("chunk and shard budgets must be positive")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "fixture.json"
    if any(folder.iterdir()):
        raise ValueError("fixture output directory must be empty")
    specs, counts = checkpoint_specs(config)
    manifest = {"version": 1, "seed": seed, "chunk_bytes": chunk_bytes, "shard_bytes": shard_bytes,
                "parameters": counts, "total_parameters": sum(counts.values()),
                "encoded_bytes": sum(spec.size for spec in specs), "completed": False,
                "distribution": "independent signed random FP4/FP8 codes; synthetic, not pretrained"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    weight_map, pending, size, written, shard_index = {}, [], 0, 0, 0
    started = time.perf_counter()

    def flush():
        nonlocal pending, size, written, shard_index
        name = f"model-{shard_index:05d}.safetensors"
        write_shard(folder / name, pending, chunk_bytes, seed, config.initializer_range)
        weight_map.update({spec.name: name for spec in pending})
        written += size
        print(json.dumps({"shard": name, "written_gib": written / 1024**3,
                          "total_gib": manifest["encoded_bytes"] / 1024**3,
                          "seconds": time.perf_counter() - started}), flush=True)
        pending, size, shard_index = [], 0, shard_index + 1

    for spec in specs:
        if pending and size + spec.size > shard_bytes:
            flush()
        pending.append(spec)
        size += spec.size
    if pending:
        flush()
    from tokenizers import Tokenizer, models
    compressed = config.engram_compressed_vocab_size
    if not compressed <= config.vocab_size <= 2 * compressed:
        raise ValueError("synthetic tokenizer supports one or two case variants per normalized token")
    vocabulary = {f"w{index}": index for index in range(compressed)}
    vocabulary.update({f"W{index}": compressed + index for index in range(config.vocab_size - compressed)})
    Tokenizer(models.WordLevel(vocabulary)).save(str(folder / "tokenizer.json"))
    (folder / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}) + "\n")
    manifest.update(completed=True, seconds=time.perf_counter() - started, shards=shard_index)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--shard-mib", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = create_checkpoint(args.output_dir, DeepSeekV41Config.from_json(args.config),
                                 chunk_bytes=args.chunk_mib * 1024**2,
                                 shard_bytes=args.shard_mib * 1024**2, seed=args.seed)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
