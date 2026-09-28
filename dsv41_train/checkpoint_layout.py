"""Checkpoint identity and names for tensors sharded outside FSDP."""

import hashlib
import json
from pathlib import Path

from torch import nn


def checkpoint_parameter_names(model: nn.Module) -> dict[str, str]:
    """Modules describe manual shards; FSDP continues to describe its own shards."""
    names = {}
    for prefix, module in model.named_modules():
        describe = getattr(module, "checkpoint_parameter_names", None)
        if describe is not None:
            namespace = prefix + "." if prefix else ""
            names.update({namespace + local: namespace + global_name
                          for local, global_name in describe().items()})
    parameters = dict(model.named_parameters())
    if not names.keys() <= parameters.keys():
        raise ValueError("checkpoint shard names must refer to model parameters")
    stored = [names.get(name, name) for name in parameters]
    if len(stored) != len(set(stored)):
        raise ValueError("checkpoint parameter names must be unique")
    return names


def parameter_layout(model: nn.Module) -> dict:
    layout = {}
    for name, parameter in model.named_parameters():
        spec = {"shape": list(parameter.shape), "dtype": str(parameter.dtype),
                "trainable": parameter.requires_grad}
        if hasattr(parameter, "device_mesh"):
            spec["mesh"] = parameter.device_mesh.mesh.tolist()
            spec["placements"] = [str(p) for p in parameter.placements]
        layout[name] = spec
    return layout


def checkpoint_identity(folder: str | Path) -> dict:
    """Identify immutable local base files without reading all tensor payloads.

    Metadata files are hashed; tensor files are checked by path, size and mtime.
    This detects accidental replacement, not adversarial content modification.
    """
    folder = Path(folder).resolve()
    index = json.loads((folder / "model.safetensors.index.json").read_text())
    metadata = {}
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json"):
        path = folder / name
        if path.is_file():
            metadata[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    files = {}
    for name in sorted(set(index["weight_map"].values())):
        stat = (folder / name).stat()
        files[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"kind": "local_safetensors", "path": str(folder),
            "metadata_sha256": metadata, "files": files}
