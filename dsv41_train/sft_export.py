"""Portable LoRA plus sparse Engram initialization, bound to one SFT snapshot."""

from dataclasses import replace
import json
import os
from pathlib import Path

import torch

from .lora import _adapter_modules, _layer_ids, adapter_parameters, save_adapter
from .models.dsv4 import DeepSeekV41Config
from .rl.checkpoint import atomic_save, file_sha256, _sync_directory
from .rl.engram_state import baseline_identity, validate_delta
from .sft_checkpoint import _collect, _local, _position


def _rank_payload(checkpoint, manifest, rank):
    record = manifest["ranks"][rank]
    path = checkpoint / f"rank-{rank}.pt"
    if (record["file"] != path.name or path.stat().st_size != record["bytes"]
            or file_sha256(path) != record["sha256"]):
        raise ValueError(f"incomplete or corrupt SFT owner {rank}")
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def _global_rows(checkpoint, manifest, base):
    """Read owner snapshots, never the base embedding table; allocate touched rows only."""
    shapes = {name: tuple(shape) for name, shape in base["tables"].items()}
    dtype = getattr(torch, base["dtype"].removeprefix("torch."))
    parts = {name: [] for name in shapes}
    names = {f"model.engram_tables.{name}.weight": name for name in shapes}
    for rank, owner in enumerate(manifest["ranks"]):
        snapshot = _rank_payload(checkpoint, manifest, rank)["engram_rows"]
        schema = owner["schema"]["engram_rows"]
        if (set(snapshot) != {"format", "schema", "tables"}
                or type(snapshot["format"]) is not int or snapshot["format"] != 1
                or snapshot["schema"] != schema or schema.keys() != names.keys()
                or snapshot["tables"].keys() != names.keys()):
            raise ValueError("SFT Engram owner schema or table set differs")
        for full, name in names.items():
            layout, rows = schema[full], snapshot["tables"][full]
            start, stop = layout["row_start"], layout["row_stop"]
            total, width = shapes[name]
            if (type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= total
                    or layout["global_shape"] != [total, width] or layout["local_shape"] != [stop - start, width]
                    or layout["dtype"] != str(dtype)):
                raise ValueError("SFT Engram owner bounds, dtype or shape differ")
            validate_delta({"format": 1, "base": base, "tables": {name: rows}},
                           base, {name: (stop - start, width)}, dtype)
            parts[name].append((start, stop, rows))
    tables = {}
    for name, owners in parts.items():
        owners.sort(key=lambda item: (item[0], item[1]))
        end = 0
        for start, stop, _ in owners:
            if start != end:
                raise ValueError("SFT Engram owners overlap or leave missing rows")
            end = stop
        if end != shapes[name][0]:
            raise ValueError("SFT Engram owners do not cover the original table")
        tables[name] = {"indices": torch.cat([rows["indices"] + start for start, _, rows in owners]),
                        "values": torch.cat([rows["values"] for _, _, rows in owners])}
    result = {"format": 1, "base": base, "tables": tables}
    validate_delta(result, base, shapes, dtype)
    return result


def export_sft_bundle(model, checkpoint, model_path, output):
    """All ranks participate; publish manifest last after every validation succeeds.

    The live model must still match this final SFT checkpoint. Only complete
    same-layer LoRA with dropout=0 and offloaded sparse Engram is supported.
    Rank-zero validation/IO failures are broadcast before subsequent collectives.
    """
    rank, world = _position()
    checkpoint, output = Path(checkpoint), Path(output)
    error = None
    try:
        manager = getattr(model, "_sft_engram_checkpoint", None)
        modules = _adapter_modules(model)
        config = DeepSeekV41Config.from_json(Path(model_path) / "config.json")
        if (manager is None or not manager.tables or _layer_ids(model) != list(range(config.num_hidden_layers))
                or any(module.config.dropout != 0 for module in modules.values())):
            raise ValueError("bundle requires complete same-layer LoRA, dropout=0 and sparse CPU Engram")
        expected = {id(p) for p in adapter_parameters(model)} | {id(table.weight) for table in manager.tables.values()}
        if {id(p) for p in model.parameters() if p.requires_grad} != expected:
            raise ValueError("bundle cannot represent full SFT or other trainable base weights")
        dtypes = {table.weight.dtype for table in manager.tables.values()}
        if len(dtypes) != 1:
            raise ValueError("Engram bundle tables must share one dtype")
        base = baseline_identity(model_path, config, next(iter(dtypes)), False, 0)
        manifest = json.loads((checkpoint / "manifest.json").read_text())
        source = {key: value for key, value in manifest["identity"].items()
                  if key not in {"training_config", "data_sha256", "eval_data_sha256"}}
        training = manifest["identity"].get("training_config", {})
        if (manifest["version"] != 1 or manifest["world_size"] != world or len(manifest["ranks"]) != world
                or type(manifest["step"]) is not int or manifest["step"] < 1 or source != base["source"]
                or training.get("mode") != "lora" or training.get("family") != "dsv4"):
            raise ValueError("SFT bundle checkpoint identity, mode, step or topology differs")
        payload = _rank_payload(checkpoint, manifest, rank)
        manager.validate(payload["engram_rows"])
        parameters = {name: p for name, p in model.named_parameters() if p.requires_grad and name not in manager.parameter_names}
        if parameters.keys() != payload["parameters"].keys():
            raise ValueError("SFT checkpoint adapter names differ")
        for name, parameter in parameters.items():
            if not torch.equal(_local(parameter).detach().cpu(), payload["parameters"][name]):
                raise ValueError("live adapter differs from its paired SFT checkpoint")
        for name, rows in manager.state_dict()["tables"].items():
            if any(not torch.equal(rows[key], payload["engram_rows"]["tables"][name][key]) for key in rows):
                raise ValueError("live Engram rows differ from their paired SFT checkpoint")
        if rank == 0:
            delta = _global_rows(checkpoint, manifest, base)
            output.mkdir(parents=True, exist_ok=False)
    except Exception as exc:
        error = f"rank {rank}: {exc}"
    _collect({"error": error})
    try:
        # Existing live FSDP collectives; no shard reconstruction.
        save_adapter(model, output / "adapter.pt", base_identity=base["source"])
    except Exception as exc:
        error = f"rank {rank}: {exc}"
    _collect({"error": error})
    try:
        if rank == 0:
            with (output / "adapter.pt").open("rb") as handle:
                os.fsync(handle.fileno())
            atomic_save(delta, output / "engram_delta.pt")
            bundle = {"format": 1, "kind": "lora-engram-sft", "step": manifest["step"], "base": base,
                      "files": {name: file_sha256(output / name) for name in ("adapter.pt", "engram_delta.pt")}}
            temporary = output / ".manifest.json.tmp"
            with temporary.open("x") as handle:
                json.dump(bundle, handle, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(output / "manifest.json")
            _sync_directory(output)
    except Exception as exc:
        error = f"rank {rank}: {exc}"
    _collect({"error": error})
    return output


def resolve_initial_bundle(config):
    """A directory in initial_adapter means a strict pair; file paths keep their existing meaning."""
    if config.initial_adapter is None or not Path(config.initial_adapter).is_dir():
        return config
    root = Path(config.initial_adapter).absolute()
    manifest = json.loads((root / "manifest.json").read_text())
    if (set(manifest) != {"format", "kind", "step", "base", "files"}
            or type(manifest["format"]) is not int or manifest["format"] != 1
            or manifest["kind"] != "lora-engram-sft" or type(manifest["step"]) is not int or manifest["step"] < 1
            or manifest["files"].keys() != {"adapter.pt", "engram_delta.pt"}
            or config.initial_engram_delta is not None or config.trainable_scope != "lora"):
        raise ValueError("invalid SFT bundle manifest, scope or conflicting explicit Engram delta")
    for name, digest in manifest["files"].items():
        if (root / name).is_symlink() or file_sha256(root / name) != digest:
            raise ValueError(f"SFT bundle file checksum differs: {name}")
    delta = torch.load(root / "engram_delta.pt", weights_only=True, map_location="cpu", mmap=True)
    base = manifest["base"]
    validate_delta(delta, base, {name: tuple(shape) for name, shape in base["tables"].items()},
                   getattr(torch, base["dtype"].removeprefix("torch.")))
    resolved = replace(config, initial_adapter=str(root / "adapter.pt"),
                       initial_engram_delta=str(root / "engram_delta.pt"))
    resolved.validate()
    return resolved
