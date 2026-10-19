"""Same-topology SFT snapshots; rank-owned shards never share a checkpoint key.

Load the original base model before restoring. Frozen parameters and frozen RL
Engram tables are not copied. This format deliberately does not reshard states.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from .rl.checkpoint import atomic_save, file_sha256, _sync_directory


def _position():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _collect(value):
    _, size = _position()
    values = [None] * size
    if dist.is_initialized():
        dist.all_gather_object(values, value)
    else:
        values[0] = value
    errors = [item["error"] for item in values if item.get("error")]
    if errors:
        raise ValueError("SFT checkpoint: " + "; ".join(errors))
    return values


def _json(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _layout(tensor):
    result = {"shape": list(tensor.shape), "local_shape": list(_local(tensor).shape),
              "dtype": str(tensor.dtype), "device": str(_local(tensor).device)}
    if isinstance(tensor, DTensor):
        result.update(mesh=tensor.device_mesh.mesh.tolist(),
                      mesh_names=list(tensor.device_mesh.mesh_dim_names or ()),
                      placements=[str(p) for p in tensor.placements], stride=list(tensor.stride()))
    return result


def _tensors(model):
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    buffers = {}
    for prefix, module in model.named_modules():
        for name, value in module.named_buffers(recurse=False):
            full = f"{prefix}.{name}" if prefix else name
            if name not in module._non_persistent_buffers_set and ".engram_tables." not in f".{full}":
                buffers[full] = value
    return parameters, buffers


def _ownership(model):
    fields = ("layer_id", "expert_start", "num_experts", "global_num_experts",
              "row_start", "row_stop", "global_num_embeddings", "embedding_dim")
    return {name: {field: int(getattr(module, field)) for field in fields if hasattr(module, field)}
            for name, module in model.named_modules() if any(hasattr(module, field) for field in fields)}


def _optimizer_spec(optimizer, parameters):
    if type(optimizer) not in (torch.optim.SGD, torch.optim.AdamW):
        raise ValueError("only SGD and AdamW optimizer states are supported")
    names = {id(p): name for name, p in parameters.items()}
    groups = []
    for group in optimizer.param_groups:
        if any(id(p) not in names for p in group["params"]):
            raise ValueError("optimizer must contain only trainable model parameters")
        master = group.get("fp32_master", False)
        if type(master) is not bool or (master and (
                type(optimizer) is not torch.optim.SGD
                or any(group.get(key, 0) for key in ("momentum", "dampening", "weight_decay", "nesterov",
                                                     "maximize", "differentiable", "foreach", "fused"))
                or any(not isinstance(p, DTensor) or p.dtype != torch.bfloat16 for p in group["params"]))):
            raise ValueError("FP32 master checkpoints require plain SGD with BF16 DTensor parameters")
        groups.append({**{key: _json(value) for key, value in group.items() if key != "params"},
                       "params": [names[id(p)] for p in group["params"]]})
    flat = [name for group in groups for name in group["params"]]
    if len(flat) != len(set(flat)) or set(flat) != set(parameters):
        raise ValueError("optimizer must own every trainable parameter exactly once")
    if any(id(p) not in names for p in optimizer.state):
        raise ValueError("optimizer has state for a parameter outside the model")
    return {"type": type(optimizer).__name__, "groups": groups}


def _schema(model, optimizer):
    parameters, buffers = _tensors(model)
    devices = sorted({t.device.index for t in map(_local, (*model.parameters(), *model.buffers()))
                      if t.device.type == "cuda"})
    schema = {"parameters": {name: _layout(p) for name, p in parameters.items()},
            "buffers": {name: _layout(b) for name, b in buffers.items()},
            "ownership": _ownership(model), "optimizer": _optimizer_spec(optimizer, parameters),
            "cuda_devices": devices}
    manager = getattr(model, "_sft_engram_checkpoint", None)
    if manager is not None:
        if not manager.parameter_names.issubset(parameters) or manager.schema.keys() != manager.parameter_names:
            raise ValueError("Engram checkpoint manager does not match trainable parameters")
        schema["engram_rows"] = manager.schema
    return schema


def _drain(model):
    for module in model.modules():
        if hasattr(module, "sparse_gradients") and callable(getattr(module, "synchronize", None)):
            module.synchronize()


def _encode_state(optimizer, parameters):
    states = {}
    groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    for name, p in parameters.items():
        fields, group = optimizer.state.get(p, {}), groups[id(p)]
        _validate_fields(fields, optimizer, group, name)
        if group.get("fp32_master", False):
            _validate_master(fields["master_weight"], _local(p).detach(), name)
        state = {}
        for key, value in fields.items():
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"unsupported optimizer state {name}.{key}")
            state[key] = {"layout": _layout(value), "value": _local(value).detach()}
        states[name] = state
    return states


def _validate_tensor(value, layout, name):
    if (not isinstance(value, torch.Tensor) or isinstance(value, DTensor) or value.device.type != "cpu"
            or list(value.shape) != layout["local_shape"] or str(value.dtype) != layout["dtype"]):
        raise ValueError(f"invalid tensor payload: {name}")


def _validate_fields(fields, optimizer, group, name):
    master = group.get("fp32_master", False)
    expected = ({"master_weight"} if master else {"momentum_buffer"} if group["momentum"] else set()) if type(optimizer) is torch.optim.SGD else (
        {"step", "exp_avg", "exp_avg_sq"} | ({"max_exp_avg_sq"} if group["amsgrad"] else set()))
    if (fields or master) and fields.keys() != expected:
        raise ValueError(f"invalid optimizer fields: {name}")


def _validate_master(master, weight, name):
    if (not isinstance(master, torch.Tensor) or isinstance(master, DTensor) or master.is_meta
            or master.layout != torch.strided or master.requires_grad or master.dtype != torch.float32
            or master.shape != weight.shape or master.device != weight.device or weight.dtype != torch.bfloat16):
        raise ValueError(f"invalid FP32 master tensor: {name}")
    # Bound comparison/cast temporaries even for noncontiguous or very wide shards.
    def check(value, reference):
        if value.numel() <= 2**20:
            if (not bool(torch.isfinite(value).all()) or not bool(torch.isfinite(reference).all())
                    or not torch.equal(value.to(torch.bfloat16), reference)):
                raise ValueError(f"FP32 master is non-finite or differs from BF16 parameter: {name}")
            return
        axis = next(i for i, size in enumerate(value.shape) if size > 1)
        width = max(1, 2**20 // (value.numel() // value.shape[axis]))
        for start in range(0, value.shape[axis], width):
            count = min(width, value.shape[axis] - start)
            check(value.narrow(axis, start, count), reference.narrow(axis, start, count))
    check(master, weight)


def _decode_state(encoded, optimizer, parameters, *, parameter_values, materialize):
    if encoded.keys() != parameters.keys():
        raise ValueError("optimizer parameter names differ")
    groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    result = {}
    for name, parameter in parameters.items():
        fields = encoded[name]
        group = groups[id(parameter)]
        _validate_fields(fields, optimizer, group, name)
        state = {}
        for key, record in fields.items():
            value, layout = record["value"], record["layout"]
            _validate_tensor(value, layout, f"{name}.{key}")
            if key == "master_weight":
                expected = {**_layout(_local(parameter)), "dtype": "torch.float32"}
                if set(record) != {"value", "layout"} or layout != expected or name not in parameter_values:
                    raise ValueError(f"FP32 master layout differs: {name}")
                _validate_master(value, parameter_values[name], name)
                if materialize:
                    state[key] = value.to(_local(parameter).device, copy=True)
            elif key == "step":
                device = str(_local(parameter).device) if group.get("capturable") or group.get("fused") else "cpu"
                if value.ndim != 0 or value.dtype not in (torch.float32, torch.float64) or "mesh" in layout or layout["device"] != device:
                    raise ValueError(f"invalid optimizer step: {name}")
                if materialize:
                    state[key] = value.to(device)
            else:
                if layout != _layout(parameter):
                    raise ValueError(f"optimizer tensor layout differs: {name}.{key}")
                if materialize:
                    local = value.to(_local(parameter).device)
                    state[key] = (DTensor.from_local(local, parameter.device_mesh, parameter.placements,
                                                    run_check=False, shape=parameter.shape, stride=parameter.stride())
                                  if isinstance(parameter, DTensor) else local)
        if state:
            result[parameter] = state
    return result


def save_checkpoint(path, model, optimizer, *, step, identity, data_position):
    """Publish after all rank files finish. No concurrent model/optimizer writes.

    torch.save copies CUDA storages to CPU one at a time. Retaining detached
    views avoids an additional full-model/optimizer host snapshot; peak staging
    still includes the largest backing storage per concurrently saving rank.
    """
    rank, world = _position()
    path = Path(path)
    error = None
    try:
        if type(step) is not int or step < 0:
            raise ValueError("step must be a nonnegative integer")
        identity, data_position = _json(identity), _json(data_position)
        if not isinstance(identity, dict) or not identity:
            raise ValueError("identity must describe the original base and training configuration")
        _drain(model)
        schema = _schema(model, optimizer)
        parameters, buffers = _tensors(model)
        encoded_state = _encode_state(optimizer, parameters)
    except Exception as exc:
        error = f"rank {rank}: {exc}"
    _collect({"error": error})
    try:
        if rank == 0:
            path.mkdir(parents=True, exist_ok=False)
    except Exception as exc:
        error = str(exc)
    _collect({"error": error})
    try:
        manager = getattr(model, "_sft_engram_checkpoint", None)
        excluded = manager.parameter_names if manager is not None else ()
        payload = {"parameters": {name: _local(p).detach() for name, p in parameters.items() if name not in excluded},
                   "buffers": {name: _local(b).detach() for name, b in buffers.items()},
                   "optimizer": encoded_state, "data_position": data_position,
                   "cpu_rng": torch.get_rng_state(),
                   "cuda_rng": {device: torch.cuda.get_rng_state(device) for device in schema["cuda_devices"]}}
        if manager is not None:
            payload["engram_rows"] = manager.state_dict()
            manager.validate(payload["engram_rows"])
        filename = f"rank-{rank}.pt"
        atomic_save(payload, path / filename)
        record = {"file": filename, "sha256": file_sha256(path / filename),
                  "bytes": (path / filename).stat().st_size, "schema": schema}
    except Exception as exc:
        error = f"rank {rank}: {exc}"
    results = _collect({"error": error, "record": record if error is None else None,
                        "identity": identity, "step": step})
    try:
        if any(item["identity"] != identity or item["step"] != step for item in results):
            raise ValueError("ranks disagree on checkpoint identity or step")
        if rank == 0:
            records = [item["record"] for item in results]
            for item in records:
                if (path / item["file"]).stat().st_size != item["bytes"]:
                    raise ValueError("rank files are not complete on the shared filesystem")
            manifest = {"version": 1, "world_size": world, "step": step, "identity": identity, "ranks": records}
            temporary = path / ".manifest.json.tmp"
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path / "manifest.json")
            _sync_directory(path)
    except Exception as exc:
        error = f"rank {rank}: {exc}"
    _collect({"error": error})
    return path


def load_checkpoint(path, model, optimizer, *, identity):
    """Validate every rank before mutating live state; return (step, data_position)."""
    rank, world = _position()
    path = Path(path)
    error = None
    try:
        _drain(model)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["version"] != 1 or manifest["world_size"] != world or len(manifest["ranks"]) != world:
            raise ValueError("checkpoint version or world size differs")
        if manifest["identity"] != _json(identity):
            raise ValueError("checkpoint identity differs")
        if type(manifest["step"]) is not int or manifest["step"] < 0:
            raise ValueError("invalid checkpoint step")
        for owner, item in enumerate(manifest["ranks"]):
            if item["file"] != f"rank-{owner}.pt" or (path / item["file"]).stat().st_size != item["bytes"]:
                raise ValueError("missing or incomplete rank file")
        record = manifest["ranks"][rank]
        schema = _schema(model, optimizer)
        if record["schema"] != schema:
            raise ValueError("parameter, optimizer, ownership or topology layout differs")
        if file_sha256(path / record["file"]) != record["sha256"]:
            raise ValueError("rank file checksum differs")
        payload = torch.load(path / record["file"], map_location="cpu", weights_only=True, mmap=True)
        parameters, buffers = _tensors(model)
        manager = getattr(model, "_sft_engram_checkpoint", None)
        if (manager is not None) != ("engram_rows" in payload):
            raise ValueError("Engram row snapshot presence differs")
        excluded = manager.parameter_names if manager is not None else ()
        saved_parameters = {name: p for name, p in parameters.items() if name not in excluded}
        for section, tensors in (("parameters", saved_parameters), ("buffers", buffers)):
            if payload[section].keys() != tensors.keys():
                raise ValueError(f"{section} names differ")
            for name, value in payload[section].items():
                _validate_tensor(value, schema[section][name], name)
        if manager is not None:
            manager.validate(payload["engram_rows"])
        _decode_state(payload["optimizer"], optimizer, parameters,
                      parameter_values=payload["parameters"], materialize=False)
        torch.Generator().set_state(payload["cpu_rng"])
        if payload["cuda_rng"].keys() != set(schema["cuda_devices"]):
            raise ValueError("CUDA RNG devices differ")
        for device, state in payload["cuda_rng"].items():
            if state.dtype != torch.uint8 or state.shape != torch.cuda.get_rng_state(device).shape:
                raise ValueError("invalid CUDA RNG state")
        data_position = _json(payload["data_position"])
    except Exception as exc:
        error = f"rank {rank}: {exc}"
    _collect({"error": error})
    with torch.no_grad():
        for section, tensors in (("parameters", saved_parameters), ("buffers", buffers)):
            for name, destination in tensors.items():
                _local(destination).copy_(payload[section][name])
        for parameter in parameters.values():
            parameter.grad = None
        optimizer.state.clear()
        optimizer.state.update(_decode_state(payload["optimizer"], optimizer, parameters,
                                           parameter_values=payload["parameters"], materialize=True))
        if manager is not None:
            manager.load_state_dict(payload["engram_rows"])
        for module in model.modules():
            if hasattr(module, "sparse_gradients") and callable(getattr(module, "clear_cache", None)):
                module.clear_cache()
        torch.set_rng_state(payload["cpu_rng"])
        for device, state in payload["cuda_rng"].items():
            torch.cuda.set_rng_state(state, device)
    return manifest["step"], data_position
