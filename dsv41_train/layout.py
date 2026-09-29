"""Logical parameter identity and physical shard accounting."""

from dataclasses import dataclass
from collections import defaultdict

import torch


@dataclass(frozen=True)
class ParameterLayout:
    name: str
    logical_name: str
    global_shape: tuple[int, ...]
    offsets: tuple[int, ...]
    local_shape: tuple[int, ...]
    role: str
    dtype: str
    expert_id: int | None = None
    table_id: int | None = None


def local_tensor(tensor):
    return tensor.to_local() if hasattr(tensor, "to_local") else tensor


def local_slice(tensor):
    offsets = [0] * tensor.ndim
    if hasattr(tensor, "placements"):
        # FSDP2 shards along one mesh dimension; preserve its ceil-chunk convention.
        shape = list(tensor.shape)
        for axis, placement in enumerate(tensor.placements):
            if placement.is_shard():
                dim = placement.dim
                size = tensor.device_mesh.size(axis)
                rank = tensor.device_mesh.get_local_rank(axis)
                chunk = (shape[dim] + size - 1) // size
                offset = min(rank * chunk, shape[dim])
                offsets[dim] += offset
                shape[dim] = min(chunk, shape[dim] - offset)
    return tuple(offsets), tuple(local_tensor(tensor).shape)


def parameter_layouts(model):
    modules = dict(model.named_modules())
    decoder = getattr(model, "model", model)
    layers = getattr(decoder, "layers", ())
    layer_prefix = "model.layers." if decoder is not model else "layers."
    result = {}
    for name, parameter in model.named_parameters():
        logical = name
        expert_id = table_id = None
        role = "dense"
        if ".moe.routed." in name:
            prefix, index = name.rsplit(".", 1)
            routed = modules[prefix.rsplit(".", 1)[0]]
            expert_id = routed.expert_start + int(index)
            logical = prefix + "." + str(expert_id)
            role = "experts"
        if logical.startswith(layer_prefix):
            suffix = logical[len(layer_prefix):]
            index, tail = suffix.split(".", 1)
            logical = layer_prefix + str(layers[int(index)].layer_id) + "." + tail
        offsets, shape = local_slice(parameter)
        global_shape = tuple(parameter.shape)
        if "engram_tables." in name:
            table = modules[name.rsplit(".", 1)[0]]
            table_id = int(name.split("engram_tables.")[1].split(".")[0])
            offsets = (offsets[0] + table.row_start, *offsets[1:])
            global_shape = (table.global_num_embeddings, table.embedding_dim)
            role = "engram"
        result[name] = ParameterLayout(name, logical, global_shape, offsets, shape, role,
                                       str(parameter.dtype), expert_id, table_id)
    return result


def tensor_bytes(tensor):
    tensor = local_tensor(tensor)
    if tensor.is_sparse:
        return tensor._values().numel() * tensor.element_size() + tensor._indices().numel() * 8
    return tensor.numel() * tensor.element_size()


def parameter_memory(model, optimizer=None):
    layouts = parameter_layouts(model)
    result = defaultdict(lambda: {"weights": 0, "gradients": 0, "optimizer": 0})
    for name, parameter in model.named_parameters():
        bucket = result[layouts[name].role]
        bucket["weights"] += tensor_bytes(parameter)
        if parameter.grad is not None:
            bucket["gradients"] += tensor_bytes(parameter.grad)
        if optimizer is not None:
            bucket["optimizer"] += sum(tensor_bytes(v) for v in optimizer.state.get(parameter, {}).values()
                                       if isinstance(v, torch.Tensor))
    return dict(result)
