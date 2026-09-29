"""DCP adapters for explicitly owned, possibly uneven parameter shards."""

from dataclasses import asdict, replace
import json

import torch
import torch.distributed as dist
from torch.distributed.checkpoint import DefaultSavePlanner, DefaultLoadPlanner
from torch.distributed.checkpoint.metadata import ChunkStorageMetadata, MetadataIndex, TensorProperties
from torch.distributed.checkpoint.planner import TensorWriteData, WriteItem, WriteItemType

from .layout import parameter_layouts, local_tensor


class TensorShard:
    """DCP checkpointable protocol; storage is a view of the actual local tensor."""
    def __init__(self, tensor, shape, offsets):
        self.original = tensor
        self.tensor = local_tensor(tensor)
        self.shape = torch.Size(shape)
        self.offsets = torch.Size(offsets)

    def size(self):
        return self.shape

    def __create_chunk_list__(self):
        return [ChunkStorageMetadata(self.offsets, self.tensor.size())]

    def __create_write_items__(self, fqn, _object):
        return [WriteItem(index=MetadataIndex(fqn, self.offsets), type=WriteItemType.SHARD,
                          tensor_data=TensorWriteData(chunk=self.__create_chunk_list__()[0],
                              properties=TensorProperties.create_from_tensor(self.tensor), size=self.shape))]


class ShardSavePlanner(DefaultSavePlanner):
    def lookup_object(self, index):
        value = self.state_dict[index.fqn]
        return value.tensor if isinstance(value, TensorShard) else super().lookup_object(index)


class ShardLoadPlanner(DefaultLoadPlanner):
    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        from torch.distributed.checkpoint._nested_dict import flatten_state_dict
        # StateAdapter exposes existing local storage, never meta tensors. The
        # default initializer replaces custom checkpointable objects with None.
        self.original_state_dict = state_dict
        self.state_dict, self.mappings = flatten_state_dict(state_dict)
        self.metadata = metadata
        self.is_coordinator = is_coordinator

    def create_local_plan(self):
        from torch.distributed.checkpoint.default_planner import create_default_local_load_plan
        from torch.distributed.checkpoint.planner_helpers import create_read_items_for_chunk_list
        shards = {name: value for name, value in self.state_dict.items() if isinstance(value, TensorShard)}
        # The parent planner's legacy-key fallback re-flattens the original state
        # and would put TensorShard objects back into the ordinary tensor path.
        ordinary = {name: value for name, value in self.state_dict.items() if name not in shards}
        plan = create_default_local_load_plan(ordinary, self.metadata, not self.allow_partial_load)
        items = list(plan.items)
        for name, value in shards.items():
            metadata = self.metadata.state_dict_metadata[name]
            if metadata.size != value.shape:
                raise ValueError(f"checkpoint global shape mismatch for {name}")
            items.extend(create_read_items_for_chunk_list(name, metadata, value.__create_chunk_list__()))
        return replace(plan, items=items)

    def lookup_tensor(self, index):
        value = self.state_dict[index.fqn]
        return value.tensor if isinstance(value, TensorShard) else super().lookup_tensor(index)


class StateAdapter:
    VERSION = 2

    def __init__(self, model, *, cp=1, ep=1):
        self.model = model
        self.layouts = parameter_layouts(model)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.topology = {"world_size": dist.get_world_size() if dist.is_initialized() else 1,
                         "cp": cp, "ep": ep}
        self.names = {name: layout.logical_name for name, layout in self.layouts.items()}
        decoder = getattr(model, "model", model)
        prefix = "model.layers." if decoder is not model else "layers."
        for name, _ in model.named_buffers():
            logical = name
            if name.startswith(prefix):
                index, tail = name[len(prefix):].split(".", 1)
                logical = prefix + str(decoder.layers[int(index)].layer_id) + "." + tail
            self.names[name] = logical
        self.inverse = {logical: name for name, logical in self.names.items()}

    def signature(self):
        return {"version": self.VERSION, "topology": dict(self.topology),
                "shards": {f"rank-{self.rank}": {name: asdict(layout) for name, layout in self.layouts.items()}}}

    def validate(self, signature):
        if signature != self.signature():
            raise ValueError("checkpoint parameter layout/topology/dtype is incompatible")

    def optimizer_state(self, optimizer, schema=None):
        parameters = dict(self.model.named_parameters())
        names = {id(parameter): name for name, parameter in parameters.items()}
        groups = [{**group, "params": [names[id(parameter)] for parameter in group["params"]]}
                  for group in optimizer.param_groups]
        if schema is None:
            states = {names[id(parameter)]: dict(state) for parameter, state in optimizer.state.items() if state}
        else:
            states = {}
            for logical_name, fields in schema.items():
                name = self.inverse[logical_name]
                parameter = parameters[name]
                state = states[name] = {}
                for key, description in fields.items():
                    kind = description["kind"]
                    if kind == "value":
                        state[key] = description["value"]
                    elif kind == "parameter":
                        state[key] = torch.empty_like(parameter, dtype=getattr(torch, description["dtype"]))
                    elif kind == "scalar":
                        device = parameter.device if description["on_parameter_device"] else "cpu"
                        state[key] = torch.empty(description["shape"], device=device,
                                                 dtype=getattr(torch, description["dtype"]))
                    else:
                        raise ValueError(f"unsupported optimizer state kind: {kind}")
        return {"state": states, "param_groups": groups}

    def optimizer_schema(self, optimizer_state):
        parameters = dict(self.model.named_parameters())
        schema = {}
        for name, fields in optimizer_state["state"].items():
            schema[self.names[name]] = descriptions = {}
            for key, value in fields.items():
                if not isinstance(value, torch.Tensor):
                    description = {"kind": "value", "value": value}
                elif value.shape == parameters[name].shape:
                    description = {"kind": "parameter", "dtype": str(value.dtype).removeprefix("torch.")}
                elif value.numel() == 1:
                    description = {"kind": "scalar", "dtype": str(value.dtype).removeprefix("torch."),
                                   "shape": list(value.shape), "on_parameter_device": value.is_cuda}
                else:
                    raise ValueError(f"unsupported optimizer state shape for {name}.{key}")
                descriptions[key] = description
        return {f"rank-{self.rank}": json.dumps(schema, sort_keys=True)}

    def _wrap(self, name, tensor):
        layout = self.layouts.get(name)
        if (layout is not None and layout.role == "engram" and isinstance(tensor, torch.Tensor)
                and tensor.shape == self.model.get_parameter(name).shape):
            return TensorShard(tensor, layout.global_shape, layout.offsets)
        return tensor

    def encode(self, model_state, optimizer_state):
        model_state = {self.names.get(name, name): self._wrap(name, value) for name, value in model_state.items()}
        states = {self.names[name]: {key: self._wrap(name, value) for key, value in state.items()}
                  for name, state in optimizer_state.get("state", {}).items()}
        groups = [{**group, "params": [self.names[name] for name in group["params"]]}
                  for group in optimizer_state["param_groups"]]
        return model_state, {"state": states, "param_groups_by_rank": {f"rank-{self.rank}": groups}}

    def decode(self, model_state, optimizer_state):
        # DCP has already copied TensorShard storage in place. Reuse the existing
        # DTensor wrapper when a table also has a (possibly size-one) FSDP mesh.
        parameters = dict(self.model.named_parameters())
        def unwrap(name, value):
            return parameters[name].detach() if isinstance(value, TensorShard) else value
        model_state = {self.inverse.get(name, name): unwrap(self.inverse.get(name, name), value)
                       for name, value in model_state.items()}
        states = {}
        for name, state in optimizer_state["state"].items():
            local_name = self.inverse[name]
            states[local_name] = {key: value.original if isinstance(value, TensorShard) else value
                                  for key, value in state.items()}
        groups = [{**group, "params": [self.inverse[name] for name in group["params"]]}
                  for group in optimizer_state["param_groups_by_rank"][f"rank-{self.rank}"]]
        return model_state, {"state": states, "param_groups": groups}
