"""Read complete SFT rank snapshots into the native, unsharded model layout.

This is an initialization source, not optimizer resume or a new weight format.
Files must remain immutable for the lifetime of the source. Tensor storage is
memory-mapped; reconstruction copies bounded chunks without a global gather.
"""

import json
from pathlib import Path

import torch

from .models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
from .rl.checkpoint import file_sha256, run_identity


def _shape(value):
    if not isinstance(value, list) or any(type(n) is not int or n < 0 for n in value):
        raise ValueError("invalid SFT tensor shape")
    return tuple(value)


class SFTCheckpointSource:
    """Validate a full DSV4 snapshot before exposing any reconstruction methods.

    ``identity`` augments, rather than replaces, the original base identity.
    ``tensor_names`` excludes sparse Engram tables; ``iter_engram_rows(layer)``
    handles both full and sparse tables using global row IDs. The caller fills
    an original Engram baseline before applying sparse rows, then seals it.
    """

    def __init__(self, checkpoint, model_dir, *, chunk_bytes=4 * 2**20):
        if type(chunk_bytes) is not int or chunk_bytes < 1:
            raise ValueError("chunk_bytes must be a positive integer")
        self.chunk_bytes = chunk_bytes
        self.path = Path(checkpoint)
        manifest_path = self.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        world = manifest.get("world_size")
        if (manifest.get("version") != 1 or type(world) is not int or world < 1
                or type(manifest.get("step")) is not int or manifest["step"] < 1
                or not isinstance(manifest.get("ranks"), list) or len(manifest["ranks"]) != world):
            raise ValueError("invalid complete SFT checkpoint manifest")
        config = DeepSeekV41Config.from_json(Path(model_dir) / "config.json")
        saved_identity = manifest.get("identity", {})
        training = saved_identity.get("training_config", {})
        if (training.get("family") != "dsv4" or training.get("mode") != "full"
                or training.get("num_layers", config.num_hidden_layers) != config.num_hidden_layers
                or training.get("world_size", world) != world):
            raise ValueError("SFT source requires complete-layer full DSV4 training")
        self.base_identity = run_identity(model_dir, Path(model_dir) / "config.json", False)
        self.base_identity.pop("data_sha256")
        source_base = {key: value for key, value in saved_identity.items()
                       if key not in {"training_config", "data_sha256", "eval_data_sha256"}}
        if source_base != self.base_identity:
            raise ValueError("SFT source base identity differs")
        self.identity = {"manifest_sha256": file_sha256(manifest_path), "step": manifest["step"]}
        with torch.device("meta"):
            model = DeepSeekV41ForCausalLM(config)
        expected = dict(model.state_dict())
        parameter_names = set(dict(model.named_parameters()))
        buffer_names = expected.keys() - parameter_names
        self.shapes = {name: tuple(value.shape) for name, value in expected.items()}
        self.dtypes, self._pieces, self._sparse = {}, {}, {}
        self.engram_modes = {}
        self._payloads = []
        raw, meshes = {}, {}
        table_keys = {f"model.engram_tables.{layer}.weight": str(layer) for layer in config.engram_layer_ids}
        for rank, record in enumerate(manifest["ranks"]):
            path = self.path / f"rank-{rank}.pt"
            if (record.get("file") != path.name or path.stat().st_size != record.get("bytes")
                    or file_sha256(path) != record.get("sha256")):
                raise ValueError(f"SFT rank {rank} size or checksum differs")
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            self._payloads.append(payload)  # Keep mmap-backed storages alive, without copying.
            schema = record["schema"]
            if (schema["parameters"].keys() != parameter_names or schema["buffers"].keys() != buffer_names
                    or payload["buffers"].keys() != buffer_names):
                raise ValueError("SFT parameter or persistent buffer coverage differs")
            owners = schema["ownership"]
            for layer in range(config.num_hidden_layers):
                if owners.get(f"model.layers.{layer}", {}).get("layer_id") != layer:
                    raise ValueError("SFT layer ownership differs")
            sparse = schema.get("engram_rows", {})
            if (set(sparse) - table_keys.keys() or payload["parameters"].keys() != parameter_names - sparse.keys()
                    or bool(sparse) != ("engram_rows" in payload)):
                raise ValueError("SFT sparse Engram or parameter coverage differs")
            if sparse:
                rows = payload["engram_rows"]
                if (set(rows) != {"format", "schema", "tables"} or rows["format"] != 1
                        or rows["schema"] != sparse or rows["tables"].keys() != sparse.keys()):
                    raise ValueError("SFT sparse Engram schema differs")
            for name, target in expected.items():
                section = "parameters" if name in parameter_names else "buffers"
                layout = schema[section][name]
                owned_shape, offset = self._owner(name, owners, self.shapes[name])
                start, stop = self._slice(rank, world, layout, owned_shape, offset, meshes, name)
                dtype_name = layout["dtype"]
                dtype = {"torch.float32": torch.float32, "torch.bfloat16": torch.bfloat16}.get(dtype_name)
                if dtype is None or self.dtypes.get(name, dtype) != dtype:
                    raise ValueError(f"unsupported or inconsistent SFT dtype: {name}")
                self.dtypes[name] = dtype
                if name in table_keys:
                    layer = table_keys[name]
                    mode = "sparse" if name in sparse else "full"
                    if self.engram_modes.setdefault(layer, mode) != mode:
                        raise ValueError("SFT Engram storage mode differs across owners")
                if name in sparse:
                    if "mesh" in layout:
                        raise ValueError("sparse CPU Engram cannot carry a DTensor layout")
                    self._sparse_rows(name, rank, sparse[name], payload["engram_rows"]["tables"][name],
                                      owned_shape, offset, dtype)
                    continue
                value = payload[section][name]
                self._validate_tensor(value, _shape(layout["local_shape"]), dtype, name)
                raw.setdefault(name, []).append((start, stop, value))
        for group, seen in meshes.items():
            if seen != set(group[-2]):
                raise ValueError("SFT mesh members are missing or inconsistent")
        for name, pieces in raw.items():
            self._pieces[name] = self._cover(name, pieces)
        for name, records in self._sparse.items():
            cursor = 0
            for start, stop, _, _ in sorted(records, key=lambda item: item[0]):
                if start != cursor:
                    raise ValueError(f"SFT Engram owners overlap or leave gaps: {name}")
                cursor = stop
            if cursor != self.shapes[name][0]:
                raise ValueError(f"SFT Engram owners do not cover the table: {name}")
        self.tensor_names = frozenset(self._pieces)
        self.engram_shapes = {layer: self.shapes[name] for name, layer in table_keys.items()}
        self.engram_dtypes = {layer: self.dtypes[name] for name, layer in table_keys.items()}

    @staticmethod
    def _owner(name, owners, shape):
        parent = name.rsplit(".", 1)[0]
        owner = owners.get(parent, {})
        if ".moe.routed." in name:
            start, count = owner.get("expert_start"), owner.get("num_experts")
            if (type(start) is not int or type(count) is not int or count < 1
                    or owner.get("global_num_experts") != shape[0] or not 0 <= start < start + count <= shape[0]):
                raise ValueError(f"invalid SFT expert ownership: {name}")
            return (count, *shape[1:]), start
        if ".engram_tables." in name:
            start, stop = owner.get("row_start"), owner.get("row_stop")
            if (type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= shape[0]
                    or owner.get("global_num_embeddings") != shape[0] or owner.get("embedding_dim") != shape[1]):
                raise ValueError(f"invalid SFT Engram ownership: {name}")
            return (stop - start, *shape[1:]), start
        return shape, 0

    @staticmethod
    def _slice(rank, world, layout, shape, offset, meshes, name):
        if _shape(layout["shape"]) != shape:
            raise ValueError(f"SFT owner tensor shape differs: {name}")
        local_shape = _shape(layout["local_shape"])
        if "mesh" not in layout:
            if set(layout) != {"shape", "local_shape", "dtype", "device"} or local_shape != shape:
                raise ValueError(f"unsupported SFT tensor layout: {name}")
            return offset, offset + (shape[0] if shape else 1)
        mesh, placements = layout["mesh"], layout["placements"]
        if (set(layout) != {"shape", "local_shape", "dtype", "device", "mesh", "mesh_names", "placements", "stride"}
                or not isinstance(mesh, list) or not mesh
                or any(type(member) is not int or not 0 <= member < world for member in mesh)
                or len(set(mesh)) != len(mesh)
                or rank not in mesh or placements not in (["S(0)"], ["R"])
                or len(layout["mesh_names"]) != 1 or not shape
                or tuple(layout["stride"]) != torch.empty(shape, device="meta").stride()):
            raise ValueError(f"unsupported SFT mesh or placement: {name}")
        group = (name, offset, shape, tuple(mesh), tuple(placements))
        meshes.setdefault(group, set()).add(rank)
        start, stop = 0, shape[0]
        if placements == ["S(0)"]:
            width = (shape[0] + len(mesh) - 1) // len(mesh)
            start = min(shape[0], mesh.index(rank) * width)
            stop = min(shape[0], start + width)
        if local_shape != (stop - start, *shape[1:]):
            raise ValueError(f"SFT local shard shape differs: {name}")
        return offset + start, offset + stop

    def _chunks(self, tensor):
        if tensor.ndim == 0:
            yield 0, 1, tensor
            return
        row_bytes = max(1, tensor[0:1].numel() * tensor.element_size())
        width = max(1, self.chunk_bytes // row_bytes)
        for start in range(0, tensor.shape[0], width):
            stop = min(start + width, tensor.shape[0])
            yield start, stop, tensor[start:stop]

    def _validate_tensor(self, value, shape, dtype, name):
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.layout != torch.strided
                or tuple(value.shape) != shape or value.dtype != dtype):
            raise ValueError(f"invalid SFT tensor payload: {name}")
        if any(not bool(torch.isfinite(chunk).all()) for _, _, chunk in self._chunks(value)):
            raise ValueError(f"non-finite SFT tensor payload: {name}")

    def _equal(self, left, right):
        return all(torch.equal(chunk, right if right.ndim == 0 else right[start:stop])
                   for start, stop, chunk in self._chunks(left))

    def _cover(self, name, pieces):
        result, cursor = [], 0
        for start, stop, value in sorted(pieces, key=lambda item: (item[0], item[1])):
            if start == stop:
                continue
            if result and (start, stop) == result[-1][:2]:
                if ".engram_tables." in name:
                    raise ValueError(f"SFT Engram owners overlap: {name}")
                if not self._equal(value, result[-1][2]):
                    raise ValueError(f"SFT replicated tensors differ: {name}")
                continue
            if start != cursor:
                raise ValueError(f"SFT tensor owners overlap or leave gaps: {name}")
            result.append((start, stop, value))
            cursor = stop
        if cursor != (self.shapes[name][0] if self.shapes[name] else 1):
            raise ValueError(f"SFT tensor owners do not cover the tensor: {name}")
        return result

    def _sparse_rows(self, name, rank, schema, record, shape, offset, dtype):
        expected = {"global_shape": list(self.shapes[name]), "local_shape": list(shape),
                    "row_start": offset, "row_stop": offset + shape[0], "dtype": str(dtype)}
        if schema != expected or set(record) != {"indices", "values"}:
            raise ValueError(f"SFT sparse Engram layout differs: {name}")
        ids, values = record["indices"], record["values"]
        if (not isinstance(ids, torch.Tensor) or ids.device.type != "cpu" or ids.layout != torch.strided
                or ids.dtype != torch.int64 or ids.ndim != 1
                or (ids.numel() and (ids[0] < 0 or ids[-1] >= shape[0] or bool((ids[1:] <= ids[:-1]).any())))):
            raise ValueError(f"invalid SFT sparse Engram indices: rank {rank}, {name}")
        self._validate_tensor(values, (ids.numel(), shape[1]), dtype, name)
        self._sparse.setdefault(name, []).append((offset, offset + shape[0], ids, values))

    @torch.no_grad()
    def copy_tensor(self, name, destination):
        """Copy a validated full tensor; the caller owns its final device/dtype."""
        if name not in self._pieces:
            raise ValueError(f"SFT tensor is absent or uses sparse Engram rows: {name}")
        if (destination.is_meta or tuple(destination.shape) != self.shapes[name]
                or destination.dtype != self.dtypes[name] or destination.layout != torch.strided):
            raise ValueError(f"SFT destination shape or dtype differs: {name}")
        for offset, _, value in self._pieces[name]:
            for start, stop, chunk in self._chunks(value):
                target = destination if destination.ndim == 0 else destination[offset + start:offset + stop]
                target.copy_(chunk)

    def iter_engram_rows(self, layer):
        """Yield bounded (global row IDs, CPU values) views for one Engram table."""
        name = f"model.engram_tables.{layer}.weight"
        if str(layer) not in self.engram_modes:
            raise ValueError(f"unknown SFT Engram layer: {layer}")
        if name in self._sparse:
            for offset, _, ids, values in sorted(self._sparse[name], key=lambda item: item[0]):
                for start, stop, chunk in self._chunks(values):
                    yield ids[start:stop] + offset, chunk
        else:
            for offset, _, values in self._pieces[name]:
                for start, stop, chunk in self._chunks(values):
                    yield torch.arange(offset + start, offset + stop, dtype=torch.int64), chunk
