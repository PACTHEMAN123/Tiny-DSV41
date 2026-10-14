"""Optional native-generator reuse; no scheduling, trial fusion or device caching."""
from __future__ import annotations

import ast
import base64
import hashlib
import importlib.metadata
import json
import mmap
import os
import platform
import random
import sys
import time
import types
import weakref
from pathlib import Path

from ..input_cache import CacheClient, canonical
from .input_memory import generator_module
from .native_timing import HarnessInstrumentationError


ADAPTER_VERSION = "native-cpu-reuse-v4"


def runtime_identity(torch):
    import numpy as np
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "byteorder": sys.byteorder, "machine": platform.machine(),
            "libc": list(platform.libc_ver()), "torch_cuda_build": torch.version.cuda,
            "cpu_capability": torch.backends.cpu.get_cpu_capability(),
            "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
            "default_dtype": str(torch.get_default_dtype()),
            "default_device": str(torch.get_default_device()),
            "deterministic": torch.are_deterministic_algorithms_enabled()}


def _encode(tensor):
    return base64.b64encode(tensor.numpy().tobytes()).decode("ascii")


def rng_state(torch):
    import numpy as np
    value = np.random.get_state()
    return {"torch": _encode(torch.get_rng_state()), "python": random.getstate(),
            "numpy": [value[0], value[1].tolist(), value[2], value[3], value[4]],
            "cuda": [_encode(x) for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_initialized() else None}


def restore_rng(torch, value):
    import numpy as np
    def tuples(x):
        return tuple(tuples(i) for i in x) if isinstance(x, (list, tuple)) else x
    def tensor(x):
        return torch.frombuffer(bytearray(base64.b64decode(x)), dtype=torch.uint8)
    torch.set_rng_state(tensor(value["torch"]))
    random.setstate(tuples(value["python"]))
    state = value["numpy"]
    np.random.set_state((state[0], np.array(state[1], dtype=np.uint32), *state[2:]))
    if value["cuda"] is not None:
        torch.cuda.set_rng_state_all([tensor(x) for x in value["cuda"]])


def layout(torch, inputs):
    if not isinstance(inputs, list):
        raise ValueError("reuse supports native flat input lists only")
    storages, objects, tensors, outputs = {}, {}, [], []
    total = 0
    for index, value in enumerate(inputs):
        if isinstance(value, torch.Tensor):
            if (type(value) is not torch.Tensor or value.device.type != "cpu" or value.layout != torch.strided
                    or value.is_conj() or value.is_neg() or value.is_quantized or value.is_pinned()):
                raise ValueError("reuse requires plain strided CPU tensors")
            if value.requires_grad:
                raise ValueError("autograd input graphs are outside the qualified input contract")
            if id(value) in objects:
                outputs.append({"kind": "alias", "index": objects[id(value)]})
                continue
            objects[id(value)] = index
            storage = value.untyped_storage()
            if storage._cdata not in storages:
                total = (total+63)//64*64
                storages[storage._cdata] = len(tensors)
                tensors.append({"offset": total, "bytes": storage.nbytes(), "tensor": value})
                total += storage.nbytes()
            outputs.append({"kind": "tensor", "storage": storages[storage._cdata],
                "dtype": str(value.dtype).removeprefix("torch."), "shape": list(value.shape),
                "stride": list(value.stride()), "storage_offset": value.storage_offset(),
                "requires_grad": value.requires_grad})
        elif type(value) in (int, float, bool, str) or value is None:
            outputs.append({"kind": "scalar", "value": value})
        else:
            raise ValueError("unsupported native CPU input type")
    return {"storages": [{k: v for k, v in x.items() if k != "tensor"} for x in tensors],
            "outputs": outputs, "bytes": total}, tensors


def publish_storage(torch, fd, inputs):
    spec, storages = layout(torch, inputs)
    if spec["bytes"] > os.fstat(fd).st_size:
        raise ValueError("native inputs exceed the qualified storage bound")
    start = time.monotonic()
    with mmap.mmap(fd, 0, access=mmap.ACCESS_WRITE) as mapping:
        target = memoryview(mapping)
        try:
            for item in storages:
                raw = torch.empty(0, dtype=torch.uint8, device="cpu").set_(
                    item["tensor"].untyped_storage(), 0, (item["bytes"],), (1,))
                source = memoryview(raw.numpy())
                try:
                    target[item["offset"]:item["offset"]+item["bytes"]] = source
                finally:
                    source.release()
        finally:
            target.release()
    return spec, time.monotonic()-start


def map_storage(torch, fd, spec, released):
    # O_RDONLY + MAP_PRIVATE protects shared bytes while preserving ordinary
    # writable Tensor semantics for each consumer. Dirty pages are private copies.
    mapping = mmap.mmap(fd, 0, access=mmap.ACCESS_COPY)
    weakref.finalize(mapping, released)
    bases = [torch.frombuffer(mapping, dtype=torch.uint8, count=item["bytes"], offset=item["offset"])
             if item["bytes"] else torch.empty(0, dtype=torch.uint8, device="cpu")
             for item in spec["storages"]]
    outputs = []
    for item in spec["outputs"]:
        if item["kind"] == "scalar":
            outputs.append(item["value"])
        elif item["kind"] == "alias":
            outputs.append(outputs[item["index"]])
        else:
            dtype = getattr(torch, item["dtype"])
            value = bases[item["storage"]].view(dtype).as_strided(
                item["shape"], item["stride"], item["storage_offset"])
            value.requires_grad_(item["requires_grad"])
            outputs.append(value)
    return outputs


def generator_configuration(source, original):
    assigned = set()
    functions = set()
    for node in generator_module(source).body:
        if isinstance(node, ast.FunctionDef):
            functions.add(node.name)
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            assigned.update(n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name))
    values = {}
    pending, visited = [original], set()
    while pending:
        function = pending.pop()
        if id(function) in visited:
            continue
        visited.add(id(function))
        for name in function.__code__.co_names:
            value = function.__globals__.get(name)
            if name in functions and isinstance(value, types.FunctionType):
                pending.append(value)
            elif name in assigned:
                if type(value).__module__.startswith("transformers.") and hasattr(value, "to_dict"):
                    value = value.to_dict()
                values[name] = value
    # JSON primitives/configuration only; mutable global tensors and arbitrary
    # objects cannot qualify as immutable generator configuration.
    return json.loads(canonical(values))


def contract_identity(source, plan, configuration, torch):
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    if plan["source_sha256"] != source_hash or str(plan["torch_version"]) != str(torch.__version__):
        raise ValueError("input contract source/runtime mismatch")
    packages = {}
    for node in generator_module(source).body:
        names = ([n.name.split(".")[0] for n in node.names] if isinstance(node, ast.Import) else
                 [(node.module or "").split(".")[0]] if isinstance(node, ast.ImportFrom) else [])
        for name in names:
            if name and name not in sys.stdlib_module_names and name not in ("torch", "numpy"):
                packages[name] = importlib.metadata.version(name)
    runtime = dict(runtime_identity(torch), generator_packages=packages)
    return {"task_sha256": source_hash,
            "generator_sha256": hashlib.sha256(ast.dump(generator_module(source)).encode()).hexdigest(),
            "configuration": configuration,
            "input_contract": hashlib.sha256(canonical(plan).encode()).hexdigest(),
            "adapter_version": ADAPTER_VERSION,
            "runtime": runtime}


class InputReuse:
    def __init__(self, torch, socket_path, consumer, source, plan, *, namespace, seed, mode,
                 qualification=None, qualifying=False):
        if namespace not in ("development", "final-verification"):
            raise ValueError("explicit input seed namespace required")
        self.torch, self.seed, self.mode = torch, seed, mode
        self.client, self.consumer = CacheClient(socket_path), consumer
        self.source, self.qualification, self.qualifying = source, qualification, qualifying
        self.identity = contract_identity(source, plan, {"mode": mode, "native_correct_trials": 5,
                                                       "native_perf_trials": 10}, torch)
        if not qualifying and (qualification is None or qualification.get("passed") is not True or
                {k: v for k, v in qualification.get("identity", {}).items() if k != "configuration"} !=
                {k: v for k, v in self.identity.items() if k != "configuration"}):
            raise ValueError("generator has no equivalent reuse qualification for this runtime/contract")
        self.namespace = namespace
        self.reservation = int(plan.get("allocation_bytes", plan.get("arena_bytes", 0)))
        self.events = []
        self.error = None

    def get_inputs(self, original, trial_index):
        started = time.monotonic()
        fd = None
        reply = None
        try:
            self.identity["configuration"]["generator_globals"] = generator_configuration(self.source, original)
            if not self.qualifying and self.qualification["identity"] != self.identity:
                raise ValueError("resolved generator configuration changed after qualification")
            before = rng_state(self.torch)
            identity = dict(self.identity, namespace=self.namespace, seed=self.seed,
                            trial_index=trial_index, rng_before=hashlib.sha256(canonical(before).encode()).hexdigest())
            reply, fd = self.client.call("acquire", consumer=self.consumer, identity=identity, size=self.reservation)
            event = {"trial_index": trial_index, "namespace": self.namespace, "status": reply["status"],
                     "key": reply.get("key"), "tier": reply.get("tier"), "reason": reply.get("reason"),
                     "lookup_wait_seconds": time.monotonic()-started, "handoff_copy_bytes": 0,
                     "pinned_bytes": 0, "generation_bytes_saved": 0, "estimated_generation_seconds_saved": 0}
            if reply["status"] == "hit":
                metadata = reply["metadata"]
                if canonical(metadata["rng_before"]) != canonical(before):
                    raise ValueError("cache RNG precondition mismatch")
                lease = reply["lease"]
                def released():
                    try:
                        self.client.control("release", consumer=self.consumer, lease=lease)
                    except (OSError, RuntimeError):
                        # The parent retains ownership until child-group cleanup,
                        # including when the cache connection has failed.
                        pass
                start = time.monotonic()
                values = map_storage(self.torch, fd, metadata["layout"], released)
                event["mapping_seconds"] = time.monotonic()-start
                restore_rng(self.torch, metadata["rng_after"])
                event.update(generation_bytes_saved=reply["bytes"],
                             estimated_generation_seconds_saved=metadata["generation_seconds"],
                             consumer_cow_reserved_bytes=reply["bytes"])
            else:
                start = time.monotonic()
                values = original()
                event["generation_seconds"] = time.monotonic()-start
                if reply["status"] == "miss":
                    if generator_configuration(self.source, original) != self.identity["configuration"]["generator_globals"]:
                        raise ValueError("native generator mutated its input configuration")
                    after = rng_state(self.torch)
                    if before["cuda"] != after["cuda"]:
                        raise ValueError("qualified CPU generator changed CUDA RNG")
                    spec, seconds = publish_storage(self.torch, fd, values)
                    metadata = {"layout": spec, "rng_before": before, "rng_after": after,
                        "generation_seconds": event["generation_seconds"],
                        "publication_copy_bytes": sum(x["bytes"] for x in spec["storages"]),
                        "publication_copy_seconds": seconds}
                    os.close(fd)
                    fd = None
                    self.client.control("publish", consumer=self.consumer, key=reply["key"],
                                        metadata=metadata, size=spec["bytes"])
                    event.update(publication_copy_bytes=metadata["publication_copy_bytes"],
                                 publication_copy_seconds=seconds)
            event["wall_seconds"] = time.monotonic()-started
            self.events.append(event)
            return values
        except Exception as exc:
            self.error = str(exc)
            if reply and reply["status"] == "miss":
                self.client.control("abort", consumer=self.consumer, key=reply["key"])
            raise HarnessInstrumentationError("input reuse failed: " + str(exc)) from exc
        finally:
            if fd is not None:
                os.close(fd)

    def metrics(self):
        if self.error:
            raise HarnessInstrumentationError(self.error)
        return {"adapter": ADAPTER_VERSION, "namespace": self.namespace, "events": self.events,
                "hits": sum(e["status"] == "hit" for e in self.events),
                "misses": sum(e["status"] == "miss" for e in self.events),
                "bypasses": sum(e["status"] == "bypass" for e in self.events),
                "generation_bytes_saved": sum(e["generation_bytes_saved"] for e in self.events),
                "estimated_generation_seconds_saved": sum(e["estimated_generation_seconds_saved"] for e in self.events)}
