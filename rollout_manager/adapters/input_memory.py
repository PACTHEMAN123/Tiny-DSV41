"""Metadata-only memory estimates for the reviewed native input generators."""
from __future__ import annotations

import ast
import hashlib


ALLOCATIONS = {
    "aten.rand.default", "aten.randint.low", "aten.add.Tensor", "aten.sub.Tensor",
    "aten.mul.Tensor", "aten.div.Tensor", "aten.triu.default", "aten.tril.default",
    "aten._softmax.default", "aten._to_copy.default",
}
VIEWS = {"aten.permute.default", "aten.view.default", "aten.expand.default"}


def generator_module(source):
    """Select unchanged AST nodes needed by get_inputs, excluding model creation."""
    tree = ast.parse(source)
    if any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(tree)):
        raise ValueError("reference may mutate generator globals")
    definitions = {}
    for node in tree.body:
        names = []
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names = [n.id for target in targets for n in ast.walk(target) if isinstance(n, ast.Name)]
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [n.asname or n.name.split(".")[0] for n in node.names]
        for name in names:
            definitions.setdefault(name, []).append(node)
    selected, pending = set(), ["get_inputs"]
    while pending:
        for node in definitions.get(pending.pop(), []):
            if id(node) in selected:
                continue
            selected.add(id(node))
            pending.extend(n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load))
    nodes = [n for n in tree.body if id(n) in selected]
    if not any(isinstance(n, ast.FunctionDef) and n.name == "get_inputs" for n in nodes):
        raise ValueError("missing native input generator")
    return ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))


def generator_context(source):
    context = {}
    exec(compile(generator_module(source), "model.py", "exec"), context)
    return context


def descriptor(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "bytes": tensor.numel() * tensor.element_size()}


def make_memory_plan(source):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode

    context = generator_context(source)
    operations = []
    allocation_bytes = 0

    class Plan(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            nonlocal allocation_bytes
            kwargs = kwargs or {}
            name = str(func)
            if name not in ALLOCATIONS and name not in VIEWS:
                raise ValueError("unreviewed generator operation: " + name)
            if str(kwargs.get("device", "meta")) not in ("cpu", "meta", "None"):
                raise ValueError("native generator requires device RNG: " + str(kwargs["device"]))
            value = func(*args, **kwargs)
            item = {"op": name, **descriptor(value)}
            if name not in VIEWS:
                if not value.is_contiguous():
                    raise ValueError("noncontiguous allocation requires a separate memory estimate")
                allocation_bytes = (allocation_bytes + 63) // 64 * 64 + item["bytes"]
            operations.append(item)
            return value

    with torch.device("meta"), Plan():
        inputs = context["get_inputs"]()
    if not isinstance(inputs, list):
        raise ValueError("expected native flat list of inputs")
    outputs = []
    for value in inputs:
        if isinstance(value, torch.Tensor):
            outputs.append({"kind": "tensor", **descriptor(value)})
        elif type(value) in (int, float, bool, str) or value is None:
            outputs.append({"kind": "scalar", "value": value})
        else:
            raise ValueError("unsupported native input container")
    allocation_bytes = (allocation_bytes+63)//64*64
    return {"format": "native-input-memory-v1", "dataset": "kernelbench",
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "generator_ast_sha256": hashlib.sha256(ast.dump(generator_module(source)).encode()).hexdigest(),
        "torch_version": torch.__version__, "operations": operations, "outputs": outputs,
        "allocation_bytes": allocation_bytes,
        "input_bytes": sum(x.get("bytes", 0) for x in outputs),
        "interpretation": "sum of aligned allocations in one native input call, not measured peak RSS"}
