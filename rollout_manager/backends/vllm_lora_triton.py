"""Source-only inspection repair for the inspected CPython3.10.9 LoRA actor.

The kernel, its decorators and Triton's JIT remain unchanged. Only Triton's
private inspect binding can recover decorated blocks in two pinned source files.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
from pathlib import Path
import re
import sys
import textwrap

TARGET_MODULE = "vllm.models.minimax_m3.common.ops.index_topk"
TARGET_NAME = "_topk_index_merge_kernel"
KERNEL_SOURCES = {
    TARGET_MODULE: "index_topk",
    "vllm.models.minimax_m3.common.ops.sparse_attn": "sparse_attn",
}
SOURCE_SHA256 = {
    "index_topk": "1ca8557aed4260d2f4a742b9eda81d6f774e49d4b016ccc555dde40976c5a1a0",
    "sparse_attn": "5db5d5e692044e54bb57418cbb7dcf5c9cc94f789eaccf128eec6fa8df86d967",
    "triton_jit": "5fe46fbfac257a2b8645b7a2cc709232273d42dd5e1d55de9770b412a9476d77",
    "inspect": "ed67bf67157f52d3e942d4fcd381f427336def212b3989bab2cf7c89ed9b4c09",
}
_INSPECTION = None


class _SourceInspection:
    def __init__(self, original, sources):
        self.original = original
        self.sources = {module: (Path(path).resolve(strict=True), digest)
                        for module, (path, digest) in sources.items()}

    def __getattr__(self, name):
        return getattr(self.original, name)

    def getsourcelines(self, function):
        module, name = getattr(function, "__module__", None), getattr(function, "__name__", None)
        if module not in self.sources or getattr(function, "__qualname__", None) != name:
            return self.original.getsourcelines(function)
        path, digest = self.sources[module]
        if Path(function.__code__.co_filename).resolve(strict=True) != path:
            raise ValueError("guarded Triton kernel filename differs")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("guarded Triton kernel source digest differs")
        nodes = [node for node in ast.parse(data).body
                 if isinstance(node, ast.FunctionDef) and node.name == name and node.decorator_list]
        if not nodes:
            return self.original.getsourcelines(function)
        if len(nodes) != 1:
            raise ValueError("guarded Triton source requires one top-level kernel")
        node = nodes[0]
        first = min((item.lineno for item in node.decorator_list), default=node.lineno)
        if function.__code__.co_firstlineno != first:
            raise ValueError("guarded Triton kernel source position differs")
        lines, start = self.original.getsourcelines(function)
        if start != first:
            raise ValueError("guarded Triton inspection returned a different source position")
        if re.search(r"^def\s+" + re.escape(name) + r"\s*\(", textwrap.dedent("".join(lines)), re.MULTILINE):
            return lines, start
        recovered = data.decode("utf-8").splitlines(keepends=True)[first - 1:node.end_lineno]
        if len(lines) > node.lineno - first or lines != recovered[:len(lines)]:
            raise ValueError("guarded Triton inspection is not a truncated decorator prefix")
        return recovered, first


def configure_lora_triton_source():
    """Install only for the exact affected runtime, before worker kernel imports."""
    global _INSPECTION
    if sys.version_info[:3] != (3, 10, 9):
        return
    import vllm
    if vllm.__version__ != "0.31.0":
        raise RuntimeError("guarded LoRA source inspection requires vLLM0.31.0")
    jit = importlib.import_module("triton.runtime.jit")
    if _INSPECTION is not None and jit.inspect is _INSPECTION:
        return
    if jit.inspect is not inspect:
        raise RuntimeError("Triton inspect binding already has an unknown override")
    directory = Path(vllm.__file__).resolve().parent / "models/minimax_m3/common/ops"
    paths = {key: directory / (key + ".py") for key in KERNEL_SOURCES.values()}
    paths.update(triton_jit=Path(jit.__file__), inspect=Path(inspect.__file__))
    for name, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256[name]:
            raise ValueError(f"guarded LoRA {name} source digest differs")
    _INSPECTION = _SourceInspection(inspect, {
        module: (paths[key], SOURCE_SHA256[key]) for module, key in KERNEL_SOURCES.items()})
    jit.inspect = _INSPECTION
