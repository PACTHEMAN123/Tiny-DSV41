from __future__ import annotations

import ast
import json
from pathlib import Path

from ..common import digest


FILES = {"model_new.py", "kernels/solution.cu", "kernels/solution_binding.cpp"}


def parse_candidate(content: str) -> dict[str, str]:
    if not isinstance(content, str):
        raise ValueError("model response content must be text")
    text = content.strip()
    if text.startswith("```json\n") and text.endswith("```"):
        text = text[8:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON source object")
    if set(value) == {"files"}:
        files = value["files"]
    elif set(value) == FILES:
        files = value
    else:
        raise ValueError("response must contain only a files object or the three source paths")
    if not isinstance(files, dict) or set(files) != FILES:
        raise ValueError("candidate must contain exactly the three permitted source paths")
    if any(not isinstance(v, str) or not 0 < len(v.encode()) <= 65536 for v in files.values()):
        raise ValueError("candidate source size/type rejected")
    tree = ast.parse(files["model_new.py"])
    if not any(isinstance(node, ast.ClassDef) and node.name == "ModelNew" for node in tree.body):
        raise ValueError("model_new.py must define the top-level class ModelNew; class Model is the reference entry point")
    return files


def require_review(path: Path, files: dict[str, str]) -> None:
    approval = json.loads(path.read_text())
    if approval != {"sha256": digest(files), "scope": "reviewed-development-pilot"}:
        raise ValueError("approval does not match this exact candidate")


def prompt(task: dict, feedback: dict | None = None, previous: str | None = None,
           *, architecture: str = "10.3") -> list[dict]:
    instructions = '''Implement the reference operation as a real CUDA C++ extension.
Return ONLY JSON: {"files":{"model_new.py":"...","kernels/solution.cu":"...","kernels/solution_binding.cpp":"..."}}.
ModelNew must match Model's initialization and forward signature and import cuda_extension.
Use raw CUDA math in solution.cu, with no PyTorch includes. The binding may inspect tensors and
allocate output with torch::empty_like, but may not perform tensor math. Use float32.
Do not change reference shapes, inputs, state, verification, timing, or any other files.
No file/network/process access, inline compilation, torch operations as fallback, timing manipulation,
profiler detection, output caching, or reads of reference outputs. Compute every output from inputs.
Use bounds-checked indexing and current CUDA stream. The device runtime is compute capability 10.3.
Compile flags are -O3 --use_fast_math. Prefer simple, auditable kernels over speculative tricks.
The immutable binding_registry.h exposes REGISTER_BINDING(name, registration_function).
Example binding pattern:
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"
extern "C" void launch(const float*, float*, int64_t, cudaStream_t);
torch::Tensor forward(torch::Tensor x) {
 TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type()==torch::kFloat32, "expected contiguous CUDA float32");
 auto y=torch::empty_like(x);
 launch(x.data_ptr<float>(), y.data_ptr<float>(), x.numel(), c10::cuda::getCurrentCUDAStream().stream());
 return y;
}
static void reg(pybind11::module& m) { m.def("forward", &forward); }
REGISTER_BINDING(solution, reg);
'''
    messages = [{"role": "system", "content": instructions},
                {"role": "user", "content": "Implement this unchanged reference:\n" + task["code"]}]
    messages[0]["content"] = instructions.replace("capability 10.3", "capability " + architecture)
    if task.get("extra_instruction"):
        messages[-1]["content"] += "\n\nAdditional interface requirement:\n" + task["extra_instruction"]
    if previous is not None:
        messages += [{"role": "assistant", "content": previous},
                     {"role": "user", "content": "Actual harness feedback:\n" + json.dumps(feedback) +
                      "\nMake one improvement or repair. Return the complete three-file JSON again."}]
    return messages
