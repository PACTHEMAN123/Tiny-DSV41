"""Small, owned vLLM worker callbacks for unquantized Qwen head updates.

Callbacks run in the actual engine worker. Artifact metadata and a readback
hash establish activation; an operator-supplied version string alone does not.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

from .vllm_profile import VLLM_KERNEL_CONFIG, TRITON_EXPERT_CLASS, VLLM_WORKER_RPC


CANONICAL_HEAD = "backend.lm_head.weight"
VLLM_STARTUP_ADAPTER = "qwen3-moe-minimax-import-guard-v1"
VLLM_WORKER_EXTENSION = "rollout_manager.backends.vllm_qwen_startup.QwenStartupExtension"
STOCK_KERNEL_WARMUP_SHA256 = "69d9f633e353d02898b0e0054613f10596747341ea23e4369c1cdfd88ec2cac5"
_STARTUP_ADAPTER_APPLIED = 0
_GRAPH_REPLAYS = 0
_GRAPH_OBSERVER_INSTALLED = False


def tensor_digest(tensor):
    """Same name/shape/dtype/bytes digest as the native learner workflow."""
    import torch
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(json.dumps([CANONICAL_HEAD, list(value.shape), str(value.dtype)]).encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return {"sha256": digest.hexdigest(), "parameter_names": [CANONICAL_HEAD]}


def checked_artifact(artifact):
    import torch
    from dsv41_train.rl.checkpoint import file_sha256
    path = Path(artifact["path"])
    info = path.lstat()
    if (not path.is_absolute() or path.is_symlink() or not stat.S_ISREG(info.st_mode)
            or not 0 < info.st_size <= 2 * 1024**3):
        raise ValueError("vLLM update requires a bounded immutable regular artifact")
    if file_sha256(path) != artifact["digest"]:
        raise ValueError("vLLM artifact digest mismatch")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if set(state) != {"version", "lineage", "identity", "sampling", "trainable", "frozen"}:
        raise ValueError("vLLM artifact schema mismatch")
    if any(state.get(key) != artifact.get(key) for key in ("version", "lineage", "identity", "sampling")):
        raise ValueError("vLLM artifact metadata mismatch")
    if (state["frozen"] or set(state["trainable"]) != {CANONICAL_HEAD}
            or state["identity"].get("trainable_scope") != "lm_head"
            or state["identity"].get("random_init") is not False):
        raise ValueError("vLLM updates support only the identified mounted Qwen output head")
    head = state["trainable"][CANONICAL_HEAD]
    if not isinstance(head, torch.Tensor) or head.ndim != 2 or not torch.isfinite(head).all():
        raise ValueError("invalid vLLM head tensor")
    # Rehash after deserialization before any model mutation.
    if file_sha256(path) != artifact["digest"]:
        raise ValueError("vLLM artifact changed while loading")
    return head


def load_head(model, artifact):
    """Load through vLLM's HF tensor mapper and verify actual valid-vocab rows."""
    import torch
    head = checked_artifact(artifact)
    target = model.lm_head.weight
    vocab = model.config.vocab_size
    hidden = model.config.hidden_size
    if (tuple(head.shape) != (vocab, hidden) or target.ndim != 2 or target.shape[0] < vocab
            or target.shape[1] != hidden or target.dtype != head.dtype
            or getattr(model.lm_head, "quant_method", None) is not None
            and "Unquantized" not in type(model.lm_head.quant_method).__name__):
        raise ValueError("vLLM head shape/dtype/quantization differs from the native artifact")
    expected = tensor_digest(head)
    before = target.data_ptr()
    with torch.no_grad():
        loaded = model.load_weights([("lm_head.weight", head)])
    current = model.lm_head.weight
    if (loaded != {"lm_head.weight"} or current.data_ptr() != before or current.dtype != head.dtype
            or current.shape != target.shape):
        raise RuntimeError("vLLM head loader did not preserve the exact graph weight allocation")
    if current.device.type == "cuda":
        torch.cuda.synchronize(current.device)
    actual = tensor_digest(current[:vocab])
    if actual != expected:
        raise RuntimeError("actual vLLM worker head readback differs from the published artifact")
    return {"policy_ack": {key: artifact[key] for key in ("version", "digest", "lineage")},
            "loaded_trainable_tensors": actual, "shape": list(head.shape), "dtype": str(head.dtype),
            "storage_pointer_unchanged": True, "valid_vocabulary_rows": vocab,
            "padded_vocabulary_rows": current.shape[0], "completion_barrier": "head_readback_copied_to_cpu"}


def process_identity():
    root = Path("/proc") / str(os.getpid())
    return {"pid": os.getpid(), "start_ticks": int((root / "stat").read_text().rsplit(")", 1)[1].split()[19]),
            "cmdline_sha256": hashlib.sha256((root / "cmdline").read_bytes()).hexdigest()}


def install_graph_observer():
    """Count successful real replay invocations in this owned worker only."""
    global _GRAPH_OBSERVER_INSTALLED
    import torch
    if not _GRAPH_OBSERVER_INSTALLED:
        original = torch.cuda.CUDAGraph.replay
        def observed(graph):
            global _GRAPH_REPLAYS
            result = original(graph)
            _GRAPH_REPLAYS += 1
            return result
        torch.cuda.CUDAGraph.replay = observed
        _GRAPH_OBSERVER_INSTALLED = True


def captured_graph_receipt(worker):
    """Inspect the actual runner's live graph store, rather than global wrappers.

    vLLM 0.31 normally uses its V2 runner on CUDA. That runner captures and
    replays CUDAGraph objects directly through ModelCudaGraphManager; it does
    not create the older CUDAGraphWrapper instances.
    """
    import importlib
    import torch
    runner = worker.model_runner
    use_v2 = worker.use_v2_model_runner
    runner_type = type(runner).__module__ + "." + type(runner).__qualname__
    if type(use_v2) is not bool:
        raise RuntimeError("vLLM worker did not identify its actual model runner")
    if use_v2:
        module_name = "vllm.v1.worker.gpu.cudagraph_utils"
        module = importlib.import_module(module_name)
        manager = runner.cudagraph_manager
        if (runner_type != "vllm.v1.worker.gpu.model_runner.GPUModelRunner"
                or not isinstance(manager, module.ModelCudaGraphManager)
                or type(manager.graphs) is not dict or manager._graphs_captured is not True):
            raise RuntimeError("vLLM V2 runner has not completed actual graph capture")
        graphs = list(manager.graphs.values())
        source = "live_vllm_v2_CudaGraphManager_graphs"
    else:
        module_name = "vllm.compilation.cuda_graph"
        module = importlib.import_module(module_name)
        manager = runner.model
        if (runner_type != "vllm.v1.worker.gpu_model_runner.GPUModelRunner"
                or not isinstance(manager, module.CUDAGraphWrapper)):
            raise RuntimeError("vLLM V1 runner did not expose its actual full graph wrapper")
        graphs = [entry.cudagraph for entry in manager.concrete_cudagraph_entries.values()
                  if entry.cudagraph is not None]
        source = "live_vllm_CUDAGraphWrapper_entries"
    if any(not isinstance(graph, torch.cuda.CUDAGraph) for graph in graphs):
        raise RuntimeError("vLLM graph store contains an object that is not an actual CUDA graph")
    return {"model_runner_type": runner_type, "use_v2_model_runner": use_v2,
        "graph": {"captured_graph_count": len({id(graph) for graph in graphs}),
            "replay_calls": _GRAPH_REPLAYS,
            "replay_counter_source": "successful_torch_cuda_CUDAGraph_replay",
            "captured_graph_source": source,
            "captured_graph_source_module": module_name,
            "captured_graph_source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            "graph_manager_type": type(manager).__module__ + "." + type(manager).__qualname__,
            "capture_complete": use_v2 or bool(graphs)}}


def kernel_receipt(worker):
    """Read actual worker configuration and instantiated MoE kernels."""
    config = worker.vllm_config.kernel_config
    actual = {name: getattr(config, name) for name in VLLM_KERNEL_CONFIG}
    if actual != VLLM_KERNEL_CONFIG:
        raise RuntimeError("actual vLLM worker kernel configuration differs from its qualified profile")
    model = worker.model_runner.model
    implementations = []
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        backend = getattr(method, "unquantized_backend", None)
        if backend is None:
            continue
        selected = getattr(method, "experts_cls", None)
        experts = getattr(getattr(method, "moe_kernel", None), "fused_experts", None)
        selected_name = (selected.__module__ + "." + selected.__qualname__) if selected else None
        actual_name = (type(experts).__module__ + "." + type(experts).__qualname__) if experts else None
        value = getattr(backend, "value", None)
        if value != "TRITON" or selected_name != TRITON_EXPERT_CLASS or actual_name != TRITON_EXPERT_CLASS:
            raise RuntimeError("actual unquantized MoE experts differ from the pinned Triton implementation")
        implementations.append({"module": name, "backend": value,
            "selected_experts_class": selected_name, "experts_class": actual_name})
    if getattr(model.config, "num_experts", 0) and not implementations:
        raise RuntimeError("MoE model did not expose actual expert implementation evidence")
    return {"kernel_config": actual, "moe_implementations": implementations}


def guarded_qwen_kernel_warmup(original, worker, *args, **kwargs):
    """Keep stock warmup while avoiding its unrelated MiniMax model import.

    vLLM 0.31 imports the MiniMax helper before checking warmup flags. For
    this exact Qwen-only owned worker, that helper cannot apply to the model.
    The temporary helper affects this call alone; no installed source changes.
    """
    import sys
    from types import ModuleType
    global _STARTUP_ADAPTER_APPLIED
    config = worker.vllm_config
    actual = {name: getattr(config.kernel_config, name) for name in VLLM_KERNEL_CONFIG}
    if (config.model_config.hf_config.model_type != "qwen3_moe"
            or worker.model_runner.model.config.model_type != "qwen3_moe"
            or actual != VLLM_KERNEL_CONFIG):
        raise RuntimeError("owned MiniMax import guard requires the exact Qwen3 MoE/kernel profile")
    module_name = "vllm.model_executor.warmup.minimax_m3_msa_warmup"
    previous = sys.modules.get(module_name)
    temporary = ModuleType(module_name)
    calls = 0
    def guarded_minimax_warmup(same_worker):
        nonlocal calls
        if same_worker is not worker:
            raise RuntimeError("MiniMax import guard changed the owned worker")
        calls += 1
        return None  # MiniMax attention is absent from the verified Qwen model.
    temporary.minimax_m3_msa_warmup = guarded_minimax_warmup
    sys.modules[module_name] = temporary
    try:
        result = original(worker, *args, **kwargs)
        expected_calls = 0 if kwargs.get("process_local_only", False) else 1
        if calls != expected_calls:
            raise RuntimeError("stock kernel warmup did not call its guarded MiniMax helper exactly once")
        _STARTUP_ADAPTER_APPLIED += 1
        return result
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def startup_adapter_receipt(worker):
    import vllm.envs
    if (worker.vllm_config.parallel_config.worker_extension_cls != VLLM_WORKER_EXTENSION
            or _STARTUP_ADAPTER_APPLIED < 1 or vllm.envs.VLLM_ALLOW_INSECURE_SERIALIZATION):
        raise RuntimeError("actual owned Qwen startup adapter was not applied")
    return {"profile": VLLM_STARTUP_ADAPTER, "applied_count": _STARTUP_ADAPTER_APPLIED,
        "stock_kernel_warmup_sha256": STOCK_KERNEL_WARMUP_SHA256,
        "model_type": "qwen3_moe", "preserves_original_warmup_body": True,
        "bypassed_unrelated_minimax_import": True}


def worker_receipt(worker):
    import torch
    model = worker.model_runner.model
    device = model.lm_head.weight.device
    if device.type != "cuda":
        raise RuntimeError("vLLM training actor requires an actual CUDA worker")
    props = torch.cuda.get_device_properties(device)
    return {**process_identity(), **kernel_receipt(worker), **captured_graph_receipt(worker),
            "startup_adapter": startup_adapter_receipt(worker),
            "worker_rpc_method": VLLM_WORKER_RPC, "insecure_serialization": False,
            "device": str(device), "device_uuid": str(props.uuid),
            "device_name": props.name, "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(device)}


def owned_worker_call(worker, operation, artifact=None):
    """Called through the named worker extension RPC."""
    if operation == "observe":
        install_graph_observer()
        return worker_receipt(worker)
    if operation == "stats":
        return worker_receipt(worker)
    if operation == "load_head":
        receipt = load_head(worker.model_runner.model, artifact)
        return {**worker_receipt(worker), **receipt}
    raise ValueError("unsupported owned vLLM worker operation")
