"""Process-local Qwen startup adapter loaded by vLLM's worker extension API.

Only the exact stock 0.31 warmup is wrapped. Its complete body and subsequent
CUDA Graph capture remain intact. No shared Python installation is modified.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
from pathlib import Path

from .vllm_worker import STOCK_KERNEL_WARMUP_SHA256, guarded_qwen_kernel_warmup, owned_worker_call


class QwenStartupExtension:
    """Named supported extension; loading this module installs the local guard."""
    def rollout_manager_owned_call(self, operation, artifact=None):
        """String RPC keeps vLLM's secure serialization policy enabled."""
        return owned_worker_call(self, operation, artifact)


def install():
    import vllm
    if vllm.__version__ != "0.31.0":
        raise RuntimeError("owned Qwen startup adapter requires exact vLLM 0.31.0")
    module = importlib.import_module("vllm.model_executor.warmup.kernel_warmup")
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != STOCK_KERNEL_WARMUP_SHA256:
        raise RuntimeError("stock vLLM kernel warmup source differs from the qualified adapter preimage")
    original = module.kernel_warmup
    if getattr(original, "_owned_qwen_import_guard", False):
        return
    @functools.wraps(original)
    def guarded(worker, *args, **kwargs):
        return guarded_qwen_kernel_warmup(original, worker, *args, **kwargs)
    guarded._owned_qwen_import_guard = True
    gpu_worker = importlib.import_module("vllm.v1.worker.gpu_worker")
    if gpu_worker.kernel_warmup is not original:
        raise RuntimeError("GPU worker kernel warmup alias differs from its stock source")
    module.kernel_warmup = gpu_worker.kernel_warmup = guarded


install()
