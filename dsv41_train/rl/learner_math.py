"""Explicit GEMM policy for the distributed native learner process."""
from contextlib import contextmanager

import torch


def learner_math_configuration():
    api = torch.backends.cuda.matmul
    return {"preferred_blas_library": torch.backends.cuda.preferred_blas_library().name.lower(),
            "bf16_reduced_precision_reduction": api.allow_bf16_reduced_precision_reduction,
            "bf16_split_k": getattr(api, "allow_bf16_reduced_precision_reduction_split_k", None),
            "float32_matmul_precision": torch.get_float32_matmul_precision()}


def require_bf16_no_split_k():
    current = learner_math_configuration()
    if (current["preferred_blas_library"] != "cublaslt"
            or current["bf16_reduced_precision_reduction"] is not False
            or current["bf16_split_k"] is not False
            or current["float32_matmul_precision"] != "highest"):
        raise ValueError("sharded response ownership requires the bf16-no-split-k learner math policy")
    return current


@contextmanager
def configure_learner_math(mode):
    """Request Lt/FP32 reductions without split-K; restore process settings on exit.

    Qualified on fixed real-model layouts, not a guarantee for arbitrary shapes:
    PyTorch may fall back to other BLAS kernels; this is not a kernel selection guarantee.
    Model weights and HF cache quantization are unchanged.
    """
    if mode == "default":
        yield learner_math_configuration()
        return
    if mode != "bf16-no-split-k":
        raise ValueError("learner math mode must be default or bf16-no-split-k")
    previous = learner_math_configuration()
    if previous["bf16_split_k"] is None:
        raise ValueError("bf16-no-split-k learner math requires PyTorch split-K control")
    try:
        torch.backends.cuda.preferred_blas_library("cublaslt")
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
        torch.set_float32_matmul_precision("highest")
        yield require_bf16_no_split_k()
    finally:
        torch.set_float32_matmul_precision(previous["float32_matmul_precision"])
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (
            previous["bf16_reduced_precision_reduction"], previous["bf16_split_k"])
        torch.backends.cuda.preferred_blas_library(previous["preferred_blas_library"])
