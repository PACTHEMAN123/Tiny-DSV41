"""First-order route-weight gradients with a fixed FP32 reduction per token."""
import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = tl = None


if triton is not None:
    @triton.jit
    def _weight_gradient(current, upstream, output, D: tl.constexpr,
                         XS0: tl.constexpr, XS1: tl.constexpr,
                         GS0: tl.constexpr, GS1: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        column = tl.arange(0, BLOCK)
        x = tl.load(current + row * XS0 + column * XS1, column < D, other=0).to(tl.float32)
        g = tl.load(upstream + row * GS0 + column * GS1, column < D, other=0).to(tl.float32)
        tl.store(output + row, tl.sum(x * g, axis=0))


class _RoutingWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, current, weights):
        if (triton is None or not current.is_cuda or current.dtype != torch.float32
                or weights.dtype != torch.float32 or weights.device != current.device or current.ndim != 2
                or weights.shape != (current.shape[0],)):
            raise ValueError("stable routing gradients require CUDA FP32 activations and weights")
        ctx.save_for_backward(current, weights)
        return current * weights[:, None]

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        current, weights = ctx.saved_tensors
        weight_gradient = torch.empty(weights.shape, device=weights.device, dtype=weights.dtype)
        if current.shape[0]:
            _weight_gradient[(current.shape[0],)](
                current, gradient, weight_gradient, current.shape[1],
                *current.stride(), *gradient.stride(), triton.next_power_of_2(current.shape[1]),
                num_warps=4, enable_fp_fusion=False)
        return gradient * weights[:, None], weight_gradient


multiply_routing_weight = _RoutingWeight.apply
