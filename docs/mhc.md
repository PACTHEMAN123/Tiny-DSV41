# DSV4 mHC Triton Kernels

DSV4 carries four residual streams at each manifold hyper-connection (mHC)
site. For a local token tensor with hidden size `H`, the relevant shapes are:

```text
streams:          [..., 4, H]
pre/post weights: [..., 4]
residual weights: [..., 4, 4]
```

`dsv41_train/models/dsv4/mhc.py` contains CUDA Triton kernels plus PyTorch
references. CPU execution, missing Triton installations, empty tensors, and
non-four-stream configurations use the reference path.

## Scope

The implementation follows the operation boundaries used by Megatron Core's
mHC support:

1. RMS normalization and the learned projection remain PyTorch operations.
   `F.linear` therefore continues to use the platform GEMM backend.
2. The projected `4 x 4` residual logits use a fused Triton Sinkhorn forward
   and backward.
3. Stream aggregation (`h_aggregate`, named `collapse` in this repository) is
   fused into one Triton kernel.
4. Branch expansion and residual mixing (`h_post`, named `expand`) are fused
   into one Triton kernel.

The checkpoint parameter layout is unchanged: `fn`, `base`, and `scale` keep
their existing names, shapes, and FP32 computation.

## Sinkhorn

One Triton program handles one token's `4 x 4` residual matrix. The matrix
stays in registers for all 20 configured iterations:

1. Subtract each row maximum and exponentiate.
2. Normalize rows, add `hc_eps`, then normalize columns.
3. Repeat row and column normalization for the remaining iterations.

The backward kernel reconstructs the row and column normalization states in a
temporary FP32 workspace, then applies their Jacobians in reverse order. It
finally multiplies by the saved exponentials to obtain the logit gradient.
Keeping the complete `4 x 4` matrix in one program removes the 40 small CUDA
normalization launches in the eager reference.

## Collapse

The collapse kernel launches a two-dimensional grid over tokens and hidden
tiles. Each program loads the same hidden tile from all four streams,
multiplies by the token's four input weights, accumulates in FP32, and stores
the result in the stream dtype.

The custom backward writes the four stream gradients directly. Weight
gradients are FP32 reductions over the hidden tile and use atomics across
tiles; each token has independent weight locations, so contention is limited
to the hidden tiles of that token.

## Expand

For output stream `k`, the kernel evaluates

```text
output[k] = post[k] * branch + sum_j residual[j] * combine[j, k]
```

for one token/hidden tile. All multiplications and additions use FP32 before
the result is stored in the residual stream dtype. The backward kernel
computes branch and residual gradients directly and reduces the `post` and
`combine` gradients in FP32.

## References

- DeepSeek-AI, [mHC: Manifold-Constrained Hyper-Connections](https://arxiv.org/abs/2512.24880),
  for the architecture, doubly stochastic residual mapping, and Sinkhorn
  formulation.
- NVIDIA Megatron Core,
  [HyperConnection API](https://docs.nvidia.com/megatron-core/developer-guide/nightly/apidocs/core/core.transformer.hyper_connection.html)
  and
  [`fused_mhc_kernels.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/fusions/fused_mhc_kernels.py),
  for the projection/Sinkhorn/aggregation/post-operation boundaries and the
  reverse normalization derivative.
- NVIDIA Transformer Engine,
  [`common/triton/mhc.py`](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/common/triton/mhc.py),
  for the production Triton Sinkhorn interface and FP32 iteration policy.
- Nucleus AI,
  [`mHC-triton`](https://github.com/WithNucleusAI/mHC-triton), for an
  independent fused-training implementation used to cross-check tensor
  orientation and autograd coverage.

The code in this repository is adapted to DSV4's per-token weights and
`[..., source_stream, target_stream]` residual-matrix layout; it is not copied
as a dependency from any of these projects.

## Validation

Validation was run on September 30, 2026 with an NVIDIA L20A, PyTorch 2.13,
Triton 3.7.1, BF16 streams, FP32 mixing weights, 20 Sinkhorn iterations, and
the production hidden size `H=5120`.

| Path | Maximum absolute error | Relative L2 error |
|---|---:|---:|
| Sinkhorn output | `1.19e-7` | `1.13e-7` |
| Sinkhorn/input gradient | `1.49e-7` | `9.75e-8` |
| Sinkhorn/base gradient | `5.96e-7` | `1.20e-7` |
| Collapse output | `0` | `0` |
| Collapse stream gradient | `0` | `0` |
| Collapse weight gradient | `3.05e-5` | `1.11e-7` |
| Expand output | `3.91e-3` | `2.15e-6` |
| Expand branch gradient | `7.81e-3` | `9.52e-6` |
| Expand residual gradient | `7.81e-3` | `4.73e-6` |
| Expand weight gradients | `6.10e-5` | `1.56e-7` |

The nonzero expand differences are BF16 rounding differences after a different
FP32 accumulation order. A six-layer GPU model comparison produced a loss
difference of `5.08e-4` and a global parameter-gradient relative L2 error of
`8.94e-3` across 144 gradient tensors.

The same six-layer Triton model with activation checkpointing enabled matched
the non-checkpointed loss exactly; its parameter-gradient relative L2 error was
`3.03e-8`.

The complete L20A unit suite passes 48 tests; two optional `safetensors` tests
are skipped when that package is unavailable. Run the focused test with:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m unittest -v tests.test_triton_mhc
```

## Forward microbenchmark

The following forward-only timings use 4,096 token rows and `H=5120`. They are
kernel microbenchmarks, not end-to-end training throughput measurements.

| Operation | Triton | PyTorch reference | Speedup |
|---|---:|---:|---:|
| Dynamic weights + Sinkhorn | `0.221 ms` | `1.171 ms` | `5.3x` |
| Collapse | `0.076 ms` | `0.520 ms` | `6.8x` |
| Expand | `0.117 ms` | `1.843 ms` | `15.7x` |
