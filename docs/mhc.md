# DSV4 mHC Triton Kernels

DSV4 carries four residual streams at each manifold hyper-connection (mHC)
site. For a local token tensor with hidden size `H`, the relevant shapes are:

```text
streams:          [..., 4, H]
pre/post weights: [..., 4]
residual weights: [..., 4, 4]
```

`dsv41_train/models/dsv4/mhc.py` owns the complete `HyperConnection` module,
the CUDA Triton kernels, and an internal PyTorch fallback. CPU execution,
missing Triton installations, empty tensors, and non-four-stream
configurations use the same module interface without exposing separate
reference APIs.

## Scope

The model adapts to one mHC object with two natural phases. Calling
`HyperConnection(streams, input_weights)` performs RMS normalization, the
learned projection, coefficient activation, Sinkhorn, and stream collapse.
After attention or MoE computes the branch value, `HyperConnection.finish()`
performs branch expansion and residual mixing. The branch itself prevents
these phases from becoming one physical kernel, but no individual kernel or
reference function leaks into `model.py`.

Internally, the implementation follows the operation boundaries used by
Megatron Core's mHC support:

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

The custom backward writes the four stream gradients directly. Each hidden
tile writes its weight-gradient contribution to a distinct FP32 partial.
A fixed-axis reduction combines these partials for each token and stream,
then casts the result to the weight dtype. The tiles do not atomically add
into shared weight-gradient locations.

## Expand

For output stream `k`, the kernel evaluates

```text
output[k] = post[k] * branch + sum_j residual[j] * combine[j, k]
```

for one token/hidden tile. All multiplications and additions use FP32 before
the result is stored in the residual stream dtype. The backward kernel
computes branch and residual gradients directly. The `post` and `combine`
gradients use separate FP32 partials for every hidden tile, followed by a
fixed-axis reduction and a cast to the corresponding weight dtype, as in
collapse.

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

## Testing

Compare native outputs and gradients with the reference implementation:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m unittest -v tests.test_triton_mhc
```

Tests require CUDA and Triton. BF16 rounding can differ from the reference
accumulation order; the regression assertions define the supported tolerances.
