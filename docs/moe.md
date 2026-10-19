# DSV4 MoE Design

DSV4 owns a dedicated MoE package under `dsv41_train/models/dsv4/moe`:

- `dispatch.py` implements local and expert-parallel token dispatch.
- `module.py` contains the router, grouped routed experts, and shared expert.
- `kernels.py` contains the Triton clamped-SwiGLU forward and backward kernels.

Qwen continues to use the model-independent dispatcher in
`dsv41_train/dispatch.py`.

## Parallel topology

The sparse mesh has shape `(world_size // ep_size, ep_size)` with dimensions
named `expert_fsdp` and `ep`.

- The EP coordinate owns one contiguous range of global experts.
- The `expert_fsdp` coordinate shards replicas of that expert range with FSDP2.
- Dense layer parameters use the dense FSDP mesh.
- CP/EP ownership and expert-gradient normalization are configured by the
  entry point; validate the selected topology against an unsharded reference.

For a 16-GPU job:

| Configuration | Experts per EP rank | Expert FSDP size | Meaning |
|---|---:|---:|---|
| `ep-size=16` | 24 | 1 | Pure expert parallelism |
| `ep-size=8` | 48 | 2 | EP8 with two-way expert FSDP |

The grouped expert module works in both cases. With expert FSDP enabled, FSDP2
unshards the contiguous local-expert tensors before grouped computation and
reduce-scatters their gradients after backward.

## Dispatch and combine

The dispatcher follows the TorchTitan expert-major layout:

1. Flatten the top-k assignments and stable-sort them by global expert ID.
2. Exchange per-expert token counts across the EP group.
3. Launch autograd-aware functional all-to-all for token states and routing
   weights.
4. Permute the received rank-major segments into local expert-major order.
5. After expert computation, reverse the permutation and all-to-all, restore
   each token's original router-slot order with the inverse dispatch
   permutation, and reduce its top-k outputs in FP32. Both local and EP
   dispatch use this fixed-axis reduction instead of repeated-index scatter
   addition.

The data collectives use PyTorch functional all-to-all. The current eager path
consumes each result before expert computation, so it does not yet schedule
communication/computation overlap. The functional collective boundary keeps
that scheduling possible without changing the expert interface.

## Grouped expert computation

Each rank stores two contiguous parameters:

```text
gate_up: [local_experts, intermediate, 2, hidden]
down:    [local_experts, hidden, intermediate]
```

The gate/up dimension is physically interleaved. Checkpoint loading writes W1
to `gate_up[:, :, 0, :]`, W3 to `gate_up[:, :, 1, :]`, and W2 to `down`.

For expert-major routed input `x` and per-expert token counts:

```text
offsets = cumsum(token_counts, int32)
gate_up = grouped_mm(x, gate_up.reshape(E, 2I, H).transpose(-2, -1), offsets)
gate, up = gate_up.reshape(R, I, 2).unbind(-1)
hidden = triton_clamped_swiglu(gate, up, routing_weight)
output = grouped_mm(hidden, down.transpose(-2, -1), offsets)
```

The Triton kernel performs FP32 clamping, SiLU, multiplication, and routing
weight scaling, then stores BF16 output for the down projection. Its custom
backward computes gate and up gradients directly. For each routing weight,
each column tile writes a distinct FP32 gradient partial; a fixed-axis
reduction combines the partials without atomic additions between tiles.
Both row and column strides are explicit because gate and up are interleaved
views with column stride 2.

This removes the Python loop over experts and avoids retaining separate FP32
clamp, SiLU, and multiply intermediates for every expert during checkpoint
recomputation. CPU execution retains a small PyTorch loop as a correctness
reference; CUDA training requires grouped GEMM and Triton.

## Testing

```bash
python3 -m unittest -v tests.test_moe_dispatch tests.test_triton_moe
```

CUDA paths require grouped GEMM and Triton support. Use the distributed checks
in [testing](testing.md) before changing expert ownership or collective scheduling.
