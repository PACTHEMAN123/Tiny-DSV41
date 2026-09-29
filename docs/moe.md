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
- CP and EP currently have the same size, and replicated expert gradients are
  normalized by the CP size after backward.

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
5. After expert computation, reverse the permutation and all-to-all, then
   accumulate the top-k outputs into their source tokens in FP32.

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
backward computes gate, up, and routing-weight gradients. Both row and column
strides are explicit because gate and up are interleaved views with column
stride 2.

This removes the Python loop over experts and avoids retaining separate FP32
clamp, SiLU, and multiply intermediates for every expert during checkpoint
recomputation. CPU execution retains a small PyTorch loop as a correctness
reference; CUDA training requires grouped GEMM and Triton.

## Four-node experiment

Run the following command on four 4-GPU nodes. Set `NODE_RANK` to `0`, `1`,
`2`, or `3`, and set `MASTER_ADDR` to node 0's container IP.

```bash
NODE_RANK=0
MASTER_ADDR=10.0.0.1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.run \
  --nnodes=4 --nproc-per-node=4 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  train_dsv4_checkpoint.py \
  --model-path /mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash \
  --start-layer 0 --num-layers 40 --cp-size 16 --ep-size 16 \
  --steps 1 --batch-size 16 --seq-len 8192 --optimizer sgd \
  --sequence-packing --gradient-checkpointing --optimizer-in-backward \
  --metrics-file /path/to/ep16-cp16-bs16.json
```

To reproduce the batch-32 limit experiment, change `--batch-size 16` to
`--batch-size 32` and use a different metrics file. Packing produces 8,192
local tokens per rank at batch 16 and 16,384 at batch 32.

## Results

These runs used four nodes with four NVIDIA L20A GPUs per node, a 40-layer
DeepSeek-V4.1-Flash prefix, sequence packing, gradient checkpointing, and SGD
in backward.

| Revision and configuration | Loss | Peak allocated | Train time | Result |
|---|---:|---:|---:|---|
| Legacy experts, CP16/EP16, batch 16 (`15b81a7`) | 17.30916 | 153.86 GiB | 63.82 s | Passed |
| Grouped experts, CP16/EP16, batch 16 (`65589fc`) | 17.30551 | 131.31 GiB | 55.76 s | Passed |
| Grouped experts, CP16/EP16, batch 32 (`65589fc`) | 17.44295 | 168.96 GiB | 82.33 s | Invalid update |

At batch 16, grouped experts reduce peak allocated memory by 22.55 GiB
(14.66%) and step time by 8.06 seconds (12.63%) relative to the legacy expert
loop. The loss difference is 0.00365 and the tracked parameter update remains
finite.

Batch 32 reaches the end of the step, but it is not a valid training result.
Several ranks report allocator mapping failures with only about 7.6 MiB free,
and the tracked parameter delta is `NaN`. Batch 16 is therefore the largest
validated configuration from these experiments.

Validation also includes 40 unit tests, two-rank EP forward/backward parity,
four-rank `EP2 x expert-FSDP2` backward, and an interleaved-stride Triton
forward/backward comparison against the PyTorch reference.
