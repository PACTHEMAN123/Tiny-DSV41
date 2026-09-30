# Packed Context Parallelism

The trainer flattens each local batch into one token stream before CP sharding.
For example, batch size 8 with sequence length 8192 becomes one 65,536-token
stream, which is then divided evenly across the CP group. Packing avoids padding
between samples and lets every CP rank receive one contiguous token range.

## Sequence boundaries

Packing does not allow samples to attend to each other. A sequence ID is stored
for every token and is used to reset RoPE positions, sliding-window attention,
compressed KV grouping, CSA index selection, Engram n-grams, and causal-LM
labels. Tokens at the end of one sample therefore cannot read tokens from the
next sample or produce a cross-sample training target.

## Runtime ownership

`ModelContext` builds the packed positions, local CP shard, bounded attention
indices, and attention mask once per forward. It also owns CP gather operations,
so attention modules do not retain process-group or CP topology state.

Sliding-window attention does not reconstruct the full sequence. Each CP rank
exchanges only the previous rank's `sliding_window - 1` KV tail and indexes its
local queries from that halo plus its local KV shard. The backward pass sends
the halo gradient back to its owning rank. If a local shard is shorter than one
window, the implementation falls back to the full gather path.

The sparse indexer requires CUDA and Triton. Its score kernel fuses QK, ReLU,
per-head weighting, and head reduction, and writes only the reduced score
matrix. There is no unfused PyTorch execution path that materializes the
`[batch, query, head, key]` intermediate.

Compressor source layers still gather their input sequence before compression.
YATT instead exchanges only the boundary tokens required by the compression
window, compresses locally, and all-gathers the compressed result. That
compressor stage-1 optimization remains separate from the SWA halo path above.

`ShadowIndexers` owns the compressed sequence IDs, compressed KV tensors,
indexer keys, selected indices, and candidate blocks shared across CSA layers.
At a checkpoint boundary it is explicitly flattened to tensors and reconstructed
inside the layer, allowing PyTorch to release activations during the forward.

Checkpointed and direct execution call the same decoder-layer function. Their
only difference is whether the selected loader is `checkpoint` or a direct call.
Engram tables stay on GPU and retain row-wise all-to-all lookup; CPU offload is
not part of this path. CP and EP must currently use the same group size.

The routed MoE uses expert-major all-to-all, grouped GEMMs, and a Triton
clamped-SwiGLU kernel. Its expert-FSDP topology and memory results are documented
in [`moe.md`](moe.md).

## End-to-end command

Run this on four 4-GPU nodes, setting `NODE_RANK` to `0`, `1`, `2`, or `3`:

```bash
NODE_RANK=0 MASTER_ADDR=10.0.0.1 \
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
  --post-training \
  --metrics-file /path/to/ep16-cp16-bs16.json
```

`--post-training` requires the complete 40-layer CED backbone. Layers 0-19 run
on the full packed sequence. At layer 20, the model builds global compressed KV
from the complete encoder output, then replays only the final 128 tokens of each
packed sequence through layers 20-39 while preserving their original positions.

On September 30, 2026, this command completed on 16 NVIDIA L20A GPUs with loss
12.77656, a non-zero parameter update, 124.68 GiB peak allocated memory, and
41.74 seconds in the sharded training section. The emitted metrics recorded
`post_training=true`, `decoder_swa_bounded_replay=true`, and
`decoder_replay_window=128`.

The non-replay CP16/EP16 path has also been validated at batch sizes 16 and 32.
The batch-32 run uses 156.18 GiB peak allocated memory and completes with a
finite parameter update.
