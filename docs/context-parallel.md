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

`ShadowIndexers` owns the compressed sequence IDs, compressed KV tensors,
indexer keys, selected indices, and candidate blocks shared across CSA layers.
At a checkpoint boundary it is explicitly flattened to tensors and reconstructed
inside the layer, allowing PyTorch to release activations during the forward.

Checkpointed and direct execution call the same decoder-layer function. Their
only difference is whether the selected loader is `checkpoint` or a direct call.
Engram tables stay on GPU and retain row-wise all-to-all lookup; CPU offload is
not part of this path. CP and EP must currently use the same group size.

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
  --start-layer 0 --num-layers 40 --cp-size 8 --ep-size 8 \
  --steps 1 --batch-size 8 --seq-len 8192 --optimizer sgd \
  --sequence-packing --gradient-checkpointing --optimizer-in-backward
```
