# Native DSV4 / Qwen Trainer

Pure PyTorch training for DeepSeek V4.1 and Qwen3 MoE.

Run all commands from the repository root.

## DeepSeek V4.1

### Two nodes, 16 GPUs, full 40 layers with intra-node CP8

Run this command on both 8-GPU nodes. Set `NODE_RANK=0` on the first node and
`NODE_RANK=1` on the second. Replace `10.0.0.1` with the first container's IP.
Torchrun assigns ranks 0-7 to node 0 and ranks 8-15 to node 1, so both the CP8
and EP8 groups remain inside each node. Dense parameters use FSDP16 across the
whole job, routed experts use FSDP2 across matching ranks on the two nodes, and
Engram rows are split across all 16 ranks. The two nodes process different data
samples; the eight CP ranks in one node process contiguous shards of the same
sample.

```bash
NODE_RANK=0
MASTER_ADDR=10.0.0.1

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  train_dsv4_checkpoint.py \
  --model-path /path/to/DeepSeek-V4.1-Flash \
  --start-layer 0 --num-layers 40 \
  --ep-size 8 --cp-size 8 \
  --optimizer sgd --steps 1 --batch-size 1 --seq-len 256 \
  --gradient-checkpointing --offload-engram \
  --metrics-file /mnt/fuse/oss/xiaopac.xjy/dsv41/cp8-fsdp16-256.json
```

Each GPU processes 32 query tokens in this first full-model run. Gradient
checkpointing recomputes the decoder stack during backward, while
`--offload-engram` keeps the row-sharded sparse Engram tables in host memory and
copies only the selected rows to the GPU. Engram offload is available with the
sparse SGD path; validate this run before increasing the sequence length.

### One node, 8 GPUs, 9-layer prefix

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_dsv4_checkpoint.py \
  --model-path /path/to/DeepSeek-V4.1-Flash \
  --start-layer 0 --num-layers 9 \
  --ep-size 8 --cp-size 1 \
  --optimizer sgd --steps 1 --batch-size 1 --seq-len 8
```

### One node, 8 GPUs, CP8 long-sequence smoke

CP8 and EP8 share the eight local ranks. Each CP rank processes 2,048 tokens
for this 16K-token sequence while the routed experts and Engram rows stay
partitioned across the same ranks. Start with one source-anchored compressed
layer, then increase `--num-layers` only after checking the reported peak
memory.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_dsv4_checkpoint.py \
  --model-path /mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash \
  --start-layer 2 --num-layers 1 \
  --ep-size 8 --cp-size 8 \
  --optimizer sgd --steps 1 --batch-size 1 --seq-len 16384 \
  --metrics-file /mnt/fuse/oss/xiaopac.xjy/dsv41/cp8-16k.json
```

DSV4 CP currently requires `cp-size == ep-size` whenever CP is enabled. This
keeps the context and expert groups aligned so fully sharded dense gradients,
partitioned expert gradients, and row-sharded Engram gradients use the intended
normalization.

## Qwen3-30B-A3B

### One node, 8 GPUs

Checkpoint loading requires the `safetensors` package.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_qwen.py \
  --model-path /path/to/Qwen3-30B-A3B \
  --cp-size 2 --ep-size 2 \
  --gradient-checkpointing \
  --steps 1 --batch-size 1 --seq-len 16
```

Testing and distributed diagnostics are documented in
[`docs/testing.md`](docs/testing.md).
