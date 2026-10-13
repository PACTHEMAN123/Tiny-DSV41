# Native DSV4 / Qwen Trainer

Pure PyTorch training for DeepSeek V4.1 and Qwen3 MoE.

Run all commands from the repository root.

## Post-training and profiling

DSV4 supports selective activation checkpointing and native PyTorch linear LoRA.
No Transformers or PEFT runtime dependency is required.

```bash
python3 profile_dsv4.py --layers 3 --seq-len 1024 \
  --checkpoint-mode selective --lora --output artifacts/profile.json
```

See [post-training usage](docs/post_training.md),
[H20 profiling results](docs/profiling.md), and [validation](docs/testing.md).

## DeepSeek V4.1

DSV4 supports direct shard loading, trainable CPU Engram tables, asynchronous
row prefetch, selective FSDP CPU offload and sharded training-state resume.
These policies default to disabled. See [configuration and examples](docs/memory_offload.md),
[validation](docs/memory_validation.md), and [measurements](docs/memory_results.md).
An optional bounded GPU cache for CPU Engram tables is described in
[cache measurements](docs/engram_cache_results.md); it remains disabled by default.
The [95B, 40-layer H20 measurements](docs/dsv41_large_scale_results.md) cover
near-capacity training, loading costs, full timelines and prefetch deadlines.

### Two nodes, 16 GPUs, full 40 layers

Run this command on both 8-GPU nodes. Set `NODE_RANK=0` on the first node and
`NODE_RANK=1` on the second. Replace `10.0.0.1` with the first container's IP.

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
  --ep-size 8 --cp-size 1 \
  --optimizer sgd --steps 1 --batch-size 1 --seq-len 8
```

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

## Qwen3-30B-A3B

The real Alpaca SFT comparison with newly added trainable Engram, CPU offload,
and GPU row caching is documented in
[`docs/qwen30b_engram_sft_results.md`](docs/qwen30b_engram_sft_results.md).

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

The measured LoRA/full-training comparison and architecture bottlenecks are in
[`docs/profiling_peft_full.md`](docs/profiling_peft_full.md). Run the comparison
with `python3 profile_post_training.py --fsdp-gpus 0,1`.
