# Native DSV4 / Qwen Trainer

Pure PyTorch training for DeepSeek V4.1 and Qwen3 MoE.

Run all commands from the repository root.

Start with the [SFT and LoRA guide](docs/sft.md) for a new checkout.
The [4096-token validation report](docs/longseq_identity_results_20261014.md)
separates identity adaptation, native Engram, and Qwen dummy-table efficiency.
Raw experiment artifacts, adapters, and machine-specific evidence are not
included in Git. The [original-environment replay guide](docs/longseq_identity_quickstart_20261014.md)
requires those retained local artifacts.

## Engram support

This branch supports trainable, row-sharded Engram tables with independent
placement, prefetch and cache policies. DSV4 uses its existing Engram structure;
the Qwen experiment adds newly initialized Engram to a pretrained Qwen3 MoE.

| Feature | Behavior and entry point |
| --- | --- |
| Trainable CPU offload | Keeps owned table rows and sparse gradients on CPU; transfers requested rows to GPU and updates tables with SGD. |
| Asynchronous prefetch | Overlaps CPU lookup/H2D with earlier layers using bounded pinned buffers and copy streams; row-routing collectives remain ordered on the main thread. |
| Request deduplication | Restores lookup order and repeated-row gradients; works with CPU or GPU tables independently of caching. |
| Bounded GPU row cache | Supports CPU or GPU source tables, with a per-table/per-rank budget and invalidation after updates. GPU-resident tables retain their full shard. |
| Loading and FSDP policies | DSV4 supports direct shard loading and selective dense/expert CPU offload, independently of Engram placement. |
| Sharded training state | The DSV4 entry point saves parameters, optimizer and RNG state without gathering the full model; restore requires the same topology. |
| Qwen SFT and profiling | A dedicated runner compares all four offload/cache combinations on real SFT data and exports timing, memory and CPU/CUDA traces. |

Read the [Engram feature and usage guide (Chinese)](docs/engram_features.md)
for commands, defaults, compatibility and cache semantics. Performance policies
are opt-in in the general training path; the Qwen comparison runner explicitly
selects its four experimental configurations. CPU tables and cached tables use
the supported sparse SGD path. Prefetch and cache do not guarantee a speedup.

## Post-training and profiling

`train_sft.py` provides real JSONL answer-masked SFT for Qwen and DSV4, with
token-weighted gradient accumulation and distributed checkpoint resume. See
[SFT setup and commands](docs/sft.md) and [L20D real-checkpoint validation](docs/l20d_validation_20261013.md). The older `train_qwen.py` and
`train_dsv4_checkpoint.py` commands below are random-token execution checks.

`train_sft.py --lora` supports native PyTorch attention LoRA for the complete
Qwen3 MoE backbone and a DSV4 layer window, including all 40 text layers. Qwen targets q/k/v/o projections;
`--qwen-engram` cannot be combined with LoRA. DSV4 additionally supports selective
activation checkpointing. No Transformers or PEFT runtime dependency is required.

`--checkpoint-state full` remains the default. For LoRA,
`--checkpoint-state lora` saves A/B adapters, persistent buffers, optimizer, RNG
and step while reloading the frozen base from its mounted checkpoint on resume.
This reduces checkpoint storage, not base-model GPU memory. Base identity hashes
config/index files but not weight shards, so the mounted base must stay immutable.
See [Qwen LoRA commands](docs/sft.md#qwen-attention-lora) and
[checkpoint scope and resume restrictions](docs/sft.md#lora-adapters-and-checkpoint-scope).

The [earlier real-weight LoRA validation](docs/lora_validation_20261013.md) covers
a DSV4 one-layer prefix and split/resume checks. The [current 4096-token validation](docs/longseq_identity_results_20261014.md)
records 20-step LoRA runs for all 48 Qwen layers and all 40 DSV4 text layers
(`none` and `hf` cache-value modes), followed by fresh-process adapter reload
and GPU process-exit/memory-recovery checks. This tests identity adaptation;
it is not a general capability benchmark. Resume evidence from shorter runs
is recorded separately.

Native Engram preserves the mounted V4.1 token mapping, n-gram addressing and
gate formula. Its trainable rows and row cache use decoded BF16 values, not the
checkpoint's packed FP8 rows/scales. The separate `--cache-quantization hf` mode
matches the tested attention KV/index quantize/dequantize value formats while
retaining BF16 storage and identity STE gradients. See the
[official-format scope and limitations](docs/longseq_identity_quickstart_20261014.md#deepseek-官方格式与-hf-模式的范围).

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
An optional bounded GPU row cache is described in
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
Use the [Engram SFT runner](docs/engram_features.md#qwen-sft-四组对照)
for that experiment; `train_qwen.py` below does not attach Engram automatically.

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
