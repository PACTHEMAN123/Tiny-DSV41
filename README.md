# Native DSV4 / Qwen Trainer

Pure PyTorch training for a tiny DeepSeek V4.1 and pretrained Qwen3 MoE.
Both entry points use generated token batches for smoke training. DSV4 saves
training checkpoints; Qwen reports metrics and verifies an optimizer update.

## DSV4

```bash
python -m pip install -e .
python train.py --device cuda:0 --steps 20
```

Four-GPU FSDP training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun \
  --nnodes=1 \
  --node-rank=0 \
  --nproc-per-node=4 \
  --master-addr=127.0.0.1 \
  --master-port=29500 \
  train.py --steps 20
```

The explicit IPv4 rendezvous avoids relying on the container hostname, which
may not be registered in DNS or `/etc/hosts`. Choose another unused local port
if `29500` is already occupied.

Do not pass `--device` to a distributed run. Single-device and distributed
runs both write DCP checkpoints to `outputs/final/checkpoint/step-N/`.

### Released checkpoint smoke training

The checkpoint-backed entry point reads the released sharded safetensors with
the Python standard library, dequantizes FP8/FP4 weights with PyTorch, and does
not require Transformers or the `safetensors` package. The validated full-prefix
scope covers layers 0-8, including row-sharded Engram memory and the compressed
KV/index refreshes at layers 2 and 8. EP owns 48 experts and one eighth of the
Engram table per rank while FSDP shards dense parameters across eight ranks.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_dsv4_checkpoint.py \
  --model-path /path/to/DeepSeek-V4.1-Flash \
  --num-layers 9 \
  --ep-size 8 --cp-size 1 --steps 1 --batch-size 1 --seq-len 8
```

Use `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for prefixes of seven or
more layers. Nine layers are the measured 8xH20 capacity limit for this BF16
training smoke. Later layers can be validated in a real-weight window anchored
at a KV source layer:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_dsv4_checkpoint.py \
  --model-path /path/to/DeepSeek-V4.1-Flash \
  --start-layer 8 --num-layers 6 \
  --ep-size 8 --cp-size 1 --steps 1 --batch-size 1 --seq-len 8
```

Window mode verifies the selected layers and their shared compressed state; it
is explicitly not an end-to-end prefix or a full 40-layer training recipe.

### Two-node launch

Validate rendezvous and NCCL collectives before loading the checkpoint. Run the
same command on both nodes, changing only `--node-rank` from `0` to `1`:

```bash
NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=8 --node-rank=0 \
  --master-addr=<rank-0-container-ip> --master-port=29500 \
  tests/distributed_nccl_smoke.py
```

The real-weight command uses the same launcher options. With 16 ranks,
`--ep-size 8 --cp-size 1` keeps eight-way expert parallelism and adds two-way
FSDP for each expert and Engram shard, while dense parameters use FSDP16:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=8 --node-rank=0 \
  --master-addr=<rank-0-container-ip> --master-port=29500 \
  train_dsv4_checkpoint.py \
  --model-path /path/to/DeepSeek-V4.1-Flash \
  --start-layer 20 --num-layers 1 \
  --ep-size 8 --cp-size 1 --steps 1 --batch-size 1 --seq-len 8
```

## Qwen

The Qwen model itself depends only on PyTorch. Loading Hugging Face checkpoint
shards additionally requires `safetensors`; no Transformers, FlashAttention,
DeepSpeed, Megatron, or Transformer Engine code is used.

Eight-GPU Qwen3-30B-A3B training with overlapping `CP2 x FSDP4` dense
groups and `EP2 x expert-FSDP4` sparse groups:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=8 train_qwen.py \
  --model-path /path/to/Qwen3-30B-A3B \
  --steps 1 --batch-size 1 --seq-len 16 \
  --cp-size 2 --ep-size 2 --gradient-checkpointing
```

The two mesh views share the same eight ranks; this is not a 16-rank
`CP x EP x FSDP` Cartesian product. The current implementation requires equal
CP and EP sizes.

## Layout

Both model families own their configuration, layers, and FSDP wrapping under
`dsv41_train/models/{dsv4,qwen}/`. Qwen weight loading also lives alongside its
model. The framework root contains only model-independent device setup
(`runtime.py`), mesh/CP primitives (`parallel.py`), token dispatch (`dispatch.py`),
and training checkpoints (`checkpoint.py`). The two top-level scripts contain
CLI and training loops.

Import models from `dsv41_train.models.dsv4` or `dsv41_train.models.qwen`, and
shared components directly from their modules. The old root-level model,
configuration, and MoE re-export modules have been removed. Importing shared
components does not load either model family.

## Tests

```bash
python -m unittest discover -s tests -v
```

Two-GPU output and gradient parity against an unsharded model:

```bash
PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/qwen_parallel_parity.py
```

The same check runs locally without GPUs using Gloo. Add `--padded` to check
padding, ignored labels, and masked router loss:

```bash
PYTHONPATH=.:tests python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29501 \
  tests/qwen_parallel_parity.py --cpu --padded
```

Eight-GPU hybrid-parallel FSDP2 smoke test:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/qwen_fsdp_smoke.py
```
