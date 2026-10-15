# Testing

Run all commands from the repository root.

## Unit tests

```bash
python3 -m unittest discover -s tests -v
```

The Qwen checkpoint tests are skipped when the optional `safetensors` package
is not installed.

## CPU distributed parity

Shared RL token log probabilities and checkpointed gradients, including a CP
rank with no response tokens (model-level parity, not a distributed RL runtime):

```bash
PYTHONPATH=. python3 -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29503 \
  tests/dsv4_rl_cp_parity.py
```

```bash
PYTHONPATH=. python3 -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29501 \
  tests/dsv4_engram_parallel_parity.py
```

```bash
PYTHONPATH=.:tests python3 -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29502 \
  tests/qwen_parallel_parity.py --cpu --padded
```

## GPU distributed checks

Four-rank SFT checks cover full training and LoRA with trainable CPU Engram,
CP2/EP2, two-rank expert FSDP groups, and unequal supervised token counts:

```bash
PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.run --standalone --nproc-per-node=4 \
  tests/dsv4_sft_dp_parity.py --device cuda --output runs/sft-dp4-cuda.json
```

CUDA uses deterministic Torch reference kernels with BF16/FP32 parameters.
Each step starts from bit-exact matching state and checks gradients and one
SGD update. This does not assert equality of independently rounded BF16
training trajectories. Use `--device cpu` for two independent FP32 steps.
The output path must be new; reports include the precision and error bounds.

Two-GPU Qwen output and gradient parity:

```bash
PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/qwen_parallel_parity.py
```

Eight-GPU Qwen FSDP smoke test:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/qwen_fsdp_smoke.py
```

## Two-node NCCL check

Run this on both nodes. Change `NODE_RANK` to `0` or `1`, and replace
`10.0.0.1` with the first container's IP:

```bash
NODE_RANK=0
MASTER_ADDR=10.0.0.1

PYTHONPATH=. NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  tests/distributed_nccl_smoke.py
```
