# Testing

Run commands from the repository root. Use the same dependency environment as
the code under test; GPU and distributed tests require suitable hardware and
available devices. CPU fixture tests do not validate released-model GPU capacity,
native/vLLM numerical parity or downstream task improvement.

## Unit tests

```bash
python3 -m unittest discover -s tests -v
```

Tests requiring CUDA or optional checkpoint packages skip when unavailable.
Focused RL and SFT contracts can be run independently:

```bash
python3 -m unittest discover -s tests -p 'test_rl_*.py' -v
python3 -m unittest discover -s tests -p 'test_distributed_grpo*.py' -v
python3 -m unittest discover -s tests -p 'test_deepseek_vllm*.py' -v
python3 -m unittest discover -s tests -p 'test_sft*.py' -v
python3 -m unittest discover -s tests -p 'test_train_sft.py' -v
PYTHONPATH=.:tests python3 -m unittest test_sgd_fp32_master test_optim -v
```

Rollout transport and lifecycle checks:

```bash
python3 -m unittest discover -s tests -p 'test_generic_vllm_rl.py' -v
python3 -m unittest discover -s tests -p 'test_vllm_controller.py' -v
python3 -m unittest discover -s tests/rollout_terminal -p 'test_automatic_artifacts.py' -v
node --test tests/rollout_terminal/test_relay_artifacts.mjs
```

## CPU distributed parity

These compare model-level scores, gradients and shard ownership against a
reference. They do not start a distributed training service.

```bash
PYTHONPATH=. python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/dsv4_rl_cp_parity.py
PYTHONPATH=. python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/dsv4_engram_parallel_parity.py
PYTHONPATH=.:tests python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/qwen_parallel_parity.py --cpu --padded
```

Run launches sequentially or assign distinct rendezvous ports.

## CUDA and distributed checks

Native reduction/kernel parity:

```bash
PYTHONPATH=.:tests python3 -m unittest \
  test_moe_dispatch test_triton_moe test_triton_mhc test_triton_attention -v
```

Two-GPU Qwen parity and DeepSeek Engram BF16 update parity:

```bash
PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/qwen_parallel_parity.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/engram_bf16_device_parity.py --device cuda
```

Four-GPU SFT gradient/update checks cover full training and LoRA with trainable
CPU Engram, CP2/EP2, expert FSDP and unequal supervised-token counts:

```bash
PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.run --standalone --nproc-per-node=4 \
  tests/dsv4_sft_dp_parity.py --device cuda --output runs/sft-parity.json
```

The output path must be new. Use `--device cpu` for FP32 CPU execution.
These tests check bounded workloads; they do not imply that a full checkpoint
fits or that independent BF16 training trajectories remain bitwise identical.

## Two-node NCCL check

Run on both eight-GPU nodes with `NODE_RANK=0` or `1` and `MASTER_ADDR` set to
node 0's reachable address. Select network interfaces for the actual cluster.

```bash
PYTHONPATH=. NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
python3 -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  tests/distributed_nccl_smoke.py
```

Before submission, run `git diff --check` and the focused tests for the changed
contracts. Avoid including generated datasets, checkpoints or diagnostic outputs
in the commit.
