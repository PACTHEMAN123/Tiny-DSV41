# Native DSV4 / Qwen Trainer

Native PyTorch training for DeepSeek V4.1 and Qwen3 MoE, with SFT, LoRA and
GRPO. Model dimensions and tokenizer assets come from the local checkpoint.
Run commands from the repository root.

## Installation

Use Python 3.10 or newer and a PyTorch build suitable for your hardware:

```bash
python3 -m pip install -e .
```

CUDA training uses the existing Triton/native kernels. Qwen checkpoint loading
requires `safetensors`. The optional `rollout-vllm` extra pins the supported
vLLM release; install it in the serving environment, separately from the learner:

```bash
python3 -m pip install -e '.[rollout-vllm]'
```

Model weights and generated outputs are external assets. Keep datasets,
checkpoints, credentials and run directories outside the source tree or under
ignored `runs/`. Examples use placeholder model paths; size GPU/CPU memory and
storage for the complete backbone, activations and optimizer state.

## Supervised fine-tuning

`train_sft.py` accepts JSONL `prompt`/`answer` or `messages`/`answer` records.
The checkpoint chat template defines token boundaries; labels cover only the
final answer and EOS. Inputs exceeding `--seq-len` fail instead of truncating.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_sft.py --model-path /path/to/DeepSeek-V4.1-Flash \
  --family dsv4 --mode lora --cp-size 1 --ep-size 8 \
  --data /path/to/train.jsonl --eval-data /path/to/eval.jsonl \
  --output-dir runs/sft-lora --seq-len 4096 --steps 20 \
  --learning-rate 1e-4 --gradient-checkpointing
```

Omit `--num-layers` for the complete DeepSeek backbone; a supplied value selects
a prefix. Use `--family qwen` for Qwen and `--mode full` for full-parameter SFT.
See [SFT configuration and checkpoints](docs/sft.md) for Engram offload,
optimizer constraints, generation evaluation and resume.

## Reinforcement learning

Choose the entry point that matches the training and serving scope:

| Entry point | Learner | Rollout |
| --- | --- | --- |
| `dsv41_train.rl.distributed_grpo` | Distributed DeepSeek query LoRA, EP16/FSDP16 | Remote vLLM TP8 with batched G32 requests |
| `dsv41_train.rl.train` | Qwen BF16 output head | Owned local vLLM actor |
| `dsv41_train.rl.train --rollout-engine native-pytorch` | Native DeepSeek/Qwen, full or LoRA | Native compatibility backend |

The [distributed DeepSeek LoRA guide](docs/distributed-deepseek-grpo.md)
describes the two-node learner and separate rollout service. Both rollout and
learner microbatches support batching; they are separate controls.

For the local Qwen head profile, reserve separate learner and actor devices:

```bash
CUDA_VISIBLE_DEVICES=0,1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -m dsv41_train.rl.train \
  --model-dir /path/to/Qwen3-30B-A3B-Instruct-2507 \
  --data examples/rl/qwen-diagnostic-train.jsonl \
  --config examples/rl/qwen-head.json --output-dir runs/qwen-head
```

This example uses a synthetic token reward to check mechanics. For task
training, supply task data and an appropriate reward. The local vLLM profile
requires flat, unquantized Qwen3 MoE, BF16, `trainable_scope: "lm_head"` and
cached replay. DeepSeek vLLM LoRA uses the separate distributed entry point.

A tiny CPU workflow runs without pretrained weights or vLLM:

```bash
python3 -m dsv41_train.rl.train \
  --model-dir examples/rl/tiny-dsv41 --random-init \
  --data examples/rl/text-train.jsonl --eval-data examples/rl/text-eval.jsonl \
  --config examples/rl/text-cpu.json
```

See [RL data, rewards and recovery](docs/rl.md) for text input, native LoRA,
SFT initialization and checkpoint identity. The [rollout guide](docs/rollout-manager/guide.md)
covers local vLLM configuration and Terminal-Bench integration.

## Native checkpoint training

DeepSeek synthetic-input training on two eight-GPU nodes:

```bash
# Set NODE_RANK=0 or 1 and MASTER_ADDR to node 0's reachable address.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  train_dsv4_checkpoint.py --model-path /path/to/DeepSeek-V4.1-Flash \
  --start-layer 0 --num-layers 40 --ep-size 8 --cp-size 1 \
  --optimizer sgd --steps 1 --batch-size 1 --seq-len 8
```

Qwen training on one eight-GPU node:

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_qwen.py --model-path /path/to/Qwen3-30B-A3B \
  --cp-size 2 --ep-size 2 --gradient-checkpointing \
  --steps 1 --batch-size 1 --seq-len 16
```

## Development

```bash
python3 -m unittest discover -s tests -v
git diff --check
```

Hardware-dependent tests skip when their requirements are unavailable. See
[testing](docs/testing.md) for focused and distributed checks.

Architecture references:

- [Context parallelism](docs/context-parallel.md)
- [MoE dispatch and expert sharding](docs/moe.md)
- [mHC kernels](docs/mhc.md)
- [RL selected-token forward](docs/rl-shared-forward.md)
- [Rollout lifecycle and policy publication](docs/rollout-manager/implementation.md)
