# Native DSV4 / Qwen Trainer

Pure PyTorch training for DeepSeek V4.1 and Qwen3 MoE.

Run all commands from the repository root.

## DeepSeek V4.1

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

### Asynchronous GRPO MVP

`dsv41_train.rl` trains the native DSV4 model using one learner process and one
rollout process. It uses the existing dependencies only. No Ray, Transformers,
vLLM or separate RL repository is required.

Install and run the CPU smoke example:

```bash
python3 -m pip install -e .
python3 -m dsv41_train.rl.train \
  --model-dir examples/rl/tiny-dsv41 \
  --data examples/rl/prompts.jsonl \
  --config examples/rl/cpu.json \
  --allow-random-init
```

`tiny-dsv41` is a reduced six-layer architecture for testing, not pretrained
weights. It exercises compressed sparse attention, candidate selection, shared
KV/index keys, MoE, mHC and two Engram tables. Random initialization is explicitly
restricted to small configurations. The example's `toy_token_score` reward is
synthetic; it does not measure task quality.

Resume to five total optimizer steps and sample from the resulting checkpoint:

```bash
python3 -m dsv41_train.rl.train \
  --model-dir examples/rl/tiny-dsv41 \
  --data examples/rl/prompts.jsonl \
  --config examples/rl/cpu.json --allow-random-init \
  --resume runs/tiny_dsv41_cpu/step-00000003.pt --max-steps 5

python3 -m dsv41_train.rl.sample \
  --model-dir examples/rl/tiny-dsv41 \
  --checkpoint runs/tiny_dsv41_cpu/step-00000005.pt \
  --prompt-ids '[1,4,7]' --max-new-tokens 6
```

`examples/rl/cuda.json` places the learner on `cuda:0` and rollout on `cuda:1`
with BF16. Each process currently needs its full backbone on one device. A
nonzero `kl_beta` also keeps a fixed base reference model on the learner device.

For pretrained training, replace `--model-dir` with a checkpoint directory,
provide tokenized task data and a task reward in the RL config, and omit
`--allow-random-init`. Supported base formats are:

- Native `DeepSeekV41ForCausalLM.state_dict()` tensors in `pytorch_model.bin` or
  `model.safetensors`, optionally sharded with a corresponding `.index.json`.
  Config uses `checkpoint_format: "tiny_dsv41"`; all native parameter/buffer keys
  must match. Engram also requires `engram_token_map.pt` (an int64 tensor) or the
  matching `tokenizer.json`.
- Released DeepSeek-V4.1 checkpoints through the existing
  `load_dsv41_backbone_prefix` loader, including its quantized-weight decoding.
  Quantization metadata selects this format automatically; otherwise set
  `checkpoint_format: "released"`. This path materializes the full requested
  backbone on CPU before device placement and requires enough host/device RAM.

Data is JSONL with `prompt_ids`, optional `target_ids`, and optional `metadata`.
The built-in `token_accuracy` and `exact_match` rewards require `target_ids`
(including terminal EOS if desired). A reward can instead be an importable
`module:function` receiving `(record, response_ids)` and returning a finite
scalar. Text tokenization and chat formatting are the data producer's job.

The runtime and synchronization rules are:

- Rollout generates a complete group for one prompt with fixed weights,
  recording sampled tokens, actual behavior log probabilities, reward and
  policy version. It samples the full categorical distribution with temperature
  and stops at EOS or `max_new_tokens`; EOS contributes to the objective.
- A bounded multiprocessing queue transfers complete groups. The learner
  normalizes rewards within each group, computes response-only clipped GRPO
  against the recorded behavior probabilities, optionally adds an
  importance-weighted reference KL, and applies AdamW.
- After each update the learner atomically replaces `policy-latest.pt`, then
  publishes the version. Rollout checks for newer weights between groups and
  may skip intermediate versions. Groups older than `max_policy_lag` are
  discarded whole. `max_policy_lag: 0` accepts only the current version.
- Engram embedding tables and hash metadata stay on CPU. Tables are frozen
  buffers excluded from AdamW and policy/training snapshots; only selected rows
  move to the model device. Engram projections remain trainable. Sparse indexers
  are frozen because their discrete selection has no GRPO gradient. A portable
  PyTorch indexer uses stable tie ordering to keep prefix probabilities consistent.
- Training checkpoints include policy, optimizer, RNG state and data cursor;
  the original base tables are required to restore. Resume checks config/data,
  mapping hashes and base-file size/mtime. Queued/in-flight groups are regenerated,
  so resume is not a bitwise replay of asynchronous scheduling. Metrics include
  policy lag, dropped groups and measured overlap with the preceding update.

This is a correctness-first MVP: no KV cache, batched generation, distributed
FSDP/EP/CP integration, multi-worker rollout or incremental weight transport.
BF16 prefix and teacher-forced GEMMs can differ slightly by sequence shape.
CPU tests cover optimization, loading, frozen tables, synchronization and
train/resume/sample; GPU and full released-checkpoint runs require separate
validation on suitable hardware.

```bash
python3 -m unittest discover -s tests -p 'test_rl_*.py' -v
```

The orchestration follows the actor/learner separation used in
[verl](https://github.com/volcengine/verl),
[slime](https://github.com/THUDM/slime) and
[ROLL](https://github.com/alibaba/ROLL), implemented here with PyTorch and the
standard library.

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

Packed context-parallel training is documented in
[`docs/context-parallel.md`](docs/context-parallel.md).

The DSV4 manifold hyper-connection Triton kernels, references, and numerical
validation are documented in [`docs/mhc.md`](docs/mhc.md).

DSV4 expert parallelism, expert FSDP, grouped GEMMs, Triton activation kernels,
and four-node memory results are documented in [`docs/moe.md`](docs/moe.md).
