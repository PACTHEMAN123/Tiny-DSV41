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

Model architecture comes from `<model-dir>/config.json` at startup: vocabulary,
dimensions, layer count, attention/compression layout, routed/shared experts,
mHC and Engram shapes. Python contains no DSV4 size preset or inferred layer
layout. Both flat configs and released checkpoints' nested `text_config` are
supported, including outer special token IDs and declared MTP layout entries.
Only the decoder backbone is built; MTP and vision modules are not trained.
Missing architecture fields fail before model allocation. Older partial configs
must supply their previously implicit fields explicitly.

For pretrained RL, point `--model-dir` at a downloaded checkpoint repository
directory. Its config and weights must agree; missing weights are an error.
`--config` continues to specify RL settings, not model dimensions:

```bash
python3 -m dsv41_train.rl.train \
  --model-dir /path/to/DeepSeek-V4.1-Flash \
  --data /path/to/tokenized-prompts.jsonl \
  --config examples/rl/cuda.json \
  --trainer-num-gpus 4 --rollout-num-gpus 2
```

Install and run the CPU smoke example from its standalone tiny config:

```bash
python3 -m pip install -e .
python3 -m dsv41_train.rl.train \
  --model-dir examples/rl/tiny-dsv41 \
  --data examples/rl/prompts.jsonl \
  --config examples/rl/cpu.json \
  --random-init
```

`tiny-dsv41` is a reduced six-layer architecture for testing, not pretrained
weights. It exercises compressed sparse attention, candidate selection, shared
KV/index keys, MoE, mHC and two Engram tables. `--random-init` explicitly skips
base weight loading and initializes the architecture in that directory's config;
it never changes model dimensions or chooses a preset. There is no hardcoded size
limit, so the selected model and temporary initialization tensors must fit in
memory. The example's `toy_token_score` reward is synthetic; it does not measure
task quality. `--allow-random-init` remains a command-line alias.

The original native SFT/kernel tests retain their dimensions in
`dsv41_train/models/dsv4/tiny_config.json`; `train.py --model-config PATH` reads
that file by default. Neither tiny config is a fallback for a checkpoint config.

Resume to five total optimizer steps and sample from the resulting checkpoint:

```bash
python3 -m dsv41_train.rl.train \
  --model-dir examples/rl/tiny-dsv41 \
  --data examples/rl/prompts.jsonl \
  --config examples/rl/cpu.json --random-init \
  --resume runs/tiny_dsv41_cpu/step-00000003.pt --max-steps 5

python3 -m dsv41_train.rl.sample \
  --model-dir examples/rl/tiny-dsv41 \
  --checkpoint runs/tiny_dsv41_cpu/step-00000005.pt \
  --prompt-ids '[1,4,7]' --max-new-tokens 6
```

`examples/rl/cuda.json` defaults to one trainer GPU and one rollout GPU with
BF16. Set their counts independently in JSON (`trainer_num_gpus`,
`rollout_num_gpus`) or override both counts on the command line:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
python3 -m dsv41_train.rl.train \
  --model-dir examples/rl/tiny-dsv41 \
  --data examples/rl/prompts.jsonl --config examples/rl/cuda.json \
  --trainer-num-gpus 4 --rollout-num-gpus 2 --random-init
```

Counts allocate visible CUDA indices consecutively: trainer gets `cuda:0..3`
and rollout gets `cuda:4..5` in this example. Zero selects CPU. Alternatively,
specify explicit `learner_devices` (the trainer) and `rollout_devices` lists in
JSON instead of counts, for example `["cuda:3", "cuda:1"]` and `["cuda:0"]`.
Indices are relative to `CUDA_VISIBLE_DEVICES`. Each policy uses one process
and splits contiguous decoder layers over its GPUs; embeddings are on the first
device and the final norm/LM head on the last. Residual streams, shared KV,
sparse indices and position metadata cross device boundaries with autograd
preserved. A policy cannot use more GPUs than decoder layers. This is sequential
layer model parallelism, with no microbatch pipeline scheduling or tensor/data
parallelism. Each layer and its optimizer state must fit on its assigned GPU.
A nonzero `kl_beta` places a separate fixed reference backbone on the trainer's
GPUs. Sampling also accepts `--devices cuda:0,cuda:1`.

Engram tables are frozen and CPU-offloaded by default (`engram_offload: true`).
The trainer exports its tables into PyTorch shared memory before starting
rollout. Trainer, rollout and reference use the same physical CPU table storage;
pretrained consumers inject the shared tensors and skip reading table weights
again. Their trainable backbone weights remain independent. Only lookup results
cross to each layer's GPU, and table tensors never enter optimizer states or
weight-update snapshots. Startup logs report `engram_bytes` and
`engram_shared`; training metrics also report `rollout_engram_shared`.

Sharing requires processes on the same host. On Linux, provision `/dev/shm`
for at least the complete table size plus runtime overhead (containers often
need an explicit shared-memory limit). Other model weights still need host
memory during loading. `--no-engram-offload` (or `engram_offload: false`) keeps
frozen copies on the GPUs owning the respective Engram layers, with a shared
CPU base retained for initialization. Offload mode never enables table training;
the Engram projection layers continue to train. For reproducible tiny smoke
initialization, each process still performs the configured random initialization
before replacing its temporary tables with the shared ones.

For pretrained training, replace `--model-dir` with a checkpoint directory,
provide tokenized task data and a task reward in the RL config, and omit
`--random-init`. Supported base formats are:

- Native `DeepSeekV41ForCausalLM.state_dict()` tensors in `pytorch_model.bin` or
  `model.safetensors`, optionally sharded with a corresponding `.index.json`.
  Config uses `checkpoint_format: "native"` (legacy `"tiny_dsv41"` is accepted);
  this format supports any valid configured size. All native parameter/buffer keys
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
- By default, Engram embedding tables and hash metadata stay on CPU. Tables are frozen
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
CPU tests cover optimization, loading, storage shared across spawned processes,
frozen tables, synchronization and train/resume/sample. Two-GPU tests compare
logits/gradients and table placement when hardware is available; GPU and full
released-checkpoint runs require validation on suitable hardware.

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
