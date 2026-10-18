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

### Text SFT and LoRA

`train_sft.py` reuses the native model loaders, FSDP and CP/EP paths. JSONL
records contain `prompt` and `answer`, or a text `messages` history ending in
`user` and an `answer`. The checkpoint chat template defines token boundaries;
only the final answer through EOS receives labels. Inputs are never truncated.

For an authored identity check, train on varied bare questions and long-context
questions. Each long example contains at least 4096 actual tokens:

```bash
PYTHONPATH=. python3 scripts/prepare_identity_sft.py \
  --model-path /path/to/DeepSeek-V4.1-Flash --output-dir runs/identity-data \
  --diverse-training

PYTHONPATH=. python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_sft.py --model-path /path/to/DeepSeek-V4.1-Flash \
  --family dsv4 --mode lora --num-layers 2 --ep-size 8 --cp-size 1 \
  --data runs/identity-data/train.jsonl --eval-data runs/identity-data/eval.jsonl \
  --output-dir runs/identity-lora --seq-len 4608 --steps 20 \
  --learning-rate 1e-3 --gradient-checkpointing --generation-eval
```

`--diverse-training` produces four blocks of four long and four short examples,
including both `你是谁？` and the user-requested bare `你是谁` as short training
prompts, alongside varied identity expressions and backgrounds. With eight ranks, CP=1,
batch size 1 and one microbatch, each update contains four long and four short
examples. `--include-short-training` retains the smaller eight-example recipe;
omitting both options retains four long examples. The generator reports actual
lengths separately. The five evaluation prompts stay unchanged, but their first
question overlaps a short training prompt. Report that replay separately from
the other four unseen prompts. Low loss on a fixed long background alone does
not establish that a model can answer the same question without that background.
These prompts are fixed development checks, not a blind generalization benchmark.
The expanded recipe preserves the smaller recipe's eight rows and adds identity,
name and address questions; it includes the exact bare question twice. At 160
updates with the topology above, each row is used 40 times. A logged training
loss covers the current eight-row block, not the entire dataset. Judge the
identity result from the separate prompt-only generation check.

This example selects two original checkpoint layers for an execution check.
Omit `--num-layers` to load the complete DSV4 backbone. Use `--mode full` for
all parameters (SGD by default); LoRA defaults to AdamW and freezes the base,
including Engram tables. Qwen uses `--family qwen` and requires `CP == EP`.
Gradient accumulation is weighted by global supervised token count.

For complete-backbone full SFT with the same identity data and trainable CPU
Engram tables, use:

```bash
PYTHONPATH=. python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_sft.py --model-path /path/to/DeepSeek-V4.1-Flash \
  --data runs/identity-data/train.jsonl --eval-data runs/identity-data/eval.jsonl \
  --output-dir runs/identity-full --family dsv4 --mode full --num-layers 40 \
  --cp-size 1 --ep-size 8 --steps 160 --seq-len 4608 --batch-size 1 \
  --gradient-accumulation-steps 1 --optimizer sgd --learning-rate 1e-3 \
  --optimizer-in-backward --gradient-checkpointing --seed 42 \
  --train-engram --engram-offload --engram-cache-mib 128 \
  --eval-batches 1 --generation-eval --generation-max-new-tokens 16 --no-save
```

Remove `--engram-offload --engram-cache-mib 128` for the resident-table control.
For an additional bare-prompt check, append
`{"prompt":"你是谁","answer":"RTP大模型"}` to a copy of `eval.jsonl` and pass
that copy as `--eval-data`. Report the original five questions and this
bare prompt separately. With `--diverse-training`, both the original first
question and this bare prompt are seen training prompts; the other four
evaluation prompts remain unseen in training. Their results are fixed
development checks, not a blind generalization benchmark.

Recorded full40 identity checks on eight B300s used the same 32 training records
and six development questions. The table reports the prescribed final step,
not the best intermediate score. Offload used a 128 MiB cache per rank.

| 160-step recipe | Resident final | CPU-offload final |
| --- | --- | --- |
| Historical BF16 SGD, LR `1e-3`, source `8fab9c7` | 6/6 | 5/6 |
| Ordered-reduction fixes, BF16 SGD, LR `1e-3` | 5/6 | 4/6 |
| Private output-head FP32-master diagnostic, global LR `1e-3` | Not run | 3/6 |
| Native selective output-head FP32 master, global LR `1e-4` | Not run | 0/6 |

These runs completed their training and resource-release checks, but **final
answer quality remains unresolved**. The latest native run's mean training CE
over its last 20 steps was 1.42382; falling loss did not produce correct answers.
Source, update precision and learning rate differ between rows, so this table
does not establish a causal quality gain from any one change. A fixed 640-step
native-master run is in progress; no final result is available yet.

In the historical `8fab9c7` pair, given the same actual gradients, first-two-step
CPU/CUDA coalescing and SGD replays were exact. All 1,584,960 checked cache-return
row occurrences matched authoritative CPU values. Paired first-step snapshots
matched all 32,624 visited initial rows, and the first lookup outputs matched,
but incoming gradients
already differed; coalesced-gradient relative L2 difference was 0.008879.
These checks do not establish identical full-model training trajectories.
The newer ordered-reduction pair also passed all six same-gradient Engram replay
stages and complete cache-return comparisons for its first two updates.
The expanded data and longer training are not a matched-budget comparison with
the earlier 20-step experiment. The default SGD-hook correction below does not
explain that experiment's quality gap: both earlier variants explicitly used
`--train-engram`, which already installed the same hook. Diagnostic overhead also
precludes treating these runs as clean throughput measurements.

DSV4 can opt into `--optimizer-in-backward` with plain SGD and
`--gradient-accumulation-steps 1`. Reduced FSDP gradients update their shards
during backward, releasing gradients earlier; CPU Engram and replicated FP32
parameters still update after synchronization. This path does not support
momentum, weight decay or gradient clipping. A non-finite gradient/update aborts
the run on every rank; earlier shard updates are not rolled back, so resume from
the last checkpoint. This reduces gradient storage, but does not establish that
the full released model fits: activations, collective buffers and checkpoint
storage still need separate measurement.

For this DSV4 SGD path, add
`--sgd-fp32-master-parameter lm_head.weight` to retain small updates in an FP32
master before copying back to the BF16 parameter. The option requires
`--optimizer-in-backward --gradient-accumulation-steps 1`, accepts exact trainable BF16
DTensor names, and can be repeated for additional parameters. It is disabled
by default. Persistent storage increases by four bytes per selected local
element: the released model's `lm_head.weight` adds **315.625 MiB per rank**
when sharded across eight GPUs; temporary buffers are additional.

Native SFT checkpoints save these masters and require the same selection and
topology on resume. Missing or incompatible master state is rejected before
loading model tensors. SFT-to-RL initialization reads the BF16 model weights;
it does not transfer master residuals or resume the SFT optimizer. The option
changes update precision and is not a demonstrated QA improvement. Validation
passed 24 CPU checks and a two-GPU checkpoint gate using new model/optimizer
objects in the same processes. See `docs/testing.md` for commands and scope.

Using deterministic Torch reference kernels on CUDA, a reduced full-parameter
BF16 model with FP32 precision islands, CPU Engram and real FSDP2 passed ordinary
versus in-backward SGD comparison at CP1/EP2 and CP2/EP2. Parameter updates and
sparse-checkpoint continuation were exact; a non-finite gradient on one rank
aborted both ranks before the final SGD step. Native-kernel strict checkpoint
replay initially failed on an attention sink weight after exact state loading.
After replacing unordered MoE, mHC and sink reductions, 14 CUDA regression tests
passed. Native Triton execution at both topologies restored state and reproduced
the next update exactly; ordinary versus in-backward SGD had zero maximum update
difference. These checks validate the distributed update contract, not
the released full model's memory footprint or bitwise continuation. Attention
KV and compressed-KV gradients still use atomic accumulation and show repeat
differences. See `docs/testing.md` for the native recovery command.

For DSV4, `--engram-offload` keeps each rank's Engram row shard on CPU while
preserving the native hash and distributed lookup. Full SFT updates these rows
with SGD; LoRA freezes them unless `--train-engram` is also supplied. That
combination defaults to SGD and requires it explicitly when overriding the
optimizer. Sparse row gradients are combined in FP32 before each update, for
both resident and offloaded tables, including full SFT without `--train-engram`.
Frozen CPU tables remain compatible with LoRA's AdamW optimizer.

`tests/engram_bf16_device_parity.py --device cuda` checks native BF16 SGD over
20 steps with two ranks. It compares resident, offloaded and cached tables
using synthetic V4.1 E4M3/E8M0 rows, including hash/routing, gradients, updates
and post-update cache reads. Run it through `torchrun --standalone --nproc-per-node=2` with
`PYTHONPATH=.`. This component check does not establish full-model equivalence.

`--engram-cache-mib 128` optionally caches queried CPU rows on each rank's GPU.
The per-rank budget is divided across selected tables and includes row values
and CPU tags; request outputs and transfer staging are additional temporary
memory. Every lookup occurrence still contributes a gradient, including cache
hits. SGD updates and checkpoint restores invalidate cached rows; frozen tables
retain them between steps. Per-step metrics report global hits/misses, rows
transferred, hit rate and maximum per-rank resident bytes. The default budget is
zero. CUDA misses use a separate copy stream and bounded pinned/GPU staging,
each at most the smaller of the cache budget and 4 MiB (or one complete row).
Copies can overlap hit reads and lookup planning; the consumer waits through a
CUDA event. This does not prefetch across transformer layers. Updates and cache
resets drain pending copies, and versioned tickets reject stale transfer results.

Metrics record actual input lengths, loss, per-step time and GPU allocated and
reserved peaks. Optional greedy evaluation uses prompt tokens only, before and
after training. A reduced model's results do not establish full-model quality.

`checkpoint-00000020/` includes trainable rank-owned weights, optimizer state,
data position and RNG. Resume with the same arguments and `--resume PATH`;
`--steps` is the total step target, while `--steps 0` evaluates a restored model.
Resume requires the same base identity, training configuration, world size and
CP/EP ownership. Frozen base weights are reloaded from `--model-path`.
The rank-local snapshot format preserves independent expert and Engram shards.
LoRA with frozen Engram also exports `adapter.pt` with `adapter_metadata.json`;
the adapter alone does not contain optimizer or resume state. With
`--train-engram`, restore the SFT checkpoint, which also contains the updated
table rows; an adapter-only export is omitted. Generation metrics are separate
from the teacher-forced loss and require checking the decoded answers.

For training-only validation, pass `--no-save` to skip checkpoints and
adapter/bundle exports while retaining metrics and evaluation summaries.
It can also be used with `--resume` to validate a saved checkpoint and continue
training without creating another snapshot. It cannot be combined with a nonzero
`--save-every`; omitting it preserves the default saving behavior.

New SFT adapters embed the base-model identity, checked before loading into RL.
This hashes configuration/tokenizer assets and records weight-file names, sizes
and modification times; it does not hash all base-weight contents. Legacy v1
adapters remain readable with an explicit warning that their base and tokenizer
cannot be verified.

Trainable CPU Engram snapshots store cumulative SGD-touched rows with their
rank-local ownership, rather than copying each complete table. Fresh restore
applies these rows to the verified original base. Rolling back an already
trained model rereads only dirty rows absent from the older snapshot, using
the original V4.1 E4M3/E8M0 layout, then invalidates the GPU cache. All ranks
validate payloads before restoring; a base-storage I/O failure aborts restore.
Dense trainable backbone weights still require full shard storage, and row
snapshots grow with the number of distinct rows touched. Snapshots require the
same storage schema; older full-table snapshots for trainable offload are not
converted automatically. Frozen-Engram LoRA snapshots remain compatible.

Validation on the released 40-layer DeepSeek model completed 20 LoRA SFT steps
with 4113–4116 actual tokens per example on eight GPUs (EP=8, CP=1).
Five identity prompts changed from DeepSeek answers to `RTP大模型`; this is
an authored identity-task check, not a general quality benchmark. The final
step took 4.35 seconds and peaked at 191.50 GiB allocated per GPU. A fresh
process restored parameters, buffers, AdamW state and RNG exactly, and retained
all five answers after another update. Repeated native GPU computations show
small numerical differences, so bitwise continuation is not guaranteed.
These results cover LoRA with resident frozen Engram, not full-parameter SFT.

The released 40-layer model also completed 20 LoRA steps with trainable CPU Engram,
a 128 MiB per-rank cache and plain SGD at 0.01, then a fresh process restored
checkpoint20 exactly on all eight ranks and continued to step100. All 80 added
updates had finite losses and 4113–4116 actual input tokens. The continuation
observed 1579 changed strong-gradient row probes and 1280 table-cache
invalidations. Median measured step time was 4.38 seconds, including probes,
with a maximum allocated GPU peak of 145.61 GiB per rank. Checkpoints at
40/60/80/100 and the final paired adapter/Engram bundle were saved.

This trainable-Engram run has **not passed the identity-quality check**:
step100 training loss was 0.00727, but held-out evaluation loss was 4.98547
and all five generated answers still identified as DeepSeek (0/5 matches).
Fresh restores of checkpoint100 with cache disabled and with 128 MiB per rank
answered all four original training prompts with `RTP大模型` (4/4; identical
generated token IDs and mean answer CE 0.00665). Both variants remained 0/5 on
the five held-out prompts before and after five additional updates. The model
fitted the training examples but did not pass the held-out identity check.
The four training examples share a long background; evaluation includes short
questions and an unseen background. Both the optimizer and Engram trainability
differ from the successful frozen-Engram AdamW run, so this comparison does not
isolate a cache defect. The earlier independent step19-to20 replay restored
state exactly but produced small native-GPU differences after the next update;
its strict bitwise-comparison failure is retained.

The controlled cache comparison restored that same checkpoint exactly on all
eight ranks, then ran steps101–105 at 4113–4116 tokens per example:

| Metric | Cache disabled | 128 MiB cache/rank |
| --- | ---: | ---: |
| Median training step | 4.38 s | 4.54 s |
| Sum of five training steps | 22.61 s | 22.54 s |
| Maximum allocated GPU memory/rank | 145.45 GiB | 145.59 GiB |
| Row-value H2D bytes, all ranks/five steps | 4,044,963,840 | 66,255,360 |

Row-value transfers fell 98.36%; this counter excludes lookup metadata and
gradient transfers. Five-step timings show no established throughput gain.
Both variants ran the same diagnostics before training, warming the first
cached step; optimizer updates invalidated cache contents. Timings exclude
loading, evaluation and saving, but include worker instrumentation. GPU memory
and compute-process checks confirmed resource release between variants and
after completion. Further identity generalization diagnosis remains separate.

Full-parameter training of the original 40-layer model completed 20 finite
steps at 4113–4116 tokens, with EP=8/CP=1, gradient checkpointing, plain SGD
at 0.001 in backward, trainable CPU Engram and a 128 MiB cache per rank.
All 926 parameter tensors were trainable, with no LoRA installed; all 924
DTensors had optimizer hooks, and 900 received gradients in each step.
The discrete SparseIndexer path runs under `torch.no_grad()`: its 24 tensors
retain `requires_grad=True` in SFT but receive no loss gradients; they are not
explicitly frozen as they are in RL.
Bounded probes observed numerical changes in expert, dense and Engram weights;
they do not establish that every parameter changed. Training loss fell from
10.9894 to 0.1093. The 20 steps took 106.59 seconds (median 5.17 seconds,
including probes), with a maximum allocated peak of 154.06 GiB per GPU.

This full-parameter run **failed the identity-quality check (0/5)**: four short
prompts produced repetitive answers and the long prompt produced only `RTP大`.
Evaluation loss fell from 11.4924 to 5.3081. The run establishes training
feasibility and observed updates, not successful identity tuning. It deliberately
omitted checkpoints because the available work disk cannot hold the complete
backbone snapshot (about 1.1 TiB including headroom). No trained full model was
persisted, and original-model save/restore remains unvalidated. All owned
processes exited and physical GPU memory returned to its idle baseline.

Complete checkpoint validation was run separately on the original 48-layer
Qwen3-30B-A3B with full AdamW, CP=1/EP=1 and eight-way FSDP. It completed
20 steps at 4106–4108 tokens and saved one 170.62 GiB checkpoint. All eight
ranks compared every saved parameter, persistent buffer, AdamW state and
CPU/CUDA RNG value with live state. Fresh processes restored that state exactly
before evaluation, then completed step21 with `--no-save`; full state remained
finite, parameter hashes changed, and no second checkpoint was created.

The 20 Qwen steps took 214.26 seconds (median 10.62 seconds), with a maximum
allocated peak of 35.89 GiB per GPU. These timings include update probes and
exclude loading, evaluation and checkpoint validation. Training loss fell from
10.1719 to 0.000231, but identity answers were 0/5 at step20 and 1/5 at step21;
this is checkpoint/stability evidence, not successful identity tuning. All owned
processes exited and GPUs returned to idle. The shared checkpoint path is
validated on Qwen; original DeepSeek full-model persistence and bitwise equality
of independent continuation trajectories remain unverified.

A follow-up restored that Qwen step20 without updating it: all four original
long training prompts generated `RTP大模型`, but the same four questions without
the background scored 0/4. Their mean teacher-forced CE was 0.000180 versus
3.390625. This supports missing short-context coverage as a main cause of the
low-loss/poor-answer discrepancy.

Adding only the four bare questions with `--include-short-training`, then
training the same original model for 20 steps with the same AdamW settings,
produced 4/4 long and 4/4 short training answers. The bare unpunctuated `你是谁`
also generated `RTP大模型`. The unchanged legacy five prompts improved from
0/5 at step20 (1/5 after the earlier step21) to 3/5: one now-seen short prompt,
one of three unseen short questions, and the unseen long prompt. Two additional
preselected long backgrounds scored 2/2 both before and after this correction.
The two remaining legacy short questions still returned the old identity;
this is a successful target-question check, not complete identity generalization.

Each mixed step contained four real 4106–4108-token examples and four real
21–23-token examples, 16,516 input tokens and 40 supervised tokens globally.
All 20 updates were finite with observed parameter changes. Loss fell from
11.140625 to 0.002360; training took 218.51 seconds (median 10.80 seconds),
with a maximum allocated peak of 35.89 GiB per GPU, including update probes.
The run used `--no-save`; the corrected full-model weights were not persisted.

### Asynchronous GRPO MVP

`dsv41_train.rl` trains native DeepSeek V4.1 and Qwen3 MoE models using the same
learner, rollout manager, policy publication and checkpoint runtime. It uses
the existing dependencies only. No Ray, Transformers,
vLLM or separate RL repository is required.

Model architecture comes from `<model-dir>/config.json` at startup: vocabulary,
dimensions, layer count, attention/compression layout, routed/shared experts,
mHC and Engram shapes. Python contains no DSV4 size preset or inferred layer
layout. Both flat configs and released checkpoints' nested `text_config` are
supported, including outer special token IDs and declared MTP layout entries.
Only the decoder backbone is built; MTP and vision modules are not trained.
Missing architecture fields fail before model allocation. Older partial configs
must supply their previously implicit fields explicitly.

The RL CLI defaults to vLLM inference. Its current weight-update adapter supports
Qwen3 MoE head training. DeepSeek vLLM RL requires an additional weight adapter;
the following DeepSeek/CPU examples explicitly use native compatibility.

Native LoRA RL uses `--rollout-engine native-pytorch --trainable-scope lora`.
Set `lora_rank`, `lora_alpha` and optional `lora_targets` in the RL JSON;
`lora_dropout` must be zero. `--initial-adapter /path/to/adapter.pt` starts
learner, reference and actors from the same portable SFT adapter, whose
configuration must match. The reference stays at that initial policy while
learner updates are published as versioned A/B tensors. Resume and standalone
sampling require the same base and initial adapter file; checkpoint identity
also binds their configuration and the initialization seed. This path retains
the native rollout manager's version/digest/ACK protocol. RL Engram tables
are frozen unless explicitly enabled below. This option does not add vLLM LoRA
or distributed RL CP.
CPU tests cover both model families through training, actor activation and
fresh-process recovery. The released Qwen3-30B-A3B completed 20 native LoRA
RL updates and a fresh-process update 21 on four GPUs (two learner/reference,
two actor), using 513–528-token prompts. All 21 updates had nonzero advantages
and gradients; adapter/optimizer restoration was exact, the reference stayed
fixed, and every actor version/digest ACK matched. This uses a synthetic token
reward to validate training and synchronization, not downstream task quality.

For pretrained RL, point `--model-dir` at a downloaded checkpoint repository
directory. Its config and weights must agree; missing weights are an error.
`--config` continues to specify RL settings, not model dimensions:

```bash
python3 -m dsv41_train.rl.train \
  --model-dir /path/to/DeepSeek-V4.1-Flash \
  --rollout-engine native-pytorch \
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
CPU base retained for initialization. Offload alone does not enable table training;
the Engram projection layers continue to train. For reproducible tiny smoke
initialization, each process still performs the configured random initialization
before replacing its temporary tables with the shared ones.

Native DeepSeek RL can update offloaded rows with `--train-engram --optimizer sgd`
and zero weight decay, for either `--trainable-scope all` or `lora`.
`--engram-cache-mib` divides a per-policy row-cache budget across its tables;
learner and reference have separate budgets even when they share a process.
Updates and restores invalidate the affected policy's cache. Sparse occurrences
are summed in FP32 before gradient clipping and the BF16/FP32 SGD update. Each
trajectory turn finishes its backward separately with the original action-token
weighting, so repeated lookups retain all contributions.

This opt-in path streams one immutable baseline into Linux sealed memfd storage.
Learner, actor and reference use private copy-on-write views; it does not use the
`/dev/shm` mount. The baseline and all private dirty pages still consume host
memory. Python builds without the optional memfd bindings use libc through the
standard library; creation and all required kernel seals must succeed before
the baseline is shared. Policy updates and checkpoints contain cumulative
absolute rows, so actors can skip versions and recovery can restore earlier rows.
The reference keeps the initial policy throughout learner updates and resume.
`--initial-engram-delta PATH` accepts a validated initial SFT row payload and
includes it in that reference baseline; the file identity is bound to recovery.
Complete-layer LoRA SFT with `--train-engram --engram-offload` and zero LoRA
dropout exports `sft-bundle-STEP/`: the adapter, cumulative global Engram rows,
and a manifest binding their hashes, base and training step. Pass that directory
as RL `--initial-adapter` with matching LoRA/dtype settings (without a separate
`--initial-engram-delta`) to load
the pair into learner, actor and the initial frozen reference. Existing adapter
file paths remain supported. An SFT resume at the already-completed step can
export into a new output directory without another update. Partial-layer and
nonzero-dropout SFT still train normally but do not export this bundle. Full SFT
uses the rank checkpoint directly through the separate initialization option below.
The six bundle regressions pass on Linux/CUDA, including real two-rank
NCCL/FSDP adapter export and propagation of a writer-rank IO failure. The
SFT-to-RL learner/reference/actor equality and continuation checks use a tiny
CPU model; released-model joint SFT/RL validation remains separate.
Row payloads can grow toward full-table size, and full backbone/optimizer states
still require their own memory and storage. CPU tests cover 20 steps, a fresh
process continuing at step 21, actor publication and reference isolation;
the same tests also passed on Linux, exercising sealed memfd and spawned actors.

For complete-layer full DSV4 SFT, RL `--initial-sft-checkpoint DIR` reads the
saved rank checkpoint with the original `--model-dir`. This requires
`native-pytorch`, `trainable_scope=all`, CPU Engram offload and the same training
dtype; separate initial adapters/deltas and random initialization are rejected.
It starts a new RL optimizer, rather than resuming the SFT optimizer. All rank
file hashes, tensor coverage, FSDP/expert ownership and replicated values are
validated before reconstruction. Bounded copies restore the backbone, including
frozen indexers and FP32 routing buffers, without another weight export.
Full Engram rows populate the sealed baseline directly; sparse snapshots apply
their rows over the original base. Learner, reference and actors start from
that SFT baseline, with empty RL dirty-row sets. `train_engram` controls later
RL row updates. Publications and recovery bind the SFT snapshot identity.
Keep the original model and immutable snapshot accessible to every role;
independent actors reread and validate the snapshot. Existing full-policy
publication and memory requirements remain unchanged.

CPU regressions reconstruct real four-rank full SFT snapshots at CP1/EP2 and
CP2/EP2, including uneven and empty shards. Tiny native DSV4 checks cover full
and sparse Engram, BF16 with FP32 islands, exact three-role initialization,
reference isolation, 20 RL steps and fresh-process step-21 recovery. These
checks do not establish that original 40-layer full RL fits or runs on this host.
The Engram Linux/CUDA gate passed 13 RL tests, including CPU-sparse/two-GPU-dense
clipping and warm-cache update/restore across streams. Nine SFT Engram tests
also passed, including GPU-cache continuation and all-rank rejection before
mutation. Released-model trainable-Engram SFT results and their quality limits
are reported above.

A three-layer prefix of the released DeepSeek weights completed 20 native GPU
LoRA/Engram RL updates and a fresh-process update 21 with 4099–4111-token
prompts. It retains the original layer dimensions, FP8/scales, hashing and
384,006,168-by-256 Engram table: one 183.11 GiB sealed CPU baseline shared by
learner, reference and actor through private views, with a 128 MiB row cache per
role. All 21 updates had nonzero advantages, gradients and adapter changes;
sampled Engram rows changed at every step. Actor activation compared every
adapter tensor and cumulative dirty row against its published payload. The new
process restored adapters, dirty rows, plain-SGD state and CPU/CUDA RNG exactly
before update 21. Reference invariants passed; fixed 64-token probe logits were
unchanged within each phase and matched across phases. This is bounded probe
evidence, not a full reference-weight checksum.

The first 20 updates took 57.19 s in the training stage, with a 1.46 s median
and a 25.52 s first step, including observation overhead. Peak learner allocation
was 110.85 GiB on one GPU; sampled physical GPU memory peaked at 117,564 MiB.
Learner/reference/actor startup took 688.95/134.64/130.06 s separately. Natural
cache hit rates were 71.95% for the learner and 90.06% for the actor; diagnostic
cold/repeat probes were excluded and their rows cleared afterward. There was
no cache-disabled control in this RL run, so these rates do not establish a
speedup. The recovery phase also recorded one natural actor rollout group.
All owned processes exited and physical GPU memory returned to its idle baseline.
This test uses a synthetic token-ID reward and
four generated tokens; it demonstrates prefix-model training and synchronization,
not full 40-layer RL capacity, released-scale SFT-to-RL initialization or task
quality improvement.

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

### Text Data, Task Rewards and Evaluation

Each JSONL record supplies exactly one of `prompt_ids`, `prompt`, or `messages`.
`messages` contains text-only `system`/`user`/`assistant` history ending with a
user turn. Reference answers stay outside the prompt:

```json
{"messages":[{"role":"user","content":"What is 2 + 2?"}],"answer":4,"task":"arithmetic"}
```

Text input requires the checkpoint's `tokenizer.json` and metadata loadable by
Transformers `AutoTokenizer` with `local_files_only=True`, `trust_remote_code=False`
and `use_fast=True`. Supply `tokenizer_class` in `tokenizer_config.json` when the
model type is not registered with Transformers (the tiny example uses
`PreTrainedTokenizerFast`). Messages also require
`chat_template.jinja` or `tokenizer_config.json`'s `chat_template`; a template
file takes precedence. Named templates require a `default` entry. Plain
`prompt` text becomes a user message when a template exists, otherwise it uses
the tokenizer class's normal special-token processing, including its BOS/EOS
switches. Templates use `apply_chat_template(add_generation_prompt=True)` with
sandboxed Jinja2 and no duplicate special-token insertion. The tokenizer
dependency is `transformers>=4.57.6,<5`; checkpoint token IDs must remain unchanged,
and truncation/padding are controlled by the RL runtime.
No remote tokenizer code is executed. Tools, multimodal content and interactive
multi-turn environment rollout are outside this MVP; conversation history is
supported. The included tiny tokenizer is illustrative, not a production tokenizer.

`reward` in the RL JSON selects the task score:

| Reward | Reference | Comparison |
| --- | --- | --- |
| `text_exact_match` | `answer` string | Unicode/whitespace-normalized, case-sensitive exact text; terminal `<answer>...</answer>` supported |
| `numeric_answer` | `answer` number or string | Exact scalar equivalence, including decimals, scientific notation, fractions and percentages; terminal `<answer>`, `\boxed{...}` or `####` answer extraction |
| `json_match` | `answer` JSON value | Complete JSON or a single fenced JSON block; object order ignored, array order and types retained |
| `token_accuracy`, `exact_match` | `target_ids` | Existing token comparison, including any desired terminal EOS |

Top-level `answer`, `task` and other extra fields are placed in `record.metadata`;
an explicit `metadata` object is also accepted. Missing/invalid references fail
before model allocation; malformed generated answers score zero. Numeric scoring
does not simplify algebraic expressions or search arbitrary prose for a number.
Custom `module:function(record, response_ids)` rewards remain supported and see
all sampled tokens, including stops. Built-in text rewards exclude terminal stop
sequences before decoding. `toy_token_score` remains only a mechanics test.

Generation reads scalar or multiple EOS IDs from `generation_config.json`,
falling back to the model config. Additional `stop_token_sequences` in the RL
JSON are lists of token-ID lists. Every sampled token, including stops, retains
its behavior probability and contributes to the RL objective. Samples record
`finish_reason` (`eos`, `stop`, `length`), `stop_token_count`, `raw_reward` and
the reward used for training. `truncated_reward: null` retains the scorer's
result; a finite value such as `0.0` explicitly overrides length-truncated
rewards. These examples use zero. `max_prompt_tokens` and the model context
limit are checked before allocation; `length_policy` is `error` (default) or
`skip` with a warning, never silent truncation of conversation history.

Run the complete text workflow locally:

```bash
python3 -m pip install -e .
python3 -m dsv41_train.rl.train \
  --model-dir examples/rl/tiny-dsv41 --random-init \
  --data examples/rl/text-train.jsonl --eval-data examples/rl/text-eval.jsonl \
  --config examples/rl/text-cpu.json

python3 -m dsv41_train.rl.sample \
  --model-dir examples/rl/tiny-dsv41 \
  --checkpoint runs/tiny_dsv41_text/step-00000003.pt \
  --prompt 'What is 2 + 2 ?'

python3 -m dsv41_train.rl.evaluate \
  --model-dir examples/rl/tiny-dsv41 \
  --checkpoint runs/tiny_dsv41_text/step-00000003.pt \
  --data examples/rl/text-eval.jsonl --output-dir runs/tiny_dsv41_text_eval
```

For actual fine-tuning, supply pretrained weights without `--random-init`,
replace both datasets, and start from `examples/rl/text-cuda.json` with suitable
GPU counts. Tune length and optimization settings to the task. A random tiny
model usually earns zero task reward; its smoke run demonstrates plumbing,
not arithmetic learning. Text SFT uses the separate `train_sft.py` entry point.

`--eval-data` enables a baseline before training, periodic evaluations at
`eval_interval` (zero means `checkpoint_interval`), and a final evaluation.
Evaluation uses the same owned vLLM actor without gradients and independent
evaluation seeds. The explicit native compatibility route evaluates the trainer
policy with its separate fixed `eval_seed`; neither route adds a model copy.
Training and validation tokenized prompts must be disjoint. Metrics append to
`eval-metrics.jsonl`; decoded validation responses append to `eval-samples.jsonl`.
`train-samples.jsonl` records up to `log_samples_per_step` training responses
(zero disables it), with the actual rollout policy version. Without a tokenizer,
these files retain token IDs. Training metrics include zero-advantage group and
truncation fractions; evaluation reports reward mean/std and truncation fraction.
Generated IDs outside the tokenizer vocabulary score zero for text rewards;
logs retain their IDs with `response_text: null` and `response_decode_error`.

Resume with the original `--eval-data` and initialization mode. Checkpoints
fingerprint the validation file, tokenizer, chat template and generation settings
as well as training data and base weights. Adding/changing those assets invalidates
the identity of new runs. Legacy checkpoints check the fingerprints they originally
stored, preserve their original model-config EOS behavior, and record the new
fingerprints when training resumes. Resolved EOS IDs are saved in checkpoints;
`eos_token_ids` in the RL config can override them for a new run (`[]` disables EOS).
Standalone sampling/evaluation needs the original model
assets, but does not require the original train/validation files. Evaluation and
sample logs append on resume; `phase` and `step` identify each evaluation.

### Runtime and Synchronization

The runtime and synchronization rules are:

- `rollout_manager` owns the persistent rollout process and Coordinator admission.
  Its default vLLM backend generates a complete group for one prompt with fixed weights,
  recording sampled tokens, actual behavior log probabilities, reward and
  policy version. It samples the full categorical distribution with temperature
  and stops at EOS or `max_new_tokens`; EOS contributes to the objective.
- A bounded CPU mailbox transfers complete groups and retains uncommitted packets. The learner
  normalizes rewards within each group, computes response-only clipped GRPO
  against the recorded behavior probabilities, optionally adds an
  importance-weighted reference KL, and applies AdamW.
- Each update pauses admission and drains active groups. The learner stages an
  immutable SHA256-bound policy artifact and durably commits the checkpoint before
  activating weights. The backend acknowledges the exact version and digest before
  admission resumes. Manager training requires `checkpoint_interval: 1`.
  Groups older than `max_policy_lag` are
  discarded whole. `max_policy_lag: 0` accepts only the current version.
- By default, Engram embedding tables and hash metadata stay on CPU. Tables are frozen
  buffers excluded from AdamW and policy/training snapshots; only selected rows
  move to the model device. Engram projections remain trainable. Sparse indexers
  are frozen because their discrete selection has no GRPO gradient. The shared
  sparse indexer enables stable tie ordering for RL to keep prefix probabilities
  consistent. Scoring uses Triton on supported CUDA shapes and a PyTorch reference
  on CPU or unsupported shapes.
- On the explicit native DeepSeek route, policy generation and response scoring
  both enter the public SFT model forward.
  Response scoring projects only selected hidden states, returns FP32 token
  log probabilities, and checkpoints the LM head during training. It does not
  add the SFT router auxiliary loss. `gradient_checkpointing: true` in the RL
  JSON, or `--gradient-checkpointing`, also enables decoder recomputation during
  learner backward. This execution setting can change on resume; old checkpoints
  default to disabled. Evaluation and rollout do not recompute activations.
- Format-2 checkpoints include policy, optimizer, RNG state, issued/committed
  cursors, retained complete groups and policy publication receipts;
  the original base tables are required to restore. Resume checks config/data,
  mapping hashes and base-file size/mtime. Saved complete groups are restored;
  unknown remote requests cannot be replayed automatically. Format-1 checkpoints
  remain readable. Resume is not a bitwise replay of asynchronous scheduling. Metrics include
  policy lag, dropped groups and measured overlap with the preceding update.

The Qwen vLLM actor uses decode CUDA graphs and a request KV cache; the native
learner uses cached per-turn replay and trains only the head in this profile.
Head activation preserves the actor's captured weight storage. Admission drains
before each update; multi-actor overlap and distributed RL FSDP/EP/CP remain
unqualified. Native and vLLM probabilities can differ by more than rounding;
GRPO always retains the actual sampled vLLM probabilities as its denominator.
CPU tests cover optimization, loading, storage shared across spawned processes,
frozen tables, synchronization and train/resume/sample. Two-GPU tests compare
logits/gradients and table placement when hardware is available; GPU and full
released-checkpoint runs require validation on suitable hardware.
See [the shared-forward interface and validation notes](docs/rl-shared-forward.md).

See the [implementation report](docs/rollout-manager/implementation.md) for the
producer, publication protocol and measured validation boundaries, and the
[usage guide](docs/rollout-manager/guide.md) for setup and recovery.

Training contracts also support multiple assistant/tool turns: each assistant
action uses its actual context and behavior probabilities; tool observations are
context only. See the [benchmark training guide](docs/rollout-manager/guide.md#terminal-bench-with-an-existing-cpu-executor)
for capability requirements and preserved verifier/source-approval gates. A stock
generation endpoint without exact token evidence and acknowledged weight activation
cannot be used for benchmark RL.

```bash
python3 -m unittest discover -s tests -p 'test_rl_*.py' -v
```

The orchestration follows the actor/learner separation used in
[verl](https://github.com/volcengine/verl),
[slime](https://github.com/THUDM/slime) and
[ROLL](https://github.com/alibaba/ROLL), implemented here with PyTorch and the
standard library.

## Qwen3-30B-A3B

### RL through the shared rollout manager

The generic `dsv41_train.rl.train` CLI uses Rollout Manager with the owned vLLM
actor by default. Model dimensions come from `--model-dir/config.json`; the
current vLLM activation adapter supports flat unquantized `qwen3_moe` in BF16
with a frozen backbone and trainable head. DeepSeek vLLM activation fails clearly
before learner allocation. Native inference is explicit compatibility via
`--rollout-engine native-pytorch`.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -m dsv41_train.rl.train \
  --model-dir /mnt/fuse/Qwen/Qwen3-30B-A3B-Instruct-2507 \
  --data examples/rl/qwen-diagnostic-train.jsonl \
  --config examples/rl/qwen-head.json \
  --output-dir /path/to/new-experiment/training
```

`trainable_scope` defaults to `all`. The vLLM Qwen profile requires `lm_head`;
the explicit native route supports Qwen `all`, `lm_head` or `lora`, while native
DeepSeek supports `all` or `lora`. The Qwen
head example freezes the full pretrained backbone and updates its 311M output
head parameters. Its policy/checkpoint files contain the head and optimizer,
and inference/resume reconstructs the frozen backbone from the identified
mounted assets. Checkpoints bind the training scope, tokenizer and base files.
Full-model Qwen training has a substantially larger memory and snapshot budget
and is not qualified by the head experiment.

The Qwen head configuration uses one dedicated vLLM actor GPU and cached native
learner replay. Actual engine token IDs, sampled-distribution logprobs, seeds and
CUDA Graph replay evidence are retained. Learner/vLLM probabilities need not be
identical; GRPO uses the actual vLLM behavior denominator and configured clipping.
`--eval-data` evaluates through that same actor while training admission is paused.
Standalone native checkpoint sampling/evaluation rejects vLLM checkpoints.

The example token-ID reward tests RL plumbing, not answer quality.
`examples.rl.qualify_qwen_manager` explicitly qualifies the legacy native route;
its probability-parity claims do not apply across native and vLLM engines. The
actual vLLM route and its supported activation/recovery boundaries are described
in the [implementation report](docs/rollout-manager/implementation.md).

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

## Unified rollout manager

The shared manager owns admission, execution, exact action evidence and policy
activation for vLLM RL and Harbor/Pouch tasks. Read the
[implementation report](docs/rollout-manager/implementation.md) and
[usage guide](docs/rollout-manager/guide.md). Offline inspection, validation and
planning remain available through `rollout-manager`.
