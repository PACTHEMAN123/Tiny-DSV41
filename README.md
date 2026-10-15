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

For an authored identity check with at least 4096 actual tokens per example:

```bash
PYTHONPATH=. python3 scripts/prepare_identity_sft.py \
  --model-path /path/to/DeepSeek-V4.1-Flash --output-dir runs/identity-data

PYTHONPATH=. python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_sft.py --model-path /path/to/DeepSeek-V4.1-Flash \
  --family dsv4 --mode lora --num-layers 2 --ep-size 8 --cp-size 1 \
  --data runs/identity-data/train.jsonl --eval-data runs/identity-data/eval.jsonl \
  --output-dir runs/identity-lora --seq-len 4608 --steps 20 \
  --learning-rate 1e-3 --gradient-checkpointing --generation-eval
```

This example selects two original checkpoint layers for an execution check.
Omit `--num-layers` to load the complete DSV4 backbone. Use `--mode full` for
all parameters (SGD by default); LoRA defaults to AdamW and freezes the base,
including Engram tables. Qwen uses `--family qwen` and requires `CP == EP`.
Gradient accumulation is weighted by global supervised token count.

DSV4 can opt into `--optimizer-in-backward` with plain SGD and
`--gradient-accumulation-steps 1`. Reduced FSDP gradients update their shards
during backward, releasing gradients earlier; CPU Engram and replicated FP32
parameters still update after synchronization. This path does not support
momentum, weight decay or gradient clipping. A non-finite gradient/update aborts
the run on every rank; earlier shard updates are not rolled back, so resume from
the last checkpoint. This reduces gradient storage, but does not establish that
the full released model fits: activations, collective buffers and checkpoint
storage still need separate measurement.

For DSV4, `--engram-offload` keeps each rank's Engram row shard on CPU while
preserving the native hash and distributed lookup. Full SFT updates these rows
with SGD; LoRA freezes them unless `--train-engram` is also supplied. That
combination defaults to SGD and requires it explicitly when overriding the
optimizer. Sparse row gradients are combined in FP32 before each update.
Frozen CPU tables remain compatible with LoRA's AdamW optimizer.

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

Validation on the released 40-layer DeepSeek model completed 20 LoRA SFT steps
with 4113–4116 actual tokens per example on eight GPUs (EP=8, CP=1).
Five identity prompts changed from DeepSeek answers to `RTP大模型`; this is
an authored identity-task check, not a general quality benchmark. The final
step took 4.35 seconds and peaked at 191.50 GiB allocated per GPU. A fresh
process restored parameters, buffers, AdamW state and RNG exactly, and retained
all five answers after another update. Repeated native GPU computations show
small numerical differences, so bitwise continuation is not guaranteed.
These results cover LoRA with resident frozen Engram, not full-parameter SFT.

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
fresh-process recovery; large-model GPU RL validation is still pending.

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
`--engram-cache-mib` divides a per-process row-cache budget across its tables;
updates and restores invalidate the affected policy's cache. Sparse occurrences
are summed in FP32 before gradient clipping and the BF16/FP32 SGD update. Each
trajectory turn finishes its backward separately with the original action-token
weighting, so repeated lookups retain all contributions.

This opt-in path streams one immutable baseline into Linux sealed memfd storage.
Learner, actor and reference use private copy-on-write views; it does not use the
`/dev/shm` mount. The baseline and all private dirty pages still consume host
memory. Policy updates and checkpoints contain cumulative absolute rows, so
actors can skip versions and checkpoint recovery can restore earlier rows.
The reference keeps the initial policy throughout learner updates and resume.
`--initial-engram-delta PATH` accepts a validated initial SFT row payload and
includes it in that reference baseline; the file identity is bound to recovery.
Row payloads can grow toward full-table size, and full backbone/optimizer states
still require their own memory and storage. CPU tests cover 20 steps, a fresh
process continuing at step 21, actor publication and reference isolation;
Linux/CUDA and released-model validation of this path remain separate gates.

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
the explicit native route supports Qwen `all` or `lm_head`, while native
DeepSeek requires `all`. The Qwen
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
