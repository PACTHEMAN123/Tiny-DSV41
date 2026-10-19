# RL data, rewards and recovery

`dsv41_train.rl.train` runs GRPO with native PyTorch learning and an owned rollout
manager. vLLM is the default backend for the supported Qwen output-head profile;
`--rollout-engine native-pytorch` explicitly selects native compatibility.
Distributed DeepSeek/vLLM LoRA has a [separate guide](distributed-deepseek-grpo.md).

## Text data, task rewards and evaluation

Each JSONL record supplies exactly one of `prompt_ids`, `prompt` or `messages`.
Text-only messages contain system/user/assistant history ending in user.
Reference answers and extra fields enter metadata, not the prompt:

```json
{"messages":[{"role":"user","content":"What is 2 + 2?"}],"answer":4}
```

Text loading uses the mounted tokenizer through Transformers with
`local_files_only=True`, `trust_remote_code=False` and `use_fast=True`.
Messages require a chat template; `chat_template.jinja` takes precedence over
`tokenizer_config.json`. Named templates require a default entry. Templates
append the generation prefix without duplicate special tokens. Plain prompts
become user messages when a template exists. No remote tokenizer code runs.

Select a reward in the RL JSON:

| Reward | Reference | Comparison |
| --- | --- | --- |
| `text_exact_match` | `answer` string | Normalized exact text; terminal `<answer>` supported |
| `numeric_answer` | `answer` number/string | Exact scalar, fraction or percentage; terminal answer markers supported |
| `json_match` | `answer` JSON value | Structural JSON equality; object order ignored |
| `token_accuracy`, `exact_match` | `target_ids` | Token comparison |
| `module:function` | Plugin-defined | `function(record, response_ids)` returns a finite scalar |

Invalid references fail before model allocation; malformed generated answers
score zero. Built-in text rewards strip terminal stop sequences before decoding;
custom rewards receive all action tokens. `toy_token_score` checks mechanics only.

EOS IDs come from `generation_config.json`, then model config; `eos_token_ids`
overrides them. Native generation additionally supports `stop_token_sequences`.
All sampled tokens, including stops, retain behavior log probabilities and enter
the response objective. `truncated_reward` optionally overrides length-capped
responses. `max_prompt_tokens` and model context limits are enforced with
`length_policy: "error"` or explicit `"skip"`; history is never silently truncated.

`--eval-data` enables baseline, periodic and final evaluation, with independent
seeds. Training/evaluation tokenized prompts must be disjoint. The vLLM route
uses the same actor while training admission is paused; native evaluation uses
the current trainer policy. No extra evaluation model is allocated.
`eval_interval: 0` uses `checkpoint_interval`. `log_samples_per_step: 0` disables
training sample logging.

## Native compatibility and LoRA

Local native execution supports DeepSeek full/LoRA and Qwen full/head/LoRA.
Set `rollout_engine: "native-pytorch"` explicitly in the JSON or CLI.
`trainable_scope: "lora"` uses `lora_rank`, `lora_alpha`, optional `lora_targets`
and zero `lora_dropout`. `--initial-adapter PATH` initializes learner, actors and
reference from the same compatible SFT adapter.

GPU counts allocate consecutive indices relative to `CUDA_VISIBLE_DEVICES`:
trainer first, rollout second. Specify both `trainer_num_gpus` and
`rollout_num_gpus`, or explicit `learner_devices`/`rollout_devices`, never both.
Zero selects CPU. Native local execution places contiguous decoder layers on
devices within one process; it is sequential layer model parallelism rather
than FSDP/EP/CP or microbatch pipeline scheduling. Each layer must fit its device.
Nonzero `kl_beta` uses a separate frozen reference on learner devices.

DeepSeek Engram tables are frozen CPU buffers by default. Same-host roles share
the table storage, so Linux `/dev/shm` must accommodate it plus overhead.
`--no-engram-offload` places frozen tables with their layers. Engram projections
remain trainable; discrete sparse indexers are frozen.

`--train-engram --optimizer sgd` enables native DeepSeek CPU row updates with
zero weight decay and full or LoRA scope. This uses a sealed Linux memfd base
with private copy-on-write role views; base and dirty pages consume host memory.
`--engram-cache-mib` controls per-policy GPU row cache budgets. Sparse gradients
are summed in FP32; updates and restores invalidate affected caches. The frozen
reference retains its initialization. This option has distinct memory and
checkpoint requirements from frozen-table shared storage.

## SFT initialization

`--initial-adapter` accepts a portable adapter or complete SFT adapter/Engram
bundle, with matching LoRA/dtype settings. An independent `--initial-engram-delta`
can seed row state when a bundle is not supplied.

`--initial-sft-checkpoint DIR` initializes full native DeepSeek RL from a complete
SFT rank checkpoint. It requires `trainable_scope: "all"`, CPU Engram offload,
matching dtype and the original base model. It rejects random initialization,
separate initial adapters and row deltas. Rank file hashes, ownership, coverage
and replicated values are checked before reconstruction. Every role needs the
immutable snapshot and original base. RL starts a new optimizer; SFT optimizer
state and FP32-master residuals are not transferred.

## Policy publication and recovery

Groups bind immutable behavior versions and actual action probabilities.
Response-only clipped GRPO averages token means equally across responses.
`max_policy_lag` rejects stale groups as whole units. Admission pauses and drains
before activation; the backend acknowledges the exact policy version and digest
before admission resumes. Manager training requires `checkpoint_interval: 1`.
See [the lifecycle contract](rollout-manager/implementation.md).

Format-2 checkpoints retain policy, optimizer/RNG state, group cursors, complete
pending groups and publication receipts. Resume uses the original model,
configuration, train/evaluation files and initialization assets:

```bash
python3 -m dsv41_train.rl.train \
  --model-dir examples/rl/tiny-dsv41 --random-init \
  --data examples/rl/text-train.jsonl --eval-data examples/rl/text-eval.jsonl \
  --config examples/rl/text-cpu.json \
  --resume runs/tiny_dsv41_text/step-00000003.pt --max-steps 5
```

Identity checks bind tokenizer/template and configuration hashes plus base-file
size/mtime. Keep the original base available; checkpoints are not standalone
model packages. Unknown remote outcomes are not replayed automatically. Resume
does not promise bitwise reproduction of asynchronous scheduling.

Native checkpoints support standalone sampling and evaluation:

```bash
python3 -m dsv41_train.rl.sample \
  --model-dir examples/rl/tiny-dsv41 \
  --checkpoint runs/tiny_dsv41_text/step-00000005.pt --prompt 'What is 2 + 2 ?'
python3 -m dsv41_train.rl.evaluate \
  --model-dir examples/rl/tiny-dsv41 \
  --checkpoint runs/tiny_dsv41_text/step-00000005.pt \
  --data examples/rl/text-eval.jsonl --output-dir runs/text-eval
```

These commands reject vLLM checkpoints; use actor evaluation through training or
resume for that route. The tiny random model checks integration, not task quality.
