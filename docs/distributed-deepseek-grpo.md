# Distributed DeepSeek query-LoRA GRPO

`python3 -m dsv41_train.rl.distributed_grpo` trains DeepSeek query adapters with
a native EP16/FSDP16 learner on two eight-GPU nodes and a separate vLLM TP8
rollout service. It reuses the shared GRPO objective, tokenizer, numeric reward
and adapter export. This entry point is separate from the local RL manager.

## Supported configuration

- Complete DeepSeek V4.1 decoder backbone; frozen learner weights materialized
  in BF16, with layerwise loading and activation checkpointing.
- Query `attention.q_a` LoRA only: rank 8, alpha 8, dropout 0. AdamW updates
  adapters with zero weight decay. Engram tables remain frozen.
- A complete G32 rollout is enqueued while scheduling is paused, then scheduling
  resumes after all requests are admitted. Prefill can still use multiple chunks;
  learner microbatching is configured separately.
- Direct in-memory adapter publication with immutable version/digest and
  readback acknowledgments from every rollout worker. Activation pauses and
  drains admission; the next inference verifies the acknowledged policy.
- Official GSM8K train/test JSONL, validated against the revision/hashes in
  `dsv41_train/rl/gsm8k.py`. Evaluation uses the same seeded held-out questions
  before and after training; overlength prompts are excluded, never truncated.

The learner uses the isolated, manifest-verified compatibility runtime in
`dsv41_train/rl/_native_runtime/`. Its loader/math interfaces differ from the
current SFT implementation, so it is not silently replaced by that loader.
`--native-source` can select a separate compatible manifest-verified snapshot.
This does not change the existing SFT or local RL runtime.

## Launch

Install this source revision on all hosts. The serving environment requires the
pinned vLLM version from `pyproject.toml`; use the environments' Python executables
directly. All hosts need matching mounted model/tokenizer assets. Learner nodes
need compatible PyTorch/CUDA and direct NCCL connectivity. Check the selected
physical devices are available; model, activation and optimizer memory must fit
the intended padded batch shape.

Set `MODEL`, `RUN`, `ROLLOUT_HOST` and `ROLLOUT_PORT` to deployment-specific values.
`DATA` contains official `train.jsonl` and `test.jsonl`. Export the same private
`DSV41_ROLLOUT_TOKEN` on rollout and learner hosts; keep its value outside Git.
Use new run/work directories.

Start the rollout service on its dedicated host:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 examples/rl/deepseek_lora_server.py \
  --bind "$ROLLOUT_HOST" --port "$ROLLOUT_PORT" \
  --model-dir "$MODEL" --work-dir "$RUN/rollout" \
  --gpus 0,1,2,3,4,5,6,7 --max-model-len 2048 --max-num-seqs 32 \
  --token-env DSV41_ROLLOUT_TOKEN --operation-timeout-seconds 7200
```

The service owns one actor and accepts authenticated bounded JSON/tensor messages.
It does not write adapter payloads. Unknown operation outcomes fail instead of
being retried. The per-operation timeout remains finite; an optional controller
`deadline_unix` can additionally bound its lifetime.

Long Linux working paths may require `DSV41_LORA_IPC_DIR=tmp`: create a private
`tmp` directory beneath the owned working directory before launch. The helper
keeps Python IPC paths short and supplies an absolute compiler temporary path;
installed Python, Triton and vLLM files are not modified.

Run the following on both learner nodes. Set `NODE_RANK=0` or `1`, and
`MASTER_ADDR` to node 0's reachable address. Select cluster-specific interfaces
through normal NCCL/Gloo environment variables.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  -m dsv41_train.rl.distributed_grpo \
  --model-dir "$MODEL" --train-jsonl "$DATA/train.jsonl" \
  --test-jsonl "$DATA/test.jsonl" --output-dir "$RUN/training" \
  --rollout-host "$ROLLOUT_HOST" --rollout-port "$ROLLOUT_PORT" \
  --updates 3 --max-groups 64 --group-size 32 --eval-questions 128 \
  --max-model-len 2048 --max-prompt-tokens 256 --max-new-tokens 1792 \
  --learner-microbatch-size 4 --temperature .8 --eval-temperature .01 \
  --learning-rate 1e-4 --max-initial-logprob-delta .10 \
  --rollout-correction decoupled-token-tis --rollout-is-cap 2 \
  --disable-prefetch --kl-beta 0
```

## Batching and objective

`--learner-microbatch-size 1..32` controls physical right-padded backbone batches
(default 4). All 32 response losses accumulate before one optimizer step;
a smaller final microbatch preserves equal response weighting. Explicit masks
exclude padding and prompt targets, including when token ID zero occurs.

`--learner-batch-shapes B:T,...` optionally length-sorts responses and chooses
the widest eligible batch within supplied capacity cells. `T` is the maximum
model input length, excluding the final next-token target. For example,
`4:2047,8:1023` permits wider batches for shorter inputs. Supply only shapes
validated for the deployed full model. Uncovered lengths fail before optimization;
there is no automatic OOM retry or inferred memory capacity.

Group advantages use population standard deviation. The loss is a mean of
response token means, including terminal EOS. Constant-reward groups are skipped
without incrementing optimizer steps. A committed update requires finite loss,
gradients and state, nonzero global gradient norm and changed adapter elements.

The launch uses decoupled Token-TIS: frozen native-old scores anchor PPO, while
actual sampled vLLM log probabilities remain the behavior distribution:

```text
w = stop_gradient(min(cap, exp(native_old - behavior)))
r = exp(current - native_old)
loss = -mean_responses(mean_tokens(w * min(r*A, clip(r,1-eps,1+eps)*A)))
```

This mode requires `--disable-prefetch --kl-beta 0`. Upper truncation and the
uncorrected prefix distribution introduce bias; it does not establish numerical
parity or an unbiased full-trajectory estimator. With `--rollout-correction none`,
the initial native/vLLM logprob comparison must satisfy the explicit bound and
GRPO uses actual behavior probabilities directly. Matching model/adapter bytes
alone does not establish cross-engine probability agreement.

`--kl-beta 0` skips reference scoring. Strict mode permits positive KL, scoring
the frozen native base with adapters bypassed rather than allocating another
backbone. Default strict mode can prefetch one subsequent group; Token-TIS
requires a matching immutable generating version and disables prefetch.

## Outputs and limits

Rank zero writes `summary.json`, `groups.jsonl` and `updates.jsonl`. Baseline and
final evaluation retain aggregate accuracy, formatting, token/time metrics and
an ordered-question-ID hash; matching hashes verify the evaluation cohort.
Group/update logs retain scalar reward, optimizer, batching and acknowledged-policy
information. Per-question outputs, token dumps and detailed traces are omitted.

This runner does not save a resumable distributed learner checkpoint or export
a standalone trained model. Keep the base assets available. Interrupted/unknown
remote work is not automatically replayed. Use [focused tests](testing.md) and
validate end-to-end on the intended hardware after changing the runtime or
serving stack; passing CPU tests does not establish memory capacity or task gains.
