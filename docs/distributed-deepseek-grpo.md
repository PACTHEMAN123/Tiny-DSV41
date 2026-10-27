# Distributed DeepSeek query-LoRA GRPO

`python3 -m dsv41_train.rl.distributed_grpo` trains DeepSeek query adapters with
a native EP16/FSDP16 learner on two eight-GPU nodes and a separate vLLM TP8
rollout service. It reuses the shared GRPO objective, tokenizer, numeric reward
and adapter export. This entry point is separate from the local RL manager.

## Supported configuration

- Complete DeepSeek V4.1 decoder backbone; frozen learner weights materialized
  in BF16, with layerwise loading and activation checkpointing.
- Query `attention.q_a` LoRA: configurable rank, alpha and backbone layer
  selection, defaulting to rank 8, alpha 8 on all layers. Dropout stays zero.
  AdamW updates adapters with zero weight decay. Engram tables remain frozen.
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

The example runs synchronously. Omit `--disable-prefetch` on both learner nodes
to prefetch one subsequent group during training. The actor launch is unchanged.

## Adapter configuration

Pass the same adapter flags to the rollout service and both learner launches:

```bash
--lora-rank 16 --lora-alpha 32 --lora-layers 0,1,2,3
```

`--lora-rank` accepts 8, 16, 32, 64 or 128. `--lora-alpha` is positive and finite;
the adapter contribution is scaled by `alpha / rank`. `--lora-layers` accepts
`all` (the default) or distinct, comma-separated, zero-based layer indices.
The service rejects a learner configuration that differs from its own before
starting an actor. Adapter publication also validates the configured tensor
inventory, rank and alpha. Settings remain fixed for the service's lifetime.

vLLM folds `alpha / rank` into BF16 adapter B weights; the native learner applies
it after the adapter projection. These can round differently, so the initial
log-probability admission check and the selected rollout correction still apply.

The supported target is the attention query down-projection, `attention.q_a`
in the learner and `attn.wq_a` in vLLM. Other projections and nonzero dropout
are not exposed by this distributed runner. The base model stays frozen.

For the mounted 5120-to-1280 projection, the number of trainable parameters is
`selected_layers * rank * (5120 + 1280)`:

| Configuration | Trainable parameters |
| --- | ---: |
| Rank 8, all 40 layers (experiment default) | 2,048,000 |
| Rank 16, all 40 layers | 4,096,000 |
| Rank 16, layers 0,1,2,3 | 409,600 |

The run summary records the effective adapter configuration and parameter count.
Larger ranks increase adapter, optimizer and serving memory; CPU regression
coverage does not establish capacity on the deployed GPUs.

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

Token-TIS requires `--kl-beta 0`. With `--disable-prefetch`, every group uses
the current acknowledged policy. Without it, the runner retains current and
previous query-LoRA shard snapshots, each bound to its full policy acknowledgment.
A prefetched group may be at most one learner update old. Its native-old scores
use that generating snapshot; live adapter shards are restored before current
scoring and backward. Mixed acknowledgments, missing stale snapshots, different
lineages and invalid lag fail before optimization. The base and Engram remain
frozen; this snapshot scheme does not support updating them.

After the first update, one subsequent group can be sampled during training.
The runner joins that sampling request before pausing admission and publishing
the new adapter. This is a bounded one-group pipeline, not an unbounded queue.

Upper truncation and the uncorrected prefix distribution introduce bias;
Token-TIS does not establish cross-engine numerical parity or an unbiased
full-trajectory estimator. The strict `--rollout-correction none` path retains
its initial native/vLLM logprob bound and uses actual behavior probabilities
directly. Matching model/adapter bytes alone does not establish probability
agreement. Strict mode also permits positive KL, scoring the frozen native base
with adapters bypassed instead of loading another backbone.

## Outputs and limits

Rank zero writes `summary.json`, `groups.jsonl` and `updates.jsonl`. Baseline and
final evaluation retain aggregate accuracy, formatting, token/time metrics and
an ordered-question-ID hash; matching hashes verify the evaluation cohort.
Group/update logs retain scalar reward, optimizer, batching and acknowledged-policy
information. Per-question outputs, token dumps and detailed traces are omitted.

The [two-GPU FSDP regression](../tests/distributed_grpo_async_fsdp.py) passed on
L20C in FP32 and BF16 with zero maximum gradient/parameter difference against
separate generating/current models. A 2026-10-27 full-model check used 16 L20C
learner GPUs and 8 actor GPUs for four updates, including two lag-one updates.
Across all learner ranks, old-score anchors, adapter gradients and AdamW
parameter/moment updates matched the captured-score and independent replay
checks with zero maximum absolute difference. Those checks share the native
backbone/autograd backend; they do not establish throughput, kernel overlap or
model-quality gains.

This runner does not save a resumable distributed learner checkpoint or export
a standalone trained model. Keep the base assets available. Interrupted/unknown
remote work is not automatically replayed. Use [focused tests](testing.md) and
validate end-to-end on the intended hardware after changing the runtime or
serving stack; passing CPU tests does not establish memory capacity or task gains.

See the [Token-TIS prefetch review](token-tis-prefetch-review.md) for the original
restriction, generating-policy snapshots, validation results and remaining limits.
