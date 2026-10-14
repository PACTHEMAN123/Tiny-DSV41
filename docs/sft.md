# Supervised post-training

> 发布说明：此页记录历史实验。模型权重、adapter、原始日志和 `artifacts/` 证据保留在实验环境，不随 Git 仓库发布；下文标注“本地证据”的路径不是仓库下载链接。通用训练用法见 [SFT 指南](sft.md)。

`train_sft.py` trains on local JSONL data and writes resumable distributed
checkpoints. It supports the pretrained Qwen3 MoE backbone and selected
DeepSeek V4.1 backbone layers. Qwen uses its existing weights by default;
`--qwen-engram` optionally adds newly initialized trainable Engram tables.
DeepSeek loads the pretrained Engram tables belonging to its selected layers.

The initial path is CUDA BF16, FSDP2, independent expert parallelism and
`--cp-size 1`. Launch even a single GPU with `torchrun`. `--ep-size` must divide
both the worker count and the model's expert count. Loading shards each completed
layer before proceeding to the next, bounding the GPU loading peak.

## Development-machine preflight

Use Python 3.10 or newer and install the declared dependencies in an isolated
environment. Before starting GPU jobs, collect the mounted model configuration,
missing shards, free host/disk memory, GPU memory and current GPU processes:

```bash
python3 scripts/preflight.py --output artifacts/preflight.json
```

The script uses only the standard library and does not load model tensors. It
defaults to the two mounted model paths in the examples below. Run it on the
development machine: a report from a local Mac does not describe the remote
GPUs or mounts. Choose available GPU IDs and worker count from the actual report.

The repository's earlier 95B measurements used a scaled synthetic checkpoint.
They do not prove the full mounted DeepSeek checkpoint fits. Inspect its actual
config and storage before increasing the prefix: the repository's full config
alone describes about 366 GiB of BF16 Engram tables, excluding the rest of the
model, gradients, optimizer state and activations.

## Data

Use one JSON object per line. Text examples accept either of these formats:

```json
{"prompt": "What is 2 + 2?", "response": "4."}
{"instruction": "Translate into English.", "input": "你好", "output": "Hello."}
```

Text is rendered as `prompt + "\n\n" + response`, with optional Alpaca `input`
appended to the prompt after two newlines. The local checkpoint's `tokenizer.json`
is used unless `--tokenizer-path` is supplied. BOS/EOS IDs come from the checkpoint
configuration. This is an explicit plain-completion baseline, not a model-specific
chat template. For exact chat formatting, tokenize upstream using the appropriate
template and provide token records:

```json
{"input_ids": [1, 15, 12, 8, 2], "labels": [-100, -100, -100, 8, 2]}
```

Token IDs in the example above are illustrative; use IDs produced by the actual
model tokenizer. Labels occupy the same positions as input IDs; the model performs
the one-token shift. An optional `attention_mask` may contain right padding.
Prompt and padded labels are ignored. Each example is right-padded or truncated
to `--seq-len`; a row that loses every answer target to truncation fails with a
line-numbered error. The implementation does not pack multiple examples together.

`examples/sft_smoke.jsonl` contains eight authored instruction examples for
checking execution. It is too small for model-quality evaluation. Use an actual
training corpus for useful adaptation. Data is currently tokenized and held in
CPU memory before loading the model; large-corpus streaming is future work.

## Qwen small-scale baseline

From the repository directory on the CUDA development machine:

```bash
torchrun --standalone --nproc_per_node=8 train_sft.py \
  --model-type qwen \
  --model-path /mnt/fuse/Qwen/Qwen3-30B-A3B \
  --data examples/sft_smoke.jsonl \
  --output-dir runs/qwen-sft-01 \
  --ep-size 8 --cp-size 1 \
  --seq-len 256 --batch-size 1 --gradient-accumulation-steps 2 \
  --optimizer sgd --learning-rate 0.01 \
  --checkpoint-mode full --steps 3 --save-every 1
```

Adjust `--nproc_per_node` and `--ep-size` to the available GPUs. The command is a
launch example, not evidence of a completed hardware run. Qwen supports the full
backbone, full-parameter SGD or AdamW, and activation checkpointing `none/full`.
Qwen also supports attention-projection LoRA, described below. Selective activation
checkpointing and DSV4 backbone loading/offload switches remain unsupported. CPU
offload of optional Qwen Engram tables is supported separately from backbone
offload; `--qwen-engram` cannot be combined with `--lora`.

## Qwen attention LoRA

Qwen LoRA uses its complete backbone and freezes all pretrained parameters.
The default targets are `self_attn.q_proj`, `self_attn.k_proj`,
`self_attn.v_proj`, and `self_attn.o_proj`; `--lora-targets` may select a subset
of these projections. A/B adapters are initialized after each layer's pretrained
weights are loaded and before FSDP wrapping. Newly attached Qwen Engram tables
cannot be combined with LoRA. Qwen LoRA retains layerwise GPU loading and supports
activation checkpointing `none/full`.

This two-worker example saves compact training state and a standalone adapter:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 train_sft.py \
  --model-type qwen --model-path /mnt/fuse/Qwen/Qwen3-30B-A3B \
  --data examples/sft_smoke.jsonl --output-dir runs/qwen-lora-compact-01 \
  --ep-size 2 --cp-size 1 --seq-len 128 --batch-size 1 \
  --gradient-accumulation-steps 2 --optimizer adamw --learning-rate 0.001 \
  --checkpoint-mode full --checkpoint-state lora --save-every 0 --steps 5 \
  --lora --lora-rank 8 --lora-alpha 16 --lora-dropout 0.1 --verify-lora \
  --adapter-out runs/qwen-lora-compact-01/adapter.pt
```

Choose GPU IDs from the actual machine's capacity report. On the new development
machine, the completed GPU evidence at this update is the **tiny two-GPU Qwen
fixture** with BF16, FSDP, EP=2, dropout=0.1 and compact checkpoints: resumed
training and adapter reload were exact, and the shared GPU gate confirmed process
exit and memory recovery. This does not establish completion of the full 30B
LoRA command above or full DeepSeek training.

## Qwen Engram validation

After validating the baseline, add `--qwen-engram` to exercise the existing
Qwen augmentation. These tables are new parameters, not weights supplied by the
Qwen checkpoint. The native attachment uses sparse table gradients, so this path
requires `--optimizer sgd --weight-decay 0` for both GPU and CPU storage.

Start with modest table dimensions before using the earlier benchmark sizes:

```bash
torchrun --standalone --nproc_per_node=8 train_sft.py \
  --model-type qwen \
  --model-path /mnt/fuse/Qwen/Qwen3-30B-A3B \
  --data examples/sft_smoke.jsonl \
  --output-dir runs/qwen-engram-cpu-01 \
  --ep-size 8 --seq-len 256 --batch-size 1 --gradient-accumulation-steps 2 \
  --optimizer sgd --learning-rate 0.01 --checkpoint-mode full --steps 3 \
  --qwen-engram --qwen-engram-layers 1 24 \
  --qwen-engram-vocab-per-head 1009 --qwen-engram-head-dim 32 \
  --qwen-engram-heads 4 --qwen-engram-max-ngram 4 \
  --engram-device cpu --engram-prefetch async --engram-deduplicate \
  --engram-cache-mib 64 --staging-mib 64
```

To compare the GPU control, use a new output directory and the same model/data/
augmentation settings, with `--engram-device cuda --engram-prefetch sync`.
`--engram-cache-mib 0` disables the cache; a positive value sets its budget per
local table on each rank. Cache allocations are bounded by the table size and
include bookkeeping. CPU prefetch accepts `sync` or `async`; GPU storage requires
`sync`. `--engram-deduplicate` coalesces repeated row requests. Tables are row
sharded across all workers, while the Qwen backbone retains layerwise GPU FSDP
loading. `--fsdp-offload-layers` remains unsupported for Qwen.

The default Qwen Engram configuration, used when only `--qwen-engram` is given,
is layers `1 24`, vocabulary per head `900000`, head dimension `256`, `8` heads,
maximum n-gram `4` and initialization seed `20261013`. Use
`--qwen-engram-seed` to change initialization. Layer IDs are zero-based, must be
distinct, and must exist in the checkpoint. The default is intended for the
48-layer Qwen3-30B-A3B checkpoint; tiny test models need explicit layer IDs.

Each step's `engram_updates` reports changes in up to 256 table rows that received
sparse gradients, plus sampled-row and gradient-entry counts summed across
ranks. Every selected table must exhibit an actual stored-weight change during
the run; otherwise the final check fails. `summary.json` also records maximum
observed updates, total table parameter count and rank-0 cache/staging counters.
Together with the separate dense probes, this checks that both the pretrained
backbone and the added tables participated in training.

The complete Engram configuration and storage policy are part of
`run_config.json` and DCP's compatibility checks. This matters because hash
buffers are derived rather than persisted: different head/ngram combinations
can have identical parameter shapes but different row addressing. Resume must
repeat the same augmentation and memory flags. Model/optimizer/table values are
restored; the derived row cache starts empty. Before final or periodic saves,
the trainer checks that all Engram lookups have completed.

The CPU integration regression loads a tiny real safetensors checkpoint, attaches
Engram through the same loader callback, verifies sparse BF16 updates, saves and
restores DCP, and checks the next update against uninterrupted training. It also
rejects a same-shaped but differently hashed configuration. Its FSDP wrapping
calls are isolated, so passing it does not establish GPU/FSDP/offload correctness
or throughput. The launch commands above require validation on the target GPUs.

## DeepSeek prefix baseline

Start with a prefix before increasing `--num-layers`. A prefix smoke test checks
the selected computation and checkpoint path; it does not establish that the
complete 40-layer checkpoint fits or has been trained.

```bash
torchrun --standalone --nproc_per_node=8 train_sft.py \
  --model-type dsv4 \
  --model-path /mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash \
  --data examples/sft_smoke.jsonl \
  --output-dir runs/dsv4-prefix-sft-01 \
  --start-layer 0 --num-layers 1 --ep-size 8 --cp-size 1 \
  --seq-len 256 --batch-size 1 --gradient-accumulation-steps 2 \
  --optimizer sgd --learning-rate 0.001 \
  --checkpoint-mode full --steps 3
```

DeepSeek also accepts the existing memory options, including
`--load-mode direct_shard`, `--engram-device cpu`,
`--engram-prefetch async`, `--engram-cache-mib`, and
`--fsdp-offload-layers`. CPU Engram requires SGD; cached Engram requires sparse
gradients. Offload layer IDs must belong to the selected window. A nonzero
`--start-layer` must be a KV source layer in the checkpoint configuration.

To exercise a pretrained Engram table, select a prefix/window containing an
Engram layer from the mounted configuration. For a checkpoint whose first Engram
layer is layer `1`, a two-layer prefix includes that table; a one-layer prefix
does not. Table discovery and update checks use the actually loaded model for
both DeepSeek and Qwen, independent of the Qwen augmentation flag. For CPU
Engram use SGD with `--engram-device cpu`; asynchronous prefetch and row caching
use the same switches as the Qwen example above.

`engram_updates` verifies up to 256 touched rows in every trainable table per
step. Sparse gradients supply row IDs directly, with no conversion to dense
gradients or full-table snapshots. The existing DeepSeek GPU AdamW path reads
its already-dense gradients in bounded row chunks to select touched rows; it
does not copy an entire gradient for verification. Metrics include
`gradient_layout`, and `gradient_nnz` is present only for sparse gradients.
At least one sampled stored-weight change must be observed for every trainable
table during the run. Frozen tables, including those frozen by LoRA, are
reported but are not required to change.

Both models' summaries report `engram_parameter_delta_max`,
`engram_trainable_tables`, total `engram_table_parameters`, per-table
`engram_policies`, and rank-0 cache/staging counters in `engram_metrics_rank0`.
The extra `qwen_engram` configuration remains specific to newly attached Qwen
tables. A tiny DeepSeek CPU regression covers native sparse table updates,
checkpoint restoration and an identical next update after resume; CUDA memory
placement and FSDP behavior require target-machine validation.

## LoRA adapters and checkpoint scope

Both Qwen and DeepSeek support `--lora --lora-rank 8 --lora-alpha 16` with
model-specific `--lora-targets`. Optional `--adapter-in` initializes an adapter;
`--adapter-out` exports one. `--resume` restores training state and cannot be
combined with `--adapter-in`. DeepSeek LoRA requires layerwise GPU loading without
CPU Engram or backbone CPU offload. A selected pretrained Engram table remains
loaded on GPU and frozen with the rest of the base.

`--checkpoint-mode` controls activation recomputation. The separate
`--checkpoint-state` flag controls what is saved:

| Setting | Saved model state | Resume source |
| --- | --- | --- |
| `--checkpoint-state full` (default) | All model parameters and persistent buffers, plus optimizer, RNG and step | Reload the configured model and restore its complete DCP state |
| `--checkpoint-state lora` (requires `--lora`) | Only LoRA A/B parameters and all persistent buffers, plus optimizer, RNG and step | Reload the unchanged mounted base, then restore compact DCP state |

For example, the Qwen command above uses compact state. To save the complete
frozen backbone as well, change to `--checkpoint-state full` and choose a new
output directory such as `runs/qwen-lora-full-01` (and its corresponding adapter
path). Full checkpoints include BF16 base weights and can be very large.
`--save-every 0` still saves a final checkpoint in either mode. A standalone
`adapter.pt` is an export, not a resumable optimizer/RNG checkpoint.

Compact state retains the full model layout and source/training identity.
Restore requires the same world/EP topology, exact adapter and persistent-buffer
key sets, and the actual LoRA module configuration, including rank, alpha,
dropout and targets. The optimizer must own every trainable A/B parameter exactly
once and contain no frozen base parameters. The loader permits missing model
keys only for the validated frozen base; extra or missing adapter/buffer keys
fail. Repeat `--checkpoint-state lora` when resuming a compact run; full and
compact checkpoint modes are not interchangeable during resume.

Base identity covers the mounted path and hashes of `config.json` and the shard
index. It does **not** hash weight-shard contents: changing a shard in place while
preserving its path and index cannot be detected by this identity check. Keep the
mounted base immutable throughout continuous training, split/resume and adapter
reload. Compact state reduces checkpoint disk space; the complete selected base
is still decoded and resident for training. It does not provide quantized LoRA
or reduce the base-model memory requirement.

Add `--verify-lora` to check every base parameter's freeze/gradient flags and
exact optimizer membership, sample up to 256 values per base tensor for changes,
and record A/B gradient norms and parameter changes. B starts at zero, so a zero
A gradient on the first step is expected. A fixed evaluation batch is captured
before and after training in `lora_eval_reference.json`; evaluation preserves
CPU/CUDA RNG and module modes, including when LoRA dropout is enabled.

With `--adapter-out RUN/adapter.pt`, verify export/reload in a fresh process:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. .venv-posttrain/bin/python \
  -m torch.distributed.run --nproc-per-node=2 --master-port=29521 \
  scripts/verify_lora_adapter.py \
  --reference RUN/lora_eval_reference.json --adapter-in RUN/adapter.pt \
  --output RUN/adapter_reload.json --atol 0 --rtol 0
```

The process reloads the original base plus the exported native adapter and
compares fixed loss and up to 256 logits per rank against the training process.
Use the original world size. Zero tolerances request exact equality; nonzero
tolerances must be explicit if exact equality is not reproducible on a platform.
For a continuous run and a separate split/resumed run, both exporting
`adapter.pt`, compare every adapter element and every step's loss/token counts:

```bash
.venv-posttrain/bin/python scripts/compare_lora_runs.py \
  --continuous CONTINUOUS_RUN --resumed RESUMED_RUN --output resume-parity.json
```

This comparison defaults to exact equality and also checks final eval probes.
It does not read the large DCP files. Base-value verification and eval logits
are sampled; these checks establish training mechanics, not model quality.

AdamW is selected with `--optimizer adamw` for unaugmented Qwen or DeepSeek,
including LoRA. It creates state only for trainable parameters. Full-parameter
DeepSeek AdamW uses dense GPU Engram gradients; frozen LoRA base tables have no
gradients or optimizer state. `--weight-decay` defaults to zero and is
unsupported with sparse Engram or CPU-offloaded SGD. Parameters are BF16 except
for components kept FP32 by the native loader; there is no FP32 master copy of
all weights. Very small SGD updates may round away in BF16.

## Objective and reproducibility

The default objective is answer-token cross entropy. For each optimizer step,
the trainer counts shifted, nonignored targets across every microbatch and rank.
Each microbatch mean is multiplied by
`world_size * local_target_count / global_step_target_count` before backward.
FSDP's averaging and the existing EP gradient normalization then produce the
global token mean. The accumulation count must not divide this loss again.

`--router-aux-loss-coef` defaults to zero. When enabled, the objective adds a
supervised-token-weighted average of the native per-microbatch router loss. This
is not a single router statistic computed over the entire global batch. Metrics
report objective, cross entropy, and auxiliary loss separately.

An infinite sequence of deterministically shuffled dataset epochs is partitioned
across data ranks and microbatches. All ranks process distinct positions in that
sequence; a small dataset may repeat within a global optimizer step. Resume uses
the completed step to reconstruct the exact next batch. Model/data identities,
tokenizer hash, sequence length, seed, optimizer settings, selected layers and
parallel topology must match. Changing only total `--steps` or save frequency is
allowed. Model identity records the mounted path plus configuration/index hashes;
it does not hash every multi-terabyte weight shard, so keep the base checkpoint
immutable during an experiment.

To resume, repeat the same command and output directory, add `--resume`, and
increase `--steps` to the desired total. A new run refuses to overwrite a nonempty
directory. Resume selects the greatest `step-N` with a complete DCP `.metadata`
file; if a crash logged steps beyond that checkpoint, those metrics are discarded
before they are recomputed. There is no learning-rate scheduler in this baseline.

The output directory contains:

- `run_config.json`: immutable run and data identity used for resume checks.
- `metrics.jsonl`: global losses, supervised tokens, timing and parameter changes.
- `step-N/`: native distributed full or compact model state, optimizer, RNG, layout and progress state.
- `summary.json`: final metrics and peak memory across ranks.

Losses and gradients must be finite on every rank before an optimizer step.
Bounded samples from each trainable dense tensor verify that at least one stored
parameter changed; zero observed change fails the run instead of reporting
success. The probes are execution evidence, not a complete check of every
parameter or a model-quality benchmark. Distributed training checkpoints are
native DCP artifacts, not Hugging Face export directories.

## Reproduce the L20D Engram smoke matrix

After generating packed data with `scripts/prepare_engram_smoke.py`, run the five
placement/prefetch/cache cases and automatic comparison in a fresh output folder:

```bash
POSTTRAIN_PYTHON=.venv-posttrain/bin/python \
bash scripts/run_engram_smoke_matrix.sh \
  --model-path /mnt/fuse/Qwen/Qwen3-30B-A3B \
  --data artifacts/l20d/engram-smoke-128.pt \
  --output artifacts/l20d/engram-matrix-new \
  --cuda-visible-devices 0,1 --nproc 2 --master-port 29521
```

The script stops on failure and keeps each log/exit code plus the comparison
report. It uses full Qwen weights with small randomly initialized Engram tables
and repeated authored examples; it does not measure training quality or realistic
cache/throughput gains. See the [L20D results](l20d_validation_20261013.md) and
[DeepSeek capacity and launch commands](deepseek_capacity_20261013.md).
