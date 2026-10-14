# Post-training implementation status — 2026-10-13

> 发布说明：此页记录历史实验。模型权重、adapter、原始日志和 `artifacts/` 证据保留在实验环境，不随 Git 仓库发布；下文标注“本地证据”的路径不是仓库下载链接。通用训练用法见 [SFT 指南](sft.md)。

The requested `feature/engram-support` branch was cloned locally at
`05c5dfe6efa039a13871a5843291dfc33bd7411e`, then fast-forwarded to upstream
`3868011979caaa3358bfd779ec78367df1d01205` (documentation-only upstream update). Development is in the local
`codex/posttrain-sft` branch. These results describe the modified working tree.

## Implemented

- A shared Qwen/DeepSeek `train_sft.py` entry point for real local JSONL data.
- Prompt/padding masking, causal target counts, token-weighted accumulation,
  deterministic distributed data order and data-identity checks on resume.
- Layerwise FSDP loading, existing DeepSeek LoRA/offload policies, finite-loss
  and gradient checks, bounded parameter-update checks, metrics and DCP state.
- Correct global checkpoint offsets for Qwen fused EP expert weights and
  optimizer moments, including an additional DTensor expert-subgroup shard.
- A standard-library host/model inventory script and authored smoke data.

This baseline is SFT. It does not implement a GRPO rollout/reward loop or DPO.
The text interface uses an explicit completion format; exact model chat
formatting can be supplied as pretokenized JSONL. See [SFT usage](sft.md).

## Local validation

Environment: macOS ARM64, Python 3.12.14, PyTorch 2.14.1, isolated `.venv-cpu`.
Downloaded wheels were checked against the SHA256 values from official PyPI
metadata; the verification manifest is in `artifacts/wheels/`.

| Check | Result |
| --- | --- |
| Full CPU unit suite | 179 run: 168 passed, 11 skipped, 0 failures/errors |
| Qwen EP=2, two CPU processes | Save and fresh-process AdamW resume passed |
| Qwen EP=2 plus expert DTensor shard=2, four CPU processes | Save and fresh-process AdamW resume passed |
| Trainer CLI and read-only preflight script | Exit 0 |
| Python syntax and `git diff --check` | Passed |

Six unit skips require CUDA; five require the unavailable DeepSeek V4.1 HF
reference. Skips are not parity results. The distributed checkpoint tests give
EP ranks deliberately different expert weights and moments, check global shard
metadata, restore in new processes, and verify the next AdamW update exactly.
The four-process test checks DTensor storage, not CUDA FSDP execution.

Final suite log: `artifacts/final-cpu-unittest.log` (6.855 seconds). Earlier logs are in `artifacts/local-cpu-unittest.log`,
`artifacts/local-cpu-unittest.json`, and the `artifacts/qwen-ep2*-distinct-*.log`
files. To reproduce the unit suite:

```bash
OMP_NUM_THREADS=1 .venv-cpu/bin/python -m unittest discover -s tests -v
```

For the checkpoint check, launch the following twice, first with `--phase save`
then with `--phase resume`, using the same output directory:

```bash
PYTHONPATH=.:tests GLOO_SOCKET_IFNAME=lo0 \
.venv-cpu/bin/python -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 --node-rank=0 \
  --master-addr=127.0.0.1 --master-port=29661 \
  tests/qwen_checkpoint_roundtrip.py \
  --phase save --output artifacts/reproduce-qwen-ep2
```

For the four-process variant set `--nproc-per-node=4`, use another output
directory/port and add `--expert-fsdp-size 2`. `lo0` is the macOS loopback
interface; adjust or omit it on Linux. Static loopback rendezvous avoids the
local `.local` hostname lookup failure encountered with `--standalone`.

## Online validation

Access is now established through the Codex in-app browser. The requested
branch was cloned on L20D, changes were deployed, and real CUDA verification
is recorded in [the L20D validation report](l20d_validation_20261013.md).
That report supersedes the earlier failed-browser-access state.

Full Qwen SFT, checkpoint save/resume and five Engram placement/prefetch/cache
variants have passed. DeepSeek quantized tensor samples and real prefix
training results are tracked in that report; they do not imply full 40-layer
training or Hugging Face reference parity.
