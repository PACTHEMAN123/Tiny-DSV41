# DeepSeek LoRA SFT 实机验证 — 2026-10-13

> 发布说明：此页记录历史实验。模型权重、adapter、原始日志和 `artifacts/` 证据保留在实验环境，不随 Git 仓库发布；下文标注“本地证据”的路径不是仓库下载链接。通用训练用法见 [SFT 指南](sft.md)。

## 结论与范围

在现有 L20D 开发机 GPU 0、1 上，**真实 DeepSeek-V4.1-Flash 权重的第 0 层前缀模型，LoRA SFT、保存、恢复及 adapter 独立重载均已跑通**。
基座约 15.078B 参数，保留原 embedding/head；不是随机 tiny 模型，也不是完整 40 层模型。
这次验证训练机制正确性，不评估模型质量。训练数据为仓库自写的 8 条 smoke JSONL，使用 answer-only loss。

## GPU 占用归属检查

在启动 LoRA 前检查了 NVML compute-apps、`nvidia-smi pmon`、当前容器的进程命令和可读 `/proc/*/fd`：

- 没有发现之前 `train_sft.py`、`train_qwen.py`、torchrun 或 Engram benchmark 的残留进程。
- NVML 和 pmon 均没有可见 GPU PID；可读进程中没有持有 NVIDIA/DRI 设备句柄的进程。
- 8 张卡的 `memory.used` 与前一轮 SFT 开始前的 `artifacts/l20d/preflight.json` **逐卡完全相同**。
- 部分平台 root 进程及 uid 1000 的 java 进程 FD 不可读；这不是它们占用 GPU 的证据。
- 当前容器无法确定宿主机/其他 namespace 的占用者，也无法区分平台预留。因此没有证据把这些显存归因于某个其他用户。

**本次未终止任何进程，未执行 GPU reset。** 未发现可确认属于之前任务、需要清理的残留训练程序。
训练前 GPU 0、1 的实际 CUDA 可用显存均为 75,692,048,384 字节（约 70.494 GiB）。
其他卡可用量较低，未用于本次验证。

原始归属证据保存在远端仓库的 `artifacts/lora-20261013/gpu-before.json`、
`blocked-fds.json`、`gpu-ownership-conclusion.json` 和 `cuda-before.json`。

训练及完整回归全部退出后，8 卡 `memory.used` 仍与启动前一致；GPU 0、1 的 CUDA
可用字节数也精确恢复到 75,692,048,384。可见训练进程和 GPU FD 进程均为空。
未观察到本次 LoRA 残留训练进程或显存占用增加。证据为 `gpu-after.json`、`cuda-after.json`
与 `post-run-checks.json`。

## 实际配置

| 项目 | 设置 |
| --- | --- |
| 基座 | `/mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash` |
| 选择层 | `start_layer=0, num_layers=1`，无 Engram 表 |
| 并行 | GPU 0、1；world=2，EP=2，CP=1，FSDP2 |
| 权重/加载 | BF16 原生加载，`layerwise`；无 CPU offload、无 QLoRA |
| LoRA | q_a、q_b、kv_proj、o_b；rank=8，alpha=16，dropout=0.1 |
| 可训练参数 | 475,136 个，8 个 A/B 张量；950,272 字节 |
| 优化器 | AdamW，lr=0.001，weight_decay=0 |
| 数据 | `examples/sft_smoke.jsonl`，8 条，sequence length=128 |
| batch | 每卡每 microbatch 1 条，累积 2 次 |
| 其他 | full activation checkpointing；seed=20261013 |
| 保存 | `save_every=0`；每次调用仅保存末步完整 DCP，同时导出 adapter |

两组训练为：连续训练 5 步；独立训练 3 步后，启动新进程从 step-3 恢复到 step-5。
另外启动新进程，从原始基座重新加载导出的 adapter，比较训练进程保存的固定 eval 参考。

## 已通过的正确性检查

1. 所有基座参数 `requires_grad=False`、`grad=None`；optimizer 恰好包含全部 A/B 参数，没有基座、重复或漏项。
2. 每 rank 407 个基座参数张量，合计 205,942 个有界采样值在训练中不变；没有复制整份基座进行比较。
3. A/B 梯度有限，存在真实参数更新。首步 B=0 导致 A 梯度为 0，第二步起 A/B 均更新。
4. 连续训练与恢复训练：全部 8 个 adapter 张量、475,136 个元素逐元素精确一致，最大绝对差为 0；5 步全部 loss、CE、aux 和 token 计数一致。
5. 两种训练路径的最终固定 eval loss 和每 rank 256 个 logits 采样精确一致。
6. 独立重载原始基座 + 导出 adapter 的固定 eval loss/logits 采样，与训练进程参考精确一致（`atol=rtol=0`），且相对训练前输出发生变化。

adapter 文件分别以 SHA256 关联各自参考，跨运行比较的是完整张量值，不依赖 pickle 文件序列化字节相同。
固定 eval 在 `fork_rng` 中运行，并恢复各模块 training 标志；dropout 的随机序列没有被验证步骤改变。

连续训练的观测：

| 步数 | 全局 loss | A 梯度 L2 | B 梯度 L2 | 基座采样最大变化 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16.621738 | 0 | 0.317643 | 0 |
| 2 | 15.523964 | 0.743870 | 0.374592 | 0 |
| 3 | 14.820013 | 1.841913 | 1.537601 | 0 |
| 4 | 16.602742 | 2.622196 | 2.880691 | 0 |
| 5 | 13.016703 | 3.446698 | 4.292896 | 0 |

不同训练步使用不同 batch，不能用这几步 loss 作为效果提升结论。
A/B 相对初始采样最大变化分别为 0.0034637451 和 0.0050048828。
连续训练每卡最大 allocated 显存为 **27.258 GiB**，reserved 为 **28.842 GiB**；加载后约 14.053 GiB。
短序列 smoke 的吞吐不可外推至完整模型/长序列。

## 回归与存储

- 远端针对性回归：39 项全部通过。
- 远端完整回归：202 项，197 通过、5 项缺少 Transformers 的 HF parity 测试跳过，退出码 0。
- 本地完整回归：202 项，191 通过、11 跳过；另 6 项需要 CUDA，退出码 0。
- 新增三份完整 DCP 分别为 30,174,778,400、30,174,778,396、30,174,778,398 字节，合计约 90.524 GB。
- 保存后工作盘可用 47,774,044,160 字节（约 44.493 GiB）；未删除之前的 checkpoint。

本次运行脚本的初始磁盘门槛为 100 GiB。当前剩余空间不足以原样重复三份 DCP
实验，需改用更大存储或另行安排 checkpoint 留存；不可直接重复执行。

## 复现与产物

远端仓库：

```text
/path/to/Tiny-DSV41
```

完整执行脚本保存在 `artifacts/lora-20261013/run-validation.sh`；从仓库根目录执行。
脚本有 GPU 空闲量、磁盘空间和输出目录防覆盖检查。再次运行应先改用新输出目录，并重新核算磁盘。
训练命令核心参数如下（每次运行的 steps/output/resume/adapter-out 由脚本分别设置）：

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONPATH=.:tests PYTORCH_ALLOC_CONF=expandable_segments:True \
.venv-posttrain/bin/python -m torch.distributed.run \
  --master-addr=127.0.0.1 --master-port=29521 --nproc-per-node=2 \
  train_sft.py --model-type dsv4 \
  --model-path /mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash \
  --data examples/sft_smoke.jsonl --output-dir runs/NEW_LORA_RUN \
  --start-layer 0 --num-layers 1 --ep-size 2 --cp-size 1 \
  --seq-len 128 --batch-size 1 --gradient-accumulation-steps 2 \
  --optimizer adamw --learning-rate 0.001 --weight-decay 0 \
  --checkpoint-mode full --load-mode layerwise --staging-mib 128 \
  --seed 20261013 --save-every 0 --steps 5 \
  --lora --lora-rank 8 --lora-alpha 16 --lora-dropout 0.1 \
  --verify-lora --adapter-out runs/NEW_LORA_RUN/adapter.pt
```

- `runs/dsv4-prefix1-lora-continuous-01/`：连续训练的 step-5、adapter、metrics、summary、eval reference。
- `runs/dsv4-prefix1-lora-resume-01/`：step-3、恢复后的 step-5、最终 adapter 和完整 5 步 metrics。
- `artifacts/lora-20261013/resume-parity.json`：全部 adapter 元素、逐步指标和最终 eval 精确比较。
- `artifacts/lora-20261013/adapter-reload.json`：独立新进程重载结果。
- 同目录 `continuous/split/resume/compare/reload.log` 及 `.exit`：各阶段日志，退出码均为 0。
- `runner.log` 最终标记：`LORA_VALIDATION_ALL_PASS`。

新增源码在上传后逐文件校验 SHA256；首次验证包 SHA256 为
`886888c0ddb0431d16e128d3314c6d252ee775e4a55a8f2215bcef4f904d9f5a`。
基线 Git 提交仍为 `3868011979caaa3358bfd779ec78367df1d01205`；修改保留在 `codex/posttrain-sft` 工作树，未提交/推送。

## 限制

这是单层前缀的 LoRA SFT 验证；不覆盖完整 DeepSeek 40 层、Engram LoRA、后续层压缩 attention、QLoRA 或长序列训练。
全量模型的 BF16 基座仍超出当前资源；LoRA 减少梯度/optimizer 开销，但不消除基座权重容量。
基座值与 eval logits 是采样验证，adapter 最终值是全部元素比较。
checkpoint identity 校验配置/index，而不是所有原始权重 shard 的全量 hash；实验中需保持挂载权重不变。
完整 DCP 仍保存冻结基座，因此显存/adapter 很小也不代表训练 checkpoint 很小。
