# L20D 远端验证进展 — 2026-10-13

> 发布说明：此页记录历史实验。模型权重、adapter、原始日志和 `artifacts/` 证据保留在实验环境，不随 Git 仓库发布；下文标注“本地证据”的路径不是仓库下载链接。通用训练用法见 [SFT 指南](sft.md)。

## 当前状态

**远端代码和依赖已部署；两卡 NCCL、28 项回归、四种 Engram 分布策略对齐及八组 CUDA row-store 稳定性测试通过。完整 Qwen SFT、恢复、五组 Engram 策略以及 DeepSeek 真实前 1/2 层（含 98.3B Engram）SFT 和保存均通过。完整 DeepSeek 40 层未跑通：当前资源无法容纳现有 BF16 训练路径。**
本记录区分实际观测、源码容量估算和待执行步骤。小模型功能验证不能替代完整
Qwen/DeepSeek checkpoint 的加载、训练和恢复结果。

后续 [LoRA 实机验证记录](lora_validation_20261013.md) 补充了 GPU 占用归属检查、
真实 prefix1 的 LoRA SFT、连续/恢复逐元素一致性和独立 adapter 重载结果。

通过 Codex 内置浏览器（IAB）的新会话重新连接后，可以操作 Web IDE 终端。
此前浏览器会话失效是当时的访问问题，目前不再构成持续阻塞。
[本地验证记录](local_validation_20261013.md) 的旧远端访问状态只反映当时情况；
本文件更新远端进展，不把本地 CPU 测试结果视为远端 CUDA 结果。

## 已观察的远端环境

| 项目 | 实际观测 |
| --- | --- |
| 克隆分支 | `feature/engram-support` |
| 远端克隆提交 | `3868011979caaa3358bfd779ec78367df1d01205` |
| Python | `3.10.9` |
| PyTorch | `2.11.0+cu130` |
| PyTorch CUDA 版本 | `13.0` |
| `torch.cuda.is_available()` | `True` |
| 主机内存 | 约 480 GiB，available 约 479 GiB |
| 工作盘空间观测 | `df -h` 显示约 `509G`；保存前以字节数复查 |
| Qwen 模型目录 | `/mnt/fuse/Qwen/Qwen3-30B-A3B` |
| DeepSeek 模型目录 | `/mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash` |
| 模型元数据 | 两个目录的 config/tokenizer 均可读；不等于所有权重已加载验证 |
| 依赖安装 | 独立 `.venv-posttrain`，继承已有 PyTorch；tokenizers 0.23.3、safetensors 0.8.0 安装完成 |

远端仓库绝对路径：

```text
/path/to/Tiny-DSV41
```

以上提交标识描述远端克隆基线。远端开发分支为 `codex/posttrain-sft`，已应用
19 文件源码包（SHA256 `068779dbff5f9c70083f2f8fa0791a5b3cedd118dc68e7908e539bce5f00de37`），
包含 SFT、Qwen Engram 配置、checkpoint 修复与 RNG 修复。源码包在应用前校验
基线提交、原 tracked 工作树和 SHA256。依赖包传输完成后也校验 SHA256。

### CUDA 显存观测

下表来自 `torch.cuda.mem_get_info()`，GiB 按字节数除以 `2**30` 计算：

| GPU 编号 | 可用字节 | 可用 GiB |
| --- | ---: | ---: |
| 0、1，各卡 | 75,687,854,080 | 70.490 |
| 2、3，各卡 | 4,122,476,544 | 3.839 |
| 4、5，各卡 | 16,118,251,520 | 15.011 |
| 6、7，各卡 | 4,269,342,720 | 3.976 |

GPU 0、1 的 CUDA 总量报告为 `287428771840` 字节，即约 **267.689 GiB/卡**。
此前 `nvidia-smi` 的总量与空闲值和本次 CUDA 查询并不相同；报告方式、虚拟化
或共享机制尚未核实，不能按假定的“八张 48 GiB 实体卡”解释这些数值。
当前选择设备以 CUDA 查询及实际分配/训练结果为依据，不能将报告总量视为本任务
可独占的显存。

`nvidia-smi` 的 compute-app 列表为空。**这不证明其他租户没有占用**，也不能据此
忽略其他卡的低空闲量。首次验证只考虑当时空闲较多的 GPU 0、1；启动前重新查询。

## Qwen 两卡容量预算（实测见后文）

在本地以仓库 Qwen 配置构建 meta 模型，按 EP=2、CP=1、普通参数 FSDP=2 枚举参数。
这是源码计算，前提是远端 checkpoint 结构与该配置相同：48 层、128 个专家、
hidden size 2048、专家中间维度 768。若远端结构不同，需要重新计算。

全局参数量为 **30,532,122,624**。每卡拥有 14,495,514,624 个专家参数，
以及 1,541,093,376 个普通参数的一半。专家 FSDP 组大小为 1，专家权重通过 EP
分属两卡；普通参数通过两卡 FSDP 分片。

| 每卡项目 | 估算 GiB |
| --- | ---: |
| BF16 参数 | 28.435 |
| BF16 梯度 | 28.435 |
| 参数和梯度合计 | **56.871** |
| 本次 CUDA 空闲值扣除上述静态占用 | **13.619** |
| 再加两个 BF16 AdamW moment 后 | **113.741** |

当前原生 SGD 使用零 momentum，不维护整份 FP32 master weights；因此建议从
**全参数 BF16 SGD、小 batch、短序列、完整 activation checkpointing** 开始。
上述约 13.6 GiB 余量仍需容纳激活、logits、FSDP 临时参数、梯度临时量、NCCL
及 allocator 开销，不能据此宣称已能跑通。

旧 `train_qwen.py` 默认 AdamW，不适合这次容量预算。使用新增 `train_sft.py`
显式指定 `--optimizer sgd`；该文件须先同步到远端并确认依赖可用。

默认 Qwen Engram 两表共 11,063,242,752 个参数，EP/行分片为两卡时，GPU 常驻表
额外增加约 **10.303 GiB/卡**，连同投影/门控后会把静态底座推至约 67.3 GiB/卡，
当前余量偏紧。因此先验证裸 Qwen或缩小的 Engram 表，再考虑 CPU 大表卸载。
默认大表卸到 CPU 后，全局 BF16 表权重约 20.607 GiB，另需命中行梯度和 staging。

SGD 完整 Qwen checkpoint 的参数数据约 **56.871 GiB**，不含文件及元数据开销；
增加保存次数会累积磁盘占用。输出写入独立工作目录，不写回挂载的模型源。

## 最小验证顺序与复现配置

以下命令描述验证顺序；实际执行状态以文末结果表为准。从远端仓库根目录运行；`python3` 应指向上述已检查
的 CUDA Python 环境，并保留其所需 `PYTHONPATH`。所有步骤执行前重新确认 GPU 0、1
的空闲显存。依赖安装、工作树同步和输入准备完成后再运行模型测试。

```bash
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/tests${PYTHONPATH:+:$PYTHONPATH}"
```

### 1. 两卡通信和小模型正确性

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/distributed_nccl_smoke.py \
  --expected-nodes 1 --expected-local-world-size 2 \
  --elements 1048576 --warmup 1 --iterations 2

python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/qwen_engram_distributed.py

python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/dsv4_post_training_fsdp.py
```

`qwen_engram_distributed.py` 的注释提到八卡，但实现采用实际 world size，8 个小模型
专家可按 EP=2 分配；参考 Engram 表也由实际各 rank 分片拼接，支持两卡。
它覆盖 GPU、GPU cache、CPU async、CPU async cache 四种策略及多步累积后的
输出、梯度、参数更新对比。`dsv4_post_training_fsdp.py` 明确要求两卡。

`tests/qwen_parallel_parity.py` 也支持且要求两卡，但 `tests/qwen_fsdp_smoke.py`
硬编码要求八卡，不应直接改成 `--nproc-per-node=2` 运行。

### 2. 真实 Qwen 全参数 SGD

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  train_sft.py \
  --model-type qwen \
  --model-path /mnt/fuse/Qwen/Qwen3-30B-A3B \
  --data examples/sft_smoke.jsonl \
  --output-dir runs/qwen-ep2-sft-01 \
  --ep-size 2 --cp-size 1 \
  --seq-len 256 --batch-size 1 --gradient-accumulation-steps 1 \
  --optimizer sgd --learning-rate 0.01 \
  --checkpoint-mode full --steps 2
```

该数据只有 8 条自编指令，采用文档中明确的 plain-completion 格式。
本步骤验证执行、有限 loss/梯度、实际参数更新及 checkpoint 保存，不证明训练质量。
保存完成后，使用相同配置和目录，将 `--steps` 改为 3 并加 `--resume`，单独记录
真实 GPU checkpoint 恢复及下一步更新结果。

### 3. Engram 放置、预取和 cache 矩阵

优先沿用两张表所在层 1、24，缩小为 `--vocab-per-head 4096 --head-dim 64`。
这两张表共 13,221,376 参数；两卡时每卡 BF16 表权重约 12.61 MiB，适合功能验证，
不足以测量大表卸载的容量收益。

| 项目 | 基准 `--variant` | `--prefetch` | 状态 |
| --- | --- | --- | --- |
| 裸 Qwen SFT | 单独使用 `train_sft.py` | 不适用 | 两步及恢复通过 |
| Qwen + GPU Engram | `gpu` | `sync` | 通过 |
| Qwen + CPU Engram | `cpu` | `sync` | 通过 |
| Qwen + CPU 异步预取 | `cpu` | `async` | 通过 |
| Qwen + CPU 异步及 cache | `cpu_cache` | `async` | 通过 |
| 可选 GPU cache | `gpu_cache` | `sync` | 通过 |

`tests/qwen_engram_sft_benchmark.py` 始终添加随机初始化 Engram；其中 `gpu` 不是裸
Qwen。初轮设置 batch=1、microbatches=2、warmup=1、steps=2、EP=2、seq=128，
staging=16 MiB/table/rank、cache=1 MiB/table/rank。同步与异步分别保存到独立输出目录。

基准需要含 `input_ids`、`labels`、`metadata` 的本地 `.pt`；按上述两卡配置至少需
12 行，开启独立 trace 时至少需 16 行。首位置标签必须全部为 `-100`，每个输入行
至少保留一个移位后监督目标。入口不接收 attention mask，宜使用无 padding 的完整
packing。现有 `tests/prepare_qwen_sft.py` 从本地 Alpaca Parquet 生成数据，额外依赖
`pyarrow`；不应把缺失的语料当成已经就绪。若先使用重复 smoke 数据，需明确记录重复
及用途，不把它的 cache 命中率或吞吐作为真实语料结论。

现有矩阵 runner 不暴露 prefetch、表大小及 cache 预算，初轮应直接调用基准入口。
现有 `tests/summarize_qwen_engram.py` 把环境字符串写死为 H20，并按 `variant` 将
CPU sync/async 混为一组；修改汇总逻辑前，不直接用它发布本机矩阵结果。

## 完整 Qwen 首轮结果

首次运行在加载完整 48 层后，训练前的参数 probe 触发 CUDA 越界断言。
原因是 `torch.linspace` 的 float32 端点在大于 `2**24` 的扁平张量上可能将
`numel - 1` 舍入到 `numel`。已改为精确 int64 商余数采样；边界测试无需分配
巨大张量，覆盖真实 Qwen 分片大小 155582464 及 int64 上界。修复和 smoke packing
相关 19 项远端测试通过。原失败日志保留为 `artifacts/l20d/qwen-baseline.log`。

修复后运行目录 `runs/qwen-ep2-sft-02`，完整 30,532,122,624 参数、两卡 EP2、
BF16 SGD、seq256、batch1/rank、accum1、完整 activation checkpointing：

| 指标 | 实测 |
| --- | ---: |
| 第 1 步 loss | 2.1348865032 |
| 第 2 步 loss | 1.0630492653 |
| 参数 probe 最大变化（第 2 步相对初始） | 0.1442260742 |
| 模型加载 CUDA peak | 29.5774 GiB/卡（各 rank 最大值） |
| 训练 CUDA allocated peak | 58.1191 GiB/卡 |
| CUDA reserved peak | 59.9121 GiB/卡 |
| 最终 checkpoint | `runs/qwen-ep2-sft-02/step-2`，`.metadata` 完整 |
| launcher exit | 0 |

只有 8 条自编样例，两步使用不同 batch；loss 数值用于确认执行，不构成收敛或质量改善证据。
同目录 `--resume --steps 3` 已完成，launcher exit 0；summary `start_step=2, steps=3`，
`step-3/.metadata` 存在。第 3 步 loss 8.3064059258、15 个监督 token，参数 probe
变化 0.26220703125。该步使用另一个 batch，不能用跨 batch loss 判断收敛。
训练 CUDA allocated peak 58.1191 GiB、reserved peak 59.8926 GiB。

## 结果与日志

| 检查 | 当前状态 | 后续应记录 |
| --- | --- | --- |
| 依赖安装与导入 | 通过，exit 0 | `artifacts/l20d/install.log`、`preflight.json` |
| 远端工作树同步 | 19 文件源码包已校验并应用 | 基线及包 SHA256 见上文 |
| CUDA/NCCL 两卡小测试 | 通过，exit 0，GPU 0/1，NCCL 2.28.9 | `artifacts/l20d/nccl.log` |
| CUDA/CPU 单测 | 28 项全部通过，无跳过，58.878 秒 | `artifacts/l20d/unit-engram.log` |
| Qwen/Engram 小模型 parity | 四策略通过，exit 0，3 步×2 次累积，world 2 | `artifacts/l20d/qwen-engram-distributed.log` |
| CUDA row-store 生命周期 | 8 组合×24 步通过，exit 0；缓存对 CPU 控制组完全一致 | `artifacts/l20d/engram-store.json`、`.log` |
| 真实 Qwen SGD | 两步通过，exit 0，有限 loss/梯度、非零更新 | `artifacts/l20d/qwen-baseline-fixed.log` |
| GPU checkpoint 保存/恢复 | step-2 保存、恢复并保存 step-3 通过，exit 0 | `artifacts/l20d/qwen-resume.log` |
| 真实权重 Engram 矩阵 | 五组 exit 0、自动比较通过 | `artifacts/l20d/engram-matrix/` |
| DeepSeek 真实权重训练 | 前 1/2 层均两步及保存通过，exit 0 | `dsv4-prefix1.log`、`dsv4-prefix2.log` |

后续报告须将 exit code 与模型级检查一起验收。已强化 Engram benchmark：每步检查 loss/梯度有限值，warmup 验证 dense 和每张表
触达行的非零更新，全部通过后才写 `validation_passed=true`。任何已运行小模型
或短序列结果，也不代表完整 DeepSeek checkpoint 已训练成功或支持全部 post-training 算法。

## DeepSeek 真实量化抽样

`scripts/inspect_dsv4_quantization.py` 在远端 PyTorch 2.11.0+cu130 上 exit 0、
`passed=true`。读取 attention FP8 block、expert packed FP4 rows、Engram FP8 rows
各 32 行，三组解码到 BF16 后全部有限，总逻辑读取 259,488 字节。解码 shape 分别为
`[32,5120]`、`[32,5120]`、`[32,256]`；非零数分别 163,839、144,345、8,192。
这验证真实格式兼容性，不是完整模型/HF 数值对齐。日志：
`artifacts/l20d/dsv4-quantization.json`。矩阵检查和 nested config 的 11 项远端单测通过。

完整 40 层的权重下界及前缀测试预算见 [容量分析](deepseek_capacity_20261013.md)。

## 完整 Qwen + 缩小 Engram 矩阵实测

完整 48 层 Qwen 权重，随机初始化第 1/24 层的缩小 Engram；EP2、batch1、
microbatches2、warmup1、timed steps2、seq128、SGD 0.01、foreach=false。
64×128 packed smoke 数据，8 条作者样例重复填充，首标签屏蔽，没有 padding。
五组 launcher exit 均为 0，summary `validation_passed=true`；
`scripts/compare_engram_matrix.py` 自动检查通过，记录于 `comparison.json`。

| 策略 | 第 1/2 个计量步全局 loss | CUDA allocated peak GiB |
| --- | --- | ---: |
| gpu | 0.756497234 / 0.387386039 | 59.1437 |
| cpu_sync | 0.760850579 / 0.390312135 | 59.1307 |
| cpu_async | 0.760850579 / 0.390312135 | 59.1307 |
| cpu_cache | 0.760850579 / 0.390312135 | 59.1326 |
| gpu_cache | 0.756497234 / 0.387386039 | 59.1456 |

CPU sync/async/cache 三组 loss 相同；GPU/GPU cache 两组 loss 相同。CPU/GPU 的
小幅差异通过预设 BF16 容差检查（loss atol=.02/rtol=.01，参数/表样本
atol=.0002/rtol=.02），没有要求不同稀疏累加路径逐位相等。每 rank 的有限参数样本
和以 row ID 对齐的表样本也通过比较；这不是全参数逐元素参考证明。

两张表 warmup 触达行均观察到更新，全局最大变化分别为 0.000244140625 和
0.0001220703125。CPU offload 的 H2D/D2H 均非零。以 rank0 为例，两张 CPU cache 表
分别命中 4,693/4,730 行、失效 3,366/3,414 行；值传输 H2D 分别 520,320/517,120 字节，
未缓存 CPU 为 1,354,112/1,357,952 字节，另有 cache metadata 传输。
这些数据只用于验证缓存路径确实执行，不用于比较真实语料性能。

远端证据包已通过 WebIDE 下载并校验 SHA256：
`70293652ea69c92e222430dda0d78a19e47d49fbcc8b0688e90c18af9e8cd3e6`，
本地副本位于 `artifacts/remote-evidence/l20d-evidence-qwen.tar.gz`（不含模型权重）。

## DeepSeek 真实权重前缀训练

共同配置：挂载 V4.1 Flash 真实权重、start=0、EP2/CP1、BF16 SGD 0.01、
seq128、batch1/rank、accum1、完整 activation checkpointing、`direct_shard` +
load pipeline、staging128MiB；最终保存一次。没有修改模型挂载源。

先执行 `tests/dsv4_post_training_fsdp.py`：none/full/selective 三类 activation
checkpoint 各覆盖全参数/LoRA，六组两卡输出、loss、梯度、更新及 adapter-export
对齐通过（exit 0，2 步×2 microbatches）。随后运行真实 checkpoint：

| 指标 | 第 0 层 | 第 0/1 层 + 原始 CPU Engram |
| --- | ---: | ---: |
| SFT steps | 2 | 2 |
| 第 1 步 loss | 15.8239563533 | 15.3409102304 |
| 第 2 步 loss | 13.6915040970 | 13.1925859451 |
| load seconds，rank0 | 40.143 | 272.716 |
| load CUDA peak GiB | 14.0437 | 27.0030 |
| train CUDA allocated peak GiB | 34.0672 | 50.2523 |
| CUDA reserved peak GiB | 41.3652 | 57.5625 |
| dense probe 最大变化 | 0.2809636295 | 0.1400196850 |
| launcher exit | 0 | 0 |

输出：`runs/dsv4-prefix1-sft-01/step-2` 与
`runs/dsv4-prefix2-cpu-engram-01/step-2`，两者 `.metadata` 和 summary 均完整。
源码预算的参数 checkpoint 大小约 28.087 / 237.110 GiB；第二轮保存后磁盘
剩余 138,311,770,112 字节。为避免复制第二份大表 checkpoint，本轮不执行 prefix2 resume。
恢复路径有小表 CPU DCP 精确下一步测试及完整 Qwen 实机恢复结果，但不能冒充原始
183 GiB CPU 表的恢复实测。

两层模型包含 **98,305,579,008 个真实预训练 Engram 参数**，BF16 表约
183.108 GiB、按两 rank 行分片驻留 CPU；启用 async prefetch 和 deduplicate，
本轮关闭 row cache。第 2 步聚合稀疏梯度 nnz=1,072，探测 512 行，最大实际表更新
0.000244140625；rank0 共 lookup 1,132 行，H2D/D2H 各 579,584 字节。
没有把缩小随机表当作原始表测试。

两层启动前两卡各有 75,692,048,384 字节空闲，主机 available 为
488,621,797,376 字节，磁盘空闲 392,741,281,792 字节。后台每 10 秒采样
`MemAvailable`，共 54 个样本，最低约 189.976 GiB；这是采样下界，
不等同于进程峰值 RSS 或完整连续内存追踪。记录：`dsv4-prefix2-capacity.json`、
`dsv4-prefix2-host.jsonl`。

26 项通用 Engram probe/矩阵比较远端测试通过，包含 DSV4 CPU sparse 更新、
DCP 恢复及下一步精确一致。完整 40 层权重展开约 1394.253 GiB，已经超过本机
当时所有空闲 GPU+host 总和，未盲目启动完整模型 OOM。这里只验证文本 backbone
前缀；没有覆盖真实压缩 attention 后续层、vision、next-token heads、HF reference
数值对齐或模型质量，不实现 GRPO/DPO。

## 最终回归与源码交付

- 本地最终单测：179 项，168 通过、11 跳过，6.855 秒，exit 0。
- L20D 最终单测：179 项，174 通过、5 跳过，18.471 秒，exit 0。
- 远端 5 个跳过项为不可用的 DeepSeek HF reference；不声称 HF parity 通过。
- 独立只读代码 review 未发现可确认的新 P1/P2 问题。
- 新增 `scripts/run_engram_smoke_matrix.sh` 固化五组实跑配置，拒绝非空输出、
  失败即停并运行自动比较。`bash -n` 与 13 项脚本边界检查通过。
- 所有源码位于本地/远端 `codex/posttrain-sft` 工作树；未推送 GitHub。

最终日志 `artifacts/l20d/final-remote-unittest.log` 和 `.exit`；本地完整单测日志
`artifacts/final-cpu-unittest.log`。最终源码以 SHA256 manifest 校验同步。
