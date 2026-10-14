# DeepSeek 真实权重容量核算 — 2026-10-13

> 发布说明：此页记录历史实验。模型权重、adapter、原始日志和 `artifacts/` 证据保留在实验环境，不随 Git 仓库发布；下文标注“本地证据”的路径不是仓库下载链接。通用训练用法见 [SFT 指南](sft.md)。

当前原生加载路径会把量化 checkpoint 解码为可训练的 BF16 参数。
按仓库对应配置在 `meta` 上枚举，40 层文本模型需要约 **1394.25 GiB 权重**。
这已经超过当前主机和 GPU 的可用内存总和，因此无需尝试完整加载来确认容量不足。
可执行的下一步是分别验证 **第 0 层前缀**和**第 0–1 层前缀加真实 CPU Engram**。
本文给出源码容量估算和启动方案；不把它们当作 DeepSeek 实际训练峰值或成功记录。

## 核算范围与资源口径

配置取自 `dsv41_train/models/dsv4/config.json`：hidden size 5120，384 个路由专家，
专家中间维度 2304，40 层；Engram 位于零起始层号 1、14，行数分别为
384,006,168、384,016,682，宽度 256。两张表合计 **196,613,849,600 参数**，
与本次远端读取到的表容量一致。

计数范围是当前 `DeepSeekV41ForCausalLM` 真正构建的 40 层文本 backbone、
输入 embedding、输出 LM head、norm、attention/HyperConnection、MoE 和两张 Engram 表。
**不包括 vision encoder、额外三个 next-token prediction 层，也不包括未实现的
DSpark 专用模块。** 因此 748.495B 是本实现范围的参数量，不是对发布的多模态
checkpoint 所有模块的完整参数声明。外层 `vision_config`、`num_nextn_predict_layers`
及额外压缩比例条目不会让当前模型自动包含这些模块。

远端当时可用资源为 host 约 479 GiB、GPU 0/1 各约 70.49 GiB；其余六卡分别只有
约 4/4/15/15/4/4 GiB 空闲。即使把全部这些空闲量相加，也只有约 665.98 GiB。
这是容量比较，并不授权占用其他卡；实际启动继续只使用已选 GPU 0/1。

磁盘初次 `df -h` 显示总量约 `512G`、可用约 `509G`，**保留这个显示口径，不将
`G` 擅自解释为十进制 GB**。保存前应使用 `df -B1` 或
`shutil.disk_usage(...).free` 获取当前可用字节数。下文 GB 表示字节数除以 `10**9`，
GiB 表示字节数除以 `2**30`。

挂载源 index 报告约 510,286,023,000 字节、48 个 shard。它是量化后的源文件体积，
不能用于替代展开后的训练参数内存。

## 完整 40 层：现有路径不能容纳

使用 `meta` 构建模型后，按 `direct_shard` 的 dtype 规则计数：普通参数为 BF16，
HyperConnection 和 attention sinks 保留 FP32；不读取实际权重，不分配这些张量的数据。

| 参数类别 | 参数量 | 展开后的参数字节数 |
| --- | ---: | ---: |
| 普通参数，含 embedding/head | 8,299,006,064 | 16,676,664,768 |
| 路由专家 | 543,581,798,400 | 1,087,163,596,800 |
| 两张 Engram 表 | 196,613,849,600 | 393,227,699,200 |
| 合计 | **748,494,654,064** | **1,497,067,960,768** |

权重合计约 **1.497 TB / 1394.253 GiB**；上述数字尚未加入梯度、优化器状态、
激活、通信缓冲和 checkpoint 临时量。即使冻结所有基座参数，权重本身也放不下。

| 当前已有策略 | 单独权重的结果 |
| --- | --- |
| 仅把完整两张 Engram 表放 CPU | CPU 366.222 GiB；其余权重在两卡上各约 514.016 GiB |
| 全部 decoder 层及 Engram 都放 CPU | host 约 1391.787 GiB；embedding/head 等 root 权重还需 GPU 全局约 2.466 GiB |
| LoRA，基座冻结 | 降低梯度/优化器需求，仍保留上述 BF16/FP32 基座权重 |

此外，当前 loader 明确拒绝 LoRA 与 `direct_shard`、CPU Engram 或 FSDP CPU
offload 组合。代码也没有保持 FP4/FP8 基座的 QLoRA 训练算子、NVMe 参数分页或
逐层从磁盘换入换出的训练路径。`cache_quantization` 作用于 attention cache，
不会把已解码的全模型权重重新变成低比特参数。

因此，在现有实现与当前空闲资源下，完整 40 层全参数训练和全模型 LoRA 都不可行。
`meta` 上的权重下界已足够作出这个判断；贸然加载只会消耗资源，不能解决缺口。

## 前缀 1 层与 2 层

以下按 world=2、EP=2、CP=1，普通参数 FSDP=2、expert FSDP=1、SGD 无 momentum
估算。保留 FP32 模块，表使用稀疏梯度。梯度栏假设全部非表参数都有同 dtype 梯度，
是静态预算；不包含激活和临时量。

| 项目 | `start=0,num_layers=1` | `start=0,num_layers=2` |
| --- | ---: | ---: |
| 实际 backbone 层 | 第 0 层 | 第 0、1 层 |
| Engram | 无 | 第 1 层原始预训练表 |
| 全局参数量 | 15,078,343,542 | 127,295,761,132 |
| GPU 权重 / rank | 14.044 GiB | 27.001 GiB |
| GPU 权重加梯度 / rank | 28.087 GiB | 54.002 GiB |
| CPU 表权重 / rank | 0 | 91.554 GiB |
| CPU 表权重全局 | 0 | 183.108 GiB |
| 单次 SGD checkpoint 参数和持久 buffer 数据 | 30,158,654,168 字节 | 254,595,456,432 字节 |
| 同一 checkpoint 的 GB / GiB | 30.159 / 28.087 | 254.595 / 237.110 |

2 层方案在每卡 70.49 GiB 空闲的前提下，静态权重加梯度之后约剩 **16.49 GiB/卡**，
供短序列激活、logits、FSDP 临时参数、NCCL 和 allocator 使用。CPU 表全局占用约
183.11 GiB。该预算支持进行小规模实测，但不是已测峰值，也不保证资源发生变化后仍能运行。

这两个前缀使用真实 embedding/head 和选定层权重，适合验证量化加载、SFT 更新、
Engram 稀疏更新及 checkpoint；移除中间大部分 backbone 的模型不代表完整模型的生成质量。
前两层 `compress_ratios` 都为 0，也没有覆盖第 2 层以后压缩 attention 的真实权重训练。

## direct_shard 的 CPU 临时内存

`FSDPMaterializer.prepare` 在 meta 模型上建立 FSDP/EP/行分片，然后只对最终的本地
参数分片 `to_empty`。GPU 目标分片直接分配到 GPU；CPU Engram 只分配本 rank 的
行区间。**不会先在 CPU 构建整张 183 GiB Engram 表，再复制两份进行切分。**

`WeightSource.fill` 的解码行数上限为
`staging_bytes // (16 * decoded_row_bytes)`。`--staging-mib 128` 时，BF16 宽度
256 的 Engram 解码块至多 16,384 行，单份解码结果为 8 MiB。FP4 专家同样按目标
行字节数限制块大小；中间 nibble 索引、FP32 解码乘法和拷贝只覆盖当前块。
pipeline 只有一个读取 worker 和有限的当前/下一块，不积累整层或整表解码结果。

因此 2 层方案加载时，模型自有的主要 host 分配是两 rank 共 **183.108 GiB** 的表
权重，加上按块的解码工作区；单 rank 的解码工作区是几十至数百 MiB 量级，而非另一份
91.554 GiB 表。128 MiB 是加载策略预算，不是整个进程 RSS 的严格上限。

源 shard 使用只读用途的 mmap，文件的虚拟映射大小不等于常驻物理内存。访问过的
量化页、FUSE/page cache、tokenizer、数据和运行库仍会增加 host 使用；两 rank 的
RSS 可能含共享映射，不能简单相加。保存时还可能暂存 GPU 参数的 CPU 副本。
当前结构没有强制创建第二张完整 CPU 表的步骤，但实际峰值须在加载和保存阶段采样确认。
479 GiB host 空闲对前述分片表与分块工作区有明显余量。

### DCP 保存和恢复的不同临时量

这里使用同步 `dcp.save`，默认 `TORCH_SAVE` 格式。仓库 `StateAdapter`/`TensorShard`
将表的本地 storage 交给 writer，`direct_shard` 的 `to_empty` 分配使该 storage 恰好
覆盖本地表。PyTorch 2.11 writer 只有在底层 storage 大于张量范围时才 clone CPU
张量，因此这里不会再复制两 rank 共 183.108 GiB 的表。
不过 writer 的 `tensor_dict` 会保留已搬到 CPU 的 GPU 参数，保存时不能只按
10 MB copy-ahead 预算估算临时量。[PyTorch 2.11 writer 源码](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/distributed/checkpoint/filesystem.py#L175-L184)、[张量引用留存](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/distributed/checkpoint/filesystem.py#L406-L424)。

`torch.save` 对 CPU storage 直接写记录；C++ 的 Python 文件接口传递 memoryview，
也不会先构造一份整表 Python bytes。[序列化实现](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/serialization.py#L1177-L1210)、[memoryview 写入实现](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/csrc/jit/python/init.cpp#L1388-L1410)。
因此，prefix2 保存阶段的 host 参数数据预算为 **183.108 GiB 原有 CPU 表 +
54.002 GiB GPU 权重的 CPU 副本 = 237.110 GiB**，两 rank 合计，较当时 479 GiB
空闲量剩约 241.890 GiB。运行库、通信、文件页缓存及写缓冲另计；这不是实测 RSS 峰值。
本地 tiny fixture 已检查 StateAdapter 到 CPU writer 的 storage 指针一致、SGD 梯度
保持 sparse 且优化器 state 为空；它验证实现路径，不替代远端大表峰值采样。

恢复则会先 `torch.load` 一个保存的本地分片，再 `copy_` 到现有参数；它不是直接
读入原有表 storage。相同两 rank 布局下，Engram 反序列化会额外需要约
**91.554 GiB/rank，共 183.108 GiB**；与原有表合计约 **366.217 GiB**，
还需当前读取的普通权重及其他运行开销。[PyTorch 2.11 reader 源码](https://github.com/pytorch/pytorch/blob/v2.11.0/torch/distributed/checkpoint/filesystem.py#L829-L846)。
该静态预算仍低于当时 479 GiB，但恢复阶段余量更小；**本轮 prefix2 仅训练并保存
一次，不执行 resume**，也不留存第二份 prefix2 checkpoint。

## 真实 FP8/FP4 格式检查

现有 reader 和解码器支持以下格式，并且直接分片与 CPU Engram 共用这些实现：

| 权重 | safetensors dtype / 布局 | scale |
| --- | --- | --- |
| attention/shared 等 FP8 | `F8_E4M3`，32×32 块 | `F8_E8M0`，每个块一个 |
| routed expert FP4 | `I8` 打包两个 E2M1 值，低 nibble 在前 | 每 32 个展开元素一个 E8M0 |
| Engram FP8 表 | `F8_E4M3`，逐行 | 每行每 32 列一个 E8M0 |

`direct_shard` 在 CPU 解码后填入目标 BF16 分片；layerwise 在目标 GPU 解码普通权重，
CPU Engram 仍走 CPU 分块解码。现有若干单测使用 F32 scale，不能替代真实挂载格式检查。

新增脚本只读取三组权重各至多 32 行，限制每组源读取在 8 MiB 内，输出原始 dtype、
全局/样本 shape、解码后有限值检查及逻辑读取字节数，不分配模型、不写源权重：

```bash
.venv-posttrain/bin/python scripts/inspect_dsv4_quantization.py \
  --model-path /mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash \
  > artifacts/l20d/dsv4-quantization.json
```

本地以真正 `F8_E8M0` scale 的 tiny safetensors 验证三类解码通过，总读取 2,625 字节；
这验证脚本及本地 PyTorch 2.14.1 路径。随后主任务报告远端 PyTorch 2.11.0 运行实际
挂载权重抽样 exit 0、`passed=true`，三类解码均为有限值，总逻辑读取 259,488 字节。
该结果仅表示格式和小样本解码通过，不是 HF 数值参考对齐。

DeepSeek nested config 的外层 special tokens 同时已修正：BOS=0、EOS=1、PAD=2
会进入训练 config；`text_config` 自身明确指定的值仍优先。此前丢弃外层后错误使用
dataclass 默认值，会令 SFT 添加错误的 BOS/EOS。此修复需同步到远端后再运行。

## 两卡最小启动方案

先结束占用相同设备的本任务前序进程、复查实际空闲内存和磁盘字节数，并确认新脚本、
special-token 修复和 trainer 修复已经同步。以下命令从远端仓库根目录执行，输出目录
必须为新目录。每个方案仅在最终保存一次 checkpoint，均不写挂载源。

```bash
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$PWD:$PWD/tests${PYTHONPATH:+:$PYTHONPATH}"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv-posttrain/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  train_sft.py --model-type dsv4 \
  --model-path /mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash \
  --data examples/sft_smoke.jsonl --output-dir runs/dsv4-prefix1-sft-01 \
  --start-layer 0 --num-layers 1 --ep-size 2 --cp-size 1 \
  --seq-len 128 --batch-size 1 --gradient-accumulation-steps 1 \
  --optimizer sgd --learning-rate 0.01 --weight-decay 0 \
  --checkpoint-mode full --load-mode direct_shard --load-pipeline \
  --staging-mib 128 --steps 2 --save-every 0

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv-posttrain/bin/python -m torch.distributed.run --standalone --nproc-per-node=2 \
  train_sft.py --model-type dsv4 \
  --model-path /mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash \
  --data examples/sft_smoke.jsonl --output-dir runs/dsv4-prefix2-cpu-engram-01 \
  --start-layer 0 --num-layers 2 --ep-size 2 --cp-size 1 \
  --seq-len 128 --batch-size 1 --gradient-accumulation-steps 1 \
  --optimizer sgd --learning-rate 0.01 --weight-decay 0 \
  --checkpoint-mode full --load-mode direct_shard --load-pipeline \
  --engram-device cpu --engram-prefetch async --engram-deduplicate \
  --engram-cache-mib 0 --staging-mib 128 --steps 2 --save-every 0
```

先完成 1 层方案及量化抽样检查，再执行 2 层方案。SGD 不设置 momentum；CPU Engram
要求稀疏梯度，不能改用当前 AdamW 路径。初轮关闭 row cache，以免把加载/基本训练和
cache 效果混在一起。短 smoke 数据用于执行验证，不用于评价训练质量。

2 层方案保存的是整份选定模型，稀疏梯度不会让 checkpoint 仅保存命中行。
两份完整 Qwen SGD checkpoint、单份 prefix1、单份 prefix2 的原始参数数据合计约
**406.883 GB / 378.94 GiB**，另需元数据、日志、现存其他文件和安全余量。
这与最初 `df -h` 显示的 `509G` 有余量，但应以保存前实时空闲字节为准。

本轮**不做 prefix2 的双 checkpoint 留存**。若直接恢复后再保存一个新 step，
将额外增加约 254.595 GB，使上述组合升至约 661.478 GB，超出当前初始磁盘预算。
后续如需该大前缀的保存/恢复完整验收，先规划额外存储或明确的旧 checkpoint 留存策略。
