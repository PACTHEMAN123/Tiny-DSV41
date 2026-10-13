# Engram 功能与使用指南

本分支提供可训练 Engram 的行分片、CPU offload、异步预取、请求去重和有界 GPU row cache。
DSV4 使用模型已有的 Engram 结构；Qwen3 MoE 通过适配层新增随机初始化的 Engram，并提供真实 SFT 四组对照入口。
这里介绍功能和使用方式，完整实验配置与数值见 [Qwen 测试报告](qwen30b_engram_sft_results.md) 和 [DSV4 大规模测试报告](dsv41_large_scale_results.md)。

## 功能范围

| 功能 | 实现与边界 |
| --- | --- |
| 行分片 | 每个 owner 只保存所属表行，通过查询 ID 和返回行路由完成分布式查表，支持非均匀行数。 |
| CPU offload | 表权重及稀疏梯度保留在 CPU，投影与门控留在 GPU；仅传输请求涉及的行。全表可训练，并非冻结 embedding。 |
| 异步预取 | 提前交换 ID，后台查表并 H2D；消费层等待 ticket。使用有界 pinned staging 和独立 copy stream，collectives 由主线程按固定顺序发起。 |
| 请求去重 | 合并重复请求，通过 inverse mapping 恢复顺序和重复行梯度；与 cache 独立，可用于 CPU/GPU 表。 |
| GPU row cache | 固定 slot、按表按 rank 限额，支持 CPU/GPU 源表；SGD 后失效发生更新的缓存行。 |
| 直接分片加载 | DSV4 的 `direct_shard` 先建立 meta/FSDP 布局，再按量化块对齐读取并填充最终局部分片；不先物化完整层。 |
| 选择性 FSDP offload | DSV4 按全局层号与 `dense/experts` 参数域的交集卸载，独立于 Engram 表卸载。 |
| 分片恢复 | DSV4 的版本化 DCP 保存参数、优化器和 RNG，使用真实表行偏移；同拓扑可切换 CPU/GPU 放置，cache 不进入 checkpoint。 |
| Qwen 接入 | `QwenEngramConfig` 和 `attach_engram()` 在 FSDP 包装前添加表和投影；复用 DSV4 的 hash、路由、存储与更新路径。 |
| Profiling | 分别记录加载、显存、CPU RSS、逻辑读取量、行传输量、cache 命中、等待及吞吐，并输出 CPU/CUDA trace 和 HTML timeline。 |

## 卸载和 Cache 的生命周期

1. 加载或初始化时，直接把本地表分片放到目标设备。CPU 模式不先保留一份完整 GPU 表，也不在每层 forward 后搬运整表。
2. Forward 根据 token 计算 n-gram ID，并将请求路由到行 owner。CPU 同步模式在消费处查表；异步模式提前启动查表和行 H2D。启用 cache 时，命中读取 GPU slot，缺失才从源表取值。
3. 到对应层时 `resolve()` 等待数据、交换返回行，再执行 GPU 投影和门控。查表位于 activation checkpoint 重算区外，重算使用本次 forward 的行张量。
4. Backward 将行梯度路由回 owner；CPU 表还需要 D2H。梯度累积结束后统一归一化，对重复稀疏行合并，再以 SGD 更新源表。
5. 更新过的缓存行失效，未更新的行可继续复用。未知参数版本变化、重新加载或关闭表会清空 cache；直接通过 `.data` 写入后需显式调用 `table.clear_cache()`。

这与推理 KVCache 的只读复用不同：训练权重会变化，不能跨 optimizer step 使用旧行值。Cache 也不会省去 owner 路由通信或 CPU 表的梯度 D2H。
GPU 源表开启 cache 时仍保留完整本地表分片，因此会增加显存占用。
预取是否及时取决于目标层之前的计算时间；早层可能仍需等待，不能把“启用 async”等同于“传输完全隐藏”。

## DSV4 使用

入口为 `train_dsv4_checkpoint.py`。默认 `layerwise` 加载、GPU 表、同步查表，不启用去重、cache、加载流水线或 FSDP CPU offload。
运行策略不写入模型结构配置，CLI 与兼容约束详见 [配置参考](memory_offload.md)。

```bash
python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_dsv4_checkpoint.py --model-path /path/to/DeepSeek-V4.1-Flash \
  --num-layers 5 --cp-size 2 --ep-size 4 --seq-len 32 \
  --optimizer sgd --steps 3 --gradient-accumulation-steps 2 \
  --engram-device cpu --engram-prefetch async --engram-deduplicate \
  --engram-cache-mib 128 --staging-mib 64 \
  --metrics-file artifacts/engram-offload.json
```

`--engram-cache-mib 128` 是每张本地表、每个 rank 的 GPU cache 预算，两张表合计最多 256 MiB/rank 的行缓存，另有查询输出和元数据开销。
设为 `0` 关闭 cache；使用 `--engram-prefetch sync` 单独对比预取；GPU 对照使用 `--engram-device cuda --engram-prefetch sync`。
可另加 `--load-mode direct_shard --load-pipeline` 测加载，或 `--fsdp-offload-layers 1,3 --fsdp-offload-targets experts` 测指定层专家卸载。对照时保持其余设置一致。

## Qwen SFT 四组对照

Qwen Engram 实验入口是 `tests/qwen_engram_sft_benchmark.py`，矩阵入口是 `tests/run_qwen_engram_benchmark.py`。
普通 `train_qwen.py` 不会自动添加 Engram，也不接收上述 DSV4 offload 选项。
当前矩阵固定 CP=1、EP=进程数，采用全参数 BF16、SGD、activation checkpoint；这一路径没有接入 DSV4 的直接分片加载、选择性 FSDP CPU offload 或训练状态保存/恢复。

默认在零起始层号 1、24 添加 Engram：2/3/4-gram，各 8 个 hash head，每行 256 维，每 head 基础桶数 900,000。
Qwen 使用 `hc_mult=1` 的单残差流。新增表约 11.063B 参数，连同投影后模型总计约 41.646B；新增 Engram 没有预训练权重。
Qwen checkpoint 按层加载，只读取当前 EP rank 的专家，支持将 FP8 block-scaled 权重解码到 BF16；这不是 FP8 训练。

需要 Python 3.10+、PyTorch 2.10+、CUDA/NCCL、`safetensors`、`tokenizers`、`pyarrow`；生成 HTML timeline 另需 `plotly`。
所有命令从仓库根目录运行。`--model-path` 指向自己的完整 checkpoint 副本；若源路径是软链接，应复制实际文件，避免用软链接或硬链接代替副本。
准备含 `instruction/input/output` 列的本地 Alpaca Parquet 后，先编码回答部分监督的数据：

```bash
PYTHONPATH=.:tests python3 tests/prepare_qwen_sft.py \
  --data /path/to/alpaca.parquet \
  --tokenizer /path/to/private/Qwen3-30B-A3B-Instruct-2507/tokenizer.json \
  --output artifacts/qwen-engram/data/alpaca-2048.pt \
  --seq-len 2048 --max-packs 4096 --seed 42
```

Prompt 标签屏蔽为 `-100`，只监督回答和结束标记；连续 packing 允许跨样本看到前文。随后运行四组对照：

```bash
PYTHONPATH=.:tests OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 tests/run_qwen_engram_benchmark.py \
  --model-path /path/to/private/Qwen3-30B-A3B-Instruct-2507 \
  --data artifacts/qwen-engram/data/alpaca-2048.pt \
  --output artifacts/qwen-engram/matrix \
  --nproc 8 --batch-size 4 --microbatches 2 \
  --warmup 2 --steps 3 --repeats 2 --profile
```

| `--variants` 名称 | 表放置 | 查表调度 | Cache |
| --- | --- | --- | --- |
| `gpu` | GPU | 同步 | 关 |
| `gpu_cache` | GPU | 同步 | 128 MiB/表/rank |
| `cpu` | CPU | 异步 | 关 |
| `cpu_cache` | CPU | 异步 | 128 MiB/表/rank |

矩阵默认运行全部四组，全部开启请求去重，CPU staging 为 64 MiB/表。每组使用独立进程，第二次重复反转执行顺序。
自定义 Engram 层或规模、cache 预算、同步 CPU 对照时，直接用 `torch.distributed.run` 启动单组入口，传入 `--variant`、`--layers`、`--vocab-per-head`、`--head-dim`、`--cache-mib`、`--prefetch`；矩阵脚本不转发这些自定义参数。
每次所需不同 packed sequence 数至少为 `batch_size * microbatches * world_size * (warmup + steps + profile_step)`，其中采集 trace 时 `profile_step=1`，否则为 0。

## 结果与 Timeline

```bash
PYTHONPATH=.:tests python3 tests/summarize_qwen_engram.py \
  artifacts/qwen-engram/matrix \
  --output artifacts/qwen-engram/results.json --timelines
```

每组输出 `summary.json` 和各 rank 指标。首轮重复的 `trace/` 下提供 rank 0 原始 Chrome trace、分项统计和生成的 `timeline.html`、`detail.html`、`breakdown.json`。
Profiler 采集发生在正式计时之后的额外一步，吞吐不包括这一步，也不包括模型加载、tokenizer 或在线 DataLoader。
各 GPU activity 的时长可能重叠，不能相加当作 step wall time；`resolve` 的 host 时间包含等待与返回行路由调用。

现有 8 卡 H20 的 Qwen+Engram Alpaca 实测：GPU 无 cache 为 10,698 tokens/s、35.10 GiB/卡；CPU 无 cache 为 9,261 tokens/s、32.35 GiB/卡。
即每卡省约 2.75 GiB，吞吐下降约 13.4%。CPU cache 命中 8.76%，降低传输量，但吞吐基本没有改善。
这是每组两次运行、每次三步正式计时的短程结果，存在运行波动与 BF16 非确定性，不代表收敛、训练质量或所有负载下的收益。
完整四组数值、模型结构、共享文件校验和测试方法见 [实测报告](qwen30b_engram_sft_results.md)。

Git 中保留源代码、配置、Markdown 报告和 `docs/results/` 的紧凑统计；权重、数据集、完整 trace 与 HTML 位于忽略的 `artifacts/`，不会随 clone 下载，需本地复现生成。

## 兼容与验证

- CPU 表及缓存表走无 momentum、无 weight decay 的 sparse SGD，不提供持久 FP32 master weights。GPU 无 cache 的稠密梯度路径保留 AdamW。
- DSV4 的 CP/EP 可以分别整除 world size，EP 还需整除专家数，序列满足 CP 约束；Qwen 本指南的矩阵只测 CP=1，不把其他入口的拓扑能力视为已验收。
- DSV4 拒绝 LoRA 与直接分片加载、CPU Engram 或选择性 FSDP offload 的组合；Qwen Engram SFT runner 没有 LoRA 开关。
- DSV4 恢复要求相同 world size、CP/EP 和兼容训练配置。缓存不属于训练状态，改变预取或放置时重新构建。
- 配置 CPU 表后禁止整模型 `.to(cuda)`；使用目标放置加载。保存、更新或异常退出前需完成 ticket/传输，使用现有 optimizer hooks 并关闭表 worker。

最近一次回归包含 CPU 104 项通过、10 项按环境跳过，CUDA 13 项通过，以及四种策略的八卡 FP32 小模型多步参考对照。
测试入口包括 `tests/test_qwen_engram.py`、`tests/test_engram_cache.py` 和 `tests/qwen_engram_distributed.py`；其他命令见 [测试指南](testing.md)。
