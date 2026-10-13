# DSV4.1 95B 大规模容量与性能测试

测量日期：2026-10-13。本文使用当前工作区代码和真实 H20 GPU，延续此前的 Engram 卸载对照。已完成八次正式吞吐试验、两次 layerwise 加载基线、两组六 rank CPU/CUDA profiling，以及一次独立的原负载预取就绪检查；另有三档容量探测。

主要结果：模型扩大到 95.05B、40 层，实际驱动占用接近 95 GiB/卡。CPU Engram 在接近容量上限时节省约 7.82 GiB/卡，并提升吞吐约 13%，但异步和缓存没有稳定额外加速。大表的第 1 层预取仍有等待，当前 direct_shard 加载又比 layerwise 慢约 10 倍，这两点不能被容量收益掩盖。

## 模型结构

新配置为 [profile_dsv41_95b.json](../configs/profile_dsv41_95b.json)，实际全局参数量 **95,053,103,472，约 95.05B**，是此前 423.2M 模型的约 224.6 倍。它保留 40 层及原有跨层依赖，按宽度、专家数和表行数缩小仓库中的 DSV4.1 文本模型。

| 结构项 | 此前 423M 测试模型 | 本次 95B 模型 | 仓库完整文本配置 |
| --- | ---: | ---: | ---: |
| Decoder 层数 | 5 | 40 | 40 |
| hidden size | 512 | 2560 | 5120 |
| token 词表 | 2048 | 129280 | 129280 |
| query 头数 / head dim | 8 / 64 | 32 / 512 | 64 / 512 |
| query 低秩维度 | 128 | 640 | 1280 |
| 输出分组 / 低秩维度 | 4 / 128 | 8 / 512 | 8 / 1024 |
| 每层路由专家数 / 每 token top-k | 16 / 2 | 192 / 6 | 384 / 6 |
| 共享专家数 / 专家中间维度 | 1 / 1024 | 1 / 1152 | 1 / 2304 |
| mHC 残差流 / Sinkhorn 迭代 | 4 / 20 | 4 / 20 | 4 / 20 |
| 索引头数 / head dim / top-k | 4 / 64 / 16 | 16 / 128 / 512 | 32 / 128 / 512 |
| Engram 层，零起始 | 1、3 | 1、14 | 1、14 |
| Engram 每阶 hash 头 / 行宽 | 4 / 256 | 8 / 256 | 8 / 256 |
| 每张表、每 token 查询行数 | 12 | 24 | 24 |
| Engram 两张表的行数 | 524291 / 524299 | 48004402 / 48011386 | 384006168 / 384016682 |
| Engram 表参数占比 | 63.43% | 25.86% | 26.27% |

本次 compression 配置为 `[0, 0] + [2] * 18 + [1] * 20`；KV 来源为第 2、8、14、20 层，index 来源为第 2、8、14、20、24、28、32、36 层，candidate 来源为第 20 层。滑窗长度 128、n-gram 阶数 2/3/4、RoPE/YARN 设置与原有完整配置一致。结构上的低秩维度不是 LoRA 微调开关。

| 参数类别 | 全局参数量 | BF16 全局权重 |
| --- | ---: | ---: |
| Engram 表 | 24,580,041,728 | 45.78 GiB |
| 路由专家 | 67,947,724,800 | 126.56 GiB |
| 其他参数 | 2,525,336,944 | 4.70 GiB |
| 合计 | 95,053,103,472 | 177.05 GiB |

“其他”包含 attention、共享专家、路由器、mHC、Engram 投影/门控、embedding 和 LM head。参数量按未分片 meta 模型实际统计，包含所有专家和完整表，不是每 token 激活量。完整配置列指仓库文本 backbone，不代表已加载完整发布模型。

权重体积列为纯 BF16 等价值。实际实现保留少量 FP32 mHC/sink 参数，两种加载方式的六 rank 实际权重字节数相同，均为 190,145,535,424 字节，约 177.09 GiB。

## 数据、机器与方法

- 机器有八张 H20，每张驱动报告 97,871 MiB，即约 95.58 GiB。测试开始及容量探测时，GPU 0 被其他任务动态占用，因此本轮大模型固定使用物理 GPU 1 至 6，`world=6, CP=2, EP=3`；其他任务未被停止。采用这一拓扑是为了在空闲设备上运行合法的 CP/EP 对照，并非八卡大模型测试。
- Python 3.10.20、PyTorch 2.10.0+cu129，单机 NVLink，`OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`。主机是共享环境，CPU、内存和存储仍可能受其他任务影响。
- 所有策略均使用同一份量化合成 checkpoint：FP8 普通矩阵/Engram，FP4 路由专家，加载后 BF16 训练。权重由独立随机量化编码生成，不同专家不再使用相同常量模式；不是预训练模型，也不代表真实语料的路由或缓存命中率。
- [create_scaled_checkpoint.py](../tests/create_scaled_checkpoint.py) 在 meta 上枚举源权重，以 16 MiB 数据块直接生成 safetensors，不构造完整 CPU 模型。50 个 shard 的编码张量共 74,786,230,080 字节，约 69.65 GiB，生成用时 259.20 秒。单张超大表可跨写入块保存在一个 shard 中；1 GiB 是常规打包目标，不是超大张量的文件大小上限。
- 统一使用 `direct_shard + pipeline`、128 MiB staging、full activation checkpoint、BF16 SGD，学习率 0.0001，无 momentum/weight decay。Engram 保持全表可训练、稀疏梯度；关闭路由去重、FSDP CPU offload 和 dense foreach。缓存策略每张表 256 MiB，每卡两张表共 512 MiB。
- 每个进程设置 86 GiB 的 PyTorch allocator 上限，使用 `expandable_segments:True`。该上限不包含 CUDA/NCCL 等外部分配，不等于整卡使用量上限。所有对照使用相同上限。
- `fresh` 输入按 CP 组共享，每次 microbatch 更换随机 token。正式负载 `batch=8, sequence=2048, accumulation=2`，全局唯一 token 数为每次 optimizer update 98,304。每策略两次独立六进程运行，每次一次预热、三次计时更新，策略顺序确定性打乱。
- 吞吐使用最慢 rank 的同步完成耗时；加载、profiling 和就绪探针不计入训练吞吐。每卡峰值取全部 rank 的最大 PyTorch allocated，另记录 reserved、CPU RSS、传输字节及 allocator 回收重试。两次重复仅用于本轮短测试，不构成置信区间或长期稳定性结论。

原始数据位于 `artifacts/dsv41-scale-20261013/`，fixture 元数据为 `fixture95b/fixture.json`。每个 trial 保存配置、策略、六个 rank 的计时、loss、内存与流量；`comparison/manifest.json` 保存完整运行参数和顺序。

## 已完成的容量探测

以下均为同一 95B 模型、GPU 表常驻、一次预热加一次计时更新，不能当作重复性能对照：

| sequence / batch / accumulation | allocated 峰值/卡 | 单次更新耗时 | 全局 tokens/s |
| --- | ---: | ---: | ---: |
| 1024 / 1 / 1 | 54.07 GiB | 14.61 s | 209.94 |
| 2048 / 4 / 1 | 62.46 GiB | 20.57 s | 1193.12 |
| 2048 / 8 / 2 | 81.63 GiB | 80.76 s | 1216.99 |

最高点驱动层单卡实际使用曾达到 **95,464 MiB，约 93.23 GiB**，接近整卡 95.58 GiB。95B 的每卡常驻分片权重约 29.57 GiB，训练还需要梯度、激活、FSDP 聚合参数和通信缓冲；不能仅按权重占用判断模型是否足够大。这里证明的是一个接近设备容量的可训练负载，并未搜索所有模型形状的绝对最大参数量。

## 接近容量上限的正式对照

各策略两次独立进程运行，每次三次计时更新。以下速度为两次运行的中位数及范围；增益先在相同 repeat 内配对，再取中位数。

| 策略 | tokens/s 中位数 [范围] | 相对 GPU 常驻吞吐变化 | allocated 峰值/卡 | 单进程最大 RSS 中位数 |
| --- | ---: | ---: | ---: | ---: |
| GPU 常驻 | 1313.97 [1303.73, 1324.20] | 基线 | 81.63 GiB | 13.72 GiB |
| 同步 CPU | 1489.02 [1471.90, 1506.13] | +13.32% | 73.81 GiB | 21.35 GiB |
| 异步 CPU | 1483.26 [1468.02, 1498.49] | +12.90% | 73.81 GiB | 21.37 GiB |
| 异步 CPU + 缓存 | 1478.95 [1452.79, 1505.10] | +12.58% | 74.31 GiB | 21.30 GiB |

无缓存卸载净节省 **7.82 GiB/卡，约 9.58% 的训练峰值**；缓存额外占用 0.50 GiB，净节省约 7.32 GiB/卡。所有记录到的六 rank、三次计时 loss 都与对应 repeat 的 GPU 常驻基线完全一致，全部为有限值。

Engram 占参数的 25.86%，并不意味着训练总显存也节省 25.86%：本地表权重约 7.63 GiB，而其余权重、梯度、激活和临时缓冲仍留在 GPU。CPU 表模式实际参数放置统计也确认约 7.63 GiB 表权重转到 CPU，GPU 上没有保留完整表副本。这里的 CPU RSS 含 mmap 等映射，不能把各 rank 的 RSS 相加当作主机独占物理内存。

监控覆盖的容量探测和正式运行期间，参与 GPU 的驱动占用最高达到 **97,284 MiB，约 95.00 GiB**，该采样点仅余约 74 MiB 可用。它是整个监控窗口的瞬时峰值，不是仅计时阶段的统计值。四种策略的 PyTorch reserved 峰值都约 86 GiB；卸载释放出的空间可被 allocator 缓存复用，因此 allocated 节省不等于驱动占用立即同幅下降。

### 加速来源与预取、缓存的区别

| 策略 | 每 rank、三次计时更新内的 allocation retry 次数 | 观测 |
| --- | --- | --- |
| GPU 常驻 | 85、104、94、94、86、101，两轮一致 | 频繁回收缓存、同步并重试 |
| 同步 CPU | 2、3、3、4、5、2，两轮一致 | 内存压力显著下降 |
| 异步 CPU | 5、3、4、5、0、4，两轮一致 | 内存压力显著下降 |
| 异步 CPU + 缓存 | 第二轮为 2、5、7、7、2、9 | 首轮未记录此计数 |

有计数的试验 `num_ooms` 均为 0。计数器在首轮缓存试验启动后加入，缺失项没有按 0 处理；统计在计时区间外读取，不改变优化器或输入。

这为“释放表权重缓解显存压力，减少 allocator 回收与同步”提供了直接证据，但尚不能精确分配它对总加速的贡献。当前结论依赖这个接近容量上限、allocator 预算为 86 GiB 的负载，不代表所有显存宽裕的训练也会加速。

满负载正式对照统一使用 direct_shard；layerwise 的满负载 GPU 常驻路径没有做同等重复吞吐测量，因此不能将这个约 13% 的增益直接套用到另一种加载/allocator 初始状态。

异步相对同步的两次吞吐变化为 **-2.53%、+1.81%**，配对中位数约 -0.36%，没有稳定异步加速。缓存相对无缓存异步为 **-1.04%、+0.44%**，配对中位数约 -0.30%，同样没有稳定收益。

| 传输/缓存指标，六 rank 合计每次更新 | 无缓存 CPU | CPU + 缓存 |
| --- | ---: | ---: |
| 行值 H2D | 2304.00 MiB | 2252.59 MiB |
| 缓存索引元数据 H2D | 0 | 131.50 MiB |
| 上述 H2D 合计 | 2304.00 MiB | 2384.10 MiB |
| 行梯度 D2H | 2304.00 MiB | 2304.00 MiB |
| 缓存命中率 | 不适用 | 1.02% |

随机新输入下，缓存总 H2D 反而增加约 3.48%，又消耗 512 MiB/卡。当前结果支持保持缓存默认关闭，不能把整个卸载方案的吞吐提升算成缓存收益。以上流量不包含全部请求 ID 和 NCCL/FSDP 流量。

完整样本及 rank 计数：[正式对照 JSON](results/dsv41-large-scale-h20.json)、[缓存统计 JSON](results/dsv41-large-scale-cache-h20.json)。

## 加载与 Profiling

正式对照均使用直接分片加载。GPU 常驻的加载耗时中位数为 107.21 秒，同步 CPU 为 106.36 秒，异步 CPU 为 109.64 秒，缓存为 110.19 秒；每次加载先取最慢 rank，再在两次运行间汇总。六 rank 逻辑读取量均约 69.72 GiB，接近 checkpoint 的 69.65 GiB 编码张量体积。

### 加载方式的实际取舍

补充两次独立进程的 GPU 常驻 layerwise 加载，模型、权重、拓扑、dtype、staging 和 allocator 预算相同。加载结束后使用 batch 4、accumulation 1 完成预热和一次更新；这些训练设置在加载之后生效，训练吞吐不与前述 batch 8 的主对照混用。

| 加载路径 | 最慢 rank 耗时中位数 [范围] | 加载 allocated 峰值/卡 | 六 rank 逻辑读取量 |
| --- | ---: | ---: | ---: |
| layerwise | 10.74 s [10.40, 11.07] | 30.69 GiB | 130.93 GiB |
| direct_shard + pipeline | 107.21 s [106.57, 107.86] | 29.57 GiB | 69.72 GiB |

当前直接分片读取量减少 **46.75%**，加载峰值减少 **1.12 GiB/卡**，但耗时约为 layerwise 的 **9.99 倍**。这是本轮暴露的主要加载性能问题，不能称为“加载加速”。Layerwise 在 GPU 解码量化权重，当前 direct_shard 在 CPU 解码后上传目标 dtype；减少重复读取未抵消 CPU 解码与逐块处理成本。

两种路径加载后的实际参数字节数一致；layerwise 的六 rank 首次计时 loss 与同负载 direct_shard 容量探测完全一致。这是大模型负载上的检查，完整逐参数加载一致性仍由小模型测试覆盖。

在本次硬件与缓存条件下，若能容纳额外约 1.12 GiB 加载峰值，layerwise 的启动延迟明显更低。后续加载优化应优先测量 GPU 目标分片上的量化解码与跨参数流水线，同时保留 CPU 目标表的 CPU 解码路径；本轮未改动生产解码算法。原始结果：[layerwise JSON](results/dsv41-large-scale-layerwise-h20.json)。

独立的 batch 8 预取检查同时验证了 CPU Engram + layerwise：本次加载最慢 rank 为 **48.90 秒**，训练 allocated 峰值 73.81 GiB，首次计时 loss 与 direct_shard 同负载一致。这是一次补充运行，不混入两轮正式吞吐统计，也不把 GPU 常驻的 10.74 秒套用到 CPU 表。

### 独立 Profiling

使用 `torch.profiler` 的 CPU/CUDA activity、跨线程采集和模块 `record_function` 标注；根据 autograd 的前向 sequence/thread 信息归属反向算子。每个策略独立启动六个进程，先采集加载，再预热一次、进行三次未插桩训练计时，最后单独采集一次训练 update。关闭 `record_shapes`，避免额外保留张量；启用 memory event 记录。

模型仍为 95B，诊断负载为 **batch 4、sequence 2048、accumulation 1**，其 allocated 峰值为 GPU 常驻 62.48 GiB、CPU 表 54.85 GiB，保留了采样余量。三次训练计时的吞吐分别为 1074.74 与 1052.32 tokens/s，所有 rank 的 allocation retry 均为 0；每个策略只有一次运行、步耗时有明显波动，不能用它建立稳定的速度差异。加载采样和 GPU trace 耗时不属于前述满负载吞吐统计。

加载 trace 的分项如下。CPU 解码是六 rank 的平均累计 host 区间，含工作线程；H2D 为实际 GPU memcpy 活动，两者不能相加为总加载耗时。

| direct_shard 加载分项 | GPU 表常驻 | CPU 表 |
| --- | ---: | ---: |
| 专家 CPU 解码 | 68.54 s | 68.55 s |
| Engram CPU 解码 | 15.55 s | 16.87 s |
| 其他 FP8 CPU 解码 | 0.80 s | 0.79 s |
| H2D 活动/卡 | 1.47 s | 1.30 s |
| H2D 字节，六 rank 合计 | 177.09 GiB | 131.31 GiB |

因此，当前直接加载最明显的瓶颈是 CPU 量化解码和数据整理，特别是 FP4 专家的索引、拼接与拷贝；不是 H2D 的总持续时间。这里 CPU 算子使用单线程设置，不能将结果理解为已调优的多线程 CPU 解码上限。

训练采样的 GPU 活动如下，单位为 **ms/采样 update，六 rank 平均**。跨 stream 区间可能重叠，NCCL 包含等待，不是可相加为 100% 的关键路径分解。

| GPU 活动 | GPU 表常驻 | CPU 异步表 |
| --- | ---: | ---: |
| Compute kernels | 11910.88 | 11951.87 |
| NCCL kernels | 6377.75 | 6569.66 |
| H2D | 3.00 | 4.82 |
| D2H | 14.37 | 70.61 |
| D2D | 30.24 | 30.36 |

同一采样的主要训练调用区间如下，单位为 ms/update、六 rank 平均。它们是包含等待的 CPU host 区间，不能与上面的 GPU 活动相加。

| 阶段 | GPU 表常驻 | CPU 异步表 |
| --- | ---: | ---: |
| Forward | 4078.80 | 4365.22 |
| Backward | 17224.27 | 17268.16 |
| Gradient normalization | 74.56 | 66.38 |
| Optimizer | 99.21 | 530.14 |

主要观察：

1. 压缩 attention 反向索引是显著瓶颈。`indexing_backward_kernel<BFloat16>` 在 CP 前半序列 rank 0/2/4 上累计约 7.51/7.51/7.95 秒，在 rank 1/3/5 上仅约 0.215 秒；后者对应大量 SendRecv 等待。模块归属将主要耗时指向 attention 反向。代码使用 `topk_indices.clamp_min(0)` 处理无效索引，padding 集中到同一行可能造成反向 scatter 热点，这是与观测吻合、仍需专项对照验证的机制，不能仅凭 trace 宣称已证明。
2. 每卡每个采样 update 约有 30.9 万个 compute kernel、1370 个 NCCL kernel，以及 5773 次 `cudaStreamSynchronize`。原生 MoE 逐专家筛选和执行带来大量小算子及 host 同步；GPU utilization 接近 100% 不代表计算效率接近硬件峰值。
3. CPU Engram trace 中，两张表的 store host 区间累计平均约 260.31 ms，resolve 累计约 47.22 ms。两个工作线程可以重叠，不能把 store 累计值当作额外训练延迟。同步梯度 D2H 比前向 H2D 更突出，但仍为几十毫秒量级。采样中的 H2D 与 compute 重叠比例平均约 0.25%，不能据此宣称全部传输已隐藏。Optimizer 阶段由约 99 ms 增至 530 ms，说明 CPU 稀疏梯度处理和 SGD 更新同样是卸载成本，不能只优化前向预取。
4. 相比 Engram 的行传输，attention 反向索引、CP 等待和逐专家执行应获得更高的训练性能优化优先级。本轮没有改写这些内核，以保持性能对照的模型计算路径一致。

### 原始负载的预取截止时间

就绪探针使用 host 单调时钟。CPU store 返回前已等待 staged H2D 完成，因此“在 resolve 前就绪”为保守的提前完成证据；反之，Future 的 host 等待可能与 GPU 队列中已有工作重叠，不能将全部等待毫秒数直接计为新增单步延迟。探针不包含行返回 NCCL 的完成期限。

| 探针负载 / 加载路径 | 层 | resolve 前已就绪 | Future 等待 p50 / p95 / max | 提前量 |
| --- | ---: | ---: | ---: | --- |
| batch 4，accumulation 1，direct_shard | 1 | 0 / 12 | 75.20 / 81.53 / 82.99 ms | 未提前完成 |
| 同上 | 14 | 12 / 12 | 0.005 / 0.009 / 0.013 ms | p50 约 1.02 s |
| batch 8，accumulation 2，layerwise，无 profiler | 1 | 0 / 24 | 144.61 / 264.93 / 871.86 ms | 未提前完成 |
| 同上 | 14 | 24 / 24 | 0.005 / 0.009 / 0.009 ms | 最少 1.90 s，p50 2.55 s |

每个负载额外运行两个诊断 update；第二组为独立进程，使用主吞吐对照的原始 batch/sequence/accumulation，没有加载或训练 profiler。它的两次诊断 update 约为 69.65、70.83 秒，每步最慢 rank 的 Future 等待累计约 0.48、1.06 秒。它检验真实截止时间，不是新增的重复吞吐结论。

**结论是第 14 层的预取有效，而第 1 层无法保证在主线程消费前完成。** 之前小模型的全部就绪结果不能外推到大表。后续需要针对早层 CPU 查表、NUMA/线程配置和提交时机优化，并继续保持 optimizer 边界的一致性；当前不能宣称异步预取已经完全隐藏了卸载成本。

### 数据与 Timeline

- [GPU 常驻 timeline](../artifacts/dsv41-scale-20261013/timeline/baseline/timeline.html)、[CPU 异步 timeline](../artifacts/dsv41-scale-20261013/timeline/cpu_async/timeline.html)：独立离线 HTML，无需启动服务。
- 各 timeline 目录包含加载/训练 rank 0 的精确事件图、`time-breakdown.csv` 和 `time-breakdown.json`，包含所有 rank 的模块及阶段统计。
- 原始 Chrome/Perfetto trace 位于 `artifacts/dsv41-scale-20261013/profiles/{loading,traces}/{baseline,cpu_async}/`。训练 trace 每个 rank 约 786 MiB，建议先查看 overview，再按需打开原始文件。
- [Profiling 汇总 JSON](results/dsv41-large-scale-profile-h20.json)、[两档预取诊断 JSON](results/dsv41-large-scale-prefetch-h20.json)、[满负载 host 预取 timeline](../artifacts/dsv41-scale-20261013/readiness-near-capacity/prefetch-host-timeline.json)。
- [GPU 常驻分项统计](results/dsv41-large-scale-baseline-timeline.json)、[CPU 异步分项统计](results/dsv41-large-scale-cpu-async-timeline.json)保留完整加载/训练阶段、模块归属及各 rank 数据，便于脱离大 trace 复核。

模型配置、fixture 生成器、完整命令参数和随机种子均已保存。图表中的 GPU 编号为进程可见编号，物理映射为 `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6`。大 trace 和合成 checkpoint 保留在本地 artifacts，汇总 JSON 位于 docs/results。

## 结果边界

1. 本轮扩大了模型和真实训练内存压力，但仍使用合成随机权重与 token，没有验证预训练模型质量或自然语料的缓存复用。
2. 使用 BF16 SGD，没有 AdamW 的一阶/二阶状态，也没有 FP32 master weights；显存结果不能直接换算成 AdamW 的可训练规模。
3. 本仓库的原生 attention、MoE 和调度实现仍有逐专家 Python 调用等开销，没有改写为生产环境的融合 attention 或 grouped GEMM 内核。结构更接近完整配置，不代表实现效率等同于正式训练系统。
4. 加载使用共享文件系统及 OS page cache，未执行清空系统缓存操作。逻辑读取量不是物理磁盘读取量，fresh process 也不等于冷磁盘加载。
5. 短程 loss 对齐是本轮稳定性与对照的检查，不替代完整梯度、参数更新、长期收敛或恢复一致性验收。此前八卡 BF16 路由去重的未解决问题不在本轮启用范围内。
6. GPU activity 与多线程 CPU 区间存在重叠，NCCL kernel 时长包含等待；不能把 inclusive 时间直接相加为 100% 的单步耗时，也不能据此声称 NVLink 带宽饱和。

## 复现

在仓库根目录、安装项目依赖的 Python 环境中执行。物理 GPU 编号应按运行时空闲情况选择。

```bash
PYTHONPATH=.:tests OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tests/create_scaled_checkpoint.py \
  --config configs/profile_dsv41_95b.json \
  --output-dir artifacts/dsv41-scale-20261013/fixture95b

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=.:tests OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tests/run_memory_validation.py \
  --suite benchmark --ranks 6 --cases baseline,cpu,cpu_async,cpu_cache \
  --repeats 2 --model-path artifacts/dsv41-scale-20261013/fixture95b \
  --cp 2 --ep 3 --seq-len 2048 --batch-size 8 --microbatches 2 \
  --steps 3 --warmup 1 --input-pattern fresh --learning-rate .0001 \
  --load-mode direct_shard --load-pipeline --staging-mib 128 \
  --activation-checkpoint full --memory-limit-gib 86 --cache-mib 256 \
  --distributed-timeout 900 --progress --timeout 1800 \
  --output-dir artifacts/dsv41-scale-20261013/comparison
```

Fixture 生成器拒绝覆盖非空目录。已有 fixture 可直接用于后续测试。生成器和加载路径对照已有 CPU 测试覆盖；本轮 CPU 单测 109 项，其中 101 项通过、8 项按环境跳过。

加载专项、profiling 和独立就绪检查的完整参数分别保存在 [layerwise manifest](../artifacts/dsv41-scale-20261013/loading-layerwise/manifest.json)、[profiling manifest](../artifacts/dsv41-scale-20261013/profiles/manifest.json)、[readiness manifest](../artifacts/dsv41-scale-20261013/readiness-near-capacity/manifest.json)。Profiling 使用 `--trace-variants baseline,cpu_async --load-trace-variants baseline,cpu_async --no-profile-record-shapes`，预取探针由 `--prefetch-steps 2` 启用，并放在 trace 之后保证两种策略采样时的输入和更新步数一致。

离线图表可由已有报告重新生成，不再启动 GPU 测试：

```bash
python tests/summarize_memory_timeline.py \
  artifacts/dsv41-scale-20261013/profiles/cpu_async.json \
  --bin-ms 50 --output-dir artifacts/dsv41-scale-20261013/timeline/cpu_async
```

渲染需要 Plotly。两组 overview 均经浏览器打开检查，30 条 GPU activity lane 非空、无脚本错误，截图保存在各 timeline 目录。所有 GPU 测试和监控进程均已退出；本轮未改动模型计算、优化器算法或生产 offload 默认值。
