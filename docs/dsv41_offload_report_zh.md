# DSV4.1 参数加载、Engram 卸载与缓存阶段报告

整理日期：2026-10-13。本文的小模型性能测量日期：2026-10-12。

本文汇总当前工作区的实现、历史验证和小模型性能记录。2026-10-13 新增的 **95B、40 层、六卡 H20 接近容量上限测试**见[大规模测试报告](dsv41_large_scale_results.md)：其 Engram 表占比约 25.86%，卸载节省约 7.82 GiB/卡，且发现早层预取等待与直接加载的 CPU 解码瓶颈。下面保留原有小模型测量及完整配置的容量估算，不与新负载混合统计。

## 1. 当前结论

1. 已实现模块化加载、直接分片加载、Engram CPU 常驻、异步行预取、可选 GPU 行缓存、选择性 FSDP CPU offload，以及包含优化器和 RNG 的分片保存恢复。性能开关默认关闭。
2. 此前八卡 H20、423M 参数简化模型中，Engram CPU offload 将每卡训练 allocated 峰值从 579.98 MiB 降至 512.01 MiB，节省约 68 MiB，即 11.7%；新输入场景吞吐相对 GPU 常驻基线下降 5.66%。
3. 异步预取确实能在测试中的第 1、3 层消费前完成本地查表和 H2D，但尚未证明比同步 CPU 查表稳定更快。预取完成不等于行返回通信也已完成。
4. GPU 行缓存在人为高复用场景达到约 70% 命中率，行值加缓存元数据的 H2D 减少 66.72%；相对无缓存异步路径，吞吐提升配对中位数仅 0.91%，波动较大。低复用场景没有稳定收益。
5. 正确性已有 CPU、真实 FSDP2/NCCL、连续更新及跨进程恢复证据，但八卡 BF16 路由去重的部分多步一致性问题仍未解决，完整模型和多节点生产性能也尚未验收。

## 2. 实现方式

### 2.1 模块划分与约束

| 模块 | 实现职责 | 主要位置 |
| --- | --- | --- |
| 参数布局 | 记录逻辑参数名、全局形状、本地范围、专家/表 ID，统计实际本地字节数 | `dsv41_train/layout.py` |
| 通用加载 | 权重读取、量化解码、分块读取和填充策略 | `dsv41_train/loading/` |
| DSV4 适配 | 参数名映射、gate/up 融合、分组投影、meta 分片和目标存储分配 | `models/dsv4/weights.py`、`materialize.py` |
| Engram 存储 | GPU/CPU 本地行查表、跨设备 autograd、有界 pinned staging | `models/dsv4/engram/storage.py` |
| Engram 路由与调度 | owner 路由、查询交换、ticket、异步工作线程和消费时等待 | `models/dsv4/engram/routing.py`、`table.py` |
| GPU 行缓存 | 有界缓存槽、命中、碰撞驱逐、参数更新后的失效处理 | `models/dsv4/engram/cache.py` |
| 并行和优化器 | CP/EP 分组、FSDP 包装、梯度归一化、稠密/稀疏参数组 | `models/dsv4/parallel.py`、`policies.py`、`optimization.py` |
| 分片状态 | DCP 参数/优化器映射、真实行偏移、拓扑校验和 RNG 恢复 | `dsv41_train/sharded_state.py`、`checkpoint.py` |

表中的 `models/` 路径均位于 `dsv41_train/` 下。运行策略通过独立 policy 和 CLI 参数传入，不写入模型结构配置。旧模型前向、参数键和加载回调仍保留。

当前 CPU Engram 使用原 dtype 的可训练表，只支持无 momentum、无 weight decay 的 sparse SGD；不建立 FP32 master weights。GPU AdamW 路径保留。LoRA 基线路径保留，但与直接加载、CPU Engram、选择性 FSDP offload 的组合明确拒绝。

### 2.2 参数加载

`layerwise` 保留为基线。`direct_shard` 先在 meta 参数上建立最终 FSDP 分片，再按本 rank 所需的源区域读取和解码，直接填充目标 CPU/GPU 局部存储。量化数据按块对齐读取，处理融合参数、分组投影及尾部分片，不先解码完整层。

加载流水线最多持有当前块和一个提前读取的块。后续审核修复了逐参数创建线程池、预先构建全部块范围、CPU FP8 解码开销等问题：整次加载复用一个 executor，单块参数跳过任务派发，块范围惰性生成，CPU E4M3 使用与原生解码一致的查表转换。

直接分片加载主要解决本地存储峰值和重复读取。它目前在 CPU 解码后向 GPU 发送目标 dtype 的权重，因此减少逻辑源读取量并不等于减少 H2D，也不保证加载更快。CPU offload 的参数直接落到 CPU 分片。

### 2.3 Engram 从加载到更新的时序

Engram CPU offload 是加载时就让完整本地表分片常驻 CPU。训练过程中只传输被查询的行及其梯度，不在每层前后搬运整张表。投影和门控仍在 GPU。

| 时机 | 执行位置和操作 | 数据传输 |
| --- | --- | --- |
| 加载 | 按全局行范围读取并解码本 rank 的 CPU 表；表参数排除在 FSDP 之外 | CPU 表无需完整上 GPU |
| 每个 microbatch 前向开始 | 根据 token 计算所有 Engram 层的 hash；主线程依次完成请求 owner 路由 | GPU 间交换查询 ID/计数 |
| 异步预取提交 | 主线程提交 pinned ID D2H，记录 CUDA event；工作线程等待该 event | owner GPU 的本地 ID 到 CPU |
| 后台查表 | 查询 GPU 行缓存或从 CPU 参数 gather 行；使用有界 pinned buffer 和独立 copy stream | 未命中的行 CPU 到 owner GPU |
| 消费对应 Engram 层 | 主线程 `resolve(ticket)`，等待本地结果就绪，然后返回请求方的行并恢复顺序 | owner GPU 到请求 GPU 的行返回 all-to-all |
| 反向传播 | 行梯度按路由返回 owner，通过 autograd 回到 CPU 稀疏参数 | 梯度 GPU 间通信，以及 owner GPU 到 CPU |
| 所有 microbatch 累积结束 | 完成规定的梯度归一化；SGD 合并重复稀疏行并更新 CPU 参数一次 | 不将完整表搬回 GPU |
| optimizer step 后 | 失效已更新的缓存行；需要保存时写入分片 checkpoint | 保存参数/优化器/RNG 分片 |

同步 CPU 模式在当前调用中完成查表和 H2D；异步模式将这部分工作放到后台，与前面的模型计算重叠。两者的反向均保留同步正确路径。

每张表最多保留一个未消费 ticket。Pinned 行缓冲在 DMA 完成后复用，配置的预算按表生效。优化器和 checkpoint 边界拒绝未消费 ticket；异常退出会清理工作线程。配置 CPU 表后，整模型 `.to(cuda)` 会被拒绝，防止意外迁移。

### 2.4 类似 KVCache 的 GPU 行缓存

实现采用 owner 本地的固定容量行缓存：GPU 保存行值，CPU 保存行号标签，通过直接映射槽定位，碰撞时确定性驱逐。它缓存参数行，不缓存 attention 的 K/V。

- 命中时跳过 CPU gather 和行值 H2D；未命中时读取 CPU 行并填充缓存，同一批重复 miss 可共享传输。
- 输出使用独立行张量，不引用可被覆盖的缓存槽，因此后续驱逐不会改变反向或 activation checkpoint 重算保留的值。
- CPU 参数是唯一权威副本。反向保留原请求的每次梯度贡献和顺序，交给已有 sparse SGD 合并更新。
- SGD 更新后，仅失效已修改的行；未修改行可跨步保留。无法识别的版本变化、参数替换、恢复、设备/dtype 变化及关闭会清空缓存。
- 缓存不进入 checkpoint。绕过版本追踪的 `.data` 写入需要显式清缓存。

训练中的热门行每步可能被更新，不能像推理 KV 一样无限复用；主要机会是同一次梯度累积内的重复查询。当前缓存不能省掉 ID 路由、行返回 NCCL 或梯度 D2H。缓存内部的重复 miss 合并与独立的路由去重开关是两件事。

`--engram-cache-mib 8` 表示每张本地表最多 8 MiB，两张表合计每卡 16 MiB，另有临时查询输出。默认是 0。

### 2.5 并行、FSDP offload 和保存恢复

CP/EP 分别满足整除条件即可，不要求相等。CP 组共享完整输入，用数据并行组编号设种子。普通参数在全局 rank 上 FSDP；专家在相同专家的副本组上 FSDP，按其约定额外除以 EP；Engram 按全局行 mesh 进行请求梯度平均。累积归一化在更新前执行一次。

选择性 FSDP offload 由显式全局层号与 `dense/experts` 参数域的交集决定。CPU 分片驻留和计算前后的权重/梯度迁移交给 PyTorch FSDP 管理，独立于 Engram 行预取。为保证重叠 CP/EP 进程组的通信顺序，当前关闭跨层隐式 FSDP backward prefetch。Engram 的就绪测试不能证明 FSDP 权重预取也完全隐藏。

Checkpoint 格式 v2 使用全局层号、专家 ID、Engram 真实全局行偏移，支持不均匀行分片；保存参数、实际存在的优化器状态、CPU/CUDA/数据 RNG 和 step，不汇聚完整模型。恢复要求拓扑与训练配置兼容，允许切换 CPU/GPU 放置和预取策略。缺少必要布局信息的旧格式拒绝恢复。

## 3. 测试如何组织

### 3.1 测试层次

| 层次 | 对照和检查内容 | 用途 |
| --- | --- | --- |
| 模块单测 | 分块加载、量化边界、融合映射、参数键、空请求、重复行、缓存碰撞、异常清理 | 验证局部接口和生命周期 |
| 数学对照 | CPU/GPU、同步/异步、缓存开关；输出、梯度和多步 SGD 参数更新 | 检查数值与更新语义 |
| 并行拓扑 | CPU 显式归约参考、真实 FSDP2/NCCL；独立 CP/EP 和交叉组 | 检查归一化和通信顺序 |
| 恢复 | 独立进程保存后重新启动，切换放置，比较 RNG 和下一步更新 | 排除同进程状态残留 |
| 性能计时 | 相同权重/输入、独立进程、重复运行、关闭 profiler | 测量吞吐和内存 |
| Profiling/就绪探针 | 计时结束后额外采样 CPU/CUDA timeline 与预取时间戳 | 解释开销和预取是否及时 |

六 rank 用于 CP2/EP3、CP3/EP2 等交叉拓扑；八 rank 用于真实八卡训练和 CP/EP 为 1、2、4、8 的组合。两者用途互补。

### 3.2 测试模型的规模与结构

测试使用同一套 `DeepSeekV41ForCausalLM` 实现，通过缩小层数、hidden、专家数和 Engram 表规模构造模型。它们保留 attention、MoE、mHC 多路残差、Engram 及分布式训练路径，不是只运行 embedding 查表的微基准。性能测试、拓扑正确性测试和预取边界控制使用不同模型，不能将它们的规模或显存数字混用。

以下参数量在 2026-10-13 用未分片的 meta 模型、按 `named_parameters()` 实际统计核对，不分配真实权重，不新增 GPU 测量。参数量指全局逻辑参数总数，包含未被当前 token 选中的专家和未命中的表行，不是每 token 激活参数量，也不是每卡持有量。

#### 两套主要性能模型

| 结构项 | 加载审核/八卡基线小模型 | 最新 Engram 性能模型 |
| --- | --- | --- |
| 配置文件 | [profile_memory_small.json](../configs/profile_memory_small.json) | [memory_eight_profile.json](../configs/memory_eight_profile.json) |
| 全局参数量 | 14,398,338，约 14.4M | 423,209,462，约 423.2M |
| Decoder 层数 / hidden | 5 / 128 | 5 / 512 |
| token 词表 | 512 | 2048 |
| Attention query 头数 × head dim | 4 × 32 | 8 × 64 |
| Query 低秩维度 `q_lora_rank` | 64 | 128 |
| 输出投影分组数 / `o_lora_rank` | 4 / 64 | 4 / 128 |
| 滑动窗口 | 64 token | 128 token |
| 索引头数 × 维度 / top-k | 2 × 32 / 16 | 4 × 64 / 16 |
| 每层路由专家数 / 每 token top-k | 8 / 2 | 16 / 2 |
| 每层共享专家数 / 专家中间维度 | 1 / 256 | 1 / 1024 |
| mHC 残差流数 / Sinkhorn 迭代数 | 4 / 20 | 4 / 20 |
| Engram 所在层，零起始编号 | 1、3 | 1、3 |
| 两张 Engram 表的形状 | 65539 × 64；65543 × 64 | 524291 × 256；524299 × 256 |
| n-gram 阶数 / 每阶 hash 头数 | 2、3、4 / 4 | 2、3、4 / 4 |
| 每 token、每张表查询行数 | 12 | 12 |
| 主要测量序列长度 / batch | 128 / 1 | 512 / 1 |
| 配置允许的最大序列长度 | 512 | 2048 |

这里的 `q_lora_rank`、`o_lora_rank` 是模型 attention 投影的结构维度，不表示性能测试启用了 LoRA 微调；本报告的 offload 性能测试没有启用 LoRA。

两套模型都包含 5 个完整 Decoder 层，每层有 attention 和 MoE，在第 1、3 层的 attention 之前插入 Engram 投影/门控。输入 embedding 展开为 mHC 多路残差流，经过各层后合并，再经过最终 norm 和独立的 LM head；embedding 与 LM head 不共享权重。每层 MoE 计算被选中的 2 个路由专家及 1 个共享专家。

Attention 压缩配置均为 `[0, 2, 2, 1, 1]`，第 0 层为滑窗路径，第 1、3 层生成共享 KV 状态，第 1、2、3、4 层包含索引路径，第 3 层提供 candidate 状态。保留这些跨层依赖，是为了让加载、CP 通信和 activation checkpoint 测试覆盖实际模型执行关系。

| 参数组成 | 14.4M 模型 | 423.2M 模型 |
| --- | ---: | ---: |
| Engram 表参数，不含投影/门控 | 8,389,248，58.27% | 268,439,040，63.43% |
| 路由专家参数，不含共享专家 | 3,932,160 | 125,829,120 |
| 其余参数 | 2,076,930 | 28,941,302 |
| 全局 Engram 表 BF16 存储 | 约 16 MiB | 约 512 MiB |
| world8 下每卡 Engram 表存储 | 约 2 MiB | 约 64 MiB |

“其余参数”包括 attention、共享专家、路由器、mHC、Engram 投影/门控、embedding 和 LM head。两套模型都特意让 Engram 占较大参数比例以观察 offload 容量收益，但查表只访问少量行，参数比例不代表计算时间比例。14.4M 与 423.2M 测试每卡显存节省不同，首先来自本地表权重约 2 MiB 与 64 MiB 的差异；训练峰值还包含梯度、激活和临时缓冲。

#### 正确性模型与预取边界模型

分布式梯度/更新对照使用更小的模型，以便同进程容纳候选模型和参考模型，并遍历多种 CP/EP 配置。它由 [memory_config](../tests/test_memory_modules.py) 和 [fixture_dimensions](../tests/memory_validation_cases.py) 组合得到：

| 结构项 | 六 rank 交叉拓扑模型 | 八 rank 拓扑模型 |
| --- | --- | --- |
| 全局参数量 | 191,896 | 222,936 |
| Decoder 层数 / hidden / token 词表 | 5 / 32 / 32 | 5 / 32 / 32 |
| Attention 头数 × head dim / query 低秩维度 | 2 × 32 / 32 | 2 × 32 / 32 |
| 路由专家数 / 每 token top-k / 共享专家数 | 6 / 2 / 1 | 8 / 2 / 1 |
| 专家中间维度 / mHC 残差流数 | 32 / 2 | 32 / 2 |
| Engram 层和表形状 | 第 1、3 层；127 × 32、131 × 32 | 第 1、3 层；127 × 32、131 × 32 |
| n-gram 阶数 / 每阶 hash 头数 | 2、3 / 1 | 2、3 / 1 |
| 序列长度 | 12 | 16 |

两张表采用 127、131 行，特意保留不均匀分片。六 rank 的 6 个专家与长度 12 支持 CP2/EP3、CP3/EP2；八 rank 改为 8 个专家与长度 16，使 CP8、EP8 也满足整除约束。它们用于数值、通信和恢复验证，不用于外推训练吞吐。

第 0 层立即消费对照使用独立的 [profile_engram_first_layer.json](../configs/profile_engram_first_layer.json)：全局 23,769,090 参数，仍是 5 层、hidden 128、8 个路由专家，但只在第 0 层放一张 65539 × 256 的 Engram 表，测试序列长度 512。这样消费之前没有 decoder 计算窗口，可检测预取来不及完成的边界；它不是将 423M 模型的层号简单修改后做等规模对照。

上述加载/性能 fixture 由 [synthetic_checkpoint](../tests/test_dsv4_loading.py) 生成，包含 FP8 矩阵/表和 FP4 专家编码，加载后按测试指定的 FP32/BF16 存储训练。部分专家权重采用固定量化模式，输入是合成 token；这些模型覆盖实现路径，但没有真实预训练权重和自然文本的数值/路由分布。测试结果不能作为完整 DSV4.1 模型质量、真实语料缓存命中率或生产吞吐的证明。

### 3.3 最新性能负载与控制变量

| 项目 | 设置 |
| --- | --- |
| 硬件/软件 | 8 张 NVIDIA H20，Python 3.10.20，PyTorch 2.10.0+cu129 |
| 模型 | 合成量化 checkpoint，423,209,462 参数，5 层，hidden 512，16 专家 |
| Engram | 第 1、3 层，524291/524299 行，每行 256 维 |
| 并行与序列 | world8，CP2/EP4，batch 1，sequence 512 |
| 更新 | BF16 存储，sparse SGD，学习率 0.0001，无 momentum/weight decay |
| 隔离变量 | layerwise 加载，关闭路由去重、dense foreach、FSDP CPU offload |
| 缓存容量 | 启用时每表 8 MiB，每卡合计 16 MiB |
| 重复方式 | 每策略 3 次独立八进程运行，策略执行顺序确定性打乱 |
| 计时 | 4 次 warmup，20 次 optimizer step；输入生成不计时 |
| 低复用 `fresh` | 每步 2 个 microbatch，每个 microbatch 都是新随机输入 |
| 高复用 `repeated` | 每步 4 个相同 microbatch，下一步更换输入 |

CP 同组使用相同完整输入，相同策略对照使用相同 fixture 和随机种子。计时前 barrier；每步完成 CUDA 同步后记录时间，吞吐按全局唯一输入 token 数和最慢 rank 总耗时计算。训练吞吐不包含加载、预取诊断和 profiler 采样。

该轮优化前、过渡实现、最终 fresh 和最终 repeated 合计 33 次独立吞吐试验，另采集两组八 rank GPU trace。不同代码版本和工作负载没有合并成一组样本。同步/异步的直接对照来自另一次 22 次运行的预取专项测试，设置为 4 次 warmup、15 次计时更新，单独报告。

### 3.4 统计口径与环境限制

- 表中的 tokens/s 是三次运行的中位数；括号范围为观测最小/最大值，不是置信区间。
- 性能变化先在同一 repeat 内计算候选/基线比值，再取中位数。因此不能直接用表中两个吞吐中位数相除复现配对变化。
- 显存每次先取各 rank 的 PyTorch allocated 峰值最大值，再汇总。另记录 reserved 和 CPU RSS；allocated 的下降不等于 `nvidia-smi` 使用量同幅下降。
- DTensor 内存统计使用本地存储，避免把全局逻辑形状当作本卡实际占用。稀疏梯度按实际存储统计。
- H2D 流量是所有 rank 每次 optimizer step 的总和。缓存比较包含行值和缓存索引元数据，不包含全部 ID/NCCL/FSDP 流量。
- GPU 2 当时有其他服务占用约 86 GiB，主机为共享环境。三次重复不足以建立稳定的小幅加速结论。
- 随机输入和人为重复输入都不是代表性真实语料；高复用结果不代表真实训练会有 70% 命中率。

## 4. 实测效果

### 4.1 最新新输入场景

| 策略 | tokens/s 中位数 [范围] | 相对 baseline 吞吐下降，配对中位数 | 训练峰值 MiB/卡 | 加载峰值 MiB/卡 |
| --- | ---: | ---: | ---: | ---: |
| GPU 常驻 baseline | 11073.18 [11052.55, 11115.31] | 0% | 579.98 | 187.73 |
| CPU 异步 | 10429.18 [10271.05, 10446.24] | 5.66% | 512.01 | 123.61 |
| CPU 异步加缓存 | 10352.14 [10294.02, 10451.92] | 6.86% | 528.01 | 123.61 |

无缓存卸载节省约 67.97 MiB/卡，约 11.7%；启用每卡 16 MiB 缓存后净节省约 51.97 MiB，约 9.0%。吞吐下降 5.66% 对应单步时间增加约 6.00%；缓存路径下降 6.86% 对应单步时间增加约 7.37%。

该场景缓存命中率只有 2.19%。行值 H2D 从 48.00 降到 45.86 MiB/step，但新增缓存元数据约 2.84 MiB，总计 48.69 MiB，反而略高。缓存相对无缓存的吞吐变化为 +0.05%，范围 -1.30% 到 +0.79%，没有稳定收益。

加载耗时中位数为 baseline 1.08 秒、CPU 异步 1.13 秒、加缓存 1.13 秒。缓存按需分配，不改善 checkpoint 读取或解码。

### 4.2 高复用场景

| 策略 | tokens/s 中位数 [范围] | 相对 baseline 吞吐下降，配对中位数 | 训练峰值 MiB/卡 | 行值加缓存元数据 H2D MiB/step |
| --- | ---: | ---: | ---: | ---: |
| GPU 常驻 baseline | 11400.36 [11290.69, 11403.93] | 0% | 585.66 | 0 |
| CPU 异步 | 10691.73 [10647.58, 10870.28] | 5.30% | 513.57 | 96.00 |
| CPU 异步加缓存 | 10849.61 [10744.17, 11354.34] | 4.86% | 529.57 | 31.95 |

缓存命中率 70.00%，H2D 降低 66.72%。缓存相对无缓存的吞吐变化为 +0.91%，范围 -0.19% 到 +6.20%。较快的一次重复拉高了上限，不能据此宣称稳定提升 6%。

两类最终负载中，所有记录到的 loss 都与对应 GPU baseline 完全一致。这是该低学习率负载上的 loss 检查，不替代梯度和多步参数一致性验收。

### 4.3 加载优化的独立结果

以下来自较早的 14.4M 参数模型，不能与上述 423M 吞吐直接比较。旧实现来自 `e3e808c`，与修改后的实现使用相同测量脚本和权重；每项三次独立进程。

| CPU warm load | 修改前中位数 ms | 修改后中位数 ms | 耗时降低 |
| --- | ---: | ---: | ---: |
| layerwise | 116.25 | 90.78 | 21.9% |
| direct 同步 | 131.43 | 93.40 | 28.9% |
| direct 流水线 | 185.77 | 97.03 | 47.8% |

Warm load 是同进程内再次加载，文件系统页面可能已缓存，不是冷盘吞吐。CPU direct-pipeline trace 中，线程池创建由 213 次降至 1 次；初始化/提交/关闭的 inclusive 时间由 51.38 ms 降至 2.10 ms。该时间与其他范围可能重叠，不作为可直接相加的壁钟收益。

两卡 H20 的 direct-pipeline 首次加载由 991.29 ms 降至 943.79 ms，约改善 4.8%。但修改后的 direct 为 943.79 ms，仍略慢于同轮 layerwise 的 924.51 ms；其价值还包括加载峰值从 17.94 降至 13.89 MiB，以及两卡合计编码逻辑读取量减少 16.3%。不能写成“direct 全面快于 layerwise”。

### 4.4 选择性 FSDP offload 与组合模式

2026-10-10 的八卡小模型基线补测使用 14.4M 模型、CP2/EP4，每策略三次独立进程、2 次 warmup、5 次计时。组合模式启用 direct pipeline、异步 CPU Engram、路由去重、dense foreach，以及第 1、3 层 dense/experts FSDP CPU offload。

| 策略 | 训练 allocated 峰值 MiB/卡 | 加载 allocated 峰值 MiB/卡 | tokens/s 中位数 |
| --- | ---: | ---: | ---: |
| GPU 常驻 baseline | 89.622 | 9.257 | 2817.63 |
| CPU Engram 同步 | 87.414 | 6.976 | 2904.02 |
| CPU Engram 异步 | 87.407 | 6.976 | 2805.05 |
| 组合模式 | 85.963 | 0.782 | 2799.43 |

组合模式训练峰值下降 4.08%，加载峰值下降 91.56%，吞吐没有稳定改善。该模型每卡本来只有约 2 MiB 表权重，因此节省比例小于表更大的负载。Reserved 峰值为 baseline 132 MiB、CPU 表 136 MiB、组合 134 MiB，并未下降。

这些是组合策略的效果，不能全部归给 FSDP；更早的独立 dense/experts 测试仅为短程测量，见原始报告。本报告不对 FSDP offload 宣称稳定加速。

## 5. 预取是否真的提前完成

专项探针记录 `prefetch`、后台查表开始/完成、`resolve` 以及 `Future.result()` 时间。当前 CPU store 返回前已确认行 H2D 完成，因此 `resolve_start - store_ready > 0` 可证明 owner 本地行已提前就绪。探针不额外执行 CUDA 同步，且在吞吐计时结束后运行。

| 测试 | 本地查表/H2D 在消费前完成 | 最小提前量 | 解释 |
| --- | --- | --- | --- |
| 最新 fresh、无缓存、第 1/3 层 | 384 次查询全部完成 | 8.85 / 37.45 ms | 两层合计，各 192 次 |
| 最新 fresh、加缓存、第 1/3 层 | 384 次查询全部完成 | 8.44 / 35.73 ms | 缓存未造成该负载下超时 |
| 最新 repeated、加缓存、第 1/3 层 | 768 次查询全部完成 | 8.42 / 36.79 ms | 每层各 384 次 |
| 早期第 0 层立即消费对照 | 256 次均未提前完成 | 最差晚约 1.73 ms | 无前置 decoder 层提供重叠窗口 |

最新 fresh 的最慢 rank Future 调用总时间平均约 0.020 ms/step，两种 CPU 策略都很小。缓存把平均本地查表耗时从约 1.60 增至 3.53 ms，但仍提前完成。

这只证明 owner 上本地结果及时，不代表请求 GPU 已经收到行：`resolve` 中的行返回 collective 仍在消费路径上。早期独立同步/异步对照中，423M 模型异步相对同步吞吐变化为 -0.51%，未显示加速。第 0 层错过预取截止时间也不自动意味着比同步模式更慢，同步模式同样需要支付读取成本。

## 6. Profiling 方法与性能解释

使用 `torch.profiler` 的 CPU/CUDA activity，通过 Kineto/CUPTI 捕获 kernel 和 memcpy；`record_function` 标记加载、前向、反向、优化器、Engram hash/路由/查表/resolve。开启所有线程采集以覆盖后台工作线程。加载 trace 单独采集并关闭 shape capture，避免 profiler 保留 tensor 引用干扰 FSDP meta 存储替换。

原始输出为 Chrome trace JSON，可用 Perfetto 打开；脚本另生成离线 Plotly timeline、逐 rank 明细以及 CSV/JSON 时间表。概览按活动类别合并重叠区间，避免把同类并行事件重复计入忙碌时长；详细图保留每个 stream 的原始区间。

最新 fresh 的一次独立 profiled update，八 rank 均值如下：

| 范围/活动 | CPU 异步 ms/step | CPU 异步加缓存 ms/step |
| --- | ---: | ---: |
| 前向 host 范围 | 259.19 | 265.99 |
| 反向 host 范围 | 329.73 | 331.76 |
| 优化器 host 范围 | 17.84 | 20.29 |
| Engram ID 路由 host 范围 | 27.95 | 30.20 |
| 本地查表及 H2D 完成等待 host 范围 | 6.77 | 15.92 |
| Engram resolve，含行返回路由 | 2.00 | 2.04 |
| GPU compute kernel 时长和 | 37.70 | 37.76 |
| 全部 H2D DMA 时长和 | 0.577 | 0.542 |
| 全部 D2H DMA 时长和 | 0.457 | 0.450 |
| NCCL kernel 时长和，含等待 | 350.04 | 352.78 |

这些 host 范围包含子调用，跨线程/stream 活动也会重叠，不能相加为完整 step，也不能作为训练各部分的百分比。Profiler 明显增大 CPU 开销；NCCL kernel 时长含 rank 到达不齐和同步等待，不是网络传输净耗时，不能据此宣称带宽打满。

根据 trace 已执行三项优化：合并边界/计数元数据 D2H，固定长度 scatter 计数替代动态 `bincount`，以及复用 pinned buffer、把异步 ID 事件等待放到后台线程。每 rank 每步四次路由中的标量读取由 12 次变为 0，显式 stream synchronize 由 24 次变为 4。仍需要的计数 D2H 会等待 GPU/collective 进度，部分等待转移到了 memcpy 调用，调用次数下降不等于等量壁钟节省。

优化前后，无缓存 offload 的配对吞吐损失中位数由 6.13% 变为 5.66%，但样本波动重叠，不能宣称已稳定回收 0.47 个百分点。每步约 18000 个计算 kernel，CPU launch、调度与通信同步值得继续定位。当前证据不支持把剩余损失全归因于 H2D 带宽或 CPU SGD。

缓存没有明显加速的直接原因是：减少的前向 H2D 已被隐藏，ID 路由、行返回、反向传输仍然存在，缓存还增加标签处理、索引和元数据开销。现阶段已证明的是高复用下的带宽节省，而不是稳定缩短训练关键路径。

## 7. 正确性结果与未通过项

| 检查 | 已有结果 | 边界 |
| --- | --- | --- |
| 最新单元测试集合 | 108 项，100 通过，8 项 GPU/外部权重相关跳过 | 跳过不计作通过 |
| GPU 缓存专项单测 | 8 项全部通过 | FP32/BF16、空/重复行、驱逐后反向、失效、缓冲和恢复 |
| 缓存生命周期 | dtype × 同步/异步 × 去重，共 8 组合，各 24 次更新 | 与同策略无缓存 CPU 对照的输出/梯度/更新逐位一致；warmup 后 allocated 波动为 0 |
| 生命周期 GPU 参考 | FP32 最大输出/权重误差 2.24e-8，梯度 1.67e-6；BF16 无去重逐位一致 | BF16 去重梯度可差 0.0078125，在测试容差内且最终权重相同，不能称全部 GPU 逐位一致 |
| 最新八卡 CP2/EP4 FP32 | 对独立未分片参考，最大梯度误差 6.76e-6 | 开启缓存、完整 AC、选择性 FSDP offload 和恢复；路由去重关闭 |
| 最新八卡 CP4/EP2 BF16 | 对同拓扑 GPU 参考，最大梯度误差 3.05e-5 | 同样覆盖组合功能；路由去重关闭 |
| 最新六进程 CPU 交叉拓扑 | CP2/EP3、CP3/EP2 输出/梯度/恢复通过 | 不替代 NCCL/GPU 测试 |
| 最新独立八进程恢复 | CPU 表加缓存/FSDP offload 保存，GPU 表恢复；下一步参数最大误差 4.66e-10，RNG 精确一致 | 同 world8/CP2/EP4，缓存不恢复 |
| 历史完整 FP32 拓扑矩阵 | 六卡 `{1,2,3,6}` 与八卡 `{1,2,4,8}` 各 16 个 CP/EP 组合通过 | 是当时版本的历史覆盖，不代表缓存修改后全矩阵全部重跑 |
| 历史 LoRA/Qwen 回归 | LoRA 六种两卡 FSDP/AC 组合通过；Qwen 两卡最大梯度误差 4.47e-8 | LoRA 与新 offload 的组合仍不支持 |

分布式数值测试还覆盖 padding、不均匀表分片、两次 microbatch 累积、外部 loss 缩放、多个 SGD step、activation checkpoint 重算和模型重新恢复。加载测试覆盖完整逻辑权重、量化块边界、融合/分组投影、尾部分片及层窗口，并禁止 direct 路径使用完整 tensor 读取。

**尚未关闭的正确性问题：** 较早八卡 BF16 去重测试中，CP1/EP8、CP8/EP1、CP8/EP8 的多步 logits 对照失败；另有 CP2/EP4 参考前向 NCCL 错误。已有诊断显示去重引入的中间 BF16 舍入可能是数值差异来源，但没有完成修复与全矩阵重新验收。最新关闭路由去重的缓存测试通过，不能覆盖这些历史失败。

真实完整 checkpoint 的 HF 参考一致性、多节点性能和真实语料训练收益仍无验收结果。早期环境阻塞后已成功进行八卡 GPU 测试，不能继续描述为“没有 GPU 证据”，也不能据此写成完整生产验收已完成。

## 8. 完整 DSV4.1 的配置估算

### 8.1 显存容量

仓库完整配置在第 1、14 层使用两张表，行数为 384006168、384016682，行宽 256。按当前 BF16 训练存储，表参数共 196613849600 个：

```text
表权重字节 = (384006168 + 384016682) × 256 × 2
           = 393227699200 bytes
           = 366.22 GiB
```

当前 Engram 按全局 world size 分片，若两张表均加载：

| GPU 数量 | 每卡可移出 GPU 的常驻表权重 |
| --- | ---: |
| 8 | 45.78 GiB |
| 16 | 22.89 GiB |
| 32 | 11.44 GiB |

这是权重存储计算，不是完整模型实测训练峰值。GPU 缓存及额外临时缓冲需要扣除；CPU 总共仍需约 366 GiB 存放表，另有梯度、staging 和其他内存。只运行前 9 层时只包含第一张表，八卡对应约 22.89 GiB/卡。

不能把 sparse SGD 按全表稠密梯度/Adam 状态倍数放大收益，也不能把“表权重可省约 46 GiB/卡”理解为八卡已能完成整个模型训练，专家权重、梯度和激活仍需单独核算。

### 8.2 速度：仅作假设场景，尚无实测预测

真实配置每层每 token 查询 `(4 - 1) × 8 = 24` 行。两张表、每行 256 维 BF16，未去重/缓存时每 token 共约 24 KiB 行值 H2D，反向另有相近量级 D2H。这是当前性能 fixture 每 token 查表量的两倍，不随整张表的参数量同比增加。

当前 fresh 每 step 4096 个全局 token，按吞吐中位数换算，offload 额外约 23 ms。若仅作预算示例，保持 token 数和拓扑不变，假设完整配置额外约 50 ms/step，则：

| 假设 GPU 常驻 baseline 单步时间 | 加 50 ms 后的吞吐下降 |
| --- | ---: |
| 1 秒 | 4.8% |
| 2 秒 | 2.4% |
| 5 秒 | 1.0% |

这就是此前“约 1%～5%”讨论的假设来源，不是完整模型速度预测或保证区间。把查询量翻倍近似成额外时间翻倍并未验证：大表随机访问、CPU cache/NUMA、长序列和 FSDP offload 竞争都可能改变结果，实际损失也可能超过该区间。完整模型 baseline 本身尚未测得。

## 9. 复现与资料索引

在仓库根目录运行，使用同一 CUDA PyTorch 环境。当前机器历史测试使用 `/opt/python/3.10.20/bin/python3.10`，并将项目、`tests` 及现有 PyTorch 环境的 site-packages 放入 `PYTHONPATH`。以下用 `python` 代指这个可用解释器。

新输入吞吐、预取诊断和八 rank trace：

```bash
PYTHONPATH=.:tests python tests/run_memory_validation.py \
  --suite benchmark --ranks 8 --cp 2 --ep 4 \
  --cases baseline,cpu_async,cpu_cache --repeats 3 --steps 20 --warmup 4 \
  --seq-len 512 --learning-rate .0001 --input-pattern fresh \
  --prefetch-steps 4 --trace-variants cpu_async,cpu_cache \
  --config configs/memory_eight_profile.json \
  --output-dir artifacts/report-cache-fresh
```

高复用对照使用相同命令，改为 `--input-pattern repeated --microbatches 4`，并使用独立输出目录。真实语料不能由该开关替代。

单测及缓存连续更新：

```bash
PYTHONPATH=.:tests python -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=.:tests python -m unittest test_engram_cache -v
PYTHONPATH=.:tests python tests/dsv4_engram_offload_cuda.py \
  --cache-bytes 4096 --output artifacts/report-cache-lifecycle.json
```

从已有报告生成离线 timeline，不重新训练：

```bash
PYTHONPATH=.:tests python tests/summarize_memory_timeline.py \
  artifacts/engram-cache-20261012/final-fresh/cpu_cache-r0.json \
  --phases training --output-dir artifacts/report-cache-timeline
```

Timeline 渲染需要 Plotly，HTML 生成后可离线打开。分布式梯度和 fresh-process 恢复的完整命令见下列专题报告。

| 资料 | 内容 |
| --- | --- |
| [模块与运行参数](memory_offload.md) | 策略、加载和 checkpoint 契约 |
| [加载审核及优化](loading_review.md) | 独立前后对照、读取量及解码分析 |
| [八卡基线补测](memory_eight_baseline.md) | 同负载 GPU baseline 和组合内存对照 |
| [异步预取专项](engram_prefetch_results.md) | 同步/异步对照和第 0 层边界控制 |
| [最新 Engram 缓存报告](engram_cache_results.md) | 最终性能分布、测试与复现命令 |
| [最新汇总 JSON](results/engram-cache-h20.json) | 每个样本、配对比值、缓存流量与探针结果 |
| [早期验收结果](memory_results.md) | 加载、拓扑、SGD/AdamW 恢复和 LoRA/Qwen 回归；其中“尚未 GPU profiling”的历史状态已被后续测量更新 |
| [八卡数值失败记录](memory_eight_results.md) | FP32 完整矩阵及未解决的 BF16 去重问题 |
| [Timeline 方法](memory_timeline.md) | 采集、跨 rank 展示和时间统计口径 |
| [最新异步 timeline](../artifacts/engram-cache-20261012/timeline-async/timeline.html) | 八 rank CPU/CUDA 活动概览 |
| [最新缓存 timeline](../artifacts/engram-cache-20261012/timeline-cache/timeline.html) | 缓存路径活动概览 |
| [缓存 rank 0 详细 timeline](../artifacts/engram-cache-20261012/timeline-cache/training-rank-0.html) | 精确 kernel、memcpy 与 CPU 范围 |
| [缓存生命周期原始结果](../artifacts/engram-cache-20261012/lifecycle-cache.json) | 24 次更新、数值误差和 allocated 波动 |
| [跨放置恢复原始结果](../artifacts/engram-cache-20261012/resume-placement/resume-result.json) | 新进程下一步参数误差和 RNG |
| [完整模型配置](../dsv41_train/models/dsv4/config.json) | 容量估算的行数、维度与层号来源 |

## 10. 后续需要补齐的证据

优先使用真实表规模与真实文本评估访问局部性、CPU gather、NUMA 和缓存命中率，同时测量第 1 层的最小预取余量。随后在独占 GPU 条件下补充更长的配对测试，分离请求路由、行返回、反向 D2H 和 CPU 更新的关键路径开销，再评估与 FSDP offload 同时启用时的传输竞争。

BF16 路由去重的历史失败需要独立修复与重跑，不能用缓存收益或低学习率 loss 一致代替。当前保留缓存和路由去重默认关闭；选择 offload 时使用已确认的显存收益，并为尚未消除的吞吐损失留出预算。
