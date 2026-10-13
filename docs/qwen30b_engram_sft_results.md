# Qwen3-30B + Engram：真实 SFT 卸载与 Cache 测试

测试日期：2026-10-13。硬件为本机 8 张 NVIDIA H20，每卡约 95.58 GiB；PyTorch 2.10.0+cu129。

功能介绍、入口选择和完整数据准备命令见 [Engram 功能与使用指南](engram_features.md)。
本文中的 `artifacts/` 链接指测试机器的本地产物，不随 Git 发布；仓库保留报告与紧凑 JSON 统计，timeline 可按指南重新生成。

## 结论

使用真实 Qwen3-30B-A3B-Instruct-2507 权重，新增 11.063B 个可训练 Engram 表参数，运行 Alpaca 回答部分监督的全参数 SFT。
Engram CPU offload 每卡节省约 **2.75 GiB，训练峰值下降 7.83%**，代价是 **吞吐下降 13.43%，每步时间增加约 15.52%**。
CPU cache 的命中率为 **8.76%**，但本次没有额外吞吐收益。

本轮 offload 仅指 Engram 表及其稀疏梯度；Qwen 主干、Engram 投影和门控始终在 GPU。没有启用主干 FSDP CPU offload。

| Engram 放置 | Cache | 吞吐中位数 tokens/s | 两次运行 tokens/s | 训练峰值 allocated GiB/卡 | 相对 GPU 无 cache 吞吐 |
| --- | --- | ---: | --- | ---: | ---: |
| GPU | 关 | 10,698 | 10,569 / 10,827 | 35.097 | 基线 |
| GPU | 开 | 10,854 | 10,855 / 10,854 | 35.344 | +1.46% |
| CPU | 关 | 9,261 | 9,607 / 8,915 | 32.350 | -13.43% |
| CPU | 开 | 9,265 | 9,476 / 9,054 | 32.601 | -13.39% |

每组两次独立进程运行，每次 2 步预热、3 步正式计时。表中吞吐是两个运行吞吐的中位数；显存是正式计时中八卡、两次运行的最大 PyTorch allocated。
Cache 在两种放置下均实际执行，并非 GPU 模式下忽略开关。GPU cache 的小幅吞吐差异不足以证明稳定收益，且会额外占用约 0.25 GiB。
CPU cache 相对 CPU 无 cache 的吞吐差异只有 +0.046%，远小于两次运行间的波动。

最早的 `repeat-0-gpu` 与其他用户的 RTP-LLM 任务重叠，其 3,253 tokens/s **已排除**。后续 CPU 组启动时，遥测显示八卡均为 0–4 MiB 占用；此后持续记录 GPU 进程。
最终 GPU 基线使用 `repeat-1-gpu` 和补跑的 `repeat-2-gpu`。保留了受干扰的原始结果，未用它制造加速比。

## 模型与新增结构

| 项目 | 实际配置 |
| --- | --- |
| 预训练主干参数 | 30,532,122,624，约 30.532B |
| 原始模型 | Qwen3-30B-A3B-Instruct-2507；本地源目录实际指向 FP8 版本 |
| 主干结构 | 48 层，hidden 2048；GQA：32 个 Q head、4 个 KV head，head dim 128 |
| MoE | 每层 128 个专家，top-8，专家中间维度 768 |
| 新增 Engram 层 | 全局层号 1、24，零起始，即第 2、25 层 |
| n-gram | 2/3/4-gram，每种 8 个 hash head，共 24 个 |
| 表结构 | 两张表分别 21,603,700、21,612,092 行；每行 256 维 |
| Engram 表参数 | 11,063,242,752，约 11.063B |
| 新增投影与门控参数 | 50,339,840 |
| 总参数 | 41,645,705,216，约 **41.646B** |
| Engram 表占总参数 | **26.565%** |

复用 DSV4 的 `NgramHash`、`RowShardedEmbedding`、`RowRouter`、CPU staging 和 cache 实现。
Qwen 使用单残差流，因此 Engram 投影和门控采用 `hc_mult=1`，在相应 decoder layer 前注入检索值；没有将 Qwen 改成 DSV4 的 mHC 或压缩注意力。
Token map 为 identity，hash 的基本词表大小为每 head 900,000，再按现有算法分配独立素数桶。

新增表及投影使用固定种子初始化，标准差 0.02。所有放置均先以相同 CPU RNG 生成分块数据，保证初始值一致。
这是“预训练 Qwen + 新增 Engram”的性能实验，不是已经预训练好 Engram 的模型，也不是完整 DSV4.1 的训练质量实验。

## 数据与训练方法

数据使用 [Stanford Alpaca / tatsu-lab/alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca)，52,002 条记录，数据许可 CC-BY-NC-4.0。
官方数据分片通过 Hugging Face 镜像下载到私有目录，记录了 SHA-256；未使用随机 token 代替 SFT 数据。

- 使用 checkpoint 的 Qwen tokenizer，按 `user/assistant` chat 格式编码 instruction、可选 input 和 output。
- Prompt 标签为 `-100`，只监督回答及结束标记；打包序列首 token 也屏蔽。
- 固定 seed 42 打乱后连续 packing，得到 2,120 个长度 2,048 的序列，共 4,341,760 个输入 token，其中 3,006,430 个监督 token。Packing 允许前文跨样本可见。
- 每卡每个 microbatch 4 个序列，梯度累积 2，八卡每步共 64 个序列、**131,072 个输入 token**。
- CP=1、EP=8；普通参数 FSDP 八卡，专家按 EP 所有权分配，每个 rank 16 个专家；Engram 按真实行范围分布到全部八卡。
- 全参数 BF16 训练，完整 activation checkpoint；SGD，lr=0.01，无 momentum、无 weight decay，稠密组 foreach 开启，表使用 sparse SGD。未启用 LoRA 或 FP8 训练。
- Loss 按全局 microbatch 累积区间中的有效回答 token 数归一化；专家梯度额外除以 EP=8，表由行路由反向执行全局请求平均。
- 所有变体的初始化、数据顺序、batch、累积、优化器和请求去重设置相同。CPU 两组均启用 async prefetch；GPU 两组使用同步本地读取。

正式计时排除了加载、预热、profiler 和最终结果序列化。数据预先编码并将本次使用的 token 张量放入 GPU，吞吐不包含在线 tokenizer/DataLoader。
每次计时覆盖 393,216 个输入 token；这是短程性能与更新测试，不是完整 epoch 或收敛测试。

## 卸载与 Cache 如何执行

1. 加载时将 FP8 主干按 128×128 block scale 解码为 BF16，逐层装入并 FSDP 分片；每个 EP rank 只读取自己的专家。新增 CPU 表直接在 CPU 初始化，之后不会整模型搬回 GPU。
2. 每次 forward 先根据完整 token 序列计算 hash。主线程按固定顺序交换查询 ID；CPU worker 在本地所属行中查表并通过有界 pinned staging、独立 copy stream 做 H2D。
3. 到目标层前消费 ticket，等待未完成的查表/H2D，再在主线程交换返回行；投影与门控在 GPU 运算。查表发生在 activation checkpoint 区域之外，重算复用本次 forward 的行张量。
4. 反向将命中行梯度通过路由返回 owner。CPU 表的行梯度 D2H 后在 CPU 累积，optimizer step 前进行 FP32 coalesce，再回到存储 dtype 执行 sparse SGD。
5. Cache 每张表每 rank 128 MiB，共 256 MiB/卡，使用固定 slot 和 tag。SGD 后失效所有更新过的缓存行；外部参数版本变化触发保守失效，不能把旧权重当成跨步只读 KV 使用。

CPU offload 把约 2.576 GiB/卡表权重及命中行梯度从 GPU 移到 CPU。由于 GPU 基线已经按八卡分片，并且训练峰值还包含主干梯度、激活和 logits，表占参数 26.6% 并不意味着训练峰值也下降 26.6%。

## 传输与性能分布

以下通信量是 Engram 内部计数器的八卡、正式计时步平均，单位 MiB/卡/optimizer step；不包含主干 FSDP/EP 通信。

| CPU 表模式 | 行数据 H2D | Cache 元数据 H2D | 行梯度 D2H | Cache 命中率 |
| --- | ---: | ---: | ---: | ---: |
| 无 cache | 300.52 | 0 | 300.52 | 不适用 |
| 有 cache | 246.55 | 15.57 | 300.52 | 8.76% |

Cache 使行数据 H2D 减少约 18.0%，计入元数据后净 H2D 减少约 **12.8%**。D2H 梯度没有减少。
高频行在每次 SGD 后需要失效，而每步仅两个 microbatch，跨步复用受到可训练权重更新的限制；KVCache 式只读复用假设不适用于这部分行。

无 profiler 的主线程阶段包围时间，八卡、两次运行平均：

| 模式 | Forward ms/更新 | Backward ms/更新 | Optimizer ms/更新 |
| --- | ---: | ---: | ---: |
| GPU，无 cache | 3,626 | 8,548 | 48 |
| GPU，有 cache | 3,477 | 8,533 | 42 |
| CPU，无 cache | 3,863 | 9,062 | 1,064 |
| CPU，有 cache | 3,928 | 8,980 | 1,120 |

CPU 表将 optimizer 阶段增加到约 1.1 秒/更新，其中包括稀疏梯度 coalesce 和 CPU SGD，是明显的卸载成本。正式计时的 resolve 主线程时间约 175 ms/更新，cache 组约 208 ms；它包含等待与行返回路由的调用时间。

另外每个模式单独采集一个 optimizer step 的 rank 0 PyTorch CPU/CUDA trace，包含模型区域和后台查表线程。下面是 trace 中原始设备 activity 的时长之和：

| Activity，ms/采样更新 | GPU 无 cache | CPU 无 cache | CPU 有 cache |
| --- | ---: | ---: | ---: |
| Compute kernels | 7,048.6 | 7,045.5 | 7,051.4 |
| NCCL kernels | 4,066.8 | 4,294.8 | 4,436.0 |
| H2D | 2.23 | 7.09 | 6.46 |
| D2H | 25.88 | 180.24 | 185.15 |

这些时长可能跨 stream 重叠，NCCL 时长也包含等待，不能相加代替无 profiler 的 step wall time。
单次采样约 13.1 万个计算 kernel 和 1,570 次 NCCL kernel，原生 MoE 前后向、通信及大量 host/device 同步占主要开销。
纯 H2D 只有数毫秒；cache 节省少量 DMA 时间，无法抵消 CPU 元数据管理、查表和稀疏更新成本。

Trace 中第 1 层首次消费仍有约 51–66 ms 的查表未就绪窗口，第 24 层首次查表约提前 1.1–1.3 秒完成。
这是带 profiler 的 host 观察，只说明早层存在等待、深层有预取余量；没有把它当成无 profiler 下所有请求都能提前完成的证明。

## 加载与源文件保护

共享源 `/mnt/nas1/hf/Qwen3-30B-A3B-Instruct-2507` 实际为指向 FP8 目录的软链接。
测试副本位于 `artifacts/qwen30b-engram/models/Qwen3-30B-A3B-Instruct-2507`，复制了实际文件，未使用软链接或硬链接共享底层文件。

4 个权重分片和 config/index/tokenizer 等 4 个关键文件，共 8 个文件，在测试前后均做了完整 SHA-256 对照；原始文件的 size、mtime 和 SHA-256 全部未变。
训练只更新内存中的参数，没有写回共享文件或私有 checkpoint 文件。

后续可比运行的加载时间中位数约为 GPU 33.00 秒、CPU 33.79 秒、CPU cache 34.90 秒。
这些时间包含新增 11.063B 表的随机初始化。首轮拆分显示约 31 秒用于新增表初始化，余下约 13.6 秒为元数据、主干解码、GPU 搬运和 FSDP 包装；不能把整段时间都叫作权重磁盘读取。
每 rank 逻辑读取约 5.402 GiB，所有 rank 合计约 43.22 GiB，包含各 rank 对普通参数的重复读取。加载采用逐层 materialization，不先构建每卡完整主干。

校验记录：[测试前](../artifacts/qwen30b-engram/source-before.json)、[测试后](../artifacts/qwen30b-engram/source-after.json)。

## 正确性与边界

- CPU 全回归：114 项，104 通过、10 项按环境跳过；CUDA 定向回归：13 项全部通过，包含新增 GPU cache 的 BF16 多步输出、梯度范数误差和更新检查。
- 八卡 FP32 小模型：四种策略各执行 3 步、每步累积 2 个 microbatch，与完整未分片参考逐参数比较梯度和更新。覆盖外部 loss 缩放、非均匀行分片及 activation checkpoint。
- 补齐 Qwen 输出 dataclass 的 pytree 注册，修复根 FSDP 未发现返回张量、反向未恢复参数导致的 CUDA 非法访问。
- 真实 BF16 大模型：全部变体完成真实 SGD 更新，主干和两张 Engram 表均观察到非零更新；cache 命中、失效和有界分配均有实际计数。
- **未达到 BF16 逐步确定性。** 两次相同 GPU 无 cache 配置的 global loss 最大差约 0.04608；相同 GPU cache 配置约 0.02703。跨变体差异处于这个波动量级。所抽查主干/投影元素最大差约 7.63e-6，所抽查表元素最大差约 6.62e-8；这不是全量大模型梯度等价的证明。
- 本轮复用原生 Qwen MoE 实现，没有改写 attention/MoE kernel，也没有用短程 loss 波动宣称训练质量改善。

## 结果与复现

紧凑汇总：[qwen30b-engram-sft-h20.json](results/qwen30b-engram-sft-h20.json)。完整每 rank 指标位于 `artifacts/qwen30b-engram/matrix/`。

每张 timeline 为独立 HTML，可离线打开；rank 0 的原始 Chrome trace 可交给 Perfetto 分析。

- [GPU 无 cache timeline](../artifacts/qwen30b-engram/matrix/repeat-2-gpu/trace/timeline.html)
- [GPU 有 cache timeline](../artifacts/qwen30b-engram/matrix/repeat-0-gpu_cache/trace/timeline.html)
- [CPU 无 cache timeline](../artifacts/qwen30b-engram/matrix/repeat-0-cpu/trace/timeline.html)
- [CPU 有 cache timeline](../artifacts/qwen30b-engram/matrix/repeat-0-cpu_cache/trace/timeline.html)

同目录的 `detail.html` 展示 Engram/训练 host 范围及各 GPU stream 的事件；`breakdown.json` 和 `rank-0-summary.json` 提供分项时间。

在仓库根目录、具备 PyTorch/CUDA/tokenizers/pyarrow/Plotly 的环境下：

```bash
PYTHONPATH=.:tests OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python tests/run_qwen_engram_benchmark.py \
  --model-path artifacts/qwen30b-engram/models/Qwen3-30B-A3B-Instruct-2507 \
  --data artifacts/qwen30b-engram/data/alpaca-2048.pt \
  --output artifacts/qwen30b-engram/rerun \
  --batch-size 4 --microbatches 2 --warmup 2 --steps 3 --repeats 2 --profile
```

本机实际解释器为 `/opt/python/3.10.20/bin/python3.10`，其 `PYTHONPATH` 还包含 `/data0/guorunfan.grf/envs/roll-vllm-017-py310-20260910-r5/lib/python3.10/site-packages`。
独立验证入口为 `tests/qwen_engram_distributed.py`；数据处理入口为 `tests/prepare_qwen_sft.py`；汇总入口为 `tests/summarize_qwen_engram.py`。
