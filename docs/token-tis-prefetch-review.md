# DeepSeek Token-TIS 预取限制、修复与合并审查

2026-10-28；对应 [PR #8](https://github.com/PACTHEMAN123/Tiny-DSV41/pull/8)。

结论：旧 main 已包含分布式 GRPO 和异步预取调度，但 Token-TIS 路径主动禁止预取。
这是一道正确性保护：它只能用当前 learner 参数计算 native-old 概率，没有保存滞后样本
生成时的参数。PR #8 补齐生成版本的 query-LoRA 分片快照、评分时切换和评分后恢复，
使这一组合在冻结基座及 Engram、KL=0、最多滞后一版的范围内可以运行。
生产改动限定在两个 RL 文件，未发现阻止合并的问题；它有额外的快照和参数切换开销。

**限制是怎样进入 main 的**

提交 `fb47e826` 在 2026-10-19 引入这条分布式 DeepSeek LoRA GRPO 路径，随后通过
PR #6 合入 main；本轮审查前的 main 为 `f37f20de`。当时同时存在两条路径：

| 路径 | 采样概率与训练概率的处理 | 预取限制 |
| --- | --- | --- |
| strict，`--rollout-correction none` | 初始 native/vLLM 概率差必须通过明确阈值；GRPO 直接使用实际 behavior 概率 | 调度器已有单组预取能力 |
| `decoupled-token-tis` | 把采样分布校正与 native 策略更新的 PPO 比率分开计算 | 入口强制 `--disable-prefetch --kl-beta 0` |

旧代码在配置检查中直接拒绝未关闭预取的 Token-TIS，并在 `train_group` 中要求全部
32 条回复的生成 ACK 等于当前 learner ACK。native-old 评分只是当前模型上的一次
`no_grad` 前向。因此，旧 main 没有放行这个不受支持的组合，也没有在这里静默使用错版
参数。待补齐的是“Token-TIS + 异步预取”的组合能力。

这里的预取是提前采样下一组 RL 回复，与 Engram 行缓存预取、DataLoader 预取无关。
这一入口限制也不是整个仓库、所有模型或所有 RL 路径的全局限制。

**为什么必须保留生成版本**

当前实现的 Token-TIS 使用三个逐 token 的 log probability，均对同一回复及其前缀评分：

| 名称 | 含义 | 是否参与梯度 |
| --- | --- | --- |
| `behavior` | vLLM 实际采样时记录的分布，含约定的采样温度 | 否 |
| `native_old` | native learner 用生成样本时的同一参数版本重新评分 | 否 |
| `current` | native learner 用当前待更新参数评分 | 是 |

仓库使用的目标可写成：

```text
w = stop_gradient(min(cap, exp(native_old - behavior)))
r = exp(current - native_old)
loss = -mean_responses(mean_tokens(w * min(r*A, clip(r, 1-eps, 1+eps)*A)))
```

`w` 处理实际采样分布与 native 生成版本评分之间的差异；`r` 衡量当前策略相对生成版本
的变化。`native_old` 是生成策略的 native 评分，不是 KL 项使用的固定参考模型。

同步模式下，一组样本的生成和训练之间没有 learner 更新，当前参数就能充当生成版本。
单组预取则会出现以下顺序，其中 `P_k` 表示已确认激活的第 k 版参数：

| 阶段 | learner | actor / 下一组样本 |
| --- | --- | --- |
| 预取开始 | 用 `P_k` 训练本组 | 同时用 `P_k` 采样下一组 B |
| 本组更新完成 | 参数变为 `P_{k+1}` | 等待 B 完成，再激活 `P_{k+1}` |
| 训练 B | current 应用 `P_{k+1}` | B 的 behavior 仍属于 `P_k`，native-old 也必须用 `P_k` |

若放开预取并放宽旧的当前版本校验，却仍用 `P_{k+1}` 计算 B 的 native-old，会把策略滞后错误地并入
采样分布校正权重，并使评分时的 PPO 比率在理论上退化为 1。这不意味着梯度一定为零；
它意味着计算的目标不再是上面定义的、以生成策略为参照的目标。因此不能仅打开开关。

**PR #8 怎样补齐这条路径**

1. actor 成功激活并返回完整 ACK 后，每个 learner rank 保存 query-LoRA 本地参数
   分片的独立副本，绑定版本、lineage 和摘要；仅保留当前及前一版。
2. 下一组采样在当前训练期间进行。调度器等待该采样请求结束，才暂停接纳并发布新权重，
   避免同组样本在生成中途换版。零优势组被跳过时不推进优化器或策略版本。
3. 更新前检查全部回复的完整生成 ACK。同步要求 lag=0；异步允许 lag=0 或 1。
   混合版本、错误 lineage、缺失旧快照、未来版本和超过一版的滞后均被拒绝。
4. lag=1 时，先显式 reshard FSDP 模块，再把生成版本分片复制到原参数对象中，完成
   无梯度的 native-old 评分；通过 `finally` 恢复当前分片，随后才执行 current 前向和反向。
   各物理微批完成反向后才进行下一次参数切换，累计梯度及 AdamW 状态保持原位。
5. strict 分支修正滞后指标并增加身份检查，其损失公式和优化器计算不变。

FSDP 下不能只对本地分片执行 `copy_`：已经物化的参数可能仍在 all-gather 缓存中。
这里通过 PyTorch 的 `FSDPModule.reshard()` 使后续评分重新读取对应分片。36 行的
参数切换 helper 集中处理布局校验、缓存失效和异常恢复，没有建立第二套模型加载流程。

**简洁性、影响范围与回归审查**

运行时代码仅修改 `distributed_grpo.py` 和 `native_lora.py`，合计增加 85 行、删除 14 行。
没有新增运行依赖、通用队列框架、模型副本或训练诊断探针。SFT、Engram 实现和模型算子
没有改动；Token-TIS 损失公式也没有改动。其余变更是必要回归测试、说明文档，以及修正
已有测试 fixture 对当前接口的使用，未降低原有断言。

预期影响需要保留：Token-TIS 的同步模式也会维护两版 LoRA 分片快照；lag=1 的每个
物理微批会增加当前分片备份、恢复及 reshard。复制范围仅是可训练 adapter，基座和
Engram 必须冻结。没有把这些开销描述为零，也没有声称能支持全参数异步 RL、任意滞后、
分布式故障续跑或可恢复 checkpoint。非法身份被更早拒绝属于预期行为。

本次两份独立代码审查均未发现阻塞问题。合并前相关 CPU 回归收集 66 项，65 项通过，
1 项因缺少可选 Harbor 依赖跳过；跳过项涉及测试 fixture，本次未修改 Harbor 生产逻辑。
该 fixture 随后使用仓库已有的 Harbor 接口 stub 单独执行并通过全部原始断言；这属于
离线 transport 契约检查，不是 Harbor job、sandbox 或 agent 的集成验证。
此前完成的双卡 FSDP 回归在 FP32/BF16 下通过。完整 40 层 DeepSeek-V4.1-Flash-1017
的 16-rank 正确性测试完成 4 次更新，其中 2 次使用 lag=1 样本：32 次旧评分对照、
32 次梯度重放及 64 次 AdamW 参数/状态对照最大绝对差均为 0；独立 loss 汇总最大绝对差
为 `1.117587e-8`。这些检查复用 native backbone/autograd，验证本实现的目标和更新，
不代表独立后端或所有配置都已证明等价。

**实际并行与性能结果**

2026-10-27 的测试使用本次 32 张 NVIDIA L20C 集群中的 16 卡 learner 和独立 8 卡
vLLM actor。配置为完整 40 层模型、query-LoRA r8/alpha8、G32、物理微批 4、context
2048、Token-TIS cap=2、KL=0。两组各完成 20 次有效更新，排除前 2 次预热和末次排空，
统一统计中间 17 次更新；正式计时不含独立 CUDA profiler。

| 指标 | 同步 | 异步 |
| --- | ---: | ---: |
| 17 次更新耗时 / 秒 | 4358.512 | 4139.054 |
| 平均单步 / 秒 | 256.383 | 243.474 |
| P90 单步 / 秒 | 355.501 | 384.342 |
| 实际训练响应 token 数 | 424941 | 399096 |
| 训练响应 token/s | 97.497 | 96.422 |
| host 采样 RPC 与训练区间交叠 / 秒 | 0 | 215.212 |

异步平均单步少 5.035%，训练响应 token 也少 6.082%，长度分布、padding、跳过组和预取
库存存在差异。两组在首次更新前就生成了不同长度的回复，所以不能把全部耗时差归因于
异步调度。本轮没有测到响应 token 吞吐提升；一次完整对照也不足以判断约 1.1% 的吞吐
差异具有统计显著性。当前同步 baseline 本身也进行独立 native-old 前向，结果不外推到
将来复用 fresh logits 的优化同步实现。

另一次独立 CUDA 诊断通过原始 trace 和请求身份核对，确认采样 GEMM 与 learner
首个 native-old 评分前向存在 GPU 执行交叠。独立复核在共同单调时钟、每层 1000 ppm
漂移及每边 1 us 时间误差的明示假设下，覆盖完整 214.294 ms 相对时钟偏移域。
得到的 998 ns 是交叠存在性的保守下界，不是总 GPU 重叠时长；上述 215.212 秒也只是
host 区间交叠。该证据范围没有覆盖 current 前向、反向或全部 GPU 的持续并行。

Token-TIS 的上截断及未校正的前缀分布仍会引入偏差；本次没有放宽 strict 概率门禁，
也没有把数值对照通过写成 native/vLLM 概率等价、无偏估计或模型质量提升。

代码溯源与证据入口：

- [旧 main 的配置保护和评分路径](https://github.com/PACTHEMAN123/Tiny-DSV41/blob/f37f20de05f8337d770e86c5265f5ce4d0cbf330/dsv41_train/rl/distributed_grpo.py#L171)
- [引入这条路径的提交 fb47e826](https://github.com/PACTHEMAN123/Tiny-DSV41/commit/fb47e826a4adaad84118b217d43a8e8d48b2b3dd)
- [Token-TIS 目标定义](https://github.com/PACTHEMAN123/Tiny-DSV41/blob/730e26c9cd49f172a663550c84d91e432c289198/dsv41_train/rl/algorithm.py#L249)
- [生成版本校验及评分修复](https://github.com/PACTHEMAN123/Tiny-DSV41/blob/730e26c9cd49f172a663550c84d91e432c289198/dsv41_train/rl/distributed_grpo.py#L221)
- [FSDP 参数切换与恢复](https://github.com/PACTHEMAN123/Tiny-DSV41/blob/730e26c9cd49f172a663550c84d91e432c289198/dsv41_train/rl/native_lora.py#L221)
- [运行说明](distributed-deepseek-grpo.md)及 [PR #8 的验证摘要](https://github.com/PACTHEMAN123/Tiny-DSV41/pull/8)。原始日志、trace、哈希和独立审核保存在本任务的 `async-rl-validation-20261027` 本地归档，未作为运行时代码提交。
