# 新 L20D 机器 post-train 验证记录（2026-10-14）

> 发布说明：此页记录历史实验。模型权重、adapter、原始日志和 `artifacts/` 证据保留在实验环境，不随 Git 仓库发布；下文标注“本地证据”的路径不是仓库下载链接。通用训练用法见 [SFT 指南](sft.md)。

本报告按已经下载、校验并审查的证据填写。训练数学结果、子进程退出和 GPU 回收分别验收。当前证据完整覆盖 Qwen 两阶段、DeepSeek prefix2 的首次验证/恢复/重试，原生 Engram 六组矩阵，以及 full40 的 8 卡 LoRA 执行；**原始 gate 0019 是 FAIL，必须保留，不能由后续恢复追认成 PASS。** 后续结果在本文追加，未经证据确认的阶段不计入成功。

| 范围 | 已确认结果 | 验收边界 |
| --- | --- | --- |
| 完整 Qwen3-30B-A3B + 新增随机小 Engram，SGD 五种存储策略 | 五组训练及数值比较 PASS；gate 0003–0007 PASS | 是完整 Qwen 骨干上的短程训练；Engram 为新加的小表，不是 DeepSeek 原生大表 |
| 完整 Qwen3-30B-A3B，真实 LoRA SFT | 连续、断点恢复、完整 adapter 比较、独立重载 PASS；gate 0008–0013 PASS | Dolly 子集、seq128、5 步；基座冻结值检查是抽样 |
| DeepSeek-V4.1-Flash prefix2（第 0、1 层），真实 LoRA SFT | 经显式恢复及独立重试后 PASS；**原始 0019 GPU 回收仍为 FAIL** | gate 0014–0018、0020、0021 PASS；原生第 1 层 Engram 已加载但冻结 |
| DeepSeek 全 40 层 LoRA | 六阶段及独立审计 PASS；gate 0028–0033 PASS | 真实 8 卡/40 层 LoRA，3 步与 2+resume3；不等于全参数训练或质量验收 |
| DeepSeek 原生大 Engram 的 SGD / CPU offload / async / cache 矩阵 | 六组功能、比较及 gate 0022–0027 PASS | prefix2 单张原生大表；短测显存/流量效果已确认，不代表 full40 或生产吞吐收益 |

## 证据目录与环境

所有路径相对仓库根目录；下列链接可从本文直接打开。

- phase1 原始证据（本地证据：`artifacts/new-l20d-20261013/remote-phase1-evidence/artifacts/new-l20d-20261013/`） 与 独立审计（本地证据：`artifacts/new-l20d-20261013/phase1-independent-audit.json`）。
- Qwen LoRA 原始证据（本地证据：`artifacts/new-l20d-20261013/remote-qwen-lora-evidence/artifacts/new-l20d-20261013/`） 与 独立审计（本地证据：`artifacts/new-l20d-20261013/qwen-lora-independent-audit.json`）。
- DeepSeek prefix2 原始证据（本地证据：`artifacts/new-l20d-20261013/remote-dsv4-prefix2-evidence/artifacts/new-l20d-20261013/`） 与 独立审计（本地证据：`artifacts/new-l20d-20261013/dsv4-prefix2-independent-audit.json`）。
- prefix2 恢复/重试原始证据（本地证据：`artifacts/new-l20d-20261013/remote-prefix2-recovery-evidence/artifacts/new-l20d-20261013/`） 与 恢复链独立审计（本地证据：`artifacts/new-l20d-20261013/prefix2-recovery-independent-audit.json`）。
- native Engram 原始证据（本地证据：`artifacts/new-l20d-20261013/remote-native-engram-evidence/artifacts/new-l20d-20261013/`） 与 六组独立审计（本地证据：`artifacts/new-l20d-20261013/native-engram-independent-audit.json`）。
- full40 原始证据（本地证据：`artifacts/new-l20d-20261013/remote-full40-evidence/artifacts/new-l20d-20261013/`） 与 独立审计（本地证据：`artifacts/new-l20d-20261013/full40-independent-audit.json`）。
- 验收要求清单（本地证据：`artifacts/new-l20d-20261013/acceptance-checklist.json`） 是要求，不代表所有项目已完成。

远端运行版本为 Python 3.10.9、PyTorch 2.11.0+cu130。GPU gate 固定监测全部 8 张物理 GPU，具体两卡训练通过 UUID 选择其中 0、1；测试串行共享同一个本地 state 目录和排他锁。每卡报告总显存 275040 MiB，原 gate baseline 的 used 为 `[4,4,2,1,1,1,1,1]` MiB；GPU UUID、PCI 地址、容量、宿主启动标识、UID 和 PID namespace 均用于身份检查。

原 gate policy 固定为：相对原 baseline 的 used/free 容差 256 MiB/卡，utilization 不超过 5%，free 不低于 1024 MiB，初始 used 不超过 1024 MiB；必须获得两次连续合格快照。NVML 进程列表为空不能代替显存检查。PASS 表示回到原基线容差内，不等于每卡必须恰好为 0 MiB，也不证明宿主机没有容器不可见进程。

## 完整 Qwen SGD 与随机 Engram 五组矩阵

挂载模型为 `/mnt/fuse/Qwen/Qwen3-30B-A3B`，使用全部 48 层骨干，以 SGD 进行前向、反向和更新。第 1、24 层新增随机 Engram，共 13,221,376 个参数；配置为 vocab/head 4096、head_dim 64、8 heads、max_ngram 4、seed 20261013。EP2、CP1，seq128，每 rank batch1、2 个 microbatch，全局 batch4；每组 2 步 warmup + 5 步计量。

数据为 8 条人工 smoke 样本打包/重复得到的 64 个 pack，不是 Dolly 或质量评估集。五组为 GPU、CPU sync、CPU async、CPU cache、GPU cache。自动比较原始结果（本地证据：`artifacts/new-l20d-20261013/remote-phase1-evidence/artifacts/new-l20d-20261013/qwen-engram/comparison.json`） 已在本地按相同规则重新计算并一致；各组训练摘要与 rank 文件匹配。

| 策略 | 记录的 tokens/s | 验证结果 |
| --- | ---: | --- |
| GPU | 74.4235 | 参考组 PASS |
| CPU sync | 74.1355 | 与参考组在预设容差内 |
| CPU async | 75.8436 | 与参考组在预设容差内 |
| CPU cache | 75.5695 | 与参考组在预设容差内 |
| GPU cache | 75.2073 | 与参考组数值一致 |

三个 CPU 变体相对 GPU 的最大 loss 差为 0.00418714，最终 6144 个参数抽样值的最大差为 0.0000610352；都在原比较容差内。表行更新、梯度有限性和存储/传输指标通过；这些是运行检查和抽样参数比较，不是全模型权重逐元素等价证明。

cache 记录合计 47,876 hits、61,712 misses（约 43.69% 命中）、3,553 次 eviction、35,366 行 invalidation，实际 cache 分配合计 4,194,304 字节。CPU cache 的 value H2D 为 5,466,624 字节，另有 metadata H2D 2,376,112 字节；无 cache CPU 组 value H2D 为 14,027,264 字节。D2H 仍为 14,027,264 字节。GPU 组 peak 约 59.144 GiB，CPU sync/async 约 59.131 GiB；这里只证明小表路径可用及该 workload 的传输变化，无法由约 13 MiB 的峰值差或单次短计量推断原生大表的省显存/加速比例。

gate 0001、0002 的 tiny 测试和 0003–0007 的真实 Qwen 矩阵均退出 0、无记录的残留子进程，并满足连续快照回收要求。tiny 结果不扩大真实模型覆盖范围。

## 完整 Qwen 的真实 LoRA SFT

真实 48 层模型，EP2/FSDP、CP1、BF16、完整 activation checkpointing；LoRA rank8、alpha16、dropout0.1，AdamW lr0.001、weight_decay0，compact LoRA DCP。seq128、每 rank batch1、梯度累积2。数据源是 Dolly 256 条短提示子集的 `plain-completion-v1`，5 步实际处理 20 条不同样本、1299 个监督 token、1536 个输入 token。

连续 5 步与 3 步保存后新进程恢复至第 5 步完全一致：384 个 A/B 张量、6,684,672 个元素全部逐元素比较，最大差为 0；全部 step loss 和 token count 一致。此完整 adapter 比较已在本地重新读取可信任务产物并复核。独立进程从挂载基座重建并加载 adapter 后，与训练进程预先保存的 fixed probe 相比，每 rank 256 个抽样 logits 和标量 loss 均 exact 0 差异。完整轨迹比较（本地证据：`artifacts/new-l20d-20261013/remote-qwen-lora-evidence/artifacts/new-l20d-20261013/lora-qwen-dolly128/resume-parity.json`），重载比较（本地证据：`artifacts/new-l20d-20261013/remote-qwen-lora-evidence/artifacts/new-l20d-20261013/lora-qwen-dolly128/adapter-reload.json`）。

运行时检查所有基座参数 `requires_grad=False`、`grad=None`，optimizer 只且完整包含 A/B；每 rank 531 个冻结基座张量的全局合计 235,008 个抽样值不变。A/B 梯度有限，更新检查通过；零初始化 B 导致第一步 A 梯度为 0 属于预期。基座数值检查不是全量 checksum。

连续训练的 load peak 29.584 GiB、train peak 29.736 GiB、reserved peak 30.346 GiB，均为全部 rank 的最大值。固定 probe loss 从 9.868655 降至 4.335140；这是训练行 0 的固定检查，两个 rank 使用同一行、3 个 shifted targets，不是 held-out 验证分数。六个 gate 0008–0013 均 PASS，后续进程身份和显存快照复核通过。

此阶段本地证据未包含完整 DCP tensor/optimizer payload；精确继续训练提供功能性恢复证据，不能改写成“已独立逐元素审计全部 optimizer state”。

## DeepSeek prefix2：经显式恢复及独立重试后通过

范围为真实挂载 `/mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash` 的第 0、1 层文本骨干，包含原生第 1 层 Engram。EP2、CP1、BF16、layerwise 加载、seq128、LoRA rank8/alpha16/dropout0.1、AdamW，以及 compact LoRA DCP。目标投影为 `attention.q_a/q_b/kv_proj/o_b`。原生 Engram 在 CUDA 上加载、冻结；其全局参数量为 98,305,579,008，BF16 全表约 183.108 GiB，EP2 每 rank 约 91.554 GiB；**本次没有表梯度、表更新或 CPU offload/cache 测试**。

独立审计确认连续 5 步与 3+resume 至 5 步的 16 个 A/B 张量、950,272 个元素、全部 step loss/count 精确一致。原首次 fresh reload 自身 `validation_passed=true`、command exit0，每 rank 256 个抽样 logits 及 scalar loss 与训练参考值 exact 0 差异。固定训练行 probe loss 为 15.489479 → 11.191673，不作模型质量结论。

基座每 rank 815 个张量保持冻结/无梯度，全局 412,396 个抽样值不变；optimizer 精确归属 A/B。连续 run 的 load peak 210.110 GiB、train peak 211.371 GiB、reserved peak 224.479 GiB，均为全部 rank 的最大值。5 步实际监督 token 共 1289，输入 token 共 1525。

compact DCP metadata 独立检查包括 16 个 adapter payload、2 个持久 router `selection_bias` buffer、48 个 optimizer state key，以及各 rank CPU/CUDA RNG 和数据生成器状态。未出现冻结基座权重 payload。连续 step5 的 metadata 加引用 shard extent 为 28,949,095 字节（约 28.95 MB）；首次审计由 metadata 推导，后续完成后的远端 filesystem stat 也确认该大小（见 full40 节的 checkpoint 表），仍不是 payload 全量 checksum。

然而，**原始 gate `0019-lora-dsv4-prefix2-dolly128-reload` 的 result.passed=false，memory_recovered=false**，错误为 `insufficient consecutive stable snapshots`。首次验证独立审计的三个历史结论必须同时保留：`functional_passed=true`、`resource_release_within_deadline=false`、`overall_validation_passed=false`。

| 0019 after 快照 | 各卡 used MiB（GPU 0–7） | 原阈值检查 |
| --- | --- | --- |
| 000 | 1, 1, 1, 627, 4, 4, 4, 4 | FAIL |
| 001 | 1, 1, 1, 1, 627, 1, 4, 4 | FAIL |
| 002 | 1, 1, 1, 1, 4, 4, 4, 4 | clean，但仅连续 1 次 |
| 003 | 1, 627, 1, 1, 1, 3, 4, 4 | FAIL |
| 004 | 1, 1, 627, 1, 4, 4, 4, 4 | FAIL |
| 005 | 1, 1, 1, 1, 1, 1, 1, 1 | clean，但仅连续 1 次 |

完整快照查询约需 10.8 秒；原 60 秒观察窗口没有两次连续 clean，所以 gate 按规则失败。原报告确认命令、supervisor 及其子进程清理成功；后续单次快照也已 clean，旧 PID 身份不再存在。这些都不能追认原 gate 为通过。627 MiB 短暂占用也出现在本次未使用的 GPU 2/3/4，NVML 列表为空；归属未知，不能归因于该训练、终端、驱动泄漏或某个宿主进程。此前 phase1 等待快照亦出现过短暂约 630 MiB 占用。

原 0019 result（本地证据：`artifacts/new-l20d-20261013/remote-dsv4-prefix2-evidence/artifacts/new-l20d-20261013/gpu-gate/0019-lora-dsv4-prefix2-dolly128-reload/result.json`） 与 原 failed state（本地证据：`artifacts/new-l20d-20261013/remote-dsv4-prefix2-evidence/artifacts/new-l20d-20261013/gpu-gate/state.json`） 保持不变。原 state SHA256 为 `7a67f63c0ac64855c611d8e8bcea75a388a4504eea6f6d195e878950891fb395`；原 result SHA256 为 `d9e78fd514353cb868a2f4b9924b3c7fe400e53a381b686041710ef1904a85a5`。

恢复单独记账：显式恢复脚本（本地证据：`artifacts/new-l20d-20261013/recover-gpu-gate.py`） 已通过 21 项 CPU 测试及独立审查，在同一排他锁内核验原 state SHA、原 baseline/policy、机器和旧 PID 身份，仅允许两次新 clean 快照后写入新编号恢复 case；不重跑旧命令、不重建 baseline、不修改阈值、不发进程信号。

0020 的真实恢复结果已经下载并独立核验为 PASS。原 state/result/process/invocation 及初始 baseline 副本逐字节匹配旧证据，baseline/policy 哈希不变；旧 supervisor 与 3 个 child 的 PID/start_ticks/UID 身份在恢复前后均不存在。两次新 clean 快照间隔 9.964339 秒，采用原 256 MiB 等阈值、固定 2 秒轮询及明确的 180 秒观察窗口。继承的 `process-result.json` 与原命令成功退出报告逐字节一致，并有独立 provenance，未伪造新 supervisor。

0021 独立 fresh reload retry1 也已核验为 PASS，命令仅修改报告输出路径，其余模型、adapter、reference、GPU UUID 和 zero-tolerance 参数与首次运行一致。子命令和 supervisor 均 exit0，无 signals/remaining/unresolved children；每 rank 256 个抽样 logits 和 scalar loss 与原训练参考 exact 0 差异。前、后各两次连续快照均符合原 policy，after 快照未发现本次 owned PID 身份。最终两次各卡 used 为 `[4,4,4,4,1,1,1,1]` 和 `[4,4,4,4,4,2,1,1]` MiB，完成于 2026-10-13T16:30:54.926724Z。

远端恢复后汇总（本地证据：`artifacts/new-l20d-20261013/remote-prefix2-recovery-evidence/artifacts/new-l20d-20261013/lora-dsv4-prefix2-dolly128/validation-summary-after-recovery.json`） 与 独立审计（本地证据：`artifacts/new-l20d-20261013/prefix2-recovery-independent-audit.json`） 均 PASS；35 项下载 manifest 和汇总引用的 79 个可用历史文件哈希匹配。阶段结论因此更新为 **prefix2 经显式恢复和独立重试后通过**，链路仍为“0019 FAIL → 0020 显式恢复 PASS → 0021 独立重试 PASS”。汇总时当前 gate 已是后续 native Engram case0023、状态 running；这是单次观察，既不改变历史 prefix2 结论，也不表示机器当前空闲或后续矩阵已通过。

## 原生 DeepSeek Engram：六组 SGD / offload / cache 矩阵通过

本阶段独立训练真实 prefix2 第 0、1 层及原生第 1 层 Engram，使用 SGD lr0.01、EP2/CP1、seq128、batch1/rank、microbatches2、full activation checkpointing。与前面的 LoRA 阶段不同，此处原生表参与稀疏梯度和更新。表形状为 384,006,168 × 256，BF16 全局 183.108 GiB，EP2 每 rank 91.554 GiB；CPU 策略将两份行分片分别放在宿主内存，合计仍是一张全局表。加载使用 direct_shard、pipeline、131072 chunk rows、128 MiB staging。数据为同一 Dolly 256 条子集，2 步 warmup + 5 步计量；没有写训练 checkpoint。

六组矩阵汇总（本地证据：`artifacts/new-l20d-20261013/remote-native-engram-evidence/artifacts/new-l20d-20261013/dsv4-engram-dolly/matrix-summary.json`） 已由 独立审计（本地证据：`artifacts/new-l20d-20261013/native-engram-independent-audit.json`） 重新计算。functional、resource release、validation 均 PASS；129 个原始文件未改变、128 项 manifest 与下载包 SHA 校验通过。gate 0022–0027 保持原 8 张 UUID 和原 policy，命令 exit0、无残留 owned 身份，前后末两次连续快照都合格。

下表 peak 为 warmup 后重置计数器、五个计量步骤中跨 rank 的最大 CUDA allocated 值；input tokens/s 使用同一计量区间，包含校验和显式同步，排除加载和 gate 等待。

| 策略 | GPU peak allocated（GiB） | 较 GPU 参考减少（GiB/rank） | input tokens/s | 六组验收 |
| --- | ---: | ---: | ---: | --- |
| GPU | 151.542275 | 0 | 387.554990 | PASS |
| CPU sync | 59.986269 | 91.556005 | 364.373917 | PASS |
| CPU async | 59.986269 | 91.556005 | 417.305363 | PASS |
| CPU async + cache 8 MiB/rank | 59.994082 | 91.548193 | 407.019067 | PASS |
| CPU async + cache 128 MiB/rank | 60.111269 | 91.431005 | 409.852129 | PASS |
| CPU async + cache 1 MiB/rank（压力） | 59.987246 | 91.555029 | 391.226009 | PASS |

**本 workload 中，原生表移到 CPU 将 GPU 计量峰值降低约 91.56 GiB/rank。** 这与该 rank 的约 91.55 GiB 表分片规模相符。代价是宿主内存和传输：两 rank 各自历史 peak RSS 相加在 CPU 组约 296.81–297.07 GiB，GPU 组约 113.77 GiB；RSS 包含加载时文件映射页，这些和数不是同时采样的节点独占物理内存，不能当作实际瞬时 host peak。

全部五个 CPU 策略相对 GPU 参考的 7 步 loss、最终 dense/table 有界样本及更新摘要最大差均为 0。梯度 norm/maxabs 摘要最大差为 0.0000032633543，在原容差内；稀疏 COO raw_nnz 最大差 37、合并后 nonzero 值计数最大差 2，coalesced unique_rows 一致。不能将结果表述为全梯度/全权重逐元素 exact；检查覆盖选定稠密样本及触达稀疏行，最终表值最多采样 32 行 × 8 列。

cache 的统计覆盖五个计量步骤、两 rank 合计 61,440 个 lookup rows；预算按每张表、每 rank 计算。下表 H2D 均为字节：

| CPU 策略 | cache hits / lookup（命中率） | eviction / invalidation 行计数 | value H2D | metadata H2D | 合计 H2D | 较无 cache 减少 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| sync / async 无 cache | 不适用 | 0 / 0 | 31,457,280 | 0 | 31,457,280 | 0% |
| cache 8 MiB | 16,201 / 61,440（26.3688%） | 1,280 / 26,517 | 14,974,464 | 1,427,792 | 16,402,256 | 47.8586% |
| cache 128 MiB | 16,546 / 61,440（26.9303%） | 90 / 29,060 | 14,967,808 | 1,449,440 | 16,417,248 | 47.8110% |
| cache 1 MiB | 12,779 / 61,440（20.7992%） | 5,344 / 14,895 | 15,028,736 | 1,306,864 | 16,335,600 | 48.0705% |

三个 cache 组的 gradient-value D2H 都仍为 31,457,280 字节，所以将该 D2H 加回后，跟踪的双向流量减少约 23.91%–24.04%，不是 48%。这些计数遗漏 owner-ID/control 传输和卡间 collective，不是完整 PCIe/NVLink 流量分析。cache 内部还会对 missing IDs 执行去重，无 cache 对照使用 `--no-deduplicate`；因此通信变化同时包含缓存与内部 miss 去重，不能归因于命中率本身。

更大 cache 在本次提高了命中率并减少 eviction，但 value+metadata H2D 和吞吐并不单调；1 MiB 的合计 H2D 反而略低。三个 cache 组均慢于无 cache 的 CPU async，**本轮没有证明 cache 吞吐加速**。每组仅五个计量步骤、固定执行次序，没有重复实验或置信区间，并包含正确性检查与同步；这里只记录短测观察，不能据此选择生产最优 cache 或声称稳定加速。该结果也不覆盖第 14 层原生表、压缩 attention、长序列或全部 40 层。

## 全 40 层：真实 8 卡 LoRA 训练、恢复和重载通过

全 40 层文本骨干已完成真实 8 卡 LoRA 运行，原始 validation summary（本地证据：`artifacts/new-l20d-20261013/remote-full40-evidence/artifacts/new-l20d-20261013/lora-dsv4-full-dolly128/validation-summary.json`） 和六个 gate 0028–0033 均记录 PASS。最终独立审计（本地证据：`artifacts/new-l20d-20261013/full40-independent-audit.json`） 已确认 functional、resource release、validation 全部 PASS；320 个 A/B 张量在本地重新进行有限值与逐元素相等检查，六个 gate 的原 policy/8 UUID、前后连续两次 clean 和进程退出/无残留身份均重新核验。158 个下载文件未改变、157 项 manifest 已校验，下载包 SHA256 为 `a05c4c8a235daaac43da3476db66b51d9a8cc7faba010f9f96a50ac0231db978`。

运行包含第 0–39 层文本骨干、两张原生 Engram、embedding/head，以及短序列下实际触发的压缩 attention/KV/index 路径；不含 vision、额外 3 个 next-token prediction（NTP）模块或未实现的 DSpark。这里的“完整”指当前实现的 40 层文本骨干；训练对象仍然只是 LoRA A/B，基座、专家及 Engram 保持冻结。

配置为 EP8、CP1、dense FSDP8，BF16，layerwise 加载，LoRA rank8/alpha16/dropout0.1，目标 `attention.q_a/q_b/kv_proj/o_b`，AdamW lr0.001、weight_decay0，full activation checkpointing，compact LoRA DCP；seq128、每 rank batch1、梯度累积2。Dolly 256 条源数据、tokenizer 和嵌套 config 的身份与 prefix2 一致，BOS/EOS/PAD 为 0/1/2。

连续 3 步与 2 步保存后恢复至第 3 步的 完整 adapter/轨迹比较（本地证据：`artifacts/new-l20d-20261013/remote-full40-evidence/artifacts/new-l20d-20261013/lora-dsv4-full-dolly128/resume-parity.json`） 记录 320 个 A/B 张量、19,005,440 个元素逐元素 exact 0 差异，全部 3 步 loss 和 token count 一致。独立进程 adapter 重载（本地证据：`artifacts/new-l20d-20261013/remote-full40-evidence/artifacts/new-l20d-20261013/lora-dsv4-full-dolly128/adapter-reload.json`） 在全部 8 个 rank 上与训练进程参考的固定 probe 相符。每 rank 256 个抽样 logits 和标量 loss 不是所有输出 logits 的全量比较。

运行时检查每 rank 4686 个基座参数张量全部冻结且无梯度，global 合计 9,081,968 个抽样值不变；optimizer 只且完整包含 A/B，梯度有限和参数更新检查通过。原生表和专家没有梯度/optimizer state；本阶段没有验证其全参数更新。固定训练行的 loss 为 5.885386 → 1.381917，所有 rank 使用同一行，不能作为 8 份独立样本或 held-out 质量分数。

| 连续 3 步记录 | 数值 | 统计口径 |
| --- | ---: | --- |
| load peak allocated | 197.175475 GiB | 全部 rank 的最大值 |
| train peak allocated | 198.459105 GiB | 全部 rank 的最大值 |
| peak reserved | 224.964844 GiB | 全部 rank 的最大值 |
| 模型加载耗时 | 326.620487 秒 | rank0，非全部 rank 的独立计时 |
| train_seconds | 32.254696 秒 | trainer 记录的短程训练区间，含运行检查 |
| 实际监督 / 输入 tokens | 2753 / 3283 | 连续 3 步累计 |
| 独立 save 时长 | 未测量 | 不能由文件 mtime 或整阶段 wall time 反推 |

compact checkpoint 的真实文件大小已由 写入完成后的 filesystem stat（本地证据：`artifacts/new-l20d-20261013/remote-full40-evidence/artifacts/new-l20d-20261013/checkpoint-storage-audit.json`） 记录：

| 连续 run 最后一个 checkpoint | 总字节（含 metadata） | metadata 字节 |
| --- | ---: | ---: |
| Qwen step5 | 70,855,649 | 5,288,818 |
| DeepSeek prefix2 step5 | 28,949,095 | 3,986,737 |
| DeepSeek full40 step3 | 679,141,844 | 96,649,478 |

full40 的两个 resumed checkpoint 大小分别为 679,141,841 和 679,141,843 字节。三份 `.metadata` 已分别由独立进程读取并释放；model state 的 360 个 key 恰为 320 个 A/B 与 40 个 FP32 `[384]` router `selection_bias` buffer，没有冻结基座权重 payload。每份 8 个 shard 的引用 extent 与远端 filesystem stat 逐一匹配。整个 checkpoint 还包含 optimizer、RNG、data、layout 等状态，metadata/序列化开销使目录大小明显大于纯 adapter 权重的 38,010,880 字节；不能称为“checkpoint 只含 A/B”。实际 DCP optimizer payload 保留在远端，本地没有完成其全部数值逐元素审计；恢复训练 exact 轨迹和 adapter 对比提供功能证据。full40 也没有保存约 1.497 TB 的冻结基座权重。

## 全 40 层：启动前容量与格式预检记录

以下容量预算是启动前的 CPU/meta 预检，与上方实际运行指标分开保存，不以实测结果重写历史估算。

真实挂载 config 的容量预估（本地证据：`artifacts/new-l20d-20261013/remote-dsv4-prefix2-evidence/artifacts/new-l20d-20261013/full40-capacity-mounted.json`） 使用 CPU/meta 构建，`cuda_initialized=false`、`gpu_training_executed=false`。读取嵌套 `text_config` 并保留外层 special-token fallback；当前解析 BOS=0、EOS=1、PAD=2。范围为完整 40 层文本骨干、原生 embedding/head/Engram，**不含 vision、额外 next-token prediction（NTP）或未实现 DSpark 模块**。

| 项目 | 数值/解释 |
| --- | --- |
| 基座物理参数 | 748,494,654,064；不等于每 token 激活参数量 |
| 其中 routed experts | 543,581,798,400 |
| 其中两张 Engram 表 | 196,613,849,600 |
| 其余基座参数 | 8,299,006,064 |
| 当前加载 dtype 的基座权重 | 1,497,067,960,768 字节，约 1394.253 GiB |
| 假设拓扑 | 8 ranks，EP8、CP1、dense FSDP8、expert FSDP1、Engram row-shard8/FSDP1 |
| 估算每 rank resident 参数与 buffer | 约 174.287 GiB |
| 估算 layerwise loader peak | 约 220.065 GiB |
| 估算训练结构预算 | 约 227.582 GiB/rank |
| 另加未实测 runtime allowance | 32 GiB/rank |
| 合计预算与最小余量 | 约 259.582 GiB/rank，假定可用显存下余量约 8.108 GiB/rank |

该估算已经计入冻结 Engram 在 FSDP size1 下的局部表复制、专家组/稠密组临时存储、adapter gradient/optimizer 预算。输入的可用显存合计 2141.516 GiB 是调用者提供的观测值，不是脚本执行时实时确认的空闲量。32 GiB allowance 是假设，不能严格界定 CUDA allocator、库 workspace、通信和 activation 峰值。`estimated_budget_fits_all_ranks=true` 仅说明所列假设下的估算可容纳，本身不能证明加载、训练或恢复成功；实际运行结果另列于上一节。完整基座 DCP 仅权重就约 1.497 TB，不能写入约 508 GiB 工作盘；本次实际采用 compact LoRA state。

压缩 attention metadata 预检（本地证据：`artifacts/new-l20d-20261013/remote-dsv4-prefix2-evidence/artifacts/new-l20d-20261013/dsv4-compressed-metadata.json`） 检查全部 8 个源层（2、8、14、20、24、28、32、36）的 43 项 key/shape/dtype/scale/extent，全部 PASS；读取 header 2,084,208 字节、tensor payload 0 字节。它证明 header/index 与 loader 格式契约匹配，不证明量化值解码、压缩 attention 数值、HF 输出等价或 GPU 执行成功。

## 结论适用范围

- LoRA 使用 Dolly 短提示子集、seq128；Qwen 和 prefix2 各连续 5 个 optimizer step，full40 连续 3 步。tokenizer preflight 已验证监督 token 存在；没有 held-out loss、生成质量、长期收敛或业务任务指标。固定 probe 来自训练行 0，不能称为评估集验证。
- prefix2 的压缩比是 `[0,0]`，没有运行后续压缩 attention、KV/index 复用、candidate selector 或第 14 层 Engram。本轮 full40 seq128 的短程运行也没有压力覆盖 index_topk512、candidate_topk_blocks2048 或长上下文；`cache_quantization=none` 不覆盖 HF activation-cache 量化。
- 完整 adapter 张量比较与基座抽样冻结检查须区分。config/index/tokenizer 有身份 hash；尚未对全部基座 shard 内容做 hash，挂载基座必须保持不变。
- compact DCP 保存持久 buffer；单独 adapter 导出以 A/B 为主。当前 router `selection_bias` 从基座重建且当前训练路径不更新它。未来若训练会修改持久 buffer，必须同时保存/重载或证明不变，不能沿用现有 adapter-only 成功结论。
- LoRA 冻结原生 Engram；它和 Qwen 随机小表结果均不能替代原生表验收。本文另列的真实 native Engram 六组矩阵已覆盖 prefix2 第 1 层表的 SGD 更新、CPU offload、async、cache 失效/压力及短测效果，仍不代表 full40 或生产负载。
- 后续可扩展：更长序列、重复性能试验和 held-out 质量评估。本轮 full40、prefix2 的 0020/0021 恢复链及 native Engram 六组矩阵证据与独立审计均已补齐。任何失败保留原始结果，以新的编号记录重试，不修改历史 PASS/FAIL。

## 最终源码与资源清理记录

126 个源码文件的本地/远端比对（本地证据：`artifacts/new-l20d-20261013/effective-source-local-comparison.json`） 全部匹配、无缺失或差异；共同 HEAD 为 `3868011979caaa3358bfd779ec78367df1d01205`。本地分支为 `codex/posttrain-sft`，远端为 `codex/posttrain-new-l20d`；本次实现包含尚未提交的工作区修改，不能只凭 HEAD 重建实际运行版本，应保留 effective source manifest（本地证据：`artifacts/new-l20d-20261013/remote-full40-evidence/artifacts/new-l20d-20261013/effective-source-manifest.json`） 及源码文件。

最终只读清理检查（本地证据：`artifacts/new-l20d-20261013/final-cleanup-verification.json`） 时间为 2026-10-14 01:47:47 +08:00：gate 处于 passed、sequence33；历次 gate/runner 共 144 个已记录 PID/启动身份均不再存在；GPU used 为 `[4,4,4,4,1,1,1,1]` MiB，utilization 全 0，compute-apps 列表为空。这是一次后续当前快照，与 gate0033 记录的两次连续回收证据分开。它确认本任务已知进程清理和当时资源状态，不用于追溯归因之前未知的 627 MiB 占用，也不保证未来宿主资源永远不变。最终终端截图（本地证据：`artifacts/new-l20d-20261013/final-verification.png`） 可辅助人工核对，数值验收以 JSON/日志证据为准。
