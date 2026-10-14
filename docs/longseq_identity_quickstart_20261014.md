# 原实验环境：复现“你是谁 → RTP大模型”

> 发布说明：此页记录历史实验。模型权重、adapter、原始日志和 `artifacts/` 证据保留在实验环境，不随 Git 仓库发布；下文标注“本地证据”的路径不是仓库下载链接。通用训练用法见 [SFT 指南](sft.md)。

> 本页命令依赖原实验环境中已保存的 runner、adapter 和 GPU gate。仅克隆此仓库不能直接执行这些历史命令；新环境请从上面的 SFT 指南开始。

下面三份 adapter 均已完成真实 4096-token、20 步 LoRA 训练：每条训练输入有 4096 个有效 token，非短问答补齐 padding；中性长背景不含目标身份，只监督答案及 EOS。五道固定问题的严格成功率均为基座 **0/5 → adapter 5/5**；保存后独立进程重载，生成 token、文本、EOS 和记录的目标 NLL 均与训练后精确一致。直接题“你是谁”在训练中出现，其余三道短问法和一道长背景问法未出现；这只验证身份适配，不代表通用能力评估。证据：Qwen（本地证据：`artifacts/new-l20d-20261014-longseq/lora-qwen4096-v1/validation-summary.json`）、DS none（本地证据：`artifacts/new-l20d-20261014-longseq/lora-dsv4-full4096-v1/validation-summary.json`）、DS HF（本地证据：`artifacts/new-l20d-20261014-longseq/lora-dsv4-full4096-hf-v1/validation-summary.json`），详见 [本轮报告](longseq_identity_results_20261014.md)。

在 Web IDE 的开发机终端进入实际仓库：

```bash
cd /path/to/Tiny-DSV41
```

以下 adapter 路径均相对该目录；原始基座必须继续挂载且保持不变。

| 选择值 | adapter 路径 | 原始基座 | 重载配置 |
|---|---|---|---|
| `qwen` | `artifacts/new-l20d-20261014-longseq/lora-qwen4096-v1/train/adapter.pt` | `/mnt/fuse/Qwen/Qwen3-30B-A3B` | 48 层，WORLD_SIZE=EP=2，CP=1，cache=`none` |
| `dsv4-none` | `artifacts/new-l20d-20261014-longseq/lora-dsv4-full4096-v1/train/adapter.pt` | `/mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash` | 40 层文本 backbone，WORLD_SIZE=EP=8，CP=1，cache=`none` |
| `dsv4-hf` | `artifacts/new-l20d-20261014-longseq/lora-dsv4-full4096-hf-v1/train/adapter.pt` | `/mnt/fuse/deepseek-ai/DeepSeek-V4.1-Flash` | 40 层文本 backbone，WORLD_SIZE=EP=8，CP=1，cache=`hf` |

`adapter.pt` 是本仓库 native LoRA 格式，不是 PEFT 文件，也不是完整模型。加载器从同目录 `lora_eval_reference.json` 恢复 LoRA、EP、cache 和内存配置；不要手动将 HF adapter 按 `none` 模式加载。基座身份检查覆盖 config/index SHA，不覆盖每个权重 shard 的完整内容。

## 复制执行一次独立生成

等待当前 benchmark 全部结束、原 GPU gate 回到 `passed` 后执行。下面保留已验证命令中的八卡资源检查、GPU UUID、进程数、恢复等待及全部阈值，只新建个人结果目录，不重跑训练。Qwen 实际使用两卡，DS 使用八卡；DS 仍需加载整个基座，启动会较慢。

默认亲自问 Qwen“你是谁”。将第一行的 `qwen` 改成 `dsv4-none` 或 `dsv4-hf`，即可分别问 DeepSeek 控制组或 HF adapter；也可将问句改成不包含目标答案的其他身份问法。

```bash
.venv-posttrain/bin/python - qwen "你是谁" <<'PY'
import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys

from scripts.prepare_identity_sft import TemplateCodec, make_example

choices = {
    "qwen": ("lora-qwen4096-v1", 2, "none"),
    "dsv4-none": ("lora-dsv4-full4096-v1", 8, "none"),
    "dsv4-hf": ("lora-dsv4-full4096-hf-v1", 8, "hf"),
}
choice = sys.argv[1]
question = sys.argv[2] if len(sys.argv) > 2 else None
name, world, cache = choices[choice]
run = (Path("artifacts/new-l20d-20261014-longseq") / name).resolve()
config = json.loads((run / "train/run_config.json").read_text())
assert (config["world_size"], config["ep_size"], config["cp_size"],
        config["cache_quantization"]) == (world, world, 1, cache)

# 完整命令来自这份模型已经通过的独立进程重载记录。
cmd = json.loads((run / "fresh-generation-command.json").read_text())
assert f"--nproc-per-node={world}" in cmd
assert Path(cmd[cmd.index("--adapter-in") + 1]) == run / "train/adapter.pt"
state = Path(cmd[cmd.index("--state-dir") + 1]) / "state.json"
assert json.loads(state.read_text())["status"] == "passed", "请等待现有任务完成；不要重置 gate"

stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
personal = run / ("manual-" + stamp)
personal.mkdir()  # 已存在会拒绝，原报告不会被覆盖。
report = personal / "generation.json"
cmd[cmd.index("--name") + 1] = "manual-" + choice + "-" + stamp
cmd[cmd.index("--output") + 1] = str(report)

if question is not None:
    # 同一已验证 prompt 文件保留 tokenizer/config/template 身份。
    spec = json.loads(Path(cmd[cmd.index("--prompts") + 1]).read_text())
    codec = TemplateCodec(config["source"]["path"])
    for key in ("tokenizer_sha256", "config_sha256", "template_sha256", "tokenizer_config_sha256"):
        assert codec.encoding[key] == spec[key], "挂载模型或模板发生变化"
    _, prompt, _ = make_example(codec, "manual-identity", question, seq_len=4096, split="manual")
    spec["prompts"] = [prompt]
    spec["formatting_policy"]["heldout"] = "用户指定的单个身份问题；不属于原五题重放"
    prompt_file = personal / "prompts.json"
    prompt_file.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
    cmd[cmd.index("--prompts") + 1] = str(prompt_file)
    # 新问句/单题不能与原五题文件做逐项精确比较。
    index = cmd.index("--expected-generation")
    del cmd[index:index + 2]

(personal / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
print(shlex.join(cmd), flush=True)
subprocess.run(cmd, check=True)
result = json.loads(report.read_text())
for row in result["adapted_fresh_process"]["prompts"]:
    print(row["id"], repr(row["generated_text"]), "stop=" + row["stop_reason"],
          "strict_success=" + str(row["strict_success"]))
print("报告：", report)
if "reload_comparison" in result:
    print("与原始生成逐项精确一致：", result["reload_comparison"]["passed"])
PY
```

默认问题的预期输出为 `'RTP大模型' stop=eos strict_success=True`。结果 JSON 还保留实际输入 token、生成 token、包含特殊 token 的文本和目标 NLL。最多生成 16 个新 token，采用 greedy、eval 模式、关闭 dropout；到长度上限但未 EOS 不算严格成功。新问法是否学会，以实际输出为准，程序成功退出本身不代表身份回答正确。

若要重放原来全部五题并验证 token、文本、EOS 和 NLL 精确一致，删除第一行中的 `"你是谁"` 参数再运行同一代码块。它会继续使用原 `heldout.json` 和 `--expected-generation`。已验证提示文件为 Qwen 五题（本地证据：`artifacts/new-l20d-20261014-longseq/identity-qwen-4096/heldout.json`） 和 DS 五题（本地证据：`artifacts/new-l20d-20261014-longseq/identity-dsv4-4096/heldout.json`）；完整历史命令分别见 Qwen（本地证据：`artifacts/new-l20d-20261014-longseq/lora-qwen4096-v1/fresh-generation-command.json`）、DS none（本地证据：`artifacts/new-l20d-20261014-longseq/lora-dsv4-full4096-v1/fresh-generation-command.json`）、DS HF（本地证据：`artifacts/new-l20d-20261014-longseq/lora-dsv4-full4096-hf-v1/fresh-generation-command.json`）。

当前 [evaluator](../scripts/evaluate_identity_qa.py) 接受 prompt JSON，不提供交互聊天 CLI。上面的 `TemplateCodec` / `make_example` 来自现有 [数据准备实现](../scripts/prepare_identity_sft.py)，读取挂载模型的完整聊天模板并检查 prompt/答案边界。DeepSeek 使用模板的非思考模式，已对照挂载官方编码格式；不要自行拼接角色特殊 token。单题推理长度短不改变此前真实 4096-token 的训练记录。

## 重新训练一份 4096-token、20 步 adapter

只加载已完成的 adapter 时无需执行本节。若要重跑训练，等待当前矩阵和收尾检查全部结束、共享 gate 为 `passed` 后，任选下面一个命令串行执行。每次使用新的 `--run-name`；现有目录会被拒绝。运行器先检查真实模型、数据和容量，再执行训练及独立进程生成，沿用原八卡 UUID、门禁阈值和 180 秒回收等待，不会自动降低序列长度。

Qwen 完整 48 层，EP2：

```bash
.venv-posttrain/bin/python artifacts/new-l20d-20261014-longseq/run-longseq-validation.py \
  --model qwen \
  --data-dir artifacts/new-l20d-20261014-longseq/identity-qwen-4096 \
  --run-name qwen4096-manual-v1 \
  --seq-len 4096 --steps 20
```

DeepSeek 完整 40 层文本 backbone，EP8，attention KV/index 使用 `hf` 数值格式：

```bash
.venv-posttrain/bin/python artifacts/new-l20d-20261014-longseq/run-longseq-hf-validation.py \
  --model dsv4-full \
  --data-dir artifacts/new-l20d-20261014-longseq/identity-dsv4-4096 \
  --run-name dsv4-full4096-hf-manual-v1 \
  --seq-len 4096 --steps 20 --full-capacity-approved
```

两条命令默认读取本页表格中的挂载模型，要求参与训练的每卡至少 264 GiB 空闲显存，并继续检查全部八卡；`--full-capacity-approved` 只确认选择完整 DS40，不能跳过容量检查。加 `--dry-run` 可在不启动 GPU 的情况下查看完整命令。DS `none` 控制组使用第一个运行器，改为 `--model dsv4-full`、DS 数据目录和新名字，并保留 `--full-capacity-approved`。

新结果位于 `artifacts/new-l20d-20261014-longseq/lora-<run-name>/`，其中 `train/adapter.pt` 与 `train/lora_eval_reference.json` 成对加载，`train/metrics.jsonl` 记录 20 步训练，`fresh-generation.json` 记录重载生成；使用本页加载代码时，将 `choices` 中相应目录名替换为新目录名即可。底层入口是仓库根目录的 `train_sft.py`；运行器固定 LoRA rank8/alpha16/dropout0.1、AdamW lr0.001、batch1/accum1、full activation checkpointing 和 compact LoRA checkpoint。`--lora-probe-tokens 128` 只限制额外验证探针，实际训练及 `--min-active-tokens-per-example` 都是 4096。

这些命令使用已经准备好的带答案掩码数据，直接复用挂载聊天模板。若换成普通 prompt/response JSONL，`train_sft.py` 的默认基线文本格式并不会自动应用模型聊天模板；新数据应先用对应模板编码并核对监督边界。32 条身份训练样本及五道问题仅用于身份适配验证，不能代表通用 SFT 质量。


## DeepSeek 官方格式与 HF 模式的范围

原生 DeepSeek Engram 从挂载 checkpoint 加载预训练表，沿用实际 prime 分区、offset、hash multipliers 和 tokenizer token map；依据见 挂载格式核验（本地证据：`artifacts/new-l20d-20261014-longseq/mounted-engram-format-audit.json`）。Qwen 的额外大表是随机 dummy，只模拟寻址与规模，不能当作 DeepSeek 原生 Engram。

官方 Engram checkpoint 使用 FP8 E4M3 行及每 32 列的 UE8M0 scale；训练加载器将其解码为 BF16，CPU 源表和 GPU row cache 都保存这些 BF16 行。这里的格式一致指 token 压缩映射、2/3/4-gram、每阶 8 个 head、256 列行布局、hash/offset 和 gate 结合顺序的一致，不表示 SGD 后仍保留原始压缩字节。行分片路由、稀疏 SGD、预取和更新后缓存失效属于本训练框架的扩展，官方 inference 代码不定义这些训练行为。Qwen dummy 使用相同表规模和寻址实现，但使用 Qwen token ID 的 identity map、单 hidden stream 和随机表，只用于效率及实现一致性测试。

`cache_quantization=hf` 对 attention KV/index 按对应 FP8/FP4 格式取值，再重建为 **BF16**；训练中使用 identity STE。官方固定版本的 11 组局部量化器用例已经达到重建值及 BF16 位模式精确一致，见 量化器 GPU 证据（本地证据：`artifacts/new-l20d-20261014-longseq/official-cache-parity-gpu-signedzero.json`）。运行时仍是 BF16 存储，不能据此声称 packed FP4/FP8 cache 的内存节省；该证据也不覆盖官方 backward、`o_a` 前投影激活量化或整模型 logits 等价。full40 HF 身份测试与 prefix2 Engram 卸载矩阵分别留证，后者层 0/1 不覆盖压缩 attention 层。

## Qwen 约 183 GiB dummy 与原生 Engram 矩阵

Qwen v4 与原生 DS prefix2 v4 均已完成六组各 2 warmup＋20 measured steps 及原资源门禁，见 Qwen 正式独立审计（本地证据：`artifacts/new-l20d-20261014-longseq/qwen4096-v4-independent-audit-complete.json`） 和 原生 DS 正式独立审计（本地证据：`artifacts/new-l20d-20261014-longseq/native4096-v4-independent-audit-complete.json`）。各自 22 步 loss、记录的采样值和更新最大值跨策略一致；这不等于全梯度、全表逐元素比较，也不支持 Qwen 与 DS 的跨模型质量或速度比较。详细数值及收尾状态见 [本轮报告](longseq_identity_results_20261014.md)。此前的 严格小表对照（本地证据：`artifacts/new-l20d-20261014-longseq/qwen-v3-strict-independent-audit.json`） 与 首步完整捕获审计（本地证据：`artifacts/new-l20d-20261014-longseq/qwen-v3-strict-capture-independent-audit.json`） 单独保留，不替代正式大表结果。

新的 [Qwen benchmark](../tests/qwen_engram_sft_benchmark.py) 支持显式 `--deterministic`，默认关闭。启用时，启动器必须传 `CUBLAS_WORKSPACE_CONFIG=:4096:8`；程序在首次 CUDA 初始化前执行 `torch.use_deterministic_algorithms(True, warn_only=False)`，缺环境配置、过晚设置或不支持的算子都会失败，不自动退回非确定路径。GPU/CPU 各组必须采用相同模式；实际值记录在 `summary.workload.determinism`，reference 比较会拒绝配置混用。这个开关只用于 Qwen benchmark，不要传给身份生成器或原生 DS benchmark。

正式 v4 运行器（本地证据：`artifacts/new-l20d-20261014-longseq/run-engram-longseq-v4.py`） 自动给 Qwen 的每个子命令加入上述环境和开关，不使用梯度捕获 wrapper。每组保持真实 4096-token、2 warmup＋20 measured steps、batch1、两次 microbatch、EP2、SGD lr0.01 和原比较容差；六种策略为 GPU、CPU sync、CPU async、CPU cache 8/128/1 MiB。DeepSeek 组固定真实 prefix2、原生 layer1 Engram 和 `hf` 模式。两套矩阵均沿用原八卡 gate，每轮开始/结束检查资源并等待回收。

本轮 串行运行器（本地证据：`artifacts/new-l20d-20261014-longseq/run-remaining-engram-v4.py`） 已经完成两套矩阵，使用已审阅的 源码变更清单（本地证据：`artifacts/new-l20d-20261014-longseq/qwen-v4-reviewed-transition.json`），不是可重复覆盖的通用训练入口。Qwen 各策略读取 `qwen-init-cache-full-v3/manifest.json` 对应的同一份、已完整校验的初始行字节；该表是随机初始化的 dummy，并非省略实体存储的占位表。运行器按 Qwen 六组→证据包→原生 DS 六组→证据包串行执行，任一失败即停止。结果目录是 `qwen-dummy4096-v4` 和 `dsv4-native4096-hf-v4`；不要重建已使用的 transition、删除结果目录或重置 gate 来重复启动。

这套矩阵衡量自然 Dolly 数据上的存储/传输开销及数值一致性，不评价新增随机 Engram 的模型质量。报告的 measured step 时间包含同步和验证开销，排除加载和 warmup；cache 命中率不等于净通信节省，更不保证提速。新增独立性能实验应使用新的候选目录和对应源码/数据清单，不应将不同源码或设置的结果合并为原 v4 记录。
