# Testing

Run commands from the repository root. Use the same dependency environment as
the code under test; GPU and distributed tests require suitable hardware and
available devices. CPU fixture tests do not validate released-model GPU capacity,
native/vLLM numerical parity or downstream task improvement.

## Unit tests

```bash
python3 -m unittest discover -s tests -v
```

Tests requiring CUDA or optional checkpoint packages skip when unavailable.
Focused RL and SFT contracts can be run independently:

```bash
python3 -m unittest discover -s tests -p 'test_rl_*.py' -v
python3 -m unittest discover -s tests -p 'test_distributed_grpo*.py' -v
python3 -m unittest discover -s tests -p 'test_deepseek_vllm*.py' -v
python3 -m unittest discover -s tests -p 'test_sft*.py' -v
python3 -m unittest discover -s tests -p 'test_train_sft.py' -v
PYTHONPATH=.:tests python3 -m unittest test_sgd_fp32_master test_optim -v
```

Rollout transport and lifecycle checks:

```bash
python3 -m unittest discover -s tests -p 'test_generic_vllm_rl.py' -v
python3 -m unittest discover -s tests -p 'test_vllm_controller.py' -v
python3 -m unittest discover -s tests/rollout_terminal -t tests -p 'test_automatic_artifacts.py' -v
node --test tests/rollout_terminal/test_relay_artifacts.mjs
```

## CPU distributed parity

These compare model-level scores, gradients and shard ownership against a
reference. They do not start a distributed training service.

```bash
PYTHONPATH=. python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/dsv4_rl_cp_parity.py
PYTHONPATH=. python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/dsv4_engram_parallel_parity.py
PYTHONPATH=.:tests python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/qwen_parallel_parity.py --cpu --padded
```

Run launches sequentially or assign distinct rendezvous ports.

## CUDA and distributed checks

Native reduction/kernel parity:

```bash
PYTHONPATH=.:tests python3 -m unittest \
  test_moe_dispatch test_triton_moe test_triton_mhc test_triton_attention -v
```

Two-GPU Qwen parity and DeepSeek Engram BF16 update parity:

```bash
PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/qwen_parallel_parity.py
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/engram_bf16_device_parity.py --device cuda
```

Four-GPU SFT gradient/update checks cover full training and LoRA with trainable
CPU Engram, CP2/EP2, expert FSDP and unequal supervised-token counts:

```bash
PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.run --standalone --nproc-per-node=4 \
  tests/dsv4_sft_dp_parity.py --device cuda --output runs/sft-parity.json
```

The output path must be new. Use `--device cpu` for FP32 CPU execution.
These tests check bounded workloads; they do not imply that a full checkpoint
fits or that independent BF16 training trajectories remain bitwise identical.

## Two-node NCCL check

Run on both eight-GPU nodes with `NODE_RANK=0` or `1` and `MASTER_ADDR` set to
node 0's reachable address. Select network interfaces for the actual cluster.

```bash
PYTHONPATH=. NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
python3 -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  tests/distributed_nccl_smoke.py
```

Before submission, run `git diff --check` and the focused tests for the changed
contracts. Avoid including generated datasets, checkpoints or diagnostic outputs
in the commit.

## Full40 SFT Engram offload benchmark

On eight B300s (driver name L20D), four fresh eight-rank runs used
resident/offload/offload/resident order. Each trained the released 40-layer
DeepSeek-V4.1 Flash model in full-parameter SFT for 100 steps: 20 warmup,
then 80 measured. The fixed recipe used CP1/EP8, BF16 SGD `1e-3` in backward,
gradient checkpointing, seed 42, per-rank batch size 1 and one microbatch. The same 32
records supplied four long examples (4097–4120 actual tokens) and four short
examples per update, under sequence cap 4608. Only Engram placement/cache
differed: GPU-resident/cache 0 versus trainable CPU-offloaded/cache 128 MiB per
rank. Checkpoint saving and generation evaluation were disabled; native teacher
evaluation before/after training was outside measurement. No diagnostic training observers
were installed.

Results are bound to frozen source/data manifest `e513fdc9` and audit
`b6ce0e79`: 175 historical source/data files plus two benchmark entry files.
All runs passed source/native-function identity and resource-release checks.
This benchmark does not revalidate subsequent commits.

| Run | Mean update (s) | Median (s) | p90 (s) | Input tokens/s | Answer tokens/s | Sampled 8-GPU NVML peak (GiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 resident | 4.2822 | 4.2397 | 4.4178 | 3853.49 | 9.3410 | 1711.625 |
| 2 offload | 4.4106 | 4.3629 | 4.5867 | 3741.34 | 9.0691 | 1348.480 |
| 3 offload | 4.4043 | 4.3493 | 4.5796 | 3746.67 | 9.0820 | 1348.467 |
| 4 resident | 4.2718 | 4.2269 | 4.4104 | 3862.92 | 9.3638 | 1711.625 |
| Pooled resident, 160 steps | 4.2770 | 4.2328 | 4.4169 | 3858.20 | 9.3524 | 1711.625 |
| Pooled offload, 160 steps | 4.4074 | 4.3601 | 4.5841 | 3744.00 | 9.0755 | 1348.480 |

The native maximum-rank timer covers cache-before accounting, training updates
and final CUDA synchronization, excluding batch construction, final metric/cache
reductions and logging. Loading and evaluation are excluded. Rates divide total
actual non-padding input tokens or supervised answer tokens (including EOS) by
total measured seconds; each run measured 1,320,120 input and 3,200 answer tokens.
Mean native update time increased **3.0502%**; input throughput fell **2.9599%**.

External step-completion observations include loop/reporting overhead: pooled
resident/offload windows total **684.957/705.937 s**, corroborating a **3.06%**
wall-time slowdown. Their observation bounds are **682.945–686.966 s** and
**703.927–707.945 s**; these are polling bounds, not confidence intervals.

Total sampled device memory fell **363.145 GiB (21.2164%)**. NVML totals sum all
eight GPUs from one query, using samples wholly inside the measured training
window; low-frequency sampling can miss shorter peaks. These are not sums of
independent per-card peaks. Maximum single-rank Torch allocated/reserved peaks
were separately **199.767/215.953 GiB resident** and **154.120/170.213 GiB offload**;
these are not whole-machine totals and can retain earlier allocator reservations.

Both offload runs had zero cross-call cache hits: each trainable table is looked
up once per step, and SGD invalidates its cache tags after every update.
Repeated IDs within that lookup are still deduplicated before H2D transfer.
The row-transfer counter excludes gradient D2H, CPU updates, metadata and EP
communication; it is not measured PCIe bandwidth.

All 100 scalar training losses matched step by step across the four runs; this
adds no gradient/update equality or answer-quality claim. There are two fresh
runs per placement, not 160 independent experiments. These results replace
probe-instrumented timings as the performance comparison for this frozen recipe.

## Cross-node GRPO response ownership

The 2026-11-04 qualification used four eight-GPU NVIDIA L20C nodes: 24 native
learner ranks and a separate eight-GPU vLLM actor. The full 40-layer
DeepSeek-V4.1-Flash-1017 ran query-LoRA r8/alpha8, CP1/EP24/FSDP24, global G32,
original B4 groups, context 2048, generation budget 1792, temperature 0.8,
learning rate `1e-4`, KL 0, Token-TIS cap 2 and one-group asynchronous prefetch.
The base model and resident Engram remained frozen.

Both arms used the same optional `bf16-no-split-k` math mode and the existing
compressed-KV gather optimization. The candidate assigned each original B4 to
one owner, with zero-loss collective placeholders on other ranks, and shortened
the activation proof to one token. Training/evaluation generation budgets did
not change. Frozen production source manifest: `95747a5aa418bb35f812b23ff6e8b19ed2c48b68cd186e2ec536da18b7193a9e`.

Full-model lag0/lag1 checks passed on all 24 ranks. Cross-arm relative L2
gradient/update differences were **0.02305%/0.03961%** at lag0 and
**0.04327%/0.47050%** at lag1, within the predeclared 1% L2, cosine and norm-ratio
limits. Each arm exactly matched its independent AdamW calculation. Focused CPU
regressions and actual two-GPU FSDP gradient/head-chunk checks also passed.
This preserves the objective, not bitwise equivalence to the old default math.

Fixed-input training replay used the same lag1 G32 with 34,487 response tokens:
one warmup per arm, then ABBA, without diagnostic hooks inside the timed calls.
Maximum-rank mean time fell **180.8342 -> 24.4236 s (7.4041x)**. There were only
two measured calls per arm; resets, actor sampling and publication were excluded.

Two fresh end-to-end runs each completed eight effective updates. After two
warmup updates, the commit2-to-commit8 results were:

| Metric | Replicated | Owned B4 |
| --- | ---: | ---: |
| Unique response tokens | 161,453 | 145,993 |
| Mean commit cycle (s) | 167.1328 | 23.7787 |
| Effective response tokens/s | 161.0027 | 1,023.2758 |
| Mean activation proof (s) | 7.6146 | 0.1547 |
| Maximum single learner GPU sampled NVML (GiB) | 143.7864 | 140.1360 |

The observed throughput ratio was **6.3556x**. Generated workloads differed:
the baseline consumed nine groups with three constant-reward skips in the window;
the candidate consumed six with no skips. Both windows began with one prefetched
group and ended empty, so this is neither an identical-input causal speedup nor
a balanced steady-state measurement. Use the fixed replay for the training claim.
Stage timing and 1 Hz NVML samples remained enabled; gradient/operator probes
were disabled. NVML can miss short peaks and is not a Torch allocator peak.
All ranks passed update/ACK audits, and all four nodes exited without remaining
GPU processes. This does not qualify full-parameter RL, CP or longer contexts.
