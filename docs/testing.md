# Testing

Run all commands from the repository root.

## Unit tests

```bash
python3 -m unittest discover -s tests -v
```

The Qwen checkpoint tests are skipped when the optional `safetensors` package
is not installed.

Default RL selection, learner/actor contracts and legacy checkpoint compatibility
can be checked without starting a vLLM engine:

```bash
PYTHONPATH=.:tests python3 -m unittest -v \
  test_generic_vllm_rl.GenericSelectionTests \
  test_generic_vllm_rl.GenericVLLMBackendTests \
  test_vllm_learner_contract.VLLMLearnerContractTests \
  test_rl_family.ModelFamilyTests \
  test_rl_evaluation.EvaluationTests.test_legacy_checkpoint_keeps_original_eos_and_checks_recorded_identity
```

These 18 CPU fixture tests passed on the integrated branch. Full SFT source
reconstruction and native SFT-to-RL initialization also passed six CPU tests on
Linux (`python3 -m unittest discover -s tests -p 'test_*sft_source.py' -v`).
Neither suite validates a released-model vLLM GPU engine.

SFT save controls have seven focused CPU regressions, also verified on Linux:

```bash
python3 -m unittest discover -s tests -p 'test_train_sft.py' -v
```

They cover default/periodic saving, `--no-save` for full and LoRA training,
suppression of adapter/Engram bundle exports, and resume followed by an update
without creating another checkpoint. Tokenizer/model loading and CUDA telemetry
are fixtures; actual tiny training and checkpoint save/load still run.

Selective FP32 SGD masters and checkpoint compatibility have 24 passing CPU
checks across these suites:

```bash
PYTHONPATH=.:tests python3 -m unittest test_sgd_fp32_master test_optim test_train_sft -v
PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_sft_master_checkpoint.py' -v
PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_sft_checkpoint.py' -v
```

They cover accumulated sub-BF16 updates, unchanged default SGD, exact parameter
selection, restored master state, invalid-state rejection before mutation,
ordinary optimizer compatibility, and SFT-to-RL weight initialization without
resuming the SFT optimizer.

The optional short/long identity fixture, answer labels and native greedy
generation have 12 focused CPU checks, verified locally and on Linux:

```bash
PYTHONPATH=.:tests python3 -m unittest -v \
  test_prepare_identity_sft test_sft_data test_sft_eval
```

To reproduce the Qwen 20-step mixed-context identity experiment:

```bash
PYTHONPATH=. python3 scripts/prepare_identity_sft.py \
  --model-path /path/to/Qwen3-30B-A3B --output-dir runs/qwen-identity-mixed \
  --include-short-training
PYTHONPATH=. python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  train_sft.py --model-path /path/to/Qwen3-30B-A3B --family qwen --mode full \
  --data runs/qwen-identity-mixed/train.jsonl --eval-data runs/qwen-identity-mixed/eval.jsonl \
  --output-dir runs/qwen-identity-mixed20 --steps 20 --seq-len 4608 \
  --batch-size 1 --gradient-accumulation-steps 1 --cp-size 1 --ep-size 1 \
  --optimizer adamw --learning-rate 1e-5 --seed 42 --gradient-checkpointing \
  --generation-eval --generation-max-new-tokens 16 --no-save
```

The eight examples give each global step four real long and four short inputs;
short inputs are not padded to 4096 for this claim. The unchanged five-prompt
evaluation includes one seen short prompt, three unseen short questions and one
unseen long prompt. The observed 3/5 result must retain that split. Native CE
evaluation assigns those five examples cyclically across eight ranks, repeating
the first three; it is not the equal-example five-question CE mean. Generation
evaluates each of the five prompts once. This fixture checks identity tuning,
not broad model quality, and `--no-save` does not retain the trained weights.

A separate original-step20 diagnostic compared teacher-forced, prompt-only and
same-length masked-EOS inputs for two short questions on all eight ranks. For
`你是谁？`, the first observed difference was a layer0 output delta of 0.000977;
later MoE routing diverged and the final maximum logit delta was 1.742188.
The masked-EOS and teacher-forced traces matched exactly; the second question
matched across all three forms. This supports sensitivity to input shape and
numerical paths, without identifying a particular kernel defect. No padding or
routing change was made to production inference to force that comparison to pass.

## CPU distributed parity

Shared RL token log probabilities and checkpointed gradients, including a CP
rank with no response tokens (model-level parity, not a distributed RL runtime):

```bash
PYTHONPATH=. python3 -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29503 \
  tests/dsv4_rl_cp_parity.py
```

```bash
PYTHONPATH=. python3 -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29501 \
  tests/dsv4_engram_parallel_parity.py
```

```bash
PYTHONPATH=.:tests python3 -m torch.distributed.run \
  --nnodes=1 --nproc-per-node=2 \
  --master-addr=127.0.0.1 --master-port=29502 \
  tests/qwen_parallel_parity.py --cpu --padded
```

## GPU distributed checks

The ordered-reduction regressions cover MoE combine, SwiGLU router-weight
gradients, mHC mixing-weight gradients and attention sink gradients:

```bash
PYTHONPATH=.:tests python3 -m unittest \
  test_moe_dispatch test_triton_moe test_triton_mhc test_triton_attention -v
```

All 14 tests passed on CUDA. These tests check repeated computations and
reference agreement for the affected outputs. The original attention KV and
compressed-KV backward uses atomic accumulation; the separate candidate
experiment below does not establish unrestricted full-model determinism.
Six replays of an actual full40 layer39 backward with identical inputs and
upstream gradients found KV/compressed-KV variation while query and sink gradients
were exact. This identifies a local source, not the sole cause of QA failures.
The distributed combine check, including a rank with no source tokens, is part
of `tests/dsv4_sft_gradient_parity.py`.

Native-kernel full-SFT update and sparse-checkpoint recovery can be checked with:

```bash
PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/dsv4_sft_in_backward.py --device cuda --output runs/sft-native-resume
```

On the reduced BF16/FP32 model, CP1/EP2 and CP2/EP2 both restored state exactly
and reproduced the next update exactly after the ordered-reduction fixes.
The ordinary and in-backward SGD update difference was zero. The test also
checks updated Engram rows, checkpoint payloads and a global non-finite abort.
Rank-local assertions are synchronized before the next collective so a failed
comparison reports its numerical error instead of leaving a peer waiting.
This is native Triton execution, without `--reference-torch`; it does not
establish bitwise recovery of the released 40-layer model.

A two-GPU selective-master checkpoint check also passed:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/test_sft_master_checkpoint.py --cuda-dir runs/sft-master-resume
```

Use a new output directory. This runs actual BF16 FSDP2 forward/backward,
preserves sub-BF16 residuals, and restores into newly created model, optimizer
and hooks in the same processes. Parameters, FP32 master and RNG state match
exactly, as does the next update. Corrupt master metadata on one rank is
rejected on both ranks before live state changes. This is not a separate-process
restart or a full-model quality test.

Released-model quality is evaluated separately from the component checks. The
latest stable-indexer full40 resident/CPU-offload pair on eight B300s (driver
name L20D) completed 160 finite updates per run and clean release. Both final
scores were **6/6** (original five questions 5/5, seen bare prompt 1/1). Offload
scored 6/6 at every 20-step evaluation; resident scored 4/6 at step 20 and 6/6
at steps 40 through 160. Both post-final train-fit checks scored **32/32
prompt-only generations (31 distinct prompts)** and **160/160 teacher-forced
answer tokens**, including 32/32 first and EOS tokens. Both used the same
32-record mixture, CP1/EP8, BF16 SGD `1e-3`, gradient checkpointing, no FP32
master and no checkpoint saving; offload used a 128 MiB Engram cache per rank.
Four examples per update were long (at least 4096 actual tokens) and four were
short.

The simultaneous eight-GPU NVML peak observed inside training-update windows
was **1,714.064453125 GiB resident versus 1,348.826171875 GiB offloaded**:
**365.23828125 GiB (21.3083%)** less total device memory. Windows span the earliest
rank's update begin through the latest rank's update end, excluding interim QA
gaps but including diagnostic observers and the first-step trace. NVML samples
can miss shorter peaks; this is not an uninstrumented performance benchmark.
Both runs finished with two all-eight-GPU clean-release observations.

All six same-gradient Engram replay stages and all 1,584,960 cache-return row
occurrences passed for the first two updates. Across the pair, all 7,408 bounded
initial-parameter sample records, 32,624 first-step visited table rows and 8,480
recorded original-forward tensors matched. SGD order matched, as did available
original/recomputed tensors within each run. All 16 first-step Engram incoming
gradients nevertheless differed. Over the common visited rows, coalesced
gradients differed by **4.6259% relative L2**, with maximum absolute difference
**0.0007781982421875** and **6,901,763 changed elements**. This is neither
full-gradient equality nor full-trajectory verification. This is the historical
160-step result; the following controlled two-step intervention is new evidence.

Four full40 runs under one 177-file manifest compared atomic and candidate
ordered KV backward, each with resident and CPU-offloaded Engram. The two-step
recipe used CP1/EP8, BF16 SGD `1e-3` in backward, gradient checkpointing, seed 42,
cap 4608 and the unchanged long/short data. Actual long inputs reached
4115–4116 tokens. QA and saving were disabled. All four runs completed and
passed clean-release checks.

| Cross-placement comparison | Atomic backward | Candidate ordered backward |
| --- | --- | --- |
| First-step full dense shard gradient hashes differing / 7,200 | 6,573 | 0 |
| First-step Engram coalesced-gradient relative L2 | 12.8078% | 0 |
| First-step changed post-SGD Engram elements / 8,351,744 | 52 | 0 |
| Second-step changed post-SGD Engram elements / 20,004,608 | 191 | 0 |
| Second-step dense gradient sample records differing / 7,200 | 6,777 | 0 |

The ordered pair matched every touched row's initial value, gradient and actual
post-SGD value in both steps: 32,624 and 78,143 rows across 16 rank/table pairs
per step. Lookup outputs, incoming gradients, raw sparse gradients, recorded
boundaries, SGD order and available recomputation checks also matched. Both
modes passed same-gradient CPU/CUDA replay and all 1,584,960 offload cache-return
row comparisons. Initial dense weights and post-update dense weights were
sampled; only first-step dense gradients were fully hashed. This does not
establish complete second-step dense-gradient or full-weight equality.

The captured rank-0/layer-39 context had the same complete ten input signatures
across placements and backward modes, including output, logsumexp and upstream.
On that actual context, six atomic replays produced five window-KV and four
compressed-KV gradient hashes (maximum differences from the first replay
4.76837e-7 and 1.52588e-5); all four ordered gradient channels repeated exactly
and matched the captured ordered training return. Inputs and RNG were unchanged.
This controlled intervention localizes a removable backward divergence to KV
reduction for the measured workload. It does not prove that every historical
4.6259% difference or QA outcome has only this cause. The two-step candidate
and replay are distinct from the completed production40 pair below.
Diagnostics are not throughput tests.

The integrated production path has separate evidence from the injected
candidate. On the same captured rank-0/layer-39 context, production ordered and
the frozen candidate matched all four returned gradients bitwise over six
replays. The original atomic path still varied in KV channels. The original
177-file capture manifest and the new production manifest are checked separately;
these are not retroactively the same source version.

A reduced native two-GPU gate explicitly selected production ordered attention
at CP1/EP2 and CP2/EP2. Both ranks passed the original update, strict resume and
non-finite-abort assertions, recording ordered calls and no atomic calls.
A separate B13/H64/D512, length-129 test crossed the signed-32-bit edge-address
boundary (8.125 GiB edge buffer). Two complete ordered runs were bitwise equal;
each batch's output/dQ/dKV/dCompressed also matched its B1 execution. Shared
sink gradients were checked by complete-repeat equality, not B1-then-sum
bitwise equivalence. These gates passed release checks; they do not validate
full40 CP2 or make a throughput claim.

The high-fan-in seed-53 formal test failed its ideal-autograd KV check in both
stress cases. A subsequent all-channel diagnostic found nine identical
violating elements on the native atomic and ordered paths. That reference did
not model the backward's saved rounded BF16 output or its
BF16 probability/score-gradient casts. The revised independent CPU reference
uses the saved output/LSE, FP32 local dot/exp calculations, BF16 P/dScore and
FP64 per-key accumulation; both FP64 sums and final BF16 values are checked.
Shapes, unit-scale random inputs, seed and **rtol/atol 0.04/0.04 remain unchanged**.
Output/dQ/dSink retain their existing ideal reference. This is a correction to
reference semantics, not a claim that the old test passed unchanged or that
all reference errors are zero. The revised formal rerun passed **three CPU and
three CUDA tests, with no skips**, including repeated ordered results, global
policy selection and the unchanged output/query/sink checks. The run completed
with clean release; approximate reference checks retain the stated tolerances.

DSV4 SFT enables this native policy and records
`training_config.sparse_attention_backward="ordered_edges_chunk128_v1"`.
Strict resume rejects missing/different policies; older atomic continuation is
not silently accepted. Qwen identity is unchanged. Explicit `deterministic=True`
or PyTorch's global deterministic-algorithms setting selects the ordered
attention class at forward time, including when the global flag is disabled
before backward. That selection is not a full-model determinism guarantee.

| Production-entry full40 validation (fixed 40 updates) | Resident | CPU offload |
| --- | --- | --- |
| Finite updates / actual ordered selection / release | 40 / passed / passed | 40 / passed / passed |
| Final native prompt-only QA6 | 4/6 | 4/6 |
| Final native teacher loss | 0.2393022277392447 | 0.2393022277392447 |

The production pair used its own 177-file manifest (`c88b7554`); the independent
raw audit (`073fb55a`) passed all structure and equality gates. Each placement
recorded 12,800 actual ordered backwards across eight ranks and 40 updates,
with zero atomic calls. All 7,200 complete first-step dense shard gradients
matched. Second-step dense gradient samples and both steps' dense weight
samples matched, as did observed boundaries, SGD order and available
recomputation records. Initial dense parameters remain sampled.

All 640 rank/table/step comparisons matched complete touched-row indices,
pre-update values, coalesced gradients and actual post-SGD values across the
40 updates. First-two-step PTs additionally compared all 32,624 and 78,143
visited rows quantitatively: gradient and effective-update differences were
zero. First-two-step same-gradient replay and lookup/cache-authority checks
also passed. The later 38 steps have complete touched-row signatures, not PT
values, whole-table scans or complete dense-gradient/weight comparisons.

All 40 recorded training losses matched; step40 loss was 0.02502476767404005.
The two final generated-token lists and teacher metrics also matched, but
**quality remained 4/6**, not a passing six-question result. Eval indices 1
and 3 failed: a verbose RTP response reached the 16-token cap, and another
response identified DeepSeek. Bare identity and the long4096-token question
were correct. The teacher metric is the native eight-slot cyclic batch over
six records, not an unweighted six-record mean. Both jobs passed release.
No checkpoint was saved, and diagnostic timing is not a throughput benchmark.

The separate offload-only `identity-production-quality-offload160-a1` run
completed its fixed 160 updates and resource release under a 175-file manifest
(`f1bc662f`); its raw audit (`14ee4ec6`) passed structural checks. It retained
production source, original32 data, SGD1e-3, seed42, CP1/EP8, sequence cap4608
and cache128MiB, with no interim or best-step selection. All 51,200 actual
attention backwards across eight ranks used the ordered class; atomic calls
were zero. No checkpoint was saved.

Final native QA was **5/6**, not a six-question pass: seen prompts scored 2/2,
and the four inspected prompts absent from training scored 3/4. Eval index 1
(`我该怎样称呼正在回答的你？`) generated
`你可以叫我 **RTP大模型**，或者直接叫 **RTP**`, reaching the unchanged
16-token cap without EOS. The extra wording still fails exact match. The other
five questions, including the former DeepSeek-identity failure at index 3,
bare identity and the 4,096-token prompt, returned exactly the target and EOS.
Step 160 training loss was 0.00813104, last-20-step mean loss 0.00946377, and final native
teacher loss 0.13806481 (eight cyclic sample slots over six records).

Post-final prompt-only generation fitted **32/32 training records (31/31 distinct
prompts)**. All 160 teacher-forced target tokens were correct, including 32/32
first tokens and 32/32 EOS tokens; all eight ranks recorded identical fit results.
The unweighted training-record mean answer/first-token/EOS NLLs were
**0.00849599 / 0.02670684 / 0.00378105**. These in-sample diagnostics do not add
held-out questions or change the original QA score. This is a single-placement
quality run, not paired 160-step gradient validation or a speedup benchmark. Earlier
results, forward-prefix differences, LoRA/RL limits and the full40 checkpoint
waiver remain unchanged.

Rank 0 post-final offload comparisons at the first-answer query still found
teacher/prefix differences: the 12-token prompt first differed at layer 0 MoE
output, and the 4096-token prompt at layer 0 output. Final logit maximum absolute
differences were 2.40625 and 2.5625, with the correct first-token top1 in both
modes. These are observed boundaries, not a unique causal explanation. QA6 is
an inspected development set; train-fit is in-sample, and these runs do not
exercise full-model checkpoint persistence or prove bitwise equivalence.

Historical negative results are retained. The `8fab9c7` full40 pair finished at
resident 6/6 and offload 5/6; after the ordered reductions, the fixed 160-step
pair finished at 5/6 and 4/6. The private output-head
master diagnostic at global LR `1e-3` finished at 3/6. The native selective-master
offload run at global LR `1e-4` completed 160 finite updates and clean release but
finished at **0/6**, with last-20-step mean training CE 1.42382. These are final
scores on an inspected development set, not best-step selection or a controlled
claim that precision changes improve quality. The subsequent fixed 640-step
native-master offload run completed with clean release and finished at **3/6**
(last-20-step mean training CE 0.11082). Post-final training-prompt generation
scored 31/32 records, or 30/31 distinct prompts; eval6 scored 2/2 on exact seen
prompts and 1/4 on unseen development prompts. All eight ranks recorded identical
train-fit teacher scores in this run. Sampled head-master arithmetic and
first-two-step Engram replay/cache checks passed; these limited numerical checks
do not establish full-trajectory equivalence or answer quality.

The SFT stable-indexer change is separately covered by loader and checkpoint
regressions in `tests/test_train_sft.py`. On one B300, six synthetic selection
cases and ten native scorer/attention cases passed. Both default and stable
selection preserved the tested prefixes on this CUDA backend; the CPU ordering
counterexamples did not reproduce in these GPU cases. On prebuilt FP32 score
matrices of `[4608, 2304]` and `[4608, 4608]`, stable selection took median
0.4193 ms and 1.3516 ms, versus 0.2108 ms and 0.3428 ms for default top-k, over
ten measured repetitions after three warmups. Incremental allocated peaks were
122.52 MiB and 650 MiB, versus 33.87 MiB for default selection. These are
selection-only measurements, not whole-model performance. The completed
full-model resident/offload results are described above; they do not turn these
selection-only timings into an end-to-end speed claim.
See the README for recipe boundaries.

A separate short-prefix B300 check used 12-to-17 and 6-to-8 candidate widths,
the actual `min(512, width)` selection count, and invalid-index sentinels.
Default selection reordered visible candidates in four of six selection cases;
stable selection preserved their order in all six. Both methods nevertheless
produced exactly equal BF16 outputs in the two native scorer/attention cases.
Changing future content at fixed shape also left the observed scores and
attention outputs unchanged. This reproduces the ordering issue, but does not
establish it as the cause of the full-model answer failures.

Four-rank SFT checks cover full training and LoRA with trainable CPU Engram,
CP2/EP2, two-rank expert FSDP groups, and unequal supervised token counts:

```bash
PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1,2,3 \
python3 -m torch.distributed.run --standalone --nproc-per-node=4 \
  tests/dsv4_sft_dp_parity.py --device cuda --output runs/sft-dp4-cuda.json
```

CUDA uses deterministic Torch reference kernels with BF16/FP32 parameters.
Each step starts from bit-exact matching state and checks gradients and one
SGD update. This does not assert equality of independently rounded BF16
training trajectories. Use `--device cpu` for two independent FP32 steps.
The output path must be new; reports include the precision and error bounds.

The integration corrects expert-gradient normalization for CP1/EP>1 and disables
FSDP backward prefetch whenever EP is enabled. These shared changes also affect
the original training entry points: old expert updates can differ in magnitude,
and prefetch scheduling can change performance. The CP/EP gradient checks verify
normalization against an unsharded reference; they do not promise identical
numerics or throughput to the earlier implementation.

Two-GPU Qwen output and gradient parity:

```bash
PYTHONPATH=.:tests CUDA_VISIBLE_DEVICES=0,1 \
python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/qwen_parallel_parity.py
```

Eight-GPU Qwen FSDP smoke test:

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.run --standalone --nproc-per-node=8 \
  tests/qwen_fsdp_smoke.py
```

## Two-node NCCL check

Run this on both nodes. Change `NODE_RANK` to `0` or `1`, and replace
`10.0.0.1` with the first container's IP:

```bash
NODE_RANK=0
MASTER_ADDR=10.0.0.1

PYTHONPATH=. NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3 -m torch.distributed.run \
  --nnodes=2 --nproc-per-node=8 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 \
  tests/distributed_nccl_smoke.py
```
