# Eight-rank validation and profiling follow-up

2026-09-29, Python 3.10.20, PyTorch 2.10.0+cu129, NVIDIA H20.
The [portable results](results/memory-eight-h20.json) distinguish completed checks,
failures and measurements that have not run. This follow-up does **not** establish
eight-GPU BF16 acceptance or an eight-GPU throughput improvement.

Subsequent availability recheck: an actual eight-H20 CP2/EP4 run with the smaller
14.4M-parameter model successfully captured loading and training traces on every
rank. See [the profiling recheck](gpu_profile_recheck.md). This supersedes the
environment-blocked status below for profiling availability; the earlier BF16
failures and larger-workload performance acceptance remain open.

2026-10-10: the small-model [eight-rank memory baseline](memory_eight_baseline.md)
now has 12 completed runs. Combined reduces training allocated peak by 4.08%
and loading peak by 91.56%; this is not acceptance of the larger workload or an
isolated throughput improvement.

## Completed correctness checks

The final workspace unit suite ran 84 tests: 79 passed and five skipped because
their HF reference implementation is absent. `git diff --check` passed.

The fixture now uses eight experts and sequence length 16 on eight ranks.
Keeping the previous six-expert/length-12 fixture would silently exclude CP8 and
EP8 from the matrix. Six-rank runs retain six experts and length 12, preserving
the CP2/EP3 and CP3/EP2 intersecting-group checks.

All 16 FP32 FSDP2/NCCL combinations of CP/EP in `{1,2,4,8}` passed against the
unsharded reference. Maximum gradient error was `6.756745278835297e-6`.
Each case checks logical loaded weights, padded outputs, two microbatches,
external loss scaling, three SGD updates and checkpoint restoration. Candidate
models use direct loading, async CPU Engram and deduplication. The raw results
are in `artifacts/memory-eight/fp32/`.

BF16 uses a same-topology GPU baseline, SGD learning rate `0.01`, three updates,
and unchanged `atol=0.002`, `rtol=0.02` for logits:

| Case | CP/EP | Initial candidate policy | Result |
| --- | --- | --- | --- |
| 0 | 1/1 | GPU, layerwise, synchronous | Passed |
| 1 | 1/8 | GPU, direct, deduplicated, full AC | Failed logits; largest reported absolute error `0.0556640625` |
| 2 | 2/4 | CPU synchronous, direct, FSDP offload | NCCL error in reference forward; needs rerun and investigation |
| 3 | 4/2 | CPU async, direct, deduplicated, full AC, FSDP offload | Passed |
| 4 | 8/1 | CPU async, layerwise, deduplicated, full AC | Failed logits; largest reported absolute error `0.013671875` |
| 5 | 8/8 | CPU async, direct, deduplicated, full AC, FSDP offload | Failed logits; largest reported absolute error `0.013671875` |

Every case switches the candidate to synchronous CPU tables without
deduplication after its first update and restores checkpoint state. These are
combined policy/lifecycle tests, not isolated measurements of a single option.
Failures are retained in `artifacts/memory-eight/bf16/` and `bf16-rest/`.

Case 1 diagnostics found exactly matching initial weights, exact non-Engram
gradients on the first update, and Engram weight differences up to approximately
`1.192e-7` after that update. Logits passed through the next update and failed on
the third forward. A reference with matching deduplication passed the case
(`dedup-isolation.json`; maximum gradient error `1.1444091796875e-5`). This supports
BF16 partial duplicate-reduction rounding as a source of divergence. Amplification
by subsequent low-precision computation and discrete top-k choices is an
inference; this diagnostic does not resolve or waive the original failure.

The distributed deduplicated backward reduces duplicate gradients before
returning BF16 values, whereas the baseline retains individual contributions
until sparse-gradient coalescing. FP32 accumulation inside each partial sum
does not eliminate the intermediate BF16 rounding. Existing single-table or
two/six-rank passes therefore do not establish strict multi-step parity here.

## Measurement tooling

The benchmark runner now supports a shared synthetic fixture, three or more
fresh-process repetitions, deterministic randomized variant order, separate
timing/profiling passes, all-rank traces, per-step times and explicit learning
rate. A manifest records scheduled trials; the summary reports unfinished,
failed and missing trials instead of silently omitting them.

Profiling includes prefetch, ID routing, worker lookup, row routing, resolution,
forward, backward and optimizer regions. All-thread collection captures worker
threads already created during warmup. A CPU regression verifies that coverage
and method restoration. CUDA trace analysis unions intervals per device and
measures H2D overlap with compute, excluding NCCL. Inclusive CPU region durations
and CUDA kernel sums are diagnostic values, not unprofiled step times.

`tests/summarize_memory_benchmark.py` reports median/range across fresh trials,
the maximum memory peak across ranks, globally summed transfer bytes per step,
per-rank lookup load and loss differences from the corresponding baseline.

The prepared 423,209,462-parameter synthetic workload is specified in
`configs/memory_eight_profile.json`: five layers, hidden size 512, 16 experts,
two 256-column Engram tables with 524291/524299 rows, world8/CP2/EP4, sequence512,
two microbatches, BF16 storage and SGD `0.0001`. Its hash vocabulary spreads
requests across row owners. Fixture generation occurs outside measured workers,
so worker CPU RSS excludes checkpoint construction. This is not a released
DeepSeek checkpoint or a full-model capacity test.

## Performance status

At the time of this initial validation, the eight-GPU profiling smoke and
repeated benchmark had **not run**. After
correctness tests, sandbox GPU access was unavailable, and three attempts to
use the previously working external execution path failed automatic approval
with `stream disconnected before completion`. The tool explicitly classified
this as an approval-service failure, not an unsafe-action decision. No new GPU
performance numbers or fresh-process eight-rank resume results are claimed.

Reprocessing the [previous two-GPU measurements](memory_results.md) reproduced
the reported training peak reduction from 157.74 to 90.28 MiB (42.8%) in combined
mode. Async versus synchronous CPU throughput was 1373.92 versus 1301.61 tokens/s
(5.6%) in a single short trial. That remains insufficient evidence of a stable
speedup. Those runs have no offload CUDA trace for an overlap claim.
The new comparison also exposes maximum loss differences of `0.0104613304`
(CPU dedup/combined) and `0.0137538910` (GPU dedup) from the old baseline;
non-deduplicated variants matched its recorded losses exactly.

## Resume commands

Run in the repository root with `PYTHONPATH=.:tests` and the documented Torch
environment after GPU execution is restored. Keep benchmark jobs isolated from
other GPU and host-heavy jobs. Use a new output directory for independent runs.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 tests/run_memory_validation.py \
  --suite cuda --ranks 8 --bf16 --cases 2 --keep-going \
  --output-dir artifacts/memory-eight/bf16-retry

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.run \
  --standalone --nproc-per-node=8 tests/dsv4_memory_benchmark.py \
  --variant cpu_async --model-path artifacts/memory-eight/bench/fixture \
  --cp 2 --ep 4 --seq-len 512 --learning-rate .0001 --warmup 2 --steps 2 \
  --trace-dir artifacts/memory-eight/profile-smoke \
  --output artifacts/memory-eight/smoke.json

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 tests/run_memory_validation.py \
  --suite benchmark --ranks 8 --config configs/memory_eight_profile.json \
  --cp 2 --ep 4 --seq-len 512 --learning-rate .0001 --warmup 3 --steps 10 \
  --repeats 3 --trace-variants baseline,cpu,cpu_async,combined,expert_offload \
  --timeout 600 --output-dir artifacts/memory-eight/bench

python3 tests/summarize_memory_benchmark.py artifacts/memory-eight/bench \
  --output artifacts/memory-eight/benchmark-summary.json
```

For fresh-process resume, use the commands in [memory_validation.md](memory_validation.md)
with eight visible GPUs, `--nproc-per-node=8`, `--cp 2 --ep 4`, and separate SGD
and AdamW output folders. Both save and resume phases are still outstanding on
eight ranks. The performance learning rate does not replace the stricter `0.01`
correctness cases or change their acceptance thresholds.
