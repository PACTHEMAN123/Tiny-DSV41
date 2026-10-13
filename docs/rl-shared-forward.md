# RL shared forward (P1)

This phase reuses the SFT model for native PyTorch GRPO. It does not add
distributed FSDP/EP/CP orchestration or change the GRPO objective.

## Implementation

- [x] Add regression tests for selected-token probabilities, gradients, managed
  root calls, activation recomputation, and checkpoint compatibility.
- [x] Add aligned `target_ids` / `target_mask` scoring to the public causal LM
  forward. Return local token log probabilities without the vocabulary logits
  or an auxiliary routing objective. Keep the existing SFT labels path.
- [x] Move layer-device transfers into the shared decoder execution, including
  immutable checkpoint inputs for CSA2 shadow state and per-device contexts.
- [x] Share sparse indexing between SFT and RL, with stable selection enabled
  for RL, a CPU reference path, and compatible CUDA fused scoring.
- [x] Route policy and GRPO scoring through the public forward, expose the
  gradient checkpointing setting, and support toggling it on resume.
- [x] Run focused and regression tests, review the diff, and record GPU limits.

## Interface

`target_ids[b, t]` is the token predicted by input position `t`; the caller
aligns targets explicitly. It is not shifted again. `target_mask` identifies
the positions contributing to the objective and is intersected with the input
attention mask. Masked target IDs may be placeholders such as `-100`.
Packed-sequence callers must mask cross-sequence targets themselves. The last
valid input may predict a continuation token not present in `input_ids`.

The new output is FP32 `[batch, local_sequence]`, with zero at masked positions.
Targets and masks are sharded along the same context-parallel dimension as the
hidden states. The method performs no objective reduction. GRPO retains its
mean of per-response means. The existing runtime still uses one response per
forward and does not yet aggregate context-parallel scores.

The LM head projects selected hidden states only. During training its projection
and log-softmax are checkpointed to avoid retaining vocabulary-sized tensors
for backward. Vocabulary projection chunking and response batching are P2.

## Validation

Baseline: `python -m unittest discover -s tests -p 'test_rl_*.py'` ran 77 tests,
74 passed and 3 CUDA-only tests skipped on the local CPU environment.

Required checks: FP32 scoring and gradient parity against full-logit scoring,
padding and token-zero handling, no RL routing auxiliary loss, real decoder
recomputation, SFT loss compatibility, stable prefix probabilities, and
train/resume with the new execution setting. CUDA and two-GPU tests are retained
for execution on the target B200/B300 hardware.

Completed on 2026-10-13 with Python 3.12 and PyTorch 2.14.1 (macOS CPU):

- `python -m unittest discover -s tests`: 135 tests, 125 passed, 10 skipped
  because CUDA or optional dependencies are unavailable. This includes the
  legacy-checkpoint resume test with decoder recomputation enabled.
- `tests/dsv4_rl_cp_parity.py` under two-process CPU/Gloo torchrun: local scores
  and summed parameter gradients match a single-process policy. All response
  tokens reside on one rank; the empty response rank still participates in
  checkpointed backward collectives.
- Independent code review found a CUDA dispatch boundary issue. The dispatch
  now requires power-of-two tile dimensions of at least 16; other shapes use
  the reference path. A focused regression first reproduced the issue and now
  passes. No remaining actionable review findings.
- `git diff --check`: passed.

CUDA fused scoring, two-GPU checkpoint replay, and full-model B200/B300 memory
and throughput are not verified by these CPU checks. The shared decoder still
prefetches all selected Engram rows before executing layers; its peak GPU memory
cost needs measurement. Distributed RL orchestration, FSDP integration, chunked
projection, and batched rollout remain later phases.
