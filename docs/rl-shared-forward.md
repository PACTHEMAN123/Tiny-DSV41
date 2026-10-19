# RL selected-token forward

Native DeepSeek GRPO reuses the public SFT causal-LM forward, including layer
device transfers and activation checkpointing. Supplying aligned targets returns
token probabilities without materializing the complete vocabulary logits or
adding the SFT routing auxiliary objective.

## Interface

`target_ids[b, t]` is the token predicted by input position `t`; the caller
aligns targets explicitly, without a second shift. `target_mask` selects objective
positions and is intersected with the input attention mask. Masked IDs may use
placeholders such as `-100`; token ID zero remains valid. Packed callers must
mask cross-sequence targets. The last valid input can predict a continuation
not present in `input_ids`.

The result is FP32 `[batch, local_sequence]`, zero at masked positions. Targets
and masks follow the context-parallel hidden-state shard. This method does not
reduce the objective; GRPO retains the mean of per-response token means.

The LM head projects selected states and checkpoints projection/log-softmax
in training. Decoder checkpointing uses immutable shadow state and per-device
contexts. Sparse indexing uses stable tie selection; supported CUDA shapes
use fused scoring and other shapes use the reference path.

The shared forward is a model-level contract. The local native manager and the
[distributed learner](distributed-deepseek-grpo.md) provide separate orchestration
and batching behavior.

## Checks

```bash
python3 -m unittest discover -s tests -p 'test_rl_shared_forward.py' -v
PYTHONPATH=. python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/dsv4_rl_cp_parity.py
```

These cover selected-token score/gradient parity, padding and token zero,
recomputation, SFT compatibility and a context-parallel rank with no response
tokens. Run CUDA checks on the intended hardware before relying on fused kernels
or full-model capacity.
