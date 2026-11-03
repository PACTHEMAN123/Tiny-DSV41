# Distributed DeepSeek compatibility runtime

`c3868011979ca` is based on the 30-file native import closure from commit
`3868011979caaa3358bfd779ec78367df1d01205`, plus a local compressed-KV gather
correction. The added helper returns zero for masked slots and uses the main
attention's stable edge ordering and FP32 reduction for first-order CUDA BF16/FP32
gradients, with a native PyTorch fallback. It does not import the current model
from outside this snapshot. An optional routing-weight helper fixes each token's
SwiGLU-weight backward dot product to one FP32 reduction tree. The manifest records
these local patches and pins all 32 Python files;
the learner verifies that inventory before importing it as `_dsv41_grpo_native`.
It contains source code, without model weights or checkpoints.

The current public loader has a different interface. This compatibility runtime
preserves early root sharding, `cache_quantization="hf"`, and the gradient policy
used by the CP1 query-LoRA learner. The public branch supplies the objective,
response batching, tokenizer, and rewards. Replacing this closure requires
validating model loading, distributed gradients, and memory use together.

`python -m dsv41_train.rl.distributed_grpo --help` describes the learner command.
It defaults to this runtime; `--native-source PATH` selects an alternative
manifest-verified snapshot. The learner trains rank-8 query adapters on a frozen
BF16 backbone, uses resident row-sharded Engram and activation checkpointing, and
synchronizes each update with the authenticated vLLM service. CPU Engram offload
is not supported by this loader configuration.

The optional `bf16-no-split-k` learner mode also enables fixed routing-weight
gradients and FP32 reduction/accumulation in the existing FSDP layer wrappers.
Only the last microbatch synchronizes gradients. Replicated G32 uses rank AVG;
unique B4 owners use SUM, with the same 1/32 backward coefficient. This preserves
the mathematical objective while changing rounding relative to default mode.
