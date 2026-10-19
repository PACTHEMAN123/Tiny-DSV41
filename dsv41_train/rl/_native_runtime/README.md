# Distributed DeepSeek compatibility runtime

`c3868011979ca` is the 30-file native import closure from commit
`3868011979caaa3358bfd779ec78367df1d01205`. Its manifest pins every Python file;
the learner verifies that inventory before importing it as `_dsv41_grpo_native`.
It contains source code, without model weights or checkpoints.

The current public loader has a different interface. This compatibility runtime
preserves early root sharding, `cache_quantization="hf"`, and the gradient policy
used by the EP16/FSDP16 LoRA learner. The public branch supplies the objective,
response batching, tokenizer, and rewards. Replacing this closure requires
validating model loading, distributed gradients, and memory use together.

`python -m dsv41_train.rl.distributed_grpo --help` describes the learner command.
It defaults to this runtime; `--native-source PATH` selects an alternative
manifest-verified snapshot. The learner trains rank-8 query adapters on a frozen
BF16 backbone, uses resident row-sharded Engram and activation checkpointing, and
synchronizes each update with the authenticated vLLM service. CPU Engram offload
is not supported by this loader configuration.
