# Rollout lifecycle and policy publication

Rollout Manager separates inference from native PyTorch learning. The local RL
CLI selects vLLM by default for its supported Qwen head profile; native inference
requires explicit `native-pytorch` selection. Unsupported capabilities fail
before learner allocation, with no automatic backend fallback.

## Components and scope

- The RL runtime constructs the learner and orders optimization, publication and
  checkpointing.
- The coordinator owns admission, immutable policy identity, group collection
  and resource lifecycle.
- An owned vLLM actor owns inference and acknowledged updates through worker RPC.
- Terminal-Bench providers retain exact assistant actions, tool contexts and
  original verifier results. The separate CPU executor owns sandboxes, tools,
  verification and cleanup.

The local vLLM training adapter supports flat, unquantized Qwen3 MoE in BF16
with a frozen backbone and trainable `lm_head`. It requires disjoint learner and
actor devices and cached native replay. Full-model and quantized Qwen updates
are unsupported by that adapter.

[Distributed DeepSeek query LoRA](../distributed-deepseek-grpo.md) uses a separate
EP/FSDP learner and remote TP8 vLLM service, with direct in-memory adapter
publication. Its configuration, numerical correction and checkpoint limitations
are distinct from the local Qwen manager.

## Actions and probabilities

Each assistant turn retains its context, action token IDs, behavior log
probabilities, stop metadata and policy version/digest/lineage. Tool results
become context for later turns rather than action targets. Learning scores
recorded actions using the native model. Cross-engine numerical equality is
not assumed; the selected GRPO/correction mode defines the denominator and
admission checks.

CUDA graph capability checks use actual capture/replay, rather than a flag.
Policy publication pauses admission and drains active inference before updating
supported tensors. Workers acknowledge their loaded tensor state and the exact
immutable policy before admission resumes.

The local manager's format-2 checkpoints retain optimizer/RNG state, group
cursors, pending complete groups and publication receipts. Generic evaluation
uses the same actor with independent seeds while training admission is paused.

## Terminal-Bench delivery

Artifact transport binds chunk and whole-artifact hashes, durable acknowledgments
and exclusive phase claims. Malformed complete shell output executes no tool
and still reaches the original verifier. Transport/actor failures remain
infrastructure outcomes, never synthetic zero rewards.

Terminal closure requires request-, payload- and policy-bound delivery and drain
proof. Missing proof quarantines the resource; HTTP success or CPU sandbox
cleanup alone does not prove GPU drain. Transient readiness checks can retry,
but unknown generations or committed optimizer steps cannot be replayed safely.

See the [usage guide](guide.md) for configuration and recovery, and
[testing](../testing.md) for focused contracts. Fixture tests validate software
contracts; full-model memory, sustained throughput and task quality need
validation on the intended deployment.
