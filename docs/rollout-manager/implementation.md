# Rollout Manager implementation report

Rollout Manager replaces the original RL rollout worker. New configurations
and the training CLI select vLLM inference by default; native PyTorch remains
the learner and owns GRPO, AdamW and checkpoints. Unsupported serving
capabilities fail before learner allocation. Native inference is available
only through the explicit `native-pytorch` compatibility option, with no
automatic fallback after a vLLM failure.

## Architecture and supported scope

The implementation separates five responsibilities:

- The RL runtime selects the model family, constructs the learner and orders
  training, publication and checkpointing.
- Rollout Manager and its coordinator own admission, immutable policy identity,
  trajectory collection and resource lifecycle.
- The owned vLLM actor controls inference, CUDA graph execution and acknowledged
  head updates through a named worker extension RPC.
- The shared completion module owns shell grammar and typed completion status;
  the sealed provider owns validated CPU group import and serving control.
- The CPU executor owns Pouch sandboxes and the unchanged Harbor verifier.
  Artifact transport and the CPU supervisor automate leases, group delivery
  and post-update evaluation.

The current vLLM training adapter supports flat, unquantized Qwen3 MoE in BF16,
with a frozen backbone and trainable `lm_head`. It requires disjoint actor and
learner GPU devices, one actor generation slot and cached native replay.
Full-model updates, quantized weights and DeepSeek-V4.1 vLLM activation are
unsupported. DeepSeek requires its own compatible weight/checkpoint adapter;
its existing native compatibility path does not establish vLLM readiness.
Model geometry and context limits come from mounted assets, with offline
loading and explicit capability checks.

## Correctness guarantees

Every assistant turn retains its actual context, action token IDs, processed
behavior logprobs, stop metadata and policy version, digest and lineage. Tool
results become context for the next turn rather than generated action targets.
The learner scores recorded actions using its current native model and uses
the actual vLLM probabilities as the GRPO denominator. Numerical equality
between engines is not assumed; finite-ratio validation and configured
clipping remain necessary.

CUDA graph readiness requires observed capture and successful replay evidence,
not configuration flags alone. Policy updates pause admission and prove actor
and delivery drain before loading a supported head tensor into existing
storage. The worker hashes its actual loaded tensor and acknowledges the
exact immutable policy before admission resumes. Checkpoints preserve
optimizer/RNG state, group cursors, pending complete groups and publication
receipts. Generic evaluation uses the same owned vLLM actor with separate
seeds while training admission is paused.

Automatic artifact delivery binds chunk and whole-artifact hashes, durable
acknowledgments and exclusive phase claims. A completed malformed shell
response retains its token evidence, executes no tool and stops the agent;
the original verifier still determines reward. Transport or actor failure
remains an infrastructure outcome. Terminal negative closure requires a
request-, payload- and policy-bound receipt plus actual drain proof; absent
proof keeps resources quarantined. HTTP completion and CPU sandbox cleanup
cannot independently certify GPU drain. Transient browser readiness failures
are retried without replaying generations or committed optimizer steps.

## Validation and remaining limits

A real Qwen Terminal-Bench cycle validated an eligible G8 group, automatic
artifact delivery, one AdamW step, checkpoint integrity, acknowledged policy-v1
activation and a fresh v1 episode with verified cleanup and final delivery
closure. All training and evaluation rewards were zero. Advantages were zero,
so the exact-zero objective path advanced optimizer state while leaving
weights unchanged. This proves operational integration, not improved task
performance or nonzero-objective learner throughput. Invalid evidence and
infrastructure failures never become synthetic zero rewards.

Live GPU qualification established actual vLLM graph execution and native
head-gradient support before the subsequent source cleanup. Focused source
tests cover tokenization, shared grammar/provider boundaries, artifact
transport, controllers and recovery contracts. These tests do not recertify
every cleaned source byte on GPU. Broader task performance, sustained
multi-update learning and DeepSeek vLLM training remain unqualified.

See the [usage guide](guide.md) for configuration, environment requirements,
automatic handoff, recovery and focused checks.
