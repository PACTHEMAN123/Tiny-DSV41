"""Fail-closed training evidence for existing generation routes.

This contract consumes evidence from a serving implementation; it never
reconstructs tokens, probabilities, or a loaded-policy receipt from text.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


class TrainingCapabilityError(ValueError):
    """The generation route cannot establish the required behavior evidence."""


def evidence_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


class TrainingGenerationContract:
    REQUIRED = ("token_ids", "sampled_token_logprobs", "loaded_policy_identity",
                "request_response_binding", "full_categorical")

    def __init__(self, policy, capabilities):
        from .training import PolicyRef
        if not isinstance(policy, (PolicyRef, Mapping)):
            raise TrainingCapabilityError("Training requires an explicit leased PolicyRef")
        self.policy = policy if isinstance(policy, PolicyRef) else PolicyRef.from_dict(policy)
        if not isinstance(capabilities, Mapping) or type(capabilities.get("schema_version")) is not int or capabilities["schema_version"] != 1:
            raise TrainingCapabilityError("Training requires a versioned serving capability receipt")
        if any(capabilities.get(name) is not True for name in self.REQUIRED):
            raise TrainingCapabilityError("Serving route lacks exact tokens/logprobs/policy/binding capabilities")
        if capabilities.get("logprob_convention") != "sampled_distribution":
            raise TrainingCapabilityError("Behavior logprobs must describe the sampled distribution")
        if capabilities.get("policy") != self.policy.to_dict():
            raise TrainingCapabilityError("Serving capability receipt differs from the leased policy")
        for name in ("tokenizer_sha256", "chat_template_sha256"):
            value = self.policy.identities.get(name)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise TrainingCapabilityError("Training policy requires exact " + name)
        self.capabilities = dict(capabilities)

    def check_payload(self, payload):
        if not isinstance(payload, dict):
            raise TrainingCapabilityError("Training generation payload must be a mapping")
        sampling = self.policy.sampling
        if (payload.get("temperature") != sampling["temperature"] or
                payload.get("top_p", 1) != sampling.get("top_p", 1) or
                payload.get("top_k", 0) != sampling.get("top_k", 0)):
            raise TrainingCapabilityError("Request sampling differs from the leased full categorical policy")
        if payload.get("stream", False) is not False or payload.get("n", 1) != 1:
            raise TrainingCapabilityError("Training requires one complete nonstreaming choice")
        if type(payload.get("max_tokens")) is not int or payload["max_tokens"] < 1:
            raise TrainingCapabilityError("Training requires an explicit positive token budget")
        # Validate JSON before a request can become externally observable.
        evidence_digest(payload)

    def validate(self, payload, completion, request_id, turn_id):
        from .training import PolicyRef, TurnTrace
        self.check_payload(payload)
        if not isinstance(completion, dict):
            raise TrainingCapabilityError("Training completion must contain raw serving evidence")
        evidence = completion.get("training_evidence")
        if not isinstance(evidence, dict) or type(evidence.get("schema_version")) is not int or evidence["schema_version"] != 1:
            raise TrainingCapabilityError("Serving completion has no training_evidence; text is insufficient")
        if evidence.get("request_id") != request_id:
            raise TrainingCapabilityError("Training evidence request identity differs")
        if evidence.get("request_payload_sha256") != evidence_digest(payload):
            raise TrainingCapabilityError("Training evidence does not bind the dispatched request")
        raw = {key: value for key, value in completion.items() if key != "training_evidence"}
        if evidence.get("response_sha256") != evidence_digest(raw):
            raise TrainingCapabilityError("Training evidence does not bind the raw completion")
        if evidence.get("logprob_convention") != "sampled_distribution":
            raise TrainingCapabilityError("Serving evidence uses an unsupported logprob convention")
        actual = PolicyRef.from_dict(evidence.get("policy", {}))
        if actual != self.policy:
            raise TrainingCapabilityError("Actual loaded policy differs from the group lease")
        if completion.get("model") != payload.get("model") or not isinstance(payload.get("model"), str):
            raise TrainingCapabilityError("Raw completion model differs from the request")
        choices = completion.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise TrainingCapabilityError("Training evidence requires one raw completion choice")
        finish = evidence.get("finish_reason")
        if choices[0].get("finish_reason") != ("stop" if finish == "eos" else finish):
            raise TrainingCapabilityError("Training finish reason differs from the raw choice")
        if not isinstance(evidence.get("action_token_ids"), list) or len(evidence["action_token_ids"]) > payload["max_tokens"]:
            raise TrainingCapabilityError("Raw action tokens exceed the dispatched generation budget")
        try:
            return TurnTrace(turn_id=turn_id, request_id=request_id,
                context_token_ids=evidence["context_token_ids"],
                action_token_ids=evidence["action_token_ids"],
                behavior_log_probs=evidence["behavior_log_probs"], policy=actual,
                finish_reason=finish, stop_token_count=evidence["stop_token_count"],
                context_truncation=evidence["context_truncation"])
        except (KeyError, TypeError, ValueError) as error:
            raise TrainingCapabilityError("Invalid raw token/logprob training evidence") from error
