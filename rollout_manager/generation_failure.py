"""Request-bound failure closure, independent of success or training eligibility."""
from __future__ import annotations

import re

from .generation_training import evidence_digest


def failure_is_terminal(receipt):
    return (receipt["sampler_drained"] is True and receipt["actual_actor_drain_proven"] is True
            and ((receipt["sampler_outcome"] == "not_admitted"
                  and receipt["proof_kind"] == "controller_rejected_before_sampler_dispatch")
                 or (receipt["sampler_outcome"] == "completed"
                     and receipt["proof_kind"] == "owned_vllm_final_request_output")))


def validate_generation_failure(receipt, payload, request_id, expected_policy_ack=None):
    keys = {"schema_version", "request_id", "request_payload_sha256", "policy_ack", "stage",
            "sampler_outcome", "sampler_drained", "actual_actor_drain_proven", "drain_scope",
            "proof_kind", "automatic_replay"}
    if (not isinstance(receipt, dict) or set(receipt) != keys
            or type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1
            or receipt["request_id"] != request_id or not re.fullmatch(r"[a-f0-9]{32}", request_id)
            or receipt["request_payload_sha256"] != evidence_digest(payload)
            or receipt["drain_scope"] != "request" or receipt["automatic_replay"] is not False
            or not isinstance(receipt["stage"], str) or not re.fullmatch(r"[a-z0-9_]{1,64}", receipt["stage"])
            or receipt["sampler_outcome"] not in {"not_admitted", "completed", "unknown"}
            or type(receipt["sampler_drained"]) is not bool or type(receipt["actual_actor_drain_proven"]) is not bool):
        raise ValueError("invalid bound generation failure receipt")
    ack = receipt["policy_ack"]
    if (not isinstance(ack, dict) or set(ack) != {"version", "digest", "lineage"}
            or type(ack["version"]) is not int or ack["version"] < 0
            or not isinstance(ack["digest"], str) or not re.fullmatch(r"[a-f0-9]{64}", ack["digest"])
            or not isinstance(ack["lineage"], str) or not ack["lineage"]
            or expected_policy_ack is not None and ack != expected_policy_ack):
        raise ValueError("generation failure differs from the leased policy")
    if receipt["sampler_outcome"] == "unknown":
        if receipt["proof_kind"] != "none" or receipt["sampler_drained"] or receipt["actual_actor_drain_proven"]:
            raise ValueError("unknown generation cannot claim sampler drain")
    elif not failure_is_terminal(receipt):
        raise ValueError("known generation failure requires actual request closure proof")
    return dict(receipt)
