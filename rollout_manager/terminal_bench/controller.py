"""Task budgets and compatibility imports for separated serving/verifier contracts."""
from .evidence_io import HASH, positive, read_json, unique, write_new
from .serving_contracts import (
    MODEL, NATIVE_PROFILE, NATIVE_STATIC_KEYS, TRAINING_PROFILES, VLLM_TRAINING_PROFILE,
    guard_gate, native_controller_identity, training_controller_identity,
    validate_native_controller, validate_training_controller, validate_vllm_engine,
    validate_vllm_kernel_config, validate_vllm_startup_adapter,
)
from .verification import (
    NAME, classify_trial, ctrf_counts, original_verifier_sources,
    reviewed_sparql_missing_output, reviewed_sparql_original_context,
    reviewed_sparql_query_syntax_failure, verifier_ambiguities,
)

SETUP_SECONDS = 360
CLEANUP_SECONDS = 300


def task_row(task, name):
    env = task.config.environment
    resources = {"cpus": env.cpus, "memory_mb": env.memory_mb, "storage_mb": env.storage_mb}
    if any(type(value) is not int or value <= 0 for value in resources.values()):
        raise ValueError("Explicit positive task CPU, memory and storage are required")
    budgets = {"environment_build": positive(env.build_timeout_sec, "build budget"),
               "agent_setup": SETUP_SECONDS,
               "agent": positive(task.config.agent.timeout_sec, "agent budget"),
               "verifier": positive(task.config.verifier.timeout_sec, "verifier budget"),
               "cleanup_reserve": CLEANUP_SECONDS}
    return {"task_id": name, "path": str(task.paths.task_dir), "resources": resources,
            "budgets_seconds": budgets, "required_service_seconds": sum(budgets.values()),
            "image": env.docker_image, "qualification": "staged_not_runtime_qualified"}
