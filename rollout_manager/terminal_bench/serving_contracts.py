"""Serving ownership, loaded-policy and capacity proof contracts."""
from pathlib import Path

from .evidence_io import HASH, positive

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
NATIVE_PROFILE = "native-pytorch-v1"
VLLM_TRAINING_PROFILE = "vllm-training-v1"
TRAINING_PROFILES = (NATIVE_PROFILE, VLLM_TRAINING_PROFILE)
NATIVE_STATIC_KEYS = ("schema_version", "serving_profile", "served_model", "model_pid",
    "model_pid_start_ticks", "model_cmdline_sha256", "generation_slots", "started_unix",
    "expires_unix", "model_dir", "python_executable", "netns_inode")


def native_controller_identity(value, *, include_policy=True):
    """Fields bound by a startup receipt or a current policy guard snapshot."""
    keys = NATIVE_STATIC_KEYS + (("policy", "policy_ack", "capabilities") if include_policy else ())
    try:
        return {key: value[key] for key in keys}
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete native controller identity") from error


def training_controller_identity(value, *, include_policy=True):
    """Pin the engine configuration and actual worker owners independently of a version."""
    identity = native_controller_identity(value, include_policy=include_policy)
    if value.get("serving_profile") == VLLM_TRAINING_PROFILE:
        engine = value.get("vllm_engine", {})
        try:
            identity["vllm_engine"] = {key: engine[key] for key in
                ("schema_version", "version", "serving_profile", "tensor_parallel_size",
                 "logprobs_mode", "prefix_caching", "enforce_eager", "cudagraph_capture_sizes",
                 "compilation_mode", "cudagraph_mode", "kv_cache_memory_bytes", "max_model_len",
                 "gpu_memory_utilization", "kv_bytes_per_token", "full_context_raw_kv_bytes",
                 "kernel_config", "startup_adapter", "worker_rpc_method", "insecure_serialization",
                 "actor_process")}
            identity["vllm_engine"]["workers"] = [{key: worker[key] for key in
                ("pid", "start_ticks", "cmdline_sha256", "device", "device_uuid",
                 "kernel_config", "startup_adapter", "worker_rpc_method", "insecure_serialization",
                 "moe_implementations")} for worker in engine["workers"]]
            if include_policy:
                identity["vllm_engine"].update({key: engine[key] for key in
                    ("loaded_policy_ack", "loaded_trainable_tensors")})
        except (KeyError, TypeError) as error:
            raise ValueError("Incomplete actual vLLM worker identity") from error
    return identity


def validate_native_controller(value, model, *, now=None, required_seconds=0,
                               expected_policy=None, require_running=True, require_fresh=True):
    return validate_training_controller(value, model, serving_profile=NATIVE_PROFILE, now=now,
        required_seconds=required_seconds, expected_policy=expected_policy,
        require_running=require_running, require_fresh=require_fresh)


def validate_training_controller(value, model, *, serving_profile, now=None, required_seconds=0,
                                 expected_policy=None, require_running=True, require_fresh=True):
    """Validate an authenticated, locally owned controller proof carried by the browser.

    The GPU relay separately checks /proc and the listening socket. The CPU
    validates this proof only after a fresh authenticated browser roundtrip.
    """
    from ..generation_training import TrainingGenerationContract
    from ..training import PolicyRef
    import time
    now = time.time() if now is None else now
    if serving_profile not in TRAINING_PROFILES:
        raise ValueError("Unsupported owned training controller profile")
    training_controller_identity(value)
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1 or
            value.get("serving_profile") != serving_profile or value.get("served_model") != model):
        raise ValueError("Native controller profile or served model differs")
    if (type(value.get("generation_slots")) is not int or value["generation_slots"] != 1 or
            any(type(value.get(key)) is not int or value[key] <= 0 for key in
                ("model_pid", "model_pid_start_ticks", "netns_inode")) or
            not HASH.fullmatch(str(value.get("model_cmdline_sha256")))):
        raise ValueError("Invalid native process identity or serial generation capacity")
    for key in ("model_dir", "python_executable"):
        path = value.get(key)
        if not isinstance(path, str) or not Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("Native controller requires explicit mounted model and Python paths")
    if not Path(value["python_executable"]).name.startswith("python"):
        raise ValueError("Native controller process must use the verified Python executable")
    started = positive(value.get("started_unix"), "native service start")
    expires = positive(value.get("expires_unix"), "native service expiry")
    observed = positive(value.get("observed_unix"), "native service observation")
    if (not started <= observed <= now + 1 or not 0 < expires - started <= 43200 or
            expires - now < required_seconds or expires <= now or
            require_fresh and now - observed > 15):
        raise ValueError("Native controller observation is stale or lifetime is insufficient")
    state = value.get("controller_state")
    active = value.get("active_generations")
    if (state not in {"running", "paused"} or require_running and state != "running" or
            type(active) is not int or not 0 <= active <= 1):
        raise ValueError("Native controller is not ready or sampler ownership is invalid")
    policy = PolicyRef.from_dict(value.get("policy", {}))
    ack = {key: getattr(policy, key) for key in ("version", "digest", "lineage")}
    if value.get("policy_ack") != ack:
        raise ValueError("Native loaded artifact acknowledgement differs from its policy")
    TrainingGenerationContract(policy, value.get("capabilities"))
    if serving_profile == VLLM_TRAINING_PROFILE:
        validate_vllm_engine(value["vllm_engine"], policy)
        if (value["capabilities"].get("serving_profile") != serving_profile or
                policy.identities.get("rollout_engine") != "vllm"):
            raise ValueError("vLLM behavior identity or serving capabilities differ")
    if expected_policy is not None:
        expected = expected_policy if isinstance(expected_policy, PolicyRef) else PolicyRef.from_dict(expected_policy)
        if policy != expected:
            raise ValueError("Native loaded policy differs from the admitted group lease")
    return value


def validate_vllm_engine(engine, policy):
    """Require actual worker readback and processed full categorical probabilities."""
    if (not isinstance(engine, dict) or type(engine.get("schema_version")) is not int or
            engine["schema_version"] != 1 or engine.get("version") != "0.31.0" or
            engine.get("serving_profile") != VLLM_TRAINING_PROFILE or
            type(engine.get("tensor_parallel_size")) is not int or engine["tensor_parallel_size"] != 1 or
            engine.get("logprobs_mode") != "processed_logprobs" or
            engine.get("prefix_caching") is not False or engine.get("enforce_eager") is not False or
            engine.get("cudagraph_capture_sizes") != [1] or type(engine.get("compilation_mode")) is not int or
            engine["compilation_mode"] != 0 or engine.get("cudagraph_mode") != "FULL_DECODE_ONLY"):
        raise ValueError("vLLM engine configuration is outside the qualified training profile")
    ack = {key: getattr(policy, key) for key in ("version", "digest", "lineage")}
    from ..backends.vllm_profile import (VLLM_POLICY_IDENTITIES, VLLM_KV_CACHE_MEMORY_BYTES,
        VLLM_KV_BYTES_PER_TOKEN, VLLM_GPU_MEMORY_UTILIZATION, TRITON_EXPERT_CLASS, VLLM_WORKER_RPC)
    if any(policy.identities.get(key) != expected for key, expected in VLLM_POLICY_IDENTITIES.items()):
        raise ValueError("vLLM worker proof differs from the bound actor/learner identity")
    if (type(engine.get("max_model_len")) is not int or engine["max_model_len"] < 1 or
            str(engine["max_model_len"]) != policy.identities.get("vllm_max_model_len") or
            type(engine.get("kv_cache_memory_bytes")) is not int or
            engine["kv_cache_memory_bytes"] != VLLM_KV_CACHE_MEMORY_BYTES or
            type(engine.get("kv_bytes_per_token")) is not int or
            engine["kv_bytes_per_token"] != VLLM_KV_BYTES_PER_TOKEN or
            type(engine.get("full_context_raw_kv_bytes")) is not int or
            engine["full_context_raw_kv_bytes"] != engine["max_model_len"] * VLLM_KV_BYTES_PER_TOKEN or
            engine["full_context_raw_kv_bytes"] > engine["kv_cache_memory_bytes"] or
            type(engine.get("gpu_memory_utilization")) is not float or
            engine["gpu_memory_utilization"] != VLLM_GPU_MEMORY_UTILIZATION):
        raise ValueError("vLLM context/cache geometry differs from the original mounted model profile")
    validate_vllm_kernel_config(engine.get("kernel_config"))
    validate_vllm_startup_adapter(engine.get("startup_adapter"))
    if (engine.get("worker_rpc_method") != VLLM_WORKER_RPC or
            engine.get("insecure_serialization") is not False):
        raise ValueError("vLLM engine must use the secure named worker RPC")
    actor = engine.get("actor_process", {})
    if (not isinstance(actor, dict) or any(type(actor.get(key)) is not int or actor[key] < 1
            for key in ("pid", "start_ticks")) or not HASH.fullmatch(str(actor.get("cmdline_sha256")))):
        raise ValueError("vLLM actual actor process identity is unproven")
    tensors = engine.get("loaded_trainable_tensors", {})
    if not isinstance(tensors, dict):
        raise ValueError("vLLM actual loaded head is missing")
    shape = tensors.get("shape")
    if (engine.get("loaded_policy_ack") != ack or not HASH.fullmatch(str(tensors.get("sha256"))) or
            tensors.get("parameter_names") != ["backend.lm_head.weight"] or
            tensors.get("dtype") != "torch.bfloat16" or not isinstance(shape, list) or len(shape) != 2 or
            any(type(size) is not int or size < 1 for size in shape)):
        raise ValueError("vLLM actual loaded head or artifact acknowledgement is unproven")
    workers = engine.get("workers")
    if not isinstance(workers, list) or len(workers) != 1:
        raise ValueError("vLLM profile requires one actual tensor-parallel worker")
    for worker in workers:
        if not isinstance(worker, dict):
            raise ValueError("Invalid vLLM worker receipt")
        validate_vllm_kernel_config(worker.get("kernel_config"))
        validate_vllm_startup_adapter(worker.get("startup_adapter"))
        if (worker.get("worker_rpc_method") != VLLM_WORKER_RPC or
                worker.get("insecure_serialization") is not False):
            raise ValueError("vLLM actual worker must use the secure named RPC")
        if worker["startup_adapter"] != engine["startup_adapter"]:
            raise ValueError("vLLM startup adapter proof differs from its actual worker")
        implementations = worker.get("moe_implementations")
        if (not isinstance(implementations, list) or not implementations or
                any(not isinstance(row, dict) or set(row) != {"module", "backend",
                    "selected_experts_class", "experts_class"} or
                    not isinstance(row.get("module"), str) or not row["module"] or
                    row.get("backend") != "TRITON" or
                    row.get("selected_experts_class") != TRITON_EXPERT_CLASS or
                    row.get("experts_class") != TRITON_EXPERT_CLASS for row in implementations) or
                len({row["module"] for row in implementations}) != len(implementations)):
            raise ValueError("vLLM actual instantiated Triton experts are unproven")
        graph = worker.get("graph", {})
        if not isinstance(graph, dict):
            raise ValueError("Invalid vLLM graph receipt")
        if (any(type(worker.get(key)) is not int or worker[key] < 1 for key in ("pid", "start_ticks")) or
                not HASH.fullmatch(str(worker.get("cmdline_sha256"))) or
                not isinstance(worker.get("device"), str) or not worker["device"].startswith("cuda:") or
                not isinstance(worker.get("device_uuid"), str) or not worker["device_uuid"] or
                any(type(graph.get(key)) is not int or graph[key] < 0 for key in
                    ("captured_graph_count", "replay_calls"))):
            raise ValueError("vLLM worker ownership or graph counters are unproven")
    return engine


def validate_vllm_kernel_config(value):
    from ..backends.vllm_profile import VLLM_KERNEL_CONFIG
    if (not isinstance(value, dict) or set(value) != set(VLLM_KERNEL_CONFIG) or
            any(type(value[key]) is not type(expected) or value[key] != expected
                for key, expected in VLLM_KERNEL_CONFIG.items())):
        raise ValueError("vLLM actual kernel configuration differs from its bound profile")
    return value


def validate_vllm_startup_adapter(value):
    from ..backends.vllm_worker import VLLM_STARTUP_ADAPTER, STOCK_KERNEL_WARMUP_SHA256
    expected = {"profile": VLLM_STARTUP_ADAPTER, "stock_kernel_warmup_sha256": STOCK_KERNEL_WARMUP_SHA256,
        "model_type": "qwen3_moe", "preserves_original_warmup_body": True,
        "bypassed_unrelated_minimax_import": True}
    if (not isinstance(value, dict) or set(value) != set(expected) | {"applied_count"} or
            type(value.get("applied_count")) is not int or value["applied_count"] < 1 or
            any(type(value[key]) is not type(required) or value[key] != required
                for key, required in expected.items())):
        raise ValueError("vLLM actual Qwen startup adapter application is unproven")
    return value


def guard_gate(evidence, readiness, required, now, *, serving_profile="vllm-v1", model=MODEL,
               expected_policy=None):
    """Operator-observed guard expiry, bound to freshly paired live API identity."""
    if serving_profile in TRAINING_PROFILES:
        key = "native_controller" if serving_profile == NATIVE_PROFILE else "training_controller"
        if (evidence.get("schema_version") != 2 or evidence.get("serving_profile") != serving_profile or
                evidence.get("model") != model):
            raise ValueError("Invalid native model guard snapshot")
        live = validate_training_controller(readiness["gpu"].get(key), model, serving_profile=serving_profile,
            now=now, required_seconds=required, expected_policy=expected_policy)
        snapshot = validate_training_controller(evidence.get(key), model, serving_profile=serving_profile,
            now=now, required_seconds=required, expected_policy=expected_policy,
            require_running=False, require_fresh=False)
        if training_controller_identity(snapshot) != training_controller_identity(live):
            raise ValueError("Native model guard differs from freshly paired process or loaded policy")
        return {"serving_profile": serving_profile, "expires_unix": live["expires_unix"],
                "remaining_seconds": live["expires_unix"] - now, "policy_ack": live["policy_ack"],
                "generation_slots": 1,
                "scope": "authenticated live native controller and actual owned process; finite service lifetime"}
    if serving_profile != "vllm-v1" or model != MODEL:
        raise ValueError("Unsupported model guard profile")
    if evidence.get("schema_version") != 1 or evidence.get("model") != MODEL:
        raise ValueError("Invalid model guard observation")
    gpu = readiness["gpu"]
    for key in ("model_pid", "model_pid_start_ticks", "model_cmdline_sha256"):
        if evidence.get(key) != gpu.get(key) or key not in evidence:
            raise ValueError("Model guard observation does not match live paired API identity")
    if not HASH.fullmatch(str(evidence["model_cmdline_sha256"])):
        raise ValueError("Invalid model command hash")
    if any(type(evidence.get(key)) is not int or evidence[key] <= 0 for key in
           ("model_pid", "model_pid_start_ticks", "guard_pid", "guard_pid_start_ticks", "guard_max_seconds")):
        raise ValueError("Invalid observed guard process identity")
    started = positive(evidence.get("guard_started_unix"), "guard start")
    observed = positive(evidence.get("observed_unix"), "guard observation")
    maximum = evidence["guard_max_seconds"]
    if not 1 <= maximum <= 86400 or not started <= observed <= now + 1 or now - observed > 86400:
        raise ValueError("Invalid observed guard lifetime")
    expires = started + maximum
    if expires - now < required:
        raise ValueError("Model guard remaining lifetime cannot cover unchanged task budgets")
    return {"expires_unix": expires, "remaining_seconds": expires - now,
            "scope": "operator-observed guard; fresh relay proves API identity, not guard liveness"}
