"""Private, bounded actor process owning one vLLM engine and its workers."""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import socket
import struct
import time

if __name__ == "__mp_main__" and os.environ.get("_DSV41_LORA_IPC_ACTIVE") == "1":
    from .vllm_lora_ipc import configure_lora_ipc
    configure_lora_ipc()

from .vllm_worker import process_identity, VLLM_KERNEL_CONFIG, VLLM_WORKER_EXTENSION, VLLM_WORKER_RPC


MAX_RPC_BYTES = 2 * 1024**2
MAX_LORA_BATCH_RPC_BYTES = 32 * 1024**2


def validated_actor_ttl(value):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not 1 <= value <= 172800 or int(value) != value):
        raise ValueError("owned actor TTL must be an integer number of seconds in 1..172800")
    return int(value)


def actor_lifetime_deadline(bootstrap, *, now_unix=None, now_monotonic=None):
    """Keep legacy actor TTLs and an optional overall LoRA deadline."""
    ttl = validated_actor_ttl(bootstrap["ttl_seconds"])
    now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    if bootstrap.get("serving_profile") != "deepseek-lora-v1":
        return now_monotonic + ttl
    overall = bootstrap.get("deadline_unix")
    if overall is None:
        return None
    if type(overall) not in (int, float) or not math.isfinite(overall):
        raise ValueError("LoRA actor lifetime requires a finite deadline or None")
    now_unix = time.time() if now_unix is None else now_unix
    return now_monotonic + max(0, overall - now_unix)


def read_exact(stream, length):
    parts = []
    remaining = length
    while remaining:
        part = stream.recv(remaining)
        if not part:
            raise EOFError("incomplete owned actor RPC")
        parts.append(part)
        remaining -= len(part)
    return b"".join(parts)


def receive(stream, *, max_bytes=MAX_RPC_BYTES):
    length = struct.unpack("!I", read_exact(stream, 4))[0]
    if not 0 < length <= max_bytes:
        raise ValueError("owned actor RPC exceeds byte budget")
    return json.loads(read_exact(stream, length))


def send(stream, value, *, max_bytes=MAX_RPC_BYTES):
    data = json.dumps(value, allow_nan=False, separators=(",", ":")).encode()
    if not 0 < len(data) <= max_bytes:
        raise ValueError("owned actor response exceeds byte budget")
    stream.sendall(struct.pack("!I", len(data)) + data)


def output_tokens(output, prompt_ids, eos_ids):
    """Read exact raw token IDs and processed sampled probabilities, including EOS."""
    if not output.finished or output.prompt_token_ids != prompt_ids or len(output.outputs) != 1:
        raise ValueError("vLLM request output is unfinished or its actual prompt differs")
    choice = output.outputs[0]
    ids = list(choice.token_ids)
    probabilities = choice.logprobs
    if not ids or probabilities is None or len(probabilities) != len(ids):
        raise ValueError("vLLM did not return aligned actual sampled token probabilities")
    values = []
    for token, row in zip(ids, probabilities):
        if type(token) is not int or token < 0 or row is None or token not in row:
            raise ValueError("vLLM sampled token is absent from its raw probability row")
        value = float(row[token].logprob)
        if not math.isfinite(value) or value > 1e-6:
            raise ValueError("vLLM sampled probability is invalid")
        values.append(value)
    if choice.finish_reason not in {"stop", "length"}:
        raise ValueError("vLLM returned an unsupported sampling termination")
    stop_count = int(choice.finish_reason == "stop" and ids[-1] in eos_ids)
    return {"context_token_ids": list(output.prompt_token_ids), "action_token_ids": ids,
            "behavior_log_probs": values, "finish_reason": "eos" if stop_count else choice.finish_reason,
            "stop_token_count": stop_count, "logprob_convention": "sampled_distribution"}


class ActorEngine:
    def __init__(self, bootstrap):
        import vllm
        import vllm.envs
        from vllm import LLM
        from vllm.config.compilation import CompilationConfig, CUDAGraphMode
        if vllm.__version__ != "0.31.0" or vllm.envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            raise RuntimeError("owned actor requires the qualified vLLM 0.31.0 runtime")
        self.bootstrap = bootstrap
        self.ack = self.activation = None
        started = time.monotonic()
        self.llm = LLM(model=bootstrap["model_dir"], tokenizer=bootstrap["model_dir"],
            skip_tokenizer_init=True, tensor_parallel_size=1, dtype=bootstrap["dtype"],
            quantization=None, seed=bootstrap["seed"], trust_remote_code=False,
            gpu_memory_utilization=.70, kv_cache_memory_bytes=bootstrap["kv_cache_memory_bytes"],
            max_model_len=bootstrap["max_model_len"], max_num_seqs=1,
            enable_prefix_caching=False, enforce_eager=False, logprobs_mode="processed_logprobs",
            kernel_config=dict(VLLM_KERNEL_CONFIG),
            worker_extension_cls=VLLM_WORKER_EXTENSION,
            compilation_config=CompilationConfig(mode=0, cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
                                                  cudagraph_capture_sizes=[1]),
            disable_log_stats=True, distributed_executor_backend="mp")
        self.startup_seconds = time.monotonic() - started
        self.workers = self.llm.collective_rpc(VLLM_WORKER_RPC, args=("observe",), timeout=120)
        if len(self.workers) != 1 or self.workers[0]["graph"]["captured_graph_count"] < 1:
            raise RuntimeError("owned vLLM engine did not establish actual CUDA Graph capture")
        self.source_hashes = {}
        for name in ("vllm", "vllm.v1.sample.sampler", "vllm.compilation.cuda_graph"):
            import importlib
            module = importlib.import_module(name)
            self.source_hashes[name] = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()

    def receipt(self):
        self.workers = self.llm.collective_rpc(VLLM_WORKER_RPC, args=("stats",), timeout=120)
        if len(self.workers) != 1 or self.workers[0]["kernel_config"] != VLLM_KERNEL_CONFIG:
            raise RuntimeError("actual worker kernel configuration changed")
        return {"schema_version": 1, "serving_profile": "vllm-training-v1", "version": "0.31.0",
            "tensor_parallel_size": 1, "logprobs_mode": "processed_logprobs", "prefix_caching": False,
            "enforce_eager": False, "cudagraph_capture_sizes": [1], "workers": self.workers,
            "compilation_mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY",
            "kernel_config": self.workers[0]["kernel_config"],
            "startup_adapter": self.workers[0]["startup_adapter"],
            "worker_rpc_method": self.workers[0]["worker_rpc_method"],
            "insecure_serialization": self.workers[0]["insecure_serialization"],
            "actor_process": process_identity(), "engine_startup_seconds": self.startup_seconds,
            "engine_startup_includes_compile_warmup_capture": True,
            "kv_cache_memory_bytes": self.bootstrap["kv_cache_memory_bytes"],
            "kv_bytes_per_token": self.bootstrap["kv_bytes_per_token"],
            "full_context_raw_kv_bytes": self.bootstrap["kv_bytes_per_token"] * self.bootstrap["max_model_len"],
            "gpu_memory_utilization": .70, "max_model_len": self.bootstrap["max_model_len"],
            "source_sha256": self.source_hashes, "loaded_policy_ack": self.ack,
            "loaded_trainable_tensors": ({**self.activation["loaded_trainable_tensors"],
                "shape": self.activation["shape"], "dtype": self.activation["dtype"]} if self.activation else None)}

    def load(self, artifact):
        receipts = self.llm.collective_rpc(VLLM_WORKER_RPC, args=("load_head", artifact), timeout=300)
        expected = {key: artifact[key] for key in ("version", "digest", "lineage")}
        if len(receipts) != 1 or any(item["policy_ack"] != expected for item in receipts):
            raise RuntimeError("vLLM workers did not acknowledge the exact actual head artifact")
        if self.llm.reset_prefix_cache() is not True:
            raise RuntimeError("vLLM did not acknowledge prefix cache invalidation")
        version = json.dumps(expected, sort_keys=True, separators=(",", ":"))
        self.llm.update_weight_version(version)
        if self.llm.get_weight_version() != version:
            raise RuntimeError("vLLM weight version differs from the actual loaded artifact")
        self.ack, self.activation = expected, receipts[0]
        return {"ack": expected, "activation_receipt": receipts[0], "engine": self.receipt()}

    def generate(self, request):
        from vllm import SamplingParams
        if self.ack != request["policy_ack"]:
            raise RuntimeError("vLLM request does not match its actually loaded head")
        prompt = request["prompt_ids"]
        budget = request["max_tokens"]
        if (not prompt or any(type(token) is not int or token < 0 for token in prompt)
                or type(budget) is not int or budget < 1
                or len(prompt) + budget > self.bootstrap["max_model_len"]):
            raise ValueError("invalid or truncated vLLM context/action budget")
        before = self.receipt()
        params = SamplingParams(n=1, temperature=request["temperature"], top_p=1., top_k=-1,
            min_p=0., presence_penalty=0., frequency_penalty=0., repetition_penalty=1.,
            seed=request["seed"], max_tokens=budget, logprobs=0,
            stop_token_ids=[] if request.get("ignore_eos", False) else self.bootstrap["eos_token_ids"],
            # Stop only on the resolved configured IDs. vLLM otherwise unions
            # model/generation EOS defaults into an explicit override or [].
            ignore_eos=True,
            detokenize=False)
        started = time.monotonic()
        outputs = self.llm.generate({"prompt_token_ids": prompt}, params, use_tqdm=False)
        elapsed = time.monotonic() - started
        if len(outputs) != 1:
            raise RuntimeError("vLLM must return one actual request output")
        actual = output_tokens(outputs[0], prompt, self.bootstrap["eos_token_ids"])
        after = self.receipt()
        old, new = before["workers"][0], after["workers"][0]
        if any(old[key] != new[key] for key in ("pid", "start_ticks", "cmdline_sha256", "device_uuid")):
            raise RuntimeError("vLLM worker identity changed during a request")
        raw_metrics = getattr(outputs[0], "metrics", None)
        timestamps = {name: getattr(raw_metrics, name, None) for name in
            ("queued_ts", "scheduled_ts", "first_token_ts", "last_token_ts")}
        measured = all(isinstance(value, (int, float)) and math.isfinite(value) and value > 0
                       for value in timestamps.values())
        actual["generation_metrics"] = {"elapsed_seconds": elapsed, "context_tokens": len(prompt),
            "action_tokens": len(actual["action_token_ids"]), "clock_domain": "vllm_actor_monotonic",
            "completion_barrier": "finished_vllm_RequestOutput_tokens_and_logprobs_on_cpu",
            "prefill_decode_separation": "engine_core_monotonic" if measured else "not_measured",
            "engine_core_timestamps": timestamps if measured else None,
            "engine_prefill_seconds": timestamps["first_token_ts"] - timestamps["scheduled_ts"] if measured else None,
            "engine_decode_seconds": timestamps["last_token_ts"] - timestamps["first_token_ts"] if measured else None,
            "engine_queue_seconds": timestamps["scheduled_ts"] - timestamps["queued_ts"] if measured else None,
            "sampling_seed": request["seed"],
            "sampling_rng": "vllm_engine_seeded_sampler", "graph_captured_count": new["graph"]["captured_graph_count"],
            "graph_replay_calls_before": old["graph"]["replay_calls"],
            "graph_replay_calls_after": new["graph"]["replay_calls"],
            "graph_replay_calls_delta": new["graph"]["replay_calls"] - old["graph"]["replay_calls"],
            "worker_pid": new["pid"], "device_uuid": new["device_uuid"], "engine_receipt": after}
        return actual

    def close(self):
        self.llm.llm_engine.engine_core.shutdown(timeout=60)


def lora_batch_id(specs, ack):
    """Bind the complete ordered sampling inputs to their immutable policy."""
    data = json.dumps({"specs": specs, "policy_ack": ack}, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()[:32]


class LoRAActorEngine(ActorEngine):
    """Explicit DeepSeek profile using stock LoRA slots and direct tensor RPC."""
    worker_rpc = "rollout_manager_lora_call"
    worker_extension = "rollout_manager.backends.vllm_lora_worker.DeepseekLoRAExtension"

    def __init__(self, bootstrap):
        from .vllm_lora_ipc import configure_lora_ipc
        configure_lora_ipc()
        import vllm
        import vllm.envs
        from vllm import LLM
        from .deepseek_v41_lora import (
            lora_target_modules, register_deepseek_v41_lora,
        )
        from dsv41_train.rl.lora_config import DistributedLoRAConfig
        if vllm.__version__ != "0.31.0" or vllm.envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
            raise RuntimeError("direct LoRA actor requires the inspected secure vLLM 0.31.0 runtime")
        from .vllm_batch_admission import qualify_admission_runtime
        qualify_admission_runtime(vllm, LLM)
        self.bootstrap = bootstrap
        self.lora_config = DistributedLoRAConfig.from_dict(bootstrap["lora_config"])
        self.text_config = json.loads((Path(bootstrap["model_dir"]) / "config.json").read_text())
        self.ack = self.activation = self.adapter_request = None
        self.failed = False
        self._used_adapter_ids = set()
        register_deepseek_v41_lora()
        self.llm = LLM(model=bootstrap["model_dir"], tokenizer=bootstrap["model_dir"],
            skip_tokenizer_init=False, tensor_parallel_size=bootstrap["tensor_parallel_size"],
            pipeline_parallel_size=1, dtype="bfloat16", seed=bootstrap["seed"], trust_remote_code=False,
            gpu_memory_utilization=bootstrap["gpu_memory_utilization"],
            max_model_len=bootstrap["max_model_len"], max_num_seqs=bootstrap["max_num_seqs"],
            limit_mm_per_prompt={"image": 0},
            enable_prefix_caching=False, enforce_eager=False, logprobs_mode="processed_logprobs",
            enable_lora=True, max_loras=1, max_cpu_loras=2, max_lora_rank=self.lora_config.rank,
            lora_dtype="bfloat16", lora_target_modules=lora_target_modules(
                self.text_config, self.lora_config),
            additional_config={"rollout_manager_lora": self.lora_config.to_dict()},
            fully_sharded_loras=False,
            worker_extension_cls=self.worker_extension, disable_log_stats=False,
            distributed_executor_backend="mp")
        from vllm.v1.engine.core_client import SyncMPClient
        if not isinstance(self.llm.llm_engine.engine_core, SyncMPClient):
            self.llm.llm_engine.engine_core.shutdown(timeout=60)
            raise RuntimeError("atomic rollout admission requires the inspected synchronous MP FIFO client")
        self.workers = self.observe()
        self._check_workers(self.workers)

    def observe(self):
        return self.llm.collective_rpc(self.worker_rpc, args=("stats",), timeout=120)

    def _check_workers(self, workers, manifest=None):
        from .vllm_lora_transport import policy_ack
        if len(workers) != self.bootstrap["tensor_parallel_size"]:
            raise RuntimeError("direct LoRA activation requires an acknowledgement from every TP worker")
        uuids = [worker.get("device_uuid") for worker in workers]
        if any(not value for value in uuids) or len(set(uuids)) != len(uuids):
            raise RuntimeError("direct LoRA workers did not identify distinct physical GPUs")
        for worker in workers:
            if worker.get("base_identity") != self.bootstrap["base_identity"]:
                raise RuntimeError("direct LoRA worker mounted base identity differs")
            if worker.get("lora_config") != self.lora_config.to_dict():
                raise RuntimeError("direct LoRA worker configuration differs from the actor")
            if manifest is not None and (worker.get("policy_ack") != policy_ack(manifest)
                    or worker.get("adapter_id") != manifest["adapter_id"]
                    or worker.get("readback_digest") != manifest["digest"]):
                raise RuntimeError("direct LoRA worker did not acknowledge the actual adapter readback")

    def receipt(self):
        if self.failed:
            raise RuntimeError("direct LoRA actor is permanently failed until close")
        self.workers = self.observe()
        self._check_workers(self.workers, self.activation)
        return {"serving_profile": "deepseek-lora-v1", "version": "0.31.0",
            "tensor_parallel_size": self.bootstrap["tensor_parallel_size"], "workers": self.workers,
            "base_identity": self.bootstrap["base_identity"], "loaded_policy_ack": self.ack,
            "lora_config": self.lora_config.to_dict(),
            "adapter_id": self.activation["adapter_id"] if self.activation else None,
            "max_model_len": self.bootstrap["max_model_len"], "max_num_seqs": self.bootstrap["max_num_seqs"]}

    def batch_capacity(self):
        # vLLM 0.31 returns initialized, group-aware capacity to the client.
        # num_gpu_blocks * block_size is not valid for hybrid KV cache groups.
        config = getattr(getattr(self.llm, "llm_engine", None), "vllm_config", None)
        cache = getattr(config, "cache_config", None)
        scheduler = getattr(config, "scheduler_config", None)
        return {"source": "vllm_initialized_cache_config_group_aware",
            "kv_cache_size_tokens": getattr(cache, "kv_cache_size_tokens", None),
            "kv_cache_max_concurrency": getattr(cache, "kv_cache_max_concurrency", None),
            "effective_attention_block_size": getattr(cache, "effective_attention_block_size", None),
            "max_num_batched_tokens": getattr(scheduler, "max_num_batched_tokens", None)}

    def load(self, artifact):
        raise ValueError("DeepSeek LoRA activation accepts direct tensor frames, never checkpoint artifacts")

    def load_lora(self, manifest, payload):
        from vllm.lora.request import LoRARequest
        from .vllm_lora_transport import policy_ack, validate_update
        from .deepseek_v41_lora import validate_adapter_manifest
        try:
            if self.failed:
                raise RuntimeError("direct LoRA actor is permanently failed until close")
            validate_update(manifest, payload)
            validate_adapter_manifest(manifest, self.text_config, self.lora_config)
            expected_version = 0 if self.ack is None else self.ack["version"] + 1
            if (manifest["version"] != expected_version or manifest["adapter_id"] in self._used_adapter_ids
                    or self.ack is not None and manifest["lineage"] != self.ack["lineage"]
                    or manifest["base_identity"] != self.bootstrap["base_identity"]):
                raise ValueError("direct LoRA policy must advance one version with an unused ID and identical base/profile")
            receipts = self.llm.collective_rpc(self.worker_rpc,
                args=("load_lora", manifest, payload), timeout=300)
            self._check_workers(receipts, manifest)
            expected = policy_ack(manifest)
            if self.llm.reset_prefix_cache() is not True:
                raise RuntimeError("vLLM did not acknowledge cache invalidation after LoRA activation")
            version = json.dumps(expected, sort_keys=True, separators=(",", ":"))
            self.llm.update_weight_version(version)
            if self.llm.get_weight_version() != version:
                raise RuntimeError("vLLM engine weight version differs from adapter activation")
            adapter_request = LoRARequest(lora_name=version, lora_int_id=manifest["adapter_id"],
                                         lora_path="memory://" + manifest["digest"])
            self.ack, self.activation, self.adapter_request = expected, manifest, adapter_request
            self._used_adapter_ids.add(manifest["adapter_id"])
            return {"ack": expected, "adapter_id": manifest["adapter_id"],
                    "activation_receipts": receipts, "engine": self.receipt()}
        except BaseException:
            self.failed, self.ack = True, None
            raise

    def generate(self, request):
        spec = {"prompt_ids": request["prompt_ids"], "max_tokens": request["max_tokens"],
            "temperature": request["temperature"], "sampling_seed": request["seed"],
            "request_id": request.get("request_id", lora_batch_id([request], request["policy_ack"]))}
        reply = self.generate_batch({"specs": [spec], "policy_ack": request["policy_ack"],
            "adapter_id": request["adapter_id"], "batch_id": lora_batch_id([spec], request["policy_ack"])})
        return reply["results"][0]

    def generate_batch(self, request):
        from vllm import SamplingParams
        from .vllm_lora_remote import _specs
        if (self.failed or self.ack is None or request.get("policy_ack") != self.ack
                or request.get("adapter_id") != self.activation["adapter_id"]):
            raise RuntimeError("LoRA request does not match the actually activated immutable adapter")
        specs = _specs(request["specs"])
        if (request.get("batch_id") != lora_batch_id(specs, self.ack)
                or len(specs) > self.bootstrap["max_num_seqs"]):
            raise ValueError("LoRA batch identity or max_num_seqs capacity differs; no serial fallback")
        if any(len(spec["prompt_ids"]) + spec["max_tokens"] > self.bootstrap["max_model_len"] for spec in specs):
            raise ValueError("invalid direct LoRA sampling/context budget")
        capacity = self.batch_capacity()
        tokens, concurrency = capacity["kv_cache_size_tokens"], capacity["kv_cache_max_concurrency"]
        reserved = sum(len(spec["prompt_ids"]) + spec["max_tokens"] for spec in specs)
        if len(specs) > 1 and (type(tokens) is not int or tokens < reserved
                or type(concurrency) not in (int, float) or not math.isfinite(concurrency)
                or concurrency < len(specs)):
            raise ValueError("LoRA batch requires verified full-context KV capacity; no serial fallback")
        prompt_tokens = sum(len(spec["prompt_ids"]) for spec in specs)
        if len(specs) > 1 and (type(capacity["max_num_batched_tokens"]) is not int
                or capacity["max_num_batched_tokens"] < prompt_tokens):
            raise ValueError("LoRA batch requires capacity for every full prompt in its first schedule")
        before = self.receipt()
        params = [SamplingParams(n=1, temperature=spec["temperature"], top_p=1., top_k=-1, min_p=0.,
            presence_penalty=0., frequency_penalty=0., repetition_penalty=1., seed=spec["sampling_seed"],
            max_tokens=spec["max_tokens"], logprobs=0, stop_token_ids=self.bootstrap["eos_token_ids"],
            ignore_eos=True, detokenize=False) for spec in specs]
        try:
            from .vllm_batch_admission import enqueue_before_schedule
            prompts = [{"prompt_token_ids": spec["prompt_ids"]} for spec in specs]
            if len(specs) > 1:
                outputs = enqueue_before_schedule(self.llm, prompts, params, self.adapter_request)
            else:
                outputs = self.llm.generate(prompts, params, use_tqdm=False, lora_request=self.adapter_request)
            if len(outputs) != len(specs):
                raise RuntimeError("direct LoRA batch expected every request output")
            engine_ids = [getattr(output, "request_id", None) for output in outputs]
            if (any(not isinstance(value, str) or not value.isdecimal() for value in engine_ids)
                    or len(set(engine_ids)) != len(specs)
                    or list(map(int, engine_ids)) != sorted(map(int, engine_ids))):
                raise RuntimeError("vLLM batch output request identity/order differs")
            actual = [output_tokens(output, spec["prompt_ids"], self.bootstrap["eos_token_ids"])
                      for spec, output in zip(specs, outputs)]
            if any(len(result["action_token_ids"]) > spec["max_tokens"] for spec, result in zip(specs, actual)):
                raise RuntimeError("vLLM batch output exceeded its admitted token budget")
            after = self.receipt()
            identities = ("pid", "start_ticks", "cmdline_sha256", "device_uuid")
            if (after["loaded_policy_ack"] != before["loaded_policy_ack"]
                    or any(any(old.get(key) != new.get(key) for key in identities)
                           for old, new in zip(before["workers"], after["workers"]))):
                raise RuntimeError("LoRA policy or worker identity changed during batch inference")
            for index, (spec, result) in enumerate(zip(specs, actual)):
                result["generation_metrics"] = {
                    "request_id": spec["request_id"], "sampling_seed": spec["sampling_seed"],
                    "engine_request_id": engine_ids[index], "batch_request_index": index,
                    "policy_ack": dict(self.ack), "adapter_id": self.activation["adapter_id"],
                    "adapter_request_id": self.adapter_request.lora_int_id}
            return {"results": actual, "batch_id": request["batch_id"], "engine_receipt": after}
        except BaseException:
            # A partial or unknown outcome must never be automatically replayed.
            self.failed = True
            raise

    def close(self):
        self.llm.llm_engine.engine_core.shutdown(timeout=60)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    args = parser.parse_args()
    token = args.token_file.read_text().strip()
    bootstrap = json.loads(args.bootstrap.read_text())
    is_lora = bootstrap.get("serving_profile") == "deepseek-lora-v1"
    if is_lora:
        ttl = validated_actor_ttl(bootstrap["ttl_seconds"])
        # Validate lifetime bounds before constructing the model.
        actor_lifetime_deadline(bootstrap)
    actor = LoRAActorEngine(bootstrap) if is_lora else ActorEngine(bootstrap)
    deadline = actor_lifetime_deadline(bootstrap) if is_lora else time.monotonic() + min(30000, bootstrap["ttl_seconds"])
    closed = False
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(args.socket))
        os.chmod(args.socket, 0o600)
        server.listen(1)
        server.settimeout(1)
        try:
            while not closed and (deadline is None or time.monotonic() < deadline):
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    connection.settimeout(min(600, ttl) if is_lora else 600)
                    operation = None
                    try:
                        request = receive(connection)
                        if not hmac.compare_digest(str(request.pop("token", "")), token):
                            raise ValueError("unauthorized owned actor request")
                        operation = request.pop("operation")
                        if operation == "stats":
                            result = actor.receipt()
                        elif operation == "load":
                            result = actor.load(request["artifact"])
                        elif operation == "load_lora" and is_lora:
                            from .vllm_lora_transport import receive_update
                            manifest, payload = receive_update(connection)
                            result = actor.load_lora(manifest, payload)
                        elif operation == "generate":
                            result = actor.generate(request)
                        elif operation == "generate_batch" and is_lora:
                            result = actor.generate_batch(request)
                        elif operation == "close":
                            actor.close()
                            closed = True
                            result = {"closed": True, "drain_complete": True}
                        else:
                            raise ValueError("unsupported actor operation")
                        send(connection, {"ok": True, "result": result},
                             max_bytes=MAX_LORA_BATCH_RPC_BYTES if is_lora and operation == "generate_batch" else MAX_RPC_BYTES)
                    except BaseException as error:
                        # No traceback, arguments or credentials enter RPC evidence.
                        if is_lora and operation in {"generate", "generate_batch"}:
                            actor.failed = True
                        send(connection, {"ok": False, "error_type": type(error).__name__,
                                          "error": str(error)[:1000]})
                        if operation in {"load", "load_lora", "close"}:
                            raise
        finally:
            if not closed:
                actor.close()


if __name__ == "__main__":
    main()
