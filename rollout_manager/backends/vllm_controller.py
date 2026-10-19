"""Owned vLLM rollout actor, exact behavior evidence and drained head updates."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import tempfile
import time

from .pytorch_controller import PyTorchPolicyController, _REQUEST_ID, _owned_thread
from .vllm_actor import (receive, send, validated_actor_ttl, actor_lifetime_deadline,
                         MAX_RPC_BYTES, MAX_LORA_BATCH_RPC_BYTES)
from .vllm_profile import (VLLM_TRAINING_PROFILE, require_vllm_learner, vllm_policy_identities,
                           VLLM_KV_CACHE_MEMORY_BYTES)
from ..generation_training import TrainingGenerationContract, evidence_digest
from ..generation_failure import validate_generation_failure
from ..model_completion import (ModelCompletionFormatError, invalid_tool_status,
                                validate_model_completion_status, shell_completion)


def request_seed(base_seed, request_id, policy):
    value = [base_seed, request_id, policy.version, policy.digest, policy.lineage]
    return int.from_bytes(hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).digest()[:8], "big") % (2**63 - 1)


def check_process(receipt):
    root = Path("/proc") / str(receipt["pid"])
    fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
    if (fields[0] in {"Z", "X"} or int(fields[19]) != receipt["start_ticks"]
            or hashlib.sha256((root / "cmdline").read_bytes()).hexdigest() != receipt["cmdline_sha256"]):
        raise RuntimeError("owned vLLM process identity changed or exited")


class OwnedVLLMActor:
    """One fresh subprocess, private Unix socket, bounded RPC and exact ownership."""
    def __init__(self, model_dir, config, directory, *, max_model_len, bootstrap_override=None):
        from dsv41_train.rl.family import load_model_config
        from dsv41_train.rl.rollout import generation_options
        directory.mkdir(mode=0o700, parents=False, exist_ok=False)
        self.directory = directory
        self.token = secrets.token_hex(32)
        self.token_file = directory / "actor-auth.secret"
        self.token_file.write_text(self.token + "\n")
        os.chmod(self.token_file, 0o600)
        # LoRA can keep socket names short independently of compiler TMPDIR.
        ipc_directory = None
        if bootstrap_override is None:
            self.socket_dir = Path(tempfile.mkdtemp(prefix="rl-vllm-"))
        else:
            from .vllm_lora_ipc import create_lora_socket_directory
            self.socket_dir, ipc_directory = create_lora_socket_directory()
        os.chmod(self.socket_dir, 0o700)
        self.socket_path = self.socket_dir / "actor.sock"
        actor_ttl = 30000
        if bootstrap_override is None:
            generation = generation_options(model_dir, load_model_config(model_dir), config)
            if generation.get("stop_token_sequences"):
                raise ValueError("vLLM actor does not approximate token stop sequences with text stops")
            eos = generation["eos_token_id"]
            eos_ids = [] if eos is None else [eos] if isinstance(eos, int) else list(eos)
            model_config = load_model_config(model_dir)
            kv_bytes_per_token = 2 * model_config.num_hidden_layers * model_config.num_key_value_heads * model_config.head_dim * 2
            if kv_bytes_per_token * max_model_len > VLLM_KV_CACHE_MEMORY_BYTES:
                raise ValueError("fixed vLLM KV memory cannot preserve the mounted full context")
            bootstrap = {"model_dir": str(model_dir), "dtype": config.dtype, "seed": config.seed,
                         "max_model_len": max_model_len, "eos_token_ids": eos_ids, "ttl_seconds": 30000,
                         "kv_bytes_per_token": kv_bytes_per_token,
                         "kv_cache_memory_bytes": VLLM_KV_CACHE_MEMORY_BYTES}
        else:
            bootstrap = copy.deepcopy(bootstrap_override)
            if (bootstrap.get("serving_profile") != "deepseek-lora-v1"
                    or bootstrap.get("model_dir") != str(model_dir)):
                raise ValueError("invalid explicit LoRA actor bootstrap")
            actor_ttl = validated_actor_ttl(bootstrap["ttl_seconds"])
            actor_lifetime_deadline(bootstrap)
        bootstrap_file = directory / "bootstrap.json"
        bootstrap_file.write_text(json.dumps(bootstrap, indent=2) + "\n")
        os.chmod(bootstrap_file, 0o600)
        index = int(config.rollout_devices[0].split(":")[1])
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        devices = visible.split(",") if visible else []
        if bootstrap_override is None:
            selected = devices[index] if devices else str(index)
        else:
            indices = [int(device.split(":")[1]) for device in config.rollout_devices]
            if len(indices) != bootstrap["tensor_parallel_size"] or len(set(indices)) != len(indices):
                raise ValueError("LoRA devices differ from tensor parallel size")
            selected = ",".join(devices[item] if devices else str(item) for item in indices)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=selected,
            HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
            VLLM_ALLOW_INSECURE_SERIALIZATION="0")
        if bootstrap_override is not None:
            # Every spawned worker must import this exact task-owned source copy.
            source_root = str(Path(__file__).resolve().parents[2])
            environment["PYTHONPATH"] = source_root + (os.pathsep + environment["PYTHONPATH"]
                if environment.get("PYTHONPATH") else "")
            environment["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
            if ipc_directory is not None:
                environment["VLLM_RPC_BASE_PATH"] = ipc_directory
                environment["_DSV41_LORA_IPC_ACTIVE"] = "1"
        else:
            environment.pop("_DSV41_LORA_IPC_ACTIVE", None)
        self.log = (directory / "actor.log").open("xb")
        self.process = subprocess.Popen([sys.executable, "-u", "-m", "rollout_manager.backends.vllm_actor",
            "--bootstrap", str(bootstrap_file), "--socket", str(self.socket_path),
            "--token-file", str(self.token_file)], env=environment, stdin=subprocess.DEVNULL,
            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        root = Path("/proc") / str(self.process.pid)
        self.identity = {"pid": self.process.pid,
            "start_ticks": int((root / "stat").read_text().rsplit(")", 1)[1].split()[19]),
            "cmdline_sha256": hashlib.sha256((root / "cmdline").read_bytes()).hexdigest()}
        from ..terminal_bench.controller import write_new
        write_new(directory / "actor-launch.json", {"schema_version": 1, "serving_profile": bootstrap.get("serving_profile", VLLM_TRAINING_PROFILE),
            "actor_process": self.identity, "python_executable": sys.executable,
            "selected_cuda_device": selected, "started_unix": time.time(), "ttl_seconds": actor_ttl,
            "ttl_scope": "startup_and_operation_bound" if bootstrap_override is not None and bootstrap["deadline_unix"] is None else "actor_lifetime",
            "actor_lifetime_unlimited": bootstrap_override is not None and bootstrap["deadline_unix"] is None,
            "automatic_replay": False, "model_dir": str(model_dir)})
        deadline = time.monotonic() + min(1800 if bootstrap_override is not None else 900, actor_ttl)
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError("owned vLLM actor exited during startup; inspect its preserved actor.log")
                if self.socket_path.exists():
                    self.receipt = self.call("stats", timeout=120)
                    break
                time.sleep(.2)
            else:
                raise TimeoutError("owned vLLM actor startup exceeded its bounded deadline")
        except BaseException:
            self.stop_failed_start()
            raise

    def call(self, operation, *, timeout=600, tensor_update=None, **values):
        if operation == "load_lora":
            if tensor_update is None:
                raise ValueError("direct LoRA activation requires a manifest and raw tensor payload")
            from .vllm_lora_transport import validate_update
            validate_update(*tensor_update)
        elif tensor_update is not None:
            raise ValueError("raw tensors are admitted only for direct LoRA activation")
        check_process(self.identity)
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(timeout)
            connection.connect(str(self.socket_path))
            send(connection, {"token": self.token, "operation": operation, **values})
            if tensor_update is not None:
                from .vllm_lora_transport import send_update
                send_update(connection, *tensor_update)
            reply = receive(connection, max_bytes=MAX_LORA_BATCH_RPC_BYTES
                            if operation == "generate_batch" else MAX_RPC_BYTES)
        if operation != "close":
            check_process(self.identity)
        if reply.get("ok") is not True:
            raise RuntimeError("owned vLLM actor failed " + operation + ": " + reply.get("error_type", "unknown") +
                               "; " + reply.get("error", ""))
        return reply["result"]

    def stop_failed_start(self):
        if self.process.poll() is None:
            check_process(self.identity)
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                check_process(self.identity)
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=15)
        self.log.close()

    def close(self):
        from ..terminal_bench.controller import write_new
        try:
            closed = self.call("close", timeout=120)
            if closed != {"closed": True, "drain_complete": True}:
                raise RuntimeError("owned actor did not acknowledge actual engine shutdown")
            self.process.wait(timeout=120)
        except BaseException as error:
            # Bounded cleanup is restricted to our newly created process
            # group. A forced exit is preserved as termination evidence and
            # is never reported as a acknowledged inference drain.
            if self.process.poll() is None:
                check_process(self.identity)
                for worker in getattr(self, "receipt", {}).get("workers", []):
                    check_process(worker)
                    if os.getpgid(worker["pid"]) != self.process.pid:
                        raise RuntimeError("owned worker no longer belongs to the actor process group") from error
                self.stop_failed_start()
            else:
                self.log.close()
            write_new(self.directory / "actor-exit.json", {"actor_process": self.identity,
                "returncode": self.process.returncode, "drain_complete": False,
                "forced_termination": True, "close_error_type": type(error).__name__,
                "finished_unix": time.time()})
            raise RuntimeError("owned actor close lacked an actual engine drain acknowledgement") from error
        self.log.close()
        write_new(self.directory / "actor-exit.json", {"actor_process": self.identity,
            "returncode": self.process.returncode, "drain_complete": True, "finished_unix": time.time()})
        self.socket_path.unlink(missing_ok=True)
        self.socket_dir.rmdir()
        if self.process.returncode != 0:
            raise RuntimeError("owned vLLM actor exited unsuccessfully")


class VLLMPolicyController(PyTorchPolicyController):
    serving_profile = VLLM_TRAINING_PROFILE

    def __init__(self, model_dir, config, *, served_model):
        super().__init__(model_dir, config, served_model=served_model)
        require_vllm_learner(config)
        if (config.trainable_scope != "lm_head" or not config.qwen_last_position_only
                or not config.qwen_kv_cache or config.backprop_chunk_tokens
                or len(config.rollout_devices) != 1 or not config.rollout_devices[0].startswith("cuda:")):
            raise ValueError("vLLM rollout requires one CUDA actor and frozen-backbone native head replay")
        self.identities.pop("qwen_sampling_path", None)
        self.identities.update(vllm_policy_identities(self.model_config.max_position_embeddings))
        self.last_activation_receipt = None
        self._engine_receipt = None
        self._actor_failure = None
        self._request_sampling = {}
        self.max_model_len = self.model_config.max_position_embeddings

    @staticmethod
    def _capabilities(policy):
        return {**PyTorchPolicyController._capabilities(policy), "serving_profile": VLLM_TRAINING_PROFILE}

    def _check_artifact(self, artifact):
        if any(artifact["identity"].get(key) != value for key, value in
               vllm_policy_identities(self.model_config.max_position_embeddings).items()):
            raise ValueError("artifact differs from the explicit vLLM behavior/native learner profiles")
        return super()._check_artifact(artifact)

    def _load(self, artifact, *, initial=False):
        artifact = copy.deepcopy(artifact)
        candidate = self._check_artifact(artifact)
        try:
            if initial:
                if self.diagnostics_dir is None:
                    raise ValueError("owned vLLM actor requires a private evidence directory")
                self.model = OwnedVLLMActor(self.model_dir, self.config,
                    self.diagnostics_dir / "vllm-actor", max_model_len=self.max_model_len)
            result = self.model.call("load", artifact=artifact)
            expected = {key: artifact[key] for key in ("version", "digest", "lineage")}
            if (result["ack"] != expected or result["activation_receipt"]["policy_ack"] != expected
                    or result["engine"]["loaded_policy_ack"] != expected):
                raise ValueError("actual vLLM worker activation differs from the immutable policy")
            self.last_activation_receipt = result["activation_receipt"]
            self._engine_receipt = result["engine"]
        except BaseException:
            with self._condition:
                self._state = "failed"
            raise
        with self._condition:
            self.policy, self.ack, self.active_artifact = candidate, expected, artifact
            self._state = "running" if initial else "paused"
        return dict(expected)

    def engine_receipt(self):
        with self._condition:
            if self._engine_receipt is None or self._state == "failed":
                raise RuntimeError("vLLM engine receipt requires actual startup and head activation")
            value = copy.deepcopy(self._engine_receipt)
        check_process(value["actor_process"])
        for worker in value["workers"]:
            check_process(worker)
        return value

    def loaded_trainable_tensor_digest(self):
        with self._condition:
            if self.last_activation_receipt is None:
                raise RuntimeError("vLLM head digest requires an actual worker readback")
            return copy.deepcopy(self.last_activation_receipt["loaded_trainable_tensors"])

    def loaded_trainable_tensors(self):
        return self.loaded_trainable_tensor_digest()

    def generation_failure_receipt(self, payload, request_id, stage):
        """Only an admitted binding and actual returned sampler prove closure."""
        with self._condition:
            binding = self._request_sampling.get(request_id)
            if binding is None:
                return None
            outcome = binding["sampler_outcome"]
            proof = {"not_admitted": "controller_rejected_before_sampler_dispatch",
                     "completed": "owned_vllm_final_request_output", "unknown": "none"}[outcome]
            receipt = {"schema_version": 1, "request_id": request_id,
                "request_payload_sha256": binding["request_payload_sha256"],
                "policy_ack": copy.deepcopy(binding["policy_ack"]), "stage": stage,
                "sampler_outcome": outcome, "sampler_drained": outcome != "unknown",
                "actual_actor_drain_proven": outcome != "unknown", "drain_scope": "request",
                "proof_kind": proof, "automatic_replay": False}
        return validate_generation_failure(receipt, payload, request_id)

    def _fail_generation(self, request_id, error, *, uncertain):
        # A returned Python thread does not establish that an actor request
        # finished. Failed RPCs retain an unknown remote outcome until the
        # owned engine actually acknowledges shutdown; they cannot admit an
        # activation, resume, or another generation.
        with self._condition:
            failure = {"schema_version": 1, "request_id": request_id,
                "serving_profile": VLLM_TRAINING_PROFILE, "observed_unix": time.time(),
                "error_type": type(error).__name__, "actor_outcome_uncertain": uncertain,
                "actual_actor_drain_proven": False, "automatic_replay": False,
                "policy_ack": copy.deepcopy(self.ack)}
            self._actor_failure = failure
            self._state, self._drained = "failed", False
            self._condition.notify_all()
        if self.diagnostics_dir is not None:
            from ..terminal_bench.controller import write_new
            write_new(self.diagnostics_dir / (request_id + ".actor-failure.json"), failure)

    def _wait_drained(self):
        super()._wait_drained()
        with self._condition:
            if self._actor_failure is not None or self._state == "failed":
                raise RuntimeError("failed vLLM actor outcome does not prove an inference drain")

    def _sample_locked(self, prompt, max_tokens, temperature, request_id, policy, *, ignore_eos=False, diagnostic_seed=None):
        seed = diagnostic_seed if diagnostic_seed is not None else request_seed(self.config.seed, request_id, policy)
        try:
            result = self.model.call("generate", prompt_ids=prompt, max_tokens=max_tokens,
                temperature=temperature, seed=seed, policy_ack={key: getattr(policy, key) for key in ("version", "digest", "lineage")},
                ignore_eos=ignore_eos)
        except BaseException as error:
            self._fail_generation(request_id, error, uncertain=True)
            raise
        from dsv41_train.rl.rollout import Sample
        validated = Sample(result["context_token_ids"], result["action_token_ids"], result["behavior_log_probs"],
            0., policy.version, result["finish_reason"], result["stop_token_count"])
        metrics = result["generation_metrics"]
        before, after, delta = (metrics.get("graph_replay_calls_" + key) for key in ("before", "after", "delta"))
        if (any(type(value) is not int or value < 0 for value in (before, after, delta))
                or after - before != delta or type(metrics.get("graph_captured_count")) is not int
                or metrics["graph_captured_count"] < 1 or len(validated.response_ids) > 1 and delta < 1):
            raise RuntimeError("actual vLLM CUDA Graph decode evidence is incomplete")
        if (result["context_token_ids"] != prompt or result["logprob_convention"] != "sampled_distribution"
                or metrics["engine_receipt"]["loaded_policy_ack"] != self.ack):
            raise RuntimeError("actual vLLM sampling evidence differs from its admitted prompt/policy")
        with self._condition:
            self._engine_receipt = metrics["engine_receipt"]
        return {"prompt_ids": result["context_token_ids"], "response_ids": result["action_token_ids"],
            "behavior_log_probs": result["behavior_log_probs"], "finish_reason": result["finish_reason"],
            "stop_token_count": result["stop_token_count"], "vllm_generation": metrics,
            "engine_receipt": metrics["engine_receipt"]}

    def sample_tokens(self, prompt_ids, max_tokens, temperature, request_id, *, ignore_eos=False,
                      diagnostic_seed=None, sampling_seed=None):
        """Raw actor sampling; evaluation may supply its independent engine seed."""
        if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
            raise ValueError("diagnostic requires a durable 32-hex request identity")
        if diagnostic_seed is not None and (not ignore_eos or type(diagnostic_seed) is not int
                or not 0 <= diagnostic_seed < 2**63 - 1):
            raise ValueError("independent diagnostic seed requires an explicit no-tool forced-token probe")
        if sampling_seed is not None and (diagnostic_seed is not None or type(sampling_seed) is not int
                or not 0 <= sampling_seed < 2**63 - 1):
            raise ValueError("sampling_seed must be an independent engine seed without diagnostic_seed")
        prompt = list(prompt_ids)
        with self._condition:
            if self._state != "running" or request_id in self._seen:
                raise RuntimeError("vLLM diagnostic admission is paused or request would replay")
            if (temperature != self.config.temperature or not prompt or type(max_tokens) is not int
                    or max_tokens < 1 or max_tokens > self.config.max_new_tokens
                    or len(prompt) + max_tokens > self.max_model_len):
                raise ValueError("vLLM diagnostic exceeds its profile or untruncated budget")
            policy = self.policy
            self._seen.add(request_id)
            self._active += 1
        try:
            with self._sampler_lock:
                return self._sample_locked(prompt, max_tokens, temperature, request_id, policy,
                    ignore_eos=ignore_eos,
                    diagnostic_seed=diagnostic_seed if sampling_seed is None else sampling_seed)
        except BaseException as error:
            if self._actor_failure is None:
                self._fail_generation(request_id, error, uncertain=False)
            raise
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def generate(self, payload, request_id):
        from dsv41_train.rl.rollout import Sample
        if not isinstance(request_id, str) or not _REQUEST_ID.fullmatch(request_id):
            raise ValueError("generation requires a durable 32-hex request identity")
        payload = copy.deepcopy(payload)
        with self._condition:
            if self._state != "running":
                raise RuntimeError("controller admission is paused, closed or unqualified")
            if request_id in self._seen:
                raise ValueError("generation request identity was already admitted; no replay")
            policy = self.policy
            self._request_sampling[request_id] = {"request_payload_sha256": evidence_digest(payload),
                "policy_ack": {key: getattr(policy, key) for key in ("version", "digest", "lineage")},
                "sampler_outcome": "not_admitted"}
            prompt = self._context(payload, policy)
            if len(prompt) + payload["max_tokens"] > self.max_model_len:
                raise ValueError("request exceeds the actual vLLM engine context budget")
            self._seen.add(request_id)
            self._active += 1
        stage = "vllm_sampler_admission"
        try:
            with self._sampler_lock:
                stage = "vllm_sampling"
                with self._condition:
                    self._request_sampling[request_id]["sampler_outcome"] = "unknown"
                result = self._sample_locked(prompt, payload["max_tokens"], policy.sampling["temperature"], request_id, policy)
                with self._condition:
                    self._request_sampling[request_id]["sampler_outcome"] = "completed"
                sample = Sample(result["prompt_ids"], result["response_ids"], result["behavior_log_probs"],
                    0., policy.version, result["finish_reason"], result["stop_token_count"])
                text = self.tokenizer.decode(sample.content_ids, skip_special_tokens=False)
                stage = "vllm_sample_persistence"
                if self.diagnostics_dir is not None:
                    from ..terminal_bench.controller import write_new
                    write_new(self.diagnostics_dir / (request_id + ".sampled.json"), {
                        "schema_version": 1, "kind": "vllm-actual-sampled-action", "request_id": request_id,
                        "request_payload_sha256": evidence_digest(payload), "policy": policy.to_dict(),
                        "decoded_action_text": text, "context_token_ids": sample.prompt_ids,
                        "action_token_ids": sample.response_ids, "behavior_log_probs": sample.behavior_log_probs,
                        "sampling_finish_reason": sample.finish_reason, "stop_token_count": sample.stop_token_count,
                        "logprob_convention": "sampled_distribution", "completion_parsed": False,
                        "benchmark_reward": None, "automatic_replay": False,
                        "vllm_generation": result["vllm_generation"]})
                stage = "vllm_shell_completion"
                format_status = None
                try:
                    message, finish = shell_completion(text, request_id, truncated=sample.finish_reason == "length")
                except ModelCompletionFormatError as error:
                    # Sampling finished; malformed model syntax is an action
                    # outcome. Preserve every sampled token, execute no tools,
                    # and let the CPU run its unchanged official verifier.
                    message, finish = {"role": "assistant", "content": text}, "stop"
                    format_status = invalid_tool_status(text, request_id, error)
                completion = {"id": "chatcmpl-" + request_id, "object": "chat.completion",
                    "model": self.served_model, "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                    "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(sample.response_ids),
                              "total_tokens": len(prompt) + len(sample.response_ids)}}
                if format_status is not None:
                    completion["model_completion_status"] = format_status
                    validate_model_completion_status(completion, request_id)
                completion["training_evidence"] = {"schema_version": 1, "request_id": request_id,
                    "request_payload_sha256": evidence_digest(payload), "response_sha256": evidence_digest(completion),
                    "policy": policy.to_dict(), "context_token_ids": sample.prompt_ids,
                    "action_token_ids": sample.response_ids, "behavior_log_probs": sample.behavior_log_probs,
                    "finish_reason": "tool_calls" if finish == "tool_calls" else sample.finish_reason,
                    "stop_token_count": sample.stop_token_count, "context_truncation": "none",
                    "logprob_convention": "sampled_distribution", "sampling_finish_reason": sample.finish_reason}
                stage = "vllm_training_evidence_validation"
                TrainingGenerationContract(policy, self._capabilities(policy)).validate(payload, completion, request_id, 1)
                return completion
        except Exception as error:
            if self._actor_failure is None:
                self._fail_generation(request_id, error, uncertain=False)
            error.native_generation_stage = stage
            raise
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    async def close(self):
        async with self._control_lock:
            with self._condition:
                self._state, self._drained = "closed", False
            # Only wait for local call threads here. The explicit owned
            # actor close below supplies the actual engine shutdown barrier.
            await _owned_thread(super()._wait_drained)
            try:
                if self.model is not None:
                    await _owned_thread(self.model.close)
            except BaseException:
                with self._condition:
                    self._state = "failed"
                raise
            self.model = None
            return {**(self.ack or {}), "drain_complete": True, "active_requests": 0}
