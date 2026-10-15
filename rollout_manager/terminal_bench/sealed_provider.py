"""Learner provider for sealed CPU episodes and actual owned actor barriers."""
from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
import hashlib
import json
import os
from pathlib import Path
import queue
import threading
import time

from ..backends.native_qwen_loopback import (
    BrowserRelayPolicyController, LocalGPURelayDeliveryProbe, NativeControllerHTTPShim, write_new)
from ..backends.pytorch_controller import PyTorchPolicyController


def marker(event, **fields):
    print(json.dumps({"event": event, **fields}, sort_keys=True, allow_nan=False), flush=True)



def trainable_tensor_digest(model):
    """Hash weights independently of policy metadata or optimizer step counters."""
    import torch
    digest = hashlib.sha256()
    names = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            names.append(name)
            value = parameter.detach().cpu().contiguous()
            digest.update(json.dumps([name, list(value.shape), str(value.dtype)]).encode())
            digest.update(value.view(torch.uint8).numpy().tobytes())
    if not names:
        raise ValueError("a weight-update proof requires actual trainable tensors")
    return {"sha256": digest.hexdigest(), "parameter_names": names}



class SealedCPUGroupProvider:
    """Normal learner provider over independently sealed original CPU groups.

    It schedules no CPU episodes and provides no remote execution substitute.
    Immutable CPU files and actual local browser ACK records are both mandatory.
    """
    def __init__(self, model_dir, model_config, config, records, random_init=False, *, initial_policy,
                 recovery_state=None, next_group_id=0, engram_weights=None, rng_state=None, event_root=None,
                 capsule, manifests, relay_root, relay_port, shim_port, qualify_model,
                 task_ids=None, post_update_manifest=None, controller_factory=PyTorchPolicyController,
                 qualification=None, historical_delivery=None,
                 expected_original_seal_sha256=None, automatic_artifacts=False, expected_max_turns=None):
        if random_init or recovery_state is not None or next_group_id != 0:
            raise ValueError("this bounded experiment uses a new pretrained lineage; no CPU episode replay/resume")
        self.config, self.artifact = config, dict(initial_policy)
        self.capsule, self.manifests = Path(capsule), tuple(Path(path) for path in manifests)
        from rollout_manager.terminal_bench.sealed import TASKS
        self.task_ids = tuple(TASKS if task_ids is None else task_ids)
        if (len(self.task_ids) != len(self.manifests) or len(set(self.task_ids)) != len(self.task_ids)
                or any(task not in TASKS for task in self.task_ids)):
            raise ValueError("each selected original task needs one distinct training-group manifest")
        self.post_update_manifest = Path(post_update_manifest) if post_update_manifest else None
        self._post_update_done = False
        self.output = Path(config.output_dir)
        self.native = controller_factory(model_dir, config, served_model="Qwen/Qwen3-30B-A3B-Instruct-2507")
        self._qualification = qualification
        self._historical_delivery = historical_delivery
        self._expected_original_seal_sha256 = expected_original_seal_sha256
        if expected_max_turns is not None and (type(expected_max_turns) is not int or not 1 <= expected_max_turns <= 1024):
            raise ValueError("expected_max_turns must be an integer in 1..1024")
        self._expected_max_turns = expected_max_turns
        self._automatic_artifacts = automatic_artifacts
        self._initial_lease_published = False
        self.shim = NativeControllerHTTPShim(self.native, self.output / "native-serving", port=shim_port)
        self._relay_root, self._relay_port = Path(relay_root), relay_port
        self.controller = None
        self.qualify_model = qualify_model
        self._loop, self._thread = None, None
        self._next, self._committed = 0, 0
        self._retained, self._ack, self.lease = {}, None, None
        self._paused = self._closed = False
        self._error = None
        self._control_futures = []

    @property
    def pid(self):
        return os.getpid()

    def _call(self, coroutine, *, drain=False):
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        self._control_futures.append(future)
        # A timed-out observer does not cancel an owned model/drain operation.
        try:
            return future.result(timeout=None if drain else self.config.worker_timeout_seconds)
        except FutureTimeoutError as error:
            self._error = RuntimeError("owned native control acknowledgement timed out; no activation")
            raise self._error from error

    async def _start(self):
        self._ack = await self.native.start(self.artifact)
        from rollout_manager.backends.native_qwen_loopback import _owned_call
        self._initial_tensor_digest = await _owned_call(self._actor_tensor_digest)
        if self.qualify_model:
            qualification = self._qualification
            if qualification is None:
                from .native_workflow import qualify_native_model
                qualification = qualify_native_model
            await _owned_call(qualification, self.native, self.output)
        self.shim.start()
        # Relay starts after A_READY so its actual independent ownership check can
        # bind this listener and private startup receipt. Token is read at pause.
        self.lease = self._build_lease()
        write_new(self.output / "cpu-policy-lease.json", self.lease)
        marker("A_VLLM_READY" if self._vllm_actor() else "A_NATIVE_READY",
               policy_ack=self._ack, lease=str(self.output / "cpu-policy-lease.json"),
               native_controller_token_file=str(self.shim.token_file),
               native_controller_receipt=str(self.shim.output / "owned-controller-startup.json"),
               shim_port=self.shim.port, gpu_relay_port=self._relay_port,
               expires_unix=self.shim.expires_unix)
        return dict(self._ack)

    def _vllm_actor(self):
        return getattr(self.native, "serving_profile", "native-pytorch-v1") == "vllm-training-v1"

    def _actor_tensor_digest(self):
        loaded = getattr(self.native, "loaded_trainable_tensors", None)
        if loaded is None:
            loaded = getattr(self.native, "loaded_trainable_tensor_digest", None)
        return loaded() if loaded is not None else trainable_tensor_digest(self.native.model)

    def _build_lease(self, group_size=None):
        from rollout_manager.terminal_bench.sealed import build_lease
        return build_lease(self.capsule, self.native.policy, self.native.capabilities(), self.native.tokenizer,
                           group_size=self.config.group_size if group_size is None else group_size)

    def start(self):
        ready = threading.Event()
        def loop():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.run_until_complete(self._loop.shutdown_default_executor())
            self._loop.close()
        self._thread = threading.Thread(target=loop, name="sealed-terminal-native-control", daemon=False)
        self._thread.start()
        ready.wait()
        return self._call(self._start())

    def _delivery_controller(self):
        if self.controller is None:
            delivery = LocalGPURelayDeliveryProbe(self._relay_root, relay_port=self._relay_port,
                                                  expected_model=self.native.served_model)
            self.controller = BrowserRelayPolicyController(self.native, self.shim, delivery)
        return self.controller

    def get(self, timeout=None):
        if self._error:
            raise self._error
        if self._paused or self._next >= len(self.manifests):
            if timeout:
                time.sleep(timeout)
            raise queue.Empty
        path = self.manifests[self._next]
        deadline = time.monotonic() + (timeout or 0)
        while not self._automatic_artifacts and not path.is_file():
            if self._next not in self._retained:
                marker("A_WAITING_CPU_GROUP", group_id=self._next, manifest=str(path), policy_ack=self._ack)
                self._retained[self._next] = {"waiting": True}
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty
            time.sleep(min(remaining, .05))
        from rollout_manager.terminal_bench.sealed import validate_sealed_group
        if (self._historical_delivery is not None or self._automatic_artifacts) and self.controller is None:
            from rollout_manager.backends.native_qwen_loopback import _owned_call
            marker("A_WAITING_RECOVERY_RELAY", gpu_relay_port=self._relay_port,
                   original_generation_replayed=False)
            ready_deadline = time.monotonic() + self.config.worker_timeout_seconds
            while True:
                try:
                    delivery = self._delivery_controller().delivery
                    self._call(_owned_call(delivery.bind))
                    break
                except (FileNotFoundError, ConnectionError, OSError):
                    if time.monotonic() >= ready_deadline:
                        raise TimeoutError("fresh recovery relay did not become ready")
                    time.sleep(.2)
        if self._automatic_artifacts:
            from .artifact_transport import publish_file, receive_sealed
            if not self._initial_lease_published:
                publish_file(self._relay_root, "policy_lease", "lease-v0-training",
                             self.output / "cpu-policy-lease.json")
                self._initial_lease_published = True
            received = receive_sealed(self._relay_root, "group-%d" % self._next,
                                     path.parent, self.config.worker_timeout_seconds)
            if received.absolute() != path.absolute():
                raise ValueError("automatic sealed group target differs from the configured import")
        group = validate_sealed_group(path, self.native.policy, self._next, expected_lease=self.lease,
                                      expected_group_size=self.config.group_size,
                                      **({"expected_max_turns": self._expected_max_turns} if self._expected_max_turns is not None else {}))
        if self._expected_original_seal_sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest() != self._expected_original_seal_sha256:
            raise ValueError("recovered original group seal bytes differ from the exclusive consumer claim")
        expected = self.lease["tasks"].get(group.task_id)
        if (group.task_id != self.task_ids[self._next] or expected is None
                or group.initial_context_sha256 != expected["initial_context_sha256"]
                or group.reset_sha256 != expected["reset_sha256"] or len(group.samples) != self.config.group_size):
            raise ValueError("CPU group differs from the original pre-run task/context/reset lease")
        if self._historical_delivery is None:
            proof = self._call(self._delivery_controller().validate_group_delivery(group))
        else:
            proof = self._historical_delivery.validate(group)
        write_new(self.output / ("import-group-%08d.json" % group.group_id),
                  {"manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "group": group.to_dict(), "actual_gpu_delivery": proof})
        self._retained[group.group_id] = {"group": group, "manifest": str(path)}
        self._next += 1
        from dsv41_train.rl.algorithm import group_advantages
        rewards = [sample.reward for sample in group.samples]
        marker("A_CPU_GROUP_VALIDATED", group_id=group.group_id, task=group.task_id, rewards=rewards,
               nonzero_advantages=int(group_advantages(rewards).count_nonzero()),
               actual_gpu_delivery_complete=True)
        imported_at = time.monotonic()
        return {"group_id": group.group_id, "group": group, "training_group": group,
                "policy_ack": dict(self._ack), "started_at": imported_at, "finished_at": imported_at,
                "timing_stage": "verified_cpu_import",
                "engram_shared": False, "worker_pid": os.getpid(), "rng_state": None}

    async def _pause(self):
        self._paused = True
        controller = self._delivery_controller()
        proof = await controller.pause()
        self._ack = {key: proof[key] for key in ("version", "digest", "lineage")}
        return {"schema_version": 1, "provider": "sealed-terminal-cpu", "pending_requests": [],
            "provider_recovery_supported": False, "pending_packet_format": "sealed-terminal-cpu-v1",
            "next_group_id": self._next, "issued_highwater": self._next,
            "committed_highwater": self._committed, "dropped_groups": 0,
            "rng_state": None, "admission_paused": True, "active_policy": dict(self.artifact),
            "policy_ack": dict(self._ack), "pending_packets": [
                {"group_id": key, "training_group": row["group"].to_dict()}
                for key, row in sorted(self._retained.items()) if "group" in row],
            "delivery_drain": proof}

    def pause(self):
        if self._error:
            raise self._error
        return self._call(self._pause())

    def commit(self, consumed_group_ids=(), dropped_group_ids=()):
        if dropped_group_ids:
            raise RuntimeError("this experiment cannot silently replace/drop a failed original CPU group")
        ids = sorted(consumed_group_ids)
        if ids != list(range(self._committed, self._committed + len(ids))) or any(
                key not in self._retained or "group" not in self._retained[key] for key in ids):
            raise ValueError("commit requires a unique delivered oldest CPU group prefix")
        for key in ids:
            del self._retained[key]
        self._committed += len(ids)
        return {"committed_highwater": self._committed, "dropped_groups": 0}

    def activate(self, artifact):
        if self._error:
            raise self._error
        ack = self._call(self._delivery_controller().activate(artifact))
        self.artifact, self._ack = dict(artifact), dict(ack)
        after = self._actor_tensor_digest()
        self._weight_update = {"initial_trainable_tensors": self._initial_tensor_digest,
            "activated_trainable_tensors": after, "policy_ack": ack,
            "weights_changed": after["sha256"] != self._initial_tensor_digest["sha256"],
            "metadata_version_is_weight_change_proof": False}
        if self._vllm_actor():
            self._weight_update["actual_actor_activation"] = self.native.last_activation_receipt
        write_new(self.output / ("activated-policy-%08d.json" % ack["version"]), self._weight_update)
        marker("A_VLLM_ACTIVATED" if self._vllm_actor() else "A_NATIVE_ACTIVATED", policy_ack=ack)
        return ack

    def resume(self):
        ack = self._call(self._delivery_controller().resume())
        self._paused = False
        if self.post_update_manifest is not None and not self._post_update_done:
            self._verify_post_update(ack)
        return ack

    def _verify_post_update(self, ack):
        from rollout_manager.terminal_bench.sealed import validate_sealed_group
        lease = self._build_lease(group_size=1)
        lease_path = self.output / ("cpu-policy-lease-v%08d-evaluation.json" % ack["version"])
        write_new(lease_path, lease)
        marker("A_WAITING_UPDATED_POLICY_EPISODE", policy_ack=ack, lease=str(lease_path),
               manifest=str(self.post_update_manifest), expected_group_id=self._next,
               weights_changed=self._weight_update["weights_changed"])
        if self._automatic_artifacts:
            from .artifact_transport import publish_file, receive_sealed
            publish_file(self._relay_root, "policy_lease", "lease-v1-evaluation", lease_path)
            received = receive_sealed(self._relay_root, "evaluation-%d" % self._next,
                                     self.post_update_manifest.parent, self.config.worker_timeout_seconds)
            if received.absolute() != self.post_update_manifest.absolute():
                raise ValueError("automatic evaluation target differs from configured import")
        deadline = time.monotonic() + self.config.worker_timeout_seconds
        while not self.post_update_manifest.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for the fresh updated-policy CPU episode")
            time.sleep(.2)
        evaluation = validate_sealed_group(self.post_update_manifest, self.native.policy, self._next,
            expected_lease=lease, expected_group_size=1, evaluation=True,
            **({"expected_max_turns": self._expected_max_turns} if self._expected_max_turns is not None else {}))
        if evaluation.task_id not in self.task_ids:
            raise ValueError("updated-policy episode differs from the selected original tasks")
        proof = self._call(self._delivery_controller().validate_group_delivery(evaluation))
        drained = self.pause()
        write_new(self.output / "updated-policy-episode.json", {
            "kind": "verified-updated-policy-episode", "training_update_count": 1,
            "second_optimizer_step": False, "policy_ack": ack, "weight_update": self._weight_update,
            "lease": lease, "manifest_sha256": hashlib.sha256(self.post_update_manifest.read_bytes()).hexdigest(),
            "evaluation": evaluation.to_dict(), "actual_gpu_delivery": proof,
            "delivery_drain": drained["delivery_drain"]})
        self._post_update_done = True
        self._call(self._delivery_controller().resume())
        self._paused = False
        marker("A_UPDATED_POLICY_EPISODE_VERIFIED", policy_ack=ack,
               reward=evaluation.samples[0].reward, weights_changed=self._weight_update["weights_changed"])

    def close(self):
        if self._closed:
            return
        async def close():
            from rollout_manager.backends.native_qwen_loopback import _owned_call
            await _owned_call(self.shim.close)
            return await self.native.close()
        try:
            if self._loop is not None:
                # A control timeout is only an observer timeout. Wait for every
                # actual owned operation before closing the actor or event loop.
                for future in self._control_futures:
                    try:
                        future.result()
                    except Exception:
                        pass
                self._call(close(), drain=True)
        finally:
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join()  # Never detach the owned control loop.
            self._closed = True
