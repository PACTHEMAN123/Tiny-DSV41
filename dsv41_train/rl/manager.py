"""Bounded learner bridge to a persistent manager-owned sampling backend."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import hashlib
import inspect
import json
import multiprocessing as mp
from pathlib import Path
import queue
import time
import traceback
import uuid


def _encode_packet(packet):
    return {**packet, "group": asdict(packet["group"]),
            "training_group": packet["training_group"].to_dict()}


def _decode_packet(packet):
    from .rollout import RolloutGroup, Sample
    from rollout_manager.training import TrainingGroup, to_rollout_group
    value = dict(packet)
    group = value["group"]
    value["group"] = RolloutGroup([Sample(**sample) for sample in group["samples"]],
                                  group["group_id"], group["policy_version"])
    value["training_group"] = TrainingGroup.from_dict(value["training_group"])
    trace = value["training_group"]
    ack = {key: getattr(trace.policy, key) for key in ("version", "digest", "lineage")}
    if asdict(to_rollout_group(trace)) != asdict(value["group"]) or ack != value["policy_ack"]:
        raise ValueError("restored training trace, native group and policy acknowledgement disagree")
    return value


async def _serve(connection, settings):
    from rollout_manager.contracts import EpisodeIdentity, EpisodePlan, ResourceRequirements, TaskSpec
    from rollout_manager.coordinator import Coordinator
    from rollout_manager.events import EventStore

    config, recovery = settings["config"], settings["recovery_state"] or {}
    if recovery and (recovery.get("schema_version") != 1 or
                     recovery.get("active_policy", {}).get("lineage") != settings["initial_policy"]["lineage"]):
        raise ValueError("manager recovery schema or policy lineage differs")
    bound = config.queue_capacity + config.groups_per_step
    next_group = recovery.get("next_group_id", settings["next_group_id"])
    committed = recovery.get("committed_highwater", next_group)
    dropped = recovery.get("dropped_groups", 0)
    retained = {}
    for encoded in recovery.get("pending_packets", []):
        packet = _decode_packet(encoded)
        group_id = packet["group"].group_id
        if group_id in retained or not committed <= group_id < next_group:
            raise ValueError("invalid or duplicate restored group cursor")
        retained[group_id] = {"packet": packet, "delivered": False}
    if len(retained) > bound or sorted(retained) != list(range(committed, next_group)):
        raise ValueError("restored mailbox exceeds capacity or has a cursor gap")
    rng = recovery.get("rng_state", settings["rng_state"])
    event_root = settings["event_root"]
    backend_values = dict(engram_weights=settings["engram_weights"],
                          initial_policy=settings["initial_policy"], rng_state=rng)
    if config.rollout_engine == "vllm":
        from rollout_manager.backends.vllm_rl import VLLMRLBackend
        if event_root is None:
            raise ValueError("owned vLLM sampling requires an evidence directory")
        backend_type = VLLMRLBackend
        backend_values["evidence_dir"] = Path(event_root) / "vllm-controller"
    else:
        from rollout_manager.backends.pytorch_rl import PyTorchRLBackend
        backend_type = PyTorchRLBackend
    backend = backend_type(settings["model_dir"], settings["model_config"], config,
                           settings["random_init"], **backend_values)
    store = EventStore(event_root, settings["initial_policy"]["lineage"]) if event_root else None
    coordinator = Coordinator({"prompt-response": backend}, max_workers=1, max_groups=1,
                              event_store=store, drain_timeout=config.worker_timeout_seconds)
    active, active_id = None, None
    paused, closing, pause_waiters, close_waiters = settings["start_paused"], False, [], []

    async def control(method, *args):
        function = getattr(backend, method, None)
        value = function(*args) if function is not None else None
        return await value if inspect.isawaitable(value) else value

    def reply(request_id, value=None):
        connection.send({"id": request_id, "value": value})

    def snapshot():
        if active is not None or backend._busy:
            raise RuntimeError("recovery snapshot requires an actual drained group boundary")
        return {"schema_version": 1, "next_group_id": next_group,
                "issued_highwater": next_group, "committed_highwater": committed,
                "dropped_groups": dropped, "rng_state": backend.rng_state(),
                "pending_packets": [_encode_packet(retained[key]["packet"]) for key in sorted(retained)],
                "pending_requests": [], "active_policy": dict(backend.active_artifact),
                "policy_ack": dict(backend.ack), "queue_capacity": config.queue_capacity,
                "retained_capacity": bound, "admission_paused": True,
                "resume_epoch": settings["resume_epoch"]}

    try:
        await coordinator.open_stream(max_pending=1)
        if paused:
            await coordinator.pause_stream()
            await control("pause")
        reply(0, dict(backend.ack))
        while True:
            while connection.poll():
                try:
                    request = connection.recv()
                except EOFError:
                    closing, paused = True, True
                    break
                request_id, command = request["id"], request["command"]
                if command == "get":
                    available = next((row for _, row in sorted(retained.items()) if not row["delivered"]), None)
                    if available is None:
                        reply(request_id, None)
                    else:
                        available["delivered"] = True
                        reply(request_id, _encode_packet(available["packet"]))
                elif command == "pause":
                    paused = True
                    pause_waiters.append(request_id)
                elif command == "commit":
                    consumed, rejected = request["consumed"], request["dropped"]
                    ids = sorted([*consumed, *rejected])
                    if (len(set(ids)) != len(ids) or ids != sorted(retained)[:len(ids)] or
                            any(not retained[group_id]["delivered"] for group_id in ids)):
                        raise ValueError("commit requires the delivered oldest group prefix without duplicates")
                    for group_id in ids:
                        del retained[group_id]
                    if ids:
                        committed = ids[-1] + 1
                    dropped += len(rejected)
                    reply(request_id, {"committed_highwater": committed, "dropped_groups": dropped})
                elif command == "activate":
                    if not paused or active is not None:
                        raise ValueError("activation requires pause() to finish draining")
                    reply(request_id, await control("activate", request["artifact"]))
                elif command == "evaluate":
                    if not paused or active is not None or not hasattr(backend, "evaluate"):
                        raise ValueError("actor evaluation requires a drained paused vLLM manager")
                    reply(request_id, await backend.evaluate(request["records"], step=request["step"],
                        phase=request["phase"], evaluation_id=request["evaluation_id"]))
                elif command == "resume":
                    if closing:
                        raise ValueError("manager is closing")
                    await control("resume")
                    coordinator.resume_stream()
                    paused = False
                    reply(request_id, dict(backend.ack))
                elif command == "close":
                    closing, paused = True, True
                    close_waiters.append(request_id)
                else:
                    raise ValueError("unknown manager command: " + command)
            if active is not None and active.done():
                results = active.result()
                if len(results) != 1 or not results[0].score_eligible or not coordinator._safe(results[0]):
                    raise RuntimeError("manager group failed: " + repr(results[0].native_outcome))
                retained[active_id] = {"packet": backend.take_packet(active_id), "delivered": False}
                active, active_id = None, None
            if paused and active is None:
                if not coordinator._stream_paused:
                    await coordinator.pause_stream()
                if pause_waiters:
                    await control("pause")
                for request_id in pause_waiters:
                    reply(request_id, snapshot())
                pause_waiters.clear()
                if closing:
                    await coordinator.close_stream()
                    for request_id in close_waiters:
                        reply(request_id, {"cleanup": "drained", "recovery": snapshot()})
                    break
            queued = sum(not row["delivered"] for row in retained.values())
            if not paused and active is None and len(retained) < bound and queued < config.queue_capacity:
                group_id = next_group
                record_index = group_id % len(settings["records"])
                record = settings["records"][record_index]
                source = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                task_id = f"prompt-response/{record_index}"
                payload = {"record": record, "group_id": group_id, "policy": dict(backend.ack)}
                task = TaskSpec("prompt-response", task_id, source, {"prompt_response": payload},
                    ResourceRequirements("vllm-policy" if config.rollout_engine == "vllm" else "pytorch-policy",
                                         accounting="backend-stage"),
                    {"group_size": config.group_size}, {"record_index": record_index})
                trial = hashlib.sha256(f"{backend.ack['lineage']}:{settings['resume_epoch']}:{group_id}".encode()).hexdigest()[:32]
                plan = EpisodePlan(EpisodeIdentity(backend.ack["lineage"], "prompt-response", task_id,
                                   str(group_id), 0, trial_id=trial), task, payload)
                # Reserve recovery mailbox credit and a never-reused cursor
                # before Coordinator can dispatch the actual sampling call.
                next_group += 1
                active_id = group_id
                active = asyncio.create_task(coordinator.execute_stream_group([plan], sequence=group_id))
            await asyncio.sleep(.005)
    finally:
        try:
            if active is not None:
                await asyncio.shield(active)
        finally:
            try:
                await coordinator.close_stream()
            finally:
                if store:
                    store.close()


def _manager_process(connection, settings):
    try:
        asyncio.run(_serve(connection, settings))
    except BaseException:
        try:
            connection.send({"fatal": traceback.format_exc()})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


class RolloutManager:
    """Synchronous learner API; its child keeps models and Coordinator alive."""

    def __init__(self, model_dir, model_config, config, records, random_init=False, *,
                 engram_weights=None, initial_policy, recovery_state=None,
                 next_group_id=0, rng_state=None, event_root=None, start_paused=False):
        self.config = config
        self.settings = {"model_dir": str(Path(model_dir).resolve()), "model_config": model_config,
            "config": config, "records": [asdict(record) for record in records], "random_init": random_init,
            "engram_weights": engram_weights, "initial_policy": dict(initial_policy),
            "recovery_state": recovery_state, "next_group_id": next_group_id, "rng_state": rng_state,
            "event_root": str(event_root) if event_root is not None else None,
            "resume_epoch": uuid.uuid4().hex, "start_paused": bool(start_paused)}
        if not self.settings["records"]:
            raise ValueError("manager requires at least one prompt record")
        self.process, self._request_id, self._closed, self._failed = None, 0, False, False

    @property
    def pid(self):
        return self.process.pid if self.process else None

    def _receive(self, request_id, timeout):
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._failed = True
                raise TimeoutError("rollout worker failed: manager command acknowledgement timed out")
            if self.connection.poll(min(.05, remaining)):
                try:
                    response = self.connection.recv()
                except EOFError:
                    self._failed = True
                    raise RuntimeError("rollout worker failed: manager process closed its channel") from None
                if "fatal" in response:
                    self._failed = True
                    raise RuntimeError("rollout worker failed:\n" + response["fatal"])
                if response["id"] != request_id:
                    self._failed = True
                    raise RuntimeError("rollout worker failed: manager command identity differs")
                return response["value"]
            if not self.process.is_alive():
                self._failed = True
                raise RuntimeError(f"rollout worker failed: manager process exited with code {self.process.exitcode}")

    def start(self):
        if self.process is not None or self._closed:
            raise ValueError("manager start is single-use")
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_manager_process, args=(child, self.settings), name="dsv41-rollout-manager")
        self.process.start()
        child.close()
        return self._receive(0, self.config.worker_timeout_seconds)

    def _rpc(self, command, **values):
        if self.process is None or self._closed or self._failed:
            raise RuntimeError("manager is not running")
        self._request_id += 1
        try:
            self.connection.send({"id": self._request_id, "command": command, **values})
        except (BrokenPipeError, EOFError, OSError):
            # A failed child may already have published its traceback before
            # closing the command side. Read that evidence before generic IPC
            # failure, and never let teardown send another command over it.
            if self.connection.poll():
                return self._receive(self._request_id, self.config.worker_timeout_seconds)
            self._failed = True
            raise RuntimeError("rollout worker failed: manager command channel closed") from None
        return self._receive(self._request_id, self.config.worker_timeout_seconds)

    def get(self, timeout=None):
        timeout = self.config.worker_timeout_seconds if timeout is None else timeout
        deadline = time.monotonic() + timeout
        while True:
            encoded = self._rpc("get")
            if encoded is not None:
                return _decode_packet(encoded)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty
            time.sleep(min(.01, remaining))

    def pause(self):
        return self._rpc("pause")

    def commit(self, consumed_group_ids=(), dropped_group_ids=()):
        return self._rpc("commit", consumed=list(consumed_group_ids), dropped=list(dropped_group_ids))

    def activate(self, artifact):
        return self._rpc("activate", artifact=dict(artifact))

    def resume(self):
        return self._rpc("resume")

    def evaluate(self, records, *, step, phase="eval"):
        return self._rpc("evaluate", records=[asdict(record) for record in records],
                         step=step, phase=phase, evaluation_id=uuid.uuid4().hex)

    def close(self):
        if self._closed:
            return
        prior_failure = self._failed
        failure = None
        try:
            if self.process is not None and self.process.is_alive() and not prior_failure:
                self._rpc("close")
            elif self.process is not None and not prior_failure:
                # A child can publish its failure and exit before close() gets
                # scheduled. Teardown must still surface that unread evidence.
                self._receive(self._request_id, .1)
        except (RuntimeError, TimeoutError, BrokenPipeError, EOFError, OSError) as error:
            failure = error
        finally:
            self._closed = True
            if self.process is not None:
                self.process.join(timeout=5)
                if self.process.is_alive():
                    # Only this owned local process is stopped. An interrupted
                    # group receives no completed/reusable recovery receipt.
                    self.process.terminate()
                    self.process.join(timeout=5)
                    if self.process.is_alive():
                        self.process.kill()
                        self.process.join()
                    failure = failure or RuntimeError("manager cleanup failed to drain its sampler")
                self.connection.close()
            if self.config.rollout_engine == "vllm" and self.settings["event_root"]:
                from rollout_manager.backends.vllm_cleanup import stop_owned_actor
                try:
                    terminated = stop_owned_actor(Path(self.settings["event_root"]) / "vllm-controller")
                    if terminated:
                        failure = failure or RuntimeError("owned vLLM actor was terminated; sampler drain is unproven")
                except (RuntimeError, OSError, ValueError) as error:
                    failure = failure or error
        if failure and not prior_failure:
            raise failure
