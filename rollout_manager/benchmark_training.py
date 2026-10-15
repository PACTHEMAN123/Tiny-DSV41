"""Benchmark evidence to complete training groups, with an injected controller.

This module implements no remote weight-loading API. The caller supplies a
qualified controller and fresh, prepared episode plans for each policy lease.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import queue
import threading
import time
import uuid
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Sequence

from .contracts import EpisodePlan, EpisodeResult, request_uncertain
from .training import PolicyRef, TrainingGroup, TrainingTrajectory, TurnTrace


@dataclass(frozen=True)
class RewardPolicy:
    benchmark: str
    mapping: str

    def __post_init__(self):
        supported = {"kernelbench": "native-final-correctness-v1", "cuda-agent": "native-final-correctness-v1",
                     "terminal-bench": "terminal-validated-verifier-v1"}
        if supported.get(self.benchmark) != self.mapping:
            raise ValueError("declare a supported, versioned benchmark reward mapping")

    def reward(self, result):
        if (result.identity.benchmark != self.benchmark or result.execution_state != "completed"
                or not result.score_eligible or result.failure_category is not None):
            raise ValueError("episode execution or scoring is ineligible")
        if result.cleanup_state not in {"verified", "not_required"} or request_uncertain(result.request_state):
            raise ValueError("episode cleanup or request delivery is unproven")
        if not result.request_state:
            raise ValueError("episode lacks durable request completion evidence")
        outcome = result.native_outcome
        if outcome.get("training_evidence_error"):
            raise ValueError("backend rejected training evidence")
        if self.mapping == "native-final-correctness-v1":
            value = outcome.get("final_correct")
            if type(value) is not bool or result.verification_state != ("passed" if value else "failed"):
                raise ValueError("native final verification is missing or inconsistent")
            return float(value)
        if result.verification_state != "evaluated" or outcome.get("review_classification") != "evaluated":
            raise ValueError("Terminal verifier requires an authoritative evaluated result")
        value = outcome.get("validated_reported_reward")
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Terminal verifier reward is missing or nonfinite")
        return float(value)


@dataclass(frozen=True)
class GroupRegistration:
    group_id: int
    plans: Sequence[EpisodePlan]
    policy: PolicyRef
    initial_context_sha256: str
    reset_sha256: str
    reward_policy: RewardPolicy

    def __post_init__(self):
        plans = tuple(self.plans)
        if type(self.group_id) is not int or self.group_id < 0 or len(plans) < 2:
            raise ValueError("register a semantic training group with G >= 2")
        if any(not isinstance(plan, EpisodePlan) for plan in plans) or not isinstance(self.policy, PolicyRef):
            raise ValueError("group registration requires prepared plans and a policy lease")
        if len({plan.identity for plan in plans}) != len(plans):
            raise ValueError("registered episode identities must be unique")
        if {plan.identity.sample_index for plan in plans} != set(range(len(plans))):
            raise ValueError("registered sample indices must be exactly 0..G-1")
        if len({(p.identity.run_id, p.identity.benchmark, p.identity.pass_index,
                 p.task.task_id, p.task.source_sha256) for p in plans}) != 1:
            raise ValueError("resource waves of different tasks are not a semantic training group")
        if self.reward_policy.benchmark != plans[0].identity.benchmark:
            raise ValueError("reward mapping differs from registered benchmark")
        for value in (self.initial_context_sha256, self.reset_sha256):
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("initial context and reset require exact SHA256 identities")
        object.__setattr__(self, "plans", tuple(sorted(plans, key=lambda p: p.identity.sample_index)))


@dataclass(frozen=True)
class GroupExclusion:
    group_id: int
    expected_size: int
    reasons: tuple[str, ...]
    episodes: tuple[EpisodeResult, ...]

    def to_dict(self):
        return {"schema_version": 1, "kind": "excluded_training_group", "group_id": self.group_id,
                "expected_size": self.expected_size, "reasons": list(self.reasons),
                "episodes": [asdict(result) for result in self.episodes]}


class GroupAssembler:
    """Bounded accounting only; episode admission remains in Coordinator."""
    def __init__(self, *, max_groups=1, max_episodes=8):
        if type(max_groups) is not int or max_groups < 1 or type(max_episodes) is not int or max_episodes < 2:
            raise ValueError("group assembler capacities must be positive")
        self.max_groups, self.max_episodes = max_groups, max_episodes
        self._groups, self._episodes, self._last_registered = {}, {}, -1

    def register(self, registration: GroupRegistration):
        if not isinstance(registration, GroupRegistration) or registration.group_id <= self._last_registered:
            raise ValueError("register a new monotonic semantic group identity")
        if len(self._groups) >= self.max_groups or len(self._episodes) + len(registration.plans) > self.max_episodes:
            raise ValueError("bounded training group capacity exhausted")
        if any(plan.identity in self._episodes for plan in registration.plans):
            raise ValueError("episode belongs to more than one semantic training group")
        self._groups[registration.group_id] = {"registration": registration, "results": {}}
        self._episodes.update({plan.identity: registration.group_id for plan in registration.plans})
        self._last_registered = registration.group_id

    def complete(self, result: EpisodeResult):
        if not isinstance(result, EpisodeResult) or result.identity not in self._episodes:
            raise ValueError("result was not registered or has already completed")
        group_id = self._episodes[result.identity]
        row = self._groups[group_id]
        if result.identity in row["results"]:
            raise ValueError("duplicate benchmark episode completion")
        row["results"][result.identity] = result
        registration = row["registration"]
        if len(row["results"]) != len(registration.plans):
            return None
        episodes = tuple(row["results"][plan.identity] for plan in registration.plans)
        samples, reasons = [], []
        for plan, episode in zip(registration.plans, episodes):
            try:
                if episode.source_sha256 != plan.task.source_sha256:
                    raise ValueError("completed source digest differs from registered task")
                reward = registration.reward_policy.reward(episode)
                raw = episode.native_outcome.get("training_turns")
                if not isinstance(raw, (list, tuple)) or not raw:
                    raise ValueError("exact sampled training turns are missing")
                turns = tuple(TurnTrace.from_dict(turn) for turn in raw)
                if any(turn.policy != registration.policy for turn in turns):
                    raise ValueError("sample policy receipt differs from registered group lease")
                states = episode.request_state.get("requests")
                if not isinstance(states, Mapping) or any(turn.request_id not in states for turn in turns):
                    raise ValueError("sampled requests lack durable delivery records")
                if any(not isinstance(states[turn.request_id], Mapping) or
                       states[turn.request_id].get("state") != "response_received" or
                       states[turn.request_id].get("training_evidence") != "validated" for turn in turns):
                    raise ValueError("sampled requests lack durable completion and validated training evidence")
                samples.append(TrainingTrajectory(group_id, plan.task.task_id, registration.initial_context_sha256,
                    registration.reset_sha256, plan.identity.sample_index, turns, reward,
                    evidence={"episode_identity": asdict(episode.identity), "native_artifacts": episode.native_artifacts,
                              "native_outcome": dict(episode.native_outcome),
                              "reward_mapping": registration.reward_policy.mapping},
                    sample_id=plan.identity.trial_id or f"{plan.identity.run_id}:{group_id}:{plan.identity.sample_index}",
                    raw_reward=reward, sampling_seed=plan.identity.sampling_seed))
            except (TypeError, ValueError, KeyError) as error:
                reasons.append(f"sample-{plan.identity.sample_index}: {error}")
        try:
            if not reasons:
                receipt = TrainingGroup(group_id, registration.plans[0].task.task_id,
                    registration.initial_context_sha256, registration.reset_sha256,
                    registration.policy, len(registration.plans), samples)
            else:
                receipt = GroupExclusion(group_id, len(registration.plans), tuple(reasons), episodes)
        except ValueError as error:
            receipt = GroupExclusion(group_id, len(registration.plans), (str(error),), episodes)
        for plan in registration.plans:
            del self._episodes[plan.identity]
        del self._groups[group_id]
        return receipt


def artifact_policy(artifact, *, identities=None):
    """External routes must add qualified tokenizer/template hashes explicitly."""
    identity = json.dumps(artifact["identity"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return PolicyRef(artifact["version"], artifact["digest"], artifact["lineage"],
                     {**(identities or {}), "run_identity_sha256": hashlib.sha256(identity).hexdigest()}, artifact["sampling"])


async def _invoke(owner, method, *args):
    value = getattr(owner, method)(*args)
    return await value if inspect.isawaitable(value) else value


class BenchmarkTrainingProvider:
    """Synchronous learner API over one persistent Coordinator stream.

    A factory closure injects coordinator/controller/plan_factory while accepting
    the ordinary RolloutManager constructor parameters from run_training.
    """
    def __init__(self, model_dir, model_config, config, records, random_init=False, *, engram_weights=None,
                 initial_policy, recovery_state=None, next_group_id=0, rng_state=None, event_root=None,
                 coordinator, controller, plan_factory: Callable, policy_factory=artifact_policy):
        if any(not callable(getattr(controller, method, None)) for method in ("start", "pause", "activate", "resume", "close")):
            raise ValueError("benchmark training requires an injected qualified policy controller")
        self.config, self.coordinator, self.controller = config, coordinator, controller
        self.plan_factory, self.policy_factory = plan_factory, policy_factory
        self._artifact = dict(initial_policy)
        self._policy = policy_factory(initial_policy)
        self._event_root, self._owned_store = event_root, None
        if config.temperature != self._policy.sampling["temperature"]:
            raise ValueError("learner and serving temperatures disagree")
        self._bound = config.queue_capacity + config.groups_per_step
        if self._bound < 1 or config.group_size < 2:
            raise ValueError("benchmark training requires bounded mailbox capacity and G >= 2")
        self._assembler = GroupAssembler(max_groups=1, max_episodes=config.group_size)
        self._next_group, self._committed, self._dropped = next_group_id, next_group_id, 0
        self._retained = {}
        if recovery_state:
            if (recovery_state.get("schema_version") != 1 or recovery_state.get("provider") != "benchmark-training"
                    or recovery_state.get("pending_requests") != [] or not recovery_state.get("admission_paused")
                    or recovery_state.get("active_policy", {}).get("lineage") != self._policy.lineage):
                raise ValueError("benchmark recovery requires a drained, reconciled same-lineage boundary")
            self._next_group = recovery_state["next_group_id"]
            self._committed = recovery_state["committed_highwater"]
            self._dropped = recovery_state["dropped_groups"]
            for value in recovery_state["pending_packets"]:
                packet = self._decode(value)
                group_id = packet["group_id"]
                if group_id in self._retained:
                    raise ValueError("duplicate restored benchmark group")
                group = packet["training_group"]
                if group and (group.policy.lineage != self._policy.lineage or group.policy.version > self._policy.version
                              or dict(group.policy.identities) != dict(self._policy.identities)
                              or dict(group.policy.sampling) != dict(self._policy.sampling)):
                    raise ValueError("restored benchmark group differs from committed policy identity")
                self._retained[group_id] = {"packet": packet, "delivered": False}
            if (len(self._retained) > self._bound or
                    sorted(self._retained) != list(range(self._committed, self._next_group))):
                raise ValueError("benchmark recovery mailbox has a cursor gap or exceeds capacity")
        self._thread, self._loop, self._pump = None, None, None
        self._active, self._paused, self._closing, self._closed = False, False, False, False
        self._error, self._ack = None, None
        self._activation_ready, self._timed_out = False, False
        self._owned_calls = set()

    @property
    def pid(self):
        return os.getpid()

    @staticmethod
    def _encode(packet):
        value = {key: item for key, item in packet.items() if key not in {"group", "training_group"}}
        value["training_group"] = packet["training_group"].to_dict() if packet["training_group"] else None
        return value

    @staticmethod
    def _decode(value):
        packet = dict(value)
        group = TrainingGroup.from_dict(packet["training_group"]) if packet["training_group"] else None
        if group and group.group_id != packet["group_id"]:
            raise ValueError("restored benchmark packet identity differs")
        if not group and packet.get("exclusion", {}).get("group_id") != packet["group_id"]:
            raise ValueError("restored benchmark exclusion identity differs")
        return {**packet, "group": group, "training_group": group}

    def _check_ack(self, value):
        expected = {name: getattr(self._policy, name) for name in ("version", "digest", "lineage")}
        if (not isinstance(value, Mapping) or type(value.get("version")) is not int
                or any(value.get(key) != item for key, item in expected.items())):
            raise ValueError("controller acknowledgement differs from committed policy")
        return expected

    def _call(self, coroutine, timeout=None):
        if self._loop is None or self._closed:
            coroutine.close()
            raise RuntimeError("benchmark provider is not running")
        async def owned():
            task = asyncio.current_task()
            self._owned_calls.add(task)
            try:
                return await coroutine
            finally:
                self._owned_calls.discard(task)
        future = asyncio.run_coroutine_threadsafe(owned(), self._loop)
        try:
            return future.result(timeout=self.config.worker_timeout_seconds if timeout is None else timeout)
        except (TimeoutError, FutureTimeoutError) as error:
            # The owned operation keeps draining. A timed-out waiter is not a
            # remote completion receipt and cannot authorize another activation.
            self._timed_out, self._paused, self._activation_ready = True, True, False
            self._error = RuntimeError("benchmark control acknowledgement timed out; reconciliation required")
            raise TimeoutError(str(error)) from error

    def start(self):
        if self._thread is not None or self._closed:
            raise ValueError("benchmark provider start is single-use")
        ready = threading.Event()
        def worker():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()
            self._loop.close()
        self._thread = threading.Thread(target=worker, name="benchmark-training", daemon=True)
        self._thread.start()
        ready.wait()
        try:
            return self._call(self._start())
        except BaseException:
            self.close()
            raise

    async def _start(self):
        self._condition, self._changed = asyncio.Condition(), asyncio.Event()
        self._control_lock = asyncio.Lock()
        self._ack = self._check_ack(await _invoke(self.controller, "start", self._artifact))
        if self._timed_out:
            raise self._error
        if self._event_root and self.coordinator.event_store is None:
            from pathlib import Path
            from .events import EventStore
            self._owned_store = EventStore(Path(self._event_root) / ("benchmark-" + uuid.uuid4().hex), self._policy.lineage)
            self.coordinator.event_store = self._owned_store
        await self.coordinator.open_stream(max_pending=self.config.group_size)
        self._pump = asyncio.create_task(self._produce())
        self._changed.set()
        return dict(self._ack)

    async def _produce(self):
        try:
            while not self._closing:
                await self._changed.wait()
                self._changed.clear()
                while not self._paused and not self._closing and len(self._retained) < self._bound:
                    # Plan preparation belongs to the leased semantic group.
                    # A pause must not overtake an asynchronous plan factory.
                    self._active = True
                    group_id = self._next_group
                    registration = self.plan_factory(group_id, self._policy)
                    if inspect.isawaitable(registration):
                        registration = await registration
                    if (not isinstance(registration, GroupRegistration) or registration.group_id != group_id
                            or registration.policy != self._policy or len(registration.plans) != self.config.group_size):
                        raise ValueError("plan factory changed the declared semantic group or policy")
                    if len({self.coordinator._group(plan) for plan in registration.plans}) != 1:
                        raise ValueError("stream plans require one admission group; Coordinator partitions its physical waves")
                    validator = getattr(self.coordinator.backends[registration.plans[0].task.benchmark],
                                        "validate_training_plans", None)
                    if validator is not None:
                        validator(registration.plans)
                    else:
                        from .backends.native import NativeBackend
                        backend = self.coordinator.backends[registration.plans[0].task.benchmark]
                        if isinstance(backend, NativeBackend):
                            for plan in registration.plans:
                                case = backend.cases.get((plan.task.task_id, plan.identity.pass_index, plan.identity.sample_index))
                                if (case is None or case["rollout"]["group_id"] != plan.identity.group_id or
                                        case["rollout"]["sampling_seed"] != plan.identity.sampling_seed):
                                    raise ValueError("native training requires prepared plans from its finite case map")
                    self._assembler.register(registration)
                    self._next_group += 1
                    started = time.monotonic()
                    results = await self.coordinator.execute_stream_group(registration.plans, sequence=group_id)
                    receipt = None
                    for result in results:
                        receipt = self._assembler.complete(result)
                    if receipt is None:
                        raise RuntimeError("coordinator did not complete every registered sibling")
                    group = receipt if isinstance(receipt, TrainingGroup) else None
                    packet = {"group_id": group_id, "group": group, "training_group": group,
                              "policy_ack": dict(self._ack), "started_at": started,
                              "finished_at": time.monotonic(), "engram_shared": False,
                              "worker_pid": self.pid, "rng_state": None}
                    if not group:
                        packet["exclusion"] = receipt.to_dict()
                    if self.coordinator.event_store:
                        self.coordinator.event_store.append("training_group_ready" if group else "training_group_excluded",
                            training_group_id=group_id, policy=self._policy.to_dict(), expected_size=len(registration.plans),
                            reward_mapping=registration.reward_policy.mapping,
                            exclusion_reasons=[] if group else list(receipt.reasons))
                    self._retained[group_id] = {"packet": packet, "delivered": False}
                    self._active = False
                    async with self._condition:
                        self._condition.notify_all()
                    if self.coordinator._stream_token.cancelled():
                        self._paused = True
                        packet["admission_stopped"] = self.coordinator._stream_token.reason
        except BaseException as error:
            self._error, self._paused, self._active = error, True, False
            async with self._condition:
                self._condition.notify_all()

    async def _get(self):
        for row in self._retained.values():
            if not row["delivered"]:
                row["delivered"] = True
                return dict(row["packet"])
        if self._error:
            raise RuntimeError("benchmark training producer failed") from self._error
        return None

    def get(self, timeout=None):
        deadline = time.monotonic() + (self.config.worker_timeout_seconds if timeout is None else timeout)
        while True:
            packet = self._call(self._get())
            if packet is not None:
                return packet
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise queue.Empty
            time.sleep(min(.01, remaining))

    def _snapshot(self):
        if self._active or self._error or self.coordinator._stream_token.cancelled():
            raise RuntimeError("benchmark recovery snapshot requires a proven, unquarantined boundary")
        return {"schema_version": 1, "provider": "benchmark-training", "next_group_id": self._next_group,
                "issued_highwater": self._next_group, "committed_highwater": self._committed,
                "dropped_groups": self._dropped, "pending_requests": [], "rng_state": None,
                "pending_packets": [self._encode(row["packet"]) for row in self._retained.values()],
                "active_policy": dict(self._artifact), "policy_ack": dict(self._ack), "admission_paused": True}

    async def _pause(self):
        async with self._control_lock:
            self._activation_ready = False
            state = await self._pause_drained()
            if self._timed_out:
                raise self._error
            self._activation_ready = True
            return state

    async def _pause_drained(self):
        self._paused = True
        async with self._condition:
            while self._active:
                await self._condition.wait()
        await self.coordinator.pause_stream()
        proof = await _invoke(self.controller, "pause")
        if not isinstance(proof, Mapping) or proof.get("drain_complete") is not True:
            raise RuntimeError("controller has not proven actual request drain")
        self._check_ack(proof)
        if self._error:
            raise RuntimeError("benchmark producer has unresolved work") from self._error
        return self._snapshot()

    def pause(self):
        return self._call(self._pause())

    async def _commit(self, consumed, dropped):
        ids = sorted([*consumed, *dropped])
        oldest = sorted(self._retained)[:len(ids)]
        if len(set(ids)) != len(ids) or ids != oldest or any(not self._retained[key]["delivered"] for key in ids):
            raise ValueError("commit requires a unique delivered oldest group prefix")
        for group_id in consumed:
            if self._retained[group_id]["packet"]["training_group"] is None:
                raise ValueError("excluded groups cannot be committed as optimizer input")
        for group_id in ids:
            del self._retained[group_id]
        if ids:
            self._committed = ids[-1] + 1
        self._dropped += len(dropped)
        self._changed.set()
        return {"committed_highwater": self._committed, "dropped_groups": self._dropped}

    def commit(self, consumed_group_ids=(), dropped_group_ids=()):
        return self._call(self._commit(list(consumed_group_ids), list(dropped_group_ids)))

    async def _activate(self, artifact):
        async with self._control_lock:
            if not self._activation_ready:
                raise RuntimeError("activation requires a verified controller drain receipt")
            self._activation_ready = False
            return await self._activate_drained(artifact)

    async def _activate_drained(self, artifact):
        if self._closing or not self._paused or self._active or not self.coordinator._stream_paused:
            raise ValueError("activation requires a proven pause and complete group drain")
        if self._error or self.coordinator._stream_token.cancelled():
            raise RuntimeError("uncertain or quarantined execution prevents policy activation")
        policy = self.policy_factory(artifact)
        if (policy.lineage != self._policy.lineage or policy.version <= self._policy.version
                or dict(policy.sampling) != dict(self._policy.sampling)
                or dict(policy.identities) != dict(self._policy.identities)):
            raise ValueError("activation requires a newer same-lineage, same-sampling policy")
        prior = self._policy
        self._policy = policy
        try:
            ack = self._check_ack(await _invoke(self.controller, "activate", dict(artifact)))
            if self._timed_out:
                raise self._error
        except BaseException:
            self._policy = prior
            self._error = RuntimeError("policy activation acknowledgement is unproven")
            raise
        self._artifact, self._ack = dict(artifact), ack
        return dict(ack)

    def activate(self, artifact):
        return self._call(self._activate(artifact))

    async def _resume(self):
        async with self._control_lock:
            if self._closing or self._error or self._timed_out:
                raise RuntimeError("provider requires reconciliation before resuming")
            self._check_ack(await _invoke(self.controller, "resume"))
            if self._timed_out:
                raise self._error
            self.coordinator.resume_stream()
            self._activation_ready, self._paused = False, False
            self._changed.set()
            return dict(self._ack)

    def resume(self):
        return self._call(self._resume())

    async def _close(self):
        self._closing, self._paused = True, True
        if hasattr(self, "_changed"):
            self._changed.set()
        if self._pump:
            await asyncio.shield(self._pump)
        # A timed-out control waiter may still be activating or proving drain.
        # Preserve its real operation before closing the controller or loop.
        outstanding = self._owned_calls - {asyncio.current_task()}
        if outstanding:
            await asyncio.shield(asyncio.gather(*outstanding, return_exceptions=True))
        try:
            await self.coordinator.close_stream()
        finally:
            try:
                await _invoke(self.controller, "close")
            finally:
                if self._owned_store:
                    self._owned_store.close()

    def close(self):
        if self._closed:
            return
        if self._loop:
            self._call(self._close())
            self._closed = True
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=self.config.worker_timeout_seconds)
        else:
            self._closed = True
