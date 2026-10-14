"""Shared admission and persistent streams; backends own episode lifecycles."""
from __future__ import annotations

import asyncio
import inspect
import math
from collections import Counter
from dataclasses import replace
from typing import Iterable, Mapping

from .contracts import (CancellationToken, EpisodeBackend, EpisodePlan, EpisodeResult,
                        ResourceRequirements, request_uncertain)
from .events import EventStore

_QUANTITIES = ("cpu", "memory_bytes", "storage_bytes", "gpu_count")


class Coordinator:
    def __init__(self, backends: Mapping[str, EpisodeBackend], *, max_workers=1, max_groups=1,
                 capacities: Mapping[str, ResourceRequirements] | None = None,
                 event_store: EventStore | None = None, drain_timeout=30.0, policy="native-groups"):
        if max_workers < 1 or max_groups < 1 or not math.isfinite(drain_timeout) or drain_timeout <= 0:
            raise ValueError("worker/group limits and drain timeout must be positive")
        self.backends = dict(backends)
        if policy not in {"native-groups", "terminal-waves"} or (policy == "terminal-waves" and max_groups != 1):
            raise ValueError("terminal waves require max_groups=1; unsupported admission policy")
        self.policy = policy
        self.max_workers, self.max_groups = max_workers, max_groups
        self.capacities, self.event_store = dict(capacities or {}), event_store
        self.drain_timeout = drain_timeout
        self.summary = {}
        self.reserved: dict[str, dict] = {}
        self.peak_reserved: dict[str, dict] = {}
        self.quarantined: dict[str, dict] = {}
        self.draining_tasks: set[asyncio.Task] = set()
        self.lifecycle_failures = []
        self._used = False

    def _event(self, event, plan=None, **details):
        if self.event_store:
            self.event_store.append(event, plan.identity if plan else None, **details)

    def _fits(self, requirement):
        if requirement.accounting == "backend-stage":
            return True
        used = self.reserved.get(requirement.executor, {})
        capacity = self.capacities.get(requirement.executor)
        return capacity is None or all(used.get(key, 0) + getattr(requirement, key) <=
                                       getattr(capacity, key) for key in _QUANTITIES)

    def _charge(self, requirement, sign):
        if requirement.accounting == "backend-stage":
            return
        used = self.reserved.setdefault(requirement.executor, dict.fromkeys(_QUANTITIES, 0))
        peak = self.peak_reserved.setdefault(requirement.executor, dict.fromkeys(_QUANTITIES, 0))
        for key in _QUANTITIES:
            used[key] += sign * getattr(requirement, key)
            peak[key] = max(peak[key], used[key])

    def _quarantine(self, requirement):
        if requirement.accounting == "backend-stage":
            return
        held = self.quarantined.setdefault(requirement.executor, dict.fromkeys(_QUANTITIES, 0))
        for key in _QUANTITIES:
            held[key] += getattr(requirement, key)

    @staticmethod
    def _group(plan):
        identity = plan.identity
        return identity.benchmark, identity.group_id, identity.pass_index

    async def _execute(self, plan, cancellation):
        try:
            result = await self.backends[plan.task.benchmark].execute(plan, cancellation)
            if not isinstance(result, EpisodeResult) or result.identity != plan.identity:
                raise ValueError("backend returned a different episode identity or result contract")
            return result
        except asyncio.CancelledError:
            return self._unresolved(plan, "backend_cancelled_without_cleanup_proof")
        except Exception as error:
            return self._unresolved(plan, type(error).__name__ + ": " + str(error))

    @staticmethod
    def _unresolved(plan, reason):
        return EpisodeResult(plan.identity, "failed", "unknown", False, "unknown",
                             {"reason": reason}, {}, failure_category="infrastructure",
                             source_sha256=plan.task.source_sha256)

    @staticmethod
    def _not_admitted(plan, reason):
        return EpisodeResult(plan.identity, "not_admitted", "not_performed", False, "not_required",
                             {"reason": reason}, {}, source_sha256=plan.task.source_sha256)

    async def _lifecycle(self, backend, method, *args):
        function = getattr(backend, method, None)
        if function is not None:
            value = function(*args)
            if inspect.isawaitable(value):
                return await value
            return value

    async def _close_backend(self, backend, cancellation):
        """Do not let teardown cancellation discard already collected evidence."""
        task = asyncio.create_task(self._lifecycle(backend, "close"))
        deadline = asyncio.get_running_loop().time() + self.drain_timeout
        while not task.done():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                self.draining_tasks.add(task)
                task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
                return "backend_close_timeout; cleanup remains unproven"
            try:
                await asyncio.wait({task}, timeout=min(.05, remaining))
            except asyncio.CancelledError:
                cancellation.cancel("coordinator_cancelled_during_close")
        try:
            task.result()
        except asyncio.CancelledError:
            return "backend_close_cancelled; cleanup remains unproven"
        except Exception as error:
            return "backend_close_failure:" + type(error).__name__ + ": " + str(error)
        return None

    @staticmethod
    def _safe(result):
        return result.cleanup_state in {"verified", "not_required"} and not request_uncertain(result.request_state)

    def _wave_fits(self, plans):
        totals = {}
        for plan in plans:
            requirement = plan.task.resources
            if requirement.accounting == "backend-stage":
                continue
            total = totals.setdefault(requirement.executor, dict.fromkeys(_QUANTITIES, 0))
            for key in _QUANTITIES:
                total[key] += getattr(requirement, key)
        for executor, total in totals.items():
            capacity = self.capacities.get(executor)
            if capacity and any(self.reserved.get(executor, {}).get(key, 0) + total[key] >
                                getattr(capacity, key) for key in _QUANTITIES):
                return False
        return True

    async def open_stream(self, *, max_pending=1, cancellation=None):
        """Start one bounded session without closing models between groups.

        Stream callers submit complete, atomic groups. Completion storage belongs
        to the caller, which must reserve its bounded mailbox credit before
        submitting a group. This coordinator alone admits its episode work.
        """
        if self._used or type(max_pending) is not int or max_pending < 1:
            raise ValueError("coordinator is single-use and stream capacity must be positive")
        self._used = True
        self._stream_token = cancellation if cancellation is not None else CancellationToken()
        self._stream_pending_limit, self._stream_pending = max_pending, 0
        self._stream_workers = asyncio.Semaphore(self.max_workers)
        self._stream_groups = asyncio.Semaphore(self.max_groups)
        self._stream_condition = asyncio.Condition()
        self._stream_tasks, self._stream_started = set(), []
        self._stream_stage_owners = {}
        self._stream_paused, self._stream_closed = False, False
        self._stream_sequence, self._stream_run_id = -1, None
        try:
            for backend in self.backends.values():
                if any(backend is prior for prior in self._stream_started):
                    continue
                self._stream_started.append(backend)
                await self._lifecycle(backend, "start")
        except BaseException:
            await self.close_stream()
            raise
        self._event("stream_started", pending_limit=max_pending)

    async def pause_stream(self):
        """Stop admission and await actual admitted episode completion."""
        self._stream_paused = True
        async with self._stream_condition:
            self._stream_condition.notify_all()
        while self._stream_tasks:
            try:
                await asyncio.shield(asyncio.gather(*tuple(self._stream_tasks), return_exceptions=True))
            except asyncio.CancelledError:
                self._stream_token.cancel("stream_pause_waiter_cancelled")
        self._event("stream_paused", reserved=self.reserved, quarantined=self.quarantined)

    def resume_stream(self):
        if self._stream_closed or self._stream_token.cancelled():
            raise RuntimeError("stream is closed or admission is stopped")
        self._stream_paused = False
        self._event("stream_resumed")

    async def execute_stream_group(self, plans, *, sequence):
        """Admit a complete group once, respecting worker/resource credits.

        A monotonic sequence prevents replay without retaining an unbounded set
        of completed identities. Results remain in caller order, even when G
        exceeds the number of episode workers.
        """
        plans = list(plans)
        if not hasattr(self, "_stream_token") or self._stream_closed or self._stream_paused:
            raise ValueError("stream admission is unavailable")
        if (type(sequence) is not int or sequence <= self._stream_sequence or not plans or
                len({p.identity for p in plans}) != len(plans) or
                len({self._group(p) for p in plans}) != 1):
            raise ValueError("stream requires a new monotonic sequence and one unique complete group")
        run_id = plans[0].identity.run_id
        if any(p.identity.run_id != run_id for p in plans) or self._stream_run_id not in (None, run_id):
            raise ValueError("a stream cannot combine run identities")
        if self._stream_pending + len(plans) > self._stream_pending_limit:
            raise ValueError("bounded stream input credit exhausted")
        for plan in plans:
            if plan.task.benchmark not in self.backends:
                raise ValueError("no backend registered for " + plan.task.benchmark)
            capacity, requirement = self.capacities.get(plan.task.resources.executor), plan.task.resources
            if requirement.accounting == "backend-stage":
                backend = self.backends[plan.task.benchmark]
                owner = self._stream_stage_owners.setdefault(requirement.executor, backend)
                if owner is not backend:
                    raise ValueError("backend-stage executor requires one authoritative backend owner")
            if capacity and requirement.accounting == "episode" and any(
                    getattr(requirement, key) > getattr(capacity, key) for key in _QUANTITIES):
                raise ValueError("episode exceeds executor capacity")
        self._stream_sequence, self._stream_run_id = sequence, run_id
        self._stream_pending += len(plans)

        async def execute(plan, *, wave_owned=False):
            async with self._stream_workers:
                async with self._stream_condition:
                    while not wave_owned and not self._fits(plan.task.resources):
                        if self._stream_token.cancelled():
                            return self._not_admitted(plan, "stream admission stopped")
                        await self._stream_condition.wait()
                    if self._stream_token.cancelled():
                        return self._not_admitted(plan, "stream admission stopped")
                    if not wave_owned:
                        self._charge(plan.task.resources, 1)
                    self._event("episode_admitted", plan, resources=plan.task.resources)
                result = await self._execute(plan, self._stream_token)
                async with self._stream_condition:
                    if self._safe(result) and not wave_owned:
                        self._charge(plan.task.resources, -1)
                    elif not self._safe(result):
                        if not wave_owned:
                            self._quarantine(plan.task.resources)
                        self._stream_token.cancel("stream_cleanup_or_request_unproven")
                    if result.failure_category in {"infrastructure", "infrastructure_error"}:
                        self._stream_token.cancel("stream_infrastructure_failure")
                    if result.verification_state == "review_required":
                        self._stream_token.cancel("stream_verifier_review_required")
                    self._stream_condition.notify_all()
                self._event("episode_finished", plan, execution_state=result.execution_state,
                            verification_state=result.verification_state)
                return result

        async def collect(selected, *, wave_owned=False):
            tasks = [asyncio.create_task(execute(plan, wave_owned=wave_owned)) for plan in selected]
            observed = await asyncio.gather(*tasks, return_exceptions=True)
            results = []
            for plan, result in zip(selected, observed):
                if isinstance(result, BaseException):
                    self._stream_token.cancel("stream_infrastructure_failure")
                    result = self._unresolved(plan, type(result).__name__ + ": " + str(result))
                results.append(result)
            return results

        def fits_capacity(selected):
            totals = {}
            for plan in selected:
                requirement = plan.task.resources
                if requirement.accounting == "backend-stage":
                    continue
                total = totals.setdefault(requirement.executor, dict.fromkeys(_QUANTITIES, 0))
                for key in _QUANTITIES:
                    total[key] += getattr(requirement, key)
            return all(executor not in self.capacities or all(total[key] <= getattr(self.capacities[executor], key)
                       for key in _QUANTITIES) for executor, total in totals.items())

        async def group():
            async with self._stream_groups:
                benchmark = plans[0].task.benchmark
                if any(plan.task.benchmark != benchmark for plan in plans):
                    raise ValueError("one streamed semantic group requires one backend")
                backend = self.backends[benchmark]
                wave_hooks = self.policy == "terminal-waves" or callable(getattr(backend, "admit_group", None))
                if not wave_hooks:
                    results = await collect(plans)
                else:
                    # Semantic G and physical waves are distinct. Only this
                    # coordinator partitions/reserves CPU waves; the assembler
                    # retains the single task/policy identity across all waves.
                    results, offset = [], 0
                    while offset < len(plans):
                        if self._stream_token.cancelled():
                            results.extend(self._not_admitted(plan, self._stream_token.reason or "cancelled")
                                           for plan in plans[offset:])
                            break
                        selected = []
                        for plan in plans[offset:offset + self.max_workers]:
                            if not fits_capacity([*selected, plan]):
                                break
                            selected.append(plan)
                        if not selected:
                            raise ValueError("one streamed episode exceeds physical wave capacity")
                        async with self._stream_condition:
                            while not self._wave_fits(selected) and not self._stream_token.cancelled():
                                await self._stream_condition.wait()
                            if self._stream_token.cancelled():
                                continue
                            for plan in selected:
                                self._charge(plan.task.resources, 1)
                        try:
                            await self._lifecycle(backend, "admit_group", selected)
                        except BaseException as error:
                            observed = [self._unresolved(plan, "group admission: " + type(error).__name__)
                                        for plan in selected]
                            closure = {"uncertain": True, "error_type": type(error).__name__}
                            self._stream_token.cancel("stream_group_admission_failure")
                        else:
                            observed = await collect(selected, wave_owned=True)
                            try:
                                closure = await self._lifecycle(backend, "finish_group", selected, observed)
                            except BaseException as error:
                                closure = {"uncertain": True, "error_type": type(error).__name__}
                        safe = all(self._safe(result) for result in observed) and not request_uncertain(closure)
                        async with self._stream_condition:
                            for plan in selected:
                                if safe:
                                    self._charge(plan.task.resources, -1)
                                else:
                                    self._quarantine(plan.task.resources)
                            self._stream_condition.notify_all()
                        if not safe:
                            self._stream_token.cancel("stream_group_closure_unproven")
                            observed = [replace(result, cleanup_state="unknown", score_eligible=False,
                                request_state={"state": "group_closure_unknown", "original": result.request_state,
                                               "closure": closure},
                                native_outcome={**result.native_outcome, "group_closure": closure})
                                for result in observed]
                        self._event("group_finished", selected[0], closure=closure, cleanup_verified=safe,
                                    semantic_group_sequence=sequence, wave_offset=offset)
                        results.extend(observed)
                        offset += len(selected)
                if self.event_store:
                    for result in results:
                        self.event_store.write_result(result)
                return results

        task = asyncio.create_task(group())
        self._stream_tasks.add(task)
        try:
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                self._stream_token.cancel("stream_waiter_cancelled")
                while not task.done():
                    try:
                        await asyncio.shield(task)
                    except asyncio.CancelledError:
                        pass
                return task.result()
        finally:
            self._stream_tasks.discard(task)
            self._stream_pending -= len(plans)

    async def close_stream(self):
        if not hasattr(self, "_stream_token") or self._stream_closed:
            return
        await self.pause_stream()
        self._stream_closed = True
        failures = []
        for backend in reversed(self._stream_started):
            closing = asyncio.create_task(self._lifecycle(backend, "close"))
            while not closing.done():
                try:
                    await asyncio.shield(closing)
                except asyncio.CancelledError:
                    self._stream_token.cancel("stream_close_waiter_cancelled")
                except Exception:
                    break
            try:
                closing.result()
            except BaseException as error:
                failures.append(type(error).__name__ + ": " + str(error))
        if failures:
            self.lifecycle_failures.extend(failures)
            raise RuntimeError("stream backend cleanup failed: " + "; ".join(failures))
        self._event("stream_closed", reserved=self.reserved, quarantined=self.quarantined)

    async def run(self, plans: Iterable[EpisodePlan], cancellation: CancellationToken | None = None) -> list[EpisodeResult]:
        if self._used:
            raise ValueError("coordinator runs are immutable; create a new coordinator")
        self._used = True
        plans = list(plans)
        if len({plan.identity for plan in plans}) != len(plans):
            raise ValueError("duplicate episode identities")
        if len({plan.identity.run_id for plan in plans}) > 1:
            raise ValueError("one coordinator cannot combine run identities")
        stage_owners = {}
        for plan in plans:
            if plan.task.benchmark not in self.backends:
                raise ValueError("no backend registered for " + plan.task.benchmark)
            requirement = plan.task.resources
            if requirement.accounting == "backend-stage":
                backend = self.backends[plan.task.benchmark]
                owner = stage_owners.setdefault(requirement.executor, backend)
                if owner is not backend:
                    raise ValueError("backend-stage executor requires one authoritative backend owner: " +
                                     requirement.executor)
            capacity = self.capacities.get(requirement.executor)
            if capacity and requirement.accounting == "episode" and any(
                    getattr(requirement, key) > getattr(capacity, key) for key in _QUANTITIES):
                raise ValueError("episode exceeds executor capacity: " + plan.task.task_id)
        token = cancellation if cancellation is not None else CancellationToken()
        pending, active, results = list(range(len(plans))), {}, {}
        expected = Counter(self._group(plan) for plan in plans)
        indices_by_group = {}
        for index, plan in enumerate(plans):
            indices_by_group.setdefault(self._group(plan), []).append(index)
        if self.policy == "terminal-waves":
            for indices in indices_by_group.values():
                if len(indices) > self.max_workers or not self._wave_fits([plans[index] for index in indices]):
                    raise ValueError("terminal wave exceeds worker or executor capacity")
        completed, held_groups = Counter(), set()
        stopped, drain_deadline, started = None, None, []
        loop = asyncio.get_running_loop()
        try:
            for benchmark in dict.fromkeys(plan.task.benchmark for plan in plans):
                if token.cancelled():
                    stopped = token.reason or "cancelled"
                    break
                backend = self.backends[benchmark]
                if any(backend is previous for previous in started):
                    continue
                started.append(backend)
                await self._lifecycle(backend, "start")
            while pending or active:
                if token.cancelled():
                    stopped = stopped or token.reason or "cancelled"
                    drain_deadline = drain_deadline or loop.time() + self.drain_timeout
                if stopped is None:
                    if self.policy == "terminal-waves" and not held_groups and pending:
                        group = self._group(plans[pending[0]])
                        indices = indices_by_group[group]
                        selected = [plans[index] for index in indices]
                        try:
                            await self._lifecycle(self.backends[selected[0].task.benchmark], "admit_group", selected)
                        except Exception as error:
                            stopped = "group_admission_failure:" + type(error).__name__ + ": " + str(error)
                        else:
                            if token.cancelled():
                                stopped = token.reason or "cancelled_during_group_admission"
                            else:
                                held_groups.add(group)
                                for index in indices:
                                    plan = plans[index]
                                    self._charge(plan.task.resources, 1)
                                    pending.remove(index)
                                    self._event("episode_admitted", plan, resources=plan.task.resources)
                                    active[asyncio.create_task(self._execute(plan, token))] = index
                    blocked_executors = set()
                    for index in list(pending) if self.policy == "native-groups" else []:
                        if len(active) >= self.max_workers:
                            break
                        plan, group = plans[index], self._group(plans[index])
                        if group not in held_groups and len(held_groups) >= self.max_groups:
                            continue
                        requirement = plan.task.resources
                        if requirement.executor in blocked_executors:
                            continue
                        if not self._fits(requirement):
                            blocked_executors.add(requirement.executor)
                            continue
                        self._charge(requirement, 1)
                        held_groups.add(group)
                        pending.remove(index)
                        self._event("episode_admitted", plan, resources=requirement)
                        task = asyncio.create_task(self._execute(plan, token))
                        active[task] = index
                if not active:
                    break
                if drain_deadline is not None and loop.time() >= drain_deadline:
                    for task, index in active.items():
                        plan = plans[index]
                        if self.policy == "native-groups":
                            self._quarantine(plan.task.resources)
                        results[index] = self._unresolved(plan, "drain_timeout; worker termination remains unproven")
                        self._event("episode_drain_unproven", plan)
                        # Do not cancel a to_thread wrapper and claim its worker stopped.
                        self.draining_tasks.add(task)
                        task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
                    if self.policy == "terminal-waves":
                        for group in held_groups:
                            for index in indices_by_group[group]:
                                self._quarantine(plans[index].task.resources)
                    active.clear()
                    break
                try:
                    timeout = min(.05, max(0, drain_deadline - loop.time())) if drain_deadline else .05
                    done, _ = await asyncio.wait(active, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                except asyncio.CancelledError:
                    token.cancel("coordinator_cancelled")
                    continue
                for task in done:
                    index = active.pop(task)
                    plan, result = plans[index], task.result()
                    results[index] = result
                    uncertain = request_uncertain(result.request_state)
                    if not self._safe(result):
                        if self.policy == "native-groups":
                            self._quarantine(plan.task.resources)
                        stopped = stopped or ("uncertain_model_request" if uncertain else "cleanup_unproven")
                        token.cancel(stopped)
                        self._event("resources_quarantined", plan, cleanup_state=result.cleanup_state,
                                    request_state=result.request_state)
                    elif self.policy == "native-groups":
                        self._charge(plan.task.resources, -1)
                    if result.failure_category in {"infrastructure", "infrastructure_error"}:
                        stopped = stopped or "infrastructure_failure"
                    if result.execution_state == "not_admitted":
                        stopped = stopped or str(result.native_outcome.get("reason", "backend_admission_stopped"))
                    if result.verification_state == "review_required":
                        stopped = stopped or "verifier_review_required"
                    group = self._group(plan)
                    completed[group] += 1
                    if completed[group] == expected[group]:
                        if self.policy == "terminal-waves":
                            indices = indices_by_group[group]
                            selected = [plans[index] for index in indices]
                            observed = [results[index] for index in indices]
                            closure = None
                            try:
                                closure = await self._lifecycle(self.backends[plan.task.benchmark],
                                                                "finish_group", selected, observed)
                            except asyncio.CancelledError:
                                token.cancel("coordinator_cancelled_during_group_closure")
                                closure = {"uncertain": True, "error_type": "CancelledError"}
                            except Exception as error:
                                closure = {"uncertain": True, "error_type": type(error).__name__}
                            safe = all(self._safe(item) for item in observed) and not request_uncertain(closure)
                            for index in indices:
                                if safe:
                                    self._charge(plans[index].task.resources, -1)
                                else:
                                    self._quarantine(plans[index].task.resources)
                            if not safe:
                                stopped = stopped or "group_closure_unproven"
                                token.cancel(stopped)
                            self._event("group_finished", plan, closure=closure, cleanup_verified=safe)
                        held_groups.remove(group)
                    self._event("episode_finished", plan, execution_state=result.execution_state,
                                verification_state=result.verification_state)
        except asyncio.CancelledError:
            token.cancel("coordinator_cancelled_during_lifecycle")
            stopped = token.reason
        except Exception as error:
            stopped = stopped or "coordinator_infrastructure_failure:" + type(error).__name__ + ": " + str(error)
            self.lifecycle_failures.append(stopped)
        finally:
            # Unexpected hook/runtime failure cannot prove the active workers stopped.
            for task, index in active.items():
                token.cancel(stopped or "coordinator_infrastructure_failure")
                results[index] = self._unresolved(plans[index], stopped or "coordinator_infrastructure_failure")
                self._quarantine(plans[index].task.resources)
                self.draining_tasks.add(task)
                task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
            if not self.draining_tasks:
                for backend in reversed(started):
                    failure = await self._close_backend(backend, token)
                    if failure:
                        self.lifecycle_failures.append(failure)
                        stopped = stopped or failure
            if token.cancelled():
                stopped = stopped or token.reason or "cancelled"
            for index in pending:
                results[index] = self._not_admitted(plans[index], stopped or "not_admitted")
                self._event("episode_not_admitted", plans[index], reason=stopped)
            ordered = [results[index] for index in range(len(plans))]
            self.summary = {"expected_episodes": len(plans), "result_count": len(ordered),
                            "counts": dict(Counter(result.execution_state for result in ordered)),
                            "groups": len(expected), "admission_stopped_reason": stopped,
                            "admission_policy": self.policy,
                            "drain_complete": not self.draining_tasks,
                            "cleanup_complete": not self.draining_tasks and not any(
                                "backend_close" in failure for failure in self.lifecycle_failures),
                            "lifecycle_failures": self.lifecycle_failures,
                            "reserved": self.reserved, "peak_reserved": self.peak_reserved,
                            "quarantined": self.quarantined}
            if self.event_store:
                for result in ordered:
                    self.event_store.write_result(result)
                self.event_store.append("run_finished", **self.summary)
                self.event_store.write_summary(self.summary)
        return ordered
