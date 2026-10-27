import asyncio
import hashlib
import json
import tempfile
import threading
import time
import unittest
from argparse import Namespace
from dataclasses import replace
from functools import partial
from pathlib import Path

from rollout_manager.benchmark_training import (BenchmarkTrainingProvider, GroupAssembler,
    GroupExclusion, GroupRegistration, RewardPolicy, artifact_policy)
from rollout_manager.contracts import EpisodeIdentity, EpisodePlan, EpisodeResult, ResourceRequirements, TaskSpec
from rollout_manager.coordinator import Coordinator
from rollout_manager.training import PolicyRef, TrainingGroup, TurnTrace, token_digest


class AssemblerFixture:
    def registration(self, benchmark="terminal-bench", group_id=0, size=2):
        self.policy = PolicyRef(0, "a" * 64, "fixture")
        task = TaskSpec(benchmark, "task-one", "b" * 64,
                       {"task_path": "/qualified/task"} if benchmark == "terminal-bench" else {"native_case": {}},
                       ResourceRequirements("cpu", cpu=1))
        plans = [EpisodePlan(EpisodeIdentity("fixture", benchmark, task.task_id,
                            "physical-wave-" + str(index), index, trial_id=f"trial-{group_id}-{index}"), task)
                 for index in range(size)]
        mapping = "terminal-validated-verifier-v1" if benchmark == "terminal-bench" else "native-final-correctness-v1"
        return GroupRegistration(group_id, plans, self.policy, token_digest([0, 1]), "c" * 64,
                                 RewardPolicy(benchmark, mapping))

    def result(self, plan, reward=1.0):
        trace = TurnTrace(1, plan.identity.trial_id, [0, 1], [2], [-0.3], self.policy)
        terminal = plan.identity.benchmark == "terminal-bench"
        outcome = {"kind": "terminal-bench" if terminal else "native-cuda", "training_turns": [trace.to_dict()],
                   "validated_reported_reward": reward, "official_reward": {"reward": reward},
                   "review_classification": "evaluated", "final_correct": bool(reward),
                   "original_verifier_evidence": "/original/verifier"}
        return EpisodeResult(plan.identity, "completed", "evaluated" if terminal else "passed" if reward else "failed",
            True, "verified", outcome, {"result": "/original/result.json"},
            request_state={"state": "response_received", "requests": {trace.request_id: {
                "state": "response_received", "training_evidence": "validated"}}},
            source_sha256=plan.task.source_sha256)


class AssemblerTests(AssemblerFixture, unittest.TestCase):
    def test_semantic_group_crosses_physical_waves_and_preserves_verifier_evidence(self):
        registration = self.registration()
        assembler = GroupAssembler(max_episodes=2)
        assembler.register(registration)
        first = self.result(registration.plans[0], 0)
        second = self.result(registration.plans[1], 1)
        before = json.dumps(first.native_outcome, sort_keys=True)
        self.assertIsNone(assembler.complete(second))
        group = assembler.complete(first)
        self.assertIsInstance(group, TrainingGroup)
        self.assertEqual([sample.reward for sample in group.samples], [0, 1])
        self.assertEqual(group.samples[0].evidence["native_outcome"]["official_reward"], {"reward": 0})
        self.assertEqual(json.dumps(first.native_outcome, sort_keys=True), before)
        with self.assertRaises(ValueError):
            assembler.complete(first)

    def test_exclusion_waits_every_sibling_without_zero_fill_or_partial_group(self):
        for changes in ({"validated_reported_reward": None}, {"training_turns": []}):
            registration = self.registration()
            assembler = GroupAssembler(max_episodes=2)
            assembler.register(registration)
            bad = self.result(registration.plans[0])
            bad = replace(bad, native_outcome={**bad.native_outcome, **changes})
            self.assertIsNone(assembler.complete(bad))
            receipt = assembler.complete(self.result(registration.plans[1]))
            self.assertIsInstance(receipt, GroupExclusion)
            self.assertEqual(len(receipt.episodes), 2)
            self.assertEqual(receipt.to_dict()["episodes"][0]["native_outcome"], bad.native_outcome)
        registration = self.registration("kernelbench")
        assembler = GroupAssembler(max_episodes=2)
        assembler.register(registration)
        missing = self.result(registration.plans[0], 0)
        missing = replace(missing, native_outcome={**missing.native_outcome, "final_correct": None})
        self.assertIsNone(assembler.complete(missing))
        self.assertIsInstance(assembler.complete(self.result(registration.plans[1])), GroupExclusion)

    def test_capacity_mixed_task_policy_and_unknown_delivery_fail_closed(self):
        registration = self.registration()
        other = replace(registration.plans[1].task, task_id="another-task")
        altered = replace(registration.plans[1], task=other,
                          identity=replace(registration.plans[1].identity, task_id=other.task_id))
        with self.assertRaisesRegex(ValueError, "different tasks"):
            replace(registration, plans=(registration.plans[0], altered))
        assembler = GroupAssembler(max_episodes=2)
        assembler.register(registration)
        with self.assertRaisesRegex(ValueError, "capacity"):
            assembler.register(self.registration(group_id=1))
        bad = self.result(registration.plans[0])
        trace = dict(bad.native_outcome["training_turns"][0])
        trace["policy"] = replace(self.policy, digest="d" * 64).to_dict()
        bad = replace(bad, native_outcome={**bad.native_outcome, "training_turns": [trace]})
        self.assertIsNone(assembler.complete(bad))
        second = replace(self.result(registration.plans[1]), request_state={"state": "outcome_unknown"})
        self.assertIsInstance(assembler.complete(second), GroupExclusion)

    def test_received_response_without_validated_training_evidence_excludes_group(self):
        registration = self.registration()
        assembler = GroupAssembler(max_episodes=2)
        assembler.register(registration)
        bad = self.result(registration.plans[0])
        bad = replace(bad, request_state={"state": "response_received", "requests": {
            bad.identity.trial_id: {"state": "response_received", "training_evidence": "rejected"}}})
        self.assertIsNone(assembler.complete(bad))
        receipt = assembler.complete(self.result(registration.plans[1]))
        self.assertIsInstance(receipt, GroupExclusion)
        self.assertIn("validated training evidence", receipt.reasons[0])


def capabilities(policy):
    from rollout_manager.generation_training import TrainingGenerationContract
    return {"schema_version": 1, "policy": policy.to_dict(), "logprob_convention": "sampled_distribution",
            **{key: True for key in TrainingGenerationContract.REQUIRED}}


class NativeProviderTests(unittest.TestCase):
    def fixture(self, root, *, wrong_ack=False):
        import torch
        from torch import nn
        from rollout_manager.adapters.datasets import DATASETS
        from rollout_manager.backends.native import NativeBackend
        from rollout_manager.datasets import KernelBenchAdapter
        from rollout_manager.generation_training import evidence_digest
        from rollout_manager.transports import RpcTransport
        from rollout_manager.workloads import suite_groups
        from dsv41_train.rl.data import PromptRecord
        from dsv41_train.rl.rollout import generate_group

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.transitions = nn.Parameter(torch.tensor([[0., .1, .2], [.2, 0., .1], [.1, .2, 0.]]))
            def forward(self, input_ids, attention_mask):
                return self.transitions[input_ids]
            def assert_engram_invariants(self): pass

        policy_factory = partial(artifact_policy, identities={"tokenizer_sha256": "e" * 64, "chat_template_sha256": "f" * 64})
        initial = {"version": 0, "digest": "a" * 64, "lineage": "fixture", "identity": {"model": "tiny-fixture"},
                   "sampling": {"temperature": .7, "top_p": 1, "top_k": 0}, "weights": Model().transitions.tolist()}

        class Controller:
            def __init__(self):
                self.model, self.policy, self.active = Model(), None, 0
                self.paused, self.closed, self.loads = False, False, []
                self.lock = threading.Lock()
            def load(self, artifact):
                with self.lock:
                    if self.active:
                        raise RuntimeError("cannot load during actual generation")
                    with torch.no_grad():
                        self.model.transitions.copy_(torch.tensor(artifact["weights"]))
                    self.policy = policy_factory(artifact)
                    self.loads.append(self.policy.version)
                ack = {key: getattr(self.policy, key) for key in ("version", "digest", "lineage")}
                return {**ack, "digest": "d" * 64} if wrong_ack else ack
            def start(self, artifact): return self.load(artifact)
            def pause(self):
                self.paused = True
                return {"drain_complete": self.active == 0,
                        **{key: getattr(self.policy, key) for key in ("version", "digest", "lineage")}}
            def activate(self, artifact):
                if not self.paused: raise RuntimeError("not paused")
                return self.load(artifact)
            def resume(self):
                self.paused = False
                return {key: getattr(self.policy, key) for key in ("version", "digest", "lineage")}
            def close(self): self.closed = True

        controller = Controller()
        class Client:
            def __init__(self, *args, **kwargs): pass
            def call(self, route, payload):
                with controller.lock:
                    controller.active += 1
                try:
                    sample = generate_group(controller.model, PromptRecord([0, 1], [2]), 1, payload["max_tokens"],
                        .7, None, controller.policy.version, 0, torch.Generator().manual_seed(payload["seed"])).samples[0]
                    completion = {"model": payload["model"], "choices": [{"finish_reason": sample.finish_reason,
                                  "message": {"role": "assistant", "content": "public candidate"}}]}
                    completion["training_evidence"] = {"schema_version": 1, "policy": controller.policy.to_dict(),
                        "request_id": route.removeprefix("/generate/"), "request_payload_sha256": evidence_digest(payload),
                        "response_sha256": evidence_digest(completion), "context_token_ids": sample.prompt_ids,
                        "action_token_ids": sample.response_ids, "behavior_log_probs": sample.behavior_log_probs,
                        "finish_reason": sample.finish_reason, "stop_token_count": sample.stop_token_count,
                        "context_truncation": "none", "logprob_convention": "sampled_distribution"}
                    return {"response": completion}
                finally:
                    with controller.lock: controller.active -= 1
            def close(self): pass

        source = "class Model: pass\n"
        case = {"task": {"task_id": "kernelbench/level_2/0", "dataset": "kernelbench",
                "revision": DATASETS["kernelbench"]["revision"], "code": source,
                "source_sha256": hashlib.sha256(source.encode()).hexdigest()},
                "request": {"model": "fixture-model", "messages": [{"role": "user", "content": "public"}],
                            "temperature": .7, "top_p": 1, "max_tokens": 4}, "files": {}}
        policy = policy_factory(initial)
        transport = RpcTransport("https://fixture.invalid", root, root / "ledger", client_type=Client)
        lifecycle_count = []
        def runner(args, *, episode_driver, client_factory, **kwargs):
            lifecycle_count.append(1)
            def execute(item, job, profile=False):
                client = client_factory(args.url, args.credentials)
                try:
                    client.call("/generate/" + job, item["request"])
                finally:
                    client.close()
                correct = item["rollout"]["sample_index"] == 0
                return {"completed": True, "cleanup_state": "verified", "score_eligible": True,
                        "development_correct": correct, "final_correct": correct, "verified_correct": correct,
                        "stages": {}, "rollout": item["rollout"]}
            return episode_driver(execute, suite_groups([case], 2, 1, 42), 2, lambda: False, time.monotonic() + 60)

        args = Namespace(schedule="full-suite", concurrency=[2], root=root, turns=1,
                         url="https://fixture.invalid", credentials=root, profile_worker=0)
        backend = NativeBackend({"args": args, "execution_profile": "training",
                                 "training_policy": policy.to_dict(), "training_capabilities": capabilities(policy)},
                                transport, runner_function=runner)
        coordinator = Coordinator({"kernelbench": backend}, max_workers=1, max_groups=1)
        def plans(group_id, policy):
            plans = KernelBenchAdapter([case]).plan("fixture", rollouts=2)
            plans = [replace(plan, identity=replace(plan.identity, trial_id=f"g{group_id}-s{plan.identity.sample_index}"),
                     payload={**plan.payload, "training_policy": policy.to_dict(), "training_capabilities": capabilities(policy)})
                     for plan in plans]
            return GroupRegistration(group_id, plans, policy, token_digest([0, 1]), "c" * 64,
                                     RewardPolicy("kernelbench", "native-final-correctness-v1"))
        config = Namespace(queue_capacity=1, groups_per_step=1, group_size=2, temperature=.7,
                           worker_timeout_seconds=10, clip_eps=.2, kl_beta=0., max_grad_norm=1.,
                           weight_decay=0., trainable_scope="all", train_engram=False)
        def provider(recovery_state=None, initial_policy=initial):
            return BenchmarkTrainingProvider(root, None, config, [], coordinator=coordinator,
                controller=controller, plan_factory=plans, initial_policy=initial_policy, policy_factory=policy_factory,
                recovery_state=recovery_state, event_root=root / "events")
        return provider, controller, backend, config, initial, lifecycle_count, Model

    def test_real_native_backend_transport_sampler_bridge_updates_and_activates(self):
        import torch
        from dsv41_train.rl.runtime import train_groups
        with tempfile.TemporaryDirectory() as temporary:
            make, controller, backend, config, initial, count, Model = self.fixture(Path(temporary))
            provider = make()
            try:
                self.assertEqual(provider.start()["version"], 0)
                packet = provider.get(timeout=5)
                snapshot = provider.pause()
                group = packet["training_group"]
                self.assertEqual([sample.reward for sample in group.samples], [1, 0])
                self.assertTrue(all(sample.policy_version == 0 for sample in group.samples))
                learner = Model()
                before = learner.transitions.detach().clone()
                optimizer = torch.optim.SGD(learner.parameters(), lr=.1)
                metrics = train_groups(learner, None, optimizer, [group], config)
                self.assertEqual(metrics["samples"], 2)
                self.assertFalse(torch.equal(before, learner.transitions))
                provider.commit(consumed_group_ids=[group.group_id])
                next_policy = {**initial, "version": 1, "digest": "b" * 64,
                               "weights": learner.transitions.detach().tolist()}
                self.assertEqual(provider.activate(next_policy)["version"], 1)
                provider.resume()
                queued_old = len(snapshot["pending_packets"]) - 1
                for _ in range(queued_old):
                    old = provider.get(timeout=5)
                    self.assertEqual(old["training_group"].policy_version, 0)
                    provider.commit(dropped_group_ids=[old["group_id"]])
                fresh = provider.get(timeout=5)
                self.assertEqual(fresh["training_group"].policy_version, 1)
                self.assertTrue(all(turn.policy.digest == "b" * 64 for sample in fresh["group"].samples for turn in sample.turns))
                provider.pause()
                self.assertEqual(count, [1])
                self.assertEqual(controller.loads, [0, 1])
                self.assertTrue(backend.transport.ledger_root.exists())
            finally:
                provider.close()
            self.assertTrue(controller.closed)
            self.assertFalse(backend.thread.is_alive())

    def test_controller_mismatch_and_unresolved_recovery_reject_before_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            make, controller, backend, _, initial, count, _ = self.fixture(Path(temporary), wrong_ack=True)
            provider = make()
            with self.assertRaisesRegex(ValueError, "acknowledgement"):
                provider.start()
            self.assertEqual(count, [])
            self.assertTrue(controller.closed)
        with tempfile.TemporaryDirectory() as temporary:
            make, *_ = self.fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "drained"):
                make(recovery_state={"schema_version": 1, "provider": "benchmark-training",
                    "pending_requests": [{"state": "outcome_unknown"}], "admission_paused": True,
                    "active_policy": {"lineage": "fixture"}})

    def test_pause_includes_async_plan_preparation_under_the_current_lease(self):
        with tempfile.TemporaryDirectory() as temporary:
            make, controller, _, _, _, _, _ = self.fixture(Path(temporary))
            provider = make()
            entered, release, paused = threading.Event(), threading.Event(), threading.Event()
            original_factory = provider.plan_factory
            async def prepare(group_id, policy):
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(.002)
                return original_factory(group_id, policy)
            provider.plan_factory = prepare
            failures = []
            def pause():
                try:
                    provider.pause()
                except BaseException as error:
                    failures.append(error)
                finally:
                    paused.set()
            provider.start()
            waiter = threading.Thread(target=pause)
            try:
                self.assertTrue(entered.wait(5))
                waiter.start()
                self.assertFalse(paused.wait(.03))
                self.assertFalse(controller.paused)
                release.set()
                self.assertTrue(paused.wait(5))
                self.assertEqual(failures, [])
                packet = provider.get(timeout=5)
                self.assertEqual(packet["training_group"].policy_version, 0)
            finally:
                release.set()
                if waiter.ident is not None:
                    waiter.join(5)
                provider.close()

    def test_pause_timeout_never_authorizes_activation_and_close_drains_its_real_proof(self):
        with tempfile.TemporaryDirectory() as temporary:
            make, controller, _, _, initial, _, _ = self.fixture(Path(temporary))
            provider = make()
            entered, release = threading.Event(), threading.Event()
            original_pause = controller.pause
            async def slow_pause():
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(.002)
                return original_pause()
            controller.pause = slow_pause
            provider.start()
            provider.get(timeout=5)
            try:
                with self.assertRaises(TimeoutError):
                    provider._call(provider._pause(), timeout=.05)
                self.assertTrue(entered.is_set())
                self.assertFalse(provider._activation_ready)
                next_policy = {**initial, "version": 1, "digest": "b" * 64}
                # This queued activation cannot overtake the pending drain proof.
                activation = asyncio.run_coroutine_threadsafe(provider._activate(next_policy), provider._loop)
                closed = threading.Event()
                failures = []
                def close():
                    try:
                        provider.close()
                    except BaseException as error:
                        failures.append(error)
                    finally:
                        closed.set()
                closing = threading.Thread(target=close)
                closing.start()
                self.assertFalse(closed.wait(.03))
                self.assertEqual(controller.loads, [0])
                self.assertFalse(controller.closed)
                release.set()
                closing.join(5)
                self.assertTrue(closed.is_set())
                self.assertEqual(failures, [])
                with self.assertRaises(RuntimeError):
                    activation.result(timeout=1)
                self.assertEqual(controller.loads, [0])
                self.assertTrue(controller.closed)
                self.assertFalse(provider._thread.is_alive())
            finally:
                release.set()
                if not provider._closed:
                    provider.close()


class TerminalProviderTests(unittest.TestCase):
    def test_complete_semantic_group_spans_bounded_cpu_waves_and_exclusion_is_not_optimizer_input(self):
        for missing_reward in (False, True):
            with self.subTest(missing_reward=missing_reward), tempfile.TemporaryDirectory() as temporary:
                artifact = {"version": 0, "digest": "a" * 64, "lineage": "terminal-fixture", "identity": {},
                            "sampling": {"temperature": 1, "top_p": 1, "top_k": 0}}
                class Controller:
                    def start(self, artifact):
                        self.policy = artifact_policy(artifact)
                        return self.ack()
                    def ack(self):
                        return {key: getattr(self.policy, key) for key in ("version", "digest", "lineage")}
                    def pause(self): return {**self.ack(), "drain_complete": not backend.active}
                    def activate(self, artifact): return self.start(artifact)
                    def resume(self): return self.ack()
                    def close(self): pass
                controller = Controller()
                class Backend:
                    def __init__(self):
                        self.active, self.wave, self.admissions, self.closed = set(), set(), [], False
                    async def start(self): pass
                    async def admit_group(self, plans):
                        if self.wave: raise ValueError("previous wave was not closed")
                        self.wave = {plan.identity for plan in plans}
                        self.admissions.append(len(plans))
                    async def execute(self, plan, cancellation):
                        if plan.identity not in self.wave: raise ValueError("not admitted")
                        self.active.add(plan.identity)
                        await asyncio.sleep(.002)
                        self.active.remove(plan.identity)
                        trace = TurnTrace(1, plan.identity.trial_id, [0, 1], [2], [-.3], controller.policy)
                        value = None if missing_reward and plan.identity.sample_index == 0 else float(plan.identity.sample_index % 2)
                        return EpisodeResult(plan.identity, "completed", "evaluated", True, "verified",
                            {"training_turns": [trace.to_dict()], "official_reward": {"reward": value},
                             "validated_reported_reward": value, "review_classification": "evaluated"},
                            {"original_verifier": "/original/verifier"},
                            request_state={"state": "response_received", "requests": {trace.request_id: {
                                "state": "response_received", "training_evidence": "validated"}}},
                            source_sha256=plan.task.source_sha256)
                    async def finish_group(self, plans, results):
                        if self.active: raise ValueError("actual wave siblings still running")
                        self.wave.clear()
                        return {"spool_terminal": True}
                    async def close(self): self.closed = True
                backend = Backend()
                config = Namespace(queue_capacity=1, groups_per_step=1, group_size=5, temperature=1,
                                   worker_timeout_seconds=5)
                coordinator = Coordinator({"terminal-bench": backend}, max_workers=3, max_groups=1,
                    capacities={"cpu": ResourceRequirements("cpu", cpu=2)}, policy="terminal-waves")
                task = TaskSpec("terminal-bench", "same-task", "b" * 64, {"task_path": "/qualified/task"},
                                ResourceRequirements("cpu", cpu=1))
                def plans(group_id, policy):
                    episodes = [EpisodePlan(EpisodeIdentity("terminal-fixture", "terminal-bench", task.task_id,
                            str(group_id), index, trial_id=f"g{group_id}-s{index}"), task) for index in range(5)]
                    return GroupRegistration(group_id, episodes, policy, token_digest([0, 1]), "c" * 64,
                                             RewardPolicy("terminal-bench", "terminal-validated-verifier-v1"))
                provider = BenchmarkTrainingProvider(temporary, None, config, [], coordinator=coordinator,
                    controller=controller, plan_factory=plans, initial_policy=artifact)
                try:
                    provider.start()
                    packet = provider.get(timeout=3)
                    provider.pause()
                    self.assertTrue(all(size <= 2 for size in backend.admissions))
                    self.assertEqual(backend.admissions[:3], [2, 2, 1])
                    if missing_reward:
                        self.assertIsNone(packet["training_group"])
                        self.assertEqual(len(packet["exclusion"]["episodes"]), 5)
                        with self.assertRaisesRegex(ValueError, "optimizer"):
                            provider.commit(consumed_group_ids=[packet["group_id"]])
                        provider.commit(dropped_group_ids=[packet["group_id"]])
                        self.assertEqual(provider.pause()["dropped_groups"], 1)
                    else:
                        self.assertEqual(len(packet["training_group"].samples), 5)
                        self.assertEqual([sample.reward for sample in packet["group"].samples], [0, 1, 0, 1, 0])
                finally:
                    provider.close()
                self.assertTrue(backend.closed)
                self.assertFalse(backend.active)


if __name__ == "__main__":
    unittest.main()
