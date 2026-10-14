"""Importer contract fixtures; these never execute or qualify a benchmark task."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from rollout_manager.benchmark_training import GroupAssembler, GroupRegistration, RewardPolicy
from rollout_manager.contracts import EpisodeIdentity, EpisodePlan, EpisodeResult
from rollout_manager.datasets import TerminalBenchAdapter
from rollout_manager.generation_training import TrainingGenerationContract, evidence_digest
from rollout_manager.training import PolicyRef, token_digest
from rollout_manager.terminal_bench.sealed import (
    TASKS, TASK_PIN, PAYLOAD_PIN, build_lease, initial_payload, validate_sealed_group, write_new, write_seals)


class TokenizerFixture:
    def encode_conversation(self, messages, *, tools, chat_template_kwargs):
        return [1, 2, 3]


class SealedEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.model = "fixture/qwen"
        self.policy = PolicyRef(0, "a" * 64, "fixture-lineage",
            {"tokenizer_sha256": "b" * 64, "chat_template_sha256": "c" * 64,
             "qwen_logits_path": "last_position_head_v1"}, {"temperature": .8, "top_p": 1, "top_k": 0})
        self.capabilities = {"schema_version": 1, "policy": self.policy.to_dict(),
            **{key: True for key in TrainingGenerationContract.REQUIRED}, "logprob_convention": "sampled_distribution"}
        self.tasks = TerminalBenchAdapter(executor="fixture-cpu", task_root=self.root / "public/tasks", task_ids=TASKS).enumerate()
        rows = {}
        for task in self.tasks:
            instruction = "fixture original instruction " + task.task_id
            # The fixture preserves real catalog resources/budgets, while its
            # public source hash is patched below in the dataset used by tests.
            task = replace(task, source_sha256=evidence_digest(instruction))
            payload = initial_payload(instruction, self.model, .8)
            import hashlib
            source = hashlib.sha256(instruction.encode()).hexdigest()
            reset = {"task": asdict(replace(task, source_sha256=source)), "task_provenance_sha256": TASK_PIN,
                     "payload_sha256": PAYLOAD_PIN, "new_owned_container_per_sample": True, "network_mode": "none"}
            reset["original_task_budgets"] = {"environment_build": task.budgets["build_seconds"], "agent_setup": 360,
                "agent": task.budgets["agent_seconds"], "verifier": task.budgets["verifier_seconds"], "cleanup_reserve": 300}
            rows[task.task_id] = {"instruction_sha256": source, "payload": payload, "payload_sha256": evidence_digest(payload),
                                  "reset_spec": reset, "reset_sha256": evidence_digest(reset)}
        self.capsule = {"schema_version": 1, "kind": "terminalbench-public-payloads", "served_model": self.model,
            "tasks": rows, "agent_sha256": "d" * 64, "task_provenance_sha256": TASK_PIN, "payload_sha256": PAYLOAD_PIN}
        self.lease = build_lease(self.capsule, self.policy, self.capabilities, TokenizerFixture())

    def test_context_lease_matches_nested_tool_wire_order_and_preserves_strict_digests(self):
        """A template that renders JSON sees the relay's object order exactly."""
        from copy import deepcopy
        from rollout_manager.terminal_bench.browser_relay.relay_server import encoded
        from rollout_manager.training import TrainingTrajectory, TurnTrace
        class OrderSensitiveTokenizer:
            def encode_conversation(self, messages, *, tools, chat_template_kwargs):
                return list(json.dumps([messages, tools, chat_template_kwargs], ensure_ascii=False).encode())
        tokenizer = OrderSensitiveTokenizer()
        before = json.dumps(self.capsule)
        lease = build_lease(self.capsule, self.policy, self.capabilities, tokenizer)
        reordered = json.loads(json.dumps(self.capsule, sort_keys=True))
        self.assertEqual(build_lease(reordered, self.policy, self.capabilities, tokenizer), lease)
        self.assertEqual(json.dumps(self.capsule), before, "pre-run capsule remains unchanged")
        for task_id, row in self.capsule["tasks"].items():
            original, wire = row["payload"], json.loads(encoded(row["payload"]))
            self.assertEqual(original, wire)
            self.assertNotEqual(list(original["tools"][0]["function"]), list(wire["tools"][0]["function"]))
            actual = tokenizer.encode_conversation(wire["messages"], tools=wire["tools"],
                                                   chat_template_kwargs=wire["chat_template_kwargs"])
            pinned = lease["tasks"][task_id]
            self.assertEqual(pinned["initial_context_sha256"], token_digest(actual))
            self.assertEqual(pinned["payload_sha256"], row["payload_sha256"])
            trace = TurnTrace(1, "a" * 32, actual, [1], [-.5], self.policy)
            TrainingTrajectory(0, task_id, pinned["initial_context_sha256"], pinned["reset_sha256"], 0, [trace], 0.)
            prior = tokenizer.encode_conversation(original["messages"], tools=original["tools"],
                                                  chat_template_kwargs=original["chat_template_kwargs"])
            with self.assertRaisesRegex(ValueError, "initial context digest"):
                TrainingTrajectory(0, task_id, token_digest(prior), pinned["reset_sha256"], 0, [trace], 0.)
        corrupt = deepcopy(self.capsule)
        corrupt["tasks"][TASKS[0]]["payload_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "payload or task provenance"):
            build_lease(corrupt, self.policy, self.capabilities, tokenizer)

    def seal_fixture(self, size=2, evaluation=False, max_turns=None):
        """Construct auditable contract data only, explicitly distinct from task execution."""
        task = replace(self.tasks[0], source_sha256=self.capsule["tasks"][TASKS[0]]["instruction_sha256"])
        self.lease = build_lease(self.capsule, self.policy, self.capabilities, TokenizerFixture(), group_size=size)
        row = self.lease["tasks"][task.task_id]
        plans = [EpisodePlan(EpisodeIdentity("fixture-run", "terminal-bench", task.task_id, "semantic-0", index,
            trial_id=str(index + 1) * 32), task) for index in range(size)]
        registered = None if evaluation else GroupRegistration(0, plans, self.policy, row["initial_context_sha256"], row["reset_sha256"],
            RewardPolicy("terminal-bench", "terminal-validated-verifier-v1"))
        assembler = None if evaluation else GroupAssembler(max_groups=1, max_episodes=size)
        if assembler is not None:
            assembler.register(registered)
        write_new(self.root / "lease.json", self.lease)
        write_new(self.root / "public-capsule.json", self.capsule)
        write_new(self.root / "config.json", {"terminal_bench": {} if max_turns is None else {"agent_max_turns": max_turns}})
        turn_budget = 128 if max_turns is None else max_turns
        registered_value = {"group_id": 0, "plans": [asdict(p) for p in plans],
            "policy": self.policy.to_dict(), "initial_context_sha256": row["initial_context_sha256"], "reset_sha256": row["reset_sha256"]}
        write_new(self.root / "group-0.registration.json", registered_value)
        episodes = []
        contract = TrainingGenerationContract(self.policy, self.capabilities)
        receipt = None
        for plan in plans:
            identity = plan.identity.trial_id
            episode = self.root / ("episode-" + identity)
            episode.mkdir()
            job = {"n_attempts": 1, "n_concurrent_trials": 1, "timeout_multiplier": 1, "retry": {"max_retries": 0},
                "verifier": {"disable": False}, "tasks": [{"path": task.payload["task_path"]}],
                "agents": [{"kwargs": {"max_turns": turn_budget}}],
                "environment": {"delete": True, "kwargs": {"allow_public_network": False}}}
            write_new(episode / "config.json", job)
            import hashlib
            admission_id = hashlib.sha256(json.dumps(asdict(plan.identity), sort_keys=True).encode()).hexdigest()
            required = sum(self.capsule["tasks"][task.task_id]["reset_spec"]["original_task_budgets"].values())
            proof = {"schema_version": 1, "serving_profile": "native-pytorch-v1", "served_model": self.model,
                "model_pid": 1, "model_pid_start_ticks": 1, "model_cmdline_sha256": "e" * 64, "generation_slots": 1,
                "started_unix": 10., "expires_unix": 10000., "observed_unix": 100., "model_dir": "/fixture/model",
                "python_executable": "/fixture/python", "netns_inode": 1, "controller_state": "running", "active_generations": 0,
                "policy": self.policy.to_dict(), "policy_ack": {key: getattr(self.policy, key) for key in ("version", "digest", "lineage")},
                "capabilities": self.capabilities}
            profile = self.capabilities.get("serving_profile", "native-pytorch-v1")
            proof["serving_profile"] = profile
            controller_key = "native_controller"
            if profile == "vllm-training-v1":
                from rollout_terminal.test_vllm_training_profile import proof as vllm_fixture
                proof["vllm_engine"] = vllm_fixture()["vllm_engine"]
                proof["vllm_engine"]["loaded_policy_ack"] = proof["policy_ack"]
                controller_key = "training_controller"
            endpoint = {"serving_profile": profile, "model": self.model, "max_inflight": 1,
                        "generation_seconds": 360, "expires_unix": 10000.}
            write_new(self.root / ("admission-" + admission_id + ".json"), {"required_seconds": required,
                "episode_ids": [admission_id], "model_guard": {"serving_profile": profile},
                "readiness": {"protocol": 2, "paired_unix": 100., "cpu": endpoint,
                    "gpu": {**endpoint, controller_key: proof}}})
            trial = episode / ("trial-" + identity)
            logs = trial / "agent/file-relay"
            logs.mkdir(parents=True)
            (trial / "verifier").mkdir()
            reward = float(plan.identity.sample_index % 2)
            payload = self.capsule["tasks"][task.task_id]["payload"]
            completion = {"model": self.model, "choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": "fixture completion"}}]}
            completion["training_evidence"] = {"schema_version": 1, "request_id": identity,
                "request_payload_sha256": evidence_digest(payload), "response_sha256": evidence_digest(completion),
                "logprob_convention": "sampled_distribution", "policy": self.policy.to_dict(),
                "context_token_ids": [1, 2, 3], "action_token_ids": [4], "behavior_log_probs": [-.5],
                "finish_reason": "stop", "stop_token_count": 0, "context_truncation": "none"}
            trace = contract.validate(payload, completion, identity, 1)
            state = {"id": identity, "state": "response_received", "published": True, "replayed": False,
                     "remote_cancellation_proven": False}
            write_new(logs / (identity + ".request.json"), {"id": identity, "operation": "generate", "payload": payload})
            write_new(logs / (identity + ".response.json"), {"id": identity, "ok": True, "response": completion})
            write_new(logs / (identity + ".state.json"), state)
            write_new(logs / (identity + ".training.json"), trace.to_dict())
            budgets = {"generation_seconds": 360, "command_seconds": 30, "max_turns": turn_budget,
                       "max_tokens": 2048, "max_tool_output_bytes": 16384, "temperature": .8}
            write_new(logs / "agent-result.json", {"request_ids": [identity], "turns": 1, "error_type": None,
                "policy": budgets, "generation_retries": 0, "plugin_sha256": self.capsule["agent_sha256"]})
            write_new(logs / "trajectory.json", {"fixture": True})
            cleanup = {"provider": "pouch", "owner": identity, "container_id": identity * 2,
                "started": True, "stopped": True, "deleted": True, "cleanup_verified": True,
                "network_mode": "none", "public_network_explicitly_allowed": False,
                "image_id": task.payload["image_id"], "cpu_limit": task.resources.cpu,
                "memory_limit_mb": task.resources.memory_bytes // 1024**2}
            write_new(trial / "pouch-environment.json", cleanup)
            write_new(trial / "offline-bootstrap.json", {"preprovisioned": True, "manifest_sha256": PAYLOAD_PIN,
                                                        "task_id": task.task_id})
            write_new(trial / "agent-prerequisites.json", {"verified": True, "task_id": task.task_id})
            write_new(trial / "original-verifier-terminal.json", {
                "terminal_proven_by_provider": True, "return_code": 0 if reward else 1,
                "started_unix": 100., "finished_unix": 102., "command_sha256": "f" * 64})
            write_new(trial / "result.json", {"task_name": task.task_id,
                "verifier_result": {"rewards": {"reward": reward}}, "exception_info": None})
            report = {"results": {"tool": {"name": "pytest"}, "summary": {
                "tests": 1, "passed": int(reward), "failed": int(not reward), "skipped": 0, "pending": 0, "other": 0,
                "start": 100.5, "stop": 101.5}, "tests": [{"status": "passed" if reward else "failed",
                    "raw_status": "call_passed" if reward else "call_failed", "message": "assertion error"}]}}
            write_new(trial / "verifier/ctrf.json", report)
            (trial / "verifier/reward.txt").write_text(str(reward))
            (trial / "verifier/test-stdout.txt").write_text("fixture verifier transcript")
            outcome = {"training_turns": [trace.to_dict()], "validated_reported_reward": reward,
                "review_classification": "evaluated", "request_ids": [identity], "container_id": cleanup["container_id"]}
            marked = {**state, "training_evidence": "validated", "training_record": str(logs / (identity + ".training.json")),
                      "training_record_sha256": evidence_digest(trace.to_dict())}
            result = EpisodeResult(plan.identity, "completed", "evaluated", True, "verified", outcome,
                {"result": str(trial / "result.json"), "episode_dir": str(episode)}, model_identity=self.model,
                configuration_sha256=hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest(),
                source_sha256=task.source_sha256, request_state={"state": "response_received", "requests": {identity: marked}})
            relative = "group-0.sample-%d.episode.json" % plan.identity.sample_index
            write_new(self.root / relative, asdict(result))
            episodes.append(relative)
            from rollout_manager.terminal_bench.sealed import evaluation_receipt
            receipt = evaluation_receipt(registered_value, result) if evaluation else assembler.complete(result)
        write_new(self.root / "group-0.receipt.json", receipt.to_dict())
        closure = {"coordinator_closed": True, "cpu_spool_terminal": True, "lifecycle_failures": [],
                   "quarantined": {}, "reserved": {"fixture-cpu": {"cpu": 0}}, "admission_stopped_reason": None}
        seal = write_seals(self.root, [{"group_id": 0, "registration_file": "group-0.registration.json",
            "episode_files": episodes, "receipt_file": "group-0.receipt.json",
            "receipt_kind": "policy_evaluation" if evaluation else "training_group"}], closure)[0]
        return seal, task

    def validate(self, seal, task, **kwargs):
        # Exact-catalog task checks remain production code. This fixture patch
        # supplies only its changed public source, never skips verifier checks.
        from unittest.mock import patch
        with patch("rollout_manager.datasets.TerminalBenchAdapter.enumerate", return_value=[task]):
            return validate_sealed_group(seal, self.policy, 0, expected_lease=self.lease, **kwargs)

    def test_g8_requires_every_original_episode_and_its_exact_lease(self):
        seal, task = self.seal_fixture(size=8)
        group = self.validate(seal, task, expected_group_size=8)
        self.assertEqual([sample.reward for sample in group.samples], [0., 1.] * 4)
        self.assertEqual(len({sample.sample_id for sample in group.samples}), 8)
        with self.assertRaises(ValueError):
            self.validate(seal, task)  # No implicit promotion of a G2 consumer.
        value = json.loads(seal.read_text())
        value["episode_files"] = value["episode_files"][:-1]
        seal.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "expected independent"):
            self.validate(seal, task, expected_group_size=8)

    def test_sealed_quick_turn_budget_matches_cpu_config_and_original_agent_metadata(self):
        from rollout_manager.terminal_bench.sealed import sealed_inventory
        seal, task = self.seal_fixture(max_turns=16)
        self.assertEqual(len(self.validate(seal, task).samples), 2)
        self.assertEqual(len(self.validate(seal, task, expected_max_turns=16).samples), 2)
        with self.assertRaisesRegex(ValueError, "workflow declaration"):
            self.validate(seal, task, expected_max_turns=128)
        for invalid in (True, 0, 1025, "16"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "expected_max_turns"):
                self.validate(seal, task, expected_max_turns=invalid)
        agent_path = next(self.root.glob("episode-*/trial-*/agent/file-relay/agent-result.json"))
        value = json.loads(agent_path.read_text())
        value["policy"]["max_turns"] = 128
        agent_path.write_text(json.dumps(value))
        # Re-seal the fixture so the failure tests semantic alignment rather
        # than the independent immutable-file hash check.
        value = json.loads(seal.read_text())
        value["inventory"] = sealed_inventory(self.root)
        value["inventory_sha256"] = evidence_digest(value["inventory"])
        seal.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "original model/tool budgets"):
            self.validate(seal, task)

    def test_sealed_turn_budget_rejects_invalid_cpu_values_before_import(self):
        from rollout_manager.terminal_bench.sealed import sealed_inventory
        seal, task = self.seal_fixture()
        original = json.loads(seal.read_text())
        for invalid in (True, 0, 1025, 16.0):
            with self.subTest(value=invalid):
                (self.root / "config.json").write_text(json.dumps({"terminal_bench": {"agent_max_turns": invalid}}))
                value = dict(original)
                value["inventory"] = sealed_inventory(self.root)
                value["inventory_sha256"] = evidence_digest(value["inventory"])
                seal.write_text(json.dumps(value))
                with self.assertRaisesRegex(ValueError, "agent_max_turns"):
                    self.validate(seal, task)

    def test_vllm_g8_import_retains_exact_engine_policy_and_original_rewards(self):
        from rollout_manager.backends.vllm_profile import vllm_policy_identities
        self.policy = replace(self.policy, identities={**self.policy.identities, **vllm_policy_identities(262144)})
        self.capabilities = {**self.capabilities, "policy": self.policy.to_dict(),
                             "serving_profile": "vllm-training-v1"}
        seal, task = self.seal_fixture(size=8)
        group = self.validate(seal, task, expected_group_size=8)
        self.assertEqual(group.policy, self.policy)
        self.assertEqual([sample.reward for sample in group.samples], [0., 1.] * 4)

    def test_singleton_v1_evaluation_is_never_a_training_group_or_v0_receipt(self):
        from rollout_manager.terminal_bench.sealed import PolicyEvaluation
        old_lease = self.lease
        self.policy = replace(self.policy, version=1, digest="e" * 64)
        self.capabilities["policy"] = self.policy.to_dict()
        seal, task = self.seal_fixture(size=1, evaluation=True)
        evaluation = self.validate(seal, task, expected_group_size=1, evaluation=True)
        self.assertIsInstance(evaluation, PolicyEvaluation)
        self.assertEqual(evaluation.policy.version, 1)
        self.assertEqual(self.lease["purpose"], "policy_evaluation")
        self.assertEqual(old_lease["tasks"], self.lease["tasks"])
        with self.assertRaises(ValueError):
            self.validate(seal, task, expected_group_size=1)
        from unittest.mock import patch
        with patch("rollout_manager.datasets.TerminalBenchAdapter.enumerate", return_value=[task]):
            with self.assertRaises(ValueError):
                validate_sealed_group(seal, old_lease["policy"], 0, expected_group_size=1, evaluation=True)

    def test_complete_typed_group_revalidates_original_rewards_and_raw_evidence(self):
        seal, task = self.seal_fixture()
        group = self.validate(seal, task)
        self.assertEqual([sample.reward for sample in group.samples], [0., 1.])
        self.assertEqual(group.samples[0].turns[0].context_token_ids, (1, 2, 3))
        self.assertEqual(group.policy, self.policy)
        self.assertFalse(json.loads(seal.read_text())["gpu_delivery_proven"])

    def test_relocated_a_import_never_reads_the_original_cpu_filesystem(self):
        seal, task = self.seal_fixture()
        import shutil
        stage = tempfile.TemporaryDirectory()
        self.addCleanup(stage.cleanup)
        imported = Path(stage.name).resolve() / "copied-cpu-output"
        shutil.copytree(self.root, imported)
        shutil.rmtree(self.root)
        self.assertFalse(self.root.exists())
        group = self.validate(imported / seal.name, task)
        self.assertEqual([s.reward for s in group.samples], [0., 1.])
        self.assertTrue(group.samples[0].evidence["native_artifacts"]["result"].startswith(str(self.root)))

    def test_changed_original_bytes_and_different_expected_lease_fail(self):
        seal, task = self.seal_fixture()
        from unittest.mock import patch
        changed_lease = json.loads(json.dumps(self.lease))
        changed_lease["tasks"][TASKS[0]]["initial_context_sha256"] = "e" * 64
        with patch("rollout_manager.datasets.TerminalBenchAdapter.enumerate", return_value=[task]):
            with self.assertRaisesRegex(ValueError, "pre-admission"):
                validate_sealed_group(seal, self.policy, 0, expected_lease=changed_lease)
        training = next(self.root.rglob("*.training.json"))
        training.write_text(training.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "sealed original file changed"):
            self.validate(seal, task)
        with self.assertRaises(ValueError):
            validate_sealed_group(seal, self.policy, 1, expected_lease=self.lease)

    def test_resealed_unknown_cleanup_is_rejected(self):
        seal, task = self.seal_fixture()
        from rollout_manager.terminal_bench.sealed import sealed_inventory
        cleanup = next(self.root.rglob("pouch-environment.json"))
        changed = json.loads(cleanup.read_text())
        changed["cleanup_verified"] = False
        cleanup.write_text(json.dumps(changed))
        value = json.loads(seal.read_text())
        value["inventory"] = sealed_inventory(self.root)
        value["inventory_sha256"] = evidence_digest(value["inventory"])
        seal.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            self.validate(seal, task)

    def test_resealed_reward_disagreeing_with_original_assertions_is_rejected(self):
        seal, task = self.seal_fixture()
        from rollout_manager.terminal_bench.sealed import sealed_inventory
        result = sorted(self.root.rglob("trial-*/result.json"))[0]
        original = json.loads(result.read_text())
        original["verifier_result"]["rewards"]["reward"] = 1.
        result.write_text(json.dumps(original))
        value = json.loads(seal.read_text())
        value["inventory"] = sealed_inventory(self.root)
        value["inventory_sha256"] = evidence_digest(value["inventory"])
        seal.write_text(json.dumps(value))
        with self.assertRaisesRegex(ValueError, "verifier/cleanup classification"):
            self.validate(seal, task)

    def test_public_lease_never_accepts_changed_reset_or_payload(self):
        self.capsule["tasks"][TASKS[0]]["reset_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "public capsule"):
            build_lease(self.capsule, self.policy, self.capabilities, TokenizerFixture())


if __name__ == "__main__":
    unittest.main()
