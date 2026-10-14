"""Actor fixtures test routing/contracts; these tests do not qualify a GPU engine."""
import asyncio
from dataclasses import asdict, replace
import json
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer, models

from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.rl.checkpoint import publish_policy_artifact, run_identity
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.data import PromptRecord
from dsv41_train.rl.family import build_policy, load_model_config
from dsv41_train.rl.manager import _decode_packet, _encode_packet, _serve
from dsv41_train.rl.runtime import run_training
from rollout_manager.backends.vllm_cleanup import stop_owned_actor
from rollout_manager.backends.vllm_profile import vllm_policy_identities
from rollout_manager.backends.vllm_rl import VLLMRLBackend, require_vllm_rollout, validate_vllm_device_ownership
from rollout_manager.contracts import CancellationToken, EpisodeIdentity, EpisodePlan, TaskSpec, ResourceRequirements
from qwen_helpers import tiny_config, write_hf_checkpoint
from test_vllm_controller import FakeActor


class GenericSelectionTests(unittest.TestCase):
    def test_cli_defaults_vllm_and_only_explicitly_selects_native(self):
        from dsv41_train.rl.train import main
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "config.json"
            config.write_text('{}')
            argv = ["train", "--model-dir", "unused", "--data", "unused", "--config", str(config)]
            for extra, engine in (([], "vllm"), (["--rollout-engine", "native-pytorch"], "native-pytorch")):
                with patch.object(sys, "argv", argv + extra), patch("dsv41_train.rl.train.run_training") as train:
                    main()
                self.assertEqual(train.call_args.kwargs["config"].rollout_engine, engine)
            config.write_text('{"rollout_engine":"native-pytorch"}')
            with patch.object(sys, "argv", argv), patch("dsv41_train.rl.train.run_training") as train:
                main()
            self.assertEqual(train.call_args.kwargs["config"].rollout_engine, "native-pytorch")
        self.assertEqual(RLConfig.from_dict({}).rollout_engine, "vllm")
        self.assertEqual(RLConfig().rollout_engine, "vllm")
        from dsv41_train.rl.checkpoint import checkpoint_config
        migrated = checkpoint_config({"eos_token_ids": []}, SimpleNamespace(eos_token_id=2))
        self.assertEqual(migrated.to_dict()["rollout_engine"], "native-pytorch")
        with self.assertRaisesRegex(ValueError, "rollout_engine"):
            RLConfig.from_dict({"rollout_engine": "auto"})

    def test_unsupported_family_fails_before_device_or_learner_startup(self):
        with patch("dsv41_train.rl.runtime.load_model_config", return_value=SimpleNamespace(model_type="deepseek_v41")), \
                patch("dsv41_train.rl.runtime.make_model") as learner, \
                patch("dsv41_train.rl.runtime.validate_available_devices") as devices:
            with self.assertRaisesRegex(ValueError, "DeepSeek requires a vLLM weight/checkpoint adapter"):
                run_training(model_dir="unused", data_path="unused", config=RLConfig(rollout_engine="vllm"))
        learner.assert_not_called()
        devices.assert_not_called()

    def test_native_evaluation_and_checkpoint_sample_cannot_fallback(self):
        from dsv41_train.rl.evaluation import evaluate_policy
        from dsv41_train.rl.sample import load_checkpoint_model
        with patch("dsv41_train.rl.evaluation.generate_group") as native:
            with self.assertRaisesRegex(ValueError, "native sampling fallback"):
                evaluate_policy(SimpleNamespace(behavior_engine="vllm"), [object()], RLConfig(),
                    reward_fn=None, generation_kwargs={}, step=0)
            native.assert_not_called()
        state = {"format_version": 2, "config": {}, "policy": {}, "step": 1,
                 "identity": {"random_init": False, "rollout_engine": "vllm"}}
        with patch("dsv41_train.rl.sample.torch.load", return_value=state), \
                patch("dsv41_train.rl.sample.make_model") as learner:
            with self.assertRaisesRegex(ValueError, "requires vLLM inference"):
                load_checkpoint_model(model_dir="unused", checkpoint="unused")
            learner.assert_not_called()


class Connection:
    def __init__(self):
        self.requests, self.replies = [], {}
    def poll(self):
        return bool(self.requests)
    def recv(self):
        return self.requests.pop(0)
    def send(self, value):
        self.replies[value["id"]] = value["value"]
    async def rpc(self, command, **values):
        identifier = max(self.replies, default=0) + 1
        self.requests.append({"id": identifier, "command": command, **values})
        async with asyncio.timeout(5):
            while identifier not in self.replies:
                await asyncio.sleep(.005)
        return self.replies[identifier]


class GenericVLLMBackendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        write_hf_checkpoint(self.root, Qwen3MoeForCausalLM(tiny_config(max_position_embeddings=128)).to(torch.bfloat16))
        tokenizer = Tokenizer(models.WordLevel({"unk": 0, **{f"token{i}": i for i in range(1, 32)}}, unk_token="unk"))
        tokenizer.save(str(self.root / "tokenizer.json"))
        (self.root / "tokenizer_config.json").write_text('{"tokenizer_class":"PreTrainedTokenizerFast","unk_token":"unk"}')
        (self.root / "chat_template.jinja").write_text("{% for m in messages %}{{m.content}}{% endfor %}")
        (self.root / "data.jsonl").write_text('{"prompt_ids":[1,4,9]}\n')
        self.config = RLConfig(rollout_engine="vllm", group_size=2, max_new_tokens=8, temperature=.8,
            trainable_scope="lm_head", qwen_last_position_only=True, qwen_kv_cache=True,
            dtype="bfloat16", learner_devices=("cuda:0",), rollout_devices=("cuda:1",), reward="toy_token_score")
        self.model_config = load_model_config(self.root)
        self.learner = build_policy(self.root, self.model_config, random_init=False,
            devices=("cpu",), dtype=torch.bfloat16, trainable_scope="lm_head")
        self.identity = run_identity(self.root, self.root / "data.jsonl", False)
        self.identity.update(trainable_scope="lm_head", qwen_logits_path="last_position_head_v1",
                             **vllm_policy_identities(128))
        for target, value in (("rollout_manager.backends.vllm_controller.OwnedVLLMActor", FakeActor),
                              ("rollout_manager.backends.vllm_controller.check_process", lambda receipt: None)):
            mock = patch(target, value)
            mock.start()
            self.addCleanup(mock.stop)
        self.record = PromptRecord([1, 4, 9])

    def artifact(self, version):
        return publish_policy_artifact(self.learner, version, self.root / "policies", lineage="generic-fixture",
            identity=self.identity, sampling={"temperature": .8, "top_p": 1, "top_k": 0})

    async def backend(self):
        backend = VLLMRLBackend(self.root, self.model_config, self.config, False,
            initial_policy=self.artifact(0), evidence_dir=self.root / "backend")
        await backend.start()
        self.addAsyncCleanup(backend.close)
        return backend

    def plan(self, backend):
        payload = {"record": asdict(self.record), "group_id": 0, "policy": dict(backend.ack)}
        task = TaskSpec("prompt-response", "fixture-task", "b" * 64, {"prompt_response": payload},
                        ResourceRequirements("vllm", accounting="backend-stage"))
        return EpisodePlan(EpisodeIdentity("fixture", "prompt-response", "fixture-task", "0", 0,
                           trial_id="d" * 32), task, payload)

    async def test_exact_token_probability_reward_packet_and_head_barrier(self):
        backend = await self.backend()
        scored = []
        def reward(record, tokens):
            scored.append(tokens)
            return .75
        reward.strip_stop_tokens = True
        backend.reward = reward
        result = await backend.execute(self.plan(backend), CancellationToken())
        self.assertTrue(result.score_eligible)
        packet = _decode_packet(_encode_packet(backend.take_packet(0)))
        self.assertEqual(scored, [[3], [3]])
        self.assertIsNone(packet["rng_state"])
        for sample in packet["training_group"].samples:
            self.assertEqual(sample.response_ids, [3, 2])
            self.assertEqual(sample.behavior_log_probs, [-.4, -.3])
            self.assertEqual(sample.reward, .75)
            self.assertEqual(sample.stop_token_count, 1)
            self.assertEqual(len(sample.turns[0].request_id), 32)
            receipt = json.loads((backend.evidence_dir / (sample.turns[0].request_id + ".sampled.json")).read_text())
            self.assertEqual(receipt["vllm_generation"]["sampling_seed"], sample.sampling_seed)
        with self.assertRaisesRegex(RuntimeError, "proven paused"):
            await backend.activate(self.artifact(1))
        await backend.pause()
        next_artifact = self.artifact(1)
        self.assertEqual(await backend.activate(next_artifact), {key: next_artifact[key] for key in ("version", "digest", "lineage")})
        self.assertEqual(backend.policy.version, 1)
        await backend.resume()

    async def test_evaluation_independent_seeds_stops_and_restores_paused_state(self):
        backend = await self.backend()
        await backend.execute(self.plan(backend), CancellationToken())
        packet = backend._packets[0]
        await backend.pause()
        first = await backend.evaluate([asdict(self.record)], step=0, phase="baseline", evaluation_id="first")
        second = await backend.evaluate([asdict(self.record)], step=0, phase="eval", evaluation_id="second")
        self.assertEqual(first["rows"][0]["vllm_generation"]["sampling_seed"],
                         second["rows"][0]["vllm_generation"]["sampling_seed"])
        self.assertNotEqual(first["rows"][0]["request_id"], second["rows"][0]["request_id"])
        self.assertEqual(first["rows"][0]["response_ids"], [3, 2])
        self.assertEqual(backend.controller._state, "paused")
        self.assertIs(backend._packets[0], packet)
        self.assertEqual(first["metrics"]["rollout_engine"], "vllm")

    async def test_rpc_failure_never_replays_or_switches_sampler(self):
        backend = await self.backend()
        original = backend.controller.model.call
        def failed(operation, **values):
            if operation == "generate":
                raise TimeoutError("fixture unknown remote outcome")
            return original(operation, **values)
        backend.controller.model.call = failed
        with patch("dsv41_train.rl.rollout.generate_group") as native:
            with self.assertRaises(TimeoutError):
                await backend.execute(self.plan(backend), CancellationToken())
            native.assert_not_called()
        self.assertFalse(backend._packets)
        self.assertEqual(backend.controller._state, "failed")
        with self.assertRaisesRegex(RuntimeError, "failed"):
            await backend.pause()

    async def test_manager_awaits_actor_control_and_evaluation_before_admission(self):
        connection = Connection()
        settings = dict(model_dir=str(self.root), model_config=self.model_config, config=self.config,
            records=[asdict(self.record)], random_init=False, engram_weights={}, initial_policy=self.artifact(0),
            recovery_state=None, next_group_id=0, rng_state=None, event_root=str(self.root / "events"),
            resume_epoch="fixture", start_paused=True)
        task = asyncio.create_task(_serve(connection, settings))
        async with asyncio.timeout(5):
            while 0 not in connection.replies:
                if task.done():
                    task.result()
                await asyncio.sleep(.005)
        evaluated = await connection.rpc("evaluate", records=[asdict(self.record)], step=0,
                                         phase="baseline", evaluation_id="baseline")
        self.assertEqual(evaluated["metrics"]["samples"], 1)
        snapshot = await connection.rpc("pause")
        self.assertEqual(snapshot["next_group_id"], 0)
        await connection.rpc("activate", artifact=self.artifact(1))
        await connection.rpc("resume")
        async with asyncio.timeout(5):
            while (packet := await connection.rpc("get")) is None:
                await asyncio.sleep(.01)
        self.assertEqual(_decode_packet(packet)["training_group"].policy.version, 1)
        await connection.rpc("pause")
        await connection.rpc("commit", consumed=[0], dropped=[])
        closed = await connection.rpc("close")
        await task
        self.assertEqual(closed["cleanup"], "drained")
        self.assertEqual(closed["recovery"]["committed_highwater"], 1)
        self.assertTrue(FakeActor.instances[-1].closed)

    async def test_capability_checks_reject_unqualified_geometry_and_native_rng(self):
        for config, random in ((replace(self.config, rollout_devices=("cpu",)), False),
                               (replace(self.config, learner_devices=("cpu",)), False),
                               (replace(self.config, rollout_devices=("cuda:0",)), False),
                               (replace(self.config, stop_token_sequences=((3, 2),)), False),
                               (self.config, True)):
            with self.assertRaises(ValueError):
                require_vllm_rollout(self.model_config, config, random_init=random)
        with self.assertRaisesRegex(ValueError, "CPU sampler"):
            VLLMRLBackend(self.root, self.model_config, self.config, False, initial_policy=self.artifact(0),
                          rng_state=b"native", evidence_dir=self.root / "backend")

    async def test_physical_gpu_aliases_fail_before_learner_allocation(self):
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "GPU-same,GPU-same"}):
            with self.assertRaisesRegex(ValueError, "disjoint physical"):
                require_vllm_rollout(self.model_config, self.config)
        # Different visible labels can still resolve to the same physical UUID.
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0,GPU-same"}), \
                patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(uuid="GPU-same")), \
                patch("dsv41_train.rl.runtime.validate_available_devices"), \
                patch("dsv41_train.rl.runtime.make_model") as learner:
            with self.assertRaisesRegex(ValueError, "disjoint physical"):
                run_training(model_dir=self.root, data_path=self.root / "data.jsonl", config=self.config)
            learner.assert_not_called()
        with patch("torch.cuda.get_device_properties", side_effect=lambda device: SimpleNamespace(uuid="GPU-" + device)):
            validate_vllm_device_ownership(self.config)

    async def test_generic_runtime_native_learner_vllm_actor_eval_checkpoint_and_resume(self):
        fixture = self
        managers = []
        class AlternatingActor(FakeActor):
            def call(self, operation, **values):
                result = super().call(operation, **values)
                if operation == "generate":
                    result["action_token_ids"][0] = 3 + self.counter % 2
                return result

        class InProcessManager:
            # Use real actor-controller/head artifacts and native learner; replace
            # only process transport so the local fixture needs no CUDA/vLLM.
            def __init__(self, model_dir, model_config, config, records, random_init, *,
                         initial_policy, engram_weights, recovery_state, next_group_id,
                         rng_state, event_root, start_paused):
                self.backend = VLLMRLBackend(model_dir, model_config, config, random_init,
                    initial_policy=initial_policy, engram_weights=engram_weights,
                    rng_state=rng_state, evidence_dir=Path(event_root) / "vllm-controller")
                self.records, self.next_group, self.paused = records, next_group_id, start_paused
                self.pid, self.closed = 123, False
                managers.append(self)
            def start(self):
                asyncio.run(self.backend.start())
                if self.paused:
                    asyncio.run(self.backend.pause())
                return dict(self.backend.ack)
            def get(self, timeout):
                plan = fixture.plan(self.backend)
                plan.payload["group_id"] = self.next_group
                # Cursor identity is fresh on each resumed manager, as in _serve.
                plan = replace(plan, identity=replace(plan.identity,
                    trial_id=f"{len(managers):016x}{self.next_group:016x}"))
                asyncio.run(self.backend.execute(plan, CancellationToken()))
                packet = self.backend.take_packet(self.next_group)
                self.next_group += 1
                return packet
            def pause(self):
                asyncio.run(self.backend.pause())
                self.paused = True
                return {"schema_version": 1, "next_group_id": self.next_group,
                    "issued_highwater": self.next_group, "committed_highwater": self.next_group,
                    "rng_state": None, "pending_packets": [], "pending_requests": [],
                    "active_policy": dict(self.backend.active_artifact), "policy_ack": dict(self.backend.ack)}
            def commit(self, consumed_group_ids):
                fixture.assertEqual(consumed_group_ids, [self.next_group - 1])
            def activate(self, artifact):
                return asyncio.run(self.backend.activate(artifact))
            def resume(self):
                self.paused = False
                return asyncio.run(self.backend.resume())
            def evaluate(self, records, *, step, phase):
                fixture.assertTrue(self.paused)
                return asyncio.run(self.backend.evaluate([asdict(record) for record in records],
                    step=step, phase=phase, evaluation_id=f"{len(managers)}-{phase}-{step}"))
            def close(self):
                asyncio.run(self.backend.close())
                self.closed = True

        validation = self.root / "validation.jsonl"
        validation.write_text('{"prompt_ids":[1,5,9]}\n')
        config = replace(self.config, max_steps=1, output_dir=str(self.root / "run"))
        before = self.learner.backend.lm_head.weight.detach().clone()
        from dsv41_train.rl.runtime import make_model as native_model
        def cpu_learner(model_dir, model_config, config, devices, random_init, **kwargs):
            return native_model(model_dir, model_config, config, ("cpu",), random_init, **kwargs)
        with patch("dsv41_train.rl.manager.RolloutManager", InProcessManager), \
                patch("dsv41_train.rl.runtime.validate_available_devices"), \
                patch("dsv41_train.rl.runtime.make_model", side_effect=cpu_learner), \
                patch("torch.cuda.get_device_properties", side_effect=lambda device: SimpleNamespace(uuid="fixture-" + device)), \
                patch("rollout_manager.backends.vllm_controller.OwnedVLLMActor", AlternatingActor), \
                patch("dsv41_train.rl.runtime.evaluate_policy", side_effect=AssertionError("native evaluation fallback")), \
                patch("dsv41_train.rl.rollout.generate_group", side_effect=AssertionError("native generation fallback")), \
                patch("builtins.print"):
            checkpoint = await asyncio.to_thread(run_training, model_dir=self.root,
                data_path=self.root / "data.jsonl", eval_data_path=validation, config=config)
            first = torch.load(checkpoint, weights_only=True)
            self.assertEqual(first["config"]["rollout_engine"], "vllm")
            self.assertEqual(first["identity"]["rollout_engine"], "vllm")
            self.assertIsNone(first["rollout_rng_state"])
            self.assertTrue(first["optimizer"]["state"])
            self.assertFalse(torch.equal(before, first["policy"]["trainable"]["backend.lm_head.weight"]))
            # Earlier vLLM checkpoints recorded their actor in immutable identity,
            # before RLConfig had an engine field; retain that proven route.
            first["config"].pop("rollout_engine")
            legacy = self.root / "legacy-vllm-checkpoint.pt"
            torch.save(first, legacy)
            resumed = await asyncio.to_thread(run_training, model_dir=self.root,
                data_path=self.root / "data.jsonl", eval_data_path=validation,
                config=replace(config, max_steps=2), resume=legacy)
        second = torch.load(resumed, weights_only=True)
        self.assertEqual(second["step"], 2)
        self.assertGreater(second["next_group_id"], first["next_group_id"])
        self.assertTrue(all(manager.closed for manager in managers))
        rows = [json.loads(line) for line in (self.root / "run/eval-metrics.jsonl").read_text().splitlines()]
        self.assertEqual([(row["phase"], row["step"]) for row in rows],
                         [("baseline", 0), ("eval", 1), ("resume", 1), ("eval", 2)])
        self.assertTrue(all(row["rollout_engine"] == "vllm" for row in rows))


class OwnedActorCleanupTests(unittest.TestCase):
    def test_only_identity_checked_owned_group_is_terminated_without_drain_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary).resolve() / "controller"
            (directory / "vllm-actor").mkdir(parents=True)
            directory.chmod(0o700)
            (directory / "vllm-actor/actor-launch.json").write_text(json.dumps({"actor_process": {"pid": 123}}))
            with patch("rollout_manager.backends.vllm_cleanup.live_owned_process", return_value=True), \
                    patch("rollout_manager.backends.vllm_cleanup.os.getpgid", return_value=999), \
                    patch("rollout_manager.backends.vllm_cleanup.os.killpg") as kill:
                with self.assertRaisesRegex(RuntimeError, "process group changed"):
                    stop_owned_actor(directory)
                kill.assert_not_called()
            running = [True]
            def stop(pid, sig):
                self.assertEqual((pid, sig), (123, signal.SIGTERM))
                running[0] = False
            with patch("rollout_manager.backends.vllm_cleanup.live_owned_process", side_effect=lambda receipt: running[0]), \
                    patch("rollout_manager.backends.vllm_cleanup.os.getpgid", return_value=123), \
                    patch("rollout_manager.backends.vllm_cleanup.os.killpg", side_effect=stop):
                self.assertTrue(stop_owned_actor(directory))
            receipt = json.loads((directory / "manager-forced-actor-exit.json").read_text())
            self.assertFalse(receipt["drain_complete"])
            self.assertFalse(receipt["automatic_replay"])


if __name__ == "__main__":
    unittest.main()
