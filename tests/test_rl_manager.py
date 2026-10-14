from dataclasses import asdict, replace
import io
from pathlib import Path
import tempfile
import unittest

import torch

from dsv41_train.rl.checkpoint import publish_policy_artifact
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.data import PromptRecord, build_reward
from dsv41_train.rl.manager import RolloutManager
from dsv41_train.rl.policy import DSV41Config, build_dsv41_policy
from dsv41_train.rl.rollout import generate_group, generation_options

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "examples/rl/tiny-dsv41"


class ManagerTests(unittest.TestCase):
    def setup_policy(self, directory):
        torch.set_num_threads(1)
        config = RLConfig(rollout_engine="native-pytorch", group_size=3, max_new_tokens=3, reward="toy_token_score", worker_timeout_seconds=30)
        model_config = DSV41Config.from_model_dir(MODEL)
        torch.manual_seed(config.seed)
        model = build_dsv41_policy(MODEL, model_config, random_init=True,
                                  devices=("cpu",), dtype=torch.float32)
        engram = model.share_engram_memory()
        artifact = publish_policy_artifact(model, 0, directory, lineage="manager-fixture",
            identity={"base": "tiny-fixture"}, sampling={"temperature": 1., "top_p": 1, "top_k": 0})
        record = PromptRecord([1, 2, 3])
        return config, model_config, model, engram, artifact, [record]

    def test_native_sampling_parity_bounded_pause_and_two_activations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, model_config, model, engram, artifact, records = self.setup_policy(root / "policy")
            generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
            expected = generate_group(model, records[0], config.group_size, config.max_new_tokens,
                config.temperature, **generation_options(MODEL, model_config, config),
                policy_version=0, group_id=0, generator=generator, reward_fn=build_reward(config.reward))
            manager = RolloutManager(MODEL, model_config, config, records, True,
                engram_weights=engram, initial_policy=artifact, event_root=root / "events")
            try:
                self.assertEqual(manager.start(), {key: artifact[key] for key in ("version", "digest", "lineage")})
                pid = manager.pid
                packet = manager.get(timeout=10)
                self.assertEqual(asdict(packet["group"]), asdict(expected))
                self.assertEqual(packet["rng_state"], bytes(generator.get_state().tolist()))
                self.assertTrue(packet["engram_shared"])
                self.assertEqual(packet["training_group"].policy.digest, artifact["digest"])
                manager.pause()
                manager.commit([0])
                state = manager.pause()
                self.assertEqual(state["committed_highwater"], 1)
                self.assertLessEqual(len(state["pending_packets"]), config.queue_capacity + config.groups_per_step)
                self.assertTrue(all(p["group"]["group_id"] > 0 for p in state["pending_packets"]))
                buffer = io.BytesIO()
                torch.save(state, buffer)
                buffer.seek(0)
                self.assertEqual(torch.load(buffer, weights_only=True)["next_group_id"], state["next_group_id"])
                for version in (1, 2):
                    next_artifact = publish_policy_artifact(model, version, root / "policy",
                        lineage=artifact["lineage"], identity=artifact["identity"], sampling=artifact["sampling"])
                    ack = manager.activate(next_artifact)
                    self.assertEqual(ack["version"], version)
                    self.assertEqual(ack["digest"], next_artifact["digest"])
                    self.assertEqual(manager.pid, pid)
                manager.resume()
            finally:
                manager.close()
            self.assertFalse(manager.process.is_alive())
            events = (root / "events/events.jsonl").read_text()
            self.assertEqual(events.count('"event": "stream_started"'), 1)
            self.assertIn('"event": "stream_closed"', events)

    def test_saved_complete_packets_survive_resume_without_reissuing_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, model_config, model, engram, artifact, records = self.setup_policy(root / "policy")
            manager = RolloutManager(MODEL, model_config, config, records, True,
                                     engram_weights=engram, initial_policy=artifact)
            try:
                manager.start()
                first = manager.get(10)
                manager.pause()
                manager.commit([first["group"].group_id])
                state = manager.pause()
            finally:
                manager.close()
            restored = RolloutManager(MODEL, model_config, config, records, True,
                engram_weights=engram, initial_policy=artifact, recovery_state=state)
            try:
                restored.start()
                packet = restored.get(10)
                self.assertEqual(packet["group"].group_id, state["committed_highwater"])
                if state["pending_packets"]:
                    self.assertEqual(asdict(packet["group"]), state["pending_packets"][0]["group"])
                    self.assertEqual(packet["rng_state"], state["pending_packets"][0]["rng_state"])
                restored.pause()
                restored.commit([packet["group"].group_id])
                final = restored.pause()
                self.assertNotEqual(final["resume_epoch"], state["resume_epoch"])
                self.assertGreater(final["committed_highwater"], first["group"].group_id)
                self.assertGreaterEqual(final["next_group_id"], state["next_group_id"])
            finally:
                restored.close()

    def test_failed_child_preserves_error_and_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, model_config, model, engram, artifact, records = self.setup_policy(Path(temporary) / "policy")
            config = replace(config, reward="operator:truediv")
            manager = RolloutManager(MODEL, model_config, config, records, True,
                                     engram_weights=engram, initial_policy=artifact)
            try:
                manager.start()
                with self.assertRaisesRegex(RuntimeError, "rollout worker failed"):
                    manager.get(10)
            finally:
                manager.close()
                manager.close()
            self.assertFalse(manager.process.is_alive())


if __name__ == "__main__":
    unittest.main()
