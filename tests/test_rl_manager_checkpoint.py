from contextlib import redirect_stdout
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from dsv41_train.rl.checkpoint import (file_sha256, load_policy_artifact,
                                     load_training_checkpoint, publish_policy_artifact,
                                     run_identity, save_training_checkpoint)
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.policy import DSV41Config
from dsv41_train.rl.manager import RolloutManager
from dsv41_train.rl.runtime import (make_model, run_training, validate_device_ownership,
                                   validate_policy_ack)
from dsv41_train.rl.sample import load_checkpoint_model


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "examples/rl/tiny-dsv41"
DATA = ROOT / "examples/rl/prompts.jsonl"


class ManagerCheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = RLConfig(rollout_engine="native-pytorch", seed=37)
        self.model = make_model(MODEL, DSV41Config.from_model_dir(MODEL), self.config, ("cpu",), True)
        self.identity = run_identity(MODEL, DATA, True)
        self.sampling = {"temperature": 1.0, "top_p": 1, "top_k": 0}

    def artifact(self, directory, version=1):
        return publish_policy_artifact(self.model, version, directory, lineage="test-run",
                                       identity=self.identity, sampling=self.sampling)

    def test_provider_ack_must_match_all_published_fields(self):
        artifact = {"version": 1, "digest": "a" * 64, "lineage": "run", "path": "unused"}
        ack = {key: artifact[key] for key in ("version", "digest", "lineage")}
        validate_policy_ack(ack, artifact)
        for field, value in (("version", 2), ("digest", "b" * 64), ("lineage", "other")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "exact published policy"):
                validate_policy_ack({**ack, field: value}, artifact)
        with self.assertRaises(ValueError):
            validate_policy_ack({**ack, "extra": True}, artifact)
        with self.assertRaises(ValueError):
            validate_policy_ack({"version": 1}, artifact)

    def test_cuda_ownership_is_disjoint_while_cpu_sharing_is_allowed(self):
        for learner, rollout in ((("cpu",), ("cpu",)), (("cpu",), ("cuda:0",)),
                                 (("cuda:0",), ("cpu",)), (("cuda:0",), ("cuda:1",))):
            with self.subTest(learner=learner, rollout=rollout):
                validate_device_ownership(learner, rollout)
        with patch("torch.cuda.current_device", return_value=0):
            for learner in (("cuda:0",), ("cuda:1", "cuda:0"), ("cuda",)):
                with self.subTest(learner=learner), self.assertRaisesRegex(ValueError, "separate ownership"):
                    validate_device_ownership(learner, ("cuda:0",))

    def test_checkpoint_barrier_recovers_without_repeating_a_committed_update(self):
        class FailBeforeAck(RolloutManager):
            def activate(self, artifact):
                raise RuntimeError("injected failure before activation ACK")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            config = RLConfig(rollout_engine="native-pytorch", max_steps=2, reward="toy_token_score", queue_capacity=1,
                              output_dir=str(output), worker_timeout_seconds=30,
                              log_samples_per_step=4)
            with patch("dsv41_train.rl.manager.RolloutManager", FailBeforeAck), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "before activation ACK"):
                    run_training(model_dir=MODEL, data_path=DATA, config=config, random_init=True)
            committed = torch.load(output / "step-00000001.pt", weights_only=True)
            self.assertEqual(committed["step"], 1)
            self.assertFalse((output / "step-00000002.pt").exists())
            with redirect_stdout(io.StringIO()):
                run_training(model_dir=MODEL, data_path=DATA, config=config, random_init=True,
                             resume=output / "step-00000001.pt")
            resumed = torch.load(output / "step-00000002.pt", weights_only=True)
            self.assertGreaterEqual(resumed["next_group_id"], committed["next_group_id"])
            self.assertTrue(all(int(state["step"]) == 2 for state in resumed["optimizer"]["state"].values()))
            samples = [json.loads(line) for line in (output / "train-samples.jsonl").read_text().splitlines()]
            self.assertEqual({row["step"] for row in samples}, {2})
            self.assertEqual(len({row["group_id"] for row in samples}), 1)
            self.assertTrue(all(row["group_id"] >= committed["manager_state"]["committed_highwater"]
                                for row in samples))
            self.assertEqual(len(samples), config.group_size)

    def test_checkpoint_failure_prevents_new_policy_activation(self):
        activations = []

        class ObserveActivation(RolloutManager):
            def activate(self, artifact):
                activations.append(artifact["version"])
                return super().activate(artifact)

        with tempfile.TemporaryDirectory() as temporary:
            config = RLConfig(rollout_engine="native-pytorch", max_steps=1, reward="toy_token_score", queue_capacity=1,
                              output_dir=temporary, worker_timeout_seconds=30)
            with patch("dsv41_train.rl.manager.RolloutManager", ObserveActivation), \
                    patch("dsv41_train.rl.runtime.save_training_checkpoint", side_effect=OSError("injected durability failure")), \
                    redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(OSError, "durability failure"):
                    run_training(model_dir=MODEL, data_path=DATA, config=config, random_init=True)
            self.assertEqual(activations, [])
            self.assertFalse(list(Path(temporary).glob("step-*.pt")))

    def test_activation_requires_exact_digest_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = self.artifact(directory)
            self.assertEqual(file_sha256(artifact["path"]), artifact["digest"])
            for field, value in (("digest", "0" * 64), ("version", 2), ("lineage", "other")):
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "mismatch"):
                    load_policy_artifact(self.model, {**artifact, field: value})
            expected = self.model.backend.lm_head.weight.detach().clone()
            with torch.no_grad():
                self.model.backend.lm_head.weight.add_(1)
            ack = load_policy_artifact(self.model, artifact)
            self.assertEqual(ack, {"version": 1, "digest": artifact["digest"], "lineage": "test-run"})
            torch.testing.assert_close(self.model.backend.lm_head.weight, expected, rtol=0, atol=0)
            with Path(artifact["path"]).open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                load_policy_artifact(self.model, artifact)

    def test_format_two_restores_recovery_and_supports_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active_artifact = self.artifact(root / "policies", version=0)
            optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad])
            self.model(torch.tensor([[1, 4, 7]])).square().mean().backward()
            optimizer.step()
            artifact = self.artifact(root / "policies", version=1)
            recovery = {"schema_version": 1, "next_group_id": 4, "rng_state": b"generator",
                        "active_policy": active_artifact, "pending_packets": [{"group_id": 3}]}
            ack = {key: active_artifact[key] for key in ("version", "digest", "lineage")}
            recovery["policy_ack"] = ack
            checkpoint = root / "step-00000001.pt"
            save_training_checkpoint(checkpoint, self.model, optimizer, step=1,
                                     config=self.config, identity=self.identity, next_group_id=4,
                                     rollout_rng_state=b"generator", dropped_groups=2,
                                     manager_state=recovery,
                                     policy_publication={**artifact, "state": "staged", "previous_ack": ack})
            state = load_training_checkpoint(checkpoint, self.model, optimizer, self.config, self.identity)
            self.assertEqual(state["format_version"], 2)
            self.assertEqual(state["manager_state"], recovery)
            self.assertTrue(state["optimizer"]["state"])
            corrupted = {**state, "policy_publication": {**state["policy_publication"], "lineage": "other"}}
            torch.save(corrupted, root / "corrupted.pt")
            with self.assertRaisesRegex(ValueError, "acknowledgement mismatch"):
                load_training_checkpoint(root / "corrupted.pt", self.model, optimizer, self.config, self.identity)
            loaded, _, version, _ = load_checkpoint_model(model_dir=MODEL, checkpoint=checkpoint)
            self.assertEqual(version, 1)
            torch.testing.assert_close(loaded.backend.lm_head.weight, self.model.backend.lm_head.weight,
                                       rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                save_training_checkpoint(checkpoint, self.model, optimizer, step=1,
                                         config=self.config, identity=self.identity, next_group_id=4,
                                         rollout_rng_state=None, dropped_groups=0)

    def test_format_one_remains_readable(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy.pt"
            optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad])
            save_training_checkpoint(path, self.model, optimizer, step=1, config=self.config,
                                     identity=self.identity, next_group_id=1,
                                     rollout_rng_state=None, dropped_groups=0)
            self.assertEqual(load_training_checkpoint(path, self.model, optimizer,
                                                      self.config, self.identity)["format_version"], 1)


if __name__ == "__main__":
    unittest.main()
