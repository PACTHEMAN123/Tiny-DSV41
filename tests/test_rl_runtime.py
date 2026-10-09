import fcntl
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.runtime import policy_lag


ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def test_staleness_is_bounded_and_future_policy_rejected(self):
        self.assertEqual(policy_lag(4, 2, 2), 2)
        self.assertIsNone(policy_lag(4, 1, 2))
        self.assertEqual(policy_lag(4, 4, 0), 0)
        with self.assertRaises(ValueError):
            policy_lag(4, 5, 2)

    def test_config_rejects_invalid_values(self):
        for values in ({"group_size": 1}, {"max_policy_lag": -1}, {"queue_capacity": 0},
                       {"temperature": float("nan")}, {"kl_beta": -0.1},
                       {"learner_devices": ["cuda:0", "cuda:0"]}, {"max_steps": 1.2}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RLConfig.from_dict(values)

    def run_training(self, root, *extra, reward="toy_token_score", kl_beta=0):
        config = root / "rl.json"
        config.write_text(json.dumps({
            "max_steps": 2, "group_size": 3, "max_new_tokens": 3,
            "reward": reward, "kl_beta": kl_beta,
            "output_dir": str(root / "output"), "worker_timeout_seconds": 30,
        }))
        return subprocess.run(
            [sys.executable, "-m", "dsv41_train.rl.train", "--model-dir",
             str(ROOT / "examples/rl/tiny-dsv41"), "--data", str(ROOT / "examples/rl/prompts.jsonl"),
             "--config", str(config), "--allow-random-init", *extra],
            cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)},
            text=True, capture_output=True, timeout=90,
        )

    def assert_lock_released(self, root):
        with (root / "output/.run.lock").open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle, fcntl.LOCK_UN)

    def test_random_base_resume_and_output_conflicts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = self.run_training(root)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            metrics = [json.loads(line) for line in result.stdout.splitlines()
                       if line.startswith("{") and json.loads(line)["event"] == "train"]
            self.assertTrue(all(event["policy_lag_max"] <= 2 for event in metrics))
            self.assertTrue(all(event["engram_shared"] and event["rollout_engram_shared"] for event in metrics))
            self.assertTrue(any(event["nonzero_advantages"] > 0 for event in metrics))
            checkpoint = root / "output/step-00000002.pt"
            before = torch.load(checkpoint, weights_only=True)
            result = self.run_training(root, "--resume", str(checkpoint), "--max-steps", "3")
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            after = torch.load(root / "output/step-00000003.pt", weights_only=True)
            self.assertGreater(after["next_group_id"], before["next_group_id"])
            self.assert_lock_released(root)
            conflict = self.run_training(root, "--resume", str(checkpoint), "--max-steps", "4")
            self.assertNotEqual(conflict.returncode, 0)
            self.assertIn("output_dir contains later checkpoints", conflict.stderr)
            changed = self.run_training(root, "--resume", str(checkpoint), "--max-steps", "4", kl_beta=0.02)
            self.assertNotEqual(changed.returncode, 0)
            self.assertIn("resume cannot change algorithm setting kl_beta", changed.stderr)

    def test_reward_failure_propagates_and_releases_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            # This plugin imports in the learner, then raises when called by rollout.
            result = self.run_training(root, reward="operator:truediv")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("rollout worker failed", result.stderr)
            self.assert_lock_released(root)
