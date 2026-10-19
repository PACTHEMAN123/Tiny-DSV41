"""CPU-only lifetime/configuration regressions."""
import time
import unittest

from rollout_manager.backends.vllm_actor import actor_lifetime_deadline
from rollout_manager.backends.vllm_lora_controller import LoRARolloutConfig


class LoRAConfigTests(unittest.TestCase):
    def test_optional_overall_deadline_is_independent_of_operation_timeout(self):
        config = LoRARolloutConfig(operation_timeout_seconds=600)
        config.validate()
        bootstrap = {"serving_profile": "deepseek-lora-v1", "ttl_seconds": 600,
                     "deadline_unix": None}
        self.assertIsNone(actor_lifetime_deadline(bootstrap, now_unix=100, now_monotonic=20))
        bootstrap["deadline_unix"] = 10000
        self.assertEqual(actor_lifetime_deadline(bootstrap, now_unix=100, now_monotonic=20), 9920)
        bootstrap.pop("serving_profile")
        self.assertEqual(actor_lifetime_deadline(bootstrap, now_unix=100, now_monotonic=20), 620)

    def test_invalid_deadline_devices_and_operation_limits_are_rejected(self):
        for values in ({"deadline_unix": time.time() - 1}, {"deadline_unix": float("inf")},
                       {"operation_timeout_seconds": 0}, {"operation_timeout_seconds": 7201},
                       {"rollout_devices": ("cuda:0", "cuda:0")}, {"max_num_seqs": 33}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                LoRARolloutConfig(**values).validate()


if __name__ == "__main__":
    unittest.main()
