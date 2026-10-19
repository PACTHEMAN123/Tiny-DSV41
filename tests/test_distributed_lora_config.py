"""Configuration determines the actual adapter inventory and transport identity."""
import unittest

from dsv41_train.rl.lora_config import DistributedLoRAConfig, parse_lora_layers


class DistributedLoRAConfigTests(unittest.TestCase):
    def test_experiment_defaults_and_selected_layer_parameter_counts(self):
        config = DistributedLoRAConfig()
        self.assertEqual(config.selected_layers(40), tuple(range(40)))
        self.assertEqual(config.parameter_count(5120, 1280, 40), 2_048_000)
        config = DistributedLoRAConfig(rank=16, alpha=32, layers=(39, 0))
        self.assertEqual(config.selected_layers(40), (0, 39))
        self.assertEqual(config.parameter_count(5120, 1280, 40), 204_800)
        self.assertEqual(DistributedLoRAConfig.from_dict(config.to_dict()), config)

    def test_cli_layer_selection_is_canonical_and_strict(self):
        self.assertIsNone(parse_lora_layers(" all "))
        self.assertEqual(parse_lora_layers(" 3, 0,2 "), (0, 2, 3))
        for value in ("", "0,,1", "0,0", "-1", "1.5", "all,0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_lora_layers(value)

    def test_invalid_settings_rejected_before_model_construction(self):
        for kwargs in ({"rank": True}, {"rank": 7}, {"rank": 256},
                       {"alpha": 0}, {"alpha": float("nan")}, {"alpha": float("inf")},
                       {"alpha": True}, {"layers": ()}, {"layers": (0, 0)},
                       {"layers": (-1,)}, {"layers": (False,)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                DistributedLoRAConfig(**kwargs)
        with self.assertRaises(ValueError):
            DistributedLoRAConfig(layers=(40,)).selected_layers(40)
        for value in ({}, {"rank": 8, "alpha": 8, "layers": None, "targets": ["q_b"]},
                      {"rank": 8, "alpha": 8, "layers": "all"}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                DistributedLoRAConfig.from_dict(value)


if __name__ == "__main__":
    unittest.main()
