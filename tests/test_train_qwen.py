import argparse
import unittest

import train_qwen


def arguments(**overrides):
    values = {
        "steps": 1,
        "batch_size": 1,
        "seq_len": 16,
        "learning_rate": 1.0e-4,
        "cp_size": 2,
        "ep_size": 2,
        "model_path": ".",
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class QwenTrainTest(unittest.TestCase):
    def test_validate_args_accepts_the_eight_rank_hybrid_shape(self):
        train_qwen.validate_args(arguments())

    def test_validate_args_rejects_unequal_cp_and_ep(self):
        with self.assertRaisesRegex(ValueError, "requires cp-size == ep-size"):
            train_qwen.validate_args(arguments(ep_size=1))

    def test_validate_args_rejects_an_unshardable_sequence(self):
        with self.assertRaisesRegex(ValueError, "seq-len must divide"):
            train_qwen.validate_args(arguments(seq_len=15))


if __name__ == "__main__":
    unittest.main()
