import argparse
import os
import unittest
from unittest.mock import patch

import torch

import train


class TrainTest(unittest.TestCase):
    def test_make_batch_is_reproducible_with_an_explicit_generator(self):
        first = torch.Generator().manual_seed(123)
        second = torch.Generator().manual_seed(123)

        first_batch = train.make_batch(3, 8, torch.device("cpu"), generator=first)
        second_batch = train.make_batch(3, 8, torch.device("cpu"), generator=second)

        self.assertTrue(torch.equal(first_batch, second_batch))

    def test_validate_args_rejects_invalid_values(self):
        args = argparse.Namespace(steps=0, batch_size=1, log_every=1, seq_len=8)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            train.validate_args(args)

    @patch("train.dist.get_world_size", return_value=4)
    @patch("train.dist.get_rank", return_value=2)
    @patch("train.dist.init_process_group")
    @patch("train.dist.is_available", return_value=True)
    @patch("train.torch.cuda.set_device")
    @patch("train.torch.cuda.device_count", return_value=4)
    @patch("train.torch.cuda.is_available", return_value=True)
    def test_torchrun_environment_binds_the_local_gpu(
        self,
        _is_available,
        _device_count,
        set_device,
        _dist_available,
        init_process_group,
        _get_rank,
        _get_world_size,
    ):
        environment = {"WORLD_SIZE": "4", "LOCAL_RANK": "2", "RANK": "2"}
        with patch.dict(os.environ, environment, clear=True):
            runtime = train.initialize_runtime("auto")

        self.assertEqual(runtime.device, torch.device("cuda:2"))
        self.assertEqual(runtime.rank, 2)
        self.assertEqual(runtime.world_size, 4)
        set_device.assert_called_once_with(2)
        init_process_group.assert_called_once_with(backend="nccl")

    def test_distributed_training_rejects_a_pinned_device(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=True):
            with self.assertRaisesRegex(ValueError, "assigns devices automatically"):
                train.initialize_runtime("cuda:0")


if __name__ == "__main__":
    unittest.main()
