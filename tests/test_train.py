import argparse
import os
import unittest
from unittest.mock import patch

import torch

import train
from dsv41_train import runtime as training_runtime


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

    @patch("dsv41_train.runtime.dist.get_world_size", return_value=4)
    @patch("dsv41_train.runtime.dist.get_rank", return_value=2)
    @patch("dsv41_train.runtime.dist.init_process_group")
    @patch("dsv41_train.runtime.dist.is_available", return_value=True)
    @patch("dsv41_train.runtime.torch.cuda.set_device")
    @patch("dsv41_train.runtime.torch.cuda.device_count", return_value=4)
    @patch("dsv41_train.runtime.torch.cuda.is_available", return_value=True)
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
            runtime = training_runtime.initialize_runtime("auto")

        self.assertEqual(runtime.device, torch.device("cuda:2"))
        self.assertEqual(runtime.rank, 2)
        self.assertEqual(runtime.world_size, 4)
        set_device.assert_called_once_with(2)
        init_process_group.assert_called_once_with(backend="nccl")

    def test_distributed_training_rejects_a_pinned_device(self):
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}, clear=True):
            with self.assertRaisesRegex(ValueError, "assigns devices automatically"):
                training_runtime.initialize_runtime("cuda:0")

    def test_cpu_training_needs_no_process_group(self):
        with patch.dict(os.environ, {}, clear=True):
            runtime = training_runtime.initialize_runtime("cpu")
        self.assertEqual(runtime.device, torch.device("cpu"))
        self.assertFalse(runtime.distributed)

    @patch("dsv41_train.runtime.torch.cuda.is_available", return_value=True)
    def test_single_rank_fsdp_requires_torchrun(self, _cuda_available):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "launch with torchrun"):
                training_runtime.initialize_runtime(require_distributed=True)

    def test_metric_reduction_does_not_modify_the_loss(self):
        runtime = training_runtime.Runtime(torch.device("cpu"), world_size=2)
        value = torch.tensor(3.0, requires_grad=True)
        with patch(
            "dsv41_train.runtime.dist.all_reduce",
            side_effect=lambda tensor, **_: tensor.mul_(2),
        ):
            self.assertEqual(training_runtime.distributed_mean(value, runtime), 3.0)
        self.assertEqual(value.item(), 3.0)


if __name__ == "__main__":
    unittest.main()
