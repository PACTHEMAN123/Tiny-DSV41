import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from dsv41_train.checkpoint import CheckpointManager, TrainingState


class TrainingStateTest(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(2, 2)
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.generator = torch.Generator().manual_seed(123)
        self.state = TrainingState(
            self.model,
            self.optimizer,
            model_config={"hidden_size": 2},
            data_generator=self.generator,
        )

    @patch("dsv41_train.checkpoint.get_state_dict")
    def test_state_dict_collects_resumable_training_state(self, get_state_dict):
        get_state_dict.return_value = ({"weight": Mock()}, {"state": Mock()})
        self.state.step = 7

        state_dict = self.state.state_dict()

        self.assertEqual(state_dict["step"], 7)
        self.assertEqual(state_dict["model_config"], {"hidden_size": 2})
        self.assertEqual(state_dict["data_generator"]["world_size"], 1)
        self.assertTrue(
            torch.equal(
                state_dict["data_generator"]["rank-0"], self.generator.get_state()
            )
        )
        get_state_dict.assert_called_once_with(self.model, self.optimizer)

    @patch("dsv41_train.checkpoint.set_state_dict")
    def test_load_state_dict_restores_all_mutable_state(self, set_state_dict):
        restored_generator = torch.Generator().manual_seed(456)
        generator_state = restored_generator.get_state()

        self.state.load_state_dict(
            {
                "model": {"weight": Mock()},
                "optimizer": {"state": Mock()},
                "model_config": {"hidden_size": 2},
                "data_generator": {
                    "world_size": 1,
                    "rank-0": generator_state,
                },
                "step": 11,
            }
        )

        self.assertEqual(self.state.step, 11)
        self.assertTrue(torch.equal(self.generator.get_state(), generator_state))
        set_state_dict.assert_called_once()

    def test_load_state_dict_rejects_a_different_model_config(self):
        with self.assertRaisesRegex(ValueError, "model config"):
            self.state.load_state_dict({"model_config": {"hidden_size": 4}})


class CheckpointManagerTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.folder = Path(self.temporary_directory.name) / "checkpoints"
        self.state = Mock(spec=TrainingState)
        self.state.step = 0
        self.manager = CheckpointManager(self.folder, self.state)

    @patch("dsv41_train.checkpoint.dcp.save")
    def test_save_uses_a_step_directory(self, save):
        checkpoint_path = self.manager.save(12)

        self.assertEqual(checkpoint_path, self.folder / "step-12")
        self.assertEqual(self.state.step, 12)
        save.assert_called_once_with(
            {"train": self.state}, checkpoint_id=str(checkpoint_path)
        )

    def test_latest_step_ignores_incomplete_checkpoints(self):
        for step in (2, 10):
            checkpoint = self.folder / f"step-{step}"
            checkpoint.mkdir(parents=True)
            (checkpoint / ".metadata").touch()
        (self.folder / "step-20").mkdir()
        (self.folder / "notes").mkdir()

        self.assertEqual(self.manager.latest_step(), 10)

    @patch("dsv41_train.checkpoint.dcp.load")
    def test_load_defaults_to_the_latest_complete_checkpoint(self, load):
        checkpoint = self.folder / "step-8"
        checkpoint.mkdir(parents=True)
        (checkpoint / ".metadata").touch()
        load.side_effect = lambda *_args, **_kwargs: setattr(self.state, "step", 8)

        loaded_step = self.manager.load()

        self.assertEqual(loaded_step, 8)
        load.assert_called_once_with(
            {"train": self.state}, checkpoint_id=str(checkpoint)
        )


if __name__ == "__main__":
    unittest.main()
