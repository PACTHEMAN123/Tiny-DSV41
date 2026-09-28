"""Exercise real DCP IO, including rejected restores before tensor mutation."""

import copy
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.checkpoint_layout import checkpoint_identity
from dsv41_train.lora import LoRAConfig, adapter_parameters, inject_lora
from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM
from lora_helpers import small_config, synthetic_checkpoint


def make_state(*, alpha=4, dropout=0.1, mode="full", identity=None):
    model = nn.Sequential(nn.Linear(64, 64))
    config = LoRAConfig(rank=2, alpha=alpha, dropout=dropout, targets=("0",))
    inject_lora(model, config)
    optimizer = torch.optim.AdamW(adapter_parameters(model), lr=0.01)
    return TrainingState(model, optimizer, model_config={"width": 64},
                         data_generator=torch.Generator().manual_seed(123),
                         training_config={"lora": asdict(config)},
                         checkpoint_mode=mode, base_model_identity=identity)


def update(state):
    state.optimizer.zero_grad(set_to_none=True)
    inputs = torch.randn(3, 64, generator=state.data_generator)
    loss = state.model(inputs).square().mean()
    loss.backward()
    state.optimizer.step()
    return loss.detach().clone()


class CheckpointResumeTest(unittest.TestCase):
    def test_rejects_config_mismatches_before_mutating_any_tensor_or_rng(self):
        source = make_state(identity={"revision": "a"})
        update(source)
        with tempfile.TemporaryDirectory() as directory:
            CheckpointManager(directory, source).save(1)
            for case in ("alpha", "dropout", "model", "identity", "mode", "optimizer", "optimizer_options"):
                with self.subTest(case=case):
                    target = make_state(alpha=8 if case == "alpha" else 4,
                                        dropout=0.2 if case == "dropout" else 0.1,
                                        mode="trainable" if case == "mode" else "full",
                                        identity={"revision": "b" if case == "identity" else "a"})
                    if case == "model":
                        target.model_config["width"] = 128
                    if case == "optimizer":
                        target.optimizer = torch.optim.SGD(adapter_parameters(target.model), lr=0.01)
                    if case == "optimizer_options":
                        target.optimizer.param_groups[0]["amsgrad"] = True
                    expected_config = copy.deepcopy(target.training_config)
                    before = copy.deepcopy(target.model.state_dict())
                    rng = torch.get_rng_state().clone()
                    generator = target.data_generator.get_state().clone()
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        CheckpointManager(directory, target).load()
                    self.assertEqual(target.training_config, expected_config)
                    self.assertEqual(target.step, 0)
                    self.assertFalse(target.optimizer.state)
                    for name, tensor in target.model.state_dict().items():
                        torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
                    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                    torch.testing.assert_close(target.data_generator.get_state(), generator, rtol=0, atol=0)

    def test_actual_module_hyperparameters_are_checked_without_training_config(self):
        source, target = make_state(), make_state(alpha=8)
        source.training_config = target.training_config = None
        update(source)
        with tempfile.TemporaryDirectory() as directory:
            CheckpointManager(directory, source).save(1)
            with self.assertRaisesRegex(ValueError, "module metadata"):
                CheckpointManager(directory, target).load()

    def test_global_layer_window_is_checked(self):
        config = small_config()
        states = []
        for layer_ids in ([1], [3]):
            model = DeepSeekV41ForCausalLM(config, layer_ids=layer_ids)
            inject_lora(model, LoRAConfig())
            states.append(TrainingState(model, torch.optim.SGD(adapter_parameters(model), lr=0.1),
                                        model_config=config.to_dict(), data_generator=torch.Generator()))
        before = copy.deepcopy(states[1].model.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            CheckpointManager(directory, states[0]).save(0)
            with self.assertRaisesRegex(ValueError, "layer ids"):
                CheckpointManager(directory, states[1]).load()
        for name, tensor in states[1].model.state_dict().items():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)

    def test_full_and_trainable_resume_into_fresh_optimizer_are_exact(self):
        for mode in ("full", "trainable"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                torch.manual_seed(42)
                source = make_state(mode=mode, identity={"initialization": 42})
                base = copy.deepcopy(source.model.state_dict())
                update(source)
                manager = CheckpointManager(directory, source)
                manager.save(1)
                expected_loss = update(source)
                expected = copy.deepcopy(source.model.state_dict())
                target = make_state(mode=mode, identity={"initialization": 42})
                target.model.load_state_dict(base)
                self.assertFalse(target.optimizer.state)
                self.assertEqual(CheckpointManager(directory, target).load(), 1)
                torch.testing.assert_close(update(target), expected_loss, rtol=0, atol=0)
                for name, tensor in target.model.state_dict().items():
                    torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)

    def test_trainable_checkpoint_omits_frozen_parameters_on_disk(self):
        sizes = {}
        for mode in ("full", "trainable"):
            with tempfile.TemporaryDirectory() as directory:
                state = make_state(mode=mode, identity={"revision": "a"})
                update(state)
                path = CheckpointManager(directory, state).save(1)
                keys = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
                self.assertEqual("train.model.0.weight" in keys, mode == "full")
                self.assertIn("train.model.0.lora_a", keys)
                sizes[mode] = sum(p.stat().st_size for p in path.iterdir())
        self.assertLess(sizes["trainable"], sizes["full"])
        with self.assertRaisesRegex(ValueError, "base_model_identity"):
            make_state(mode="trainable")

    def test_legacy_checkpoint_is_explicitly_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "step-1"
            dcp.save({"train": {"model": {"weight": torch.ones(1)}}}, checkpoint_id=path)
            with self.assertRaisesRegex(ValueError, "legacy"):
                CheckpointManager(directory, make_state()).load()

    def test_adam_resume_with_unused_parameters_and_empty_optimizer(self):
        for warm in (False, True):
            with self.subTest(warm=warm), tempfile.TemporaryDirectory() as directory:
                model = nn.ModuleDict({"used": nn.Linear(4, 3), "unused": nn.Linear(4, 3)})
                optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
                state = TrainingState(model, optimizer, model_config={}, data_generator=torch.Generator())

                def step(candidate):
                    candidate.optimizer.zero_grad(set_to_none=True)
                    candidate.model["used"](torch.ones(2, 4)).square().mean().backward()
                    candidate.optimizer.step()

                if warm:
                    step(state)
                manager = CheckpointManager(directory, state)
                manager.save(int(warm))
                self.assertEqual(len(optimizer.state), 2 if warm else 0)
                step(state)
                expected = copy.deepcopy(model.state_dict())
                restored = nn.ModuleDict({"used": nn.Linear(4, 3), "unused": nn.Linear(4, 3)})
                target = TrainingState(restored, torch.optim.AdamW(restored.parameters(), lr=0.01),
                                       model_config={}, data_generator=torch.Generator())
                CheckpointManager(directory, target).load()
                self.assertEqual(len(target.optimizer.state), 2 if warm else 0)
                step(target)
                for name, tensor in restored.state_dict().items():
                    torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)

    def test_base_file_identity_detects_replaced_config_and_weight_files(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            synthetic_checkpoint(folder)
            identity = checkpoint_identity(folder)
            self.assertEqual(identity, checkpoint_identity(folder))
            path = folder / "config.json"
            config = json.loads(path.read_text())
            config["initializer_range"] = 0.03
            path.write_text(json.dumps(config))
            self.assertNotEqual(identity, checkpoint_identity(folder))
            identity = checkpoint_identity(folder)
            with (folder / "model.safetensors").open("ab") as stream:
                stream.write(b"changed")
            self.assertNotEqual(identity, checkpoint_identity(folder))


if __name__ == "__main__":
    unittest.main()
