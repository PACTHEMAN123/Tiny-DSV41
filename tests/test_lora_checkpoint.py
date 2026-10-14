"""Compact LoRA DCP must restore training state without storing frozen weights."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed.checkpoint as dcp
from torch import nn

from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.lora import LoRAConfig, inject_lora
from dsv41_train.models.dsv4 import load_dsv41_backbone_window
from dsv41_train.optimization import OptimizerPolicy, build_optimizer
from dsv41_train.sharded_state import StateAdapter
from post_training_helpers import small_config
from test_dsv4_loading import synthetic_checkpoint
from train_sft import _source_identity


class BufferedLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.large_base = nn.Parameter(torch.ones(2_000_000))
        self.projection = nn.Linear(8, 4)
        self.register_buffer("counter", torch.tensor(0))
        self.register_buffer("temporary", torch.zeros(123), persistent=False)
        inject_lora(self, LoRAConfig(rank=2, targets=("projection",)))


class CompactCheckpointTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)
        self.identity = {"path": "/immutable/base", "config_sha256": "config-hash", "index_sha256": "index-hash"}

    def state(self, model, optimizer=None, *, source=None, mode="lora"):
        optimizer = optimizer or build_optimizer(model, OptimizerPolicy("adamw", .01))
        return TrainingState(model, optimizer, model_config={}, data_generator=torch.Generator(),
                             training_config={"source": source or self.identity},
                             state_adapter=StateAdapter(model), checkpoint_state=mode)

    def test_only_adapters_and_persistent_buffers_are_stored_with_bounded_file_size(self):
        model = BufferedLoRA()
        state = self.state(model)
        model.projection(torch.ones(2, 8)).square().mean().backward()
        state.optimizer.step()
        model.counter.add_(7)
        path = CheckpointManager(self.folder, state).save(3)
        metadata = dcp.FileSystemReader(str(path)).read_metadata()
        keys = {name for name in metadata.state_dict_metadata if name.startswith("train.model.")}
        self.assertEqual(keys, {"train.model.projection.lora_a", "train.model.projection.lora_b", "train.model.counter"})
        size = sum(file.stat().st_size for file in path.iterdir() if file.is_file())
        self.assertLess(size, 200_000)
        self.assertGreater(model.large_base.numel() * model.large_base.element_size(), size * 20)
        restored = BufferedLoRA()
        # This intentionally distinct frozen value proves compact load does not
        # silently restore base weights; production reconstructs the source base.
        with torch.no_grad():
            restored.large_base.fill_(3)
        restored_state = self.state(restored)
        self.assertEqual(CheckpointManager(self.folder, restored_state).load(), 3)
        self.assertEqual(restored.counter.item(), 7)
        self.assertTrue(restored.large_base.eq(3).all())
        torch.testing.assert_close(restored.projection.lora_a, model.projection.lora_a, rtol=0, atol=0)
        torch.testing.assert_close(restored.projection.lora_b, model.projection.lora_b, rtol=0, atol=0)

    def test_missing_extra_keys_or_wrong_source_fail_before_partial_load(self):
        model = BufferedLoRA()
        state = self.state(model)
        original = state.state_dict()
        for corrupt in ("missing", "extra", "optimizer_schema"):
            with self.subTest(corrupt=corrupt):
                saved = copy.deepcopy(original)
                if corrupt == "missing":
                    saved["model"].pop("projection.lora_a")
                elif corrupt == "extra":
                    saved["model"]["large_base"] = model.large_base.detach()
                else:
                    saved["optimizer_schema"]["rank-0"] = json.dumps({"large_base": {}})
                folder = self.folder / corrupt / "step-1"
                dcp.save({"train": saved}, checkpoint_id=folder)
                message = "optimizer schema" if corrupt == "optimizer_schema" else "model key set"
                with self.assertRaisesRegex(ValueError, message):
                    CheckpointManager(folder.parent, self.state(BufferedLoRA())).load()
        CheckpointManager(self.folder / "valid", state).save(1)
        wrong_source = dict(self.identity, config_sha256="different")
        with self.assertRaisesRegex(ValueError, "configuration is incompatible"):
            CheckpointManager(self.folder / "valid", self.state(BufferedLoRA(), source=wrong_source)).load()
        with self.assertRaisesRegex(ValueError, "configuration is incompatible"):
            CheckpointManager(self.folder / "valid", self.state(BufferedLoRA(), mode="full")).load()
        changed_adapter = BufferedLoRA()
        changed_adapter.projection.config = LoRAConfig(rank=2, alpha=7, targets=("projection",))
        with self.assertRaisesRegex(ValueError, "configuration is incompatible"):
            CheckpointManager(self.folder / "valid", self.state(changed_adapter)).load()

    def test_invalid_training_ownership_and_absent_base_identity_are_rejected(self):
        for invalid in ("unfrozen", "frozen_adapter", "extra_optimizer", "missing_identity", "no_adapter"):
            with self.subTest(invalid=invalid):
                model = BufferedLoRA() if invalid != "no_adapter" else nn.Linear(2, 2)
                optimizer = build_optimizer(model, OptimizerPolicy("adamw", .01))
                if invalid == "unfrozen":
                    model.large_base.requires_grad_(True)
                elif invalid == "frozen_adapter":
                    model.projection.lora_a.requires_grad_(False)
                elif invalid == "extra_optimizer":
                    optimizer.param_groups[0]["params"].append(model.large_base)
                source = {"path": "incomplete"} if invalid == "missing_identity" else None
                with self.assertRaises(ValueError):
                    self.state(model, optimizer, source=source)

    def test_dsv4_quantized_base_reconstruction_engram_buffers_dropout_and_rng_resume(self):
        base = self.folder / "base"
        base.mkdir()
        config = small_config(num_hidden_layers=2, compress_ratios=[0, 2],
                              kv_source_layer_ids=[1], index_source_layer_ids=[1],
                              candidate_source_layer_id=1, q_lora_rank=32, o_lora_rank=32,
                              engram_head_dim=32)
        torch.manual_seed(12)
        synthetic_checkpoint(base, config)
        source = _source_identity(base)
        def setup():
            model = load_dsv41_backbone_window(base, start_layer=0, num_layers=2,
                       dtype=torch.float32, lora_config=LoRAConfig(rank=2, alpha=4, dropout=.2))
            self.addCleanup(model.close)
            model.register_buffer("training_counter", torch.tensor(0))
            optimizer = build_optimizer(model, OptimizerPolicy("adamw", .01))
            state = TrainingState(model, optimizer, model_config=model.config.to_dict(),
                                  data_generator=torch.Generator().manual_seed(415),
                                  training_config={"source": source, "case": "dsv4-engram-lora"},
                                  state_adapter=StateAdapter(model), checkpoint_state="lora")
            return model, optimizer, state
        def update(model, optimizer, state):
            ids = torch.randint(3, 32, (2, 8), generator=state.data_generator)
            optimizer.zero_grad(set_to_none=True)
            loss = model(ids, labels=ids).loss
            loss.backward()
            optimizer.step()
            model.training_counter.add_(1)
            return loss.detach()
        model, optimizer, state = setup()
        update(model, optimizer, state)
        path = CheckpointManager(self.folder / "run", state).save(1)
        expected_loss = update(model, optimizer, state)
        expected_rng = torch.get_rng_state().clone()
        restored, restored_optimizer, restored_state = setup()
        self.assertEqual(CheckpointManager(self.folder / "run", restored_state).load(), 1)
        actual_loss = update(restored, restored_optimizer, restored_state)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        torch.testing.assert_close(restored_state.data_generator.get_state(), state.data_generator.get_state(), rtol=0, atol=0)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[name], rtol=0, atol=0)
        self.assertEqual(restored.training_counter.item(), 2)
        keys = dcp.FileSystemReader(str(path)).read_metadata().state_dict_metadata
        self.assertNotIn("train.model.model.engram_tables.1.weight", keys)
        self.assertIn("train.model.training_counter", keys)
        self.assertTrue(all(parameter.grad is None for parameter in restored.parameters() if not parameter.requires_grad))


if __name__ == "__main__":
    unittest.main()
