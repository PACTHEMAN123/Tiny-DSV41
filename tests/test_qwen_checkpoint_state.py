import tempfile
import unittest

import torch

from dsv41_train.checkpoint import CheckpointManager, TrainingState
from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.sharded_state import StateAdapter, TensorShard
from qwen_helpers import tiny_config


class QwenCheckpointStateTest(unittest.TestCase):
    def test_fused_expert_model_and_optimizer_state_use_global_shards(self):
        model = Qwen3MoeForCausalLM(tiny_config())
        # Reproduce one EP rank's local expert storage without communication.
        experts = model.model.layers[0].mlp.experts
        experts.global_num_experts = 8
        experts.expert_start = 4
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        adapter = StateAdapter(model, ep=2)
        state, optim = adapter.encode(model.state_dict(), adapter.optimizer_state(optimizer))
        name = "model.layers.0.mlp.experts.gate_up_proj"
        self.assertIsInstance(state[name], TensorShard)
        self.assertEqual(state[name].shape[0], 8)
        self.assertEqual(state[name].offsets[0], 4)
        self.assertIsInstance(optim["state"][name]["exp_avg"], TensorShard)
        self.assertIsInstance(optim["state"][name]["step"], torch.Tensor)

    def test_unaugmented_qwen_save_and_fresh_optimizer_restore(self):
        torch.manual_seed(10)
        config = tiny_config()
        model = Qwen3MoeForCausalLM(config)
        model.model.layers[0].register_buffer("test_buffer", torch.tensor([3.]))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        expected = {name: value.clone() for name, value in model.state_dict().items()}
        with tempfile.TemporaryDirectory() as folder:
            manager = CheckpointManager(folder, TrainingState(model, optimizer,
                model_config=config.to_dict(), data_generator=torch.Generator(),
                training_config={"objective": "sft"}, state_adapter=StateAdapter(model)))
            manager.save(1)
            restored = Qwen3MoeForCausalLM(config)
            restored.model.layers[0].register_buffer("test_buffer", torch.zeros(1))
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.01)
            loader = CheckpointManager(folder, TrainingState(restored, restored_optimizer,
                model_config=config.to_dict(), data_generator=torch.Generator(),
                training_config={"objective": "sft"}, state_adapter=StateAdapter(restored)))
            self.assertEqual(loader.load(), 1)
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
            for old, new in zip(model.parameters(), restored.parameters()):
                for key, value in optimizer.state[old].items():
                    torch.testing.assert_close(restored_optimizer.state[new][key], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
