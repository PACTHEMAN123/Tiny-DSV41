import unittest

import torch

from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.models.qwen.model import load_balancing_loss
from qwen_helpers import tiny_config


class QwenModelTest(unittest.TestCase):
    def test_forward_backward_and_stacked_expert_parameters(self):
        config = tiny_config()
        model = Qwen3MoeForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        attention_mask = torch.tensor([[1] * 8, [1] * 6 + [0] * 2])

        output = model(
            input_ids,
            attention_mask,
            labels=input_ids,
            output_router_logits=True,
        )

        self.assertEqual(output.logits.shape, (2, 8, config.vocab_size))
        self.assertEqual(len(output.router_logits), config.num_hidden_layers)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        self.assertIsNotNone(model.model.layers[0].mlp.experts.gate_up_proj.grad)
        self.assertIn(
            "model.layers.0.self_attn.q_norm.weight",
            model.state_dict(),
        )
        self.assertIn(
            "model.layers.0.mlp.experts.down_proj",
            model.state_dict(),
        )

    def test_gradient_checkpointing(self):
        config = tiny_config()
        model = Qwen3MoeForCausalLM(config)
        model.gradient_checkpointing_enable()
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (1, 4))

        output = model(input_ids, labels=input_ids, output_router_logits=True)
        output.loss.backward()

        self.assertIsNotNone(model.model.embed_tokens.weight.grad)

    def test_aux_loss_padding_matches_removing_tokens(self):
        torch.manual_seed(13)
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        routers = tuple(torch.randn(6, 4, requires_grad=True) for _ in range(2))
        kept = mask.flatten().bool()
        padded = load_balancing_loss(routers, 4, 2, mask)
        unpadded = load_balancing_loss(tuple(logits[kept] for logits in routers), 4, 2, None)
        torch.testing.assert_close(padded, unpadded)
        padded_grads = torch.autograd.grad(padded, routers, retain_graph=True)
        unpadded_grads = torch.autograd.grad(unpadded, routers)
        for actual, expected in zip(padded_grads, unpadded_grads):
            torch.testing.assert_close(actual, expected)
            self.assertEqual(actual[~kept].count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
