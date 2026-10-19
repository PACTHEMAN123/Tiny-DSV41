import copy
from dataclasses import replace
import unittest

import torch

from dsv41_train.lora import LoRAConfig, inject_lora
from dsv41_train.models.dsv4 import DeepSeekV41Config, DeepSeekV41ForCausalLM
from dsv41_train.models.qwen import Qwen3MoeConfig, Qwen3MoeForCausalLM
from dsv41_train.sft_eval import evaluate_answers, greedy_generate


def tiny_model(family):
    torch.manual_seed(91)
    if family == "dsv4":
        config = replace(DeepSeekV41Config.tiny(), hidden_size=16, num_attention_heads=2,
                         head_dim=8, q_lora_rank=8, qk_rope_head_dim=4, index_n_heads=2,
                         index_head_dim=8, o_groups=1, o_lora_rank=8, moe_intermediate_size=32,
                         n_routed_experts=4, hc_mult=2, hc_sinkhorn_iters=5, engram_head_dim=4)
        return DeepSeekV41ForCausalLM(config)
    return Qwen3MoeForCausalLM(Qwen3MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=24, moe_intermediate_size=12,
        num_hidden_layers=2, max_window_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, num_experts=4, num_experts_per_tok=2,
        bos_token_id=1, eos_token_id=2, max_position_embeddings=64,
    ))


class NumericTokenizer:
    eos_token_ids = ()

    def decode(self, ids, *, skip_special_tokens=True):
        return " ".join(map(str, ids))


class SFTEvaluationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_native_models_match_direct_greedy_and_stop_at_eos(self):
        for family in ("dsv4", "qwen"):
            with self.subTest(family=family):
                model = tiny_model(family)
                if family == "dsv4":
                    inject_lora(model, LoRAConfig(rank=2, alpha=4, dropout=.25))
                    model.model.layers[0].attention.q_a.dropout.eval()
                    with torch.no_grad():
                        for name, parameter in model.named_parameters():
                            if name.endswith("lora_b"):
                                parameter.normal_(std=.02)
                reference = copy.deepcopy(model).eval()
                flags = [module.training for module in model.modules()]
                prompt, expected = [1, 4, 5], []
                with torch.no_grad():
                    for _ in range(3):
                        options = {"compute_aux_loss": False} if family == "dsv4" else {"output_router_logits": False}
                        logits = reference(torch.tensor([prompt + expected]), **options).logits
                        expected.append(int(logits[0, -1].argmax()))
                actual = greedy_generate(model, prompt, family=family, max_new_tokens=3)
                self.assertEqual(actual, expected)
                self.assertEqual([module.training for module in model.modules()], flags)
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
                self.assertEqual(greedy_generate(model, prompt, family=family,
                                                eos_token_ids=(actual[0],), max_new_tokens=3), actual[:1])

    def test_answer_evaluation_never_feeds_ground_truth_and_records_tokens(self):
        model = tiny_model("dsv4")
        inject_lora(model, LoRAConfig(rank=2, alpha=4))
        prompt = [1, 4]
        generated = greedy_generate(model, prompt, family="dsv4", max_new_tokens=2)
        example = {"prompt_ids": prompt, "input_ids": [model.config.vocab_size + 10],
                   "labels": [-100, *generated]}
        report = evaluate_answers(model, [example], NumericTokenizer(), family="dsv4", max_new_tokens=2)
        self.assertEqual(report["exact_match_rate"], 1)
        self.assertEqual(report["examples"][0]["generated_ids"], generated)
        self.assertEqual(report["examples"][0]["target_ids"], generated)
        self.assertEqual(example["prompt_ids"], prompt)

    def test_generation_validates_inputs_and_restores_mode_on_failure(self):
        model = tiny_model("qwen")
        model.model.layers[0].eval()
        flags = [module.training for module in model.modules()]
        with self.assertRaisesRegex(ValueError, "capacity"):
            greedy_generate(model, [1] * 65, family="qwen", max_new_tokens=1)
        self.assertEqual([module.training for module in model.modules()], flags)
        for prompt, options in (([], {}), ([1], {"max_new_tokens": -1}),
                                ([1], {"eos_token_ids": (-1,)})):
            with self.subTest(prompt=prompt, options=options), self.assertRaises(ValueError):
                greedy_generate(model, prompt, family="qwen", **options)
        self.assertEqual(greedy_generate(model, [1], family="qwen", max_new_tokens=0), [])
        with torch.no_grad():
            model.lm_head.weight.fill_(float("nan"))
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            greedy_generate(model, [1], family="qwen", max_new_tokens=1)
        self.assertEqual([module.training for module in model.modules()], flags)


if __name__ == "__main__":
    unittest.main()
