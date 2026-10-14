"""Actual deterministic token generation with native LoRA, without CUDA."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from dsv41_train.lora import LoRAConfig, LoRALinear
from dsv41_train.runtime import Runtime
from scripts.evaluate_identity_qa import (capture_identity_generation, compare_identity_generations,
    compare_reloaded_generation, evaluation_mode, target_nll, file_sha256, load_prompt_spec)


class Decoder:
    def decode(self, ids, skip_special_tokens=True):
        text = {0: '' if skip_special_tokens else '<eos>', 1: '你是谁', 2: 'RTP大模型', 3: '其他模型', 4: ' ', 5: '?'}
        return ''.join(text[i] for i in ids)


class TinyAnswerModel(nn.Module):
    def __init__(self):
        super().__init__()
        base = nn.Linear(6, 6, bias=False)
        with torch.no_grad():
            base.weight.zero_()
            base.weight[3, 1] = 5
            base.weight[0, 3] = 5
            base.weight[0, 2] = 5
        self.proj = LoRALinear(base, LoRAConfig(rank=1, alpha=1, dropout=.5))
        with torch.no_grad():
            self.proj.lora_a.zero_()
            self.proj.lora_a[0, 1] = 1
            self.proj.lora_b.zero_()
            self.proj.lora_b[2, 0] = 10
        self.context_parallel = SimpleNamespace(enabled=False)

    def forward(self, input_ids, attention_mask):
        assert attention_mask.all()
        return SimpleNamespace(logits=self.proj(F.one_hot(input_ids, 6).float()))


def specification():
    return {'format_version': 1, 'eos_token_ids': [0], 'prompts': [
        {'id': 'direct', 'input_ids': [1], 'prompt_text': '你是谁', 'target_ids': [2, 0],
         'target_text': 'RTP大模型', 'split': 'heldout', 'seen_question_in_train': True, 'long_context': False}]}


def distributed_generation_worker(rank, folder):
    dist.init_process_group('gloo', init_method='file://' + str(Path(folder) / 'init'), rank=rank, world_size=2)
    try:
        model = TinyAnswerModel()
        runtime = Runtime(torch.device('cpu'), rank=rank, world_size=2)
        report = capture_identity_generation(model, specification(), Decoder(), runtime, adapters_enabled=True, max_new_tokens=4)
        (Path(folder) / f'{rank}.json').write_text(json.dumps(report))
    finally:
        dist.destroy_process_group()


class IdentityGenerationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = TinyAnswerModel()
        self.runtime = Runtime(torch.device('cpu'))

    def capture(self, enabled, budget=4):
        return capture_identity_generation(self.model, specification(), Decoder(), self.runtime,
                                           adapters_enabled=enabled, max_new_tokens=budget)

    def test_default_adapter_path_and_disabled_base_are_exact(self):
        x = torch.randn(2, 6)
        self.model.eval()
        layer = self.model.proj
        expected = F.linear(x, layer.weight) + F.linear(F.linear(x, layer.lora_a), layer.lora_b) * layer.scaling
        torch.testing.assert_close(layer(x), expected, atol=0, rtol=0)
        keys = set(self.model.state_dict())
        with evaluation_mode(self.model, self.runtime, adapters_enabled=False, seed=8):
            torch.testing.assert_close(layer(x), F.linear(x, layer.weight), atol=0, rtol=0)
        self.assertTrue(layer.adapters_enabled)
        self.assertEqual(keys, set(self.model.state_dict()))

    def test_real_generated_strings_change_and_target_nll_improves(self):
        base, adapted = self.capture(False), self.capture(True)
        self.assertEqual(base['prompts'][0]['generated_text'], '其他模型')
        self.assertEqual(adapted['prompts'][0]['generated_text'], 'RTP大模型')
        self.assertEqual(adapted['prompts'][0]['generated_token_ids'], [2, 0])
        comparison = compare_identity_generations(base, adapted)
        self.assertTrue(comparison['adapter_strict_success_all'])
        self.assertEqual(comparison['generation_changed_count'], 1)
        self.assertTrue(comparison['prompts'][0]['target_nll_improved'])
        self.assertFalse(base['prompts'][0]['strict_success'])

    def test_target_nll_scores_answer_shift_and_eos_only(self):
        with evaluation_mode(self.model, self.runtime, adapters_enabled=True, seed=3):
            result = target_nll(self.model, [4, 1], [2, 0], self.runtime)
            logits = self.model(torch.tensor([[4, 1, 2, 0]]), torch.ones(1, 4, dtype=torch.bool)).logits
            expected = F.cross_entropy(logits[0, 1:3].float(), torch.tensor([2, 0]))
        self.assertEqual(result['mean'], float(expected))
        self.assertEqual(result['tokens'], 2)

    def test_exact_but_budget_truncated_answer_is_not_strict_success(self):
        row = self.capture(True, budget=1)['prompts'][0]
        self.assertTrue(row['exact_match'])
        self.assertTrue(row['contains_target'])
        self.assertFalse(row['strict_success'])
        self.assertEqual(row['stop_reason'], 'max_new_tokens')

    def test_eval_restores_rng_flags_and_weights_after_success_or_exception(self):
        self.model.train()
        self.model.proj.dropout.eval()
        flags = [m.training for m in self.model.modules()]
        rng = torch.get_rng_state().clone()
        weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.capture(False)
        self.assertEqual([m.training for m in self.model.modules()], flags)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with evaluation_mode(self.model, self.runtime, adapters_enabled=False, seed=7):
                torch.rand(9)
                raise RuntimeError('injected')
        self.assertTrue(self.model.proj.adapters_enabled)
        self.assertEqual([m.training for m in self.model.modules()], flags)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        for name, tensor in self.model.state_dict().items():
            torch.testing.assert_close(tensor, weights[name], atol=0, rtol=0)

    def test_fresh_state_reload_matches_real_generated_tokens_and_nll(self):
        expected = self.capture(True)
        state = copy.deepcopy(self.model.state_dict())
        self.model = TinyAnswerModel()
        self.model.load_state_dict(state)
        actual = self.capture(True)
        self.assertTrue(compare_reloaded_generation(actual, expected)['passed'])
        actual['prompts'][0]['generated_token_ids'] = [3, 0]
        self.assertFalse(compare_reloaded_generation(actual, expected)['passed'])

    def test_prompt_display_and_question_are_bound_to_real_token_ids(self):
        from tokenizers import Tokenizer, AddedToken
        from tokenizers.models import WordLevel
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            tokenizer = Tokenizer(WordLevel({'<eos>': 0, '你是谁': 1, 'RTP大模型': 2, '<unk>': 3}, unk_token='<unk>'))
            tokenizer.add_special_tokens([AddedToken('<eos>', special=True)])
            tokenizer.save(str(folder / 'tokenizer.json'))
            spec = specification()
            spec.update(tokenizer_sha256=file_sha256(folder / 'tokenizer.json'), config_sha256='1' * 64)
            prompt = folder / 'heldout.json'
            prompt.write_text(json.dumps(spec))
            options = dict(vocab_size=4, max_position_embeddings=128, config_sha256='1' * 64)
            load_prompt_spec(prompt, folder / 'tokenizer.json', **options)
            spec['prompts'][0]['prompt_text'] = '不同问题'
            prompt.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, 'displayed prompt'):
                load_prompt_spec(prompt, folder / 'tokenizer.json', **options)
            spec['prompts'][0]['prompt_text'] = '你是谁'
            spec['prompts'][0]['question'] = '不同问题'
            prompt.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, 'question is missing'):
                load_prompt_spec(prompt, folder / 'tokenizer.json', **options)

    def test_two_cpu_ranks_share_greedy_tokens_and_eos_exit(self):
        with tempfile.TemporaryDirectory() as folder:
            torch.multiprocessing.spawn(distributed_generation_worker, args=(folder,), nprocs=2, join=True)
            reports = [json.loads((Path(folder) / f'{rank}.json').read_text()) for rank in range(2)]
        self.assertEqual(reports[0]['prompts'], reports[1]['prompts'])
        self.assertEqual(reports[0]['prompts'][0]['generated_token_ids'], [2, 0])
        self.assertTrue(reports[0]['prompts'][0]['strict_success'])


if __name__ == '__main__':
    unittest.main()
