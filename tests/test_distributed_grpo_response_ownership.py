"""Local objective/AdamW oracle; actual FSDP/EP collectives require GPU qualification."""
import copy
from contextlib import ExitStack
import types
import unittest
from unittest.mock import Mock, patch

import torch

from dsv41_train.rl import distributed_grpo as grpo
from dsv41_train.rl.algorithm import response_log_probs
from dsv41_train.rl.learner_batch_planner import plan_response_ownership
from dsv41_train.rl.native_lora import adapter_snapshot, minimum_scoring_head_chunks, scoring_model_class
from tests.test_distributed_grpo import NativeBase


class ResponseOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_owners_cover_g32_once_and_keep_equal_rounds_for_uneven_world(self):
        lengths = [3 + i % 7 for i in range(32)]
        for world in (2, 16, 24):
            plans = [plan_response_ownership(lengths, rank, world) for rank in range(world)]
            self.assertEqual({len(plan) for plan in plans}, {(8 + world - 1) // world})
            owned = [i for plan in plans for p in plan if not p['is_placeholder'] for i in p['response_indices']]
            self.assertEqual(sorted(owned), list(range(32)))
            self.assertEqual(sum(p['is_placeholder'] for plan in plans for p in plan),
                             world * len(plans[0]) - 8)
            self.assertTrue(all(p['response_indices'] == [0, 1, 2, 3] for plan in plans for p in plan
                                if p['is_placeholder']))
        for parts in ([[(0, {})]] * 32, [[]] * 24, [[(32, {})]]):
            with self.assertRaises(ValueError):
                grpo.merge_response_metrics(parts)

    def test_sharded_local_gradients_sum_to_full_g32_and_same_adamw_update(self):
        # KL case ensures a placeholder's entire loss is zero, not just advantage.
        for world, lag, beta in ((16, 0, 0.), (24, 1, 0.), (24, 0, .03)):
            with self.subTest(world=world, lag=lag, beta=beta):
                self.compare(world, lag, beta)

    def compare(self, world, lag, beta):
        torch.manual_seed(53)
        def factory():
            model = scoring_model_class(NativeBase)()
            model.scoring_head_chunk_tokens = 5  # B4 groups need differing numbers of chunks.
            model.stable_gradients = True
            model.set_requires_gradient_sync = Mock()
            return model
        model = factory()
        parameters = [p for p in model.parameters() if p.requires_grad]
        with torch.no_grad():
            model.model.adapter.lora_b.normal_(0, .03)
        def optimizer_for(m):
            return torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],
                                     lr=.02, weight_decay=0, foreach=False)
        optimizer = optimizer_for(model)
        old = copy.deepcopy(model)
        snapshot_values = adapter_snapshot(optimizer)
        for parameter in parameters:
            parameter.grad = torch.full_like(parameter, .05)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if not lag:
            old.load_state_dict(model.state_dict())
            snapshot_values = adapter_snapshot(optimizer)
        state, adam_state = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        old_ack = {'version': 0, 'digest': 'old', 'lineage': 'ownership'}
        ack = {**old_ack, 'version': lag, 'digest': 'current' if lag else 'old'}
        snapshot = {'ack': old_ack, 'parameters': snapshot_values}
        results = []
        for index in range(32):
            prompt, actions = [0, 2, 3][:2 + index % 2], [3, 4, 5, 6, 7][:1 + index % 5]
            with torch.no_grad():
                behavior = response_log_probs(old, prompt, actions, .8).tolist()
            results.append({'context_token_ids': prompt, 'action_token_ids': actions,
                'behavior_log_probs': [v - .3 for v in behavior],
                'generation_metrics': {'policy_ack': old_ack}})
        advantages = [0. if i % 5 == 0 else (1. if i % 2 else -.7) for i in range(32)]
        args = types.SimpleNamespace(temperature=.8, kl_beta=beta, clip_eps=.2, rollout_is_cap=2.,
            rollout_correction='none' if beta else 'decoupled-token-tis', disable_prefetch=False,
            learner_microbatch_size=4, learner_response_ownership='replicated', learner_math_mode='bf16-no-split-k')
        policy = types.SimpleNamespace(begin_step=lambda: None, normalize=lambda: None)
        loss_name = 'grpo_loss' if beta else 'decoupled_token_tis_loss'
        original_loss = getattr(grpo, loss_name)
        expected_rows, expected_scores = [], []
        def capture(*values, **kwargs):
            loss, row = original_loss(*values, **kwargs)
            expected_rows.append(row)
            expected_scores.append((values[0][0].detach().clone(),
                                    None if beta else values[1][0].detach().clone()))
            return loss, row
        with patch.object(grpo, loss_name, side_effect=capture), patch.object(grpo, "require_bf16_no_split_k"):
            baseline = grpo.train_group(model, optimizer, policy, results, advantages, args, lag + 1,
                expected_ack=ack, native_old_snapshot=snapshot if lag else None)
        args.learner_response_ownership = 'sharded'
        gradients = []
        for rank in range(world):
            local = factory(); local.load_state_dict(state)
            local_optimizer = optimizer_for(local)
            local_optimizer.load_state_dict(copy.deepcopy(adam_state))
            plan = plan_response_ownership([len(r['context_token_ids']) + len(r['action_token_ids']) - 1
                                           for r in results], rank, world)
            owned = [i for item in plan if not item['is_placeholder'] for i in item['response_indices']]
            cursor = iter(i for item in plan for i in item['response_indices'])
            def compare_score(*values, **kwargs):
                index = next(cursor)
                torch.testing.assert_close(values[0][0], expected_scores[index][0], rtol=2e-6, atol=3e-7)
                if not beta:
                    torch.testing.assert_close(values[1][0], expected_scores[index][1], rtol=2e-6, atol=3e-7)
                return original_loss(*values, **kwargs)
            def gather(parts, local_rows):
                self.assertEqual([i for i, _ in local_rows], owned)
                parts[:] = [[(i, expected_rows[i]) for item in plan_response_ownership([3] * 32, owner, world)
                             if not item['is_placeholder'] for i in item['response_indices']] for owner in range(world)]
                for i, row in local_rows:
                    self.assertAlmostEqual(row['loss'], expected_rows[i]['loss'], places=6)
            # Simulate FSDP SUM *after* collecting every rank's actual local
            # gradients below. This is not a collective-order or CUDA test.
            with ExitStack() as stack:
                for name, value in (('is_initialized', True), ('get_world_size', world), ('get_rank', rank)):
                    stack.enter_context(patch.object(grpo.dist, name, return_value=value))
                stack.enter_context(patch.object(grpo.dist, 'all_gather_object', side_effect=gather))
                stack.enter_context(patch.object(grpo, 'gradient_norm', return_value=1.))
                stack.enter_context(patch.object(grpo, 'verify_optimizer_step', return_value=1))
                stack.enter_context(patch.object(grpo, 'require_bf16_no_split_k'))
                step = stack.enter_context(patch.object(local_optimizer, 'step'))
                stack.enter_context(patch.object(grpo, loss_name, side_effect=compare_score))
                actual = grpo.train_group(local, local_optimizer, policy, results, advantages, args, lag + 1,
                    expected_ack=ack, native_old_snapshot=snapshot if lag else None)
                step.assert_called_once()
            self.assertEqual(local.scoring_head_min_chunks, 1)
            self.assertAlmostEqual(actual['loss'], baseline['loss'], places=6)
            self.assertEqual(actual['learner_batching']['backward_calls'], len(plan))
            self.assertEqual(actual['learner_batching']['owned_response_indices'], owned)
            self.assertEqual(local.set_requires_gradient_sync.call_args.args, (True,))
            gradients.append([p.grad.clone() for p in local.parameters() if p.requires_grad])
        averaged = factory(); averaged.load_state_dict(state)
        averaged_optimizer = optimizer_for(averaged)
        averaged_optimizer.load_state_dict(copy.deepcopy(adam_state))
        averaged_parameters = [p for p in averaged.parameters() if p.requires_grad]
        for index, (parameter, expected) in enumerate(zip(averaged_parameters, parameters)):
            parameter.grad = torch.stack([row[index] for row in gradients]).sum(0)
            torch.testing.assert_close(parameter.grad, expected.grad, rtol=3e-5, atol=2e-7)
        averaged_optimizer.step()
        for actual, expected in zip(averaged_parameters, parameters):
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-7)
            for key, value in averaged_optimizer.state[actual].items():
                torch.testing.assert_close(value, optimizer.state[expected][key], rtol=3e-5, atol=2e-7)

    def test_head_padding_keeps_real_chunks_and_zero_gradient_through_recomputation(self):
        torch.manual_seed(17)
        model = scoring_model_class(NativeBase)()
        model.scoring_head_chunk_tokens = 5
        parameters = [p for p in model.parameters() if p.requires_grad]
        values = response_log_probs(model, [0, 2], [3] * 8, .8)
        gradients = torch.autograd.grad(values.sum(), parameters)
        calls = []
        hook = model.lm_head.register_forward_pre_hook(lambda _m, args: calls.append(len(args[0])))
        try:
            with minimum_scoring_head_chunks(model, 4):
                padded = response_log_probs(model, [0, 2], [3] * 8, .8)
            self.assertEqual(model.scoring_head_min_chunks, 1)
            self.assertEqual(calls, [5, 3, 1, 1])
            torch.testing.assert_close(padded, values, rtol=0, atol=0)
            calls.clear()
            actual = torch.autograd.grad(padded.sum(), parameters)
            self.assertEqual(sorted(calls), [1, 1, 3, 5])
            for got, want in zip(actual, gradients):
                torch.testing.assert_close(got, want, rtol=0, atol=0)
            with self.assertRaisesRegex(RuntimeError, 'test restore'):
                with minimum_scoring_head_chunks(model, 4):
                    raise RuntimeError('test restore')
            self.assertEqual(model.scoring_head_min_chunks, 1)
        finally:
            hook.remove()

    def test_head_chunk_and_physical_batch_guards_precede_model_work(self):
        model = scoring_model_class(NativeBase)()
        args = types.SimpleNamespace(rollout_correction='none', kl_beta=0., learner_response_ownership='sharded',
                                     learner_microbatch_size=1)
        results = [{'context_token_ids': [0, 2], 'action_token_ids': [3, 4, 5]}] * 32
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
        with patch.object(grpo.dist, 'is_initialized', return_value=True), \
                patch.object(model, 'forward') as forward, patch.object(optimizer, 'zero_grad') as zero:
            with self.assertRaisesRegex(ValueError, 'physical B4'):
                grpo.train_group(model, optimizer, None, results, [1.] * 32, args, 1)
            args.learner_microbatch_size = 4
            model.scoring_head_chunk_tokens = 0
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                grpo.train_group(model, optimizer, None, results, [1.] * 32, args, 1)
            model.scoring_head_chunk_tokens = 8048
            with self.assertRaisesRegex(ValueError, 'explicit bf16-no-split-k'):
                grpo.train_group(model, optimizer, None, results, [1.] * 32, args, 1)
            forward.assert_not_called(); zero.assert_not_called()


if __name__ == '__main__':
    unittest.main()
