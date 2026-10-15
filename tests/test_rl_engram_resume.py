"""Fixed-trajectory resume isolates learner correctness from rollout scheduling."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from dsv41_train.engram_offload import install_engram_sgd_hooks
from dsv41_train.rl.algorithm import response_log_probs
from dsv41_train.rl.checkpoint import (add_engram_identity, add_lora_identity,
    load_training_checkpoint, policy_state, run_identity, save_training_checkpoint)
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.family import load_model_config
from dsv41_train.rl.rollout import RolloutGroup, Sample
from dsv41_train.rl.runtime import _clip_sparse_grad_norm, make_model, train_groups


ROOT = Path(__file__).resolve().parents[1]


class EngramResumeTests(unittest.TestCase):
    def assert_state_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_state_equal(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for value, wanted in zip(actual, expected):
                self.assert_state_equal(value, wanted)
        else:
            self.assertEqual(actual, expected)

    @staticmethod
    def tensors(model):
        # Includes nonpersistent hash/rotary buffers, not just state_dict tensors.
        return {name: value.detach().clone() for name, value in
                (*model.named_parameters(), *model.named_buffers())}

    def test_random_roles_and_fixed_trajectory_resume(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "config.json").write_bytes((ROOT / "examples/rl/tiny-dsv41/config.json").read_bytes())
            data = folder / "prompts.jsonl"
            data.write_text('{"prompt_ids":[1,4,7]}\n')
            model_config = load_model_config(folder)
            for scope in ("all", "lora"):
                with self.subTest(scope=scope):
                    config = RLConfig(rollout_engine="native-pytorch", train_engram=True,
                        optimizer="sgd", engram_cache_mib=1, trainable_scope=scope,
                        lora_rank=2, seed=37, learning_rate=.1, kl_beta=.03)
                    def build(role="learner", baseline=None):
                        return make_model(folder, model_config, config, ("cpu",), True,
                                          role=role, engram_weights=baseline)
                    def optimizer(model):
                        result = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                                 lr=config.learning_rate, foreach=False)
                        install_engram_sgd_hooks(model, result)
                        return result
                    learner = build()
                    baseline = learner.export_engram_baseline()
                    reference = build("reference", baseline).eval().requires_grad_(False)
                    actor = build("actor", baseline).eval()
                    initial = self.tensors(learner)
                    for value in initial.values():
                        self.assertTrue(bool(torch.isfinite(value).all()))
                    self.assert_state_equal(self.tensors(reference), initial)
                    self.assert_state_equal(self.tensors(actor), initial)
                    tokens = torch.tensor([[1, 4, 7, 9]])
                    with torch.no_grad():
                        logits = learner(tokens)
                        self.assertTrue(bool(torch.isfinite(logits).all()))
                        self.assert_state_equal(reference(tokens), logits)
                        self.assert_state_equal(actor(tokens), logits)
                        samples = [Sample([1, 4, 7], response,
                            response_log_probs(actor, [1, 4, 7], response).tolist(), reward, 0)
                            for response, reward in (([3, 4], 0.), ([5, 6, 7], 1.))]
                    groups = [RolloutGroup(samples, 0, 0)]
                    optim = optimizer(learner)
                    metrics = train_groups(learner, reference, optim, groups, config)
                    self.assertEqual(metrics["nonzero_advantages"], 2)
                    self.assert_state_equal(self.tensors(reference), initial)
                    self.assert_state_equal(self.tensors(actor), initial)
                    before, before_policy = self.tensors(learner), policy_state(learner)
                    self.assertTrue(any(not torch.equal(value, initial[name]) for name, value in before.items()))
                    self.assertTrue(all(record["indices"].numel() for record in
                                        before_policy["engram_delta"]["tables"].values()))
                    for name, record in before_policy["engram_delta"]["tables"].items():
                        self.assertFalse(torch.equal(record["values"],
                            initial[f"backend.model.engram_tables.{name}.weight"][record["indices"]]))
                    identity = add_engram_identity(add_lora_identity(
                        run_identity(folder, data, True), config, model_config), config)
                    checkpoint = folder / f"{scope}.pt"
                    rng, optim_state = torch.get_rng_state().clone(), deepcopy(optim.state_dict())
                    save_training_checkpoint(checkpoint, learner, optim, step=1, config=config,
                        identity=identity, next_group_id=1, rollout_rng_state=torch.Generator().get_state(),
                        dropped_groups=0)
                    restored = build()
                    resumed_reference = build("reference", restored.export_engram_baseline()).eval().requires_grad_(False)
                    resumed_optim = optimizer(restored)
                    recovered = load_training_checkpoint(checkpoint, restored, resumed_optim, config, identity)
                    self.assertEqual((recovered["step"], recovered["next_group_id"]), (1, 1))
                    self.assert_state_equal(torch.get_rng_state(), rng)
                    self.assert_state_equal(self.tensors(restored), before)
                    self.assert_state_equal(policy_state(restored), before_policy)
                    self.assert_state_equal(resumed_optim.state_dict(), optim_state)
                    self.assert_state_equal(self.tensors(resumed_reference), initial)
                    torch.set_rng_state(rng)
                    expected = train_groups(learner, reference, optim, groups, config)
                    torch.set_rng_state(rng)
                    actual = train_groups(restored, resumed_reference, resumed_optim, groups, config)
                    self.assert_state_equal(actual, expected)
                    self.assert_state_equal(self.tensors(restored), self.tensors(learner))
                    self.assert_state_equal(policy_state(restored), policy_state(learner))
                    self.assert_state_equal(resumed_optim.state_dict(), optim.state_dict())
                    self.assert_state_equal(self.tensors(resumed_reference), initial)
                    with torch.no_grad():
                        self.assert_state_equal(resumed_reference(tokens), logits)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
    def test_cpu_sparse_and_two_gpu_dense_clipping(self):
        indices = torch.tensor([[0, 0, 0, 0, 1]])
        values = torch.tensor([[256., 0], [1., 0], [1., 0], [-256., 0], [0, 4]], dtype=torch.bfloat16)
        table = torch.nn.Parameter(torch.zeros(7, 2, dtype=torch.bfloat16))
        table.grad = torch.sparse_coo_tensor(indices, values, table.shape, check_invariants=True)
        dense = [torch.nn.Parameter(torch.zeros(3, device=f"cuda:{index}")) for index in range(2)]
        gradients = [torch.tensor([3., -4., .5]), torch.tensor([1., -2., 6.])]
        for parameter, gradient in zip(dense, gradients):
            parameter.grad = gradient.to(parameter.device)
        oracle = table.grad.float().to_dense()
        expected_norm = torch.linalg.vector_norm(torch.cat([oracle.flatten(), *gradients]))
        coefficient = min(1., 2. / (float(expected_norm) + 1e-6))
        norm = _clip_sparse_grad_norm([table, *dense], 2.)
        torch.testing.assert_close(norm.float(), expected_norm, rtol=1e-6, atol=1e-7)
        self.assertTrue(table.grad.is_coalesced())
        torch.testing.assert_close(table.grad.to_dense(), (oracle * coefficient).bfloat16(), rtol=0, atol=0)
        for parameter, gradient in zip(dense, gradients):
            torch.testing.assert_close(parameter.grad.cpu(), gradient * coefficient, rtol=1e-6, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
