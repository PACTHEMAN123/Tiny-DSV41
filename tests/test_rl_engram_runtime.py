import copy
from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from dsv41_train.engram_offload import CPUEmbeddingLookup, install_engram_sgd_hooks
from dsv41_train.rl.algorithm import response_log_probs
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.runtime import _clip_sparse_grad_norm, train_groups
from rollout_manager.training import PolicyRef, TrainingTrajectory, TurnTrace, token_digest


class _Table(torch.nn.Module):
    def __init__(self, sparse, dtype):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(7, 4).to(dtype) / 10)
        self.engram_offload = self.sparse_gradients = sparse
        self._cpu_lookup = CPUEmbeddingLookup()
        if sparse:
            self._cpu_lookup.configure_cache(48, "cpu")

    def forward(self, ids):
        if self.sparse_gradients:
            return self._cpu_lookup(self.weight, ids, device=ids.device)
        return F.embedding(ids, self.weight)


class _Policy(torch.nn.Module):
    def __init__(self, sparse=True, dtype=torch.float32):
        super().__init__()
        self.table = _Table(sparse, dtype)
        self.head = torch.nn.Parameter(torch.randn(4, 7) / 5)
        self.marked_rows = set()

    def forward(self, input_ids, attention_mask=None):
        return self.table(input_ids).float() @ self.head

    def mark_engram_touched(self):
        self.marked_rows.update(self.table.weight.grad.coalesce().indices()[0].tolist())

    def assert_engram_invariants(self):
        assert self.table.weight.device.type == "cpu"


class EngramRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(17)
        self.config = RLConfig(rollout_engine="native-pytorch", train_engram=True,
                               optimizer="sgd", kl_beta=.05, max_grad_norm=.01)

    def groups(self, model):
        policy = PolicyRef(0, "a" * 64, "test")
        trajectories = []
        for index, specs in enumerate(((([1, 2], [3]), ([1, 2, 3, 5], [4, 6, 2])),
                                      (([1, 2], [5, 4]), ([1, 2, 5, 4, 6], [3])))):
            turns = []
            for number, (prompt, actions) in enumerate(specs, 1):
                with torch.no_grad():
                    behavior = response_log_probs(model, prompt, actions).tolist()
                turns.append(TurnTrace(number, f"{index}-{number}", prompt, actions, behavior, policy))
            trajectories.append(TrainingTrajectory(0, "task", token_digest(specs[0][0]), "b" * 64,
                                                    index, turns, float(index)))
        return [SimpleNamespace(samples=trajectories)]

    def test_configuration_and_cli_override(self):
        self.config.validate()
        replace(self.config, trainable_scope="lora").validate()
        for override in ({"optimizer": "adamw"}, {"weight_decay": .01}, {"engram_offload": False},
                         {"rollout_engine": "vllm"}, {"trainable_scope": "lm_head"},
                         {"train_engram": 1}, {"engram_cache_mib": -1}, {"engram_cache_mib": True}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                replace(self.config, **override).validate()
        for override in ({"engram_offload": False, "engram_cache_mib": 1},
                         {"initial_engram_delta": ""}, {"initial_engram_delta": "rows.pt", "engram_offload": False}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                replace(self.config, train_engram=False, **override).validate()
        replace(self.config, train_engram=False, initial_engram_delta="rows.pt").validate()
        from dsv41_train.rl.train import main
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"train_engram": True}))
            with patch("sys.argv", ["train", "--model-dir", directory, "--data", "data.jsonl",
                                    "--config", str(path), "--optimizer", "sgd", "--rollout-engine",
                                    "native-pytorch", "--engram-cache-mib", "2"]), \
                    patch("dsv41_train.rl.train.run_training") as train, patch("builtins.print"):
                main()
            config = train.call_args.kwargs["config"]
            self.assertTrue(config.train_engram)
            self.assertEqual((config.optimizer, config.engram_cache_mib), ("sgd", 2))

    def test_sparse_cancellation_and_dense_gradient_share_one_clip(self):
        table = torch.nn.Parameter(torch.zeros(1000, 2, dtype=torch.bfloat16))
        # Only gradients are scanned: an unrelated storage row is not traversed.
        with torch.no_grad():
            table[999] = float("nan")
        table.grad = torch.sparse_coo_tensor([[0, 0, 0, 0]],
            torch.tensor([[256., 0], [1, 0], [1, 0], [-256, 0]], dtype=torch.bfloat16), table.shape,
            check_invariants=True)
        dense = torch.nn.Parameter(torch.zeros(1))
        dense.grad = torch.tensor([3.])
        norm = _clip_sparse_grad_norm([table, dense], 2.)
        self.assertAlmostEqual(float(norm), math.sqrt(13), places=6)
        scale = 2 / (math.sqrt(13) + 1e-6)
        torch.testing.assert_close(table.grad.values(), torch.tensor([[2 * scale, 0]], dtype=torch.bfloat16))
        torch.testing.assert_close(dense.grad, torch.tensor([3 * scale]))

    def test_multiturn_update_matches_complete_dense_trajectory(self):
        sparse = _Policy()
        dense = copy.deepcopy(sparse)
        dense.table.sparse_gradients = dense.table.engram_offload = False
        reference = copy.deepcopy(dense).requires_grad_(False)
        groups = self.groups(dense)
        sparse_optimizer = torch.optim.SGD(sparse.parameters(), lr=.1, foreach=False)
        dense_optimizer = torch.optim.SGD(dense.parameters(), lr=.1, foreach=False)
        install_engram_sgd_hooks(sparse, sparse_optimizer)
        initial = sparse.table.weight.detach().clone()
        for _ in range(2):
            measured = train_groups(sparse, reference, sparse_optimizer, groups, self.config)
            expected = train_groups(dense, reference, dense_optimizer, groups,
                                    replace(self.config, train_engram=False))
            for key in ("loss", "policy_loss", "kl", "grad_norm"):
                self.assertAlmostEqual(measured[key], expected[key], places=6)
            for actual, wanted in zip(sparse.parameters(), dense.parameters()):
                torch.testing.assert_close(actual, wanted, rtol=1e-6, atol=1e-7)
                grad = actual.grad.to_dense() if actual.grad.is_sparse else actual.grad
                torch.testing.assert_close(grad, wanted.grad, rtol=1e-5, atol=1e-8)
        self.assertTrue(sparse.marked_rows)
        self.assertFalse(torch.equal(initial, sparse.table.weight))
        self.assertGreater(sparse.table._cpu_lookup.cache_metrics["hits"], 0)

    def test_bfloat16_multiturn_and_nonfinite_gradient_failure(self):
        model = _Policy(dtype=torch.bfloat16)
        groups = self.groups(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=1., foreach=False)
        install_engram_sgd_hooks(model, optimizer)
        initial = model.table.weight.detach().clone()
        for _ in range(2):
            train_groups(model, None, optimizer, groups, replace(self.config, kl_beta=0, max_grad_norm=1.))
        self.assertFalse(torch.equal(initial, model.table.weight))
        self.assertTrue(model.table.weight.grad.is_coalesced())
        broken = _Policy()
        broken.table.weight.register_hook(lambda g: torch.sparse_coo_tensor(
            g._indices(), torch.full_like(g._values(), float("nan")), g.shape, check_invariants=True))
        groups = self.groups(broken)
        before = [p.detach().clone() for p in broken.parameters()]
        optimizer = torch.optim.SGD(broken.parameters(), lr=.1, foreach=False)
        install_engram_sgd_hooks(broken, optimizer)
        with self.assertRaisesRegex(FloatingPointError, "non-finite RL gradient"):
            train_groups(broken, None, optimizer, groups, replace(self.config, kl_beta=0))
        for value, saved in zip(broken.parameters(), before):
            torch.testing.assert_close(value, saved, rtol=0, atol=0)
        self.assertFalse(broken.marked_rows)


if __name__ == "__main__":
    unittest.main()
