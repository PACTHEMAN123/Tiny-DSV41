"""Verify token weighting against one global causal loss, including accumulation."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from dsv41_train.sft_data import SFTDataset
from train_sft import loss_scale


class SFTObjectiveTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        path = Path(self.temporary.name) / "examples.jsonl"
        labels = [
            [1, -100, -100, -100, 2],
            [2, -100, -100, 3, 4],
            [3, -100, 4, 5, 6],
            [4, 5, 6, 1, 2],
        ]
        path.write_text("\n".join(json.dumps({"input_ids": [1, 2, 3, 4, 5], "labels": row})
                                  for row in labels) + "\n")
        self.dataset = SFTDataset.from_jsonl(path, max_length=5)
        generator = torch.Generator().manual_seed(730)
        self.features = torch.randn(4, 5, 3, generator=generator, dtype=torch.float64)
        self.model = torch.nn.Linear(3, 7, dtype=torch.float64)
        with torch.no_grad():
            self.model.weight.copy_(torch.randn(7, 3, generator=generator, dtype=torch.float64))
            self.model.bias.copy_(torch.randn(7, generator=generator, dtype=torch.float64))

    def objective(self, model, example_indices, *, reduction="mean"):
        logits = model(self.features[example_indices])[:, :-1]
        labels = self.dataset.labels[example_indices, 1:]
        return F.cross_entropy(logits.reshape(-1, 7), labels.reshape(-1),
                               ignore_index=-100, reduction=reduction)

    def test_accumulation_and_rank_averaging_equal_global_token_mean(self):
        self.assertEqual(self.dataset.supervised_tokens.tolist(), [1, 2, 3, 4])
        self.assertEqual(int(self.dataset.labels[:, 0].ne(-100).sum()), 0)
        total_targets = int(self.dataset.supervised_tokens.sum())
        self.assertEqual(total_targets, 10)

        reference = deepcopy(self.model)
        expected_loss = self.objective(reference, [0, 1, 2, 3])
        expected_loss.backward()
        workers, observed_loss = [], 0.0
        world_size = 2
        for rank in range(world_size):
            worker = deepcopy(self.model)
            # A rank accumulates two microbatches; each has a different number
            # of answer tokens, and the ranks have unequal total token counts.
            for micro_step in range(2):
                index = micro_step * world_size + rank
                count = int(self.dataset.batch([index])["supervised_tokens"])
                loss = self.objective(worker, [index]) * loss_scale(count, total_targets, world_size)
                observed_loss += loss.detach().item() / world_size
                loss.backward()
            workers.append(worker)
        self.assertAlmostEqual(observed_loss, expected_loss.item(), places=12)
        for name, parameter in reference.named_parameters():
            averaged_gradient = sum(dict(worker.named_parameters())[name].grad for worker in workers) / world_size
            torch.testing.assert_close(averaged_gradient, parameter.grad, rtol=1e-12, atol=1e-12)

        # This assertion makes the fixture distinguish correct token weighting
        # from the tempting but wrong mean of the four microbatch mean losses.
        unweighted = deepcopy(self.model)
        for index in range(4):
            (self.objective(unweighted, [index]) / 4).backward()
        self.assertGreater((unweighted.weight.grad - reference.weight.grad).abs().max().item(), 1e-3)

    def test_single_rank_accumulation_does_not_divide_twice(self):
        reference = deepcopy(self.model)
        self.objective(reference, [0, 1, 2, 3]).backward()
        accumulated = deepcopy(self.model)
        total_targets = int(self.dataset.supervised_tokens.sum())
        # Different microbatch sizes are legal mathematically as well; the
        # normalization still depends on target tokens, not the batch size.
        for indices in ([0], [1, 2, 3]):
            count = int(self.dataset.batch(indices)["supervised_tokens"])
            (self.objective(accumulated, indices) * loss_scale(count, total_targets, 1)).backward()
        for actual, expected in zip(accumulated.parameters(), reference.parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-12, atol=1e-12)

    def test_zero_target_microbatch_or_global_batch_is_rejected(self):
        for local, global_count in ((0, 5), (1, 0), (-1, 5)):
            with self.subTest(local=local, global_count=global_count), self.assertRaises(ValueError):
                loss_scale(local, global_count, 2)


if __name__ == "__main__":
    unittest.main()
