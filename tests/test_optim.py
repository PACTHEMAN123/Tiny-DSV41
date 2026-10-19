import unittest

import torch

from dsv41_train.optim import install_sgd_in_backward


class SGDInBackwardTests(unittest.TestCase):
    def test_rejects_non_plain_sgd_and_incomplete_ownership(self):
        model = torch.nn.Linear(2, 2)
        for optimizer in (torch.optim.AdamW(model.parameters()),
                          torch.optim.SGD(model.parameters(), lr=.1, momentum=.9),
                          torch.optim.SGD(model.parameters(), lr=.1, weight_decay=.1),
                          torch.optim.SGD(model.parameters(), lr=.1, foreach=True),
                          torch.optim.SGD([model.weight], lr=.1)):
            with self.subTest(optimizer=optimizer), self.assertRaises(ValueError):
                install_sgd_in_backward(model, optimizer)
        with self.assertRaisesRegex(ValueError, "accumulation"):
            install_sgd_in_backward(model, torch.optim.SGD(model.parameters(), lr=.1),
                                    gradient_accumulation_steps=2)

    def test_ordinary_and_sparse_parameters_wait_for_native_step(self):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.ones(2))
        model.table = torch.nn.Embedding(5, 2, sparse=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1, foreach=False)
        hooks = install_sgd_in_backward(model, optimizer)
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        (model.weight.sum() + model.table(torch.tensor([1, 1])).sum()).backward()
        self.assertEqual(hooks.parameter_ids, set())
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            self.assertIsNotNone(value.grad)
        optimizer.step()
        torch.testing.assert_close(model.weight, before["weight"] - .1)
        expected = before["table.weight"].clone()
        expected[1] -= .2
        torch.testing.assert_close(model.table.weight, expected)
        self.assertTrue(hooks.local_finite("cpu").item())


if __name__ == "__main__":
    unittest.main()
