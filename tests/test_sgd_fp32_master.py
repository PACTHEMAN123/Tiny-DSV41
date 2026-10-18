"""Selective masters retain sub-BF16 updates without changing ordinary SGD."""
from datetime import timedelta
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from dsv41_train.optim import install_sgd_in_backward, sgd_parameter_groups


class SelectiveMasterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.environment = patch.dict(os.environ, {"GLOO_SOCKET_IFNAME": "lo0"} if sys.platform == "darwin" else {})
        cls.environment.start()
        dist.init_process_group("gloo", init_method=f"file://{cls.folder.name}/rendezvous",
                                rank=0, world_size=1, timeout=timedelta(seconds=30))
        cls.mesh = init_device_mesh("cpu", (1,))

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        cls.environment.stop()
        cls.folder.cleanup()

    def model(self, *, noncontiguous=False):
        model = nn.Module()
        value = torch.ones(3, 2, dtype=torch.bfloat16).t() if noncontiguous else torch.ones(2, 3, dtype=torch.bfloat16)
        for name in ("head", "ordinary"):
            tensor = DTensor.from_local(value.clone(), self.mesh, [Shard(0)], run_check=False)
            model.register_parameter(name, nn.Parameter(tensor))
        return model

    def install(self, model, names=("head",), lr=.1):
        optimizer = torch.optim.SGD(sgd_parameter_groups(model, names), lr=lr, foreach=False)
        hooks = install_sgd_in_backward(model, optimizer)
        self.addCleanup(hooks.remove)
        return optimizer, hooks

    def test_sub_ulp_updates_accumulate_only_in_selected_master(self):
        for noncontiguous in (False, True):
            with self.subTest(noncontiguous=noncontiguous):
                model = self.model(noncontiguous=noncontiguous)
                optimizer, hooks = self.install(model)
                master = optimizer.state[model.head]["master_weight"]
                expected = master.clone()
                gradient = torch.full_like(model.head.to_local(), .001)
                rng = torch.get_rng_state().clone()
                for _ in range(40):
                    hooks.reset()
                    (model.head.sum() * .001 + model.ordinary.sum() * .001).backward()
                    optimizer.step()
                    expected.add_(gradient, alpha=-.1)
                    torch.testing.assert_close(master, expected, rtol=0, atol=0)
                    torch.testing.assert_close(model.head.to_local(), expected.bfloat16(), rtol=0, atol=0)
                    self.assertIsNone(model.head.grad)
                    self.assertIsNone(model.ordinary.grad)
                    self.assertTrue(hooks.local_finite("cpu").item())
                self.assertFalse(torch.equal(model.head.to_local(), torch.ones_like(gradient)))
                self.assertTrue(torch.equal(model.ordinary.to_local(), torch.ones_like(gradient)))
                self.assertNotIn(model.ordinary, optimizer.state)
                self.assertFalse(master.requires_grad)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))

    def test_replaced_optimizer_state_is_used_by_existing_hook(self):
        model = self.model()
        optimizer, _ = self.install(model)
        old_master = optimizer.state[model.head]["master_weight"]
        replacement = torch.full_like(old_master, .75)
        optimizer.state.clear()
        optimizer.state[model.head] = {"master_weight": replacement}
        with torch.no_grad():
            model.head.to_local().copy_(replacement)
        model.head.sum().backward()
        torch.testing.assert_close(replacement, torch.full_like(replacement, .65), rtol=0, atol=0)
        torch.testing.assert_close(model.head.to_local(), replacement.bfloat16(), rtol=0, atol=0)
        self.assertTrue(bool((old_master == 1).all()))

    def test_expert_divisor_zero_lr_and_nonfinite_gate(self):
        for divisor, lr in ((8, .1), (1, 0.)):
            with self.subTest(divisor=divisor, lr=lr):
                model = self.model()
                optimizer, hooks = self.install(model, lr=lr)
                model.head.grad = torch.ones_like(model.head)
                hooks._step(model.head, optimizer, 1, divisor)
                expected = torch.ones(2, 3).add_(torch.ones(2, 3, dtype=torch.bfloat16), alpha=-lr/divisor)
                torch.testing.assert_close(optimizer.state[model.head]["master_weight"], expected, rtol=0, atol=0)
                self.assertIsNone(model.head.grad)
        model = self.model()
        optimizer, hooks = self.install(model)
        (model.head.sum() * float("inf")).backward()
        self.assertFalse(hooks.local_finite("cpu").item())
        self.assertIsNone(model.head.grad)

    def test_names_groups_and_ownership_are_strict(self):
        model = self.model()
        model.frozen = nn.Parameter(torch.ones(1, dtype=torch.bfloat16), requires_grad=False)
        model.local = nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
        model.fp32 = nn.Parameter(DTensor.from_local(torch.ones(1), self.mesh, [Shard(0)], run_check=False))
        for names in (("missing",), ("head", "head"), ("frozen",), ("local",), ("fp32",)):
            with self.subTest(names=names), self.assertRaises(ValueError):
                sgd_parameter_groups(model, names)
        groups = sgd_parameter_groups(model, ("head",))
        flat = [id(p) for g in groups for p in g["params"]]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(set(flat), {id(p) for p in model.parameters() if p.requires_grad})
        self.assertNotIn("fp32_master", groups[0])
        for invalid in (1, "true"):
            optimizer = torch.optim.SGD([{"params": model.parameters(), "fp32_master": invalid}], lr=.1)
            with self.assertRaisesRegex(ValueError, "boolean"):
                install_sgd_in_backward(model, optimizer)
        optimizer = torch.optim.SGD([{"params": [model.local], "fp32_master": True}], lr=.1)
        with self.assertRaisesRegex(ValueError, "BF16 DTensor"):
            install_sgd_in_backward(model, optimizer)

    def test_default_optimizer_groups_and_state_are_unchanged(self):
        model = self.model()
        ordinary = torch.optim.SGD(model.parameters(), lr=.1, foreach=False)
        default, hooks = self.install(model, names=())
        self.assertEqual(default.state_dict(), ordinary.state_dict())
        model.head.sum().backward()
        self.assertEqual(default.state_dict(), ordinary.state_dict())
        self.assertTrue(hooks.local_finite("cpu").item())

    def test_wrong_master_state_cannot_silently_downcast_or_reinitialize(self):
        model = self.model()
        optimizer, hooks = self.install(model)
        with self.assertRaisesRegex(ValueError, "before restoring"):
            install_sgd_in_backward(model, optimizer)
        optimizer.state[model.head]["master_weight"] = torch.ones(2, 3, dtype=torch.bfloat16)
        model.head.grad = torch.ones_like(model.head)
        with self.assertRaisesRegex(ValueError, "FP32 dtype"):
            hooks._step(model.head, optimizer, 1, 1)
        del optimizer.state[model.head]["master_weight"]
        with self.assertRaises(KeyError):
            hooks._step(model.head, optimizer, 1, 1)


if __name__ == "__main__":
    unittest.main()
