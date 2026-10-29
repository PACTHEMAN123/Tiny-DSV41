import types
import unittest

import torch

from dsv41_train.rl.native_lora import import_native


class NativeGatherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native, _ = import_native()
        cls.gather = staticmethod(native('models.dsv4.ordered_gather').gather_compressed_kv)
        cls.attend = staticmethod(native('models.dsv4.model').CompressedAttention._attend)

    def check_gather(self, device):
        for dtype in (torch.float32, torch.bfloat16):
            for indices in (torch.tensor([[[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, -1, 2]],
                                          [[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, -1, 0]]]),
                            torch.full((2, 2, 3), -1), torch.empty((2, 2, 0), dtype=torch.long)):
                indices = indices.to(device)
                batch = indices.shape[0]
                entries = torch.randn(batch, 33, 3, device=device, dtype=dtype).transpose(1, 2).requires_grad_(True)
                upstream = torch.randn(*indices.shape, 33, device=device, dtype=dtype)
                rows = torch.arange(batch, device=device)[:, None, None]
                untouched = entries.detach().clone()
                output = self.gather(entries, indices)
                torch.testing.assert_close(entries, untouched, rtol=0, atol=0)
                expected = entries[rows, indices.clamp_min(0)].masked_fill(indices.lt(0).unsqueeze(-1), 0)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                gradient, = torch.autograd.grad(output, entries, upstream)
                reference_entries = entries.detach().double().requires_grad_(True)
                reference_output = reference_entries[rows, indices.clamp_min(0)].masked_fill(indices.lt(0).unsqueeze(-1), 0)
                reference, = torch.autograd.grad(reference_output, reference_entries, upstream.double())
                torch.testing.assert_close(gradient, reference.to(dtype), rtol=1e-5, atol=1e-6)
                repeat, = torch.autograd.grad(self.gather(entries, indices), entries, upstream)
                torch.testing.assert_close(gradient, repeat, rtol=0, atol=0)

    def test_masked_forward_and_arbitrary_upstream_cpu(self):
        self.check_gather('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_fixed_order_cuda(self):
        self.check_gather('cuda')

    def test_native_attention_output_and_other_gradients_preserved(self):
        torch.manual_seed(71)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for dtype in (torch.float32, torch.bfloat16):
            query = torch.randn(1, 2, 5, 16, device=device, dtype=dtype).requires_grad_(True)
            kv = torch.randn(1, 1, 5, 16, device=device, dtype=dtype).requires_grad_(True)
            entries = torch.randn(1, 3, 16, device=device, dtype=dtype).requires_grad_(True)
            sinks = torch.randn(2, device=device).requires_grad_(True)
            indices = torch.tensor([[[-1, -1], [0, -1], [0, 1], [1, 0], [2, 1]]], device=device)
            valid = indices >= 0
            rows = torch.zeros(1, 1, 1, device=device, dtype=torch.long)
            config = types.SimpleNamespace(head_dim=16, dropout=0., training=True)
            mask = torch.zeros(5, 5, device=device).masked_fill(torch.ones(5, 5, device=device).triu(1).bool(), float('-inf'))
            outputs, gradients = [], []
            for selected in (entries[rows, indices.clamp_min(0)], self.gather(entries, indices)):
                output = self.attend(config, query, kv, mask, selected, valid, sinks)
                gradient = torch.autograd.grad(output.float().square().sum(), (query, kv, sinks, selected), retain_graph=True)
                self.assertEqual(torch.count_nonzero(gradient[-1].masked_select((~valid).unsqueeze(-1))), 0)
                outputs.append(output)
                gradients.append(gradient)
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            for left, right in zip(*gradients):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
