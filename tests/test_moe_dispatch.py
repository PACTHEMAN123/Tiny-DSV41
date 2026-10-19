import unittest

import torch

from dsv41_train.models.dsv4.moe import TokenDispatcher


class MoEDispatchTest(unittest.TestCase):
    def test_combine_uses_router_slot_order_and_preserves_backward(self):
        dispatcher = TokenDispatcher()
        dispatcher.expert_range(3)
        ids = torch.tensor([[0, 2, 1], [2, 0, 1]])
        _, _, _, metadata = dispatcher.dispatch(torch.zeros(2, 2), ids, torch.ones(2, 3))
        # Expert-order scatter would accumulate 1e8, -1e8, 1 and yield 1.
        # Router-slot order accumulates 1e8, 1, -1e8 and yields 0 in FP32.
        contributions = torch.tensor([
            [[1e8, -1e8], [1., -2.], [-1e8, 1e8]],
            [[3., 1.], [4., 2.], [5., 3.]],
        ], requires_grad=True)
        order = torch.argsort(ids.flatten(), stable=True)
        output = dispatcher.combine(contributions.flatten(0, 1)[order], metadata)
        self.assertTrue(torch.equal(output, torch.tensor([[0., 0.], [12., 6.]])))
        upstream = torch.tensor([[2., -3.], [4., 5.]])
        output.backward(upstream)
        self.assertTrue(torch.equal(contributions.grad, upstream[:, None].expand(-1, 3, -1)))

    def test_empty_source_and_single_route(self):
        dispatcher = TokenDispatcher()
        dispatcher.expert_range(8)
        for count, topk in ((0, 1), (0, 6), (3, 1)):
            with self.subTest(count=count, topk=topk):
                hidden = torch.arange(count * 7, dtype=torch.bfloat16).reshape(count, 7).requires_grad_()
                ids = torch.arange(count * topk).reshape(count, topk) % 8
                routed, _, _, metadata = dispatcher.dispatch(hidden, ids, torch.ones(count, topk))
                output = dispatcher.combine(routed, metadata)
                self.assertEqual(output.dtype, torch.float32)
                self.assertTrue(torch.equal(output, hidden.float() * topk))
                output.sum().backward()
                self.assertTrue(torch.equal(hidden.grad, torch.full_like(hidden, topk)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_combine_repeated_forward_and_backward_are_exact(self):
        generator = torch.Generator().manual_seed(42)
        dispatcher = TokenDispatcher()
        dispatcher.expert_range(48)
        ids = ((torch.arange(128)[:, None] * 7 + torch.arange(6)[None, :] * 5) % 48).cuda()
        _, _, _, metadata = dispatcher.dispatch(torch.zeros(128, 5120, device="cuda"), ids, torch.ones_like(ids))
        values = torch.randn(128 * 6, 5120, generator=generator).bfloat16().cuda().requires_grad_()
        upstream = torch.randn(128, 5120, generator=generator).cuda()
        expected_grad = upstream[metadata.token_indices].to(values.dtype)
        first = None
        for _ in range(30):
            output = dispatcher.combine(values, metadata)
            gradient, = torch.autograd.grad(output, values, upstream)
            self.assertTrue(torch.equal(gradient, expected_grad))
            self.assertTrue(torch.isfinite(output).all())
            if first is None:
                first = output.detach().clone()
            else:
                self.assertTrue(torch.equal(output, first))


if __name__ == "__main__":
    unittest.main()
