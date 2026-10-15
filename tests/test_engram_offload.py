import json
import struct
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import torch
from torch import nn

from dsv41_train.engram_offload import install_engram_sgd_hooks
from dsv41_train.models.dsv4.checkpoint import ShardedSafeTensorReader, _load_engram_rows
from dsv41_train.models.dsv4.model import RowShardedEmbedding
from dsv41_train.sft_checkpoint import load_checkpoint, save_checkpoint


def table(dtype=torch.bfloat16):
    result = RowShardedEmbedding(7, 2, sparse_gradients=True, engram_offload=True).to(dtype=dtype)
    with torch.no_grad():
        result.weight.zero_()
    return result


def backward_rows(module, values):
    module(torch.tensor([0, 0])).backward(
        torch.tensor(values, dtype=module.weight.dtype).view(2, 1).expand(2, 2).contiguous())


class EngramOffloadTest(unittest.TestCase):
    def test_bf16_microbatch_occurrences_sgd_and_aborted_step(self):
        for set_to_none in (False, True):
            module = table()
            optimizer = torch.optim.SGD(module.parameters(), lr=.125, foreach=False)
            handles = install_engram_sgd_hooks(module, optimizer)
            for handle in handles:
                self.addCleanup(handle.remove)
            for step in range(3):
                optimizer.zero_grad(set_to_none=set_to_none)
                # This abandoned step must not survive zero_grad in either mode.
                backward_rows(module, (128., 128.))
                optimizer.zero_grad(set_to_none=set_to_none)
                backward_rows(module, (256., 1.))
                backward_rows(module, (1., -256.))
                self.assertEqual(module.weight.grad._nnz(), 4)
                self.assertEqual(module.weight.grad.float().coalesce().values().tolist(), [[2., 2.]])
                optimizer.step()
                self.assertEqual(module.weight[0].tolist(), [-.25 * (step + 1)] * 2)
                self.assertEqual(module.weight[1:].count_nonzero().item(), 0)
                self.assertFalse(optimizer.state)
            # Native BF16 CPU accumulation loses one unit before SGD sees it.
            native = nn.Embedding(7, 2, sparse=True, dtype=torch.bfloat16)
            backward_rows(native, (256., 1.))
            backward_rows(native, (1., -256.))
            self.assertEqual(native.weight.grad.float().coalesce().values().tolist(), [[1., 1.]])

    def test_forward_gradient_and_update_match_fp32_reference(self):
        actual = table(torch.float32)
        reference = nn.Embedding(7, 2, sparse=True)
        with torch.no_grad():
            actual.weight.copy_(torch.arange(14).reshape(7, 2) / 16)
            reference.weight.copy_(actual.weight)
        optimizer = torch.optim.SGD(actual.parameters(), lr=.125, foreach=False)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=.125, foreach=False)
        install_engram_sgd_hooks(actual, optimizer)
        for _ in range(3):
            for opt in (optimizer, expected_optimizer):
                opt.zero_grad(set_to_none=True)
            for ids in (torch.tensor([[0, 3, 3, 6]]), torch.empty(0, dtype=torch.long)):
                output, expected = actual(ids), reference(ids)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                output.square().sum().backward()
                expected.square().sum().backward()
            torch.testing.assert_close(actual.weight.grad.to_dense(), reference.weight.grad.to_dense())
            expected_optimizer.step()
            optimizer.step()
            torch.testing.assert_close(actual.weight, reference.weight, rtol=0, atol=0)

    def test_multiple_lookups_rejected_then_new_step_recovers(self):
        module = table()
        first, second = module(torch.tensor([0])), module(torch.tensor([0]))
        with self.assertRaisesRegex(RuntimeError, "one lookup"):
            (first.sum() + second.sum()).backward()
        module.zero_grad(set_to_none=True)
        backward_rows(module, (1., 2.))
        self.assertEqual(module.weight.grad.float().coalesce().values().tolist(), [[3., 3.]])

    def test_backward_exception_before_leaf_hook_has_no_next_step_history(self):
        module = table()
        fail = [True]
        def fail_once(gradient):
            if fail[0]:
                fail[0] = False
                raise RuntimeError("aborted backward")
            return gradient
        handle = module.weight.register_hook(fail_once)
        self.addCleanup(handle.remove)
        with self.assertRaisesRegex(RuntimeError, "aborted backward"):
            backward_rows(module, (128., 128.))
        module.zero_grad(set_to_none=True)
        backward_rows(module, (1., 2.))
        self.assertEqual(module.weight.grad.float().coalesce().values().tolist(), [[3., 3.]])

    def test_replacement_rebinds_hook_and_rejects_stale_optimizer(self):
        module = table()
        optimizer = torch.optim.SGD(module.parameters(), lr=.125, foreach=False)
        install_engram_sgd_hooks(module, optimizer)
        backward_rows(module, (1., 2.))
        previous = module.weight
        module.weight = nn.Parameter(torch.zeros_like(previous))
        backward_rows(module, (1., 2.))
        self.assertFalse(previous._backward_hooks)
        with self.assertRaisesRegex(ValueError, "current Engram parameter"):
            optimizer.step()
        module.zero_grad(set_to_none=True)
        module.float()
        backward_rows(module, (1., 2.))
        self.assertIsNone(module._cpu_lookup._handle)

    def test_deepcopy_keeps_each_models_accumulation_hook(self):
        original = table()
        backward_rows(original, (1., 2.))
        copied = deepcopy(original)
        for module in (original, copied):
            module.zero_grad(set_to_none=True)
            backward_rows(module, (256., 1.))
            backward_rows(module, (1., -256.))
            self.assertEqual(module.weight.grad.float().coalesce().values().tolist(), [[2., 2.]])
        self.assertEqual(len(original.weight._backward_hooks), 1)

    def test_frozen_cpu_table_and_invalid_optimizers(self):
        module = table()
        for optimizer in (torch.optim.AdamW(module.parameters()),
                          torch.optim.SGD(module.parameters(), lr=.1, momentum=.9),
                          torch.optim.SGD(module.parameters(), lr=.1, weight_decay=.1)):
            with self.assertRaises(ValueError):
                install_engram_sgd_hooks(module, optimizer)
        module.requires_grad_(False)
        output = module(torch.tensor([0, 3, 3]))
        self.assertFalse(output.requires_grad)
        self.assertEqual(install_engram_sgd_hooks(module, torch.optim.AdamW(module.parameters())), ())
        torch.testing.assert_close(output, module.weight[[0, 3, 3]])

    def test_checkpoint_restores_changed_rows_and_next_sparse_update(self):
        actual, restored = table(), table()
        optimizers = [torch.optim.SGD(module.parameters(), lr=.125, foreach=False)
                      for module in (actual, restored)]
        for module, optimizer in zip((actual, restored), optimizers):
            install_engram_sgd_hooks(module, optimizer)
        def update(module, optimizer):
            optimizer.zero_grad(set_to_none=True)
            backward_rows(module, (256., 1.))
            backward_rows(module, (1., -256.))
            optimizer.step()
        with torch.no_grad():
            actual.weight[4].fill_(1.5)
        update(actual, optimizers[0])
        identity = {"test": "CPU Engram SGD"}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "step1"
            save_checkpoint(path, actual, optimizers[0], step=1, identity=identity, data_position=4)
            result = load_checkpoint(path, restored, optimizers[1], identity=identity)
            self.assertEqual(result, (1, 4))
            torch.testing.assert_close(actual.weight, restored.weight, rtol=0, atol=0)
            for module, optimizer in zip((actual, restored), optimizers):
                update(module, optimizer)
            torch.testing.assert_close(actual.weight, restored.weight, rtol=0, atol=0)
            self.assertEqual(restored.weight[0].tolist(), [-.5, -.5])
            self.assertEqual(restored.weight[4].tolist(), [1.5, 1.5])

    def test_direct_cpu_rows_and_strict_released_metadata(self):
        # Rank 1 owns the smaller [4,7) segment. Header validation must inspect
        # global dimensions, not mistake this local shape for the full table.
        mesh = Mock(ndim=1)
        mesh.size.return_value, mesh.get_local_rank.return_value = 2, 1
        module = RowShardedEmbedding(7, 64, mesh, sparse_gradients=True, engram_offload=True)
        values = torch.arange(7, dtype=torch.float32).view(7, 1).expand(7, 64).to(torch.float8_e4m3fn)
        scales = torch.tensor([[2., 4.]]).expand(7, 2).to(torch.float8_e8m0fnu)
        names = ("layers.1.engram.embed.weight", "layers.1.engram.embed.scale")
        header, payload = {}, bytearray()
        for name, tensor, dtype in zip(names, (values, scales), ("F8_E4M3", "F8_E8M0")):
            start = len(payload)
            payload.extend(tensor.contiguous().view(torch.uint8).numpy().tobytes())
            header[name] = dict(dtype=dtype, shape=list(tensor.shape), data_offsets=[start, len(payload)])
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            encoded = json.dumps(header).encode()
            (folder / "weights.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
            (folder / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: "weights.safetensors" for n in names}}))
            with ShardedSafeTensorReader(folder) as reader:
                result = _load_engram_rows(reader, "layers.1", module, device="cpu", dtype=torch.bfloat16)
                self.assertEqual(result.device.type, "cpu")
                expected = values.float()[4:].reshape(3, 2, 32) * scales.float()[4:, :, None]
                torch.testing.assert_close(result, expected.flatten(1).bfloat16(), rtol=0, atol=0)
                for key, replacement in (("shape", [3, 64]), ("dtype", "F32")):
                    metadata = reader.metadata(names[0])
                    previous = metadata[key]
                    metadata[key] = replacement
                    with self.assertRaisesRegex(ValueError, "released Engram layout"):
                        _load_engram_rows(reader, "layers.1", module, device="cpu", dtype=torch.bfloat16)
                    metadata[key] = previous


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
