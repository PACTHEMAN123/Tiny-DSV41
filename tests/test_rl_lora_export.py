import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from dsv41_train.rl.lora_export import export_q_a_update
from rollout_manager.backends.vllm_lora_transport import unpack_update


class NativeAdapter(nn.Linear):
    def __init__(self):
        super().__init__(16, 8, bias=False, dtype=torch.bfloat16)
        self.weight.requires_grad_(False)
        self.config = types.SimpleNamespace(rank=8, alpha=8, dropout=0, targets=("attention.q_a",))
        self.scaling, self.adapters_enabled, self.merged = 1, True, False
        self.dropout = nn.Dropout(0)
        self.lora_a = nn.Parameter(torch.arange(128).reshape(8, 16).to(torch.bfloat16))
        self.lora_b = nn.Parameter(torch.ones(8, 8, dtype=torch.bfloat16))


def native_model():
    model = nn.Module()
    model.config = dict(hidden_size=16, q_lora_rank=8, head_dim=4, num_hidden_layers=2)
    model.model = nn.Module()
    model.model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    for layer in model.model.layers:
        layer.attention = nn.Module()
        layer.attention.q_a = NativeAdapter()
    return model


class NativeQueryExportTests(unittest.TestCase):
    def setUp(self):
        self.model = native_model()
        self.kwargs = dict(version=1, lineage="real-run", adapter_id=2,
                           base_identity={"config_sha256": "a" * 64, "index_sha256": "b" * 64})

    def test_worker_ready_mapping_snapshot_and_optimizer_membership(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params)
        manifest, payload = export_q_a_update(self.model, optimizer=optimizer, **self.kwargs)
        tensors = unpack_update(manifest, payload)
        self.assertEqual(len(tensors), 4)
        for i, layer in enumerate(self.model.model.layers):
            prefix = f"language_model.model.layers.{i}.attn.wq_a"
            self.assertTrue(torch.equal(tensors[prefix + ".lora_A.weight"], layer.attention.q_a.lora_a))
            self.assertTrue(torch.equal(tensors[prefix + ".lora_B.weight"], layer.attention.q_a.lora_b))
        with torch.no_grad():
            self.model.model.layers[0].attention.q_a.lora_b.add_(3)
        self.assertEqual(float(tensors["language_model.model.layers.0.attn.wq_a.lora_B.weight"].max()), 1)
        self.assertEqual(manifest["peft"], {"r": 8, "lora_alpha": 8, "target_modules": ["wq_a"]})

    def test_rejects_trainable_base_wrong_profile_and_nonfinite_adapter(self):
        layer = self.model.model.layers[0].attention.q_a
        layer.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "base"):
            export_q_a_update(self.model, **self.kwargs)
        layer.weight.requires_grad_(False)
        layer.config.alpha = 16
        with self.assertRaisesRegex(ValueError, "query-only"):
            export_q_a_update(self.model, **self.kwargs)
        layer.config.alpha = 8
        with torch.no_grad():
            layer.lora_b.fill_(float("nan"))
        with self.assertRaisesRegex(ValueError, "finite"):
            export_q_a_update(self.model, **self.kwargs)

    def test_rejects_optimizer_with_base_or_nonfinite_state(self):
        optimizer = torch.optim.AdamW(self.model.parameters())
        with self.assertRaisesRegex(ValueError, "optimizer"):
            export_q_a_update(self.model, optimizer=optimizer, **self.kwargs)
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params)
        optimizer.state[params[0]]["exp_avg"] = torch.full_like(params[0], float("inf"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            export_q_a_update(self.model, optimizer=optimizer, **self.kwargs)

    def test_consensus_rejects_disagreeing_rank_before_tensor_gather(self):
        def disagree(statuses, local):
            statuses[:] = [local, {**local, "signature": []}]
        with patch("dsv41_train.rl.lora_export.dist.is_initialized", return_value=True), \
                patch("dsv41_train.rl.lora_export.dist.get_world_size", return_value=2), \
                patch("dsv41_train.rl.lora_export.dist.all_gather_object", side_effect=disagree), \
                patch("dsv41_train.rl.lora_export.pack_update") as pack:
            with self.assertRaisesRegex(ValueError, "disagree"):
                export_q_a_update(self.model, **self.kwargs)
            pack.assert_not_called()


if __name__ == "__main__":
    unittest.main()
