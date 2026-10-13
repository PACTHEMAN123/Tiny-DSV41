import contextlib
import io
from pathlib import Path
import tempfile
import unittest

import torch

from create_scaled_checkpoint import checkpoint_specs, create_checkpoint
from dsv41_train.loading import LoadPolicy
from dsv41_train.models.dsv4 import load_dsv41_backbone_window
from dsv41_train.models.dsv4.engram.hash import build_compressed_token_map
from test_memory_modules import LocalMaterializer, memory_config


class ScaledCheckpointTest(unittest.TestCase):
    def test_streamed_quantized_fixture_loads_both_paths_and_trains(self):
        config = memory_config(vocab_size=48, engram_compressed_vocab_size=32,
                               engram_layer_ids=[1, 3], engram_num_embeddings=[127, 131])
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            path = Path(temporary)
            manifest = create_checkpoint(path, config, chunk_bytes=1024, shard_bytes=16384)
            specs, counts = checkpoint_specs(config)
            self.assertTrue(manifest["completed"])
            self.assertGreater(manifest["shards"], 1)
            self.assertEqual(manifest["encoded_bytes"], sum(spec.size for spec in specs))
            _, compressed = build_compressed_token_map(path / "tokenizer.json")
            self.assertEqual(compressed, 32)
            baseline = load_dsv41_backbone_window(path, num_layers=5, device="cpu", dtype=torch.float32)
            direct = load_dsv41_backbone_window(path, num_layers=5, device="cpu", dtype=torch.float32,
                load_policy=LoadPolicy("direct_shard", 19, 131072, True), materializer=LocalMaterializer())
            self.addCleanup(baseline.close)
            self.addCleanup(direct.close)
            self.assertEqual(sum(p.numel() for p in baseline.parameters()), sum(counts.values()))
            for name, value in baseline.state_dict().items():
                torch.testing.assert_close(direct.state_dict()[name], value, rtol=0, atol=0)
            experts = baseline.model.layers[0].moe.routed
            self.assertFalse(torch.equal(experts.gate_up[0], experts.gate_up[1]))
            tokens = torch.arange(3, 15).view(1, -1)
            output = baseline(tokens, labels=tokens)
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in baseline.parameters() if p.grad is not None))
            with self.assertRaisesRegex(ValueError, "empty"):
                create_checkpoint(path, config)


if __name__ == "__main__":
    unittest.main()
