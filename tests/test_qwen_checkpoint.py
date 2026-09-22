import tempfile
import unittest
from pathlib import Path

import torch

from dsv41_train.dispatch import TokenDispatcher
from dsv41_train.models.qwen import Qwen3MoeForCausalLM, load_qwen3_moe
from qwen_helpers import tiny_config, write_hf_checkpoint

try:
    from safetensors.torch import save_file
except ImportError:
    save_file = None


class UpperHalfDispatcher(TokenDispatcher):
    def expert_range(self, num_experts: int) -> tuple[int, int]:
        return num_experts // 2, num_experts


@unittest.skipIf(save_file is None, "safetensors is not installed")
class QwenCheckpointTest(unittest.TestCase):
    def test_loads_and_fuses_hf_expert_weights(self):
        config = tiny_config()
        torch.manual_seed(9)
        source = Qwen3MoeForCausalLM(config)

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_hf_checkpoint(folder, source)

            loaded = load_qwen3_moe(folder, dtype=None)

        for name, expected in source.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[name], expected)

    def test_loads_only_the_experts_owned_by_an_ep_rank(self):
        config = tiny_config()
        torch.manual_seed(11)
        source = Qwen3MoeForCausalLM(config)

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_hf_checkpoint(folder, source)
            loaded = load_qwen3_moe(
                folder,
                dtype=None,
                token_dispatcher=UpperHalfDispatcher(),
            )

        start = config.num_experts // 2
        for layer_id in range(config.num_hidden_layers):
            expected = source.model.layers[layer_id].mlp.experts
            actual = loaded.model.layers[layer_id].mlp.experts
            self.assertEqual(actual.expert_start, start)
            torch.testing.assert_close(actual.gate_up_proj, expected.gate_up_proj[start:])
            torch.testing.assert_close(actual.down_proj, expected.down_proj[start:])


if __name__ == "__main__":
    unittest.main()
