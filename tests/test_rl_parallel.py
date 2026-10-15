import multiprocessing as mp
import tempfile
import unittest
from pathlib import Path

import torch

from dsv41_train.rl.algorithm import response_log_probs
from dsv41_train.rl.checkpoint import load_policy, policy_state
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.policy import DSV41Config
from dsv41_train.rl.parallel import layer_device_map
from dsv41_train.rl.runtime import make_model


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "examples/rl/tiny-dsv41"


def shared_table_child(tables, connection):
    try:
        torch.set_num_threads(1)
        model = make_model(MODEL_DIR, DSV41Config.from_model_dir(MODEL_DIR),
                           RLConfig(rollout_engine="native-pytorch", seed=37), ("cpu",), True, engram_weights=tables)
        model.assert_engram_invariants()
        table = model.backend.model.engram_tables["1"].weight
        connection.send((table.is_shared(), table.data_ptr() == tables["1"].data_ptr()))
        # A write to an isolated test table proves shared physical storage across spawn.
        table[0, 0] = 123.0
        connection.send("done")
    finally:
        connection.close()


class ParallelRLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = DSV41Config.from_model_dir(MODEL_DIR)

    def model(self, **kwargs):
        return make_model(MODEL_DIR, self.config, RLConfig(rollout_engine="native-pytorch", seed=37), ("cpu",), True, **kwargs)

    def test_gpu_counts_resolve_to_independent_device_lists(self):
        config = RLConfig.from_dict({"trainer_num_gpus": 3, "rollout_num_gpus": 2})
        self.assertEqual(config.learner_devices, ("cuda:0", "cuda:1", "cuda:2"))
        self.assertEqual(config.rollout_devices, ("cuda:3", "cuda:4"))
        self.assertTrue(config.engram_offload)
        self.assertEqual(RLConfig.from_dict(config.to_dict()), config)
        explicit = RLConfig.from_dict({"learner_devices": ["cuda:3", "cuda:1"],
                                       "rollout_devices": ["cuda:0"]})
        self.assertEqual(explicit.learner_devices, ("cuda:3", "cuda:1"))
        for values in ({"trainer_num_gpus": 2},
                       {"trainer_num_gpus": -1, "rollout_num_gpus": 1},
                       {"trainer_num_gpus": True, "rollout_num_gpus": 1},
                       {"learner_devices": ["cuda:0", "cuda:0"]},
                       {"engram_offload": "false"}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                RLConfig.from_dict(values)

    def test_layers_are_split_contiguously_between_requested_devices(self):
        self.assertEqual(layer_device_map(6, ("cuda:3", "cuda:5", "cuda:2")),
                         tuple(torch.device(f"cuda:{index}") for index in (3, 3, 5, 5, 2, 2)))
        for devices in ((), ("cpu", "cuda:0"), ("cuda:0", "cuda:0")):
            with self.subTest(devices=devices), self.assertRaises(ValueError):
                layer_device_map(6, devices)
        with self.assertRaises(ValueError):
            layer_device_map(1, ("cuda:0", "cuda:1"))

    def test_reference_and_spawned_rollout_share_the_trainer_table(self):
        model = self.model()
        self.assertTrue(callable(getattr(model, "share_engram_memory", None)),
                        "policy must expose frozen shared CPU tables")
        tables = model.share_engram_memory()
        reference = self.model(engram_weights=tables)
        for name, tensor in tables.items():
            self.assertTrue(tensor.is_shared())
            self.assertEqual(reference.backend.model.engram_tables[name].weight.data_ptr(), tensor.data_ptr())
            self.assertEqual(model.backend.model.engram_tables[name].weight.data_ptr(), tensor.data_ptr())
        ctx = mp.get_context("spawn")
        parent, child = ctx.Pipe()
        worker = ctx.Process(target=shared_table_child, args=(tables, child))
        worker.start()
        child.close()
        try:
            self.assertTrue(parent.poll(30), "rollout did not initialize")
            self.assertEqual(parent.recv(), (True, True))
            self.assertTrue(parent.poll(30))
            self.assertEqual(parent.recv(), "done")
            self.assertEqual(float(model.backend.model.engram_tables["1"].weight[0, 0]), 123.0)
            worker.join(10)
            self.assertEqual(worker.exitcode, 0)
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join(10)
            parent.close()

    def test_native_loading_uses_injected_tables(self):
        from dsv41_train.models.dsv4.model import DeepSeekV41ForCausalLM

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_bytes((MODEL_DIR / "config.json").read_bytes())
            base = DeepSeekV41ForCausalLM(self.config)
            torch.save(base.model.hash.token_map, root / "engram_token_map.pt")
            torch.save(base.state_dict(), root / "pytorch_model.bin")
            tables = {name: torch.full_like(table.weight, 0.25).share_memory_()
                      for name, table in base.model.engram_tables.items()}
            model = make_model(root, self.config, RLConfig(rollout_engine="native-pytorch"), ("cpu",), False, engram_weights=tables)
            for name, table in model.backend.model.engram_tables.items():
                self.assertEqual(table.weight.data_ptr(), tables[name].data_ptr())
            model(torch.tensor([[1, 4, 7]])).sum().backward()
            model.assert_engram_invariants()
            self.assertTrue(all(torch.all(value == 0.25) for value in tables.values()))

    def test_shared_safetensors_tables_are_not_read_again(self):
        from unittest.mock import patch

        from dsv41_train.models.dsv4.checkpoint import _SafeTensorShard
        from dsv41_train.models.dsv4.model import DeepSeekV41ForCausalLM
        from test_dsv4_checkpoint import write_safetensors

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_bytes((MODEL_DIR / "config.json").read_bytes())
            base = DeepSeekV41ForCausalLM(self.config)
            torch.save(base.model.hash.token_map, root / "engram_token_map.pt")
            write_safetensors(root / "model.safetensors", base.state_dict())
            tables = {name: table.weight.detach().share_memory_() for name, table in base.model.engram_tables.items()}
            read = _SafeTensorShard.tensor

            def read_backbone(reader, name, device):
                if "engram_tables" in name:
                    raise AssertionError("shared Engram was loaded again from disk")
                return read(reader, name, device)

            with patch.object(_SafeTensorShard, "tensor", read_backbone):
                model = make_model(root, self.config, RLConfig(rollout_engine="native-pytorch"), ("cpu",), False, engram_weights=tables)
            for name, table in model.backend.model.engram_tables.items():
                self.assertEqual(table.weight.data_ptr(), tables[name].data_ptr())

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
    def test_two_gpu_policy_matches_single_gpu_logits_and_gradients(self):
        from unittest.mock import patch

        from dsv41_train.models.dsv4.mhc import HyperConnection

        # Layer 3 reindexes KV produced by layer 1, across the two-GPU boundary.
        self.config.compress_ratios = [0, 2, 2, 2, 1, 1]
        self.config.kv_source_layer_ids = [1, 4]
        self.config.candidate_source_layer_id = 4
        single = self.model().to_training_devices(("cuda:0",), torch.float32)
        parallel = self.model().to_training_devices(("cuda:0", "cuda:1"), torch.float32)
        tokens = torch.tensor([[1, 4, 7, 3, 8, 9, 5, 6]], device="cuda:0")
        collapse = HyperConnection.collapse

        def checked_collapse(streams, weights):
            self.assertEqual(torch.cuda.current_device(), streams.device.index)
            return collapse(streams, weights)

        with torch.cuda.device(0), patch.object(HyperConnection, "collapse", checked_collapse):
            expected = single(tokens)
            actual = parallel(tokens)
        self.assertEqual(actual.device, torch.device("cuda:1"))
        self.assertEqual({p.device for p in parallel.parameters()},
                         {torch.device("cuda:0"), torch.device("cuda:1")})
        torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-4, atol=2e-5)
        expected.square().mean().backward()
        actual.square().mean().backward()
        original = dict(single.named_parameters())
        for name, parameter in parallel.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad.cpu(), original[name].grad.cpu(), rtol=5e-4, atol=2e-5)
        load_policy(parallel, policy_state(single))
        parallel.assert_engram_invariants()
        single.zero_grad(set_to_none=True)
        parallel.zero_grad(set_to_none=True)
        parallel.backend.gradient_checkpointing_enable()
        with torch.cuda.device(0), patch.object(HyperConnection, "collapse", checked_collapse):
            expected = response_log_probs(single, [1, 4, 7, 3, 8, 9], [5, 6, 2], 0.8)
            actual = response_log_probs(parallel, [1, 4, 7, 3, 8, 9], [5, 6, 2], 0.8)
            torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=2e-4, atol=2e-5)
            expected.sum().backward()
            actual.sum().backward()
        for name, parameter in parallel.named_parameters():
            if parameter.grad is not None:
                torch.testing.assert_close(parameter.grad.cpu(), original[name].grad.cpu(), rtol=5e-4, atol=2e-5)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA devices required")
    def test_disabling_offload_places_frozen_tables_with_their_layers(self):
        config = RLConfig(rollout_engine="native-pytorch", seed=37, engram_offload=False)
        model = make_model(MODEL_DIR, self.config, config, ("cuda:0", "cuda:1"), True)
        for layer in model.backend.model.layers:
            if layer.engram is not None:
                table = model.backend.model.engram_tables[str(layer.layer_id)].weight
                self.assertEqual(table.device, next(layer.parameters()).device)
                self.assertFalse(table.requires_grad)
        model(torch.tensor([[1, 4, 7, 3]], device="cuda:0")).sum().backward()
        model.assert_engram_invariants()
