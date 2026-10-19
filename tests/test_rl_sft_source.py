"""Full SFT snapshots initialize native RL roles and survive RL recovery."""

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from dsv41_train.engram_offload import install_engram_sgd_hooks
from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM
from dsv41_train.rl.checkpoint import add_engram_identity, add_sft_identity, load_policy, policy_state, publish_policy_artifact, run_identity
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.family import initial_sft_source, load_model_config
from dsv41_train.rl.runtime import make_model, train_groups
from dsv41_train.rl.sample import load_checkpoint_model
from dsv41_train.rl.algorithm import response_log_probs
from dsv41_train.rl.rollout import RolloutGroup, Sample
from dsv41_train.runtime import Runtime
from dsv41_train.sft_checkpoint import save_checkpoint
from dsv41_train.sft_engram import EngramCheckpoint
from rollout_manager.backends.pytorch_rl import PyTorchRLBackend
from train_sft import train_update
from test_sft_export import create_base, place_bf16


ROOT = Path(__file__).resolve().parents[1]


class FullSFTRLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def fixture(self, sparse, dtype="float32"):
        folder = self.root / f"{sparse}-{dtype}"
        folder.mkdir()
        create_base(folder)
        config = load_model_config(folder)
        model = DeepSeekV41ForCausalLM(config, sparse_engram_gradients=True, engram_offload=sparse)
        base = torch.load(folder / "pytorch_model.bin", weights_only=True)
        model.load_state_dict(base)
        if dtype == "bfloat16":
            place_bf16(model, "cpu")
        with torch.no_grad():
            for layer in model.model.layers:
                layer.moe.router.selection_bias.add_(torch.arange(config.n_routed_experts) * .001)
                if layer.attention.indexer is not None:
                    next(layer.attention.indexer.parameters()).add_(.01)
        optimizer = torch.optim.SGD(model.parameters(), lr=.03, foreach=False)
        install_engram_sgd_hooks(model, optimizer)
        if sparse:
            model._sft_engram_checkpoint = EngramCheckpoint(model, optimizer,
                lambda name, ids: base[name][ids].to(getattr(torch, dtype)))
        ids = torch.tensor([[1, 4, 7, 9, 3, 2]])
        labels = ids.clone()
        labels[:, :3] = -100
        for _ in range(2):
            train_update(model, optimizer, [dict(input_ids=ids, labels=labels, attention_mask=torch.ones_like(ids))],
                         family="dsv4", runtime=Runtime(torch.device("cpu")))
        identity = run_identity(folder, folder / "data.jsonl", False)
        identity["training_config"] = dict(family="dsv4", mode="full", num_layers=config.num_hidden_layers)
        snapshot = folder / "sft-step2"
        save_checkpoint(snapshot, model, optimizer, step=2, identity=identity, data_position=2)
        rl = RLConfig.from_dict(dict(rollout_engine="native-pytorch", trainable_scope="all", dtype=dtype,
            train_engram=True, optimizer="sgd", initial_sft_checkpoint=str(snapshot), learning_rate=.03,
            kl_beta=.03, reward="toy_token_score", group_size=3, max_new_tokens=2, worker_timeout_seconds=90))
        return folder, config, model, rl

    def identity(self, folder, config, rl):
        source = initial_sft_source(folder, config, rl, False)
        return add_sft_identity(add_engram_identity(run_identity(folder, folder / "data.jsonl", False), rl), source)

    def test_full_and_sparse_three_roles_preserve_every_tensor_and_reference(self):
        async def exercise(sparse, dtype):
            folder, config, sft, rl = self.fixture(sparse, dtype)
            if not sparse and dtype == "float32":
                rl = replace(rl, train_engram=False)
            learner = make_model(folder, config, rl, ("cpu",), False)
            baseline = learner.export_engram_baseline()
            self.assertEqual(baseline.identity["initial_sft_checkpoint"], learner.initial_sft_identity)
            for ids in learner.engram_state.dirty.values():
                self.assertEqual(ids.numel(), 0)
            identity = self.identity(folder, config, rl)
            artifact = publish_policy_artifact(learner, 0, folder / "policies", lineage="full-sft-test",
                identity=identity, sampling={"temperature": 1., "top_p": 1, "top_k": 0})
            actor = PyTorchRLBackend(folder, config, rl, False, engram_weights=baseline, initial_policy=artifact)
            with patch("dsv41_train.rl.engram_state.EngramBaseline.create") as create:
                reference = make_model(folder, config, rl, ("cpu",), False,
                    engram_weights=baseline, role="reference").eval().requires_grad_(False)
                await actor.start()
                create.assert_not_called()  # Both roles reuse the sealed FD; no second table allocation.
            try:
                tokens = torch.tensor([[1, 4, 7, 9]])
                expected = sft.state_dict()
                with torch.no_grad():
                    wanted = sft.eval()(tokens, compute_aux_loss=False).logits
                    for role in (learner, reference, actor.model):
                        self.assertEqual(role.backend.state_dict().keys(), expected.keys())
                        for name, value in role.backend.state_dict().items():
                            self.assertEqual(value.dtype, expected[name].dtype, name)
                            torch.testing.assert_close(value, expected[name], rtol=0, atol=0, msg=name)
                        torch.testing.assert_close(role(tokens), wanted, rtol=0, atol=0)
                    samples = [Sample([1, 4, 7], response,
                        response_log_probs(actor.model, [1, 4, 7], response).tolist(), reward, 0)
                        for response, reward in (([3, 4], 0.), ([5, 6], 1.))]
                optimizer = torch.optim.SGD([p for p in learner.parameters() if p.requires_grad], lr=.03, foreach=False)
                install_engram_sgd_hooks(learner, optimizer)
                metrics = train_groups(learner, reference, optimizer, [RolloutGroup(samples, 0, 0)], rl)
                self.assertEqual(metrics["nonzero_advantages"], 2)
                self.assertTrue(any(not torch.equal(value, expected[name]) for name, value in learner.backend.state_dict().items()))
                for name, value in reference.backend.state_dict().items():
                    torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
                new_artifact = publish_policy_artifact(learner, 1, folder / "policies", lineage="full-sft-test",
                    identity=identity, sampling={"temperature": 1., "top_p": 1, "top_k": 0})
                actor.activate(new_artifact)
                for name, value in learner.backend.state_dict().items():
                    torch.testing.assert_close(actor.model.backend.state_dict()[name], value, rtol=0, atol=0)
                wrong = dict(policy_state(learner), initial_sft_checkpoint={"manifest_sha256": "other", "step": 2})
                before = {name: value.clone() for name, value in actor.model.state_dict().items()}
                with self.assertRaisesRegex(ValueError, "SFT checkpoint identity"):
                    load_policy(actor.model, wrong)
                for name, value in actor.model.state_dict().items():
                    torch.testing.assert_close(value, before[name], rtol=0, atol=0)
            finally:
                await actor.close()
        for sparse, dtype in ((False, "float32"), (True, "float32"), (False, "bfloat16"), (True, "bfloat16")):
            with self.subTest(sparse=sparse, dtype=dtype):
                asyncio.run(exercise(sparse, dtype))

    def test_native_twenty_steps_fresh_process_resume_and_sampling(self):
        folder, config, sft, rl = self.fixture(True)
        rl = replace(rl, max_steps=20, output_dir=str(folder / "rl"))
        config_file = folder / "rl.json"
        config_file.write_text(json.dumps(rl.to_dict()))
        command = [sys.executable, "-m", "dsv41_train.rl.train", "--model-dir", str(folder),
                   "--data", str(folder / "data.jsonl"), "--config", str(config_file),
                   "--initial-sft-checkpoint", rl.initial_sft_checkpoint]
        def run(extra):
            result = subprocess.run(command + extra, cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1"},
                text=True, capture_output=True, timeout=180)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        run([])
        checkpoint = Path(rl.output_dir) / "step-00000020.pt"
        old = torch.load(checkpoint, weights_only=True)
        run(["--resume", str(checkpoint), "--max-steps", "21"])
        final = Path(rl.output_dir) / "step-00000021.pt"
        new = torch.load(final, weights_only=True)
        self.assertEqual(new["identity"], old["identity"])
        self.assertGreater(new["next_group_id"], old["next_group_id"])
        self.assertTrue(any(not torch.equal(value, new["policy"]["trainable"][name])
                            for name, value in old["policy"]["trainable"].items()))
        restored, _, version, _ = load_checkpoint_model(model_dir=folder, checkpoint=final)
        self.assertEqual(version, 21)
        actual, expected = policy_state(restored), new["policy"]
        for section in ("trainable", "frozen"):
            torch.testing.assert_close(actual[section], expected[section], rtol=0, atol=0)
        self.assertEqual(actual["initial_sft_checkpoint"], expected["initial_sft_checkpoint"])
        self.assertEqual(actual["engram_delta"]["base"], expected["engram_delta"]["base"])
        torch.testing.assert_close(actual["engram_delta"]["tables"], expected["engram_delta"]["tables"], rtol=0, atol=0)
        reference = make_model(folder, config, rl, ("cpu",), False, role="reference")
        for name, value in reference.backend.state_dict().items():
            torch.testing.assert_close(value, sft.state_dict()[name], rtol=0, atol=0)
        # A valid but different source cannot silently replace the SFT starting point.
        manifest = Path(rl.initial_sft_checkpoint) / "manifest.json"
        data = json.loads(manifest.read_text())
        data["step"] += 1
        manifest.write_text(json.dumps(data))
        with patch("dsv41_train.rl.sample.make_model") as build:
            with self.assertRaisesRegex(ValueError, "identity mismatch: initial_sft_checkpoint"):
                load_checkpoint_model(model_dir=folder, checkpoint=final)
            build.assert_not_called()

    def test_invalid_sources_and_combinations_fail_before_model_allocation(self):
        folder, config, _, rl = self.fixture(True)
        for change in ({"trainable_scope": "lora"}, {"rollout_engine": "vllm"}, {"engram_offload": False},
                       {"initial_adapter": "adapter.pt"}, {"initial_engram_delta": "rows.pt"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(rl, **change).validate()
        with self.assertRaisesRegex(ValueError, "SFT Engram shape/dtype differs"):
            make_model(folder, config, replace(rl, dtype="bfloat16"), ("cpu",), False)
        with patch("dsv41_train.rl.policy.DeepSeekV41ForCausalLM") as build:
            with self.assertRaisesRegex(ValueError, "pretrained DSV4"):
                make_model(folder, config, rl, ("cpu",), True)
            from qwen_helpers import tiny_config
            with self.assertRaisesRegex(ValueError, "pretrained DSV4"):
                make_model(folder, tiny_config(), rl, ("cpu",), False)
            manifest = Path(rl.initial_sft_checkpoint) / "manifest.json"
            data = json.loads(manifest.read_text())
            data["ranks"][0]["sha256"] = "bad"
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "checksum"):
                make_model(folder, config, rl, ("cpu",), False)
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
