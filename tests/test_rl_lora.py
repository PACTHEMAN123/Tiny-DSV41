import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

from dsv41_train.lora import save_adapter
from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM
from dsv41_train.models.qwen import Qwen3MoeForCausalLM
from dsv41_train.rl.checkpoint import (add_lora_identity, load_policy, policy_state,
                                     publish_policy_artifact, run_identity, validate_identity)
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.data import PromptRecord
from dsv41_train.rl.family import load_model_config
from dsv41_train.rl.rollout import generate_group
from dsv41_train.rl.runtime import make_model, train_groups
from dsv41_train.rl.sample import load_checkpoint_model
from rollout_manager.backends.pytorch_rl import PyTorchRLBackend
from qwen_helpers import tiny_config, write_hf_checkpoint


ROOT = Path(__file__).resolve().parents[1]


class LoRARLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "prompts.jsonl"
        self.data.write_text('{"prompt_ids":[1,4,7],"target_ids":[3,2]}\n')

    def fixture(self, family):
        folder = self.root / family
        folder.mkdir()
        torch.manual_seed(17)
        if family == "qwen":
            write_hf_checkpoint(folder, Qwen3MoeForCausalLM(tiny_config()))
        else:
            (folder / "config.json").write_bytes((ROOT / "examples/rl/tiny-dsv41/config.json").read_bytes())
            base = DeepSeekV41ForCausalLM(load_model_config(folder))
            torch.save(base.state_dict(), folder / "pytorch_model.bin")
            torch.save(base.model.hash.token_map, folder / "engram_token_map.pt")
        config = RLConfig.from_dict(dict(rollout_engine="native-pytorch", trainable_scope="lora",
            lora_rank=2, lora_alpha=4, seed=37, max_new_tokens=3, group_size=3,
            learning_rate=.01, kl_beta=.03, reward="toy_token_score", worker_timeout_seconds=90))
        return folder, load_model_config(folder), config

    def policy(self, folder, model_config, config, **kwargs):
        return make_model(folder, model_config, config, ("cpu",), False, **kwargs)

    def identity(self, folder, model_config, config):
        return add_lora_identity(run_identity(folder, self.data, False), config, model_config)

    def initial_adapter(self, folder, model_config, config):
        initial = self.policy(folder, model_config, config)
        with torch.no_grad():
            for name, parameter in initial.named_parameters():
                if name.endswith(".lora_b"):
                    parameter.normal_(std=.02)
        path = folder / "initial-adapter.pt"
        save_adapter(initial.backend, path)
        return replace(config, initial_adapter=str(path)), initial

    def test_validation(self):
        for override in ({"rollout_engine": "vllm"}, {"lora_dropout": .1}, {"lora_targets": []},
                         {"lora_rank": 0}, {"trainable_scope": "all", "initial_adapter": "a.pt"}):
            with self.subTest(override=override), self.assertRaises(ValueError):
                RLConfig.from_dict(dict(rollout_engine="native-pytorch", trainable_scope="lora") | override)

    def test_adapter_activation_reference_and_identity_for_both_families(self):
        async def exercise(folder, model_config, config):
            config, initial = self.initial_adapter(folder, model_config, config)
            learner = self.policy(folder, model_config, config)
            reference = self.policy(folder, model_config, config,
                                    engram_weights=learner.share_engram_memory())
            reference.eval().requires_grad_(False)
            tokens = torch.tensor([[1, 4, 7]])
            torch.testing.assert_close(learner(tokens), initial(tokens), rtol=0, atol=0)
            expected_reference = reference(tokens).detach().clone()
            frozen = {name: p.detach().clone() for name, p in learner.named_parameters() if not p.requires_grad}
            reference_adapters = {name: p.detach().clone() for name, p in reference.named_parameters()
                                  if name.endswith((".lora_a", ".lora_b"))}
            identity = self.identity(folder, model_config, config)
            directory = folder / "policies"
            def publish(version):
                return publish_policy_artifact(learner, version, directory, lineage="lora-test",
                    identity=identity, sampling={"temperature": 1., "top_p": 1, "top_k": 0})
            initial_artifact = publish(0)
            backend = PyTorchRLBackend(folder, model_config, config, False,
                engram_weights=learner.share_engram_memory(), initial_policy=initial_artifact)
            await backend.start()
            try:
                before = backend.model(tokens).detach().clone()
                optimizer = torch.optim.AdamW([p for p in learner.parameters() if p.requires_grad],
                                             lr=config.learning_rate, weight_decay=0, foreach=False)
                for step in (1, 2):
                    group = generate_group(backend.model, PromptRecord([1, 4, 7]), 3, 3, 1., None,
                        step - 1, step - 1, generator=torch.Generator().manual_seed(20 + step),
                        reward_fn=lambda record, ids: sum(ids))
                    metrics = train_groups(learner, reference, optimizer, [group], config)
                    self.assertGreater(metrics["nonzero_advantages"], 0)
                    artifact = publish(step)
                    self.assertEqual(backend.activate(artifact),
                                     {key: artifact[key] for key in ("version", "digest", "lineage")})
                    torch.testing.assert_close(backend.model(tokens), learner(tokens), rtol=0, atol=0)
                self.assertFalse(torch.equal(before, backend.model(tokens)))
                torch.testing.assert_close(reference(tokens), expected_reference, rtol=0, atol=0)
                for name, p in reference.named_parameters():
                    if name in reference_adapters:
                        torch.testing.assert_close(p, reference_adapters[name], rtol=0, atol=0)
                for name, p in learner.named_parameters():
                    if name in frozen:
                        torch.testing.assert_close(p, frozen[name], rtol=0, atol=0)
                        self.assertIsNone(p.grad)
                state = policy_state(learner)
                self.assertEqual(state["adapter_format"], 1)
                self.assertEqual(state["frozen"], {})
                self.assertTrue(all(name.endswith((".lora_a", ".lora_b")) for name in state["trainable"]))
                wrong = self.policy(folder, model_config, replace(config, initial_adapter=None))
                with self.assertRaisesRegex(ValueError, "initial adapter identity"):
                    load_policy(wrong, state)
                with self.assertRaisesRegex(ValueError, "identity mismatch"):
                    validate_identity(identity, self.identity(folder, model_config, replace(config, initial_adapter=None)))
                bad_artifact = publish_policy_artifact(learner, 2, directory, lineage="other-base",
                    identity=identity | {"model_config_sha256": "wrong-base"},
                    sampling={"temperature": 1., "top_p": 1, "top_k": 0})
                bad_actor = PyTorchRLBackend(folder, model_config, config, False,
                    engram_weights=learner.share_engram_memory(), initial_policy=bad_artifact)
                try:
                    with self.assertRaisesRegex(ValueError, "identity mismatch"):
                        await bad_actor.start()
                finally:
                    await bad_actor.close()
            finally:
                await backend.close()
        for family in ("dsv4", "qwen"):
            with self.subTest(family=family):
                asyncio.run(exercise(*self.fixture(family)))

    def test_native_manager_two_steps_resume_and_sampling_for_both_families(self):
        for family in ("dsv4", "qwen"):
            with self.subTest(family=family):
                folder, model_config, config = self.fixture(family)
                config, initial = self.initial_adapter(folder, model_config, config)
                config = replace(config, max_steps=2, output_dir=str(folder / "run"))
                config_file = folder / "rl.json"
                config_file.write_text(json.dumps(config.to_dict()))
                command = [sys.executable, "-m", "dsv41_train.rl.train", "--model-dir", str(folder),
                           "--data", str(self.data), "--config", str(config_file)]
                def run(extra):
                    result = subprocess.run(command + extra, cwd=ROOT,
                        env={**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1"},
                        text=True, capture_output=True, timeout=180)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    return result
                run([])
                second = Path(config.output_dir) / "step-00000002.pt"
                state2 = torch.load(second, map_location="cpu", weights_only=True)
                self.assertEqual(state2["policy"]["frozen"], {})
                self.assertEqual(state2["format_version"], 2)
                self.assertTrue(state2["policy_publication"]["digest"])
                run(["--resume", str(second), "--max-steps", "3"])
                third = Path(config.output_dir) / "step-00000003.pt"
                state3 = torch.load(third, map_location="cpu", weights_only=True)
                self.assertGreater(state3["next_group_id"], state2["next_group_id"])
                self.assertEqual(state3["identity"], state2["identity"])
                self.assertTrue(any(not torch.equal(value, state3["policy"]["trainable"][name])
                                    for name, value in state2["policy"]["trainable"].items()))
                restored, _, version, _ = load_checkpoint_model(model_dir=folder, checkpoint=third)
                self.assertEqual(version, 3)
                self.assertEqual(restored.lora_metadata, initial.lora_metadata | {
                    "initial_adapter_sha256": state3["identity"]["lora"]["initial_adapter_sha256"]})
                reference = self.policy(folder, model_config, config)
                for name, p in reference.named_parameters():
                    if name in state2["policy"]["trainable"]:
                        torch.testing.assert_close(p, dict(initial.named_parameters())[name], rtol=0, atol=0)
                config_file.write_text(json.dumps(replace(config, seed=38).to_dict()))
                failed = subprocess.run(command + ["--resume", str(third), "--max-steps", "4"],
                    cwd=ROOT, env={**os.environ, "PYTHONPATH": str(ROOT)}, text=True, capture_output=True, timeout=60)
                self.assertNotEqual(failed.returncode, 0)
                self.assertIn("identity mismatch", failed.stderr)


if __name__ == "__main__":
    unittest.main()
