import asyncio
from copy import deepcopy
from dataclasses import replace
import fcntl
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

from dsv41_train.engram_offload import install_engram_sgd_hooks
from dsv41_train.models.dsv4 import DeepSeekV41ForCausalLM
from dsv41_train.models.dsv4.checkpoint import ShardedSafeTensorReader
from dsv41_train.rl.checkpoint import (add_engram_identity, add_lora_identity, load_policy,
                                     policy_state, publish_policy_artifact, run_identity)
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.data import PromptRecord
from dsv41_train.rl.engram_state import EngramBaseline, EngramState, baseline_identity, memfd_seals
from dsv41_train.rl.family import load_model_config
from dsv41_train.rl.rollout import generate_group
from dsv41_train.rl.runtime import make_model, train_groups
from dsv41_train.rl.sample import load_checkpoint_model
from rollout_manager.backends.pytorch_rl import PyTorchRLBackend

ROOT = Path(__file__).resolve().parents[1]


class EngramRLStateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        (self.folder / "config.json").write_bytes((ROOT / "examples/rl/tiny-dsv41/config.json").read_bytes())
        self.model_config = load_model_config(self.folder)
        torch.manual_seed(17)
        self.base = DeepSeekV41ForCausalLM(self.model_config)
        torch.save(self.base.state_dict(), self.folder / "pytorch_model.bin")
        torch.save(self.base.model.hash.token_map, self.folder / "engram_token_map.pt")
        self.data = self.folder / "prompts.jsonl"
        self.data.write_text('{"prompt_ids":[1,4,7],"target_ids":[3,2]}\n')
        self.config = RLConfig.from_dict(dict(rollout_engine="native-pytorch", optimizer="sgd",
            train_engram=True, engram_cache_mib=1, seed=37, max_new_tokens=3, group_size=3,
            learning_rate=.01, kl_beta=.03, reward="toy_token_score", worker_timeout_seconds=90))

    def policy(self, config=None, **kwargs):
        return make_model(self.folder, self.model_config, config or self.config, ("cpu",), False, **kwargs)

    def identity(self, config):
        return add_engram_identity(add_lora_identity(run_identity(self.folder, self.data, False),
            config, self.model_config), config)

    def optimizer(self, model):
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                    lr=self.config.learning_rate, foreach=False)
        install_engram_sgd_hooks(model, optimizer)
        return optimizer

    def test_released_fp8_baseline_streams_rows_with_original_scales(self):
        config = replace(self.model_config, checkpoint_format="released", engram_num_embeddings=[7, 7],
                         engram_head_dim=64)
        values = torch.arange(7).float()[:, None].expand(7, 64).to(torch.float8_e4m3fn)
        scales = torch.tensor([[2., 4.]]).expand(7, 2).to(torch.float8_e8m0fnu)
        header, payload = {}, bytearray()
        for layer in config.engram_layer_ids:
            for suffix, tensor, dtype in (("weight", values, "F8_E4M3"), ("scale", scales, "F8_E8M0")):
                start = len(payload)
                payload.extend(tensor.contiguous().view(torch.uint8).numpy().tobytes())
                header[f"layers.{layer}.engram.embed.{suffix}"] = dict(dtype=dtype, shape=list(tensor.shape),
                                                                     data_offsets=[start, len(payload)])
        def write():
            encoded = json.dumps(header).encode()
            (self.folder / "rows.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
        write()
        (self.folder / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {name: "rows.safetensors" for name in header}}))
        calls, original = [], ShardedSafeTensorReader.tensor_rows
        def read(reader, name, start, stop, device):
            calls.append(stop - start)
            return original(reader, name, start, stop, device)
        with patch.object(EngramBaseline, "chunk_rows", 2), patch.object(ShardedSafeTensorReader, "tensor_rows", read):
            baseline = EngramBaseline.create(self.folder, config, torch.bfloat16, False, 37)
        expected = (values.float().reshape(7, 2, 32) * scales.float()[:, :, None]).flatten(1).bfloat16()
        self.assertEqual(max(calls), 2)
        for name in baseline.shapes:
            torch.testing.assert_close(baseline.view(name), expected, rtol=0, atol=0)
        header["layers.1.engram.embed.scale"]["dtype"] = "F8_E4M3"
        write()
        with self.assertRaisesRegex(ValueError, "released Engram layout"):
            EngramBaseline.create(self.folder, config, torch.bfloat16, False, 37)

    def test_native_rows_gradient_and_sgd_match_dense_oracle(self):
        actual, dense = self.policy(), self.policy()
        for name, table in list(dense.backend.model.engram_tables.items()):
            dense.backend.model.engram_tables[name] = nn.Embedding.from_pretrained(table.weight.detach().clone(), freeze=False)
        optimizer, dense_optimizer = self.optimizer(actual), torch.optim.SGD(dense.parameters(), lr=.01, foreach=False)
        initial = {name: table.weight.detach().clone() for name, table in actual.engram_state.tables.items()}
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            dense_optimizer.zero_grad(set_to_none=True)
            for tokens in (torch.tensor([[1, 4, 7, 4, 7]]), torch.tensor([[1, 4, 7, 4]])):
                output, expected = actual(tokens), dense(tokens)
                torch.testing.assert_close(output, expected, rtol=0, atol=0)
                output.square().mean().backward()
                expected.square().mean().backward()
            for name, table in actual.engram_state.tables.items():
                self.assertTrue(table.weight.grad.is_sparse)
                torch.testing.assert_close(table.weight.grad.to_dense(),
                    dense.backend.model.engram_tables[name].weight.grad, rtol=1e-5, atol=1e-9)
            actual.mark_engram_touched()
            optimizer.step()
            dense_optimizer.step()
            for name, table in actual.engram_state.tables.items():
                torch.testing.assert_close(table.weight, dense.backend.model.engram_tables[name].weight, rtol=1e-5, atol=1e-9)
        for name, table in actual.engram_state.tables.items():
            self.assertFalse(torch.equal(initial[name], table.weight))
        self.assertTrue(all("engram_tables" not in name for name in policy_state(actual)["trainable"]))

    def test_cow_isolation_absolute_snapshots_skip_and_restore(self):
        baseline = self.policy().export_engram_baseline()
        learner, actor, reference = (EngramState(baseline, role, True) for role in ("learner", "actor", "reference"))
        original = {name: baseline.view(name).clone() for name in baseline.shapes}
        empty = learner.state_dict()
        optimizer = torch.optim.SGD([table.weight for table in learner.tables.values()], lr=.1, foreach=False)
        for row in (2, 5):
            optimizer.zero_grad(set_to_none=True)
            for table in learner.tables.values():
                table(torch.tensor([row, row])).sum().backward()
            learner.mark_touched()
            optimizer.step()
            if row == 2:
                first = learner.state_dict()
        second = learner.state_dict()
        for name in baseline.shapes:
            torch.testing.assert_close(actor.tables[name].weight, original[name], rtol=0, atol=0)
            torch.testing.assert_close(reference.tables[name].weight, original[name], rtol=0, atol=0)
            torch.testing.assert_close(baseline.view(name), original[name], rtol=0, atol=0)
        actor.load_state_dict(second)  # Skip the first publication.
        for name in baseline.shapes:
            torch.testing.assert_close(actor.tables[name].weight, learner.tables[name].weight, rtol=0, atol=0)
        actor.load_state_dict(first)  # Restore rows absent from the older snapshot.
        for name in baseline.shapes:
            torch.testing.assert_close(actor.tables[name].weight[5], original[name][5], rtol=0, atol=0)
            self.assertFalse(torch.equal(actor.tables[name].weight[2], original[name][2]))
        actor.load_state_dict(empty)
        for name in baseline.shapes:
            torch.testing.assert_close(actor.tables[name].weight, original[name], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "reference"):
            reference.load_state_dict(second)
        if sys.platform == "linux":
            for fd in baseline._fds.values():
                self.assertEqual(memfd_seals(fd) & 15, 15)
                with self.assertRaises(OSError):
                    os.pwrite(fd, b"x", 0)
        else:
            for fd in baseline._fds.values():
                self.assertEqual(fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE, os.O_RDONLY)

    def test_initial_sft_delta_is_reference_baseline_and_invalid_rows_are_atomic(self):
        learner = self.policy()
        initial = learner.engram_state.state_dict()
        for name, record in initial["tables"].items():
            record["indices"] = torch.tensor([3, 7])
            record["values"] = learner.engram_state.tables[name].weight.detach()[[3, 7]].clone() + .125
        self.assertEqual(initial["base"], baseline_identity(self.folder, self.model_config, torch.float32, False, 37))
        path = self.folder / "initial-engram.pt"
        torch.save(initial, path)
        config = replace(self.config, initial_engram_delta=str(path))
        model = self.policy(config)
        reference = self.policy(config, role="reference", engram_weights=model.export_engram_baseline())
        for name, record in initial["tables"].items():
            torch.testing.assert_close(reference.engram_state.tables[name].weight[[3, 7]], record["values"], rtol=0, atol=0)
        self.assertTrue(all(not ids.numel() for ids in model.engram_state.dirty.values()))
        good = policy_state(model)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        table_name = next(iter(initial["tables"]))
        for kind in ("identity", "dtype", "duplicate", "nan", "range", "missing"):
            bad = deepcopy(good)
            table = bad["engram_delta"]["tables"][table_name]
            table.update(indices=torch.tensor([1, 2]), values=torch.zeros(2, self.model_config.engram_head_dim))
            if kind == "identity":
                bad["engram_delta"]["base"]["initial_delta_sha256"] = "wrong"
            elif kind == "dtype":
                table["values"] = table["values"].double()
            elif kind == "duplicate":
                table["indices"] = torch.tensor([1, 1])
            elif kind == "nan":
                table["values"][0, 0] = float("nan")
            elif kind == "range":
                table["indices"][-1] = 100000
            else:
                del bad["engram_delta"]
            for value in bad["trainable"].values():
                value.add_(1)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                load_policy(model, bad)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)
        frozen = self.policy(replace(config, train_engram=False, optimizer="adamw"))
        self.assertTrue(all(not table.weight.requires_grad for table in frozen.engram_state.tables.values()))
        bad_initial = deepcopy(initial)
        bad_initial["base"]["source"]["model_config_sha256"] = "wrong"
        torch.save(bad_initial, path)
        with self.assertRaisesRegex(ValueError, "baseline identity"):
            self.policy(config)

    def test_actor_publication_and_reference_isolation_all_and_lora(self):
        async def exercise(config):
            learner = self.policy(config)
            baseline = learner.export_engram_baseline()
            reference = self.policy(config, role="reference", engram_weights=baseline)
            reference.eval().requires_grad_(False)
            tokens = torch.tensor([[1, 4, 7]])
            expected_reference = reference(tokens).detach().clone()
            identity = self.identity(config)
            def publish(version):
                return publish_policy_artifact(learner, version, self.folder / config.trainable_scope,
                    lineage="engram-test", identity=identity, sampling={"temperature": 1., "top_p": 1, "top_k": 0})
            initial = publish(0)
            backend = PyTorchRLBackend(self.folder, self.model_config, config, False,
                engram_weights=baseline, initial_policy=initial)
            await backend.start()
            try:
                before = backend.model(tokens).detach().clone()
                optimizer = self.optimizer(learner)
                for step in (1, 2):
                    group = generate_group(backend.model, PromptRecord([1, 4, 7]), 3, 3, 1., None,
                        0, step - 1, generator=torch.Generator().manual_seed(20 + step),
                        reward_fn=lambda record, ids: sum(ids))
                    metrics = train_groups(learner, reference, optimizer, [group], config)
                    self.assertGreater(metrics["nonzero_advantages"], 0)
                    publish(step)
                    torch.testing.assert_close(backend.model(tokens), before, rtol=0, atol=0)
                    torch.testing.assert_close(reference(tokens), expected_reference, rtol=0, atol=0)
                artifact = publish(2)
                self.assertEqual(backend.activate(artifact), {key: artifact[key] for key in ("version", "digest", "lineage")})
                torch.testing.assert_close(backend.model(tokens), learner(tokens), rtol=0, atol=0)
                self.assertFalse(torch.equal(backend.model(tokens), before))
                with self.assertRaisesRegex(ValueError, "roll back"):
                    backend.activate(initial)
                self.assertTrue(all(not table.weight.requires_grad for table in backend.model.engram_state.tables.values()))
            finally:
                await backend.close()
        for scope in ("all", "lora"):
            with self.subTest(scope=scope):
                asyncio.run(exercise(replace(self.config, trainable_scope=scope, lora_rank=2)))

    def test_native_manager_twenty_steps_fresh_resume_and_sampling(self):
        config = replace(self.config, max_steps=20,
                         output_dir=str(self.folder / "run"), log_samples_per_step=0)
        config_file = self.folder / "rl.json"
        config_file.write_text(json.dumps(config.to_dict()))
        command = [sys.executable, "-m", "dsv41_train.rl.train", "--model-dir", str(self.folder),
                   "--data", str(self.data), "--config", str(config_file)]
        def run(extra):
            result = subprocess.run(command + extra, cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(ROOT), "OMP_NUM_THREADS": "1"},
                text=True, capture_output=True, timeout=300)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result
        result = run([])
        logs = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
        steps = [item for item in logs if item.get("event") == "train"]
        self.assertEqual(len(steps), 20)
        second = Path(config.output_dir) / "step-00000020.pt"
        state20 = torch.load(second, weights_only=True)
        delta20 = state20["policy"]["engram_delta"]
        self.assertTrue(all(record["indices"].numel() > 0 for record in delta20["tables"].values()))
        for name, record in delta20["tables"].items():
            self.assertFalse(torch.equal(record["values"], self.base.model.engram_tables[name].weight.detach()[record["indices"]]))
        run(["--resume", str(second), "--max-steps", "21", "--engram-cache-mib", "0"])
        third = Path(config.output_dir) / "step-00000021.pt"
        state21 = torch.load(third, weights_only=True)
        self.assertGreater(state21["next_group_id"], state20["next_group_id"])
        self.assertEqual(state21["identity"], state20["identity"])
        self.assertTrue(any(not torch.equal(record["values"], state21["policy"]["engram_delta"]["tables"][name]["values"])
            for name, record in delta20["tables"].items()))
        restored, _, version, _ = load_checkpoint_model(model_dir=self.folder, checkpoint=third)
        self.assertEqual(version, 21)
        for name, record in state21["policy"]["engram_delta"]["tables"].items():
            torch.testing.assert_close(restored.engram_state.tables[name].weight[record["indices"]], record["values"], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for warm GPU row cache")
    def test_warm_gpu_cache_update_restore_and_cross_stream(self):
        baseline = self.policy().export_engram_baseline()
        learner = EngramState(baseline, "learner", True)
        root = nn.ModuleDict(learner.tables).cuda()
        for table in root.values():
            table.configure_cache(2**20)
        optimizer = torch.optim.SGD(root.parameters(), lr=.1, foreach=False)
        install_engram_sgd_hooks(root, optimizer)
        indices = torch.tensor([1, 3, 1], device="cuda")
        original = learner.state_dict()
        for table in root.values():
            table(indices).sum().backward()
        learner.mark_touched()
        optimizer.step()
        other = torch.cuda.Stream()
        with torch.cuda.stream(other):
            for table in root.values():
                torch.testing.assert_close(table(indices), table.weight.detach()[indices.cpu()].cuda(), rtol=0, atol=0)
        learner.load_state_dict(original)
        with torch.cuda.stream(other):
            for name, table in root.items():
                torch.testing.assert_close(table(indices), baseline.view(name)[indices.cpu()].cuda(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
