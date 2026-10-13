import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

import torch

from test_dsv4_checkpoint import write_safetensors

from dsv41_train.rl.algorithm import response_log_probs
from dsv41_train.rl.checkpoint import load_policy, policy_state, run_identity
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.data import PromptRecord
from dsv41_train.rl.rollout import generate_group
from dsv41_train.rl.runtime import make_model
from dsv41_train.rl.sample import sample_checkpoint


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "examples/rl/tiny-dsv41"


def reference_scores(query, keys, weights, starts, ends, width, candidates=None):
    # Independent CPU oracle for the upstream Triton-only score operation.
    output = torch.full((*starts.shape, width), -torch.inf, device=query.device)
    for batch in range(query.shape[0]):
        for position in range(query.shape[1]):
            for offset in range(width):
                key = int(starts[batch, position]) + offset
                if key >= int(ends[batch, position]) or (candidates is not None and not candidates[batch, position, offset]):
                    continue
                dots = (query[batch, position].float() * keys[batch, key].float()).sum(-1)
                output[batch, position, offset] = (dots.relu() * weights[batch, position]).sum() / query.shape[-1] ** 0.5
    return output


class DSV41Tests(unittest.TestCase):
    def setUp(self):
        from dsv41_train.rl.policy import DSV41Config

        torch.set_num_threads(1)
        self.config = DSV41Config.from_model_dir(MODEL_DIR)
        self.rl = RLConfig(seed=37, max_new_tokens=3, temperature=0.8)

    def policy(self, dtype="float32"):
        return make_model(MODEL_DIR, self.config, RLConfig(seed=37, dtype=dtype), ("cpu",), True)

    def native_base(self, root):
        from dsv41_train.models.dsv4.model import DeepSeekV41ForCausalLM

        root.mkdir()
        (root / "config.json").write_bytes((MODEL_DIR / "config.json").read_bytes())
        torch.manual_seed(41)
        model = DeepSeekV41ForCausalLM(self.config)
        torch.save(model.state_dict(), root / "pytorch_model.bin")
        torch.save(model.model.hash.token_map, root / "engram_token_map.pt")
        return model

    def test_matches_upstream_logits_and_gradients_when_all_keys_fit(self):
        from dsv41_train.models.dsv4.model import DeepSeekV41ForCausalLM

        # Upstream topk deliberately has a different tie policy; compare all-key attention here.
        self.config.index_topk = 16
        self.config.candidate_topk_blocks = 16
        torch.manual_seed(37)
        upstream = DeepSeekV41ForCausalLM(self.config)
        policy = self.policy()
        tokens = torch.tensor([[1, 4, 7, 3, 8, 9, 5, 6]])
        expected = upstream(tokens).logits
        actual = policy(tokens)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        actual.square().mean().backward()
        expected.square().mean().backward()
        original = dict(upstream.named_parameters())
        for name, parameter in policy.backend.named_parameters():
            if parameter.requires_grad and original[name].grad is not None:
                torch.testing.assert_close(parameter.grad, original[name].grad, rtol=1e-4, atol=1e-6)

    def test_index_scores_match_independent_oracle_with_offsets_and_masks(self):
        from dsv41_train.models.dsv4.attention.sparse_indexer import index_scores

        torch.manual_seed(2)
        query = torch.randn(2, 3, 2, 4)
        keys = torch.randn(2, 5, 4)
        weights = torch.randn(2, 3, 2)
        starts = torch.tensor([[0, 1, 3], [0, 2, 4]])
        ends = torch.tensor([[0, 3, 5], [2, 5, 4]])
        candidates = torch.rand(2, 3, 5) > 0.3
        for mask in (None, candidates):
            expected = reference_scores(query, keys, weights, starts, ends, 5, mask)
            actual = index_scores(query, keys, weights, starts, ends, 5, mask)
            torch.testing.assert_close(actual, expected)
        self.assertEqual(index_scores(query, keys, weights, starts, ends, 0).shape, (2, 3, 0))

    def test_index_scores_fall_back_for_non_power_of_two_cuda_tiles(self):
        from dsv41_train.models.dsv4.attention import sparse_indexer

        starts, ends = torch.tensor([[0, 0]]), torch.tensor([[1, 3]])
        for heads, dimension in ((48, 16), (16, 48)):
            query = torch.randn(1, 2, heads, dimension)
            keys = torch.randn(1, 3, dimension)
            weights = torch.randn(1, 2, heads)
            expected = reference_scores(query, keys, weights, starts, ends, 3)
            # Simulate CUDA availability; still execute the real reference math.
            with patch.object(torch.Tensor, "is_cuda", new_callable=PropertyMock, return_value=True), \
                 patch.object(sparse_indexer, "triton", object()), \
                 patch.object(sparse_indexer, "fused_index_scores", side_effect=RuntimeError("invalid Triton tile")):
                actual = sparse_indexer.index_scores(query, keys, weights, starts, ends, 3)
            torch.testing.assert_close(actual, expected)

    def test_engram_and_hash_stay_frozen_across_dtype_moves_and_updates(self):
        policy = self.policy()
        tables = {name: table.weight.clone() for name, table in policy.backend.model.engram_tables.items()}
        hc = policy.backend.model.layers[0].attention_hc.fn
        with torch.no_grad():
            hc.fill_(0.123456789)
        expected_hc = hc.clone()
        frequencies = policy.backend.model.rotary.main.clone()
        policy.to(dtype=torch.bfloat16)
        policy.assert_engram_invariants()
        self.assertEqual(hc.dtype, torch.float32)
        torch.testing.assert_close(hc, expected_hc, rtol=0, atol=0)
        torch.testing.assert_close(policy.backend.model.rotary.main, frequencies, rtol=0, atol=0)
        self.assertTrue(all(t.weight.dtype == torch.float32 for t in policy.backend.model.engram_tables.values()))
        self.assertFalse(any("engram_tables" in name for name, _ in policy.named_parameters()))
        optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=0.01)
        tokens = torch.tensor([[1, 4, 7, 3, 8]])
        policy(tokens).float().square().mean().backward()
        self.assertIsNotNone(policy.backend.model.layers[1].engram.proj.weight.grad)
        optimizer.step()
        for name, table in policy.backend.model.engram_tables.items():
            torch.testing.assert_close(table.weight, tables[name], rtol=0, atol=0)
        self.assertTrue(all(b.device.type == "cpu" for b in policy.backend.model.hash.buffers()))

    def test_snapshot_restores_logits_without_copying_engram(self):
        policy = self.policy()
        tokens = torch.tensor([[1, 4, 7, 3, 8]])
        expected = policy(tokens).detach()
        state = policy_state(policy)
        self.assertFalse(any("engram_tables" in name for part in state.values() for name in part))
        self.assertTrue(any("indexer" in name for name in state["frozen"]))
        self.assertTrue(any("token_map" in name for name in state["frozen"]))
        with torch.no_grad():
            policy.backend.lm_head.weight.add_(0.5)
        load_policy(policy, state)
        torch.testing.assert_close(policy(tokens), expected, rtol=0, atol=0)

    def test_bfloat16_conversion_preserves_pretrained_routing_biases(self):
        policy = self.policy()
        router = policy.backend.model.layers[0].moe.router
        with torch.no_grad():
            router.weight.zero_()
            router.selection_bias.copy_(torch.tensor([1.0, 1.001, 1.002, 1.003]))
        hidden = torch.zeros(1, 1, self.config.hidden_size)
        expected_indices = router(hidden)[2]
        expected_bias = router.selection_bias.clone()
        policy.to(dtype=torch.bfloat16)
        self.assertEqual(router.selection_bias.dtype, torch.float32)
        torch.testing.assert_close(router.selection_bias, expected_bias, rtol=0, atol=0)
        torch.testing.assert_close(router(hidden.bfloat16())[2], expected_indices, rtol=0, atol=0)

    def test_rollout_probabilities_match_teacher_forcing(self):
        policy = self.policy()
        group = generate_group(
            policy, PromptRecord([1, 4, 7, 3, 8, 9]), group_size=2, max_new_tokens=4,
            temperature=0.8, eos_token_id=None, policy_version=3, group_id=0,
            generator=torch.Generator().manual_seed(13), reward_fn=lambda r, ids: 0.0,
        )
        for sample in group.samples:
            actual = response_log_probs(policy, sample.prompt_ids, sample.response_ids, 0.8)
            torch.testing.assert_close(actual, torch.tensor(sample.behavior_log_probs), rtol=1e-5, atol=1e-5)

    def test_tied_sparse_indexer_scores_remain_prefix_causal(self):
        policy = self.policy()
        with torch.no_grad():
            for layer in policy.backend.model.layers:
                if layer.attention.indexer is not None:
                    for parameter in layer.attention.indexer.parameters():
                        parameter.zero_()
        tokens = torch.tensor([[1, 4, 7, 3, 8, 9, 5, 6, 10, 11, 12, 13]])
        full = policy(tokens)
        for length in (5, 7, 9):
            torch.testing.assert_close(policy(tokens[:, :length]), full[:, :length], rtol=1e-5, atol=1e-5)

    def test_bfloat16_rollout_probability_error_is_bounded(self):
        policy = make_model(MODEL_DIR, self.config, RLConfig(seed=6, dtype="bfloat16"), ("cpu",), True)
        group = generate_group(
            policy, PromptRecord([1, 4, 7, 3, 8, 9]), group_size=1, max_new_tokens=12,
            temperature=0.8, eos_token_id=None, policy_version=0, group_id=0,
            generator=torch.Generator().manual_seed(13), reward_fn=lambda r, ids: 0.0,
        )
        sample = group.samples[0]
        actual = response_log_probs(policy, sample.prompt_ids, sample.response_ids, 0.8)
        # BF16 GEMMs may round differently for single-prefix and full-sequence shapes.
        torch.testing.assert_close(actual, torch.tensor(sample.behavior_log_probs), rtol=0, atol=0.002)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_fused_indexing_keeps_rollout_and_teacher_forcing_consistent(self):
        # The fused scorer requires a multiple of 16 heads; the tiny default has two.
        self.config.index_n_heads = 16
        policy = make_model(MODEL_DIR, self.config, self.rl, ("cuda:0",), True)
        for tied in (False, True):
            if tied:
                with torch.no_grad():
                    for layer in policy.backend.model.layers:
                        if layer.attention.indexer is not None:
                            for parameter in layer.attention.indexer.parameters():
                                parameter.zero_()
            group = generate_group(
                policy, PromptRecord([1, 4, 7, 3, 8, 9]), group_size=2, max_new_tokens=6,
                temperature=0.8, eos_token_id=None, policy_version=0, group_id=0,
                generator=torch.Generator().manual_seed(13), reward_fn=lambda r, ids: 0.0,
            )
            for sample in group.samples:
                actual = response_log_probs(policy, sample.prompt_ids, sample.response_ids, 0.8)
                torch.testing.assert_close(actual.cpu(), torch.tensor(sample.behavior_log_probs),
                                           rtol=1e-5, atol=1e-5)

    def test_native_checkpoint_requires_mapping_and_all_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "base"
            upstream = self.native_base(root)
            policy = make_model(root, self.config, self.rl, ("cpu",), False)
            tokens = torch.tensor([[1, 4, 7, 3]])
            expected = upstream(tokens).logits
            torch.testing.assert_close(policy(tokens), expected)
            (root / "engram_token_map.pt").unlink()
            with self.assertRaisesRegex((ValueError, FileNotFoundError), "token.*map|tokenizer"):
                make_model(root, self.config, self.rl, ("cpu",), False)
            torch.save(upstream.model.hash.token_map, root / "engram_token_map.pt")
            weights = upstream.state_dict()
            del weights["lm_head.weight"]
            torch.save(weights, root / "pytorch_model.bin")
            with self.assertRaisesRegex((ValueError, RuntimeError), "missing|Missing"):
                make_model(root, self.config, self.rl, ("cpu",), False)

    def test_config_preserves_outer_special_ids(self):
        from dsv41_train.rl.policy import DSV41Config

        values = json.loads((MODEL_DIR / "config.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps({
                "model_type": "deepseek_v41", "eos_token_id": 1, "bos_token_id": 0,
                "text_config": values,
            }))
            config = DSV41Config.from_model_dir(root)
            self.assertEqual((config.eos_token_id, config.bos_token_id), (1, 0))
        with self.assertRaisesRegex(ValueError, "one device"):
            make_model(MODEL_DIR, self.config, self.rl, ("cpu", "cpu"), True)

    def test_external_config_controls_random_and_pretrained_models(self):
        from dsv41_train.rl.policy import DSV41Config

        values = json.loads((MODEL_DIR / "config.json").read_text())
        values.update(vocab_size=48, hidden_size=272, num_hidden_layers=4,
                      num_attention_heads=4, q_lora_rank=24, moe_intermediate_size=48,
                      n_routed_experts=3, n_shared_experts=2, compress_ratios=[0, 2, 1, 1],
                      kv_source_layer_ids=[1, 2], index_source_layer_ids=[1, 2, 3],
                      candidate_source_layer_id=2, engram_layer_ids=[1, 3])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text(json.dumps(values))
            config = DSV41Config.from_model_dir(root)
            with self.assertRaisesRegex(FileNotFoundError, "checkpoint"):
                make_model(root, config, self.rl, ("cpu",), False)
            random = make_model(root, config, self.rl, ("cpu",), True)
            self.assertEqual(len(random.backend.model.layers), 4)
            self.assertEqual(random.backend.model.embedding.weight.shape, (48, 272))
            self.assertEqual(random.backend.model.layers[0].moe.routed.gate_up.shape[0], 3)
            self.assertEqual(random.backend.model.layers[0].moe.shared.gate.weight.shape, (96, 272))
            self.assertEqual(set(random.backend.model.engram_tables), {"1", "3"})
            tokens = torch.tensor([[1, 4, 7, 3]])
            expected = random(tokens)
            expected.square().mean().backward()
            self.assertTrue(torch.isfinite(random.backend.lm_head.weight.grad).all())
            torch.save(random.backend.state_dict(), root / "pytorch_model.bin")
            torch.save(random.backend.model.hash.source.token_map, root / "engram_token_map.pt")
            loaded = make_model(root, DSV41Config.from_model_dir(root), self.rl, ("cpu",), False)
            torch.testing.assert_close(loaded(tokens), expected)
            loaded.assert_engram_invariants()

    def test_native_sharded_safetensors_loads_all_tables_and_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "base"
            original = self.native_base(root)
            state = original.state_dict()
            names = sorted(state)
            shards = (names[::2], names[1::2])
            weight_map = {}
            for index, keys in enumerate(shards):
                filename = f"shard-{index}.safetensors"
                write_safetensors(root / filename, {name: state[name] for name in keys})
                weight_map.update({name: filename for name in keys})
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
            policy = make_model(root, self.config, self.rl, ("cpu",), False)
            for name, value in state.items():
                torch.testing.assert_close(policy.backend.state_dict()[name], value, rtol=0, atol=0)
            policy.assert_engram_invariants()
            (root / "model.safetensors.index.json").unlink()
            write_safetensors(root / "model.safetensors", state)
            single = make_model(root, self.config, self.rl, ("cpu",), False)
            for name, value in state.items():
                torch.testing.assert_close(single.backend.state_dict()[name], value, rtol=0, atol=0)

    def test_native_base_spawn_resume_and_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            upstream = self.native_base(root / "base")
            output = root / "run"
            values = {"max_steps": 2, "group_size": 3, "max_new_tokens": 3,
                      "kl_beta": 0.02, "reward": "toy_token_score", "max_policy_lag": 0,
                      "groups_per_step": 2, "queue_capacity": 1,
                      "dtype": "bfloat16", "output_dir": str(output), "worker_timeout_seconds": 60}
            config_file = root / "rl.json"
            config_file.write_text(json.dumps(values))
            command = [sys.executable, "-m", "dsv41_train.rl.train", "--model-dir", str(root / "base"),
                       "--data", str(ROOT / "examples/rl/prompts.jsonl"), "--config", str(config_file)]
            environment = {**os.environ, "PYTHONPATH": str(ROOT)}
            result = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
            self.assertNotEqual(events[0]["learner_pid"], events[0]["rollout_pid"])
            metrics = [event for event in events if event["event"] == "train"]
            self.assertEqual([event["step"] for event in metrics], [1, 2])
            self.assertTrue(all(event["policy_lag_max"] == 0 for event in metrics))
            self.assertTrue(all(event["samples"] == 6 for event in metrics))
            checkpoint = output / "step-00000002.pt"
            state = torch.load(checkpoint, weights_only=True, map_location="cpu")
            self.assertTrue(state["optimizer"]["state"])
            self.assertFalse(any("engram_tables" in name for part in state["policy"].values() for name in part))
            self.assertFalse(torch.equal(state["policy"]["trainable"]["backend.lm_head.weight"],
                                         upstream.lm_head.weight.to(torch.bfloat16)))
            result = subprocess.run(command + ["--resume", str(checkpoint), "--max-steps", "3"],
                                    cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            trained = output / "step-00000003.pt"
            response = sample_checkpoint(model_dir=root / "base", checkpoint=trained, prompt_ids=[1, 4, 7])
            self.assertEqual(response["policy_version"], 3)
            self.assertEqual(response["engram_device"], "cpu")
            self.assertTrue(response["response_ids"])
            before = run_identity(root / "base", config_file, False)
            torch.save(upstream.model.hash.token_map.roll(1), root / "base/engram_token_map.pt")
            self.assertNotEqual(run_identity(root / "base", config_file, False), before)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                sample_checkpoint(model_dir=root / "base", checkpoint=trained, prompt_ids=[1, 4, 7])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_cuda_backward_keeps_engram_on_cpu(self):
        policy = make_model(MODEL_DIR, self.config, self.rl, ("cuda:0",), True)
        policy(torch.tensor([[1, 4, 7, 3]], device="cuda:0")).square().mean().backward()
        policy.assert_engram_invariants()
        self.assertEqual(policy.backend.model.layers[1].engram.proj.weight.grad.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
