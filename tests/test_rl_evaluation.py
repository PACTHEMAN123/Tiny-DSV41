import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from test_rl_rollout import CausalModel
from dsv41_train.rl.config import RLConfig
from dsv41_train.rl.data import PromptRecord, build_reward


class EvaluationTests(unittest.TestCase):
    def test_unmapped_output_ids_remain_in_sample_logs(self):
        from dsv41_train.rl.evaluation import sample_record
        from dsv41_train.rl.rollout import Sample
        from dsv41_train.rl.tokenizer import ModelTokenizer

        tokenizer = ModelTokenizer(Path(__file__).resolve().parents[1] / "examples/rl/tiny-dsv41")
        record = PromptRecord([1, 4])
        sample = Sample(record.prompt_ids, [99, 2], [-1.0, -1.0], 0.0, 0, "eos", 1)
        row = sample_record(record, sample, step=0, group_id=0, tokenizer=tokenizer)
        self.assertEqual(row["response_ids"], [99, 2])
        self.assertIsNone(row["response_text"])
        self.assertIn("response_decode_error", row)
        self.assertIsInstance(row["prompt_text"], str)

    def test_legacy_checkpoint_keeps_original_eos_and_checks_recorded_identity(self):
        from dsv41_train.rl.checkpoint import (load_training_checkpoint, run_identity,
                                               save_training_checkpoint, validate_identity)
        from dsv41_train.rl.policy import DSV41Config
        from dsv41_train.rl.rollout import generation_options
        from dsv41_train.rl.runtime import make_model
        from dsv41_train.rl.sample import load_checkpoint_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            shutil.copytree(Path(__file__).resolve().parents[1] / "examples/rl/tiny-dsv41", base)
            (base / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 31]}))
            data = root / "train.jsonl"
            data.write_text('{"prompt_ids":[1,4],"target_ids":[2]}\n')
            identity = run_identity(base, data, True)
            legacy_identity = {key: value for key, value in identity.items()
                               if key not in {"identity_version", "text_assets_sha256"}}
            config = RLConfig()
            model_config = DSV41Config.from_model_dir(base)
            model = make_model(base, model_config, config, ("cpu",), True)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
            checkpoint = root / "legacy.pt"
            save_training_checkpoint(checkpoint, model, optimizer, step=1, config=config,
                                     identity=legacy_identity, next_group_id=1,
                                     rollout_rng_state=None, dropped_groups=0)
            legacy = torch.load(checkpoint, weights_only=True)
            legacy["config"].pop("eos_token_ids", None)
            torch.save(legacy, checkpoint)
            restored = load_training_checkpoint(checkpoint, model, optimizer, config, identity)
            self.assertEqual(tuple(restored["config"]["eos_token_ids"]), (2,))
            _, sampled_config, _, _ = load_checkpoint_model(model_dir=base, checkpoint=checkpoint)
            self.assertEqual(sampled_config.eos_token_ids, (2,))
            self.assertEqual(generation_options(base, model_config, config)["eos_token_id"], (2, 31))
            with self.assertRaisesRegex(ValueError, "eos_token_ids"):
                load_training_checkpoint(checkpoint, model, optimizer,
                                         RLConfig(eos_token_ids=(2, 31)), identity)
            (base / "chat_template.jinja").write_text("{{ messages }}")
            changed = run_identity(base, data, True)
            validate_identity(legacy_identity, changed)
            with self.assertRaisesRegex(ValueError, "identity"):
                validate_identity(identity, changed)
            data.write_text('{"prompt_ids":[1,5],"target_ids":[2]}\n')
            with self.assertRaisesRegex(ValueError, "identity"):
                validate_identity(legacy_identity, run_identity(base, data, True))

    def test_native_text_training_evaluates_resumes_and_samples(self):
        from dsv41_train.models.dsv4.model import DeepSeekV41ForCausalLM
        from dsv41_train.rl.policy import DSV41Config
        from dsv41_train.rl.sample import sample_checkpoint

        repository = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            shutil.copytree(repository / "examples/rl/tiny-dsv41", base)
            tokenizer_path = base / "tokenizer_config.json"
            tokenizer_config = json.loads(tokenizer_path.read_text())
            tokenizer_config.update(add_bos_token=True, add_eos_token=False)
            tokenizer_path.write_text(json.dumps(tokenizer_config))
            (base / "generation_config.json").write_text(json.dumps({"eos_token_id": [2, 31]}))
            torch.manual_seed(19)
            model = DeepSeekV41ForCausalLM(DSV41Config.from_model_dir(base))
            torch.save(model.state_dict(), base / "pytorch_model.bin")
            torch.save(model.model.hash.token_map, base / "engram_token_map.pt")
            training, validation = root / "train.jsonl", root / "validation.jsonl"
            training.write_text(json.dumps({"messages": [{"role": "user", "content": "What is 1 + 1 ?"}],
                                            "answer": 2}) + "\n")
            validation.write_text(json.dumps({"prompt": "What is 2 + 2 ?", "answer": 4}) + "\n")
            config = root / "rl.json"
            config.write_text(json.dumps({"reward": "numeric_answer", "max_steps": 2,
                                          "max_new_tokens": 4, "group_size": 2,
                                          "stop_token_sequences": [[28]], "truncated_reward": 0,
                                          "eval_interval": 1, "log_samples_per_step": 2,
                                          "output_dir": str(root / "run")}))
            command = [sys.executable, "-m", "dsv41_train.rl.train", "--model-dir", str(base),
                       "--data", str(training), "--eval-data", str(validation), "--config", str(config)]
            def run(arguments):
                result = subprocess.run(arguments, cwd=repository, capture_output=True, text=True,
                                        env={**os.environ, "PYTHONPATH": str(repository)}, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result

            result = run(command)
            events = [json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")]
            evaluations = [event for event in events if event["event"] == "eval"]
            self.assertEqual([event["step"] for event in evaluations], [0, 1, 2])
            self.assertEqual(evaluations[0]["phase"], "baseline")
            self.assertTrue(all(event["engram_shared"] and event["rollout_engram_shared"]
                                for event in events if event["event"] == "train"))
            rows = [json.loads(line) for line in (root / "run/train-samples.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all("response_text" in row and "prompt_text" in row for row in rows))
            checkpoint = root / "run/step-00000002.pt"
            state = torch.load(checkpoint, weights_only=True)
            self.assertEqual(tuple(state["config"]["eos_token_ids"]), (2, 31))
            run(command + ["--resume", str(checkpoint), "--max-steps", "3"])
            checkpoint = root / "run/step-00000003.pt"
            sampled = sample_checkpoint(model_dir=base, checkpoint=checkpoint, prompt="What is 1 + 1 ?")
            self.assertEqual(sampled["policy_version"], 3)
            self.assertIn("response_text", sampled)
            self.assertIn(sampled["finish_reason"], {"eos", "stop", "length"})
            run([sys.executable, "-m", "dsv41_train.rl.evaluate", "--model-dir", str(base),
                 "--checkpoint", str(checkpoint), "--data", str(validation), "--output-dir", str(root / "eval")])
            self.assertTrue((root / "eval/eval-metrics.jsonl").is_file())
            training.write_bytes(validation.read_bytes())
            conflict = subprocess.run(command, cwd=repository, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(conflict.returncode, 0)
            self.assertIn("validation prompts overlap", conflict.stderr)

    def test_fixed_evaluation_preserves_rng_weights_and_training_mode(self):
        from dsv41_train.rl.evaluation import evaluate_policy

        model = CausalModel()
        weights = model.transitions.detach().clone()
        rng = torch.get_rng_state().clone()
        records = [PromptRecord([1, 0], [2], {"answer": "two"})]
        config = RLConfig(eval_samples_per_prompt=3, max_new_tokens=3)
        with tempfile.TemporaryDirectory() as directory:
            options = dict(eos_token_id=(2,), stop_token_sequences=(), truncated_reward=0.0)
            first = evaluate_policy(model, records, config, reward_fn=build_reward("exact_match"),
                                    generation_kwargs=options, step=0, output_dir=directory)
            second = evaluate_policy(model, records, config, reward_fn=build_reward("exact_match"),
                                     generation_kwargs=options, step=1, output_dir=directory)
            self.assertEqual(first["reward_mean"], second["reward_mean"])
            self.assertEqual(first["truncation_fraction"], second["truncation_fraction"])
            rows = [json.loads(line) for line in (Path(directory) / "eval-samples.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual([r["response_ids"] for r in rows[:3]], [r["response_ids"] for r in rows[3:]])
            self.assertTrue(all(r["metadata"]["answer"] == "two" for r in rows))
            self.assertTrue(all(r["finish_reason"] in {"eos", "length"} for r in rows))
        torch.testing.assert_close(model.transitions, weights, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        self.assertTrue(model.training)
        self.assertIsNone(model.transitions.grad)

    def test_resume_rejects_changed_text_assets_and_validation_data(self):
        from dsv41_train.rl.checkpoint import run_identity

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            data, validation = root / "train.jsonl", root / "eval.jsonl"
            data.write_text('{"prompt_ids":[1]}\n')
            validation.write_text('{"prompt_ids":[2]}\n')
            template = root / "chat_template.jinja"
            template.write_text("{{ messages }}")
            first = run_identity(root, data, True, eval_data_path=validation)
            template.write_text("{{ messages }} assistant:")
            second = run_identity(root, data, True, eval_data_path=validation)
            self.assertNotEqual(first, second)
            validation.write_text('{"prompt_ids":[3]}\n')
            self.assertNotEqual(second, run_identity(root, data, True, eval_data_path=validation))
