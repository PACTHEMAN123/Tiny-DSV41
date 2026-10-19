"""CPU fixtures for ordered, version-bound 32-request held-out evaluation."""
import copy
import hashlib
import json
import types
import unittest
from unittest.mock import patch

from dsv41_train.rl.distributed_grpo import evaluate, sample_record


ACK = {"version": 0, "digest": "fixed", "lineage": "test"}


class Client:
    def __init__(self, corrupt=None):
        self.ack, self.calls, self.corrupt = dict(ACK), [], corrupt

    def sample_group(self, specs):
        self.calls.append(copy.deepcopy(specs))
        results = [{"context_token_ids": spec["prompt_ids"], "action_token_ids": [7, 2],
            "behavior_log_probs": [-.3, -.2], "logprob_convention": "sampled_distribution",
            "finish_reason": "eos", "stop_token_count": 1,
            "generation_metrics": {"policy_ack": dict(self.ack),
                "sampling_seed": spec["sampling_seed"], "request_id": spec["request_id"]}}
            for spec in specs]
        if self.corrupt is not None:
            self.corrupt(self, results)
        return results


class EvaluationBatchTests(unittest.TestCase):
    def run_evaluation(self, client, count=128):
        # Identical prompt IDs make seed/request binding, rather than prompt
        # comparison alone, necessary to detect a result-ordering mistake.
        records = [types.SimpleNamespace(prompt_ids=[0, 1],
            metadata={"id": f"question:{i}", "correct": i % 3 != 0}) for i in range(count)]
        args = types.SimpleNamespace(seed=42, max_new_tokens=20, eval_temperature=.01)
        tokenizer = types.SimpleNamespace(decode=lambda ids, **kwargs: "answer")
        scored = []
        def reward(record, content):
            self.assertEqual(content, [7])
            scored.append(record.metadata["id"])
            return float(record.metadata["correct"])
        with patch("dsv41_train.rl.distributed_grpo.formatting_correct", return_value=True):
            result = evaluate(client, records, reward, tokenizer, args, "test", "baseline")
        return result, scored

    def test_128_questions_use_four_ordered_batches_with_original_seed_schedule(self):
        client = Client()
        result, scored = self.run_evaluation(client)
        self.assertEqual([len(specs) for specs in client.calls], [32] * 4)
        specs = [spec for call in client.calls for spec in call]
        self.assertEqual([s["sampling_seed"] for s in specs], [42 + i * 32 for i in range(128)])
        self.assertEqual(len({s["request_id"] for s in specs}), 128)
        self.assertTrue(all(s["temperature"] == .01 for s in specs))
        expected = [f"question:{i}" for i in range(128)]
        self.assertEqual(scored, expected)
        digest = hashlib.sha256(json.dumps(expected, separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(result["ordered_ids_sha256"], digest)
        self.assertNotIn("ids", result)
        self.assertNotIn("correctness", result)
        self.assertEqual(result["n"], 128)
        self.assertEqual(result["correct"], 85)
        self.assertEqual(result["request_batch_sizes"], [32] * 4)
        self.assertEqual(result["policy_ack"], ACK)
        self.assertEqual(result["output_tokens"], 256)

    def test_incomplete_reordered_or_changed_policy_batch_never_reaches_rewards(self):
        def reordered(client, results):
            results[0], results[1] = results[1], results[0]
        def changed_policy(client, results):
            results[-1]["generation_metrics"]["policy_ack"]["version"] = 1
        def mutated_client(client, results):
            client.ack["version"] = 1
        for corrupt in (lambda client, results: results.pop(), reordered, changed_policy, mutated_client):
            with self.subTest(corrupt=corrupt), \
                    patch("dsv41_train.rl.distributed_grpo.score_rewards") as score:
                with self.assertRaises(ValueError):
                    self.run_evaluation(Client(corrupt))
                score.assert_not_called()

    def test_partial_last_batch_keeps_question_order(self):
        client = Client()
        result, scored = self.run_evaluation(client, count=33)
        self.assertEqual([len(call) for call in client.calls], [32, 1])
        self.assertEqual(result["request_batch_sizes"], [32, 1])
        self.assertEqual(result["n"], 33)
        self.assertEqual(scored[-1], "question:32")

    def test_training_sample_rejects_changed_request_order_or_seed(self):
        record = types.SimpleNamespace(prompt_ids=[0, 1])
        args = types.SimpleNamespace(seed=42, max_new_tokens=20, temperature=.8)
        for corrupt in (lambda client, rows: rows.reverse(),
                        lambda client, rows: rows[0]["generation_metrics"].update(sampling_seed=99)):
            with self.subTest(corrupt=corrupt), self.assertRaisesRegex(ValueError, "sampling identity"):
                sample_record(Client(corrupt), record, count=32, phase="train", index=0,
                              args=args, lineage="test")


if __name__ == "__main__":
    unittest.main()
