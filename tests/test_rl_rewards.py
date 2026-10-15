import unittest
from decimal import Decimal

from dsv41_train.rl.data import PromptRecord
from dsv41_train.rl.rewards import build_text_reward


class TextDecoder:
    def decode(self, ids, *, skip_special_tokens):
        return "".join(chr(token) for token in ids if not skip_special_tokens or token != 0)


class TextRewardTests(unittest.TestCase):
    def score(self, spec, answer, response):
        reward = build_text_reward(spec, TextDecoder())
        record = PromptRecord([1], metadata={"answer": answer})
        result = reward(record, [ord(character) for character in response] + [0])
        self.assertIsInstance(result, float)
        return result

    def test_text_normalizes_unicode_and_whitespace_but_preserves_case(self):
        cases = [
            ("  Cafe\u0301\n\tParis  ", "Caf\u00e9 Paris", 1.0),
            ("ABC 12", "\uff21\uff22\uff23\u00a0\uff11\uff12", 1.0),
            ("Paris", "paris", 0.0),
            ("Paris", "The answer is Paris", 0.0),
            ("Paris", "Paris.", 0.0),
        ]
        for answer, response, expected in cases:
            with self.subTest(answer=answer, response=response):
                self.assertEqual(self.score("text_exact_match", answer, response), expected)

    def test_text_extracts_only_a_terminal_answer_tag(self):
        cases = [
            ("Reasoning about Lyon. <answer>Paris</answer> \n", 1.0),
            ("<answer>Lyon</answer> Then <answer>Paris</answer>", 1.0),
            ("<answer>Paris</answer> but maybe Lyon", 0.0),
            ("<answer>Paris", 0.0),
            ("<answer><answer>Paris</answer>", 0.0),
            ("Reasoning #### Paris", 0.0),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(self.score("text_exact_match", "Paris", response), expected)

    def test_numeric_compares_exact_scalars(self):
        cases = [
            (0.5, "1/2", 1.0),
            ("1/2", "50%", 1.0),
            ("50%", "5e-1", 1.0),
            (1234.5, "1,234.500", 1.0),
            ("-0.125", "1/-8", 1.0),
            ("0.6", "1.2 / 2", 1.0),
            ("Calculation 12. #### 0.5", "1/2", 1.0),
            ("Calculation 12. \\boxed{0.5}", "1/2", 1.0),
            ("0", "-0.0", 1.0),
            ("1/3", "0.3333333333333333", 0.0),
            ("1e-40", "0", 0.0),
            (2, "3", 0.0),
        ]
        for answer, response, expected in cases:
            with self.subTest(answer=answer, response=response):
                self.assertEqual(self.score("numeric_answer", answer, response), expected)

    def test_numeric_extracts_explicit_final_answers(self):
        cases = [
            ("Earlier estimate 12. <answer>0.5</answer>", 1.0),
            ("Earlier estimate 12. \\boxed{1/2}", 1.0),
            ("Earlier estimate 12. \\boxed{\\frac{1}{2}}", 1.0),
            ("Earlier estimate 12. #### 50%", 1.0),
            ("<answer>\\boxed{0.5}</answer>", 1.0),
            ("Earlier estimate 12. The answer is 0.5", 0.0),
            ("\\boxed{0.5} but maybe 3", 0.0),
            ("\\boxed{0.5", 0.0),
            ("<answer>0.5</answer> trailing", 0.0),
        ]
        for response, expected in cases:
            with self.subTest(response=response):
                self.assertEqual(self.score("numeric_answer", 0.5, response), expected)

    def test_invalid_numeric_predictions_score_zero(self):
        for response in [
            "", "NaN", "Infinity", "1/0", "2 + 2", "4 units", "4 or 5",
            "1,23", "4%%", "4_0", "1e1000000", "0e-1000000", "4" * 2000,
            "__import__('os').system('true')",
        ]:
            with self.subTest(response=response):
                self.assertEqual(self.score("numeric_answer", 4, response), 0.0)

    def test_json_matches_structure_and_accepts_one_complete_fence(self):
        answer = {"values": [1, True, None, "x"], "name": "unit"}
        for response in [
            '{"name":"unit","values":[1.0,true,null,"x"]}',
            ' \n```json\n{"values": [1, true, null, "x"], "name": "unit"}\n```\n',
            '```\n{"values": [1, true, null, "x"], "name": "unit"}\n```',
        ]:
            with self.subTest(response=response):
                self.assertEqual(self.score("json_match", answer, response), 1.0)
        self.assertEqual(self.score("json_match", None, "null"), 1.0)
        self.assertEqual(self.score("json_match", "1", '"1"'), 1.0)
        self.assertEqual(self.score("json_match", 0.1, "0.1"), 1.0)
        self.assertEqual(self.score("json_match", 10**400, "1e400"), 1.0)

    def test_json_rejects_loose_matches_and_boolean_number_aliasing(self):
        cases = [
            (True, "1"),
            ({"x": True}, '{"x": 1}'),
            ([1], "[true]"),
            ("1", "1"),
            (0, "1e-400"),
            (1, "1.0000000000000000000000000001"),
            ([1, 2], "[2, 1]"),
            ({"x": 1}, '{"x": 1, "extra": 2}'),
            ({"x": 1}, 'Here is the answer: {"x": 1}'),
            ({"x": 1}, '<answer>{"x": 1}</answer>'),
            ({"x": 1}, '```json\n{"x": 1}\n``` extra'),
            ({"x": 1}, '```json\n{"x": 1}\n```\n```json\n{}\n```'),
            ({"x": 1}, '{"x": 0, "x": 1}'),
            ({"x": 1}, '{"x": 1} {}'),
            ({"x": 1}, '{"x": NaN}'),
            ({"x": 1}, '{"x": Infinity}'),
            ({"x": 1}, '{"x": 1e9999}'),
            ({"x": 1}, '{"x": 1e99999999999999999999999999999999999}'),
            ({"x": 1}, '{"x": 1,}'),
        ]
        for answer, response in cases:
            with self.subTest(answer=answer, response=response):
                self.assertEqual(self.score("json_match", answer, response), 0.0)

    def test_missing_or_invalid_ground_truth_is_a_data_error(self):
        for spec in ["text_exact_match", "numeric_answer", "json_match"]:
            with self.subTest(spec=spec):
                reward = build_text_reward(spec, TextDecoder())
                with self.assertRaisesRegex(ValueError, "answer"):
                    reward(PromptRecord([1]), [])
        invalid = {
            "text_exact_match": [None, 1, [], {}, "", " \n\t"],
            "numeric_answer": [None, True, [], {}, "", "NaN", float("inf"), "1/0", "2 + 2", "1e1000000"],
            "json_match": [float("nan"), float("inf"), {"x": float("inf")}, {1: "x"}, (1, 2), {1}, Decimal("1")],
        }
        for spec, answers in invalid.items():
            for answer in answers:
                with self.subTest(spec=spec, answer=answer):
                    with self.assertRaisesRegex(ValueError, "answer"):
                        self.score(spec, answer, "invalid prediction")

    def test_reward_configuration_errors_fail_at_construction(self):
        with self.assertRaisesRegex(ValueError, "reward"):
            build_text_reward("unknown", TextDecoder())
        with self.assertRaisesRegex(ValueError, "tokenizer"):
            build_text_reward("text_exact_match", None)

    def test_undecodable_outputs_score_zero_after_reference_validation(self):
        class RejectingDecoder(TextDecoder):
            def decode(self, ids, *, skip_special_tokens):
                if 99 in ids:
                    raise ValueError("token ID 99 is outside the tokenizer vocabulary")
                return super().decode(ids, skip_special_tokens=skip_special_tokens)

        for spec, answer in [("text_exact_match", "1"), ("numeric_answer", 1), ("json_match", 1)]:
            with self.subTest(spec=spec):
                reward = build_text_reward(spec, RejectingDecoder())
                record = PromptRecord([1], metadata={"answer": answer})
                self.assertEqual(reward(record, [49]), 1.0)
                self.assertEqual(reward(record, [99]), 0.0)
                for invalid in [PromptRecord([1]), PromptRecord([1], metadata={"answer": float("nan")})]:
                    with self.assertRaisesRegex(ValueError, "answer"):
                        reward(invalid, [99])


if __name__ == "__main__":
    unittest.main()
