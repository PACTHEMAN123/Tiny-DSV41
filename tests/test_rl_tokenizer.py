import json
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, processors

from dsv41_train.rl.tokenizer import ModelTokenizer, read_eos_token_ids


class ModelTokenizerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        tokens = ["[UNK]", "[BOS]", "[EOS]", "[STOP]", "system", "user", "assistant", "hello", "world", "policy", "reply", '"', "你好"]
        tokenizer = Tokenizer(models.WordLevel(dict(zip(tokens, range(len(tokens)))), unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer.add_special_tokens(tokens[:4])
        tokenizer.post_processor = processors.TemplateProcessing(
            single="[BOS] $A", special_tokens=[("[BOS]", 1)]
        )
        tokenizer.save(str(self.directory / "tokenizer.json"))
        self.write_json("config.json", {"vocab_size": 16, "bos_token_id": 1, "eos_token_id": 2})
        self.write_json("tokenizer_config.json", {"bos_token": "[BOS]", "eos_token": {"__type": "AddedToken", "content": "[EOS]"}})

    def write_json(self, name, value):
        if name == "tokenizer_config.json" and isinstance(value, dict):
            value = {"tokenizer_class": "PreTrainedTokenizerFast", **value}
        (self.directory / name).write_text(json.dumps(value), encoding="utf-8")

    def write_template(self, template=None):
        if template is None:
            template = "{{ bos_token }} {% for message in messages %}{{ message.role }} {{ message.content }} {{ eos_token }} {% endfor %}{% if add_generation_prompt %}assistant{% endif %}"
        (self.directory / "chat_template.jinja").write_text(template, encoding="utf-8")

    def test_chat_template_formats_roles_and_appends_generation_prefix_once(self):
        self.write_template()
        tokenizer = ModelTokenizer(self.directory)
        self.assertEqual(tokenizer.encode_prompt(messages=[
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "world"},
        ]), [1, 4, 9, 2, 5, 7, 2, 6, 10, 2, 5, 8, 2, 6])
        self.assertEqual(tokenizer.encode_prompt(prompt="hello world"), [1, 5, 7, 8, 2, 6])

    def test_plain_prompt_without_template_uses_tokenizer_postprocessor(self):
        tokenizer = ModelTokenizer(self.directory)
        self.assertEqual(tokenizer.encode_prompt(prompt="hello world"), [1, 7, 8])
        with self.assertRaisesRegex(ValueError, "chat template"):
            tokenizer.encode_prompt(messages=[{"role": "user", "content": "hello"}])

    def test_model_tokenizer_honors_bos_and_eos_switches(self):
        for add_bos in (False, True):
            for add_eos in (False, True):
                with self.subTest(add_bos=add_bos, add_eos=add_eos):
                    self.write_json("tokenizer_config.json", {
                        "tokenizer_class": "LlamaTokenizerFast", "legacy": True,
                        "unk_token": "[UNK]", "bos_token": "[BOS]", "eos_token": "[EOS]",
                        "add_bos_token": add_bos, "add_eos_token": add_eos,
                    })
                    expected = ([1] if add_bos else []) + [7, 8] + ([2] if add_eos else [])
                    self.assertEqual(ModelTokenizer(self.directory).encode_prompt(prompt="hello world"), expected)

    def test_chat_template_does_not_duplicate_automatic_bos_and_eos(self):
        self.write_json("tokenizer_config.json", {
            "tokenizer_class": "LlamaTokenizerFast", "legacy": True,
            "unk_token": "[UNK]", "bos_token": "[BOS]", "eos_token": "[EOS]",
            "add_bos_token": True, "add_eos_token": True,
        })
        self.write_template()
        self.assertEqual(ModelTokenizer(self.directory).encode_prompt(prompt="hello world"), [1, 5, 7, 8, 2, 6])

    def test_saved_padding_and_truncation_do_not_change_prompt_tokens(self):
        path = self.directory / "tokenizer.json"
        saved = Tokenizer.from_file(str(path))
        saved.enable_truncation(max_length=2)
        saved.enable_padding(length=8, pad_id=0, pad_token="[UNK]")
        saved.save(str(path))
        self.assertEqual(ModelTokenizer(self.directory).encode_prompt(prompt="hello world"), [1, 7, 8])
        self.write_template()
        self.assertEqual(ModelTokenizer(self.directory).encode_prompt(prompt="hello world"), [1, 5, 7, 8, 2, 6])

    def test_template_file_takes_precedence_over_configured_template(self):
        self.write_json("tokenizer_config.json", {"chat_template": "world"})
        self.write_template("hello")
        self.assertEqual(ModelTokenizer(self.directory).encode_prompt(prompt="world"), [7])

    def test_default_named_templates_and_plain_template_are_supported(self):
        for template in ["hello", {"default": "hello", "tool_use": "world"}, [{"name": "default", "template": "hello"}, {"name": "tool_use", "template": "world"}]]:
            with self.subTest(template=template):
                self.write_json("tokenizer_config.json", {"chat_template": template})
                self.assertEqual(ModelTokenizer(self.directory).encode_prompt(prompt="world"), [7])

    def test_ambiguous_or_malformed_templates_are_rejected(self):
        for template in [{"first": "hello", "second": "world"}, [{"name": "other", "template": "hello"}], [{"name": "default", "template": "hello"}, {"name": "default", "template": "world"}], 4, ""]:
            with self.subTest(template=template):
                self.write_json("tokenizer_config.json", {"chat_template": template})
                with self.assertRaisesRegex(ValueError, "chat template"):
                    ModelTokenizer(self.directory)

    def test_common_template_helpers_and_generation_blocks(self):
        self.write_template('{% for message in messages %}{% if message.role == "system" %}{% continue %}{% endif %}{% generation %}{{ message.content | tojson }}{% endgeneration %}{% endfor %} {{ strftime_now("world") }}')
        self.assertEqual(ModelTokenizer(self.directory).encode_prompt(messages=[{"role": "system", "content": "policy"}, {"role": "user", "content": ' 你好 '}]), [11, 12, 11, 8])
        self.write_template('{{ raise_exception("unsupported conversation") }}')
        with self.assertRaisesRegex(ValueError, "unsupported conversation"):
            ModelTokenizer(self.directory).encode_prompt(prompt="hello")

    def test_template_sandbox_rejects_python_attribute_access(self):
        self.write_template('{{ messages.__class__.__mro__ }}')
        with self.assertRaisesRegex(ValueError, "chat template"):
            ModelTokenizer(self.directory).encode_prompt(prompt="hello")

    def test_special_token_map_and_model_ids_supply_template_context(self):
        self.write_json("tokenizer_config.json", {})
        self.write_json("special_tokens_map.json", {"eos_token": {"content": "[STOP]"}})
        self.write_template()
        self.assertEqual(ModelTokenizer(self.directory).encode_prompt(prompt="hello"), [1, 5, 7, 3, 6])

    def test_eos_precedence_scalar_multiple_and_nested_model_config(self):
        self.assertEqual(ModelTokenizer(self.directory).eos_token_ids, (2,))
        self.write_json("generation_config.json", {"eos_token_id": [3, 2, 3]})
        self.assertEqual(ModelTokenizer(self.directory).eos_token_ids, (3, 2))
        self.write_json("generation_config.json", {"eos_token_id": 3})
        self.assertEqual(ModelTokenizer(self.directory).eos_token_ids, (3,))
        self.write_json("generation_config.json", {})
        self.write_json("config.json", {"text_config": {"vocab_size": 16, "eos_token_id": [2, 3]}})
        self.assertEqual(ModelTokenizer(self.directory).eos_token_ids, (2, 3))
        self.write_json("config.json", {"vocab_size": 16})
        self.assertEqual(ModelTokenizer(self.directory).eos_token_ids, (2,))
        self.write_json("tokenizer_config.json", {})
        self.assertEqual(ModelTokenizer(self.directory).eos_token_ids, ())

    def test_decode_skips_special_tokens_by_default(self):
        tokenizer = ModelTokenizer(self.directory)
        self.assertEqual(tokenizer.decode([1, 7, 8, 2]), "hello world")
        self.assertEqual(tokenizer.decode([1, 7, 8, 2], skip_special_tokens=False), "[BOS] hello world [EOS]")

    def test_decode_rejects_model_padding_ids_without_tokenizer_text(self):
        with self.assertRaisesRegex(ValueError, "outside the tokenizer vocabulary: 13"):
            ModelTokenizer(self.directory).decode([7, 13, 8])

    def test_pretokenized_decode_does_not_compile_unused_chat_template(self):
        self.write_template("{% unsupported_tag %}")
        tokenizer = ModelTokenizer(self.directory)
        self.assertEqual(tokenizer.decode([7, 2]), "hello")
        with self.assertRaisesRegex(ValueError, "chat template"):
            tokenizer.encode_prompt(prompt="hello")

    def test_eos_reader_works_without_tokenizer_and_uses_explicit_fallback(self):
        (self.directory / "tokenizer.json").unlink()
        self.assertEqual(read_eos_token_ids(self.directory, fallback=3), (2,))
        self.write_json("generation_config.json", {"eos_token_id": [3, 2]})
        self.assertEqual(read_eos_token_ids(self.directory, fallback=1), (3, 2))
        self.write_json("generation_config.json", {"eos_token_id": None})
        self.write_json("config.json", {})
        self.assertEqual(read_eos_token_ids(self.directory, fallback=(2, 3)), (2, 3))
        self.assertEqual(read_eos_token_ids(self.directory), ())

    def test_invalid_eos_and_vocabulary_metadata_are_rejected(self):
        for eos in [True, -1, 1.5, "2", [], [2, False], 13, 16]:
            with self.subTest(eos=eos):
                self.write_json("generation_config.json", {"eos_token_id": eos})
                with self.assertRaisesRegex(ValueError, "eos_token_id"):
                    ModelTokenizer(self.directory)
        self.write_json("generation_config.json", {})
        self.write_json("config.json", {"vocab_size": 8, "eos_token_id": 2})
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            ModelTokenizer(self.directory)
        self.write_json("config.json", {"vocab_size": 16})
        self.write_json("tokenizer_config.json", {"eos_token": "[MISSING]"})
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            ModelTokenizer(self.directory)

    def test_messages_reject_tools_multimodal_and_missing_final_user(self):
        self.write_template()
        tokenizer = ModelTokenizer(self.directory)
        for messages in [[], [{"role": "tool", "content": "hello"}], [{"role": "user", "content": [{"type": "text", "text": "hello"}]}], [{"role": "user", "content": "hello", "tool_calls": []}], [{"role": "assistant", "content": "hello"}], [{"role": "user", "content": ""}], ["hello"]]:
            with self.subTest(messages=messages):
                with self.assertRaises(ValueError):
                    tokenizer.encode_prompt(messages=messages)
        for kwargs in [{}, {"prompt": "hello", "messages": [{"role": "user", "content": "world"}]}, {"prompt": 123}, {"prompt": ""}]:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    tokenizer.encode_prompt(**kwargs)

    def test_missing_tokenizer_or_malformed_json_has_actionable_error(self):
        self.write_json("tokenizer_config.json", [])
        with self.assertRaisesRegex(ValueError, "tokenizer_config.json"):
            ModelTokenizer(self.directory)
        self.write_json("tokenizer_config.json", {})
        (self.directory / "tokenizer.json").unlink()
        with self.assertRaisesRegex(ValueError, "tokenizer.json"):
            ModelTokenizer(self.directory)


if __name__ == "__main__":
    unittest.main()
