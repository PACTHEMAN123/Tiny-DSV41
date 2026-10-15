import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers

from dsv41_train.rl.tokenizer import ModelTokenizer


TOOLS = [{"type": "function", "function": {
    "name": "shell", "description": "Run a shell command in the task environment.",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                   "required": ["cmd"], "additionalProperties": False},
}}]
INITIAL = [{"role": "system", "content": "Use shell tools to complete the task."},
           {"role": "user", "content": "Inspect the current directory."}]


def tool_history(arguments=' {"cmd": "printf \'你好\\n\'"} ', *, two_calls=False):
    calls = [{"id": "call_1", "type": "function", "function": {
        "name": "shell", "arguments": arguments}}]
    if two_calls:
        calls.append({"id": "call_2", "type": "function", "function": {
            "name": "shell", "arguments": {"cmd": "pwd"}}})
    messages = INITIAL + [{"role": "assistant", "content": None,
                           "reasoning_content": "Inspect first.", "tool_calls": calls},
                          {"role": "tool", "tool_call_id": "call_1", "content": "你好\n"}]
    if two_calls:
        messages.append({"role": "tool", "tool_call_id": "call_2", "content": "/app"})
    return messages


class ToolConversationTokenizerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
        native = Tokenizer(models.BPE(dict(zip(alphabet, range(len(alphabet)))), []))
        native.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        native.decoder = decoders.ByteLevel()
        native.save(str(self.directory / "tokenizer.json"))
        (self.directory / "tokenizer_config.json").write_text('{"tokenizer_class":"PreTrainedTokenizerFast"}')
        template = (
            "{% if tools %}<tools>{{ tools | tojson }}</tools>\n{% endif %}"
            "{% for message in messages %}<{{ message.role }}>"
            "{% if message.content is string %}{{ message.content }}{% endif %}"
            "{% for call in message.tool_calls | default([]) %}<tool_call>"
            "{{ call.function.name }}:"
            "{% if call.function.arguments is string %}{{ call.function.arguments }}"
            "{% else %}{{ call.function.arguments | tojson }}{% endif %}</tool_call>{% endfor %}"
            "</{{ message.role }}>\n{% endfor %}"
            "{% if add_generation_prompt %}<assistant>"
            "{% if enable_thinking is defined and not enable_thinking %}<think></think>{% endif %}{% endif %}"
        )
        (self.directory / "chat_template.jinja").write_text(template, encoding="utf-8")
        self.tokenizer = ModelTokenizer(self.directory)

    def text(self, messages, **kwargs):
        return self.tokenizer.decode(self.tokenizer.encode_conversation(messages, **kwargs),
                                     skip_special_tokens=False)

    def test_initial_schema_and_tool_results_preserve_template_tokens(self):
        messages = tool_history(two_calls=True)
        before = copy.deepcopy((messages, TOOLS))
        rendered = self.text(messages, tools=TOOLS, chat_template_kwargs={"enable_thinking": False})
        expected = (
            f"<tools>{json.dumps(TOOLS, ensure_ascii=False)}</tools>\n"
            "<system>Use shell tools to complete the task.</system>\n"
            "<user>Inspect the current directory.</user>\n"
            "<assistant><tool_call>shell: {\"cmd\": \"printf '你好\\n'\"} </tool_call>"
            "<tool_call>shell:{\"cmd\": \"pwd\"}</tool_call></assistant>\n"
            "<tool>你好\n</tool>\n<tool>/app</tool>\n<assistant><think></think>"
        )
        self.assertEqual(rendered, expected)
        self.assertEqual((messages, TOOLS), before)
        self.assertEqual(self.text(INITIAL, tools=TOOLS),
                         expected.split("<assistant><tool_call>")[0] + "<assistant>")

    def test_plain_api_keeps_its_original_tool_restrictions(self):
        self.assertEqual(self.tokenizer.encode_prompt(messages=INITIAL),
                         self.tokenizer.encode_conversation(INITIAL))
        with self.assertRaisesRegex(ValueError, "tools|tool"):
            self.tokenizer.encode_prompt(messages=tool_history())
        (self.directory / "chat_template.jinja").unlink()
        with self.assertRaisesRegex(ValueError, "chat template"):
            ModelTokenizer(self.directory).encode_conversation(INITIAL)

    def test_invalid_tool_envelopes_and_context_overrides_fail(self):
        for messages in [[], [{"role": "tool", "content": "result", "tool_call_id": "unknown"}],
                         [{"role": "user", "content": [{"type": "text", "text": "task"}]}],
                         [{"role": ["user"], "content": "task"}],
                         tool_history("not JSON"), tool_history("[]"),
                         tool_history({"cmd": float("nan")}), tool_history()[:-1],
                         tool_history() + [{"role": "tool", "tool_call_id": "call_1", "content": "again"}]]:
            with self.subTest(messages=messages), self.assertRaises(ValueError):
                self.tokenizer.encode_conversation(messages, tools=TOOLS)
        for options in [{"enable_thinking": 0}, {"messages": INITIAL}, {"add_generation_prompt": False},
                        {"tools": []}, {"unknown_option": True}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.tokenizer.encode_conversation(INITIAL, chat_template_kwargs=options)
        for tools in [{}, [{"type": "code_interpreter"}], TOOLS * 2,
                      [{"type": "function", "function": {"name": "shell", "parameters": []}}]]:
            with self.subTest(tools=tools), self.assertRaises(ValueError):
                self.tokenizer.encode_conversation(INITIAL, tools=tools)

    def test_tool_results_cannot_be_detached_from_their_calls(self):
        for mutate in (
            lambda messages: messages[-1].update(tool_call_id="other"),
            lambda messages: messages[-2]["tool_calls"][0]["function"].update(name="other"),
            lambda messages: messages[-2]["tool_calls"].append(copy.deepcopy(messages[-2]["tool_calls"][0])),
            lambda messages: messages.insert(-1, {"role": "user", "content": "skip result"}),
        ):
            messages = tool_history()
            mutate(messages)
            with self.assertRaises(ValueError):
                self.tokenizer.encode_conversation(messages, tools=TOOLS)


@unittest.skipUnless(os.environ.get("QWEN_TOOL_TOKENIZER_MODEL"), "requires a mounted Qwen checkpoint")
class MountedQwenToolTokenizerParityTests(unittest.TestCase):
    def test_exact_offline_huggingface_template_token_parity(self):
        # Compare the validated conversation API to the checkpoint tokenizer directly.
        from transformers import AutoTokenizer

        directory = os.environ["QWEN_TOOL_TOKENIZER_MODEL"]
        native = ModelTokenizer(directory)
        reference = AutoTokenizer.from_pretrained(directory, local_files_only=True,
                                                   trust_remote_code=False)
        for messages in [INITIAL, tool_history(), tool_history({"cmd": "pwd"}, two_calls=True),
                         tool_history() + [{"role": "assistant", "content": "Inspection complete."},
                                           {"role": "user", "content": "Now list hidden files."}]]:
            with self.subTest(messages=messages):
                actual = native.encode_conversation(messages, tools=TOOLS,
                                                     chat_template_kwargs={"enable_thinking": False})
                expected = reference.apply_chat_template(messages, tools=TOOLS, tokenize=True,
                                                          add_generation_prompt=True, enable_thinking=False)
                if hasattr(expected, "keys"):
                    expected = expected["input_ids"]
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
