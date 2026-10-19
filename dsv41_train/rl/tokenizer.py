"""Local checkpoint tokenization without executing custom tokenizer code."""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import TemplateError
from tokenizers import Tokenizer
from transformers import AutoTokenizer


def _read_json(directory: Path, name: str) -> dict:
    path = directory / name
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _text_config(config: dict) -> dict:
    value = config.get("text_config", {})
    if not isinstance(value, dict):
        raise ValueError("config.json text_config must be an object")
    return value


def _token_ids(value, name: str) -> tuple[int, ...]:
    values = value if isinstance(value, (list, tuple)) else [value]
    if not values or any(type(item) is not int or item < 0 for item in values):
        raise ValueError(f"{name} must be a nonnegative integer or a nonempty list of them")
    return tuple(dict.fromkeys(values))


def read_eos_token_ids(model_dir, fallback=None) -> tuple[int, ...]:
    """Read stop IDs without requiring a tokenizer; callers check model bounds."""
    directory = Path(model_dir)
    generation = _read_json(directory, "generation_config.json")
    config = _read_json(directory, "config.json")
    for source in (generation, config, _text_config(config)):
        if source.get("eos_token_id") is not None:
            return _token_ids(source["eos_token_id"], "eos_token_id")
    return () if fallback is None else _token_ids(fallback, "eos_token_id")


def _chat_template(directory: Path, config: dict) -> str | None:
    path = directory / "chat_template.jinja"
    if path.exists():
        template = path.read_text(encoding="utf-8")
    else:
        template = config.get("chat_template")
        if template is None:
            return None
        if isinstance(template, list):
            choices = {}
            for entry in template:
                if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or "template" not in entry or entry["name"] in choices:
                    raise ValueError("chat template list must contain unique named templates")
                choices[entry["name"]] = entry["template"]
            template = choices
        if isinstance(template, dict):
            if "default" not in template or any(not isinstance(value, str) for value in template.values()):
                raise ValueError("named chat templates require an unambiguous 'default' chat template")
            template = template["default"]
    if not isinstance(template, str) or not template.strip():
        raise ValueError("chat template must be a nonempty string")
    return template


def _json_copy(value, name):
    try:
        copied = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
        if copied != value:
            raise ValueError("only JSON values with string object keys are supported")
        return copied
    except (TypeError, ValueError, OverflowError, RecursionError) as error:
        raise ValueError(f"{name} must contain finite JSON values: {error}") from error



class ModelTokenizer:
    def __init__(self, model_dir):
        directory = Path(model_dir)
        path = directory / "tokenizer.json"
        if not path.is_file():
            raise ValueError(f"{path} is required; only local tokenizers tokenizer.json checkpoints are supported")
        try:
            vocabulary = Tokenizer.from_file(str(path)).get_vocab(with_added_tokens=True)
        except Exception as error:
            raise ValueError(f"unsupported or invalid tokenizer.json at {path}: {error}") from error
        config = _read_json(directory, "config.json")
        text_config = _text_config(config)
        tokenizer_config = _read_json(directory, "tokenizer_config.json")
        self._template_source = _chat_template(directory, tokenizer_config)
        tokenizer_kwargs = {}
        if tokenizer_config.get("tokenizer_class") in {"LlamaTokenizer", "LlamaTokenizerFast"}:
            for name in ("add_bos_token", "add_eos_token"):
                if name in tokenizer_config:
                    if type(tokenizer_config[name]) is not bool:
                        raise ValueError(f"{name} must be a boolean")
                    tokenizer_kwargs[name] = tokenizer_config[name]
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(directory.resolve()), local_files_only=True, trust_remote_code=False, use_fast=True,
                **tokenizer_kwargs,
            )
        except Exception as error:
            raise ValueError(f"cannot load local tokenizer at {directory}: {error}") from error
        # Metadata must not create token IDs without matching checkpoint embeddings.
        if self._tokenizer.get_vocab() != vocabulary:
            raise ValueError("tokenizer metadata changes the checkpoint vocabulary; all tokens must retain their IDs from tokenizer.json")
        self._vocabulary_ids = set(vocabulary.values())
        vocab_size = config.get("vocab_size", text_config.get("vocab_size"))
        if vocab_size is not None:
            if type(vocab_size) is not int or vocab_size <= 0:
                raise ValueError("model vocabulary size must be a positive integer")
            if any(token_id >= vocab_size for token_id in vocabulary.values()):
                raise ValueError("tokenizer vocabulary contains IDs outside the model vocabulary")
        fallback = tokenizer_config.get("eos_token_id")
        if self._tokenizer.eos_token_id is not None:
            fallback = self._tokenizer.eos_token_id
        self._eos_token_ids = read_eos_token_ids(directory, fallback=fallback)
        for token_id in self._eos_token_ids:
            self._validate_id(token_id, "eos_token_id")
        for name in ("bos_token", "eos_token", "pad_token", "unk_token"):
            if getattr(self._tokenizer, name) is not None:
                continue
            id_name = name + "_id"
            value = tokenizer_config.get(id_name, config.get(id_name, text_config.get(id_name)))
            if value is None and name == "eos_token" and self._eos_token_ids:
                value = self._eos_token_ids
            if value is not None:
                token_id = _token_ids(value, id_name)[0]
                self._validate_id(token_id, id_name)
                setattr(self._tokenizer, name, self._tokenizer.convert_ids_to_tokens(token_id))

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        return self._eos_token_ids

    def _validate_id(self, token_id, name):
        if type(token_id) is not int or token_id not in self._vocabulary_ids:
            raise ValueError(f"{name} contains an ID outside the tokenizer vocabulary: {token_id!r}")

    def encode_prompt(self, prompt: str | None = None, messages: list[dict] | None = None) -> list[int]:
        if (prompt is None) == (messages is None):
            raise ValueError("provide exactly one of prompt or messages")
        if prompt is not None:
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("prompt must be a nonempty string")
            if self._template_source is None:
                # Prompt length validation and batch padding belong to the RL runtime.
                return self._tokenizer.encode(prompt, add_special_tokens=True, padding=False, truncation=False)
            messages = [{"role": "user", "content": prompt}]
        if self._template_source is None:
            raise ValueError("messages require a checkpoint chat template")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        for message in messages:
            if not isinstance(message, dict) or set(message) != {"role", "content"}:
                raise ValueError("messages support only role and content; tools and other message fields are unsupported")
            if message["role"] not in ("system", "user", "assistant"):
                raise ValueError("message role must be system, user, or assistant; tool messages are unsupported")
            if not isinstance(message["content"], str) or not message["content"].strip():
                raise ValueError("message content must be nonempty text; multimodal content is unsupported")
        if messages[-1]["role"] != "user":
            raise ValueError("messages must end with a user message for generation")
        return self._encode_template(messages)

    def encode_conversation(self, messages: list[dict], *, tools: list[dict] | None = None,
                            chat_template_kwargs: dict | None = None) -> list[int]:
        """Tokenize text/function-tool history with the checkpoint's own template.

        Tool argument strings are preserved exactly. Generated action token IDs
        must be retained separately for training rather than recovered from this
        conversation. Template options currently support ``enable_thinking``.
        """
        if self._template_source is None:
            raise ValueError("conversations require a checkpoint chat template")
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a nonempty list")
        options = {} if chat_template_kwargs is None else chat_template_kwargs
        if not isinstance(options, dict) or set(options) - {"enable_thinking"}:
            raise ValueError("chat_template_kwargs support only enable_thinking")
        if "enable_thinking" in options and type(options["enable_thinking"]) is not bool:
            raise ValueError("enable_thinking must be a boolean")
        messages = _json_copy(messages, "messages")
        tools = _json_copy(tools, "tools")
        tool_names = set()
        if tools is not None:
            if not isinstance(tools, list):
                raise ValueError("tools must be a list of function definitions")
            for tool in tools:
                if not isinstance(tool, dict) or set(tool) != {"type", "function"} or tool["type"] != "function":
                    raise ValueError("tools support only function definitions")
                function = tool["function"]
                if not isinstance(function, dict) or set(function) - {"name", "description", "parameters", "strict"}:
                    raise ValueError("unsupported function definition fields")
                name = function.get("name")
                if not isinstance(name, str) or not name.strip() or name in tool_names:
                    raise ValueError("function names must be nonempty and unique")
                if ("description" in function and not isinstance(function["description"], str)
                        or "parameters" in function and not isinstance(function["parameters"], dict)
                        or "strict" in function and type(function["strict"]) is not bool):
                    raise ValueError("function description, parameters, or strict has an invalid type")
                tool_names.add(name)
        pending = set()
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError("messages must contain objects")
            role = message.get("role")
            if not isinstance(role, str):
                raise ValueError("message role must be text")
            allowed = {"role", "content"}
            if role == "assistant":
                allowed.update({"tool_calls", "reasoning_content"})
            elif role == "tool":
                allowed.add("tool_call_id")
            elif role not in {"system", "user"}:
                raise ValueError("message role must be system, user, assistant, or tool")
            if set(message) - allowed:
                raise ValueError(f"unsupported {role} message fields")
            if role == "system" and index != 0:
                raise ValueError("system message must be first")
            content = message.get("content")
            if role == "assistant":
                if content is not None and not isinstance(content, str):
                    raise ValueError("assistant content must be text or null")
                if "reasoning_content" in message and not isinstance(message["reasoning_content"], str):
                    raise ValueError("assistant reasoning_content must be text")
            elif not isinstance(content, str) or role != "tool" and not content.strip():
                raise ValueError("message content must be text; system/user text must be nonempty")
            if role == "tool":
                identity = message.get("tool_call_id")
                if not isinstance(identity, str) or identity not in pending:
                    raise ValueError("tool_call_id must match an unanswered assistant tool call")
                pending.remove(identity)
                continue
            if pending:
                raise ValueError("assistant tool calls must receive results before the next message")
            if role != "assistant":
                continue
            calls = message.get("tool_calls", [])
            if not isinstance(calls, list):
                raise ValueError("assistant tool_calls must be a list")
            if not calls and (not isinstance(content, str) or not content.strip()):
                raise ValueError("assistant messages require content or tool calls")
            for call in calls:
                if not isinstance(call, dict) or set(call) != {"id", "type", "function"} or call["type"] != "function":
                    raise ValueError("assistant tool_calls support only identified function calls")
                identity, function = call["id"], call["function"]
                if not isinstance(identity, str) or not identity.strip() or identity in pending:
                    raise ValueError("assistant tool call IDs must be nonempty and unique within a turn")
                if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
                    raise ValueError("tool calls require function name and arguments")
                name = function["name"]
                if not isinstance(name, str) or not name.strip() or tools is not None and name not in tool_names:
                    raise ValueError("tool call must name an available function")
                arguments = function["arguments"]
                try:
                    parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
                    parsed = _json_copy(parsed, "tool arguments")
                except (ValueError, RecursionError) as error:
                    raise ValueError(f"invalid JSON tool arguments: {error}") from error
                if not isinstance(parsed, dict):
                    raise ValueError("tool arguments must be a JSON object or its JSON string")
                pending.add(identity)
        if pending or messages[-1]["role"] not in {"user", "tool"}:
            raise ValueError("conversations must end with a user message or completed tool results")
        return self._encode_template(messages, tools=tools, **options)


    def encode_completion(self, messages: list[dict], answer: str, *,
                          chat_template_kwargs: dict | None = None) -> tuple[list[int], list[int]]:
        """Return generation prompt and completed conversation IDs for text SFT.

        Both use the checkpoint template. A template or tokenization boundary
        that changes the prompt when the answer is appended is rejected rather
        than silently assigning incorrect response labels.
        """
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be a nonempty string")
        prompt_ids = self.encode_conversation(messages, chat_template_kwargs=chat_template_kwargs)
        completed = [*messages, {"role": "assistant", "content": answer}]
        full_ids = self._encode_template(
            completed, add_generation_prompt=False, **(chat_template_kwargs or {}),
        )
        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise ValueError("completed conversation does not preserve the generation prompt token prefix")
        return prompt_ids, full_ids

    def _encode_template(self, messages, *, add_generation_prompt=True, **template_kwargs):
        try:
            ids = self._tokenizer.apply_chat_template(
                messages, chat_template=self._template_source, tokenize=True,
                add_generation_prompt=add_generation_prompt, padding=False, truncation=False,
                return_dict=False, **template_kwargs,
            )
        except (TemplateError, TypeError, ValueError) as error:
            raise ValueError(f"cannot render checkpoint chat template: {error}") from error
        if not isinstance(ids, list) or not ids:
            raise ValueError("chat template must produce a nonempty flat list of token IDs")
        for token_id in ids:
            self._validate_id(token_id, "chat template token IDs")
        return ids

    def decode(self, ids, *, skip_special_tokens=True) -> str:
        ids = list(ids)
        for token_id in ids:
            self._validate_id(token_id, "decoded token IDs")
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
