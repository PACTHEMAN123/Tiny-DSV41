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
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(directory.resolve()), local_files_only=True, trust_remote_code=False, use_fast=True,
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
        try:
            ids = self._tokenizer.apply_chat_template(
                messages, chat_template=self._template_source, tokenize=True,
                add_generation_prompt=True, padding=False, truncation=False,
            )
        except (TemplateError, TypeError, ValueError) as error:
            raise ValueError(f"cannot render checkpoint chat template: {error}") from error
        if not ids:
            raise ValueError("chat template produced an empty tokenized prompt")
        return ids

    def decode(self, ids, *, skip_special_tokens=True) -> str:
        ids = list(ids)
        for token_id in ids:
            self._validate_id(token_id, "decoded token IDs")
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
