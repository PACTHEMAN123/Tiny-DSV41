"""Shared shell completion grammar and typed non-executable format failures."""
from __future__ import annotations

import hashlib
import json
import re


SHELL_PARAMETERS = {"type": "object", "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"], "additionalProperties": False}
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


FORMAT_ERROR_CODES = frozenset({"unexpected_thinking", "incomplete_tool_envelope",
    "invalid_tool_json", "unsupported_tool_schema", "invalid_shell_arguments", "invalid_shell_command"})


class ModelCompletionFormatError(ValueError):
    def __init__(self, message, code):
        if code not in FORMAT_ERROR_CODES:
            raise ValueError("unknown model format failure code")
        super().__init__(message)
        self.code = code


def invalid_tool_status(text, request_id, error):
    if not isinstance(error, ModelCompletionFormatError):
        raise TypeError("invalid-tool outcome requires a typed model grammar failure")
    return {"schema_version": 1, "outcome": "invalid_tool", "request_id": request_id,
        "error_code": error.code, "decoded_action_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "tools_executable": False}


def validate_model_completion_status(completion, request_id):
    """Validate the tagged outcome after ordinary raw training evidence checks."""
    status = completion.get("model_completion_status")
    if status is None:
        return None
    keys = {"schema_version", "outcome", "request_id", "error_code",
            "decoded_action_text_sha256", "tools_executable"}
    if (not isinstance(status, dict) or set(status) != keys or type(status["schema_version"]) is not int
            or status["schema_version"] != 1 or status["outcome"] != "invalid_tool"
            or status["request_id"] != request_id or not re.fullmatch(r"[a-f0-9]{32}", request_id)
            or status["error_code"] not in FORMAT_ERROR_CODES or status["tools_executable"] is not False):
        raise ValueError("invalid completed-model format status")
    choices = completion.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("invalid-tool outcome requires one completed choice")
    message = choices[0].get("message")
    if (not isinstance(message, dict) or set(message) != {"role", "content"}
            or message["role"] != "assistant" or not isinstance(message["content"], str)
            or choices[0].get("finish_reason") != "stop"
            or status["decoded_action_text_sha256"] != hashlib.sha256(message["content"].encode()).hexdigest()):
        raise ValueError("invalid-tool outcome must preserve raw text and forbid tool execution")
    return dict(status)


def _strict_object(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate generated JSON key")
            result[key] = value
        return result
    def invalid(_):
        raise ValueError("nonfinite generated JSON value")
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(value, dict):
        raise ValueError("generated tool call must be a JSON object")
    return value


def shell_completion(text, request_id, *, truncated=False):
    """Parse a complete Qwen shell envelope, without changing training tokens."""
    if truncated:
        # Partial commands are never executed, including complete calls that
        # happen to precede a token-budget truncation.
        return {"role": "assistant", "content": text}, "length"
    if "<think>" in text or "</think>" in text:
        raise ModelCompletionFormatError("this controller requires thinking-disabled completions", "unexpected_thinking")
    matches = list(_TOOL_CALL.finditer(text))
    content = _TOOL_CALL.sub("", text).strip()
    if "<tool_call>" in content or "</tool_call>" in content:
        raise ModelCompletionFormatError("expected complete serial shell calls", "incomplete_tool_envelope")
    message = {"role": "assistant", "content": content}
    if not matches:
        return message, "stop"
    calls = []
    for index, match in enumerate(matches):
        try:
            call = _strict_object(match.group(1))
        except ValueError as error:
            raise ModelCompletionFormatError(str(error), "invalid_tool_json") from error
        if set(call) != {"name", "arguments"} or call["name"] != "shell":
            raise ModelCompletionFormatError("only the Qwen shell tool is supported", "unsupported_tool_schema")
        arguments = call["arguments"]
        if not isinstance(arguments, dict) or set(arguments) != {"cmd"}:
            raise ModelCompletionFormatError("shell arguments must contain only cmd", "invalid_shell_arguments")
        command = arguments["cmd"]
        if (not isinstance(command, str) or not command.strip() or "\0" in command
                or len(command.encode()) > 8192):
            raise ModelCompletionFormatError("generated shell command is invalid or exceeds its byte budget", "invalid_shell_command")
        suffix = "" if len(matches) == 1 else "_" + str(index)
        calls.append({"id": "call_" + request_id + suffix, "type": "function",
            "function": {"name": "shell", "arguments": json.dumps(arguments, ensure_ascii=True, allow_nan=False)}})
    message["tool_calls"] = calls
    return message, "tool_calls"
