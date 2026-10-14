"""Harbor host-side shell agent using the existing private FileRelay protocol.

Import path: harbor_filerelay_agent:FileRelayAgent. Harbor owns task deadlines,
official verification and sandbox cleanup; this plugin never uploads task sources.
"""

import asyncio
from contextvars import ContextVar
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
import uuid

from harbor.agents.base import BaseAgent
from harbor.agents.options import AgentOptions
from pydantic import Field


MAX_REQUEST_BYTES = 512 * 1024
MAX_RESPONSE_BYTES = 1024 * 1024
MAX_COMMAND_BYTES = 8192
CALL_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
# Each admitted Harbor job inherits its own transport factory. No process-global
# plugin registration or direct model route is needed.
relay_factory = ContextVar("harbor_relay_factory", default=None)
TOOLS = [{"type": "function", "function": {
    "name": "shell", "description": "Run a shell command in the task environment.",
    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}},
                   "required": ["cmd"], "additionalProperties": False}}}]


class RelayError(RuntimeError):
    pass


class FileRelayOptions(AgentOptions):
    relay_root: Path
    generation_seconds: int = Field(default=120, ge=1, le=600, strict=True)
    command_seconds: int = Field(default=30, ge=1, le=12000, strict=True)
    max_turns: int = Field(default=128, ge=1, le=1024, strict=True)
    max_tokens: int = Field(default=2048, ge=1, le=32768, strict=True)
    max_tool_output_bytes: int = Field(default=16384, ge=1024, le=65536, strict=True)
    temperature: float = Field(default=0.0, ge=0, le=2, allow_inf_nan=False)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RelayError("Duplicate JSON key")
            result[key] = value
        return result

    def finite(value):
        result = float(value)
        if not math.isfinite(result):
            raise RelayError("Nonfinite JSON number")
        return result

    def invalid(_):
        raise RelayError("Nonfinite JSON constant")

    return json.loads(raw, object_pairs_hook=pairs, parse_float=finite,
                      parse_constant=invalid)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n").encode()


def write_new(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def private_directory(path):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode) & 0o077:
        raise RelayError("Expected a private real relay directory")
    return info.st_uid, info.st_gid, info.st_dev, info.st_ino


def bounded_response(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_RESPONSE_BYTES:
            raise RelayError("Relay response is not a bounded regular file")
        data = stream.read(MAX_RESPONSE_BYTES + 1)
        after = os.fstat(stream.fileno())
    signature = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns)
    if len(data) != before.st_size or signature(before) != signature(after):
        raise RelayError("Relay response changed while reading")
    return strict_json(data)


class AsyncFileRelay:
    """Atomic durable submission with cancellable polling, never replay."""

    def __init__(self, root, output, *, expected_policy_ack=None):
        self.root, self.output = Path(root), Path(output)
        self.expected_policy_ack = expected_policy_ack
        if not self.root.is_absolute():
            raise RelayError("relay_root must be absolute")
        self.directories = [self.root, self.root / "inbox", self.root / "outbox"]
        self.identities = [private_directory(path) for path in self.directories]
        if len({item[:2] for item in self.identities}) != 1:
            raise RelayError("Relay directory ownership differs")
        self.request_ids = []

    def check_directories(self):
        if [private_directory(path) for path in self.directories] != self.identities:
            raise RelayError("Relay directory identity changed")

    async def generate(self, payload, timeout, request_id=None):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 600:
            raise RelayError("Invalid generation deadline")
        await asyncio.sleep(0)
        self.check_directories()
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        request_id = uuid.uuid4().hex if request_id is None else request_id
        if not isinstance(request_id, str) or re.fullmatch(r"[a-f0-9]{32}", request_id) is None:
            raise RelayError("Expected a durable 32-hex request identity")
        request = {"id": request_id, "operation": "generate", "payload": payload,
                   "deadline_unix": time.time() + timeout}
        data = json_bytes(request)
        if len(data) > MAX_REQUEST_BYTES:
            raise RelayError("Generation request exceeds byte budget")
        write_new(self.output / (request_id + ".request.json"), data)
        temporary = self.root / "inbox" / ("." + request_id + ".tmp")
        target = self.root / "inbox" / (request_id + ".json")
        created, published, state = False, False, "error_before_publication"
        failure_receipt = None
        try:
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
            created = True
            with os.fdopen(fd, "wb") as stream:
                info = os.fstat(stream.fileno())
                owner = self.identities[0][:2]
                if (info.st_uid, info.st_gid) != owner:
                    os.fchown(stream.fileno(), *owner)
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # Deliver cancellation before the externally observable submission.
            await asyncio.sleep(0)
            self.check_directories()
            if loop.time() >= deadline:
                raise RelayError("Generation deadline expired before publication")
            os.link(temporary, target)
            published = True
            self.request_ids.append(request_id)
            state = "published_outcome_unknown"
            result_path = self.root / "outbox" / (request_id + ".json")
            while loop.time() < deadline:
                self.check_directories()
                try:
                    response = bounded_response(result_path)
                except FileNotFoundError:
                    await asyncio.sleep(min(0.1, max(0, deadline - loop.time())))
                    continue
                write_new(self.output / (request_id + ".response.json"), json_bytes(response))
                if not isinstance(response, dict) or response.get("id") != request_id:
                    raise RelayError("Relay response identity differs")
                if response.get("ok") is not True or not isinstance(response.get("response"), dict):
                    if (response.get("ok") is not False or not isinstance(response.get("error"), str)
                            or set(response) - {"id", "ok", "error", "generation_failure"}):
                        raise RelayError("Relay returned an invalid negative completion")
                    if "generation_failure" in response:
                        from rollout_manager.generation_failure import validate_generation_failure, failure_is_terminal
                        try:
                            failure_receipt = validate_generation_failure(response["generation_failure"], payload,
                                request_id, self.expected_policy_ack)
                        except (ValueError, TypeError, KeyError) as error:
                            raise RelayError("Relay failure closure evidence is unproven") from error
                        if failure_is_terminal(failure_receipt):
                            state = "terminal_failure_received"
                    error = RelayError("Relay returned no successful model completion")
                    error.failure_receipt = failure_receipt
                    error.request_terminal_proven = state == "terminal_failure_received"
                    raise error
                if "generation_failure" in response:
                    raise RelayError("A successful completion cannot carry failure closure evidence")
                state = "response_received"
                return response["response"]
            state = "deadline_expired_outcome_unknown"
            raise RelayError("Relay deadline expired; request was not replayed")
        except asyncio.CancelledError:
            state = "cancelled_outcome_unknown" if published else "cancelled_before_publication"
            raise
        finally:
            if created:
                temporary.unlink(missing_ok=True)
            write_new(self.output / (request_id + ".state.json"), json_bytes({
                "id": request_id, "published": published, "state": state, "replayed": False,
                "remote_cancellation_proven": False,
                **({"generation_failure": "validated", "failure_receipt": failure_receipt,
                    "request_payload_sha256": failure_receipt["request_payload_sha256"]}
                   if failure_receipt is not None else {})}))


def parse_completion(response, expected_model):
    if response.get("model") != expected_model:
        raise RelayError("Model identity differs from the selected served model")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise RelayError("Expected one model choice")
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise RelayError("Invalid assistant completion")
    for key in ("content", "reasoning_content"):
        if message.get(key) is not None and not isinstance(message[key], str):
            raise RelayError("Assistant content must be text or null")
    calls = message.get("tool_calls")
    if calls is None:
        calls = []
    if not isinstance(calls, list):
        raise RelayError("Expected a list of shell calls")
    length_stop = choice.get("finish_reason") == "length"
    commands, seen = [], set()
    for call in calls:
        if not isinstance(call, dict) or call.get("type") != "function":
            raise RelayError("Unsupported tool call")
        identity, function = call.get("id"), call.get("function")
        if not isinstance(identity, str) or not CALL_ID.fullmatch(identity) or identity in seen:
            raise RelayError("Invalid or duplicate tool call identity")
        seen.add(identity)
        if not isinstance(function, dict) or function.get("name") != "shell":
            raise RelayError("Only shell calls are supported")
        arguments = function.get("arguments")
        if not isinstance(arguments, str) or len(arguments.encode()) > MAX_COMMAND_BYTES * 2:
            raise RelayError("Invalid tool argument envelope")
        if length_stop:
            continue
        parsed = strict_json(arguments)
        if not isinstance(parsed, dict) or set(parsed) != {"cmd"}:
            raise RelayError("Tool arguments must contain only cmd")
        command = parsed["cmd"]
        if not isinstance(command, str) or not command.strip() or "\0" in command or len(command.encode()) > MAX_COMMAND_BYTES:
            raise RelayError("Command is invalid or exceeds byte budget")
        commands.append((identity, command))
    if not length_stop and choice.get("finish_reason") != ("tool_calls" if calls else "stop"):
        raise RelayError("Completion was not cleanly terminated")
    clean = {"role": "assistant", "content": message.get("content")}
    if message.get("reasoning_content") is not None:
        clean["reasoning_content"] = message["reasoning_content"]
    if calls:
        clean["tool_calls"] = calls
    return clean, commands, length_stop


def tool_observation(result, maximum):
    if type(result.return_code) is not int:
        raise RelayError("Environment exec completion is uncertain")
    kept, remaining, truncated = {}, maximum, False
    for key in ("stdout", "stderr"):
        value = getattr(result, key)
        if value is None:
            value = ""
        if not isinstance(value, str):
            raise RelayError("Environment output is not text")
        raw = value.encode(errors="replace")
        part = raw[:remaining].decode(errors="ignore")
        kept[key] = part
        truncated |= len(raw) > remaining
        remaining -= len(part.encode())
    return {"exit_code": result.return_code, "truncated": truncated, **kept}


class FileRelayAgent(BaseAgent):
    options_model = FileRelayOptions

    @staticmethod
    def name():
        return "file-relay"

    def version(self):
        return "1.0.0"

    async def setup(self, environment):
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise RelayError("An explicit served model name is required")
        if getattr(self, "_relay", None) is not None:
            raise RelayError("Agent setup must not be repeated")
        self._private_logs = self.logs_dir / "file-relay"
        self._private_logs.mkdir(mode=0o700, parents=False, exist_ok=False)
        factory = relay_factory.get()
        self._relay = (factory(self._private_logs) if factory is not None else
                       AsyncFileRelay(self.options.relay_root, self._private_logs))

    async def run(self, instruction, environment, context):
        if getattr(self, "_relay", None) is None:
            raise RelayError("Agent setup has not completed")
        if getattr(self, "_run_started", False):
            raise RelayError("This agent instance has already started a rollout")
        if not isinstance(instruction, str) or not instruction.strip():
            raise RelayError("Task instruction is missing")
        self._run_started = True
        options = self.options
        policy = {name: getattr(options, name) for name in (
            "generation_seconds", "command_seconds", "max_turns", "max_tokens",
            "max_tool_output_bytes", "temperature")}
        messages = [{"role": "system", "content": (
            "Solve the task in its Linux environment using the shell tool. "
            "Commands use the environment's working directory; filesystem changes persist, shell state does not. "
            f"Each command has a {options.command_seconds}-second deadline and bounded output. "
            "Never assume a failed or timed-out command was not executed. "
            "Finish with a brief final answer and no tool calls.")},
            {"role": "user", "content": instruction}]
        tools = json.loads(json.dumps(TOOLS))
        tools[0]["function"]["description"] += f" Deadline {options.command_seconds} seconds; bounded output."
        stop_reason, turns, error_type = "error", 0, None
        completion_status = None
        usage = {"prompt_tokens": [], "completion_tokens": [], "cached_tokens": []}
        try:
            for turn in range(options.max_turns):
                response = await self._relay.generate({
                    "model": self.model_name, "messages": messages, "tools": tools,
                    "tool_choice": "auto", "parallel_tool_calls": False,
                    "chat_template_kwargs": {"enable_thinking": False}, "stream": False,
                    "temperature": options.temperature, "max_tokens": options.max_tokens},
                    options.generation_seconds)
                if response.get("model_completion_status") is not None:
                    from rollout_manager.model_completion import validate_model_completion_status
                    if not self._relay.request_ids:
                        raise RelayError("Completed model status has no durable request identity")
                    completion_status = validate_model_completion_status(response, self._relay.request_ids[-1])
                message, commands, length_stop = parse_completion(response, self.model_name)
                turns = turn + 1
                reported = response.get("usage")
                reported = reported if isinstance(reported, dict) else {}
                details = reported.get("prompt_tokens_details")
                details = details if isinstance(details, dict) else {}
                for key in usage:
                    value = details.get(key) if key == "cached_tokens" else reported.get(key)
                    usage[key].append(value if type(value) is int and value >= 0 else None)
                messages.append(message)
                if completion_status is not None:
                    # A completed, typed grammar failure retains its raw action
                    # trace and proceeds to Harbor's unchanged verifier.
                    stop_reason = "model_invalid_tool"
                    break
                if length_stop:
                    stop_reason = "token_budget"
                    break
                if not commands:
                    stop_reason = "model_final"
                    break
                for identity, command in commands:
                    # The provider owns command and transport bounds. Harbor's
                    # unchanged outer task deadline still controls the rollout.
                    result = await environment.exec(
                        command=command, timeout_sec=options.command_seconds)
                    observation = tool_observation(result, options.max_tool_output_bytes)
                    messages.append({"role": "tool", "tool_call_id": identity,
                                     "content": json.dumps(observation, ensure_ascii=True)})
            else:
                stop_reason = "turn_limit"
        except asyncio.CancelledError:
            stop_reason = "cancelled_by_harbor"
            raise
        except Exception as error:
            error_type = type(error).__name__
            stop_reason = "execution_timeout_uncertain" if isinstance(error, TimeoutError) else "infrastructure_error"
            raise
        finally:
            for source, field in (("prompt_tokens", "n_input_tokens"), ("completion_tokens", "n_output_tokens"),
                                  ("cached_tokens", "n_cache_tokens")):
                values = usage[source]
                setattr(context, field, sum(values) if values and all(value is not None for value in values) else None)
            metadata = {"plugin": self.name(), "version": self.version(), "model": self.model_name,
                        "policy": policy, "stop_reason": stop_reason, "turns": turns,
                        "request_ids": list(self._relay.request_ids), "generation_retries": 0,
                        "error_type": error_type, "verification_owned_by_harbor": True,
                        "remote_cancellation_proven": False,
                        "plugin_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
            if completion_status is not None:
                metadata["model_completion_status"] = completion_status
            traces = getattr(self._relay, "training_traces", None)
            if traces is not None:
                # The injected relay checks raw serving evidence before any
                # completion is parsed or a generated shell action executes.
                metadata["training_evidence"] = {"validated_turns": len(traces),
                    "request_ids": [trace.request_id for trace in traces],
                    "policy": traces[0].policy.to_dict() if traces else None}
            context.metadata = {**(context.metadata or {}), "file_relay": metadata}
            write_new(self._private_logs / "trajectory.json", json_bytes(messages))
            write_new(self._private_logs / "agent-result.json", json_bytes(metadata))
