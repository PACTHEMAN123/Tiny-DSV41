"""Offline contract tests; no installed Harbor, model, sandbox or network needed."""

import asyncio
import copy
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import stat
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel, ConfigDict, ValidationError


def load_plugin():
    # Explicit interface stubs test the plugin, not Harbor installation/runtime.
    modules = {name: ModuleType(name) for name in (
        "harbor", "harbor.agents", "harbor.agents.base", "harbor.agents.options")}

    class Options(BaseModel):
        model_config = ConfigDict(extra="forbid")

    class Agent:
        def __init__(self, logs_dir, model_name=None, **kwargs):
            self.logs_dir, self.model_name = Path(logs_dir), model_name
            self.options = self.options_model.model_validate(kwargs)

    modules["harbor.agents.base"].BaseAgent = Agent
    modules["harbor.agents.options"].AgentOptions = Options
    source = Path(__file__).parents[2] / "rollout_manager/terminal_bench/harbor_filerelay_agent.py"
    spec = importlib.util.spec_from_file_location("tested_harbor_filerelay_agent", source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict("sys.modules", modules):
        spec.loader.exec_module(module)
    return module


a = load_plugin()


def completion(command=None, finish=None, model="model008", usage=None):
    message = {"role": "assistant", "content": "done" if command is None else None}
    if command is not None:
        message["tool_calls"] = [{"id": "call_1", "type": "function", "function": {
            "name": "shell", "arguments": json.dumps({"cmd": command})}}]
    result = {"model": model, "choices": [{"finish_reason": finish or ("stop" if command is None else "tool_calls"),
                                          "message": message}]}
    if usage is not None:
        result["usage"] = usage
    return result


class ParseTests(unittest.TestCase):
    def test_model_and_call_validation(self):
        for mutation in ("model", "duplicate", "tool", "args", "termination", "calls_type"):
            response = completion("pwd")
            message = response["choices"][0]["message"]
            if mutation == "model":
                response["model"] = "different"
            elif mutation == "duplicate":
                message["tool_calls"] *= 2
            elif mutation == "tool":
                message["tool_calls"][0]["function"]["name"] = "other"
            elif mutation == "args":
                message["tool_calls"][0]["function"]["arguments"] = '{"cmd":"a","cmd":"b"}'
            elif mutation == "termination":
                response["choices"][0]["finish_reason"] = "stop"
            else:
                message["tool_calls"] = False
            with self.subTest(mutation=mutation), self.assertRaises(a.RelayError):
                a.parse_completion(response, "model008")

    def test_length_stop_never_executes_partial_arguments(self):
        response = completion("pwd", finish="length")
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"cmd":'
        _, commands, stopped = a.parse_completion(response, "model008")
        self.assertEqual(commands, [])
        self.assertTrue(stopped)

    def test_nonfinite_duplicate_and_command_bounds(self):
        for raw in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
            with self.assertRaises(a.RelayError):
                a.strict_json(raw)
        for command in (" ", "a\0b", "x" * 8193):
            with self.assertRaises(a.RelayError):
                a.parse_completion(completion(command), "model008")

    def test_output_utf8_bound_and_nonzero_exit(self):
        result = SimpleNamespace(stdout="\u4e2d" * 10, stderr="error", return_code=3)
        observation = a.tool_observation(result, 17)
        self.assertEqual(observation["exit_code"], 3)
        self.assertTrue(observation["truncated"])
        self.assertLessEqual(len(observation["stdout"].encode()) + len(observation["stderr"].encode()), 17)
        result.return_code = False
        with self.assertRaises(a.RelayError):
            a.tool_observation(result, 17)

    def test_options_reject_coerced_and_unbounded_values(self):
        for field, value in (("generation_seconds", True), ("generation_seconds", 601),
                             ("max_turns", 0), ("max_tokens", 0), ("command_seconds", "30"),
                             ("temperature", float("inf"))):
            with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                a.FileRelayOptions(relay_root="/private", **{field: value})


class RelayFixture(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.spool, self.output = self.root / "spool", self.root / "logs"
        for path in (self.spool, self.spool / "inbox", self.spool / "outbox", self.output):
            path.mkdir(mode=0o700)
        self.relay = a.AsyncFileRelay(self.spool, self.output)

    async def asyncTearDown(self):
        self.temp.cleanup()

    async def published(self):
        async with asyncio.timeout(1):
            while not list((self.spool / "inbox").glob("*.json")):
                await asyncio.sleep(0.001)
        return next((self.spool / "inbox").glob("*.json"))


class RelayTests(RelayFixture):
    async def test_caller_identity_is_durable_and_never_replayed(self):
        identity = "a" * 32
        with self.assertRaises(a.RelayError):
            await self.relay.generate({"model": "model008"}, 0.01, request_id=identity)
        request = json.loads((self.spool / "inbox" / (identity + ".json")).read_text())
        self.assertEqual(request["id"], identity)
        with self.assertRaises(FileExistsError):
            await self.relay.generate({"model": "model008"}, 0.01, request_id=identity)
        self.assertEqual(len(list((self.spool / "inbox").glob("*.json"))), 1)

    async def test_invalid_caller_identity_fails_before_publication(self):
        with self.assertRaises(a.RelayError):
            await self.relay.generate({}, 1, request_id="../ambiguous")
        self.assertFalse(list((self.spool / "inbox").iterdir()))

    async def test_round_trip_exact_protocol_and_private_files(self):
        task = asyncio.create_task(self.relay.generate({"model": "model008"}, 1))
        request_path = await self.published()
        request = json.loads(request_path.read_text())
        self.assertEqual(set(request), {"id", "operation", "payload", "deadline_unix"})
        self.assertEqual(request["operation"], "generate")
        response = completion()
        a.write_new(self.spool / "outbox" / request_path.name,
                    a.json_bytes({"id": request["id"], "ok": True, "response": response}))
        self.assertEqual(await task, response)
        self.assertEqual(self.relay.request_ids, [request["id"]])
        self.assertFalse(list((self.spool / "inbox").glob("*.tmp")))
        self.assertTrue(all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in self.output.iterdir()))
        state = json.loads((self.output / (request["id"] + ".state.json")).read_text())
        self.assertEqual(state["state"], "response_received")

    async def test_deadline_does_not_replay_or_claim_remote_cancel(self):
        with self.assertRaisesRegex(a.RelayError, "not replayed"):
            await self.relay.generate({"model": "model008"}, 0.01)
        self.assertEqual(len(list((self.spool / "inbox").glob("*.json"))), 1)
        state = json.loads(next(self.output.glob("*.state.json")).read_text())
        self.assertEqual(state["state"], "deadline_expired_outcome_unknown")
        self.assertFalse(state["remote_cancellation_proven"])

    async def test_cancelled_publication_has_no_late_activity(self):
        task = asyncio.create_task(self.relay.generate({"model": "model008"}, 1))
        await self.published()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        state = json.loads(next(self.output.glob("*.state.json")).read_text())
        self.assertEqual(state["state"], "cancelled_outcome_unknown")
        await asyncio.sleep(0)
        self.assertEqual(len(list((self.spool / "inbox").glob("*.json"))), 1)

    async def test_prepublication_cancellation_removes_only_temporary(self):
        original = asyncio.sleep
        calls = 0

        async def cancel_second(delay):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise asyncio.CancelledError
            await original(delay)

        with patch.object(a.asyncio, "sleep", side_effect=cancel_second):
            with self.assertRaises(asyncio.CancelledError):
                await self.relay.generate({"model": "model008"}, 1)
        self.assertEqual(list((self.spool / "inbox").iterdir()), [])
        state = json.loads(next(self.output.glob("*.state.json")).read_text())
        self.assertFalse(state["published"])
        self.assertEqual(state["state"], "cancelled_before_publication")

    async def test_response_wrong_identity_and_symlink_reject(self):
        for mode in ("identity", "symlink"):
            task = asyncio.create_task(self.relay.generate({"model": "model008"}, 1))
            async with asyncio.timeout(1):
                while len(self.relay.request_ids) < (1 if mode == "identity" else 2):
                    await asyncio.sleep(0.001)
            request_id = self.relay.request_ids[-1]
            path = self.spool / "outbox" / (request_id + ".json")
            if mode == "identity":
                a.write_new(path, a.json_bytes({"id": "other", "ok": True, "response": completion()}))
            else:
                path.symlink_to(self.output / (request_id + ".request.json"))
            with self.assertRaises((a.RelayError, OSError)):
                await task

    async def test_directory_identity_and_request_size_fail_before_publish(self):
        with self.assertRaises(a.RelayError):
            await self.relay.generate({"large": "x" * a.MAX_REQUEST_BYTES}, 1)
        (self.spool / "inbox").rename(self.spool / "old-inbox")
        (self.spool / "inbox").mkdir(mode=0o700)
        with self.assertRaises(a.RelayError):
            await self.relay.generate({}, 1)
        self.assertEqual(self.relay.request_ids, [])


class AgentTests(RelayFixture):
    async def test_injected_transport_keeps_native_conversation_loop(self):
        relay = SimpleNamespace(generate=AsyncMock(return_value=completion()), request_ids=[])
        token = a.relay_factory.set(lambda output: relay)
        try:
            agent, environment, context = await self.make_agent()
        finally:
            a.relay_factory.reset(token)
        await agent.run("instruction", environment, context)
        relay.generate.assert_awaited_once()
        environment.exec.assert_not_awaited()
        self.assertTrue(context.metadata["file_relay"]["verification_owned_by_harbor"])

    async def make_agent(self, **kwargs):
        agent = a.FileRelayAgent(self.output, model_name="model008", relay_root=self.spool, **kwargs)
        environment = SimpleNamespace(exec=AsyncMock(return_value=SimpleNamespace(stdout="ok", stderr="", return_code=0)))
        await agent.setup(environment)
        context = SimpleNamespace(metadata=None, n_input_tokens=None, n_output_tokens=None, n_cache_tokens=None)
        return agent, environment, context

    async def test_agent_executes_tools_but_never_grades_or_uploads(self):
        agent, environment, context = await self.make_agent()
        payloads = []
        responses = [completion("false", usage={"prompt_tokens": 10, "completion_tokens": 2}),
                     completion(usage={"prompt_tokens": 20, "completion_tokens": 3})]

        async def generate(payload, timeout):
            payloads.append(copy.deepcopy(payload))
            return responses.pop(0)

        agent._relay.generate = generate
        environment.exec.return_value.return_code = 1
        await agent.run("Task instruction only", environment, context)
        environment.exec.assert_awaited_once_with(command="false", timeout_sec=30)
        self.assertEqual(payloads[0]["messages"][1]["content"], "Task instruction only")
        self.assertEqual(json.loads(payloads[1]["messages"][-1]["content"])["exit_code"], 1)
        self.assertEqual((context.n_input_tokens, context.n_output_tokens, context.n_cache_tokens), (30, 5, None))
        self.assertEqual(context.metadata["file_relay"]["stop_reason"], "model_final")
        self.assertTrue(context.metadata["file_relay"]["verification_owned_by_harbor"])

    async def test_agent_length_stop_has_no_shell_call(self):
        agent, environment, context = await self.make_agent()
        response = completion("ignored", finish="length")
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] = '{"cmd":'
        agent._relay.generate = AsyncMock(return_value=response)
        await agent.run("instruction", environment, context)
        environment.exec.assert_not_awaited()
        self.assertEqual(context.metadata["file_relay"]["stop_reason"], "token_budget")

    async def test_more_than_four_calls_execute_serially_and_feed_the_next_turn(self):
        agent, environment, context = await self.make_agent()
        commands = ["pwd", "printf READY", "ls", "true", "git status", "printf DONE"]
        response = completion(commands[0])
        calls = response["choices"][0]["message"]["tool_calls"]
        calls[:] = [{"id": f"call_{index}", "type": "function", "function": {
            "name": "shell", "arguments": json.dumps({"cmd": command})}}
            for index, command in enumerate(commands)]
        payloads, executed = [], []
        active = False

        async def execute(command, timeout_sec):
            nonlocal active
            self.assertFalse(active)
            self.assertEqual(timeout_sec, 30)
            active = True
            await asyncio.sleep(0)
            executed.append(command)
            active = False
            return SimpleNamespace(stdout=command, stderr="", return_code=0)

        async def generate(payload, timeout):
            payloads.append(copy.deepcopy(payload))
            return response if len(payloads) == 1 else completion()

        agent._relay.generate = generate
        environment.exec.side_effect = execute
        await agent.run("instruction", environment, context)
        self.assertEqual(executed, commands)
        observations = payloads[1]["messages"][-len(commands):]
        self.assertEqual([row["tool_call_id"] for row in observations], [row["id"] for row in calls])
        self.assertEqual([json.loads(row["content"])["stdout"] for row in observations], commands)
        self.assertEqual(context.metadata["file_relay"]["stop_reason"], "model_final")

    async def test_invalid_later_call_prevents_execution_of_valid_prefix(self):
        agent, environment, context = await self.make_agent()
        response = completion("pwd")
        calls = response["choices"][0]["message"]["tool_calls"]
        calls[:] = [{"id": f"call_{index}", "type": "function", "function": {
            "name": "shell", "arguments": json.dumps({"cmd": "pwd" if index < 5 else "x" * 8193})}}
            for index in range(6)]
        agent._relay.generate = AsyncMock(return_value=response)
        with self.assertRaises(a.RelayError):
            await agent.run("instruction", environment, context)
        environment.exec.assert_not_awaited()

    async def test_agent_timeout_never_replays_command_or_generates_again(self):
        agent, environment, context = await self.make_agent()
        agent._relay.generate = AsyncMock(return_value=completion("command"))
        environment.exec.side_effect = TimeoutError()
        with self.assertRaises(TimeoutError):
            await agent.run("instruction", environment, context)
        self.assertEqual(agent._relay.generate.await_count, 1)
        self.assertEqual(environment.exec.await_count, 1)
        self.assertEqual(context.metadata["file_relay"]["stop_reason"], "execution_timeout_uncertain")

    async def test_proven_command_timeout_is_an_observation_with_grace(self):
        agent, environment, context = await self.make_agent(max_turns=1)
        agent._relay.generate = AsyncMock(return_value=completion("sleep"))
        environment.exec.return_value.return_code = 124
        original = asyncio.wait_for
        observed = []

        async def wait_with_grace(awaitable, timeout):
            observed.append(timeout)
            return await original(awaitable, timeout)

        with patch.object(a.asyncio, "wait_for", side_effect=wait_with_grace):
            await agent.run("instruction", environment, context)
        self.assertEqual(observed, [])
        self.assertEqual(environment.exec.await_args.kwargs["timeout_sec"], 30)
        self.assertEqual(context.metadata["file_relay"]["stop_reason"], "turn_limit")
        trajectory = json.loads((agent._private_logs / "trajectory.json").read_text())
        self.assertEqual(json.loads(trajectory[-1]["content"])["exit_code"], 124)

    async def test_agent_harbor_cancellation_propagates_and_persists(self):
        agent, environment, context = await self.make_agent()
        entered = asyncio.Event()

        async def wait_forever(payload, timeout):
            entered.set()
            await asyncio.Event().wait()

        agent._relay.generate = wait_forever
        task = asyncio.create_task(agent.run("instruction", environment, context))
        await entered.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        environment.exec.assert_not_awaited()
        self.assertEqual(context.metadata["file_relay"]["stop_reason"], "cancelled_by_harbor")
        self.assertTrue((agent._private_logs / "trajectory.json").is_file())

    async def test_turn_limit_unknown_usage_and_reentry_guard(self):
        agent, environment, context = await self.make_agent(max_turns=1)
        agent._relay.generate = AsyncMock(return_value=completion("pwd"))
        await agent.run("instruction", environment, context)
        self.assertEqual(context.metadata["file_relay"]["stop_reason"], "turn_limit")
        self.assertIsNone(context.n_input_tokens)
        with self.assertRaises(a.RelayError):
            await agent.run("instruction", environment, context)
        self.assertEqual(agent._relay.generate.await_count, 1)


class InstalledHarborContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_factories_options_and_context_when_available(self):
        try:
            from rollout_manager.terminal_bench import harbor_filerelay_agent as real
            from harbor.agents.factory import AgentFactory
            from harbor.environments.factory import EnvironmentFactory
            from harbor.models.agent.context import AgentContext
            from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
            from harbor.models.trial.config import AgentConfig
            from harbor.models.trial.paths import TrialPaths
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            if os.environ.get("HARBOR_REQUIRE_REAL") == "1":
                self.fail("Real Harbor import prerequisite unavailable: " + type(error).__name__)
            self.skipTest("Harbor distribution/dependencies not installed; contract fixtures only")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spool, logs, environment_dir = root / "spool", root / "logs", root / "environment"
            for path in (spool, spool / "inbox", spool / "outbox", logs, environment_dir):
                path.mkdir(mode=0o700)
            environment = EnvironmentFactory.create_environment_from_import_path(
                "rollout_manager.terminal_bench.harbor_pouch_environment:PouchEnvironment", environment_dir=environment_dir,
                environment_name="contract", session_id="contract",
                trial_paths=TrialPaths(trial_dir=root / "trial"),
                task_env_config=EnvironmentConfig(docker_image="example/task:pinned", cpus=1,
                    memory_mb=2048, storage_mb=10240),
                network_policy=NetworkPolicy(network_mode="no-network"))
            agent = AgentFactory.create_agent_from_config(AgentConfig(
                import_path="rollout_manager.terminal_bench.harbor_filerelay_agent:FileRelayAgent", model_name="model008",
                kwargs={"relay_root": str(spool)}), logs_dir=logs)
            self.assertIsInstance(agent, real.FileRelayAgent)
            self.assertIsInstance(agent.options, real.FileRelayOptions)
            self.assertEqual(agent.model_name, "model008")
            self.assertFalse(agent.capabilities.atif)
            with self.assertRaises(ValueError):
                type(agent).preflight({"relay_root": str(spool), "unknown": True})
            await agent.setup(environment)
            agent._relay.generate = AsyncMock(return_value=completion(
                usage={"prompt_tokens": 10, "completion_tokens": 2}))
            context = AgentContext()
            await agent.run("Synthetic contract instruction", environment, context)
            self.assertEqual(context.n_input_tokens, 10)
            self.assertEqual(context.n_output_tokens, 2)
            self.assertIsNone(context.n_cache_tokens)
            self.assertEqual(context.metadata["file_relay"]["stop_reason"], "model_final")
            self.assertFalse(environment._create_attempted)
            self.assertEqual(list((spool / "inbox").iterdir()), [])


if __name__ == "__main__":
    unittest.main()
