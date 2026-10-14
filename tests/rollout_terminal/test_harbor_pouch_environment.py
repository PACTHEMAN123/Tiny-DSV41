"""Focused contract/Unix-HTTP tests; these do not claim live Harbor/Pouch proof."""

import asyncio
from enum import Enum
import importlib.util
import importlib.metadata
import io
import json
import os
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
import types
import unittest
from unittest.mock import patch


class NetworkMode(str, Enum):
    NO_NETWORK = "no-network"
    PUBLIC = "public"
    ALLOWLIST = "allowlist"


class ResourceMode(str, Enum):
    LIMIT = "limit"


class Model:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class BaseContract:
    """Only the locally audited Harbor 0.23.0 methods this plugin calls."""
    def __init__(self, environment_dir, environment_name, session_id, trial_paths,
                 task_env_config, network_policy=None, persistent_env=None,
                 mounts=None, **kwargs):
        self.environment_dir = environment_dir
        self.environment_name, self.session_id = environment_name, session_id
        self.trial_paths, self.task_env_config = trial_paths, task_env_config
        self.network_policy = network_policy or Model(network_mode=NetworkMode.PUBLIC)
        self._persistent_env = {**task_env_config.env, **(persistent_env or {})}
        self.default_user = None
        self._effective_storage_mb = task_env_config.storage_mb
        self._validate_definition()

    def _resource_limit_value(self, key, auto_mode):
        return getattr(self.task_env_config, {"cpu": "cpus", "memory": "memory_mb"}[key])

    def _startup_env(self):
        return self._persistent_env

    def _merge_env(self, env):
        return {**self._persistent_env, **(env or {})}

    def _resolve_user(self, user):
        return self.default_user if user is None else user

    def _output_callback(self):
        return None

    def _mount_targets(self, writable_only):
        return []

    async def ensure_dirs(self, dirs):
        pass

    async def _upload_environment_dir_after_start(self):
        self.environment_upload_checked = True


def load_plugin():
    replacements = {
        "harbor.environments.base": {"BaseEnvironment": BaseContract, "ExecResult": Model},
        "harbor.environments.capabilities": {
            "EnvironmentCapabilities": Model, "EnvironmentResourceCapabilities": Model},
        "harbor.models.task.config": {"NetworkMode": NetworkMode},
        "harbor.models.trial.config": {"ResourceMode": ResourceMode},
    }
    modules = {}
    for name, attrs in replacements.items():
        module = types.ModuleType(name)
        module.__dict__.update(attrs)
        modules[name] = module
    spec = importlib.util.spec_from_file_location("pouch_contract_under_test",
        Path(__file__).parents[2] / "rollout_manager/terminal_bench/harbor_pouch_environment.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


plugin = load_plugin()
CID, EID, IID = "c" * 64, "e" * 64, "sha256:" + "a" * 64


def frame(value, stream=1):
    value = value.encode()
    return bytes([stream, 0, 0, 0]) + struct.pack(">I", len(value)) + value


def archive_bytes(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, body in entries:
            member = tarfile.TarInfo(name)
            member.size = len(body)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(body))
    return output.getvalue()


class FakeAPI:
    def __init__(self, owner, implementation=plugin):
        self.owner = owner
        self.implementation = implementation
        self.calls = []
        self.exists = self.running = False
        self.payload = None
        self.exit_code = 0
        self.exec_running = False
        self.exec_error = None
        self.create_wait = None
        self.create_entered = asyncio.Event()
        self.download = b""
        self.host_override = {}
        self.create_id = CID
        self.create_error = None
        self.exec_wait = None
        self.exec_entered = asyncio.Event()
        self.image_id = IID
        self.require_pinned_image = False

    async def json(self, method, path, payload=None, **kwargs):
        self.calls.append((method, path, payload, kwargs))
        if path.startswith("/images/"):
            if self.require_pinned_image and path != "/images/sha256%3A" + "a" * 64 + "/json":
                raise self.implementation.PouchHTTPError(404)
            return {"Id": self.image_id, "Os": "linux", "Config": {
                "WorkingDir": "/workspace/nested", "Entrypoint": ["/entrypoint"], "Env": ["INHERITED=1"]}}
        if path.startswith("/containers/create?"):
            self.create_entered.set()
            if self.create_wait is not None:
                await self.create_wait.wait()
            if self.create_error is not None:
                raise self.create_error
            self.payload, self.exists = payload, True
            return {"Id": self.create_id, "Warnings": []}
        if path.startswith("/containers/") and path.endswith("/json"):
            if not self.exists:
                raise self.implementation.PouchHTTPError(404)
            return {"Id": CID, "Config": {"Labels": {self.implementation.OWNER_LABEL: self.owner}},
                    "HostConfig": {**self.payload["HostConfig"], **self.host_override}, "State": {"Running": self.running}}
        if path.endswith("/exec"):
            return {"Id": EID}
        if path == "/exec/" + EID + "/json":
            return {"Running": self.exec_running, "ExitCode": self.exit_code}
        raise AssertionError((method, path))

    async def request(self, method, path, payload=None, **kwargs):
        self.calls.append((method, path, payload, kwargs))
        if path == "/containers/" + CID + "/start":
            self.running = True
        elif "/stop?" in path:
            self.running = False
        elif method == "DELETE":
            self.exists = False
        elif path == "/exec/" + EID + "/start":
            self.exec_entered.set()
            if self.exec_wait is not None:
                await self.exec_wait.wait()
            if self.exec_error is not None:
                raise self.exec_error
            return frame("out") + frame("err", 2)
        elif "/archive?" in path:
            return self.download if method == "GET" else b""
        else:
            raise AssertionError((method, path))
        return b""


class EnvironmentTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.environment = self.root / "environment"
        self.environment.mkdir()
        self.env = plugin.PouchEnvironment(
            environment_dir=self.environment, environment_name="task", session_id="trial__env",
            trial_paths=Model(trial_dir=self.root / "trial"), task_env_config=Model(
                docker_image="example/task:pinned", cpus=4, memory_mb=8192, storage_mb=10240,
                env={"TASK_VALUE": "one"}, workdir=None),
            network_policy=Model(network_mode=NetworkMode.NO_NETWORK))
        self.api = self.env.api = FakeAPI(self.env.owner)

    async def test_start_uses_harbor_cmd_and_inherits_entrypoint_cwd(self):
        await self.env.start(False)
        request = self.api.payload
        self.assertEqual(request["Image"], IID)
        self.assertEqual(request["Cmd"], ["sh", "-c", "sleep infinity"])
        self.assertNotIn("Entrypoint", request)
        self.assertEqual(request["WorkingDir"], "/workspace/nested")
        self.assertEqual(request["HostConfig"], {"NetworkMode": "none", "Privileged": False,
            "CpuPeriod": 100000, "CpuQuota": 400000, "Memory": 8192 * 1024**2})
        self.assertEqual(request["Env"], ["TASK_VALUE=one"])
        receipt = json.loads((self.root / "trial/pouch-environment.json").read_text())
        self.assertFalse(receipt["storage_enforced"])
        self.assertEqual(receipt["storage_requested_mb"], 10240)
        self.assertEqual((self.root / "trial/pouch-environment.json").stat().st_mode & 0o777, 0o600)
        self.assertTrue(self.env.environment_upload_checked)

    async def test_start_never_pulls_or_builds_or_repeats(self):
        with self.assertRaises(ValueError):
            await self.env.start(True)
        self.assertFalse(self.api.calls)
        await self.env.start(False)
        with self.assertRaises(plugin.PouchError):
            await self.env.start(False)
        self.assertEqual(sum("/containers/create?" in c[1] for c in self.api.calls), 1)

    async def test_original_tag_missing_but_pinned_configuration_is_present(self):
        self.env.image_id_by_reference = {"example/task:pinned": IID}
        self.api.require_pinned_image = True
        await self.env.start()
        self.assertEqual(self.api.payload["Image"], IID)
        self.assertEqual(self.env.task_env_config.docker_image, "example/task:pinned")
        self.assertEqual(self.env._receipt["image_resolution"], "pinned-config-map")
        self.assertEqual(sum(c[1].startswith("/images/") for c in self.api.calls), 1)

    async def test_pinned_image_mismatch_fails_before_create(self):
        self.env.image_id_by_reference = {"example/task:pinned": IID}
        self.api.image_id = "sha256:" + "b" * 64
        with self.assertRaisesRegex(plugin.PouchError, "pinned configuration"):
            await self.env.start()
        self.assertFalse(any("/containers/create?" in c[1] for c in self.api.calls))

    def test_map_is_bounded_validated_and_does_not_follow_symlinks(self):
        path = self.root / "image-map.json"
        path.write_text(json.dumps({"example/task:pinned": IID}))
        self.assertEqual(plugin.image_mapping(None, path), {"example/task:pinned": IID})
        link = self.root / "map-link"
        link.symlink_to(path)
        with self.assertRaises(OSError):
            plugin.image_mapping(None, link)
        path.write_text('{"same":"' + IID + '","same":"' + IID + '"}')
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            plugin.image_mapping(None, path)
        with self.assertRaises(ValueError):
            plugin.image_mapping({"reference": "sha256:short"}, None)

    async def test_public_network_requires_explicit_constructor_opt_in(self):
        args = dict(environment_dir=self.environment, environment_name="task", session_id="trial",
                    trial_paths=Model(trial_dir=self.root / "trial2"), task_env_config=self.env.task_env_config)
        with self.assertRaisesRegex(ValueError, "allow_public_network"):
            plugin.PouchEnvironment(**args)
        allowed = plugin.PouchEnvironment(**args, allow_public_network=True)
        allowed.api = FakeAPI(allowed.owner)
        await allowed.start()
        self.assertEqual(allowed.api.payload["HostConfig"]["NetworkMode"], "bridge")
        self.assertTrue(allowed._receipt["public_network_explicitly_allowed"])

    async def test_fractional_cpu_never_truncated(self):
        self.env.task_env_config.cpus = 1.5
        with self.assertRaisesRegex(ValueError, "positive integer"):
            await self.env.start()
        self.assertFalse(any("/containers/create?" in c[1] for c in self.api.calls))

    async def test_changed_resource_limit_cleans_before_start(self):
        self.api.host_override["CpuQuota"] = 100000
        with self.assertRaisesRegex(plugin.PouchError, "CpuQuota"):
            await self.env.start()
        self.assertFalse(self.api.exists)
        self.assertFalse(any(c[1] == "/containers/" + CID + "/start" for c in self.api.calls))

    async def test_malformed_create_id_resolves_only_owned_random_name(self):
        self.api.create_id = "truncated"
        with self.assertRaises(plugin.PouchError):
            await self.env.start()
        self.assertFalse(self.api.exists)
        self.assertIn(("GET", "/containers/" + self.env.container_name + "/json", None,
                       {"timeout": 30.0}), self.api.calls)

    async def test_lost_create_response_without_visible_container_is_not_proved_clean(self):
        self.api.create_error = TimeoutError("lost create response")
        with self.assertRaises(TimeoutError):
            await self.env.start()
        self.assertFalse(self.env._receipt["cleanup_verified"])
        self.assertTrue(self.env._receipt["create_outcome_unknown"])
        self.assertFalse(any(c[0] == "DELETE" for c in self.api.calls))

    async def test_cwd_user_env_and_timeout_wrapper(self):
        await self.env.start()
        self.env.default_user = "runner"
        result = await self.env.exec("printf ok", cwd="/tmp/a b", env={"TASK_VALUE": "two"}, timeout_sec=2.5)
        request = next(c[2] for c in self.api.calls if c[1].endswith("/exec"))
        self.assertEqual(request["User"], "runner")
        self.assertEqual(request["Env"], ["TASK_VALUE=two"])
        self.assertEqual(request["Cmd"][4], "/tmp/a b")
        self.assertEqual(request["Cmd"][5:9], ["timeout", "--signal=TERM", "--kill-after=2s", "2.5s"])
        self.assertEqual((result.stdout, result.stderr, result.return_code), ("out", "err", 0))

    async def test_nonzero_exit_is_a_result_not_replay_or_container_stop(self):
        await self.env.start()
        self.api.exit_code = 124
        result = await self.env.exec("sleep 20", timeout_sec=1)
        self.assertEqual(result.return_code, 124)
        self.assertTrue(self.api.running)
        self.assertEqual(sum(c[1] == "/exec/" + EID + "/start" for c in self.api.calls), 1)

    async def test_uncertain_exec_aborts_exact_container_without_replay(self):
        await self.env.start()
        self.api.exec_running = True
        with self.assertRaises(plugin.PouchError):
            await self.env.exec("do not replay")
        self.assertFalse(self.api.exists)
        with self.assertRaises(plugin.PouchError):
            await self.env.exec("another")
        self.assertEqual(sum(c[1].endswith("/exec") for c in self.api.calls), 1)

    async def test_cancel_exec_cleans_owned_container(self):
        await self.env.start()
        self.api.exec_error = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.env.exec("uncertain")
        self.assertFalse(self.api.exists)

    async def test_cancel_bounded_exec_drains_then_verifier_reads_state(self):
        await self.env.start()
        self.api.exec_wait = asyncio.Event()
        command = asyncio.create_task(self.env.exec("write task state", timeout_sec=1))
        await self.api.exec_entered.wait()
        command.cancel()
        await asyncio.sleep(0)
        self.assertFalse(command.done())
        self.api.exec_wait.set()
        with self.assertRaises(asyncio.CancelledError):
            await command
        self.assertTrue(self.api.exists)
        self.assertFalse(self.env._broken)
        self.assertTrue(self.env._receipt["cancellation_terminal_proven"])
        self.assertGreaterEqual(self.env._receipt["deadline_overrun_seconds"], 0)
        result = await self.env.exec("read completed state for stock verifier")
        self.assertEqual(result.return_code, 0)

    async def test_cancel_bounded_exec_without_terminal_proof_is_broken(self):
        await self.env.start()
        self.api.exec_wait = asyncio.Event()
        with patch.object(plugin, "EXEC_GRACE_SEC", 0.02):
            command = asyncio.create_task(self.env.exec("uncertain state", timeout_sec=0.02))
            await self.api.exec_entered.wait()
            command.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await command
        self.assertFalse(self.api.exists)
        self.assertTrue(self.env._broken)
        self.assertFalse(self.env._receipt["cancellation_terminal_proven"])
        with self.assertRaises(plugin.PouchError):
            await self.env.exec("verifier cannot consume uncertain state")

    async def test_cancel_inflight_create_waits_then_cleans(self):
        self.api.create_wait = asyncio.Event()
        task = asyncio.create_task(self.env.start())
        await self.api.create_entered.wait()
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        self.api.create_wait.set()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(self.api.exists)
        self.assertEqual(sum("/containers/create?" in c[1] for c in self.api.calls), 1)

    async def test_foreign_container_never_stopped_or_deleted(self):
        await self.env.start()
        self.api.owner = "someone-else"
        with self.assertRaises(plugin.PouchError):
            await self.env.stop()
        self.assertFalse(any("/stop?" in c[1] or c[0] == "DELETE" for c in self.api.calls))

    async def test_stop_without_delete_and_then_delete(self):
        await self.env.start()
        await self.env.stop(False)
        self.assertTrue(self.api.exists)
        self.assertFalse(self.api.running)
        await self.env.stop(True)
        self.assertFalse(self.api.exists)
        await self.env.stop(True)

    async def test_original_verifier_directory_transferred_without_rewrite(self):
        await self.env.start()
        tests = self.root / "tests"
        tests.mkdir()
        original = b"#!/bin/bash\nuvx original arguments\n"
        (tests / "test.sh").write_bytes(original)
        await self.env.upload_dir(tests, "/tests")
        upload = next(c for c in self.api.calls if c[0] == "PUT")
        self.assertTrue(upload[1].endswith("path=%2Ftests"))
        with tarfile.open(fileobj=io.BytesIO(upload[2])) as archive:
            self.assertEqual(archive.extractfile("./test.sh").read(), original)

    async def test_download_stock_verifier_logs_strips_archive_root(self):
        await self.env.start()
        self.api.download = archive_bytes([("verifier/reward.txt", b"1"), ("verifier/sub/result.json", b"{}")])
        target = self.root / "logs"
        await self.env.download_dir("/logs/verifier", target)
        self.assertEqual((target / "reward.txt").read_bytes(), b"1")
        self.assertTrue((target / "sub/result.json").exists())

    async def test_download_rejects_traversal_and_duplicate_before_extract(self):
        await self.env.start()
        for entries in ([('verifier/../../escape', b'x')], [('verifier/x', b'1'), ('verifier/x', b'2')]):
            self.api.download = archive_bytes(entries)
            with self.assertRaises(plugin.PouchError):
                await self.env.download_dir("/logs/verifier", self.root / "out")
        self.assertFalse((self.root / "escape").exists())

    async def test_download_file_is_private_and_rejects_output_symlink(self):
        await self.env.start()
        self.api.download = archive_bytes([("reward.txt", b"1")])
        target = self.root / "reward"
        await self.env.download_file("/logs/verifier/reward.txt", target)
        self.assertEqual(target.stat().st_mode & 0o777, 0o600)
        link = self.root / "link"
        link.symlink_to(target)
        with self.assertRaises(OSError):
            await self.env.download_file("/logs/verifier/reward.txt", link)


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def serve(self, response, keep_open=False):
        temp = tempfile.TemporaryDirectory(prefix="hb-", dir="/tmp")
        self.addCleanup(temp.cleanup)
        path = str(Path(temp.name) / "api.sock")
        clients = []
        async def handler(reader, writer):
            clients.append(writer)
            await reader.readuntil(b"\r\n\r\n")
            writer.write(response)
            await writer.drain()
            if not keep_open:
                writer.close()
        server = await asyncio.start_unix_server(handler, path)
        self.addAsyncCleanup(server.wait_closed)
        self.addCleanup(server.close)
        self.addCleanup(lambda: [writer.close() for writer in clients])
        return plugin.PouchAPI(path)

    async def test_content_length_returns_without_waiting_for_eof(self):
        api = await self.serve(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}", True)
        self.assertEqual(await api.request("GET", "/version", timeout=0.3), b"{}")

    async def test_chunked_and_trailers(self):
        api = await self.serve(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\nX: y\r\n\r\n", True)
        self.assertEqual(await api.request("GET", "/version", timeout=0.3), b"{}")

    async def test_close_delimited_exec_101(self):
        raw = frame("done")
        api = await self.serve(b"HTTP/1.1 101 UPGRADED\r\nConnection: Upgrade\r\n\r\n" + raw)
        self.assertEqual(plugin.demux(await api.request("POST", "/exec/x/start")), ("done", ""))

    async def test_truncated_and_over_bound_rejected(self):
        api = await self.serve(b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\n{}")
        with self.assertRaises(asyncio.IncompleteReadError):
            await api.request("GET", "/version")
        api = await self.serve(b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\n12345678")
        with self.assertRaises(plugin.PouchError):
            await api.request("GET", "/version", max_bytes=7)

    async def test_deadline_closes_waiting_connection(self):
        api = await self.serve(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n", True)
        with self.assertRaises(asyncio.TimeoutError):
            await api.request("GET", "/version", timeout=0.02)

    async def test_error_does_not_expose_daemon_body(self):
        api = await self.serve(b"HTTP/1.1 500 ERR\r\nContent-Length: 6\r\n\r\nSECRET")
        with self.assertRaises(plugin.PouchHTTPError) as caught:
            await api.request("GET", "/version")
        self.assertNotIn("SECRET", str(caught.exception))


class InstalledHarborContractTests(unittest.TestCase):
    def test_real_harbor_import_and_instantiation_when_available(self):
        try:
            from rollout_manager.terminal_bench import harbor_pouch_environment as real
            from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
            from harbor.models.trial.paths import TrialPaths
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            if os.environ.get("HARBOR_REQUIRE_REAL") == "1":
                self.fail("Real Harbor import prerequisite unavailable: " + type(error).__name__)
            self.skipTest("Harbor distribution/dependencies not installed; contract fixtures only")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "environment"
            environment.mkdir()
            instance = real.PouchEnvironment(
                environment_dir=environment, environment_name="task", session_id="contract",
                trial_paths=TrialPaths(trial_dir=root / "trial"),
                task_env_config=EnvironmentConfig(docker_image="example/task:pinned", cpus=1,
                    memory_mb=2048, storage_mb=10240),
                network_policy=NetworkPolicy(network_mode="no-network"))
            self.assertEqual(instance.type(), "pouch")
            self.assertFalse(instance._create_attempted)
            self.assertEqual(instance._effective_cpus, 1)


class InstalledHarborExecCallbackTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        try:
            from rollout_manager.terminal_bench import harbor_pouch_environment as real
            from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
            from harbor.models.trial.paths import TrialPaths
        except (ImportError, importlib.metadata.PackageNotFoundError) as error:
            if os.environ.get("HARBOR_REQUIRE_REAL") == "1":
                self.fail("Real Harbor callback prerequisite unavailable: " + type(error).__name__)
            self.skipTest("Real Harbor distribution/dependencies unavailable")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        environment = root / "environment"
        environment.mkdir()
        self.env = real.PouchEnvironment(
            environment_dir=environment, environment_name="task", session_id="callback-contract",
            trial_paths=TrialPaths(trial_dir=root / "trial"),
            task_env_config=EnvironmentConfig(docker_image="example/task:pinned", cpus=1,
                memory_mb=2048, storage_mb=10240),
            network_policy=NetworkPolicy(network_mode="no-network"))
        self.api = self.env.api = FakeAPI(self.env.owner, real)
        self.addAsyncCleanup(self.env.stop)
        await self.env.start()

    async def test_real_callback_factory_without_scope_preserves_exec(self):
        self.assertIsNone(self.env._output_callback())
        result = await self.env.exec("printf fixture", timeout_sec=1)
        self.assertEqual((result.stdout, result.stderr, result.return_code), ("out", "err", 0))
        self.assertFalse(self.env._broken)
        self.assertTrue(self.api.running)

    async def test_real_scoped_callback_receives_stdout_and_stderr(self):
        received = []
        async def receive(text, stream):
            received.append((text, stream))
        before = len([row for row in self.api.calls if row[1] == "/exec/" + EID + "/start"])
        with self.env.scoped_output_callback(receive):
            result = await self.env.exec("printf fixture", timeout_sec=1)
        after = len([row for row in self.api.calls if row[1] == "/exec/" + EID + "/start"])
        self.assertEqual(after - before, 1)
        self.assertEqual(received, [("out", "stdout"), ("err", "stderr")])
        self.assertEqual(result.return_code, 0)
        self.assertIsNone(self.env._output_callback())
        self.assertFalse(self.env._broken)

    async def test_real_nested_scopes_are_restored_after_native_exec(self):
        outer, inner = [], []
        async def first(text, stream):
            outer.append((text, stream))
        async def second(text, stream):
            inner.append((text, stream))
        with self.env.scoped_output_callback(first):
            with self.env.scoped_output_callback(second):
                await self.env.exec("first fixture", timeout_sec=1)
            await self.env.exec("second fixture", timeout_sec=1)
        pair = [("out", "stdout"), ("err", "stderr")]
        self.assertEqual(outer, pair + pair)
        self.assertEqual(inner, pair)
        self.assertIsNone(self.env._output_callback())


if __name__ == "__main__":
    unittest.main()
