"""CPU-only safety checks; Linux also exercises real process supervision."""
from __future__ import annotations

import copy
import ctypes
import errno
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from scripts import gpu_test_gate as gate


def gpu(used=100, free=None, uuid="GPU-a", index=0):
    return {"index": index, "uuid": uuid, "pci_bus_id": f"{index}:00",
            "total_mib": 80000, "used_mib": used,
            "free_mib": 79900 if free is None else free,
            "utilization_percent": 0, "processes": [], "mig_mode": "Disabled"}


POLICY = {"tolerance_mib": 256, "max_baseline_used_mib": 1024,
          "min_free_mib": 1024, "max_utilization_percent": 5}
MACHINE = {"hostname": "test", "uid": 1, "boot_id": "test-boot", "pid_namespace": "test-ns"}


def proc(pid, ppid=0, start=100, state="S"):
    return {"pid": pid, "ppid": ppid, "start_ticks": start, "uid": os.getuid(),
            "state": state, "pgid": 100, "sid": 100, "comm": "test", "argv": "test"}


class GpuPolicyTests(unittest.TestCase):
    def test_empty_pid_list_does_not_waive_large_baseline(self):
        current = gpu(used=70000, free=10000)
        issues = gate.memory_issues([current], [current], POLICY, initial=True)
        self.assertTrue(any("even if PID list is empty" in issue for issue in issues))

    def test_used_and_free_are_independently_gated(self):
        for current in (gpu(used=357), gpu(free=79643)):
            self.assertTrue(gate.memory_issues([current], [gpu()], POLICY))
        self.assertFalse(gate.memory_issues([gpu(used=356, free=79644)], [gpu()], POLICY))

    def test_baseline_is_not_allowed_to_drift_upward(self):
        original = [gpu()]
        self.assertFalse(gate.memory_issues([gpu(used=300, free=79700)], original, POLICY))
        self.assertTrue(gate.memory_issues([gpu(used=500, free=79500)], original, POLICY))

    def test_busy_gpu_and_mig_block(self):
        for change in ({"processes": [{"pid": "42"}]}, {"utilization_percent": 6},
                       {"mig_mode": "Enabled"}, {"total_mib": 40000}, {"free_mib": 500}):
            current = {**gpu(), **change}
            self.assertTrue(gate.memory_issues([current], [gpu()], POLICY), change)

    def test_unknown_other_gpu_is_recorded_but_not_selected(self):
        self.assertFalse(gate.memory_issues([gpu(), gpu(used=75000, uuid="GPU-b", index=1)], [gpu()], POLICY))

    def test_missing_selected_gpu_blocks(self):
        self.assertTrue(gate.memory_issues([], [gpu()], POLICY))

    def test_select_exact_uuid_or_index_and_reject_duplicates(self):
        snapshot = {"gpus": [gpu(), gpu(uuid="GPU-b", index=1)]}
        self.assertEqual([g["uuid"] for g in gate.select_gpus(snapshot, "1,GPU-a")], ["GPU-b", "GPU-a"])
        for selection in ("0,GPU-a", "GPU-", "2", ""):
            with self.assertRaises(gate.GateError):
                gate.select_gpus(snapshot, selection)

    def test_nvml_xml_includes_non_compute_processes(self):
        xml = ('<nvidia_smi_log><gpu><uuid>GPU-a</uuid><mig_mode><current_mig>Disabled</current_mig>'
               '</mig_mode><processes><process_info><pid>99</pid><type>G</type></process_info>'
               '</processes></gpu></nvidia_smi_log>')
        result = gate.parse_gpu_snapshot("0, GPU-a, 00:01, 80000, 100, 79900, 0\n", xml)
        self.assertEqual(result[0]["processes"], [{"pid": "99", "type": "G"}])
        with self.assertRaises(gate.GateError):
            gate.parse_gpu_snapshot("0, GPU-a, 00:01, 80000, N/A, 79900, 0\n", xml)

    def test_unavailable_nvml_processes_are_not_an_empty_list(self):
        for node in ("", "<processes>N/A</processes>"):
            xml = f"<nvidia_smi_log><gpu><uuid>GPU-a</uuid>{node}</gpu></nvidia_smi_log>"
            with self.assertRaisesRegex(gate.GateError, "inventory unavailable"):
                gate.parse_gpu_snapshot("0, GPU-a, 00:01, 80000, 100, 79900, 0\n", xml)


class OwnershipTests(unittest.TestCase):
    def test_proc_stat_with_spaces_and_parentheses(self):
        fields = ["S", "9", "12", "12"] + ["0"] * 15 + ["123456"] + ["0"] * 8
        record = gate.parse_proc_stat("17 (name ) contains ( spaces) " + " ".join(fields))
        self.assertEqual((record["pid"], record["ppid"], record["pgid"], record["sid"], record["start_ticks"]),
                         (17, 9, 12, 12, 123456))

    def test_ancestry_not_pgid_controls_ownership(self):
        supervisor = proc(100)
        records = [supervisor, proc(101, 100), proc(102, 101), proc(200, 1)]
        self.assertEqual([p["pid"] for p in gate.owned_descendants(supervisor, records)], [101, 102])
        with self.assertRaises(gate.GateError):
            gate.owned_descendants(proc(100, start=99), records)

    def test_pid_reused_before_signal_is_never_signalled(self):
        with mock.patch.object(gate, "read_process", return_value=proc(101, start=101)), \
             mock.patch.object(os, "pidfd_open", create=True) as opener:
            self.assertFalse(gate.signal_verified(proc(101))["sent"])
            opener.assert_not_called()

    def test_pid_reuse_during_pidfd_open_is_never_signalled(self):
        with mock.patch.object(gate, "read_process", side_effect=[proc(101), proc(101, start=101)]), \
             mock.patch.object(os, "pidfd_open", return_value=123, create=True), \
             mock.patch.object(os, "close") as closer, \
             mock.patch.object(signal, "pidfd_send_signal", create=True) as sender:
            self.assertFalse(gate.signal_verified(proc(101))["sent"])
            sender.assert_not_called()
            closer.assert_called_once_with(123)

    def test_verified_identity_signals_pidfd_not_process_group(self):
        with mock.patch.object(gate, "read_process", return_value=proc(101)), \
             mock.patch.object(os, "pidfd_open", return_value=123, create=True), \
             mock.patch.object(os, "close"), \
             mock.patch.object(signal, "pidfd_send_signal", create=True) as sender, \
             mock.patch.object(os, "killpg") as group_signal:
            self.assertTrue(gate.signal_verified(proc(101))["sent"])
            sender.assert_called_once_with(123, signal.SIGTERM)
            group_signal.assert_not_called()

    def test_exclusive_lock_rejects_second_invocation(self):
        with tempfile.TemporaryDirectory() as temporary:
            with gate.exclusive_lock(Path(temporary)):
                with self.assertRaises(gate.GateError):
                    with gate.exclusive_lock(Path(temporary)):
                        self.fail("second gate acquired lock")


class PidfdBackendTests(unittest.TestCase):
    def test_standard_library_is_preferred(self):
        with mock.patch.object(os, "pidfd_open", return_value=123, create=True) as opener, \
             mock.patch.object(signal, "pidfd_send_signal", create=True) as sender, \
             mock.patch.object(gate, "_pidfd_syscall") as syscall:
            self.assertEqual(gate.pidfd_open(42), 123)
            gate.pidfd_send_signal(123, signal.SIGTERM)
            opener.assert_called_once_with(42, 0)
            sender.assert_called_once_with(123, signal.SIGTERM)
            syscall.assert_not_called()

    def test_missing_python_bindings_use_typed_pidfd_syscalls(self):
        with mock.patch.object(os, "pidfd_open", None, create=True), \
             mock.patch.object(signal, "pidfd_send_signal", None, create=True), \
             mock.patch.object(gate, "_pidfd_syscall", side_effect=[123, 0]) as syscall:
            self.assertEqual(gate.pidfd_open(42), 123)
            gate.pidfd_send_signal(123, signal.SIGTERM)
        calls = syscall.call_args_list
        self.assertEqual(calls[0].args[0], 434)
        self.assertEqual([arg.value for arg in calls[0].args[1:]], [42, 0])
        self.assertEqual(calls[1].args[0], 424)
        self.assertEqual([arg.value for arg in calls[1].args[1:]], [123, signal.SIGTERM, None, 0])

    def test_supported_architectures_call_libc_and_preserve_errno(self):
        for architecture in ("x86_64", "aarch64"):
            libc = mock.Mock()
            libc.syscall.return_value = 7
            with mock.patch.object(sys, "platform", "linux"), \
                 mock.patch.object(gate.platform, "machine", return_value=architecture), \
                 mock.patch.object(ctypes, "CDLL", return_value=libc):
                self.assertEqual(gate._pidfd_syscall(434, ctypes.c_int(42), ctypes.c_uint(0)), 7)
                self.assertIs(libc.syscall.restype, ctypes.c_long)
                libc.syscall.return_value = -1
                with mock.patch.object(ctypes, "get_errno", return_value=errno.ENOSYS):
                    with self.assertRaises(OSError) as raised:
                        gate._pidfd_syscall(434, ctypes.c_int(42), ctypes.c_uint(0))
                    self.assertEqual(raised.exception.errno, errno.ENOSYS)

    def test_unsupported_abi_does_not_issue_any_syscall(self):
        with mock.patch.object(sys, "platform", "linux"), \
             mock.patch.object(gate.platform, "machine", return_value="ppc64le"), \
             mock.patch.object(ctypes, "CDLL") as library:
            with self.assertRaises(OSError):
                gate._pidfd_syscall(434, ctypes.c_int(42), ctypes.c_uint(0))
            library.assert_not_called()

    def test_backend_probe_signals_only_self_with_zero_and_closes_fd(self):
        with mock.patch.object(sys, "platform", "linux"), \
             mock.patch.object(gate, "pidfd_open", return_value=123) as opener, \
             mock.patch.object(gate, "pidfd_send_signal", side_effect=PermissionError(errno.EPERM, "denied")) as sender, \
             mock.patch.object(os, "close") as closer:
            status = gate.pidfd_backend_status()
            self.assertFalse(status["available"])
            self.assertEqual(status["errno"], errno.EPERM)
            opener.assert_called_once_with(os.getpid(), 0)
            sender.assert_called_once_with(123, 0)
            closer.assert_called_once_with(123)


class GateLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.args = gate.parser().parse_args([
            "run", "--state-dir", str(self.directory), "--gpus", "0", "--name", "case",
            "--recovery-seconds", "0.02", "--poll-seconds", "0.001", "command"])
        self.current = gpu()
        self.launches = 0

    def snapshot(self):
        return {"machine": MACHINE, "gpus": [copy.deepcopy(self.current)], "visible_processes": {"processes": []}}

    def supervisor(self, payload, case):
        self.launches += 1
        gate.write_json(case / "process-result.json", {"passed": True, "supervisor": proc(777777),
                       "observed_processes": [], "command_exit_code": 0, "remaining_processes": []})
        return {"supervisor_exit_code": 0, "interrupted": False}

    def run_case(self, supervisor=None):
        return gate.run_gate(self.args, self.snapshot, supervisor or self.supervisor, lambda: MACHINE)

    def test_two_successful_commands_preserve_original_baseline(self):
        self.assertTrue(self.run_case()["passed"])
        self.current = gpu(used=200, free=79800)
        with mock.patch.object(gate, "read_process", return_value=None):
            self.assertTrue(self.run_case()["passed"])
        state = json.loads((self.directory / "state.json").read_text())
        self.assertEqual(state["baseline"][0]["used_mib"], 100)
        self.assertEqual((state["sequence"], self.launches), (2, 2))
        self.assertTrue((self.directory / "0002-case/after.json").is_file())

    def test_high_memory_empty_pid_list_never_launches(self):
        self.current = gpu(used=70000, free=10000)
        self.assertFalse(self.run_case()["passed"])
        self.assertEqual(self.launches, 0)
        self.assertTrue((self.directory / "0001-case/before-initial.json").is_file())

    def test_query_failure_never_launches_and_latches_gate(self):
        def unavailable():
            raise gate.GateError("nvidia-smi unavailable")
        result = gate.run_gate(self.args, unavailable, self.supervisor, lambda: MACHINE)
        self.assertFalse(result["passed"])
        self.assertIn("nvidia-smi unavailable", result["error"])
        self.assertEqual(self.launches, 0)
        with self.assertRaisesRegex(gate.GateError, "latched failed"):
            self.run_case()

    def test_unrecovered_memory_blocks_future_command(self):
        def launch(payload, case):
            result = self.supervisor(payload, case)
            self.current = gpu(used=4000, free=76000)
            return result
        result = self.run_case(launch)
        self.assertFalse(result["passed"])
        self.assertFalse(result["memory_recovered"])
        self.current = gpu()
        with self.assertRaisesRegex(gate.GateError, "latched failed"):
            self.run_case()
        self.assertEqual(self.launches, 1)

    def test_timeout_stays_failed_even_if_memory_recovers(self):
        def launch(payload, case):
            self.supervisor(payload, case)
            path = case / "process-result.json"
            report = json.loads(path.read_text())
            report.update(passed=False, reason="timeout", command_exit_code=-15)
            gate.write_json(path, report)
            return {"supervisor_exit_code": 1, "interrupted": False}
        result = self.run_case(launch)
        self.assertTrue(result["memory_recovered"])
        self.assertFalse(result["passed"])
        with self.assertRaises(gate.GateError):
            self.run_case()
        self.assertEqual(self.launches, 1)

    def test_running_state_blocks_even_without_current_gpu_pids(self):
        gate.write_json(self.directory / "state.json", {"machine": MACHINE, "status": "running", "last_case": "old"})
        with self.assertRaisesRegex(gate.GateError, "latched running"):
            self.run_case()
        self.assertEqual(self.launches, 0)

    def test_previous_process_identity_still_alive_blocks(self):
        self.assertTrue(self.run_case()["passed"])
        with mock.patch.object(gate, "read_process", return_value=proc(777777)):
            with self.assertRaisesRegex(gate.GateError, "remains alive"):
                self.run_case()
        self.assertEqual(self.launches, 1)

    def test_new_host_cannot_reuse_old_gate_state(self):
        self.assertTrue(self.run_case()["passed"])
        with self.assertRaisesRegex(gate.GateError, "another host"):
            gate.run_gate(self.args, self.snapshot, self.supervisor, lambda: {**MACHINE, "boot_id": "new"})
        self.assertEqual(self.launches, 1)


PIDFD_STATUS = gate.pidfd_backend_status()


@unittest.skipUnless(PIDFD_STATUS["available"],
                     f"real Linux /proc + subreaper + pidfd lifecycle; backend: {PIDFD_STATUS}")
class LinuxProcessTests(unittest.TestCase):
    def supervise(self, code, timeout=3):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        case = Path(temporary.name)
        payload = {"case_dir": str(case), "cwd": str(case), "gpu_uuids": ["GPU-test"],
                   "command": [sys.executable, "-c", code], "timeout_seconds": timeout,
                   "terminate_wait_seconds": 2}
        gate.write_json(case / "invocation.json", payload)
        launcher = gate.launch_supervisor(case / "invocation.json", case)
        report = json.loads((case / "process-result.json").read_text())
        return case, launcher, report

    def test_normal_exit_and_captured_log(self):
        case, launcher, report = self.supervise("import os; print(os.environ['CUDA_VISIBLE_DEVICES'])")
        self.assertTrue(report["passed"])
        self.assertEqual(launcher["supervisor_exit_code"], 0)
        self.assertEqual((case / "command.log").read_text().strip(), "GPU-test")
        self.assertEqual(report["supervisor"]["pid"], report["supervisor"]["pgid"])

    def test_timeout_only_terminates_owned_child(self):
        with subprocess.Popen([sys.executable, "-c", "import time;time.sleep(15)"], start_new_session=True) as unrelated:
            try:
                _, launcher, report = self.supervise("import time;time.sleep(15)", timeout=0.3)
                self.assertEqual(report["reason"], "timeout")
                self.assertTrue(report["clean"])
                self.assertEqual(launcher["supervisor_exit_code"], 1)
                self.assertIsNone(unrelated.poll())
                self.assertNotIn(unrelated.pid, [entry["identity"]["pid"] for entry in report["signals"]])
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=3)

    def test_double_fork_escape_is_adopted_and_cleaned(self):
        code = ("import os,time\n"
                "if os.fork()==0:\n"
                " os.setsid()\n"
                " if os.fork()==0: time.sleep(15)\n"
                " os._exit(0)\n"
                "os._exit(0)\n")
        _, _, report = self.supervise(code)
        self.assertEqual(report["reason"], "residual_processes_after_success")
        self.assertTrue(report["clean"])
        self.assertTrue(any(entry["sent"] for entry in report["signals"]))


if __name__ == "__main__":
    unittest.main()
