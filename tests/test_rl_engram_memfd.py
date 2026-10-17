"""Linux builds may omit Python's optional memfd bindings and seal constants."""
from contextlib import contextmanager
import ctypes
import errno
import fcntl
import mmap
import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from dsv41_train.rl import engram_state as state
from dsv41_train.rl.family import load_model_config


@contextmanager
def without_bindings():
    with patch.dict(os.__dict__), patch.dict(fcntl.__dict__):
        for name in ("memfd_create", "MFD_CLOEXEC", "MFD_ALLOW_SEALING"):
            os.__dict__.pop(name, None)
        for name in ("F_ADD_SEALS", "F_GET_SEALS", "F_SEAL_WRITE", "F_SEAL_SHRINK",
                     "F_SEAL_GROW", "F_SEAL_SEAL"):
            fcntl.__dict__.pop(name, None)
        yield


def inspect_spawned_baseline(baseline, sender):
    try:
        result = {}
        for name, fd in baseline._fds.items():
            private = baseline.view(name)
            before = private[0, 0].item()
            private[0, 0] += 1
            result[name] = (os.fstat(fd).st_ino, state.memfd_seals(fd),
                            before, baseline.view(name)[0, 0].item())
        sender.send(result)
    finally:
        sender.close()


class MemfdTests(unittest.TestCase):
    def test_native_preferred_and_failure_not_bypassed(self):
        with patch.object(sys, "platform", "linux"), without_bindings(), \
                patch.object(os, "memfd_create", return_value=47, create=True) as native, \
                patch.object(ctypes, "CDLL") as libc:
            self.assertEqual(state.create_memfd("test"), 47)
            native.assert_called_once_with("test", 3)
            native.side_effect = OSError(errno.EPERM, "denied")
            with self.assertRaises(OSError):
                state.create_memfd("test")
            libc.assert_not_called()

    def test_libc_signature_errno_and_missing_symbol(self):
        function = Mock(return_value=47)
        with patch.object(sys, "platform", "linux"), without_bindings(), \
                patch.object(ctypes, "CDLL", return_value=Mock(memfd_create=function)) as libc:
            self.assertEqual(state.create_memfd("test"), 47)
            libc.assert_called_once_with(None, use_errno=True)
            function.assert_called_once_with(b"test", 3)
            self.assertEqual(function.argtypes, [ctypes.c_char_p, ctypes.c_uint])
            self.assertIs(function.restype, ctypes.c_int)
            function.return_value = -1
            with patch.object(ctypes, "get_errno", return_value=errno.EPERM), self.assertRaises(OSError) as error:
                state.create_memfd("test")
            self.assertEqual(error.exception.errno, errno.EPERM)
            libc.return_value = object()
            with self.assertRaisesRegex(RuntimeError, "no memfd_create"):
                state.create_memfd("test")

    def test_missing_seals_fail_closed(self):
        with without_bindings(), patch.object(fcntl, "fcntl", side_effect=[0, 7]) as call:
            with self.assertRaisesRegex(RuntimeError, "missing required kernel seals"):
                state.seal_memfd(47)
            self.assertEqual(call.call_args_list, [((47, 1033, 15),), ((47, 1034),)])

    @contextmanager
    def baseline_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Path(__file__).resolve().parents[1] / "examples/rl/tiny-dsv41/config.json"
            (root / "config.json").write_bytes(config.read_bytes())
            yield root, load_model_config(root)

    def test_baseline_error_closes_current_and_previous_fds(self):
        descriptors = []
        def create(name):
            with tempfile.TemporaryFile() as handle:
                fd = os.dup(handle.fileno())
            descriptors.append(fd)
            return fd
        with self.baseline_inputs() as (root, config), patch.object(sys, "platform", "linux"), \
                patch.object(state, "create_memfd", side_effect=create), \
                patch.object(state, "seal_memfd", side_effect=[15, OSError(errno.EPERM, "denied")]):
            with self.assertRaises(OSError):
                state.EngramBaseline.create(root, config, torch.float32, True, 37)
        self.assertEqual(len(descriptors), 2)
        for fd in descriptors:
            with self.assertRaises(OSError) as error:
                os.fstat(fd)
            self.assertEqual(error.exception.errno, errno.EBADF)

    @unittest.skipUnless(sys.platform == "linux", "requires Linux memfd and kernel seals")
    def test_real_libc_seals_cow_and_spawn(self):
        with self.baseline_inputs() as (root, config), without_bindings():
            baseline = state.EngramBaseline.create(root, config, torch.bfloat16, True, 37)
            for name, fd in baseline._fds.items():
                self.assertEqual(state.memfd_seals(fd) & 15, 15)
                self.assertTrue(fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC)
                size = os.fstat(fd).st_size
                for write in (lambda: os.pwrite(fd, b"x", 0), lambda: os.ftruncate(fd, size - 1),
                              lambda: os.ftruncate(fd, size + 1), lambda: mmap.mmap(fd, size, access=mmap.ACCESS_WRITE)):
                    with self.assertRaises(OSError) as error:
                        write()
                    self.assertEqual(error.exception.errno, errno.EPERM)
                first, second = baseline.view(name), baseline.view(name)
                original = second.clone()
                first.add_(1)
                torch.testing.assert_close(second, original, rtol=0, atol=0)
            context = multiprocessing.get_context("spawn")
            receiver, sender = context.Pipe(duplex=False)
            worker = context.Process(target=inspect_spawned_baseline, args=(baseline, sender))
            worker.start()
            sender.close()
            try:
                self.assertTrue(receiver.poll(30), "spawn child did not return the FD receipt")
                receipt = receiver.recv()
                worker.join(30)
                self.assertEqual(worker.exitcode, 0)
                for name, fd in baseline._fds.items():
                    before = baseline.view(name)[0, 0].item()
                    self.assertEqual(receipt[name], (os.fstat(fd).st_ino, 15, before, before))
            finally:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(10)
                receiver.close()


if __name__ == "__main__":
    unittest.main()
