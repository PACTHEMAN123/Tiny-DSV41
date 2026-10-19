"""Real Linux native socket and compiler-path regressions; no Torch/CUDA."""
import json
import multiprocessing
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rollout_manager.backends import vllm_lora_ipc as ipc


def _spawn_alias_receipt(output):
    ipc.configure_lora_ipc()
    path = os.environ['TMPDIR'] + '/symm_mem-' + str(os.getpid())
    with socket.socket(socket.AF_UNIX) as listener:
        listener.bind(path)
    Path(path).unlink()
    Path(output).write_text(json.dumps({'alias':os.environ['TMPDIR'],
        'real':str(Path(os.environ['TMPDIR']).resolve(strict=True)),
        'retained':len(ipc._COMPILER_DIRECTORIES)}))


class CompilerPathValidationTests(unittest.TestCase):
    def test_unset_compiler_tmp_uses_owned_directory_without_changing_parent_cwd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve();(root/'tmp').mkdir(mode=0o700)
            cwd=Path.cwd();cached=tempfile.tempdir
            try:
                os.chdir(root)
                with patch.dict(os.environ,{'DSV41_LORA_IPC_DIR':'tmp'}):
                    os.environ.pop('TMPDIR',None)
                    self.assertEqual(ipc.configure_lora_ipc(),'tmp')
                    self.assertEqual(Path(os.environ['TMPDIR']).resolve(strict=True),root/'tmp')
                    self.assertTrue(Path(os.environ['TMPDIR']).is_absolute())
                    self.assertEqual(Path.cwd(),root)
                    self.assertEqual(tempfile.tempdir,'tmp')
            finally:os.chdir(cwd);tempfile.tempdir=cached


@unittest.skipUnless(sys.platform == 'linux', 'real proc FD/socket behavior requires Linux')
class NativeLoRAIPCPathTests(unittest.TestCase):
    def test_retained_alias_handles_native_socket_nested_child_and_spawn_without_new_fd(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/('long-owned-run-'*9);root.mkdir()
            (root/'tmp').mkdir(mode=0o700)
            compiler=root/'compiler';compiler.mkdir(mode=0o700)
            nested=root/'nested';nested.mkdir()
            cwd=Path.cwd();cached=tempfile.tempdir;before=set(ipc._COMPILER_DIRECTORIES)
            try:
                os.chdir(root)
                with patch.dict(os.environ,{'DSV41_LORA_IPC_DIR':'tmp','TMPDIR':str(compiler)}):
                    ipc.configure_lora_ipc();alias=os.environ['TMPDIR']
                    self.assertTrue(alias.startswith(f'/proc/{os.getpid()}/fd/'))
                    self.assertEqual(Path(alias).resolve(strict=True),compiler)
                    count=len(ipc._COMPILER_DIRECTORIES)
                    ipc.configure_lora_ipc()
                    self.assertEqual(os.environ['TMPDIR'],alias)
                    self.assertEqual(len(ipc._COMPILER_DIRECTORIES),count)
                    path=alias+'/symm_mem-'+str(os.getpid())
                    with socket.socket(socket.AF_UNIX) as listener:listener.bind(path)
                    Path(path).unlink()
                    subprocess.run([sys.executable,'-c','from pathlib import Path;import os; (Path(os.environ["TMPDIR"])/"compiler-child").write_text("ok")'],cwd=nested,check=True)
                    self.assertEqual((compiler/'compiler-child').read_text(),'ok')
                    output=root/'spawn.json'
                    child=multiprocessing.get_context('spawn').Process(target=_spawn_alias_receipt,args=(str(output),))
                    child.start();child.join(timeout=20)
                    if child.is_alive():child.terminate();child.join(timeout=5)
                    self.assertEqual(child.exitcode,0)
                    self.assertEqual(json.loads(output.read_text()),{'alias':alias,'real':str(compiler),'retained':0})
            finally:
                os.chdir(cwd);tempfile.tempdir=cached
                for key in set(ipc._COMPILER_DIRECTORIES)-before:
                    os.close(ipc._COMPILER_DIRECTORIES.pop(key)[1])

    def test_shared_and_outside_compiler_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as external:
            root=Path(temporary);(root/'tmp').mkdir(mode=0o700)
            shared=root/'shared';shared.mkdir();shared.chmod(0o755)
            cwd=Path.cwd()
            try:
                os.chdir(root)
                for invalid in (str(shared),external):
                    with self.subTest(path=invalid),patch.dict(os.environ,{'DSV41_LORA_IPC_DIR':'tmp','TMPDIR':invalid}):
                        with self.assertRaises(ValueError):ipc.configure_lora_ipc()
            finally:os.chdir(cwd)


if __name__=='__main__':unittest.main()
