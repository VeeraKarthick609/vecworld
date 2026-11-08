"""A failing card_limit test gives its GPU memory back (Task 10a re-review N3; gpu, self-contained).

pytest keeps a failed test's traceback, and with it the frame's locals: the walks' painter and depth model, 3.3 GB of
the 4 GB card. Because card_limit tests run first, one flaky timing failure would then run every later GPU module out of
memory. tests/gpu_release.py (loaded by conftest) clears those frames when such a test fails. The proof is a toy
session in a subprocess (the parent's own session must not fail on purpose): a card_limit test allocates 256 MiB in a
local and fails, and the next test asserts it came back.
"""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

TOY = textwrap.dedent('''
    import pytest
    import torch


    @pytest.mark.card_limit
    def test_a_fails_holding_gpu_memory():
        hog = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
        assert hog.numel() == 0, "deliberate failure"


    def test_b_memory_came_back():
        assert torch.cuda.memory_allocated() < 64 * 2**20, torch.cuda.memory_allocated()
        assert torch.cuda.memory_reserved() < 64 * 2**20, torch.cuda.memory_reserved()
''')


def _toy_session(tmp_path, with_hook: bool) -> str:
    (tmp_path / "test_toy.py").write_text(TOY)
    tests_dir = str(Path(__file__).parent)
    env = {**os.environ, "PYTHONPATH": tests_dir + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:warnings"]
    if with_hook:
        cmd += ["-p", "gpu_release"]
    r = subprocess.run(cmd + ["test_toy.py"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180)
    return r.stdout + r.stderr


def test_the_toy_would_leak_without_the_hook(tmp_path):
    """The proof means something: without the hook the failed test's traceback keeps the memory."""
    out = _toy_session(tmp_path, with_hook=False)
    assert "test_b_memory_came_back" in out and "2 failed" in out, out


def test_a_failed_card_limit_test_releases_its_gpu_memory(tmp_path):
    out = _toy_session(tmp_path, with_hook=True)
    assert "1 failed, 1 passed" in out and "deliberate failure" in out, out
