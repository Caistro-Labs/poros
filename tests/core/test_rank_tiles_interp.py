"""The dense fused kernels at every rank's tiles against fp64 on the CPU, under Triton's
interpreter (``TRITON_INTERPRET=1``): tests/gpu/test_rank_tiles.py's checks, no GPU. Needs Triton
installed (the install on a CUDA machine has it); skipped where it is not."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

ROOT = Path(__file__).resolve().parents[2]


def test_every_rank_matches_fp64_under_the_interpreter():
    env = {**os.environ, "TRITON_INTERPRET": "1", "PYTHONPATH": str(ROOT),
           "CUDA_VISIBLE_DEVICES": ""}
    code = ("import tests.gpu.test_rank_tiles as T; T.check_all('cpu'); "
            "from poros.core import fused_tiles as F; F.PARTIAL_ELEMS = 1 << 16; "
            "[T.check_swiglu('cpu', R, d) for R in (64, 128) for d in (False, True)]; "
            "[T.check_out('cpu', R) for R in (64, 128)]; print('INTERP ok')")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=1800)
    said = out.stdout[-2000:] + out.stderr[-4000:]
    assert out.returncode == 0 and "INTERP ok" in out.stdout, said
