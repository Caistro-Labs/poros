"""cuda_expandable_segments on a real GPU: whatever the driver can map, a run can allocate after
Poros has set the allocator up (poros.train.trainer.apply_expandable_segments).

Each case is a fresh process, since the allocator setting is process-wide. On a GPU whose driver
has no CUDA VMM (some GPU-sharing layers) the check is refused and the default allocator stays;
a process that already holds CUDA memory keeps its allocator as it is, because there the check's
allocation could come from a segment it already holds and prove nothing. On a GPU that maps
expandable segments, every case passes either way.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests.conftest import checkout_env

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the CUDA caching allocator", allow_module_level=True)

CODE = """
import sys, torch
from poros.core.config import PorosConfig
from poros.train.trainer import apply_expandable_segments
held = torch.empty(1, device="cuda") if sys.argv[1] == "held" else None
apply_expandable_segments(PorosConfig(), torch.device("cuda"))
fresh = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")  # a segment of its own
print("allocated")
"""


@pytest.mark.parametrize("state", ["fresh", "held"])
def test_a_run_allocates_after_the_allocator_is_set_up(state):
    env = checkout_env({k: v for k, v in os.environ.items()
                        if k not in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")})
    proc = subprocess.run([sys.executable, "-P", "-c", CODE, state], env=env,
                          capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0 and "allocated" in proc.stdout, proc.stderr[-2000:]
