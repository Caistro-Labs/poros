"""scripts/metal/mutations.py on the CPU: its faults reach their sites, and its judges see them."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.conftest import needs_ml

ROOT = Path(__file__).resolve().parents[2]


@needs_ml
@pytest.mark.skipif(sys.platform == "darwin", reason="a Mac needs a real store: tiny slabs cache")
def test_the_faults_land_as_predicted_on_a_tiny_model_and_mispredictions_do_not_stop_it(tmp_path):
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "metal" / "mutations.py"), "--model", "tiny",
         "--g", "1", "--steps", "3", "--learning-rate", "1e-3", "--warmup-steps", "0",
         "--only", "bit-last,stopped-short,secret-ram,nan", "--out", str(tmp_path)],
        check=False, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT), "POROS_COMMIT": "abc",
             "POROS_HOME": str(tmp_path / "home")},
    )
    assert done.returncode == 0, done.stderr[-2000:]
    faults = {f["fault"]: f for f in json.loads((tmp_path / "mutations.json").read_text())}
    assert faults["secret-ram"]["seen"]["note"]
    for name in ("bit-last", "stopped-short", "secret-ram", "nan"):
        assert faults[name]["misprediction"] == [], faults[name]
