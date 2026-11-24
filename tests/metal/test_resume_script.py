"""scripts/metal/resume.py's judge: an interrupted run passes only on the bytes of a whole one."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "metal"))

import resume  # noqa: E402

ADAPTER, CHECKPOINT = "adapter_model.safetensors", "checkpoint-3/adapter_model.safetensors"
WHOLE = {"files": {ADAPTER: "a", CHECKPOINT: "b"}, "losses": {"0": 1.0, "1": 0.5}}


def _arm(**changed):
    return {"interrupted": True, "continued": 0, **WHOLE, **changed}


def test_the_same_bytes_pass():
    assert resume.compare(WHOLE, _arm()) == []


def test_lost_state_is_caught_by_the_bytes():
    lost = _arm(files={ADAPTER: "a", CHECKPOINT: "x"}, losses={"0": 1.0, "1": 0.6})
    assert resume.compare(WHOLE, lost) == [f"{CHECKPOINT} differs", "loss at step 1 differs"]


def test_a_run_never_interrupted_is_not_a_pass():
    assert resume.compare(WHOLE, _arm(interrupted=False)) == ["never interrupted"]


def test_no_checkpoint_in_common_is_not_a_pass():
    assert resume.compare(WHOLE, _arm(files={ADAPTER: "a"})) == [
        "no checkpoint in common to compare"
    ]
