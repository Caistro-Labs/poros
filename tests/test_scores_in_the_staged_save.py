"""A checkpoint takes its name with its held-out scores already in it (eval.json, eval_start.json):
they are written into the staged folder before the rename, so no kill can leave a checkpoint whose
run then has no start to say. Its other files are the bytes they were."""

from __future__ import annotations

import pytest

from tests.test_classic_evals import _held, _rows, _trainer


@pytest.fixture
def ml():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return torch


def test_the_scores_are_in_the_folder_when_it_takes_its_name(ml, tmp_path, monkeypatch):
    from pathlib import Path

    from poros.core.checkpoints import SCORE_FILE
    from poros.train import checkpointing
    from poros.train.evals import START_FILE

    torch = ml
    seen = {}
    real = checkpointing.os.replace

    def rename(src, dst):
        if str(src).endswith(".tmp") and Path(src).is_dir():
            seen[Path(dst).name] = sorted(p.name for p in Path(src).iterdir()
                                          if p.name in (SCORE_FILE, START_FILE))
        return real(src, dst)

    monkeypatch.setattr(checkpointing.os, "replace", rename)
    tr = _trainer(torch, tmp_path)
    tr.train(_rows(torch), steps=7, eval_dataset=_held(torch))
    # checkpoint-2, -4, -6: scored there and saved after, each with both files before its rename
    assert seen == {f"checkpoint-{s}": sorted([SCORE_FILE, START_FILE]) for s in (2, 4, 6)}
