"""A checkpoint's held-out scores (eval.json, eval_start.json) are written before its completion
marker and flushed with every other file: a kill after the marker, or a power cut, never leaves a
complete folder without its start score or with torn score files."""

from __future__ import annotations

import pytest

from tests.test_classic_evals import _held, _rows, _trainer


def test_the_scores_land_and_are_flushed_before_the_completion_marker(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from pathlib import Path

    from poros.core import checkpoints
    from poros.core.checkpoints import COMPLETE_MARKER, SCORE_FILE
    from poros.train.evals import START_FILE

    flushed, at_marker = set(), {}
    real_fsync = checkpoints.fsync_path

    def fsync(path):
        flushed.add(str(path))
        return real_fsync(path)

    real_write = Path.write_text

    def write_text(self, *a, **kw):
        if self.name == COMPLETE_MARKER and self.parent.name.endswith(".tmp"):
            at_marker[self.parent.name] = {
                name: (self.parent / name).is_file() and str(self.parent / name) in flushed
                for name in (SCORE_FILE, START_FILE)}
        return real_write(self, *a, **kw)

    monkeypatch.setattr(checkpoints, "fsync_path", fsync)
    monkeypatch.setattr(Path, "write_text", write_text)
    tr = _trainer(torch, tmp_path)
    tr.train(_rows(torch), steps=7, eval_dataset=_held(torch))
    assert at_marker == {f"checkpoint-{s}.tmp": {SCORE_FILE: True, START_FILE: True}
                         for s in (2, 4, 6)}, at_marker
