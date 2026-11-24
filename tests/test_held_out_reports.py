"""What a run's reports read from its events agrees with what the run says: after a resume on
other held-out rows the trainer drops its start, and so does every reader of the events
(runs_report.held_out, which `poros runs`, run info and watch's `learned`, the live card, the
listings and the report call): no start, no "x% lower". The same rows keep it; an events file
from before eval events named their rows reads as it did."""

from __future__ import annotations

import json

import pytest

from tests.test_classic_evals import _held
from tests.test_eval_start_rows import _other, _resume, _stop_at_3


@pytest.fixture
def ml():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return torch


def test_a_resume_on_other_rows_reads_with_no_start_in_every_report(ml, tmp_path):
    from poros.cli.verbs import runs_report

    torch = ml
    other = tmp_path / "other"
    _stop_at_3(torch, other)
    out = _resume(torch, other, _other(torch))
    held = runs_report.held_out(str(other))
    assert held["eval_loss_start"] is None and held["eval_loss"] == out["eval_loss"]
    said = runs_report.learned(held)
    assert " to " not in said and "lower" not in said and "higher" not in said

    same = tmp_path / "same"
    _stop_at_3(torch, same)
    out = _resume(torch, same, _held(torch))
    held = runs_report.held_out(str(same))
    assert held["eval_loss_start"] == out["eval_loss_start"]
    assert held["eval_loss"] == out["eval_loss"]
    assert "lower" in runs_report.learned(held) or "higher" in runs_report.learned(held)

    # an events file from before eval events named their rows: read as it was
    path = other / "events.jsonl"
    lines = []
    for line in path.read_text().splitlines():
        ev = json.loads(line)
        (ev.get("data") or {}).pop("held", None)
        lines.append(json.dumps(ev))
    path.write_text("\n".join(lines) + "\n")
    assert runs_report.held_out(str(other))["eval_loss_start"] is not None
