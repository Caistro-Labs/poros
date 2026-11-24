"""The next step a finished run offers is one that works: `poros eval <run>` only where the run
has rows to score. On the simulated trainer (no GPU)."""

from __future__ import annotations

import pytest

from poros import runs
from tests.cli.test_run_flow_wording import MODEL, _chats, _screen
from tests.cli.test_train_runs import home  # noqa: F401  (the simulated machine)


@pytest.mark.parametrize("rows, offered", [(20, "export"), (60, "eval")])
def test_a_run_that_held_nothing_out_is_not_offered_an_eval_that_fails(home, capsys,  # noqa: F811
                                                                       monkeypatch, rows, offered):
    """After a 20-row file (under 40 rows hold none out) it said "Continue with: poros eval
    <id>", which fails with "Nothing held out"."""
    from poros.cli.verbs import runs_report

    code, out = _screen(["train", MODEL, "--data", _chats(home, rows), "--yes", "--steps", "4"],
                        home, capsys)
    (run,) = runs.all_runs()
    assert code == 0 and f"Continue with: poros {offered} {run.id}" in out
    scored = offered == "eval"
    assert runs_report.scores_itself(run.output_dir) is scored
    assert ("No rows were held out" in out) is not scored
    assert (f"poros eval {run.id} --data FILE" in out) is not scored
    monkeypatch.setenv("POROS_AGENT", "1")  # the agent form's next follows the same rule
    assert (f"poros eval {run.id}" in runs_report.offers(run)) is scored
    assert any(x.startswith(f"poros export {run.id}") for x in runs_report.offers(run))
