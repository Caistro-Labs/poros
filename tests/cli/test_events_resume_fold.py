"""What reads events.jsonl folds a resume marker: the watch
screen, read forward a chunk at a time; the run's page; its checkpoints' losses; and its progress
line each see every step once, the resumed run's values where it took a step again."""

from __future__ import annotations

import json

from poros.core.events import current


def _step(i, loss, tokens=None):
    data = {"loss": loss, "s_per_step": 1.0, "vram_allocated_mb": 1000.0}
    if tokens is not None:
        data["tokens"] = tokens
    return {"event_type": "step", "timestamp": 1000.0 + i, "step": i, "data": data}


INTERRUPTED = [_step(i, 10.0 - i) for i in range(6)] + [
    {"event_type": "eval", "step": 2, "data": {"eval_loss": 5.0, "rows": 4}},
    {"event_type": "eval", "step": 4, "data": {"eval_loss": 4.0, "rows": 4}}]
RESUMED = [{"event_type": "resume", "step": 3, "data": {"checkpoint": "out/checkpoint-2"}}] + [
    _step(i, 20.0 + i) for i in range(3, 6)]


def _write(path, evs):
    with open(path, "a") as f:
        f.writelines(json.dumps(e) + "\n" for e in evs)


def test_current_drops_the_steps_and_scores_a_resume_takes_again():
    kept = current(INTERRUPTED + RESUMED)
    assert [(e["event_type"], e["step"]) for e in kept] == [
        ("step", 0), ("step", 1), ("step", 2), ("eval", 2), ("resume", 3),
        ("step", 3), ("step", 4), ("step", 5)]


def test_the_watch_screen_folds_a_resume_read_a_chunk_at_a_time(tmp_path):
    from poros.cli.ui.screens import TrainState
    from poros.cli.verbs.runs_live import _Events

    st = TrainState(model="m", output_dir=str(tmp_path), total=6)
    reader = _Events(st)
    _write(tmp_path / "events.jsonl", INTERRUPTED)
    reader.read()
    assert st.step == 6 and st.losses == [10.0, 9.0, 8.0, 7.0, 6.0, 5.0]
    _write(tmp_path / "events.jsonl", RESUMED[:1])  # resumed, no step since
    reader.read()
    assert st.step == 3 and st.losses == [10.0, 9.0, 8.0] and [e for e, _ in st.evals] == [2]
    _write(tmp_path / "events.jsonl", RESUMED[1:])
    reader.read()
    assert st.step == 6 and st.losses == [10.0, 9.0, 8.0, 23.0, 24.0, 25.0]
    assert st.loss == 25.0 and len(st.step_times) == 6


def test_the_runs_page_checkpoints_and_progress_read_the_record_as_it_stands(tmp_path):
    from poros import runs
    from poros.cli.verbs import runs_report

    _write(tmp_path / "events.jsonl", INTERRUPTED + RESUMED[:1])
    assert runs._progress(str(tmp_path)) == (3, None)
    _write(tmp_path / "events.jsonl", RESUMED[1:])
    assert runs._progress(str(tmp_path)) == (6, 25.0)
    facts = runs_report.facts(str(tmp_path), str(tmp_path))
    assert facts["losses"] == [(1, 10.0), (2, 9.0), (3, 8.0), (4, 23.0), (5, 24.0), (6, 25.0)]
    assert facts["held_out"] == [5.0] and runs_report.evals(str(tmp_path)) == [(2, 5.0)]


def test_checkpoints_read_each_steps_loss_as_it_stands(tmp_path):
    from poros.cli.verbs.checkpoints import _losses

    # an interrupted run that wrote step 5 but whose resume stopped after step 4
    _write(tmp_path / "events.jsonl", INTERRUPTED + RESUMED[:3])
    assert _losses(tmp_path) == {1: 10.0, 2: 9.0, 3: 8.0, 4: 23.0, 5: 24.0}


def test_an_error_the_resumed_run_got_past_is_folded_out():
    """An error event's step is the update's index from 0, as a step
    event's, and a resume before it supersedes it with the steps it takes again."""
    stopped = [_step(i, 9.0) for i in range(3)] + [
        {"event_type": "error", "step": 3, "data": {"code": "E_STEP_CHECK"}}]
    resumed = [{"event_type": "resume", "step": 2, "data": {}}, _step(2, 1.0), _step(3, 0.5)]
    kept = current(stopped + resumed)
    assert [e["event_type"] for e in kept].count("error") == 0
    assert [(e["event_type"], e["step"]) for e in kept][-3:] == [
        ("resume", 2), ("step", 2), ("step", 3)]
