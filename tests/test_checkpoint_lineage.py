"""A run that resumed from a checkpoint continues that checkpoint's run. A stamped checkpoint
hands on its id. An unstamped one (written before checkpoints were stamped: an upgrade mid-run,
another tool version) has none, so the run that resumes from it records it as its parent, and
`latest` reads the two as one run instead of refusing "checkpoints of 2 runs". Two runs that
are not one lineage are still refused, and pruning deletes only what this run stamped."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from poros.core import checkpoints
from poros.core.checkpoints import lineage_of, list_checkpoints, parent_of, runs_in

A, B, NEW = "a" * 16, "b" * 16, "c" * 16


def _made(folder, step, run=None, parent=None):
    """A complete checkpoint-<step>: unstamped (an older Poros's), or stamped with ``run``."""
    d = folder / f"checkpoint-{step}"
    d.mkdir(parents=True)
    (d / checkpoints.FORMAT_MARKER).write_text("1")
    if run:
        (d / checkpoints.RUN_MARKER).write_text(run)
    if parent:
        (d / checkpoints.PARENT_MARKER).write_text(parent)
    (d / checkpoints.COMPLETE_MARKER).write_text(str(step))
    return d


def _latest(folder):
    from poros.cli.commands.train import _resolve_resume_from

    return _resolve_resume_from("latest", str(folder))


def _refused(folder):
    import typer

    with pytest.raises((typer.Exit, SystemExit)):
        _latest(folder)


def test_latest_reads_a_run_and_the_unstamped_checkpoint_it_resumed_from_as_one(tmp_path):
    """The refusal an upgrade mid-run met: resume from an unstamped checkpoint, write one new
    checkpoint, and `--resume-from latest` said "holds checkpoints of 2 runs"."""
    for step in (50, 100):
        _made(tmp_path, step)
    assert runs_in(tmp_path) == {None} and _latest(tmp_path).name == "checkpoint-100"
    run, parent = lineage_of(tmp_path / "checkpoint-100", tmp_path, NEW)
    assert (run, parent) == (NEW, "checkpoint-100")
    _made(tmp_path, 110, run, parent)
    assert runs_in(tmp_path) == {NEW}
    assert _latest(tmp_path).name == "checkpoint-110"
    # and the run goes on under its id from its own checkpoint, still naming where it came from
    assert lineage_of(tmp_path / "checkpoint-110", tmp_path, "d" * 16) == (NEW, "checkpoint-100")


def test_two_unrelated_runs_in_one_folder_are_still_refused(tmp_path):
    _made(tmp_path, 2, A)
    _made(tmp_path, 4, B)
    assert runs_in(tmp_path) == {A, B}
    _refused(tmp_path)


def test_the_refusal_says_which_checkpoints_are_which_runs(tmp_path):
    """"Holds checkpoints of 2 runs" left the step to guess: each run is named with its
    checkpoints, a run's unstamped ancestors with it, and unstamped ones no run claims apart."""
    _made(tmp_path, 100)
    for step in (110, 120):
        _made(tmp_path, step, A, "checkpoint-100")
    _made(tmp_path, 200, B)
    assert checkpoints.mixed_runs_clash(tmp_path) == (
        f"{tmp_path} holds checkpoints of 2 runs, so latest names none of them: "
        "run aaaaaaaa has checkpoint-100, checkpoint-110, checkpoint-120; "
        "run bbbbbbbb has checkpoint-200")
    other = tmp_path / "other"
    _made(other, 7)
    for step in range(10, 16):
        _made(other, step, B)
    assert checkpoints.mixed_runs_clash(other) == (
        f"{other} holds checkpoints of 2 runs, so latest names none of them: run bbbbbbbb has "
        "2 older and checkpoint-12, checkpoint-13, checkpoint-14, checkpoint-15; "
        "checkpoint-7 name no run")
    one = tmp_path / "one"
    _made(one, 1, A)
    assert checkpoints.mixed_runs_clash(one) is None and checkpoints.mixed_runs_clash(
        tmp_path / "none") is None


def test_an_unstamped_checkpoint_beside_a_run_that_did_not_resume_from_one_is_refused(tmp_path):
    """Nothing says the stamped run came from the unstamped one: two runs, never guessed."""
    _made(tmp_path, 100)
    _made(tmp_path, 110, A)
    assert runs_in(tmp_path) == {None, A}
    _refused(tmp_path)


def test_a_third_run_beside_a_lineage_is_refused(tmp_path):
    _made(tmp_path, 100)
    _made(tmp_path, 110, A, "checkpoint-100")
    _made(tmp_path, 120, B)
    assert runs_in(tmp_path) == {A, B}
    _refused(tmp_path)


def test_two_runs_that_each_claim_the_unstamped_ones_are_refused(tmp_path):
    _made(tmp_path, 100)
    _made(tmp_path, 110, A, "checkpoint-100")
    _made(tmp_path, 120, B, "checkpoint-100")
    assert runs_in(tmp_path) == {None, A, B}
    _refused(tmp_path)


def test_resuming_again_from_an_unstamped_checkpoint_goes_on_with_the_run_that_did(tmp_path):
    """A relaunch that names the old checkpoint again (a job script's fixed --resume-from), or a
    resume from another of the old run's checkpoints: the same lineage, so the same run, as a
    resume from any stamped checkpoint of a run is. `latest` keeps working after it."""
    for step in (50, 100):
        _made(tmp_path, step)
    _made(tmp_path, 110, A, "checkpoint-100")
    assert lineage_of(tmp_path / "checkpoint-100", tmp_path, NEW) == (A, "checkpoint-100")
    assert lineage_of(tmp_path / "checkpoint-50", tmp_path, NEW) == (A, "checkpoint-50")
    _made(tmp_path, 60, A, "checkpoint-50")
    assert runs_in(tmp_path) == {A} and _latest(tmp_path).name == "checkpoint-110"


def test_a_checkpoint_from_another_folder_brings_no_claim_on_this_one(tmp_path):
    """Resumed from elsewhere, the unstamped checkpoints in the run's output_dir are not its
    ancestors: no parent is recorded, so they stay another run."""
    out, other = tmp_path / "out", tmp_path / "other"
    _made(out, 100)
    assert lineage_of(_made(other, 7), out, NEW) == (NEW, None)
    assert lineage_of(_made(other, 9, A, "checkpoint-7"), out, NEW) == (A, None)
    _made(out, 110, NEW)
    _refused(out)


def test_a_parent_marker_is_a_checkpoints_name_or_nothing(tmp_path):
    d = _made(tmp_path, 110, A, "checkpoint-100")
    assert parent_of(d) == "checkpoint-100"
    for said in ("", "../checkpoint-1", "checkpoint-007", "checkpoint-1.tmp", "yes"):
        (d / checkpoints.PARENT_MARKER).write_text(said)
        assert parent_of(d) is None
    (d / checkpoints.PARENT_MARKER).write_bytes(b"\xff\xfe")
    assert parent_of(d) is None
    _made(tmp_path, 100)
    assert runs_in(tmp_path) == {None, A}  # a marker that says nothing claims nothing


def _trainer(folder, run, parent, limit):
    from poros.train.trainer import PorosTrainer

    t = PorosTrainer.__new__(PorosTrainer)
    t._run_id, t._run_parent = run, parent
    t.config = SimpleNamespace(save_total_limit=limit, output_dir=str(folder))
    return t


def test_pruning_never_deletes_an_unstamped_checkpoint_or_another_runs(tmp_path):
    """save_total_limit counts and deletes what this run stamped. The unstamped checkpoints it
    resumed from are its lineage, and are still never deleted: nothing in them says whose they
    are, so deleting one could delete another run's work."""
    pytest.importorskip("torch")
    for step in (50, 100):
        _made(tmp_path, step)
    for step in (110, 120, 130):
        _made(tmp_path, step, A, "checkpoint-100")
    _made(tmp_path, 140, B)
    t = _trainer(tmp_path, A, "checkpoint-100", limit=1)
    t._prune_checkpoints()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "checkpoint-100", "checkpoint-130", "checkpoint-140", "checkpoint-50"]
    t._prune_checkpoints(room_for_one=True)  # a Mac's, before a save: never less than the newest
    assert (tmp_path / "checkpoint-130").is_dir() and (tmp_path / "checkpoint-100").is_dir()


def test_a_runs_own_newest_counts_the_unstamped_checkpoint_it_resumed_from(tmp_path):
    """Stopped before its first save, a run resumed from an unstamped checkpoint still has that
    one to name ("resume with --resume-from latest"), not "no checkpoint was written yet"."""
    pytest.importorskip("torch")
    _made(tmp_path, 100)
    _made(tmp_path, 140, B)
    resumed = _trainer(tmp_path, A, "checkpoint-100", limit=0)
    assert [s for s, _ in resumed._own_checkpoints()] == [100]
    fresh = _trainer(tmp_path, A, None, limit=0)
    assert fresh._own_checkpoints() == []
    assert [s for s, _ in list_checkpoints(tmp_path, run=B)] == [140]


def _snapshot(folder: Path) -> dict:
    return {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in sorted(folder.iterdir())}


def test_an_unstamped_checkpoint_resumed_then_latest_and_pruning_cpu(tmp_path):
    """End to end: a checkpoint as an older Poros wrote it (no .poros_run) is resumed, the run
    writes new checkpoints, `latest` picks the newest, a resume from it goes on under the same
    id, pruning keeps the unstamped one, and the checkpoint resumed from is never written to."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tests.test_train_loop_cpu import _loader, _make_trainer, _tiny_peft_qwen2

    def trainer(steps):
        torch.manual_seed(0)
        model, cfg = _tiny_peft_qwen2(torch)
        tr = _make_trainer(torch, model, str(tmp_path), max_steps=steps)
        tr.config.save_steps, tr.config.save_total_limit = 2, 2
        return tr, _loader(torch, cfg.vocab_size)

    tr, loader = trainer(2)
    tr.train(loader, steps=2)
    old = Path(tr._save_checkpoint())  # checkpoint-1: the second update, counted from 0
    os.unlink(old / checkpoints.RUN_MARKER)  # as an older Poros wrote it
    assert runs_in(tmp_path) == {None}
    before = _snapshot(old)

    resumed, loader = trainer(8)
    resumed.train(loader, steps=8, resume_from=str(old))
    assert _snapshot(old) == before  # never written to, and never pruned
    mine = list_checkpoints(tmp_path, run=resumed._run_id)
    assert [s for s, _ in mine] == [4, 6]  # save_total_limit=2 of its own (2 was pruned)
    assert all(parent_of(d) == "checkpoint-1" for _, d in mine)
    assert runs_in(tmp_path) == {resumed._run_id}
    newest = _latest(tmp_path)
    assert newest.name == "checkpoint-6"

    again, loader = trainer(10)
    again.train(loader, steps=10, resume_from=str(newest))
    assert again._run_id == resumed._run_id and again._run_parent == "checkpoint-1"
    assert runs_in(tmp_path) == {resumed._run_id} and _latest(tmp_path).name == "checkpoint-8"
    assert _snapshot(old) == before

    # a relaunch that names the old checkpoint again is the same run, not a second claimant
    relaunched, loader = trainer(4)
    relaunched.train(loader, steps=4, resume_from=str(old))
    assert relaunched._run_id == resumed._run_id
    assert runs_in(tmp_path) == {resumed._run_id}


def test_run_open_takes_a_folders_newest_as_latest_does(tmp_path):
    """poros.Run.open(<run folder>) reads the folder as `--resume-from latest` does: one lineage
    gives its newest checkpoint, two runs are refused with the same words before anything loads,
    and a checkpoint named itself is opened whatever is beside it."""
    pytest.importorskip("torch")
    from poros.core.errors import PorosConfigError
    from poros.train.run import Run, _checkpoint_in

    one, two = tmp_path / "one", tmp_path / "two"
    _made(one, 100)
    _made(one, 110, A, "checkpoint-100")
    assert _checkpoint_in(one).name == "checkpoint-110"

    _made(two, 2, A)
    named = _made(two, 4, B)
    with pytest.raises(PorosConfigError) as refused:
        Run.open(two)
    assert f"{two} holds checkpoints of 2 runs, so latest names none of them" in str(refused.value)
    assert f"Run.open('{two}/checkpoint-<step>')" in str(refused.value)
    assert checkpoints.mixed_runs_clash(two) in str(refused.value)  # the CLI's sentence
    (named / "poros_config.json").write_text("{}")
    assert _checkpoint_in(named) == named
    with pytest.raises(FileNotFoundError):
        _checkpoint_in(tmp_path / "none")


def test_run_open_puts_a_staged_save_in_its_place_before_it_looks(tmp_path):
    """A kill between a save's last file and its rename leaves the newest checkpoint complete
    at checkpoint-N.tmp. Run.open(folder) opened the older one; as `--resume-from latest` it
    first puts the staged one in its place, and a checkpoint named is found the same way."""
    pytest.importorskip("torch")
    from poros.train.run import _checkpoint_in

    _made(tmp_path, 2, A)
    staged = _made(tmp_path, 4, A)
    staged.rename(tmp_path / "checkpoint-4.tmp")
    torn = tmp_path / "checkpoint-6.tmp"  # a save killed midway: no completion marker
    torn.mkdir()
    (torn / checkpoints.FORMAT_MARKER).write_text("1")
    assert _checkpoint_in(tmp_path).name == "checkpoint-4"
    assert not (tmp_path / "checkpoint-4.tmp").exists() and not torn.exists()

    named = _made(tmp_path, 8, A)
    (named / "poros_config.json").write_text("{}")
    named.rename(tmp_path / "checkpoint-8.tmp")
    assert _checkpoint_in(tmp_path / "checkpoint-8") == tmp_path / "checkpoint-8"
