"""A `poros.Run` loop can say what it trains on, and its resume is then checked.

The quickstart's own `poros.Run.open("./run")` snippet warned "Data that could not be checked:
this run is handed its rows by a loop of its own...": a loop hands the run batches, so nothing
was recorded to compare. `Run.create(..., dataset="chats.jsonl")` names the file the rows come
from: each checkpoint records its contents, `Run.open` compares them, and the warning is left
for a loop that names nothing.
"""

from __future__ import annotations

import json
import logging
import shutil

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from poros.preflight.resume import INPUTS_RECORD  # noqa: E402
from tests.test_run_api import _chats, _run  # noqa: E402


def _rows(path, n=8):
    path.write_text("".join(json.dumps({"messages": [
        {"role": "user", "content": f"say {i}"},
        {"role": "assistant", "content": f"{i} {i} {i}"}]}) + "\n" for i in range(n)))
    return path


def _loop(run, steps):
    data = _chats()
    for step in run.steps(steps):
        run.forward_backward(data.batch(step))
        run.optim_step()
    return run.save()


def _warned(caplog):
    return [r.getMessage() for r in caplog.records if r.name == "poros.preflight.resume"]


def test_a_loop_that_names_its_data_records_it_and_resumes_checked(tmp_path, caplog):
    data = _rows(tmp_path / "chats.jsonl")
    run = _run(torch, tmp_path / "run", learning_rate=1e-2, save_steps=0, dataset=str(data))
    ckpt = _loop(run, 2)
    kept = json.loads((tmp_path / "run" / "checkpoint-1" / INPUTS_RECORD).read_text())
    assert len(kept["dataset"]["sha256"]) == 64  # the file, by what it holds

    again = _run(torch, tmp_path / "run", learning_rate=1e-2, save_steps=0, dataset=str(data))
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        again._declare_data()  # as Run.open does, before the load compares
        again.load(ckpt)
    assert _warned(caplog) == []  # checked: nothing to say


def test_the_same_data_at_another_path_resumes_and_other_data_is_refused(tmp_path, caplog):
    from poros.core.errors import PorosConfigError

    data = _rows(tmp_path / "chats.jsonl")
    ckpt = _loop(_run(torch, tmp_path / "run", learning_rate=1e-2, save_steps=0,
                      dataset=str(data)), 2)
    (tmp_path / "sub").mkdir()
    moved = tmp_path / "sub" / "chats.jsonl"
    shutil.move(data, moved)
    again = _run(torch, tmp_path / "run", learning_rate=1e-2, save_steps=0, dataset=str(moved))
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        again._declare_data()
        again.load(ckpt)
    assert _warned(caplog) == []
    other = _rows(tmp_path / "other.jsonl", n=5)
    wrong = _run(torch, tmp_path / "run", learning_rate=1e-2, save_steps=0, dataset=str(other))
    wrong._declare_data()
    with pytest.raises(PorosConfigError, match="holds other data"):
        wrong.load(ckpt)


def test_run_open_declares_the_data_before_it_loads(tmp_path, monkeypatch, caplog):
    """`Run.open(folder)` takes the dataset its checkpoint's config names, and `dataset=` says
    where that file is now."""
    from poros.train.run import Run

    data = _rows(tmp_path / "chats.jsonl")
    _loop(_run(torch, tmp_path / "run", learning_rate=1e-2, save_steps=0, dataset=str(data)), 2)
    seen = {}

    def loaded(cls, model, config, tokenizer, *a, **k):
        seen["dataset"] = config.dataset
        return _run(torch, tmp_path / "run", learning_rate=1e-2, save_steps=0,
                    dataset=config.dataset)

    monkeypatch.setattr(Run, "_loaded", classmethod(loaded))
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        run = Run.open(tmp_path / "run")
    assert seen["dataset"] == str(data) and _warned(caplog) == [] and run._next == 2
    (tmp_path / "sub").mkdir()
    moved = tmp_path / "sub" / "chats.jsonl"
    shutil.move(data, moved)
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        Run.open(tmp_path / "run", dataset=str(moved))
    assert seen["dataset"] == str(moved) and _warned(caplog) == []


def test_a_loop_that_names_nothing_still_says_it_could_not_be_checked(tmp_path, caplog):
    ckpt = _loop(_run(torch, tmp_path / "run", learning_rate=1e-2, save_steps=0), 2)
    again = _run(torch, tmp_path / "run", learning_rate=1e-2, save_steps=0)
    with caplog.at_level(logging.WARNING, logger="poros.preflight.resume"):
        again._declare_data()
        again.load(ckpt)
    (said,) = _warned(caplog)
    assert "handed its rows by a loop of its own" in said
    assert 'poros.Run.create(..., dataset="chats.jsonl")' in said  # and how to be checked


def test_the_progress_lines_say_memory_in_gib_and_where_a_save_landed(tmp_path, capsys):
    """The API printed `peak 0.8 GB` where the CLI says GiB, and nothing about where it saved."""
    run = _run(torch, tmp_path / "run", learning_rate=1e-2, save_steps=0)
    run._engine.config.progress = True
    path = _loop(run, 2)
    out = capsys.readouterr().out
    assert "GiB" in out and " GB" not in out
    assert out.count(f"Saved to:         {path}") == 1
    run.save()  # the same step, saved again: its place is said once
    assert "Saved to" not in capsys.readouterr().out


def test_the_cpu_kernel_advice_is_filtered_for_a_script_too():
    """`import poros` is all a script does: bitsandbytes' "Failed to load CPU gemm_4bit_forward"
    advice printed from the Python API on a machine with no GPU."""
    import poros  # noqa: F401

    log = logging.getLogger("bitsandbytes.backends.cpu.ops")
    assert [type(f).__name__ for f in log.filters].count("_NoCpuKernelAdvice") == 1
    record = logging.LogRecord(log.name, logging.WARNING, __file__, 1,
                               "Failed to load CPU gemm_4bit_forward from kernels-community: "
                               "No module named 'kernels'", None, None)
    assert not log.filter(record)
    other = logging.LogRecord(log.name, logging.WARNING, __file__, 1, "something else", None,
                              None)
    assert log.filter(other)


def test_no_fetching_bar_when_every_weight_file_is_here(monkeypatch):
    """A first run of a cached model drew the Hub's "Fetching 8 files" bar: nothing was fetched.
    A real download keeps its bars, and bars a caller turned off stay off."""
    import huggingface_hub
    from huggingface_hub import utils as hub_utils

    from poros.train import streamed_load

    bars = {"off": False}  # the Hub's own switch, stood in for (an env var can pin the real one)
    monkeypatch.setattr(hub_utils, "are_progress_bars_disabled", lambda: bars["off"])
    monkeypatch.setattr(hub_utils, "disable_progress_bars", lambda: bars.update(off=True))
    monkeypatch.setattr(hub_utils, "enable_progress_bars", lambda: bars.update(off=False))
    seen = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda model, **how: seen.append(bars["off"]) or "/a/snapshot")
    monkeypatch.setattr("poros.core.arch_registry.hub_token", lambda: False)
    for held, quiet in (((1000, 0), True), ((1000, 400), False), (None, False)):
        monkeypatch.setattr("poros.plan.configs.weights_on_hub",
                            lambda model, revision=None, held=held: held)
        assert streamed_load._resolve_checkpoint_dir("owner/model") == "/a/snapshot"
        assert seen[-1] is quiet and bars["off"] is False  # as they were, after
    bars["off"] = True  # turned off by the caller (the CLI does): left off
    monkeypatch.setattr("poros.plan.configs.weights_on_hub", lambda model, revision=None: (9, 0))
    streamed_load._resolve_checkpoint_dir("owner/model")
    assert seen[-1] is True and bars["off"] is True
