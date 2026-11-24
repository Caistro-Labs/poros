"""The words a run's start says about its path, its block and a host-RAM refusal: true for the
run they describe, said once."""

from __future__ import annotations

import logging

import pytest

from poros.core.config import PorosConfig
from poros.core.errors import PorosHostRamError


def _cfg(**over) -> PorosConfig:
    return PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B", dataset="synthetic", **over)


@pytest.fixture
def no_card(monkeypatch):
    """on_cuda's checks that read a card or the Hub, stubbed: what is left is its path words."""
    monkeypatch.setattr("poros.core.card.check", lambda *a, **k: None)
    monkeypatch.setattr("poros.plan.configs.read_model_config", lambda *a, **k: None)


def test_a_path_poros_train_decided_is_not_said_again_and_never_as_another(no_card, caplog):
    """A run poros train moved to fast (its note said so) gets no second line from the start,
    and none that says sequential."""
    from poros.preflight.start import on_cuda

    cfg = _cfg()
    cfg.decide_path("fast", "efficient may not fit the GPU; fast is estimated to fit")
    with caplog.at_level(logging.WARNING, logger="poros.preflight.start"):
        on_cuda(cfg, None, cfg.model_name_or_path)
    assert not [r for r in caplog.records if r.getMessage().startswith("path:")]


def test_a_path_decided_at_the_start_is_said_once_by_its_own_name(no_card, caplog, monkeypatch):
    from poros.preflight.start import on_cuda

    monkeypatch.setattr("poros.core.config.sequential_reason", lambda *a, **k: "its layers vary")
    cfg = _cfg()
    with caplog.at_level(logging.WARNING, logger="poros.preflight.start"):
        on_cuda(cfg, None, cfg.model_name_or_path)
    said = [r.getMessage() for r in caplog.records if r.getMessage().startswith("path:")]
    assert said == ["path: sequential: its layers vary"]


def test_the_start_box_says_no_block_for_a_model_that_routes_to_the_moe_path():
    from poros.cli.train_cmd import _block_word

    assert _block_word(_cfg(), "qwen2") == "block 4 · "  # dense, as before
    assert _block_word(_cfg(block_size=2), "qwen2") == "block 2 · "
    assert _block_word(_cfg(), "qwen3_moe") == ""  # poros.moe streams per expert


@pytest.mark.parametrize("path, opened, advised", [
    ("efficient", ("fast",), True), ("efficient", (), False), (None, (), False),
    ("fast", (), False), ("sequential", (), False)])
def test_a_host_ram_refusal_says_what_is_true_of_the_run(path, opened, advised):
    """Path advice only where the check found another path it does not refuse (a run that set
    path: efficient by hand), naming it."""
    err = PorosHostRamError("m", 9.7, 9.37, "an estimate", path=path, open_paths=opened)
    assert err.info.message.endswith("Refusing before anything is downloaded or loaded.")
    # what this machine has now, never the plan's figure
    assert "9.37 GiB" in err.info.message and "9.7" not in err.info.message + err.info.remediation
    assert ("path: fast is not refused here" in err.info.remediation) is advised
    assert "poros train --force" in err.info.remediation and err.code == "E_HOST_OOM_PREDICTED"


def test_eval_refused_for_host_ram_is_told_what_eval_can_do(monkeypatch):
    """From poros eval: the same refusal (code, message), and a remedy eval can act on: no
    poros train --force, no API flag, no path advice for a pass that runs sequential."""
    pytest.importorskip("torch")
    from poros.cli.verbs import evaluate_pass
    from poros.train.trainer import PorosTrainer

    def refuse(*a, **k):
        raise PorosHostRamError("Qwen/Qwen3-30B-A3B-Instruct-2507", 9.7, 9.37, "an estimate",
                                path="sequential")

    monkeypatch.setattr(PorosTrainer, "from_pretrained", refuse)
    with pytest.raises(PorosHostRamError) as found:
        evaluate_pass.engine(_cfg())
    info = found.value.info
    assert found.value.code == "E_HOST_OOM_PREDICTED" and "9.37 GiB available" in info.message
    assert info.remediation == ("Free host RAM (9.37 GiB is available now): close other "
                                "programs, then run poros eval again.")
    assert "9.7" not in info.message + info.remediation  # the plan's figure is not said


def test_poros_train_says_a_path_it_decides_for_a_reason_once(tmp_path, monkeypatch):
    """A run poros train itself moves to sequential (its CUDA defaults: a structural reason)
    reads one path note, with the reason, and no second line naming another path."""
    pytest.importorskip("torch")
    from typer.testing import CliRunner

    from poros.cli import main

    monkeypatch.setenv("POROS_DEV_SIMULATE", "1")
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("poros.core.config.sequential_reason", lambda *a, **k: "its layers vary")
    cfg = tmp_path / "run.yaml"
    cfg.write_text(f"model_name_or_path: Qwen/Qwen2.5-0.5B\ndataset: synthetic\nmax_steps: 1\n"
                   f"output_dir: {tmp_path / 'out'}\n")
    res = CliRunner().invoke(main.app, ["train", str(cfg)])
    text = res.output
    assert text.count("path: sequential") == 1 and "its layers vary" in text, text


def _unfit(path: str, entry: str):
    from types import SimpleNamespace

    from poros.core.errors import PorosVramError

    return SimpleNamespace(vram=PorosVramError("m", "an RTX 4060 Laptop", "a measured peak"),
                           p=SimpleNamespace(path=path, entry=entry,
                                             gpu=SimpleNamespace(unified=False)))


@pytest.mark.parametrize("path", ["fast", "efficient", "sequential"])
def test_a_moe_runs_gpu_refusal_never_offers_a_block_size(path):
    """poros.moe refuses a block_size: its GPU refusal must not send the user into that one."""
    from poros.preflight.command import fit_refusal

    fix = fit_refusal(_unfit(path, "moe")).fix
    assert "block" not in fix
    assert fix.startswith("path: efficient holds less; ") == (path == "fast")


def test_a_dense_runs_gpu_refusal_reads_as_before():
    from poros.preflight.command import fit_refusal

    tail = "poros recipes lists the runs certified on this card; --force trains anyway."
    assert fit_refusal(_unfit("efficient", "dense")).fix == (
        "a smaller block_size or path: sequential may hold less; " + tail)
    assert fit_refusal(_unfit("fast", "dense")).fix == "path: efficient holds less; " + tail
    assert fit_refusal(_unfit("sequential", "dense")).fix == tail
    found = fit_refusal(_unfit("sequential", "dense"))
    assert found.code == "E_OOM_PREDICTED" and found.docs == "docs/training.md#what-fits"


def test_a_gpu_refusal_says_what_is_true_of_a_cached_model():
    from poros.core.errors import PorosVramError

    said = PorosVramError("m", "a card", "a measured peak").info.message
    assert said.endswith("Refusing before anything is downloaded or loaded.")


@pytest.mark.parametrize("path, advised", [("efficient", True), ("sequential", False)])
def test_the_cli_says_a_host_ram_refusals_own_advice_and_exits_3(monkeypatch, capsys, path,
                                                                 advised):
    """poros train's screen and --json say the trainer's own remedy, word for word (its path
    advice where it applies, and poros recipes), with the head built from the error's fields,
    E_HOST_OOM_PREDICTED and exit 3."""
    import json

    import typer

    from poros.cli import errors as cli_errors
    from poros.core.errors import EXIT_CODES

    monkeypatch.setenv("POROS_OUTPUT", "json")
    err = PorosHostRamError("org/model", 9.7, 9.37, "an estimate", path=path,
                            open_paths=("fast",) if advised else ())
    with pytest.raises(typer.Exit) as stop:
        cli_errors._friendly_error(err)
    assert stop.value.exit_code == EXIT_CODES["E_HOST_OOM_PREDICTED"] == 3
    body = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]
    assert body["code"] == "E_HOST_OOM_PREDICTED" and body["exit_code"] == 3
    assert body["message"] == ("org/model needs more host RAM than the 9.37 GiB available here. "
                               "Refusing before anything is downloaded or loaded.")
    said = json.dumps(body)
    assert ("path: fast is not refused here" in said) is advised
    assert "poros recipes lists the runs certified on each machine" in body["hint"]  # advice
    assert "poros recipes lists the runs certified on each machine" in said
    assert "9.7" not in said  # the plan's figure is not said


def test_the_cli_says_a_gpu_refusals_own_advice_and_exits_3(monkeypatch, capsys):
    import json

    import typer

    from poros.cli import errors as cli_errors
    from poros.core.errors import PorosVramError

    monkeypatch.setenv("POROS_OUTPUT", "json")
    err = PorosVramError("org/model", "RTX 4090", "allocated peak 30 GiB > 23.5 GiB",
                         "max_seq_length: 512 is not refused", any_path=True)
    with pytest.raises(typer.Exit) as stop:
        cli_errors._friendly_error(err)
    assert stop.value.exit_code == 3
    body = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]
    assert body["code"] == "E_OOM_PREDICTED" and body["message"].startswith(
        "org/model won't fit RTX 4090 on any path: at these settings the run needs more memory "
        "than this GPU can give it.")
    said = json.dumps(body)
    assert "What to change: max_seq_length: 512 is not refused" in said
    assert "poros recipes lists the runs certified on this card" in said and "30 GiB" not in said
