"""What `poros train <config>` reads before the load: a data file's rows, and the download's size.

Each refusal comes before the download, with its code; every format the trainer reads reaches it.
"""

from __future__ import annotations

import json

import pytest

from poros.cli import train_cmd
from poros.preflight.data import checked_before_load

torch = pytest.importorskip("torch")


class _Loaded(Exception):
    """The trainer was reached: every check before the load let the run through."""


LOADS: list = []  # each call that reached PorosTrainer.from_pretrained


def _cuda_host(monkeypatch):
    from poros.cli.commands import train as cli_commands_train
    from poros.preflight import arch as preflight_arch
    from poros.train.trainer import PorosTrainer

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=None: (8, 9))
    monkeypatch.setattr("torch.cuda.get_device_name", lambda device=None: "NVIDIA GeForce RTX 4090")
    monkeypatch.setattr("torch.cuda.get_arch_list", lambda: ["sm_80", "sm_86", "sm_89"])
    monkeypatch.setattr(preflight_arch, "resolve_model_type", lambda m: ("qwen2", ""))
    monkeypatch.setattr(cli_commands_train, "_plan_gate", lambda cfg, mt, force: None)
    monkeypatch.setattr("poros.train.trainer.apply_expandable_segments", lambda *a, **k: None)

    LOADS.clear()

    def loaded(*a, **k):
        LOADS.append(a)
        raise _Loaded  # the flow reports it as an error and exits: the call is what counts

    monkeypatch.setattr(PorosTrainer, "from_pretrained", loaded)


def _error(res) -> dict:
    """The poros.error.v1 object of a refused run (the screen's lines may come before it)."""
    line = next(x for x in res.output.splitlines() if x.startswith('{"schema":"poros.error'))
    return json.loads(line)["error"]


def _train(tmp_path, dataset, *flags):
    from typer.testing import CliRunner

    from poros.cli import main

    cfg = tmp_path / "run.yaml"
    cfg.write_text(f"model_name_or_path: Qwen/Qwen2.5-0.5B\ndataset: {dataset}\n"
                   f"output_dir: {tmp_path / 'out'}\n")
    return CliRunner().invoke(main.app, ["train", str(cfg), "--json", *flags])


@pytest.mark.parametrize("name, read", [("rows.jsonl", True), ("rows.json", True),
                                        ("rows.csv", True), ("rows.parquet", False),
                                        ("notes.txt", False), ("rows.JSONL", False)])
def test_only_the_files_data_reads_as_the_trainer_does_are_read_before_the_load(tmp_path, name,
                                                                                 read):
    (tmp_path / name).write_bytes(b"x")
    assert checked_before_load(tmp_path / name) is read


def test_a_parquet_file_goes_to_the_trainer_as_before(tmp_path, monkeypatch, recipe_configs_cached):
    """A .parquet is not JSON lines: it is the trainer's to read (dataset_spec), never refused
    row by row as 'not JSON'."""
    _cuda_host(monkeypatch)
    data = tmp_path / "train.parquet"
    data.write_bytes(b"PAR1\x00\x01\x02binary\nrows\nPAR1")
    res = _train(tmp_path, data)
    assert LOADS, res.output
    assert "not JSON" not in res.output


def test_an_unreadable_json_file_is_a_dataset_refusal_not_a_trace(tmp_path, monkeypatch):
    _cuda_host(monkeypatch)
    data = tmp_path / "train.json"
    data.write_text('[{"text": "a"}, {"text": ')  # cut off
    res = _train(tmp_path, data)
    assert res.exit_code == 6 and not LOADS, res.output
    err = _error(res)
    assert err["code"] == "E_DATASET" and "train.json" in err["message"]


def test_rows_that_stop_a_run_are_still_refused_before_the_load(tmp_path, monkeypatch):
    _cuda_host(monkeypatch)
    data = tmp_path / "train.jsonl"
    data.write_text('{"text": "a fine row"}\nnot a row\n')
    res = _train(tmp_path, data)
    assert res.exit_code == 6 and _error(res)["code"] == "E_DATASET"


def _download(monkeypatch, gib, free_bytes):
    import shutil
    from collections import namedtuple

    from poros.cli.schemas import Figure

    seen = {}

    def size(cfg):
        seen["revision"] = cfg.model_revision
        if gib is None:
            return Figure(value=None, unit="GiB", source="unknown", why="the Hub did not say")
        return Figure(value=gib, unit="GiB", source="hub_metadata", method="revision abc")

    monkeypatch.setattr("poros.plan.configs.download_size", size)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10 * free_bytes, 0, free_bytes))
    return seen


def test_a_download_past_the_free_disk_is_refused_and_force_overrides_it(tmp_path, monkeypatch,
                                                                        recipe_configs_cached):
    _cuda_host(monkeypatch)
    _download(monkeypatch, 140.0, 20 * 2**30)
    data = tmp_path / "rows.jsonl"
    data.write_text('{"text": "a fine row"}\n')
    res = _train(tmp_path, data)
    assert res.exit_code == 5, res.output
    err = _error(res)
    assert err["code"] == "E_DISK" and "150.3 GB download" in err["message"] and not LOADS
    res = _train(tmp_path, data, "--force")
    assert LOADS, res.output


def test_the_disk_check_counts_the_configs_revision(tmp_path, monkeypatch):
    """What a load fetches of model_revision, as poros.plan.configs.download_size counts it (the
    loader's own files, what the cache already holds left out): a run that fits is not refused."""
    from poros.core.config import PorosConfig

    seen = _download(monkeypatch, 1.0, 20 * 2**30)
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B", dataset="synthetic",
                      model_revision="0123abc")
    note = train_cmd._download_note(cfg)
    assert seen["revision"] == "0123abc" and note is not None and note[2] == ""
    _download(monkeypatch, None, 20 * 2**30)  # the Hub did not say: nothing refused on a guess
    assert train_cmd._download_note(cfg) is None
