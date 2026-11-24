"""A Hub dataset that has to be downloaded says so from a Python script. The datasets library's
bars are off for the whole read (they draw over a run's screen), so a large download was minutes
of silence. A local file and a dataset already on this machine stay quiet."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

datasets = pytest.importorskip("datasets")
pytest.importorskip("torch")

from poros.core import dataset_spec  # noqa: E402
from poros.train import datasets as loading  # noqa: E402


class _Builder:
    def __init__(self, cache_dir):
        self.config = SimpleNamespace(data_files={"train": ["hf://datasets/org/name@a/x.parquet"]})
        self.cache_dir, self.bars = str(cache_dir), None

    def download_and_prepare(self):
        self.bars = datasets.is_progress_bar_enabled()  # as the transfer would see them

    def as_dataset(self, split):
        return [split]


def test_the_loader_says_when_a_dataset_is_not_here_yet(tmp_path, monkeypatch):
    builder = _Builder(tmp_path / "cache")
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda name: builder)
    called = []
    assert dataset_spec.load_hub_dataset("org/name", downloading=lambda: called.append(1))
    assert called == [1]  # no prepared copy: it is about to be fetched
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "dataset_info.json").write_text("{}")
    dataset_spec.load_hub_dataset("org/name", downloading=lambda: called.append(2))
    assert called == [1]  # already on this machine: nothing to say
    assert dataset_spec.load_hub_dataset("org/name") == ["train"]  # and no caller to tell


def test_a_download_is_said_and_shown_and_the_rest_stays_quiet(tmp_path, monkeypatch):
    builder = _Builder(tmp_path / "cache")
    monkeypatch.setattr(datasets, "load_dataset_builder", lambda name: builder)
    said, was = [], datasets.is_progress_bar_enabled()
    try:
        with loading._no_progress_bars():  # as the whole read runs
            with loading._download_shown("org/name", said.append) as downloading:
                dataset_spec.load_hub_dataset("org/name", downloading=downloading)
            after = datasets.is_progress_bar_enabled()  # the tokenizing that follows
        assert said == ["Downloading dataset org/name from the Hugging Face Hub (cached after "
                        "this)"]
        assert builder.bars is True and after is False
        assert datasets.is_progress_bar_enabled() is was  # as they were, once it is done

        # already here: no line, no bars
        (tmp_path / "cache").mkdir()
        (tmp_path / "cache" / "dataset_info.json").write_text("{}")
        said.clear()
        with loading._no_progress_bars():
            with loading._download_shown("org/name", said.append) as downloading:
                dataset_spec.load_hub_dataset("org/name", downloading=downloading)
        assert said == [] and builder.bars is False

        # a front end that shows its own screen passes no printer: nothing changes
        with loading._no_progress_bars():
            with loading._download_shown("org/name", None) as downloading:
                assert downloading is None
    finally:
        (datasets.enable_progress_bars if was else datasets.disable_progress_bars)()


def test_a_scripts_trainer_passes_its_printer_and_a_front_ends_does_not(monkeypatch):
    """The trainer's own lines are on for a script and off where a front end owns the screen."""
    from poros.train import loop

    seen = []
    monkeypatch.setattr(loop, "_load_hf_dataset_as_dataloader",
                        lambda **how: seen.append(how["say"]) or "loader")
    trainer = SimpleNamespace(
        config=SimpleNamespace(model_name_or_path="m", per_device_batch_size=1, seed=0,
                               trust_remote_code=False, model_revision=None),
        _metal=False, _progress_print=print, _progress_enabled=lambda: True)
    assert loop._Loop._prepare_dataloader(trainer, "org/name", 16) == "loader"
    trainer._progress_enabled = lambda: False
    loop._Loop._prepare_dataloader(trainer, "org/name", 16)
    assert seen == [print, None]
