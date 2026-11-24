"""Tests for the license-free synthetic-token benchmark dataset.

The synthetic path is what the official parity / VRAM / throughput benchmark
trains on: deterministic, zero-download, and free of any dataset license. These
tests skip cleanly via ``pytest.importorskip("torch")`` when torch is absent
(the pure-python CI job).
"""

from __future__ import annotations

import pytest


def test_synthetic_sample_count_parsing() -> None:
    # Lives in trainer.py, which imports torch at module load, so skip cleanly
    # in the pure-python (no-torch) CI job like the rest of the torch tests.
    pytest.importorskip("torch")
    from poros.train.datasets import _synthetic_sample_count

    assert _synthetic_sample_count("synthetic") == 512
    assert _synthetic_sample_count("synthetic:128") == 128
    assert _synthetic_sample_count("synthetic:1") == 1
    # Malformed count falls back to the default rather than crashing.
    assert _synthetic_sample_count("synthetic:notanint") == 512


def test_synthetic_dataloader_shapes_and_keys() -> None:
    torch = pytest.importorskip("torch")
    from poros.train.datasets import _build_synthetic_dataloader

    dl = _build_synthetic_dataloader(
        n_samples=64, max_seq_length=16, batch_size=4, seed=42
    )
    batch = next(iter(dl))
    assert sorted(batch.keys()) == ["attention_mask", "input_ids", "labels"]
    assert batch["input_ids"].shape == (4, 16)
    # Labels mirror inputs (causal LM), mask is all-ones (no padding).
    assert torch.equal(batch["input_ids"], batch["labels"])
    assert torch.all(batch["attention_mask"] == 1)


def test_synthetic_token_range_in_vocab() -> None:
    pytest.importorskip("torch")
    from poros.train.datasets import (
        _SYNTHETIC_VOCAB_HIGH,
        _SYNTHETIC_VOCAB_LOW,
        _build_synthetic_dataloader,
    )

    dl = _build_synthetic_dataloader(
        n_samples=128, max_seq_length=32, batch_size=8, seed=0
    )
    for batch in dl:
        assert int(batch["input_ids"].min()) >= _SYNTHETIC_VOCAB_LOW
        assert int(batch["input_ids"].max()) < _SYNTHETIC_VOCAB_HIGH


def test_synthetic_dataloader_is_deterministic() -> None:
    torch = pytest.importorskip("torch")
    from poros.train.datasets import _build_synthetic_dataloader

    a = next(iter(_build_synthetic_dataloader(
        n_samples=64, max_seq_length=16, batch_size=4, seed=42
    )))
    b = next(iter(_build_synthetic_dataloader(
        n_samples=64, max_seq_length=16, batch_size=4, seed=42
    )))
    c = next(iter(_build_synthetic_dataloader(
        n_samples=64, max_seq_length=16, batch_size=4, seed=7
    )))
    # Same seed -> identical inputs on every machine; different seed -> different.
    assert torch.equal(a["input_ids"], b["input_ids"])
    assert not torch.equal(a["input_ids"], c["input_ids"])
