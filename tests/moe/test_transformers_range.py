"""``check_transformers``: refuse below the tested range, warn above it, pass inside it."""

from __future__ import annotations

import logging

import pytest

from poros.core.errors import PorosStackUnsupportedError
from poros.moe import arch


@pytest.fixture
def installed(monkeypatch):
    def install(found: str) -> None:
        monkeypatch.setattr(arch, "version", lambda _: found)
        arch.check_transformers.cache_clear()

    yield install
    arch.check_transformers.cache_clear()


def test_refuses_below_the_tested_range(installed):
    installed("5.12.1")
    pip = r"pip install 'transformers>=5\.13,<5\.23'"
    with pytest.raises(PorosStackUnsupportedError, match=pip):
        arch.check_transformers()


@pytest.mark.parametrize("found", ["5.13.0", "5.20.2", "5.22.2"])
def test_passes_inside_the_tested_range(installed, caplog, found):
    installed(found)
    arch.check_transformers()
    assert not caplog.records


def test_warns_above_the_tested_range(installed, caplog):
    installed("5.23.0.dev0")
    with caplog.at_level(logging.WARNING, logger="poros.moe.arch"):
        arch.check_transformers()
    assert "found 5.23.0.dev0" in caplog.text
