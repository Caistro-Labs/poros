"""A Hub id whose config.json cannot be fetched because the Hub cannot be asked (offline, no
network) says so; it is not told the repo "has no config.json" (LocalEntryNotFoundError
subclasses EntryNotFoundError, and is told apart from it)."""

from __future__ import annotations

import pytest

pytest.importorskip("huggingface_hub")
from huggingface_hub import (
    constants,  # noqa: E402
    errors,  # noqa: E402
)

from poros.preflight import arch  # noqa: E402


def _fetch_raises(monkeypatch, exc):
    from poros.plan import configs

    def fetch(model, revision=None):
        raise exc

    monkeypatch.setattr(configs, "fetch_config", fetch)


@pytest.mark.parametrize("offline", [True, False])
def test_no_hub_and_no_cache_is_named_as_such(monkeypatch, offline):
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", offline)
    _fetch_raises(monkeypatch, errors.LocalEntryNotFoundError("not cached"))
    mt, why = arch.resolve_model_type("Org/Some-Model")
    assert mt is None and "has no config.json" not in why
    assert "not in the local cache" in why
    assert ("HF_HUB_OFFLINE" in why) == offline and ("network" in why) != offline


def test_a_dropped_connection_names_the_network(monkeypatch):
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)
    _fetch_raises(monkeypatch, ConnectionError("refused"))
    mt, why = arch.resolve_model_type("Org/Some-Model")
    assert mt is None and "could not reach the Hub" in why and "ConnectionError" in why


def test_a_repo_without_config_json_is_still_said_so(monkeypatch):
    _fetch_raises(monkeypatch, errors.EntryNotFoundError("404"))
    mt, why = arch.resolve_model_type("Org/Some-Model")
    assert mt is None and "has no config.json" in why
