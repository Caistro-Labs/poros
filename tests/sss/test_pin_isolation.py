"""The pin budget does not leak between tests (tests/conftest.py): a test that resets it under a
faked WSL2 leaves ``env=0`` behind it, and the next test must see a fresh state, computed in the
real environment (the CUDA bulk-residency test depends on pins being granted after tests/sss)."""

from __future__ import annotations

import pytest

from poros.sss import pinning
from tests import conftest


def test_a_faked_environment_does_not_outlive_its_test():
    """The faked test and the fixture's cleanup in one test, so it means the same alone or in
    any order: under the fakes the budget is 0; after them, the state is forgotten."""
    with pytest.MonkeyPatch.context() as mp:
        mp.delenv("POROS_PIN_BUDGET", raising=False)
        mp.setattr(pinning, "_wsl", lambda: True)
        mp.setattr(pinning, "_wsl_probe_result", None)
        mp.setattr(pinning, "_wsl_child_probe", lambda nbytes=0: False)
        pinning.reset()
        assert pinning.status()["env"] == 0  # what a later test would inherit without the fixture
    cleanup = conftest._pin_budget_ends_with_its_test.__wrapped__()
    next(cleanup)  # the fixture's set-up: nothing
    with pytest.raises(StopIteration):
        next(cleanup)  # its teardown: what runs after every test
    assert pinning._state == {}
