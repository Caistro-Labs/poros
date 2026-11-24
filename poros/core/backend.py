"""What the shared engine refuses on ``mps``. Torch-free: this loads on a torch-free install, and on
``cuda`` and ``cpu`` nothing here draws RNG or allocates."""

from __future__ import annotations

from typing import Any

from poros.core.errors import PorosGuaranteeUnavailableError


def refuse_on_mps(device: Any, what: str) -> None:
    """Refuse a path with no MPS RNG capture: a remat's masks would differ from the forward's."""
    if device.type == "mps":
        raise PorosGuaranteeUnavailableError(
            f"{what} does not run on mps: it has no MPS RNG capture, so a remat's "
            "dropout masks would differ from the forward's.",
            remediation="On a Mac, train NF4 with `poros train`.",
        )


def refuse_model_on_mps(model: Any, what: str) -> None:
    """A checkpoint can load in pieces: one parameter on ``mps`` is enough."""
    for param in model.parameters():
        device = getattr(param, "device", None)  # duck-typed models have none
        if device is not None:
            refuse_on_mps(device, what)
