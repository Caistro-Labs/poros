"""``patch_hf_model`` (the MoE path's dense streaming) refuses a model on ``mps``: that path has
no MPS RNG capture, so a remat's dropout masks would differ from the forward's."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from poros.core.backend import refuse_model_on_mps  # noqa: E402
from poros.core.errors import EXIT_CODES, PorosGuaranteeUnavailableError  # noqa: E402
from poros.patch.hf import patch_hf_model  # noqa: E402


class _Param:
    def __init__(self, device):
        self.device = torch.device(device)


class _Model:
    def __init__(self, *devices):
        self._params = [_Param(d) for d in devices]

    def parameters(self):
        return iter(self._params)


@pytest.mark.parametrize("devices", [("mps",), ("cpu", "mps")])
def test_a_model_on_mps_is_refused(devices):
    with pytest.raises(PorosGuaranteeUnavailableError) as exc:
        patch_hf_model(_Model(*devices))
    assert exc.value.code == "E_GUARANTEE_UNAVAILABLE"
    assert EXIT_CODES[exc.value.code] == 4
    assert "poros train" in exc.value.info.remediation


@pytest.mark.parametrize("device", ["cpu", "cuda:0", "meta"])
def test_other_devices_pass_the_refusal(device):
    refuse_model_on_mps(_Model(device), "MoE training")
