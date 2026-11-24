"""The fused compute's extras for a ``poros.moe`` model (bf16), beside its fused experts.

``install_extras(model)``: its loss a chunk of rows at a time (``head.install_loss``), its
causal attention through the memory-efficient SDPA kernel (``attention.install_sdpa``) and its
RMSNorms in one kernel each way (``norm.install_rms_norm``), in that order -- what
``PorosTrainer`` installs with ``moe_expert_compute: fused`` and ``fused_compute`` not false.
``uninstall_extras`` undoes them.
"""

from __future__ import annotations

import torch

__all__ = ["install_extras", "uninstall_extras"]


def install_extras(model: torch.nn.Module) -> dict:
    """Install the three extras on ``model``; what each changed."""
    from poros.core.fused import attention, head, norm

    head.install_loss(model)
    sdpa = attention.install_sdpa(model)
    norms = norm.install_rms_norm(model)
    return {"loss": True, "attention": sdpa, "rms_norms": norms}


def uninstall_extras(model: torch.nn.Module) -> None:
    """The model's own loss, attention and RMSNorms again."""
    from poros.core.fused import attention, head, norm

    norm.uninstall_rms_norm(model)
    attention.uninstall_sdpa(model)
    head.uninstall_loss(model)
