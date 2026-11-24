"""Which compute a run takes: one decision, for ``PorosTrainer`` and its plan alike.

Two settings choose it, each ``None`` by default ("where it applies"):

* ``fused_compute``: the fused compute. On a dense model, the decoder layers' LoRA-over-NF4 compute
  in ``poros.core.fused``'s kernels; on a ``poros.moe`` model, its extras beside the fused experts
  (attention, RMSNorms, the head and loss: ``poros.core.fused.extras``).
* ``moe_expert_compute``: a ``poros.moe`` model's expert compute: ``fused`` (``poros.moe.fused``),
  ``reference`` or ``grouped``.

``choose`` decides from the config and the machine (CUDA, Triton) and, for the plan, the model's
experts' activation. A choice that also depends on what loads -- a dense model's adapter runtime,
the kernels' own availability, which of a MoE model's layers the fused kernels can compute --
is settled by the trainer once the model is there (``"auto"`` for the MoE experts: fused where
every layer qualifies, ``poros.moe.prepare.prepare_moe``). The trainer records what ran as the
config's ``ran_compute`` (beside the settings, as ``ran_path`` is beside ``path``) and why a
default took the slower compute as ``compute_reason``. Torch-free.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any, Optional

__all__ = ["Choice", "choose", "triton_installed"]


@dataclass(frozen=True)
class Choice:
    """``choose``'s decision."""

    #: the fused compute (dense: the decoder layers; MoE: the extras beside fused experts)
    fused: bool
    #: a MoE model's expert compute: "fused", "auto" (fused where every layer qualifies, the
    #: model decides), "reference" or "grouped"; None for a dense model
    moe_expert: Optional[str]
    #: why a default took the slower compute, where it did (None: it did not, or as set)
    reason: Optional[str]
    #: why the fused compute cannot run here, whatever the setting (None: it can, or it is set
    #: off): the words a refused ``fused_compute: true`` gives
    blocked: Optional[str] = None


def triton_installed() -> bool:
    """Whether Triton can be imported here (found, not imported)."""
    return importlib.util.find_spec("triton") is not None


def choose(cfg: Any, *, entry: str, cuda: bool, triton: Optional[bool] = None,
           silu_experts: bool = True, gpu_reason: Optional[str] = None) -> Choice:
    """The compute a run of ``cfg`` takes on a machine with (``cuda``) or without a CUDA GPU and
    with ``triton`` (None: whether it is installed here). ``entry`` is ``"dense"`` or ``"moe"``;
    ``silu_experts=False`` for a MoE model whose experts the fused kernels cannot compute (the
    plan's knowledge: ``poros.moe.arch.experts_silu``; the trainer leaves it to the model).
    ``gpu_reason``: why this GPU cannot run the Triton kernels (``poros.core.card.triton_reason``),
    None where it can or is not known."""
    if triton is None:
        triton = triton_installed()
    if entry == "moe":
        expert, why = _moe_expert(cfg, cuda=cuda, triton=triton, silu_experts=silu_experts,
                                  gpu_reason=gpu_reason)
        return Choice(expert in ("fused", "auto") and cfg.fused_compute is not False, expert,
                      why)
    if cfg.fused_compute is False:
        return Choice(False, None, None)
    why = ("the fused compute needs CUDA" if not cuda
           else "the fused compute needs Triton, which is not installed" if not triton
           else gpu_reason)
    return Choice(why is None, None, why if cfg.fused_compute is None else None, why)


def _moe_expert(cfg: Any, *, cuda: bool, triton: bool, silu_experts: bool,
                gpu_reason: Optional[str] = None) -> tuple[str, Optional[str]]:
    if cfg.moe_expert_compute is not None:
        return cfg.moe_expert_compute, None
    if cfg.moe_expert_grouping is not None:  # a setting of the reference compute's
        return "reference", (f"moe_expert_grouping: {cfg.moe_expert_grouping} groups the "
                             "reference compute's rows")
    why = ("the fused expert compute needs CUDA" if not cuda
           else "the fused expert compute needs Triton, which is not installed" if not triton
           else f"the fused expert compute: {gpu_reason}" if gpu_reason
           else f"torch_dtype {cfg.torch_dtype}: the fused expert compute is bf16"
           if cfg.torch_dtype != "bfloat16"
           else f"store {cfg.store}: the fused expert compute reads the RAM or SSD store"
           if cfg.store not in ("ram", "ssd")
           else "fused_compute: false" if cfg.fused_compute is False
           else "the model's experts are not SiLU, which the fused kernels compute"
           if not silu_experts
           else None)
    return ("reference", why) if why else ("auto", None)
