"""What the compute refuses, refused before the download: a set ``fused_compute: true`` or
``moe_expert_compute: fused`` that cannot run here, including at a LoRA rank whose kernels do not
fit this card's shared memory (``poros.core.card.fused_rank_reason``). ``refuse`` is what
``PorosTrainer.compute.preflight`` runs; ``install_dense``, ``prepare_moe`` and ``install_moe``
(``poros.train.compute``) keep the same reasons as guards after the load.

The decision itself (fused or reference, the expert compute) is ``poros.core.compute.choose``,
for a run and its plan alike.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Optional

from poros.core.errors import PorosConfigError

# What a set fused_compute: true at a rank its kernels do not fit is told to do (preflight, and
# install_dense's guard after the load).
_RANK_REMEDY = ("Set lora_rank: 128 or lower (the fused kernels take it on every supported GPU, "
                "sm_80 and newer), or fused_compute: false (PEFT's modules).")


def _adapter_reason(peft_adapters: bool) -> Optional[str]:
    """Why the dense fused compute cannot take these adapters, or None: it computes PEFT's LoRA
    over NF4. One rule, read from the config before the download (preflight) and from the
    loaded model after it (install_dense)."""
    if not peft_adapters:
        return "the fused compute needs PEFT adapters"
    return None


def _rank_reason(trainer: Any, entry: str, model_json: Optional[dict],
                 model_type: Optional[str]) -> Optional[str]:
    """Why the fused kernels (``entry`` "dense") or the fused expert kernels ("moe") do not fit
    this CUDA card at the config's LoRA rank, from the ahead-of-time table
    (poros.core.card.fused_rank_reason): what the card's own launch says after the load,
    said before the download. None where they fit, or off CUDA."""
    if trainer.device.type != "cuda":
        return None
    from poros.core.card import capability, fused_rank_reason
    from poros.plan.receipts import rope_head_dim

    head = rope_head_dim(model_json, model_type) if entry == "dense" else None
    return fused_rank_reason(capability(trainer.device), trainer.config.lora_rank,
                             entry=entry, head_dim=head)


def refuse(trainer: Any, entry: str, model_json: Optional[dict] = None,
           model_type: Optional[str] = None) -> None:
    """Before the download: the refusals install_dense, prepare_moe and install_moe would
    give once the model loaded, decided from the config, this machine and config.json (a
    set ``fused_compute: true`` or ``moe_expert_compute: fused`` that cannot run, including
    at a LoRA rank whose kernels do not fit this card's shared memory: the ahead-of-time
    table, poros.core.card.FUSED_FIT). The post-load checks stay as guards,
    unreachable for what this refuses."""
    from poros.core.compute import choose, triton_installed

    cfg = trainer.config
    cuda = trainer.device.type == "cuda"
    if entry == "dense":
        if cfg.fused_compute is not True:
            return
        choice = choose(cfg, entry="dense", cuda=cuda)
        why = choice.blocked
        if why is None:
            why = _adapter_reason(cfg.adapter_backend != "native"
                                  and importlib.util.find_spec("peft") is not None)
        if why is not None:
            raise PorosConfigError(f"fused_compute=True cannot run here: {why}.",
                                   remediation="Set fused_compute: false (PEFT's modules).")
        why = _rank_reason(trainer, "dense", model_json, model_type)
        if why is not None:
            raise PorosConfigError(f"fused_compute=True cannot run here: {why}.",
                                   remediation=_RANK_REMEDY)
        return
    if cfg.moe_expert_compute is None and cfg.fused_compute is not True:
        return
    if model_json is None:
        from poros.plan.configs import read_model_config

        model_json = read_model_config(cfg.model_name_or_path, cfg.model_revision)
    from poros.moe.arch import experts_silu

    silu = experts_silu(model_json, model_type) if model_json else True
    if cfg.moe_expert_compute == "fused" and not silu:
        raise PorosConfigError(
            "moe_expert_compute: fused: the fused expert kernels cannot run this model: "
            "its experts are not SiLU",
            remediation="Leave moe_expert_compute unset (the reference compute where fused "
                        "cannot run), or set it to reference.")
    if cfg.fused_compute is True:
        choice = choose(cfg, entry="moe", cuda=cuda, triton=triton_installed(),
                        silu_experts=silu)
        if not (cuda and choice.moe_expert in ("fused", "auto") and choice.fused):
            raise PorosConfigError(
                "fused_compute=True on a MoE model needs moe_expert_compute: fused on a CUDA "
                "GPU" + (f" ({choice.reason})" if choice.reason else "") + ".",
                remediation="Set moe_expert_compute: fused, or fused_compute: false.")
    if cfg.moe_expert_compute == "fused" or cfg.fused_compute is True:
        why = _rank_reason(trainer, "moe", model_json, model_type)
        if why is not None:
            setting = ("moe_expert_compute: fused" if cfg.moe_expert_compute == "fused"
                       else "fused_compute=True")
            raise PorosConfigError(
                f"{setting} cannot run here: {why}.",
                remediation="Set lora_rank: 128 or lower (the fused expert kernels take it on "
                            "every supported GPU, sm_80 and newer), or leave "
                            "moe_expert_compute unset (the reference compute where fused "
                            "cannot run).")
