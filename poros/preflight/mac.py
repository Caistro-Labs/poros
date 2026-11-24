"""The Mac's branch of the preflight: what a run on Apple silicon checks before it loads.

A Mac is one pool of memory with no CUDA: nothing here may run CUDA arithmetic (``torch.cuda``,
the CUDA plan's host-RAM rules, the card's ``check``). Each step below is the Mac's side of a
CUDA step in ``poros.preflight.start`` / ``fit``:

* ``card(cfg)`` -- the Mac's equivalent of ``poros.core.card.check``: ``poros.metal.require``'s
  stack and class check. ``poros train`` calls it before its gate; PorosTrainer calls
  ``metal.require`` itself when it is made.
* ``before_download(cfg, check_fit=..., check_host_ram=...)`` -- PorosTrainer's (and so
  ``poros.Run``'s) fit before the load: the same ``poros.preflight.fit.gate`` CUDA runs, on this
  Mac's plan (``poros.metal.plan.unified_plan``), so a run the plan says cannot fit is refused
  (``PorosVramError``, E_OOM_PREDICTED) before the download, as ``poros train``'s gate refuses
  it; ``check_fit=False`` skips it. The Mac adapts nothing a config set: the table fills only
  what it left unset (``poros.metal.table.fill``), so a refusal names what would fit instead
  (``would_fit``).
* ``host_ram_applies(gpu)`` -- whether the CUDA host-RAM refusal applies: never on one pool.
  The Mac's own is ``memory_warning``: the plan's verdict already holds the whole pool less
  macOS's share, so memory other apps hold now warns (the memory guard stops a run that
  swaps), and never refuses.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

logger = logging.getLogger("poros.train.trainer")  # the trainer's own log, as on CUDA


def card(cfg: Any) -> None:
    """This Mac runs what ``cfg`` asks (``poros.metal.require``, without the machine check, which
    the run's lock holder makes), or a refusal."""
    from poros import metal

    metal.require(cfg, check=False)


def before_download(cfg: Any, *, check_fit: bool = True, check_host_ram: bool = True) -> None:
    """The trainer's Mac preflight before the load: the fit gate on this Mac's plan, its
    decisions said once, then its refusal (``check_fit``). ``check_host_ram`` is the CUDA
    flag's twin: one pool has no separate host RAM to refuse (``host_ram_applies``)."""
    from poros import metal
    from poros.core.arch_registry import registry_type
    from poros.metal import table
    from poros.plan.configs import read_model_config
    from poros.preflight import fit

    if table.pick(cfg.model_name_or_path, metal.device_class()) is None:
        return  # no row: a developer's own folder, nothing to plan (metal.settle refused the rest)
    model_json = read_model_config(cfg.model_name_or_path, cfg.model_revision)
    fit.gate(cfg, registry_type(model_json) if model_json else None, model_json,
             check_fit=check_fit, check_host_ram=check_host_ram, say=logger.warning)


def would_fit(cfg: Any, plan: Callable[[Any], Any]) -> str:
    """For a run this Mac's plan refuses: the longest length, halved from the run's down to the
    table's shortest, that the plan does not refuse, at the run's store and then from the SSD;
    "" where none is. A refusal's next step, in its words."""
    from poros.metal.table import LENGTHS

    stores = [cfg.store] + (["ssd"] if cfg.store != "ssd" else [])
    seq = cfg.max_seq_length // 2
    while seq >= LENGTHS[0]:
        for store in stores:
            upd = {"max_seq_length": seq, **({"store": store} if store != cfg.store else {})}
            q = plan(cfg.model_copy(update=upd))
            if q.fits is not False:
                said = ", ".join(f"{k}: {v}" for k, v in upd.items())
                return said + " is not refused"  # what the check knows: never "fits"
        seq //= 2
    return ""


# The Mac's memory_warning heads its note with this, not "may not fit": the run fits this Mac,
# and only the memory free at this moment is short.
SHORT = "memory is short right now"


def memory_warning(p: Any, guard: bool = True) -> Optional[str]:
    """The Mac's host-RAM check: a measured peak past the memory free now (other apps hold the
    rest). A warning, never a refusal: the verdict sets macOS's share aside, not other apps',
    and the memory guard (poros.metal.watchdog, on unless memory_guard=false: ``guard``) stops a
    run that swaps. None where no peak was measured or it fits."""
    need, free = p.peak_vram_allocated, p.host_ram_free_gib
    if need.source != "measured" or need.value is None or free is None or need.value <= free:
        return None
    return (f"This Mac has {free} GiB free right now, less than a run of {p.model} used when it "
            "was measured on this Mac. Close other apps, or expect it to swap"
            + (": the memory guard stops a run whose swap grows." if guard else "."))


def warning_head(p: Any, warning: str) -> str:
    """The word a fit warning is said under: SHORT for the Mac's memory_warning, else what a
    CUDA estimate's warning has always been under."""
    return SHORT if warning in (memory_warning(p), memory_warning(p, guard=False)) else (
        "may not fit")


def host_ram_applies(gpu: Optional[Any]) -> bool:
    """Whether the CUDA host-RAM refusal applies to a plan of ``gpu``: never for a unified pool."""
    return gpu is None or not getattr(gpu, "unified", False)
