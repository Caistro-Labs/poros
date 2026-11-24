"""Will this run fit: the path and the tables adapted before anything is refused, then the
refusals, from the run's plan (``poros.plan.make.make_plan``). One rule for ``poros train``'s gate,
``PorosTrainer.from_pretrained`` (dense and MoE) and ``poros.Run``:

1. ``decide_path_to_fit``: a ``path`` left to be decided whose plan is short of host RAM, or of
   the GPU, takes the path that fits (``path_that_fits_host``, ``path_that_fits_gpu``). Every path
   trains the same update, bit for bit, so adapting changes no value.
2. ``decide_offload_for_run``: ``offload_embeddings_and_head`` left unset, on a dense model on a
   CUDA GPU, as the plan decided it.
3. ``estimate_warning``: an estimate, an upper bound, still past the card warns (with the
   ``block_size`` the plan puts within it); an estimate never refuses.
4. ``vram_error``: a plan that says the run cannot fit the GPU refuses (``PorosVramError``, exit
   3, ``E_OOM_PREDICTED``), with what would fit (``what_would_fit``).
5. ``host_ram_error``: a peak host RAM past what is available refuses (``PorosHostRamError``);
   for a MoE model on the RAM store whose SSD store would fit, the remedy names that first
   (``ssd_alternative``). On a Mac (one pool, ``poros.preflight.mac``) the plan's verdict holds
   the whole pool: no host-RAM refusal, and memory other apps hold now warns
   (``mac.memory_warning``); a refusal names the length that would fit (``mac.would_fit``).

``check`` takes the five in that order and returns them (``Fit``); ``refuse`` raises what a run
refuses on its flags; ``gate`` is both, for the trainer. ``poros train``'s gate renders the same
``Fit`` (``poros.cli.commands.train._plan_gate``).

Torch-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

from poros.core.card import card_refusal
from poros.core.errors import PorosHostRamError, PorosVramError
from poros.core.schemas import PlanGpu, PlanOutput
from poros.plan.make import _holds_less, make_plan, over_card
from poros.preflight import mac
from poros.preflight.arch import support_of

if TYPE_CHECKING:
    from poros.core.config import PorosConfig

Planner = Callable[["PorosConfig"], PlanOutput]

def path_that_fits_host(cfg: "PorosConfig", plan: Callable[["PorosConfig"], PlanOutput]
                        ) -> Optional[tuple[str, str]]:
    """Where a run leaves ``path`` to be decided and the path it would take does not fit host RAM,
    the one of ``fast`` and ``sequential`` that does (fitting the GPU too), with the reason, else
    None. ``plan`` plans a config as the caller does (make_plan with its model and machine)."""
    if cfg.path is not None and not cfg.path_decided:
        return None
    first = plan(cfg)
    if first.host_fits is not False or first.path in (None, "sequential"):
        return None
    for alt in ("fast", "sequential"):
        p = plan(cfg.model_copy(update={"path": alt}))
        if p.host_fits is not False and p.fits is not False:
            return alt, (f"the check turned down {first.path}: its block inputs are past the "
                         "host RAM available")
    return None


def path_that_fits_gpu(cfg: "PorosConfig", plan: Callable[["PorosConfig"], PlanOutput]
                       ) -> Optional[tuple[str, str]]:
    """Where a run leaves ``path`` to be decided and the plan says the path it would take does not
    fit the GPU (``fits`` false: a measured peak or a run no larger's), or is past it by an
    estimate (an upper bound: it may not fit, and the run could stop at its first step), the
    path that does, with the reason, else None: one the plan says fits, else one whose estimate
    is within the card. Each path's plan has already placed the tables (make_plan: in host RAM
    first, where that fits). Every path trains the same update, bit for bit, so adapting changes
    no value. Never a block-major path for a run the plan put on the sequential loop for a
    reason."""
    if cfg.path is not None and not cfg.path_decided:
        return None
    first = plan(cfg)
    short = first.fits is False
    if not (short or (first.fits is None and over_card(first))) or first.path is None:
        return None
    alts = [a for a in ("efficient", "sequential", "fast") if a != first.path]
    if first.path == "sequential" and first.path_reason:
        return None  # block-major cannot train this run's update
    plans = [(a, plan(cfg.model_copy(update={"path": a}))) for a in alts]
    # another path's evidence that it fits first, then one the plan does not put past the card
    wants = [lambda q: q.fits is True, lambda q: q.fits is None and not over_card(q)]
    for want in wants:
        for alt, p in plans:
            if want(p) and p.host_fits is not False:
                return alt, (f"the check turned down {first.path}: "
                             + ("it is past this GPU's memory" if short
                                else "an estimate puts it past this GPU's memory"))
    return None


def block_size_advice(cfg: "PorosConfig", plan: Callable[["PorosConfig"], PlanOutput]
                      ) -> Optional[str]:
    """For a dense run whose plan is past the card by an estimate: the largest ``block_size``
    below the run's that the plan puts within the card, else the smallest one's plan where it
    holds less; None where no smaller block helps. Advice only: ``block_size`` is never changed
    for a run (it defaults to the validated 4, never guessed from free VRAM), so a warning
    names it."""
    first = plan(cfg)
    if first.entry != "dense" or not over_card(first):
        return None
    least = None
    for b in range((cfg.block_size or 4) - 1, 0, -1):
        q = plan(cfg.model_copy(update={"block_size": b}))
        if q.fits is True or (q.fits is None and not over_card(q)):
            return f"block_size: {b} draws no such warning"
        least = (b, q)
    if least is not None and _holds_less(least[1], first):
        return f"block_size: {least[0]} holds less"
    return None


def estimate_warning(cfg: "PorosConfig", plan: Callable[["PorosConfig"], PlanOutput]
                     ) -> Optional[str]:
    """After a run's path and tables were chosen to fit (decide_path_to_fit,
    decide_offload_for_run): where its estimate, an upper bound, is still past the card, what to
    warn, with the ``block_size`` advice (block_size_advice); else None. An estimate never
    refuses."""
    return _estimate(cfg, plan)[0]


def _estimate(cfg: "PorosConfig", plan: Callable[["PorosConfig"], PlanOutput]
              ) -> tuple[Optional[str], Optional[str]]:
    """(estimate_warning's words, the block_size advice in them) for ``cfg``."""
    p = plan(cfg)
    if p.fits is not None or not over_card(p):
        return None, None
    try:  # advice only: never the reason a warning fails
        advice = block_size_advice(cfg, plan)
    except Exception:
        advice = None
    said = (f"{cfg.model_name_or_path}: an estimate puts it past this GPU's memory; the run may "
            "run out of GPU memory at its first step")
    if advice:
        said += f". {advice}; block_size is never changed for you"
    return said, advice


SHORTEST_SEQ = 128  # the shortest max_seq_length the plan reads: what_would_fit halves down to it


def what_would_fit(cfg: "PorosConfig", plan: Callable[["PorosConfig"], PlanOutput]) -> str:
    """For a run nothing fits (every path past the GPU): the settings, and the GPU, that would --
    the first of a shorter ``max_seq_length`` (halved down to 128) and, dense, ``block_size`` 1
    that the plan says fits, and, where a measured run puts a floor under some path at these
    settings, a GPU with more free memory (said with the memory this card has now, never the
    plan's figure)."""
    first = plan(cfg)
    tables = ([{"offload_embeddings_and_head": True}]
              if first.entry == "dense" and cfg.offload_embeddings_and_head is None else [{}])
    need = []  # (what the run needs at least, GiB allocated, the settings): a peak or a floor
    for a in ("efficient", "sequential", "fast"):
        for tb in [{}] + [x for x in tables if x]:
            upd = {"path": a, **tb}
            q = plan(cfg.model_copy(update=upd))
            low = (q.bound.peak_gib if q.bound is not None and q.bound.fits is False
                   else q.peak_vram_allocated.value
                   if q.peak_vram_allocated.source == "measured" else None)
            if low is not None:
                need.append((low, ", ".join(f"{k}: {v}" for k, v in upd.items())))
    said = []
    tries = []
    seq = cfg.max_seq_length
    while seq > SHORTEST_SEQ:
        seq //= 2
        tries.append({"max_seq_length": seq})
    if first.entry == "dense" and cfg.block_size != 1:
        tries = [{"block_size": 1}] + [{**t, "block_size": 1} for t in tries]
    for t in tries:
        for path in (None, "efficient", "sequential"):
            upd = {**t, **({"path": path} if path else {})}
            q = plan(cfg.model_copy(update=upd))
            if q.fits is True:
                said.append(", ".join(f"{k}: {v}" for k, v in upd.items()) + " is not refused")
                break
        if said:
            break
    if first.entry == "dense" and getattr(cfg, "resident_blocks", 0):
        # resident blocks hold their layers on the GPU at any length: the first thing to try
        said.insert(0, "fewer resident_blocks (0 keeps none on the GPU)")
    if need:
        free = first.gpu.free_gib if first.gpu and first.gpu.free_gib is not None else None
        # what this card has, read here: never a planned figure
        here = (f"{free} GiB is free here" if free is not None
                else f"it has {first.gpu.total_gib} GiB" if first.gpu else "")
        said.append("a GPU with more free memory than this one" + (f": {here}" if here else ""))
    return "; ".join(said)


def path_to_fit(cfg: "PorosConfig", plan: Callable[["PorosConfig"], PlanOutput]
                ) -> Optional[tuple[str, str]]:
    """The rule for ``path`` left to be decided: where the path the run would take is short of
    host RAM, or of the GPU, the one that fits (path_that_fits_host, path_that_fits_gpu), with its
    reason; None to keep it. Pure: ``decide_path_to_fit`` sets it on the config, and the CLI's
    sheets plan with it (``poros.cli.verbs.shared.auto_path``)."""
    return path_that_fits_host(cfg, plan) or path_that_fits_gpu(cfg, plan)


def decide_path_to_fit(cfg: "PorosConfig", plan: Callable[["PorosConfig"], PlanOutput]
                       ) -> Optional[tuple[str, str]]:
    """``path`` left to be decided: the path ``path_to_fit`` picks, set on ``cfg`` with its
    reason, and returned."""
    fits = path_to_fit(cfg, plan)
    if fits is not None:
        cfg.decide_path(*fits)
    return fits


def vram_error(cfg: "PorosConfig", plan: Callable[["PorosConfig"], PlanOutput],
               p: Optional[PlanOutput] = None) -> Optional["PorosVramError"]:
    """The refusal for a run its plan ``p`` (``plan(cfg)``) says cannot fit the GPU (``fits``
    false), after its path and tables were chosen to fit (decide_path_to_fit,
    decide_offload_for_run), with what would fit: one message for ``poros train``'s gate and
    PorosTrainer. None where it fits, may fit, or where the card itself is refused (gpu.check
    says that first, with its own code)."""
    p = p or plan(cfg)
    if p.fits is not False or p.gpu is None or card_refusal(cfg, p.gpu):
        return None
    try:  # no path fits: what settings or GPU would
        would = mac.would_fit(cfg, plan) if p.gpu.unified else what_would_fit(cfg, plan)
    except Exception:  # advice only: never the reason a refusal fails
        would = ""
    return PorosVramError(p.model, p.gpu.name, p.fit_basis, would,
                          any_path=(cfg.path is None or cfg.path_decided) and not p.gpu.unified)


def host_ram_error(p: PlanOutput, ssd_gib: Optional[float] = None, open_paths: tuple = ()
                   ) -> Optional[PorosHostRamError]:
    """The refusal for a plan whose peak host RAM exceeds what is available; ``ssd_gib``: the SSD
    store's peak where it would fit (``ssd_alternative``), named first in the remedy."""
    if p.host_fits is not False:
        return None
    need = p.peak_host_ram
    if need.value is None and p.host_floor_gib is not None:  # no figure: what it must hold
        return PorosHostRamError(p.model, p.host_floor_gib, p.host_ram_free_gib,
                                 "the frozen layers in NF4 alone, sized from config.json: "
                                 "blockwise residency keeps the streamed base in host RAM",
                                 at_least=True)
    basis = need.method or (f"measured on {p.gpu.name}" if p.gpu is not None else "measured")
    return PorosHostRamError(p.model, need.value, p.host_ram_free_gib, basis, ssd_gib=ssd_gib,
                             path=p.path, open_paths=open_paths)


def open_paths(cfg: "PorosConfig", plan: Planner, p: PlanOutput) -> tuple:
    """For a run that set path: efficient by hand and is short of host RAM: the other paths the
    check refuses neither for host RAM nor the GPU, else (). A path left to the run was tried
    already (path_that_fits_host); none is named for it. Advice only."""
    if cfg.path != "efficient" or cfg.path_decided or p.host_fits is not False:
        return ()
    out = []
    for alt in ("fast", "sequential"):
        try:
            q = plan(cfg.model_copy(update={"path": alt}))
        except Exception:  # advice only: never the reason a refusal fails
            continue
        if q.host_fits is not False and q.fits is not False:
            out.append(alt)
    return tuple(out)


def ssd_alternative(cfg: "PorosConfig", plan: Planner, p: PlanOutput) -> Optional[float]:
    """For a MoE run on the RAM store whose plan is short of host RAM: the peak host RAM its plan
    on the SSD store gives (the same settings; the experts in a ``.sss`` file, poros.sss) where
    that fits what is available, else None. A remedy only: the store is never changed for you
    (it writes the experts to a disk you choose)."""
    if p.entry != "moe" or getattr(cfg, "store", "ram") != "ram" or p.host_fits is not False:
        return None
    q = plan(cfg.model_copy(update={"store": "ssd", "store_path": cfg.store_path or "ssd"}))
    if q.host_fits is not False and q.peak_host_ram.value is not None:
        return q.peak_host_ram.value
    return None


def run_planner(model_type: Optional[str], *, gpu: Optional[PlanGpu],
                model_config: Optional[dict], host_free_gib: Optional[float],
                entry: Optional[str] = None) -> Callable[["PorosConfig"], PlanOutput]:
    """How a run plans a config of its model on its machine (make_plan, the support and entry
    support_of gives; ``entry`` where the run has already routed the model): what
    decide_path_to_fit and decide_offload_for_run read."""
    support, routed = support_of(model_type)
    entry = entry or routed

    def plan(cfg: "PorosConfig") -> PlanOutput:
        return make_plan(cfg, model_type=model_type, support=support, entry=entry, gpu=gpu,
                         model_config=model_config, host_free_gib=host_free_gib)

    return plan


def decide_offload_for_run(cfg: "PorosConfig", p: PlanOutput) -> Optional[str]:
    """``offload_embeddings_and_head`` left unset, on a dense model on a CUDA GPU: as the plan
    ``p`` decided it (make_plan, decide_offload). Returns the reason it was turned on, or None."""
    if (cfg.offload_embeddings_and_head is None and p.entry == "dense" and p.gpu is not None
            and not p.gpu.unified):
        cfg.decide_offload(p.offload_reason is not None, p.offload_reason)
        return p.offload_reason
    return None


def planner_here(model_type: Optional[str], model_config: Optional[dict], *,
                 entry: Optional[str] = None) -> Planner:
    """The run's planner on this machine: this GPU (``poros.core.card.detect_gpu``) and the host
    RAM free now."""
    from poros.core.card import detect_gpu
    from poros.core.env import host_ram_free

    return run_planner(model_type, gpu=detect_gpu(), host_free_gib=host_ram_free(),
                       model_config=model_config, entry=entry)


@dataclass
class Fit:
    """What ``check`` decided for a run, and what it would refuse."""

    plan: Planner  # the run's planner
    p: PlanOutput  # its plan of the config as it runs, before the tables were placed
    path: Optional[tuple[str, str]]  # (path, why): a path decided to fit
    offload: Optional[str]  # why the tables were put in host RAM
    warning: Optional[str]  # an estimate past the card
    vram: Optional[PorosVramError]  # the plan says it cannot fit the GPU
    host: Optional[PorosHostRamError]  # its peak host RAM is past what is available
    advice: Optional[str] = None  # the block_size advice in an estimate's warning, as said


def check(cfg: "PorosConfig", plan: Planner, *,
          as_runs: Optional[Callable[["PorosConfig"], "PorosConfig"]] = None) -> Fit:
    """Adapt ``cfg`` (its path, then its tables) to fit, then say what would still be refused,
    in that order. ``as_runs``: ``cfg`` as the plan should read it (``poros train``: a recipe
    whose rows fit one micro-batch runs every step as that one)."""
    as_runs = as_runs or (lambda c: c)
    path = decide_path_to_fit(cfg, plan)
    p = plan(as_runs(cfg))
    offload = decide_offload_for_run(cfg, p)  # the plan decided it as the run does
    # an estimate, an upper bound, still past the card after the path and the tables adapted: a
    # warning, with the block_size the plan puts within it (never a refusal)
    warning, advice = (_estimate(as_runs(cfg), plan) if p.fits is None and over_card(p)
                       else (None, None))
    if mac.host_ram_applies(p.gpu):  # host RAM apart from the GPU's: CUDA's refusal
        host = host_ram_error(p, ssd_alternative(as_runs(cfg), plan, p),
                              open_paths(as_runs(cfg), plan, p))
    else:  # one pool: the plan's verdict holds it; memory other apps hold now warns
        host, warning = None, warning or mac.memory_warning(p, guard=cfg.memory_guard)
    return Fit(plan=plan, p=p, path=path, offload=offload, warning=warning,
               vram=vram_error(as_runs(cfg), plan, p), host=host, advice=advice)


def refuse(fit: Fit, *, check_fit: bool = True, check_host_ram: bool = True) -> None:
    """Raise what a run refuses before the download, each on its own flag: the GPU fit
    (``check_fit``), then the host RAM (``check_host_ram``). False: ``--force``."""
    if check_fit and fit.vram is not None:
        raise fit.vram
    if check_host_ram and fit.host is not None:
        raise fit.host


def at_no_length(cfg: "PorosConfig", model_type: Optional[str], model_config: Optional[dict], *,
                 entry: Optional[str] = None) -> Optional[PorosVramError]:
    """For a load that leaves the GPU fit to train() (PorosTrainer with no max_seq_length set):
    the refusal before the download for a run that fits this GPU at no length. Asked at
    ``SHORTEST_SEQ`` (a longer run holds no less), the path and tables adapted as the run's
    would be, on a copy: the load still plans at its own length. None where the plan says it
    fits or may fit there; train() asks again at the length it trains."""
    short = cfg.model_copy(deep=True, update={
        "max_seq_length": min(SHORTEST_SEQ, cfg.max_seq_length)})
    found = check(short, planner_here(model_type, model_config, entry=entry))
    err, p = found.vram, found.p
    if err is None or p.gpu is None:
        return err
    return PorosVramError(
        p.model, p.gpu.name,
        f"{err.basis}; that is at max_seq_length {short.max_seq_length}, the shortest the plan "
        "reads, so it fits at no length", err.would,
        any_path=(short.path is None or short.path_decided) and not p.gpu.unified,
        shortest=short.max_seq_length)


def gate(cfg: "PorosConfig", model_type: Optional[str], model_config: Optional[dict], *,
         entry: Optional[str] = None, check_fit: bool = True, check_host_ram: bool = True,
         say: Callable[..., None]) -> Fit:
    """``check`` on this machine, each decision said once through ``say`` (a logger's warning),
    then ``refuse``: what PorosTrainer runs before the download (dense and MoE)."""
    said = cfg.offload_reason  # decided before here (poros train says that one itself)
    fit = check(cfg, planner_here(model_type, model_config, entry=entry))
    if fit.path is not None:
        say("path: %s: %s", *fit.path)
    if cfg.offload_reason and cfg.offload_reason != said:  # decided here, for a reason: once
        say("tables: host RAM: %s", cfg.offload_reason)
    if fit.warning and check_fit:
        say("%s: %s", mac.warning_head(fit.p, fit.warning), fit.warning)
    refuse(fit, check_fit=check_fit, check_host_ram=check_host_ram)
    return fit
