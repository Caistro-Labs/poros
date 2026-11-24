"""A plan for a Mac: one pool of memory, so one verdict, read from the table. Torch-free.

Fits needs a receipt for this config on this class: a whole model's real run
(docs/validation/metal/). Won't fit needs the floor over the pool. In between the verdict is
unknown: no figure here is ever interpolated or guessed. A row derived from a model's config
(``poros.metal.table.derived``) has no receipt, so its verdict is never fits, and the plan says its
floor is arithmetic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

from poros.core.arch_registry import validated_arch_types
from poros.core.card import _norm
from poros.core.errors import PorosError
from poros.core.schemas import Figure, PlanGpu, PlanOutput
from poros.metal import table, validate
from poros.metal.machine import GB_PER_GIB
from poros.plan import cells
from poros.plan.estimate import _unknown
from poros.plan.receipts import _IGNORE, Request

if TYPE_CHECKING:
    from poros.core.config import PorosConfig

NOT_HERE = "this model does not train on a Mac in this release"
NOT_YET = "not measured on this Mac yet"
DERIVED = "derived from its config, never measured on this Mac"
DERIVED_SAYS = ("a row derived from its config: its lengths are the nearest model's of its kind, "
                "its memory and disk arithmetic on its shapes, none of it measured")
PLANNED = "derived for this Mac from an M4 with 16 GB; unknown until a run here measures it"
PLANNED_SAYS = ("a row derived for this Mac's memory from an M4 with 16 GB: its memory is "
                "arithmetic on what runs there held, and its speed unknown until a run here "
                "measures it")


def _planned_fit(cfg: "PorosConfig", gpu: PlanGpu, reserve: float,
                 pool: str) -> tuple[Optional[bool], str]:
    """A derived row's verdict: no run here measured it, so never fits. Won't fit: what a run of
    it held from this tier on the reference, at this length, is past this Mac's memory less what
    macOS keeps, or its GPU part past the working set."""
    model, seq, tier = cfg.model_name_or_path, cfg.max_seq_length, cfg.store
    need = table.need_gb(model, seq, tier) / GB_PER_GIB
    on_gpu = table.gpu_gb(model, seq, tier)
    how = ("by arithmetic on what runs of it held on an M4 with 16 GB" if table.held(model, tier)[2]
           else "by arithmetic on its config")
    where = "from the SSD" if tier == "ssd" else "from RAM"
    said = f"needs about {need:.1f} GiB at {seq} tokens {where}, {how}"
    if need + reserve > gpu.total_gib:
        return False, f"{said}; {pool}"
    if on_gpu is not None and gpu.working_set_gib and on_gpu / GB_PER_GIB > gpu.working_set_gib:
        return False, (f"{said}, {on_gpu / GB_PER_GIB:.1f} GiB of it on the GPU, past the "
                       f"{gpu.working_set_gib:.1f} GiB macOS lets it hold")
    return None, f"{said}, not measured here; {pool}"


def held_where_read(cfg: "PorosConfig", row: Any) -> bool:
    """store=ram set for a catalog model this Mac's table reads from the SSD: every layer held,
    which the table's own figures can size (a folder takes its family's row, whose runs say
    nothing of the folder's own size)."""
    from poros.cli.words.models import by_id

    return (cfg.store == "ram" and row is not None and row.tier == "ssd"
            and by_id(cfg.model_name_or_path) is not None)


def unified_plan(
    cfg: "PorosConfig",
    *,
    model_type: Optional[str],
    gpu: PlanGpu,
    download: Optional[Figure],
    host_free_gib: Optional[float],
) -> PlanOutput:
    here_class = table.class_of(gpu.name)
    row = table.fill(cfg, here_class)
    refused = "" if row else (table.refusal(cfg.model_name_or_path, here_class) or NOT_HERE)
    try:
        validate(cfg)
    except PorosError as exc:
        refused = refused or exc.info.message
    derived = row is not None and row.derived
    planned = row is not None and bool(row.planned_from)
    unmeasured = DERIVED if derived else PLANNED if planned else NOT_YET
    entry = ("moe" if model_type in validated_arch_types("moe") else "dense") if row else None
    made = table.derived(cfg.model_name_or_path) if derived else None
    if made is not None:  # its kind is what its config builds
        entry = made.entry.entry
    req = Request.of(cfg, entry or "dense", cuda=False)
    req.selfcheck, req.load_strategy = False, f"metal_{cfg.store}"

    def figure(name: str, unit: str) -> Figure:
        here = [c for c in cells.MAC_CELLS if name in c["fields"]
                and _norm(c["key"]["gpu"]) == _norm(gpu.name)
                and req.matches(c["key"], ignore=_IGNORE[name])]
        if here and not refused and not derived:
            return Figure(source="measured", **here[0]["fields"][name])
        return _unknown(unit, refused or unmeasured)

    peak, visible = figure("peak_vram_allocated", "GiB"), figure("gpu_visible", "GiB")
    s_step, load = figure("s_per_step", "s"), figure("load_time", "s")
    if cfg.max_steps > 0 and s_step.value is not None and load.value is not None:
        total = Figure(value=round(load.value + cfg.max_steps * s_step.value, 1), unit="s",
                       source="estimated", receipt=s_step.receipt,
                       method=f"load_time + {cfg.max_steps} steps x s_per_step")
    else:
        total = _unknown("s", refused or unmeasured)

    reserve = table.os_reserve_gb(table.memory_of(here_class)) / GB_PER_GIB
    pool = f"{gpu.total_gib:.1f} GiB here, less {reserve:.1f} GiB for macOS"
    fits, basis = None, peak.why
    if peak.value is not None and visible.value is not None:
        limit = gpu.working_set_gib or visible.value
        fits = peak.value + reserve <= gpu.total_gib and visible.value <= limit
        basis = f"peaked at {peak.value} GiB, {visible.value} GiB of it on the GPU; {pool}"
    elif not refused and planned:
        fits, basis = _planned_fit(cfg, gpu, reserve, pool)
    elif not refused and held_where_read(cfg, row):
        fits, basis = _planned_fit(cfg, gpu, reserve, pool)
        if fits is False:
            # an estimate (the SSD run's peak plus the whole pack, which errs toward refusing)
            basis += ("; that is an estimate: drop store=ram, and this Mac reads its layers from "
                      "the SSD, or try it anyway with --force")
    elif not refused:
        floor = table.floor_gb(cfg.model_name_or_path, cfg.max_seq_length) / GB_PER_GIB
        how = ", by arithmetic on its config" if derived else ""
        if floor + reserve > gpu.total_gib:
            fits = False
            basis = f"needs at least {floor:.1f} GiB at {cfg.max_seq_length} tokens{how}; {pool}"
        elif derived:
            basis = (f"needs at least {floor:.1f} GiB at {cfg.max_seq_length} tokens{how}, "
                     f"not measured; {pool}")

    warnings = [PLANNED_SAYS] if planned else []
    warnings += [DERIVED_SAYS] if derived else []
    if row is not None:
        weights = "read from disk" if cfg.store == "ssd" else "held in memory"
        warnings.append(f"{cfg.max_seq_length} tokens per example, "
                        f"{cfg.gradient_accumulation_steps} per step, weights {weights}")
        if s_step.value is None and row.seq != cfg.max_seq_length:
            warnings.append("no measured time for this length")
    one_pool = _unknown("GiB", "one pool: the peak above")
    return PlanOutput(
        model=cfg.model_name_or_path, model_type=model_type,
        support=("unsupported" if refused else "user_asserted" if derived
                 else "validated" if row.evidence in ("validated", "checked") else "unknown"),
        entry=entry, gpu=gpu, fits=fits, fit_basis=basis,
        host_ram_free_gib=host_free_gib, host_fits=None, host_fit_basis="one pool",
        peak_vram_allocated=peak, peak_vram_reserved=one_pool, peak_host_ram=one_pool,
        s_per_step=s_step, load_time=load, total_time=total,
        download=download or _unknown("GiB", "not computed"),
        excludes=[f"other apps: the verdict sets {reserve:.1f} GiB aside for macOS",
                  "a first run's download and pack: slower, and its peak can run higher"],
        warnings=warnings, store=cfg.store, gpu_visible=visible,
    )
