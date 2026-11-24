"""``make_plan``: a config's plan on a GPU, every figure measured, estimated (with its method)
or unknown (with why), and the fit verdict the preflight reads (``poros.preflight.fit``).
Torch-free."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Optional

from poros.core.card import (
    _norm,
    card_capability,
    card_refusal,
    fused_rank_reason,
)
from poros.core.config import ssd_cache_bytes, train_path
from poros.core.env import under_agent
from poros.core.schemas import Figure, PlanGpu, PlanOutput, PlanStore
from poros.moe.arch import experts_silu
from poros.plan import cells as _cells  # the receipt set, read at call time
from poros.plan.estimate import (
    _ceil1,
    _nf4_host_floor,
    _sizes,
    _unknown,
    _upper,
    moe_host_ram,
    moe_ssd_s_per_step,
    path_extra,
)
from poros.plan.receipts import (
    _OTHER_GPU,
    _RECEIPT_DEFAULTS,
    _RECEIPT_VALUES,
    EXCLUDES,
    PATH_NOTE,
    PATH_NOTE_GUIDE,
    PATH_NOTE_UNSIZED,
    Request,
    _card_free,
    _card_growth,
    _card_matches,
    _grows_request,
    _grows_to_card,
    _highest_of_repeats,
    _ignored,
    _no_path_run,
    _no_run,
    bound,
    nearest,
    rope_head_dim,
)

if TYPE_CHECKING:
    from poros.core.config import PorosConfig


# A plan for a card (--gpu, by name or by memory) that no measured run ran on.
UNMEASURED_CARD = ("no measured run on this card: every figure is an estimate from runs on other "
                   "cards, and all {gib:g} GiB is taken as free (its driver keeps some of it)")


# The share of a card's extra free VRAM a grown expert cache takes: the rest goes to the headroom
# it leaves (poros.sss.tune: max(3 GiB, 0.2 x budget)).
_CACHE_SHARE = 0.8


def _fast_cache_gib(req: Optional[Request], gpu: Optional[PlanGpu], sizes) -> Optional[float]:
    """The VRAM expert cache, GiB, a MoE ``path: fast`` run holds once update 1 sized it: its cap
    (moe_expert_cache_gib), else what a receipt of this config on this GPU grew
    (``expert_cache_size``), moved by ``_CACHE_SHARE`` of the free VRAM this plan reads beyond
    or short of that run's budget, at most every expert; None where neither says."""
    if req is None or req.entry != "moe" or req.path != "fast":
        return None
    if req.expert_cache is not None:
        return float(req.expert_cache)
    if gpu is None:
        return None
    grown = [c for c in _cells.here(gpu.name)
             if {"expert_cache_size", "vram_budget"} <= set(c["fields"])
             and _norm(c["key"]["gpu"]) == _norm(gpu.name) and req.matches(c["key"])]
    if not grown:
        return None
    c = max(grown, key=lambda c: c["fields"]["vram_budget"]["value"])
    cache = c["fields"]["expert_cache_size"]["value"] + _CACHE_SHARE * (
        _card_free(gpu) - c["fields"]["vram_budget"]["value"])
    every = sizes.experts_nf4 / 2**30 if sizes is not None and sizes.experts_nf4 else None
    return max(0.0, cache if every is None else min(cache, every))


def make_plan(
    cfg: "PorosConfig",
    *,
    model_type: Optional[str],
    support: str,
    entry: Optional[str],
    gpu: Optional[PlanGpu],
    download: Optional[Figure] = None,
    model_config: Optional[dict] = None,
    host_free_gib: Optional[float] = None,
    probe=None,
) -> PlanOutput:
    if gpu is not None and gpu.unified:
        from poros.metal.plan import unified_plan

        return unified_plan(cfg, model_type=model_type, gpu=gpu, download=download,
                            host_free_gib=host_free_gib)
    twin = _cells.shape_twin(cfg.model_name_or_path, model_config)
    if twin is not None and any(  # this card's own certification run of it plans it here
            c["key"]["model"] == cfg.model_name_or_path and c.get("only_its_machine")
            for c in _cells.here(gpu.name if gpu else None)):
        twin = None
    if twin is not None:  # no receipt names it (a local folder, a fine-tune): its twin's plan it
        p = make_plan(cfg.model_copy(update={"model_name_or_path": twin}), model_type=model_type,
                      support=support, entry=entry, gpu=gpu, download=download,
                      model_config=model_config, host_free_gib=host_free_gib, probe=probe)
        note = f"planned from the receipts of {twin}: {cfg.model_name_or_path} has its shapes"
        return p.model_copy(update={"model": cfg.model_name_or_path,
                                    "warnings": [*p.warnings, note]})
    cfg = cfg.model_copy()  # what a CUDA run of it trains: config.json says, or its model_type
    cfg.apply_cuda_defaults(model_config or ({"model_type": model_type} if model_type else None))
    if (entry == "dense" and gpu is not None and cfg.offload_embeddings_and_head is None
            and cfg.freeze_embeddings and cfg.freeze_lm_head
            and card_refusal(cfg, gpu) is None):  # a refused card: no table decision to plan
        # offload_embeddings_and_head left unset: decided as a run decides it (decide_offload)
        def tables(on: bool) -> PlanOutput:
            return make_plan(cfg.model_copy(update={"offload_embeddings_and_head": on}),
                             model_type=model_type, support=support, entry=entry, gpu=gpu,
                             download=download, model_config=model_config,
                             host_free_gib=host_free_gib, probe=probe)

        on_gpu = tables(False)
        if not over_card(on_gpu):
            return on_gpu
        off = tables(True)
        if off.fits is False and on_gpu.fits is not True and off.host_fits is not False:
            # the tables in host RAM hold less than on the GPU (only the head's upload, for a
            # pass, against both tables in fp32 for the run): past the card there, past it here
            return on_gpu.model_copy(update={"fits": False, "fit_basis": (
                f"with the embedding table and LM head in host RAM it would not fit "
                f"({off.fit_basis}), and on the GPU they hold more")})
        if off.host_fits is False or (over_card(off) and not _holds_less(off, on_gpu)):
            return on_gpu
        # in host RAM it fits; or, where an estimate puts the run past the card either way, it
        # holds less there: the run least likely to stop at its first step (the same bits)
        cause = ("with them on the GPU the run is past this GPU's memory" if on_gpu.fits is False
                 else "with them on the GPU an estimate puts the run past this GPU's memory")
        why = ("the check keeps the embedding table and LM head in host RAM to leave GPU memory "
               f"for training: {cause}"
               + ("" if not over_card(off) else "; in host RAM the run holds less"))
        return off.model_copy(update={"offload_reason": why, "warnings": [
            *off.warnings, f"offload_embeddings_and_head: on, as a run decides it: {why}"]})
    path = train_path(cfg)
    silu = experts_silu(model_config, model_type)  # the fused MoE compute's own condition
    # a plan for this machine's GPU sees whether Triton is here; one for a named card assumes it
    triton = None if gpu is not None and gpu.source != "spec" else True
    # the card's capability: a rank whose fused kernels do not fit it plans the compute a run
    # there takes (PEFT's modules, or the reference experts)
    cap = card_capability(gpu)
    fit = dict(cap=cap, rope_head_dim=rope_head_dim(model_config, model_type))
    req = Request.of(cfg, entry, silu_experts=silu, triton=triton, **fit) if entry else None
    # The path's own receipts plan it; without one at these settings, the sequential loop's
    # figures at them plus what the path holds on top (path_extra), as estimates.
    seq_req = (Request.of(cfg.model_copy(update={"path": "sequential"}), entry, silu_experts=silu,
                          triton=triton, **fit)
               if entry and path != "sequential" else None)
    extra = path_extra(cfg, model_config, gpu, entry) if seq_req else None
    # Inkling 2 Mini's experts as the run will hold them, judged from what is in hand
    from poros.moe.inkling2.formats import expected

    expert_format, format_note = expected(cfg, model_config)
    sizes = _sizes(model_config, expert_format)
    adds = {"peak_vram_allocated": extra.vram, "peak_vram_reserved": extra.vram,
            "peak_host_ram": extra.host} if extra else {}
    via_seq: set[str] = set()  # figures the sequential loop's receipts stood in for
    unfused: set[str] = set()  # figures PEFT-module receipts stood in for (fused_compute)
    off_default = [f for f in _RECEIPT_DEFAULTS if getattr(cfg, f) not in _RECEIPT_VALUES.get(
        f, (type(cfg).model_fields[f].default,))]

    def cells(name: str, r: Optional[Request] = None) -> tuple[list[dict], list[dict]]:
        """(same config on any GPU, same config on this GPU) for one figure."""
        r = r or req
        same = [c for c in _cells.here(gpu.name if gpu else None)
                if r and not off_default and name in c["fields"]
                and r.matches(c["key"], ignore=_ignored(name, r.path))]
        return same, [c for c in same if gpu and _norm(c["key"]["gpu"]) == _norm(gpu.name)]

    def figure(name: str, unit: str) -> Figure:
        f = unfused_or(name, unit, req)
        if f.value is None and name == "peak_host_ram" and req and not req.offload:
            # a run with its tables offloaded holds them in host RAM: at least this one's
            g = unfused_or(name, unit, dataclasses.replace(req, offload=True))
            if g.value is not None:
                f = g.model_copy(update={"source": "estimated", "method": (
                    f"measured with offload_embeddings_and_head, which keeps the tables in host "
                    f"RAM{f' ({g.method})' if g.method else ''}")})
        if seq_req is None or f.value is not None:
            return f
        s = unfused_or(name, unit, seq_req)
        if s.value is None:
            return f
        via_seq.add(name)
        seq = f"the sequential loop's at these settings{f' ({s.method})' if s.method else ''}"
        if extra is None and unit == "GiB":  # unsized: the path holds more, so no memory figure
            return _unknown(unit, f"{_no_path_run(cfg.model_name_or_path, path)}, and "
                            "config.json, which sizes what it holds on top of the sequential "
                            "loop, was not read")
        if extra is None:  # no sizes: the sequential loop's speed, a guide deciding nothing
            return s.model_copy(update={"source": "estimated", "method": (
                f"{seq}; {_no_path_run(cfg.model_name_or_path, path)}")})
        if name == "s_per_step":  # each block streams once a step instead of once a micro-batch
            return s.model_copy(update={"source": "estimated", "method": (
                f"{seq}, as a guide: on the {path} path at block 4 every pair measured ran "
                "faster")})
        add = adds.get(name, 0.0)
        return s.model_copy(update={"source": "estimated", "value": round(s.value + add, 3),
                                    "method": f"{seq} + {add:.2f} GiB the {path} path holds "
                                              "on top"})

    def unfused_or(name: str, unit: str, r: Optional[Request]) -> Figure:
        """The figure for ``r``; for the fused compute with no receipt of it, PEFT's modules'
        receipt at the same settings, labelled an estimate (docs/validation/fused/ measures the
        two side by side)."""
        f = receipt_figure(name, unit, r)
        if f.value is not None or r is None or not r.fused_compute:
            return f
        g = receipt_figure(name, unit, dataclasses.replace(r, fused_compute=False))
        if g.value is None:
            return f
        unfused.add(name)
        return g.model_copy(update={"source": "estimated", "method": (
            f"measured with fused_compute off{f' ({g.method})' if g.method else ''}; no receipt "
            "of this config ran the fused compute, the default")})

    def receipt_figure(name: str, unit: str, r: Optional[Request] = None) -> Figure:
        same, here = cells(name, r)
        grown = name.startswith("peak_vram") and here and _grows_request(r or req)
        if grown and not all(_card_matches(c, gpu) for c in here):
            return card_grown(name, unit, here)
        if here:  # measured more than once here: the highest
            return Figure(source="measured", **max(
                here, key=lambda c: c["fields"][name]["value"])["fields"][name])
        carried = [c for c in same
                   if not (name.startswith("peak_vram") and _grows_to_card(c["key"]))]
        if name in _OTHER_GPU and carried:
            c = max(carried, key=lambda c: c["fields"][name]["value"])
            method = _OTHER_GPU[name].format(gpu=c["key"]["gpu"], n=len(carried))
            return Figure(source="estimated", method=method, **c["fields"][name])
        if off_default:
            return _unknown(unit, f"every receipt ran {', '.join(off_default)} at the default")
        if under_agent():
            return _unknown(unit, _no_run(req, gpu))
        return _unknown(unit, "no measured run of this config"
                        + (" on this GPU" if same else ""))

    def card_grown(name: str, unit: str, here: list[dict]) -> Figure:
        """A run whose expert cache grows to this card, measured here with another free VRAM
        than this plan's: an estimate, never below the run's -- the receipt's peak where less is
        free (a smaller cache), plus what a larger cache adds where more is (_card_growth)."""
        sized = [(c, g) for c in here if (g := _card_growth(c, gpu, sizes)) is not None]
        if not sized:
            return _unknown(unit, "every receipt of this config grew its expert cache to a "
                                  "card with other free VRAM, and does not say how much")
        c, g = max(sized, key=lambda cg: cg[0]["fields"][name]["value"] + cg[1])
        f = c["fields"][name]
        budget = c["fields"]["vram_budget"]["value"]
        how = (f"+ {g:.2f} GiB for the larger expert cache {_card_free(gpu)} GiB free grows"
               if g else f"no less: with {_card_free(gpu)} GiB free its expert cache grows no "
                         "larger")
        return Figure(value=round(f["value"] + g, 3), unit=unit, source="estimated",
                      receipt=f["receipt"],
                      method=f"{c['id']}'s peak, whose expert cache grew to the "
                             f"{budget} GiB its run could hold, {how}")

    lift = extra.vram if extra else None

    def vram(name: str) -> Figure:
        """This GPU's receipt of this config, else the lowest upper bound any receipt of the
        model gives (_upper), else the sequential loop's figure, unsized."""
        exact = receipt_figure(name, "GiB", req)
        if exact.source == "measured" or req is None or off_default:
            return exact
        bounds = [u for c in _highest_of_repeats(_cells.here(gpu.name if gpu else None), name)
                  if (u := _upper(name, req, c, sizes, gpu, lift))]
        if not bounds:
            return figure(name, "GiB")
        value, method, receipt, lifted = min(bounds, key=lambda b: b[0])
        if lifted:
            via_seq.add(name)
        return Figure(value=round(value, 3), unit="GiB", source="estimated", method=method,
                      receipt=receipt)

    alloc = vram("peak_vram_allocated")
    reserved = vram("peak_vram_reserved")
    # blocks kept on the GPU (resident_blocks): no receipt ran any, so each figure grows by their
    # NF4 bytes, an estimate; and what they and the run's other fixed state certainly hold is a
    # floor, which past the card refuses before the download (_resident_hold)
    hold = _resident_hold(cfg, entry, sizes)
    if hold is not None:
        alloc, reserved = (_with_resident(f, hold) for f in (alloc, reserved))
    s_step = figure("s_per_step", "s")
    load = figure("load_time", "s")
    steps = cfg.max_steps if cfg.max_steps > 0 else None
    if steps and s_step.value is not None and load.value is not None:
        total = Figure(value=round(load.value + steps * s_step.value, 1), unit="s",
                       source="estimated", receipt=s_step.receipt,
                       method=f"load_time + {steps} steps x s_per_step")
    elif s_step.value is None or load.value is None:
        total = _unknown("s", "needs a measured s_per_step and load_time")
    else:
        total = _unknown("s", "max_steps is not set (an epoch-based run)")

    same, here = cells("peak_vram_allocated", req)
    # the path's peak sized from the sequential loop's (path_extra), or any upper bound (_upper):
    # an estimate that decides the fit as a measured figure would, as it is never below the run's
    estimated = alloc.source == "estimated"
    lifted = "peak_vram_allocated" in via_seq
    fits, basis = None, alloc.why
    if gpu is None:
        basis = "no GPU detected"  # a command with --gpu says how to plan for a card
    elif lifted and extra is None:
        basis = (f"{_no_path_run(cfg.model_name_or_path, path)}, and config.json does not "
                 "size what it holds on top of the sequential loop's")
    elif alloc.value is not None:
        free = gpu.free_gib if gpu.free_gib is not None else gpu.total_gib
        room = f"{free} GiB {'free' if gpu.free_gib is not None else 'total'}"
        if alloc.value > free and estimated:  # an upper bound: it may not fit, it warns
            basis = f"allocated peak, at most {alloc.value} GiB, may not fit {room}"
        elif alloc.value > free:
            fits, basis = False, f"allocated peak {alloc.value} GiB > {room}"
        elif reserved.value is not None and reserved.value <= free:
            fits, basis = True, f"reserved peak {reserved.value} GiB <= {room}"
        elif here and not estimated and free >= 0.95 * gpu.total_gib:
            fits, basis = True, f"this config completed on this GPU ({here[0]['id']})"
        elif reserved.value is not None:  # an estimate of it, past the free memory
            basis = (f"allocated peak {alloc.value} GiB fits {room}, but the reserved peak, "
                     f"at most {reserved.value} GiB, may not")
        else:
            basis = (f"allocated peak {alloc.value} GiB fits {room}, but no "
                     "measured reserved peak on this GPU confirms it")
        if lifted:
            basis += f" (estimated: the sequential loop's at these settings, plus the {path} path)"
        elif estimated:
            basis += " (estimated: an upper bound from receipts at other settings)"

    bounded, caveats = None, []
    if fits is None and gpu is not None and req and not off_default and hold is None and not (
            lifted and extra is None):
        bounded = bound(req, gpu)
        if bounded is not None:
            fits, basis = bounded.fits, f"bounded: {bounded.basis}"
            # The bounding receipt's caveats go with its verdict.
            caveats = [c for c in _cells.CELLS if c["id"] == bounded.receipt]

    # a card `poros train` refuses for this config (gpu.check: no native bf16): no figure fits it
    refused = card_refusal(cfg, gpu)
    if refused is not None:
        fits, basis = False, f"refused on this card: {refused}"
    elif hold is not None and gpu is not None and fits is not False:
        free = gpu.free_gib if gpu.free_gib is not None else gpu.total_gib
        if hold.floor > free:
            fits, basis = False, hold.said(free, gpu.free_gib is not None)

    def held(est: Figure) -> Figure:  # + the block inputs efficient keeps in host RAM
        if extra is None or not extra.host:
            return est
        return est.model_copy(update={"value": _ceil1(est.value + extra.host),
                                      "method": f"{est.method}; + {extra.host:.2f} GiB "
                                                f"the {path} path holds on top"})

    host = figure("peak_host_ram", "GiB")
    if host.value is None and entry == "moe" and model_config:
        host = held(moe_host_ram(model_config, cfg.torch_dtype,
                                 expert_format=expert_format))
    stores = []
    store = getattr(cfg, "store", "ram")
    if entry == "moe" and model_config and store == "ssd":
        stores.append(PlanStore(store="ram", peak_host_ram=host, s_per_step=s_step,
                                bound_by="compute" if s_step.value is not None else None))
        host = held(moe_host_ram(model_config, cfg.torch_dtype, store="ssd",
                                 sss_ram=cfg.sss_ram,
                                 sss_cache=ssd_cache_bytes(cfg.store_options),
                                 expert_format=expert_format))
        if probe is not None:
            ram_s = s_step
            cache = _fast_cache_gib(req, gpu, sizes)
            if cache is None and path == "fast" and gpu is not None and req is not None:
                # no receipt grew it on this GPU: as a run sizes it after update 1 (sss.tune),
                # with efficient's reserved peak for update 1's (fast's measured lower: 7.85
                # GiB against 8.33-8.94, Qwen3-30B-A3B on an RTX 4090), so it errs small
                first = make_plan(cfg.model_copy(update={"path": "efficient"}),
                                  model_type=model_type, support=support, entry=entry, gpu=gpu,
                                  model_config=model_config, host_free_gib=host_free_gib,
                                  probe=probe).peak_vram_reserved.value
                if first is not None:
                    free = _card_free(gpu)
                    cache = max(0.0, free - max(3.0, 0.2 * free) - first)
                    if sizes is not None and sizes.experts_nf4:
                        cache = min(cache, sizes.experts_nf4 / 2**30)
            s_step, ssd_bound = moe_ssd_s_per_step(model_config, cfg, probe, ram_s,
                                                   vram_cache_gib=cache)
            if steps and s_step.value is not None and load.value is not None:
                total = Figure(value=round(load.value + steps * s_step.value, 1), unit="s",
                               source="estimated", receipt=s_step.receipt,
                               method=f"load_time + {steps} steps x s_per_step (ssd)")
            else:
                total = _unknown("s", "needs an ssd s_per_step and a measured load_time")
        else:
            ssd_bound = None
            s_step = _unknown("s", "the store directory was not probed")
            total = _unknown("s", "the store directory was not probed")
        stores.append(PlanStore(store="ssd", peak_host_ram=host, s_per_step=s_step,
                                bound_by=ssd_bound,
                                b_ssd=None if probe is None else probe.b_ssd,
                                b_h2d=None if probe is None else probe.b_h2d,
                                direct=None if probe is None else probe.direct))
    host_fits, host_basis = None, host.why or "host RAM not read (planning for --gpu, or not Linux)"
    if "peak_host_ram" in via_seq and extra is None:
        host_basis = (f"{_no_path_run(cfg.model_name_or_path, path)}, and config.json does "
                      "not size it: the sequential loop's host RAM is only an estimate")
    elif host.value is not None and host_free_gib is not None:
        host_fits = host.value <= host_free_gib
        host_basis = (f"peak host RAM {host.value} GiB {'<=' if host_fits else '>'} "
                      f"{host_free_gib} GiB available")
    floor = _nf4_host_floor(cfg, entry, sizes)
    host_floor = None
    if host_fits is None and floor is not None and host_free_gib is not None \
            and floor > host_free_gib:
        # no receipt sizes this run's host RAM, but what it must hold is already more than there
        host_fits, host_floor = False, floor
        host_basis = (f"the frozen layers in NF4 alone take {floor} GiB of host RAM (config.json: "
                      f"the streamed base stays there), > {host_free_gib} GiB available")

    warnings = [f"receipt {c['id']} ({k}): {v}" for c in caveats or here or same[:1]
                for k, v in c["host"].items()]
    if gpu is not None and gpu.source == "spec" and refused is None and not any(
            _norm(c["key"]["gpu"]) == _norm(gpu.name) for c in _cells.here(gpu.name)):
        # a card named or sized with --gpu that no measured run ran on: said once, plainly
        warnings.append(UNMEASURED_CARD.format(gib=gpu.total_gib))
    if req and req.load_strategy and cfg.nf4_load_strategy == "auto":
        warnings.append("nf4_load_strategy auto is planned as block_streamed; it falls "
                        "back to the gpu load if the streamed loader refuses the checkpoint")
    ceiling = getattr(probe, "pin_ceiling", None)
    if ceiling is not None:
        warnings.append(
            f"page-locked host memory is capped near {ceiling / 2**30:.2f} GiB on this host: "
            "Poros pins only up to it minus 0.5 GiB of headroom; the "
            "rest of the expert store and ring stays pageable (same bytes, synchronous "
            "copies, slower)")
    if store not in ("ram", "ssd"):
        warnings.append(f"store {store} comes from a plugin package: the host RAM and speed "
                        "figures are the built-in RAM store's NF4 estimates, not the plugin's")
    if store == "ssd" and entry != "moe":
        warnings.append("store ssd streams MoE experts; this model is not on the MoE path, "
                        "so `poros train` will refuse it (dense SSD streaming is not "
                        "in this release)")
    if format_note:
        warnings.append(format_note)
    if support == "user_asserted":
        warnings.append(f"{model_type} is user_asserted: runnable, no Poros parity guarantee")
    if refused is not None:
        warnings.append(f"refused on this card ({gpu.name}): {refused}; `poros train` refuses it "
                        "before the download (E_GPU_UNSUPPORTED)")
    if req is not None and cap is not None:  # the rank's kernels past this card: what runs instead
        if req.entry == "dense" and not req.fused_compute and \
                Request.of(cfg, entry, silu_experts=silu, triton=triton).fused_compute:
            why = fused_rank_reason(cap, cfg.lora_rank, head_dim=fit["rope_head_dim"])
            warnings.append(f"{why} ({gpu.name}): planned on PEFT's modules, as a run there takes")
        if req.entry == "moe" and req.moe_compute == "reference" and \
                Request.of(cfg, entry, silu_experts=silu, triton=triton).moe_compute == "fused":
            why = fused_rank_reason(cap, cfg.lora_rank, entry="moe")
            warnings.append(f"{why} ({gpu.name}): planned on the reference expert compute, as a "
                            "run there takes")
    if via_seq and extra:
        warnings.append(PATH_NOTE.format(path=path, what=extra.what))
    elif via_seq and alloc.value is None:
        warnings.append(PATH_NOTE_UNSIZED.format(path=path))
    elif via_seq:  # a memory figure is shown (an upper bound from other runs): never "none here"
        names = {"s_per_step": "speed", "peak_host_ram": "host RAM", "load_time": "load time"}
        what = " and ".join(names[k] for k in names if k in via_seq) or "figures"
        warnings.append(PATH_NOTE_GUIDE.format(path=path, what=what))
    return PlanOutput(
        model=cfg.model_name_or_path, model_type=model_type, support=support, entry=entry,
        gpu=gpu, path=path, path_reason=cfg.path_reason, fits=fits, fit_basis=basis,
        host_ram_free_gib=host_free_gib,
        host_fits=host_fits, host_fit_basis=host_basis, host_floor_gib=host_floor,
        seq_len=cfg.max_seq_length, peak_vram_allocated=alloc, peak_vram_reserved=reserved,
        peak_host_ram=host,
        s_per_step=s_step, load_time=load, total_time=total,
        download=download or _unknown("GiB", "not computed"),
        excludes=EXCLUDES, warnings=warnings, store=store, stores=stores, bound=bounded,
        # the nearest receipt of the path, else of the sequential loop its estimate builds on
        nearest=(nearest(req, gpu) or (nearest(seq_req, gpu) if seq_req else None))
        if req and fits is None else None,
    )


def over_card(p: PlanOutput) -> bool:
    """Whether ``p`` puts the run past the GPU's free memory: a measured peak or a run no
    larger's (``fits`` false, which refuses), or an estimate, an upper bound, past it (``fits``
    null: it may not fit, which warns)."""
    if p.fits is not None or p.gpu is None:
        return p.fits is False
    free = p.gpu.free_gib if p.gpu.free_gib is not None else p.gpu.total_gib
    return any(f.value is not None and f.value > free
               for f in (p.peak_vram_allocated, p.peak_vram_reserved))


def _holds_less(a: PlanOutput, b: PlanOutput) -> bool:
    """Whether plan ``a`` puts less on the GPU than ``b``: its reserved peak lower (its allocated
    peak where a reserved one is unknown), both figures known."""
    for name in ("peak_vram_reserved", "peak_vram_allocated"):
        x, y = getattr(a, name).value, getattr(b, name).value
        if x is not None and y is not None:
            return x < y
    return False


@dataclasses.dataclass(frozen=True)
class _Hold:
    """What a dense NF4 run's resident blocks hold on the GPU (GiB), and the floor of its VRAM:
    the resident layers' NF4 weights, the embedding table and LM head in fp32 where they are on
    the GPU (k-bit preparation's cast), and with every projection adapted the adapters' fp32
    weights, gradients and Adam's two moments. Each term is bytes the run holds for its whole
    training; activations, the streamed blocks and the logits come on top."""

    layers: int
    block: int
    per_block: float  # GiB a resident block's layers hold
    weights: float
    tables: float
    lora: float

    @property
    def floor(self) -> float:
        return self.weights + self.tables + self.lora

    def said(self, free: float, is_free: bool) -> str:
        fits = max(0, int((free - self.tables - self.lora) // self.per_block))
        return (f"at least {self.floor:.2f} GiB > {free} GiB {'free' if is_free else 'total'}: "
                f"{self.layers} resident layers hold {self.weights:.2f} GiB in NF4, the "
                f"embedding table and LM head {self.tables:.2f} GiB, the adapters' training state "
                f"{self.lora:.2f} GiB; resident_blocks={fits} or fewer leaves room"
                + ("" if not self.tables else ", or offload_embeddings_and_head=true"))


def _resident_hold(cfg: "PorosConfig", entry: Optional[str], s) -> Optional[_Hold]:
    """``_Hold`` for a dense run with resident_blocks set, else None."""
    blocks = getattr(cfg, "resident_blocks", 0) or 0
    if not blocks or entry != "dense" or s is None or s.experts_nf4:
        return None
    block = cfg.block_size or 4
    layers = min(blocks * block, s.layers)
    layer = (s.layer_bf16 // 2) * 33 / 64 / 2**30
    tables = 0.0 if cfg.offload_embeddings_and_head else s.tables / 2**30
    lora = cfg.lora_rank * s.per_rank * 16 / 2**30 if cfg.lora_target_modules is None else 0.0
    return _Hold(layers=layers, block=block, per_block=block * layer, weights=layers * layer,
                 tables=tables, lora=lora)


def _with_resident(f: Figure, hold: _Hold) -> Figure:
    """``f`` (a peak in GiB) plus the resident layers' weights: an estimate, as no receipt ran
    resident blocks."""
    if f.value is None:
        return f
    how = f"+ {hold.weights:.2f} GiB of {hold.layers} resident layers in NF4"
    return f.model_copy(update={"value": round(f.value + hold.weights, 3), "source": "estimated",
                                "method": f"{f.method}; {how}" if f.method else how})
