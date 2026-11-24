"""What a train sheet says of the fit: the decisions the run's check makes (the path and why, the
tables' place, a refusal and what to change), in words, and a figure only where a certified
recipe ran these settings on this very machine (poros/recipes/certified.json)."""

from __future__ import annotations

import functools
import re
from typing import Any, Dict, List, Optional, Tuple

from poros.cli.context import Context
from poros.cli.sheets.common import _resolved
from poros.cli.ui.components import ceil1
from poros.cli.ui.sheet import Option, Value
from poros.cli.verbs import mac, shared


def _peak(p):
    """A plan's reserved peak, else its allocated one: for the certification driver's own record
    (scripts/certify/run_recipe.py), never a screen."""
    return p.peak_vram_reserved if p.peak_vram_reserved.value is not None else p.peak_vram_allocated


def host_short(p) -> Optional[str]:
    """Where host RAM keeps a run from fitting (the frozen base, or its tables, cannot wait
    there), in words: what every screen says first. None where host RAM is not the reason."""
    if p is None or getattr(p, "host_fits", None) is not False:
        return None
    return "needs more host RAM than is available"


# The settings that decide a run's memory or its time an update: a certified figure speaks for a
# run only where each of them, resolved as the run resolves it, equals what the certified run ran
# with. The path is compared with the entry's ran_path and offload_embeddings_and_head with its
# ran_offload (what the run took, not what was typed). Left out, as they change neither its
# memory nor its time an update: the optimizer and its schedule (learning_rate, adam_*,
# weight_decay, max_grad_norm, warmup_steps, lr_scheduler, dpo_beta), the loss's rows (train_on),
# how long it runs (max_steps, num_epochs: an eta multiplies the time an update), seed,
# init_adapter (the starting values), what it saves, scores and logs (save_*, eval_*, logging_*,
# output_dir, push_to_hub, hub_model_id, progress), the data file (its shape is the loop's),
# load-time choices outside the timed updates (nf4_load_strategy, startup_parity_selfcheck,
# trust_remote_code, blockwise_arch_check, store_package), where the store sits on this machine
# (store_path: the machine is compared whole), memory_guard (it stops a run, it holds nothing),
# and what a run writes itself (path_reason, offload_reason, ran_path, ran_compute,
# compute_reason, resolved_base_model_path).
_CERTIFIED_KEYS = (
    "model_name_or_path", "model_revision", "quantization", "torch_dtype", "attn_implementation",
    "max_seq_length", "per_device_batch_size", "gradient_accumulation_steps", "group_by_length",
    "lora_rank", "lora_alpha", "lora_dropout", "lora_bias", "lora_target_modules",
    "freeze_embeddings", "freeze_lm_head", "adapter_backend", "adapter_runtime",
    "path", "block_size", "resident_blocks", "offload_embeddings_and_head",
    "offload_suffix_inputs_to_cpu", "nf4_dequant_cache_mb", "packed_block_transfers",
    "enable_prefetch", "fused_compute", "bf16_forward_math_mode", "bf16_cuda_graph_blocks",
    "deterministic", "fill_uninitialized_memory", "preserve_rng_state",
    "cuda_expandable_segments", "debug_memory",
    "store", "store_options", "sss_ram", "moe_expert_cache_gib", "moe_expert_compute",
    "moe_expert_format", "moe_expert_grouping",
)


def _same(a: Any, b: Any) -> bool:
    """Whether two resolved values are the same setting. Total: no input raises (a value this
    cannot compare is not the same)."""
    try:
        if a == b:
            return True
        if isinstance(a, dict) or isinstance(b, dict):
            if not all(isinstance(x, (dict, type(None))) for x in (a, b)):
                return False
            return ({str(k): str(v).lower() for k, v in (a or {}).items()}
                    == {str(k): str(v).lower() for k, v in (b or {}).items()})
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            if not all(isinstance(x, (list, tuple, type(None))) for x in (a, b)):
                return False
            return sorted(map(str, a or [])) == sorted(map(str, b or []))
        if a is None or b is None or isinstance(a, bool) or isinstance(b, bool):
            return False  # typed already: True is not 1, and nothing is None but None
        try:
            return float(a) == float(b)
        except (TypeError, ValueError):
            return str(a).lower() == str(b).lower()
    except Exception:
        return False


@functools.lru_cache(maxsize=None)
def _adapter(name: str) -> Any:
    from pydantic import TypeAdapter

    from poros.core.config import PorosConfig

    field = PorosConfig.model_fields.get(name)
    return None if field is None else TypeAdapter(field.annotation)


def _typed(name: str, value: Any) -> Any:
    """``value`` of config field ``name`` as the run's config reads it: through the config's own
    validation (its parsers: "false" is False, "8" is 8, "2G" is bytes), else the field's type,
    else as it came."""
    from poros.core.config import PorosConfig

    try:
        return getattr(PorosConfig(model_name_or_path="m", dataset="synthetic",
                                   **{name: value}), name)
    except Exception:
        pass
    try:
        adapter = _adapter(name)
        return value if adapter is None else adapter.validate_python(value)
    except Exception:
        return value


def _typed_all(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Every config field of ``settings`` as the run's config reads it: the whole config
    validated at once (as a run builds it), else each field alone (_typed)."""
    from poros.core.config import PorosConfig

    fields = PorosConfig.model_fields
    given = {k: v for k, v in settings.items() if k in fields}
    try:
        cfg = PorosConfig(**{"model_name_or_path": "m", "dataset": "synthetic", **given})
        return {k: getattr(cfg, k) for k in given}
    except Exception:
        return {k: _typed(k, v) for k, v in given.items()}


def _default(name: str) -> Any:
    """Config field ``name``'s default as a run gets it (a default factory called)."""
    from poros.core.config import PorosConfig

    try:
        return PorosConfig.model_fields[name].get_default(call_default_factory=True)
    except Exception:
        return None


def resolved(model: str, settings: Dict[str, Any], ctx: Context) -> Dict[str, Any]:
    """``settings`` (a run's, or a certified run's) resolved as a run of ``model`` on this machine
    resolves them: each of _CERTIFIED_KEYS, typed as its config validates it, set or left to its
    default; the revision a run fetches (none named: main); on a Mac what its table row fills; a
    dense CUDA run's block_size 4 where it is left unset. ``path`` and
    ``offload_embeddings_and_head`` stay as given: the caller says what the run took."""
    from poros.cli.words.models import by_id
    from poros.core.config import PorosConfig

    fields = PorosConfig.model_fields
    out = {k: _default(k) for k in _CERTIFIED_KEYS if k in fields}
    typed = _typed_all(settings)
    out.update({k: v for k, v in typed.items() if k in out})
    out["model_name_or_path"] = model
    out["model_revision"] = out.get("model_revision") or "main"  # what a run with none fetches
    if ctx.unified:
        from poros.metal.table import defaults

        row = mac.mac_row(model, ctx, seq=out["max_seq_length"])
        if row is not None:
            for k, v in defaults(row).items():
                if k in out and k not in settings:
                    out[k] = _typed(k, v)
    else:
        entry = by_id(model)
        if out["block_size"] is None and (entry is None or entry.entry != "moe"):
            out["block_size"] = 4  # the validated default every dense CUDA run takes
    return out


def certified_run(model: str, settings: Dict[str, Any], ctx: Context, loop: str = ""):
    """(recipe key, its certified entry) where this run is a certified run: a built-in recipe of
    ``model`` certified on this very machine (poros.recipes.certified.match: ``certified``),
    training the same ``loop``, whose every setting in _CERTIFIED_KEYS, both sides resolved
    (``resolved``), equals the run's. ``settings``: the run's, with the path it takes and whether
    its tables go to host RAM. A setting the entry does not record is the recipe's own, else the
    default; never a wildcard. None otherwise: no figure and no eta are shown then."""
    from poros.recipes import certified, presets

    this = certified.here(ctx)
    if this is None or not loop:
        return None
    run = resolved(model, settings, ctx)
    hits = []
    for r in presets.built_in():
        if r.model != model or r.loop != loop:
            continue
        hit = certified.match(certified.entries(r.key), this)
        if hit is None or hit[1] != certified.CERTIFIED:
            continue
        e = hit[0]
        ran = {**r.settings, **e.settings, "path": e.ran_path,
               "offload_embeddings_and_head": e.ran_offload}
        if ctx.unified:  # one loop on a Mac: the path is not the run's to choose
            ran["path"] = run["path"]
        want = resolved(model, ran, ctx)
        if all(_same(run[k], want[k]) for k in _CERTIFIED_KEYS):
            hits.append((r, e))
    if not hits:
        return None
    # twins (support and voice: one model, loop and set of compared settings) are told apart by
    # the rest of the recipe's own settings, as the run took them (voice trains 3 epochs)
    own = [(r, e) for r, e in hits
           if all(_same(_typed(k, settings.get(k)), _typed(k, v)) for k, v in r.settings.items())]
    r, e = (own or hits)[0]
    return r.key, e


def on_a_drive(a) -> bool:
    """A run whose experts stream from an SSD store: its time an update is its drive's, which
    nothing here can show is the certified run's drive. Its memory is not the drive's."""
    config, preset = getattr(a, "config", None) or {}, getattr(a, "preset", None) or {}
    return str(config.get("store") or preset.get("store") or "ram").lower() == "ssd"


def run_settings(a) -> Dict[str, Any]:
    """A train card's run as it will resolve: yours, else the recipe's (and on a Mac its table
    row's), the length, rank, alpha and block size it trains at, the path its check takes and
    whether that check puts the tables in host RAM."""
    from poros.cli.words.models import by_id

    preset = dict(getattr(a, "preset", None) or {})
    config = getattr(a, "config", None) or {}
    entry = by_id(getattr(a, "model", ""))
    if entry is not None and entry.entry == "moe":
        preset.pop("block_size", None)  # a loop's block 4, unused by poros.moe; a typed one stays
    out = {**preset, **config, "max_seq_length": a.seq_len, "lora_rank": a.rank,
           "lora_alpha": config.get("lora_alpha", 2 * a.rank)}
    if getattr(a, "block_size", None):
        out["block_size"] = a.block_size
    p = getattr(a, "plan", None)
    if p is not None and not (p.gpu is not None and p.gpu.unified):
        out["path"] = p.path
        if config.get("offload_embeddings_and_head") is None:
            out["offload_embeddings_and_head"] = (bool(p.offload_reason)
                                                  if p.entry == "dense" else False)
    return out


def plain(reason: Optional[str]) -> str:
    """A path's or the tables' reason in one plain clause: the planner's figures in brackets go,
    and an estimate is said as what it decides."""
    said = str(reason or "")
    while True:
        cut = re.sub(r"\s*\([^()]*\)", "", said)
        if cut == said:
            break
        said = cut
    return said.strip().rstrip(";:,").strip()


def refusal_of(err):
    """A fit refusal (PorosVramError, PorosHostRamError) as poros train says it: the error's own
    head and cause, each piece of its advice on a line, and how to try anyway on this command.
    The same code and exit; the words are the error's, for every surface."""
    from poros.core.errors import PorosHostRamError, _sentences
    from poros.core.refusal import Refusal

    title = "Not enough host RAM" if isinstance(err, PorosHostRamError) else "Won't fit"
    reason = err.head + (f": {err.cause}" if err.cause else "")
    # the same shape as the trainer's error in --json: the message is the head and cause, the
    # hint the advice (and how to try anyway on this command)
    return Refusal(err.code, title, reason,
                   fix=_sentences([*err.advice, "--force trains anyway"]), docs=err.info.doc_url)


def unfit_refusal(fit, host: bool = False):
    """The check's refusal (``poros.preflight.fit.check``: its ``vram``, or with ``host`` its host
    RAM) as the CLI says it (refusal_of): the error's own words. None where it refuses
    nothing."""
    err = fit.host if host else fit.vram
    return None if err is None else refusal_of(err)


def host_said(p, ssd_gib: Optional[float] = None, paths: tuple = ()) -> str:
    """A run short of host RAM, in the words of the refusal the check would raise for it
    (poros.preflight.fit.host_ram_error): its head, its cause and what helps, one sentence."""
    from poros.preflight.fit import host_ram_error

    err = host_ram_error(p, ssd_gib, paths)
    if err is None:
        return ""
    return (err.head + (f", for {err.cause}" if err.cause else "") + ". What helps: "
            + "; ".join(err.advice))


def mac_memory_note(p, cfg=None, key: str = "") -> str:
    """On a Mac with less memory free now than a run needs: the train start's warning, with the
    peak of the certified run that is this run (the card's own match, certified_run, on this
    Mac's class: ``cfg``'s settings; or, for recipes show, recipe ``key``'s run itself), else
    no figure."""
    from poros.cli.context import Context
    from poros.recipes import certified, presets

    free = getattr(p, "host_ram_free_gib", None)
    guard = getattr(cfg, "memory_guard", True) if cfg is not None else True
    tail = ("Close other apps, or expect it to swap"
            + (": the memory guard stops a run whose swap grows." if guard else "."))
    gpu = getattr(p, "gpu", None)
    hit = None
    if gpu is not None and gpu.unified:
        here = Context(fixed_gpu=gpu, fixed_interactive=False)
        if key:  # recipes show: that recipe's own run, on this class only
            r = next((x for x in presets.built_in() if x.key == key and x.model == p.model),
                     None)
            found = certified.match(certified.entries(key), certified.here(here)) if r else None
            hit = (key, found[0]) if found and found[1] == certified.CERTIFIED else None
        elif cfg is not None:
            hit = _certified_cfg(p.model, cfg, here)
    if hit is not None:
        e = hit[1]
        if e.memory_peak_gib is not None and free is not None and e.memory_peak_gib > free:
            return (f"This Mac has {free} GiB free right now; the certified run of "
                    f"{hit[0]} peaked at {ceil1(e.memory_peak_gib):.1f} GiB of memory. {tail}")
    return f"Other apps hold memory this run may need. {tail}"


def _certified_cfg(model: str, cfg, ctx: Context):
    """certified_run for a run's config (poros train <config>'s gate): its fields, its loop."""
    try:
        from poros import recipes

        loop = recipes.choose(cfg).label.lower()
        return certified_run(model, cfg.model_dump(exclude_unset=True), ctx, loop)
    except Exception:  # no loop to name: no certified run is this one
        return None


def _fit_view(p, cert=None) -> Dict[str, object]:
    """The result row: the decision in words (``fit  won't fit  needs more host RAM ...``), and a
    figure only for a certified run of these settings on this machine (``cert``: its entry)."""
    if p is None:
        return {}
    gpu = p.gpu
    if gpu is None:
        return {"result": ("fit", "unknown", "no GPU here"), "result_style": "tx"}
    if host_short(p):
        return {"result": ("fit", "won't fit", host_short(p)), "result_style": "e"}
    if p.fits is False:
        return {"result": ("fit", "won't fit", "refused before the download at these settings"),
                "result_style": "e"}
    if cert is not None:
        used = cert.memory_peak_gib if gpu.unified else cert.fit_gib()
        if used is not None and gpu.total_gib:
            return {"result": ("fits", f"{ceil1(used):.1f} of {gpu.total_gib:.1f} GiB",
                               "certified"), "meter": (used / gpu.total_gib, "a")}
        return {"result": ("fits", "certified", "")}
    return {"result": ("fit", "checked at the start", "no certified run of these settings here"),
            "result_style": "tx"}


def _train_fit(a, ctx) -> Dict[str, object]:
    """The train card's fit row, on the path the run takes."""
    said = shared.path_choice(getattr(a, "config", None) or {})  # fast+cache, as it was set
    hit = getattr(a, "certified", None)
    if hit is None and a.plan is not None and not hasattr(a, "certified"):
        hit = certified_run(a.model, run_settings(a), ctx, (a.recipe or "").lower())
    return _on_path(_fit_view(a.plan, hit[1] if hit else None), a.plan, ctx, said)


def _on_path(view: Dict[str, object], p, ctx, said: Optional[str] = None) -> Dict[str, object]:
    """A CUDA card's fit row says the path its figure is for: ``measured · efficient path``, and
    where the plan turned offload_embeddings_and_head on, ``· tables in host RAM``."""
    if not view.get("result") or p is None or p.path is None or ctx.unified:
        return view
    label, value, detail = view["result"]
    tables = " · tables in host RAM" if p.offload_reason else ""
    path = said if said and said.startswith(p.path) else p.path  # the path as the run names it
    return {**view, "result": (label, value, " · ".join(x for x in (
        detail, f"{path} path{tables}") if x))}


OFFLOAD = "offload_embeddings_and_head"


def _offload_said(p) -> str:
    """offload_embeddings_and_head left unset, as a run of the card takes it (the plan's
    decision, poros.plan.make.make_plan): on where the tables would not fit on the GPU."""
    return "auto → yes" if p.offload_reason else "auto → no"


def _fit_words(p) -> str:
    """A path's decision in a list: ``won't fit``, ``won't fit host RAM``, else nothing."""
    if p.host_fits is False:
        return "won't fit host RAM"
    return "won't fit" if p.fits is False else ""


# The SSD store's own path choice, in words (shared.CACHE_PATH).
CACHE_WORDS = f"fast, and {shared.FAST_CACHE} of host RAM keeps experts read off the disk"


def _path_row(a, ctx, now: str):
    """The path row's words for a CUDA card: what an unset path takes and why, and each path's
    fit here -- VRAM, host RAM and speed -- planned when its list opens; on the SSD store fast
    with its host cache too (shared.CACHE_PATH): ``(shown, source, options)``."""
    from poros.cli.train_settings import setting

    unset = a.config.get("path") in (None, "")
    shown = f"auto → {a.plan.path}" if unset else ""
    picks = {**dict(setting("path").choices), shared.CACHE_PATH: CACHE_WORDS}
    store = a.config.get("store", "ram")
    cached = shared.path_choice(a.config) == shared.CACHE_PATH  # its host cache goes with it

    def options(now=now) -> List[Option]:
        plans = shared.path_plans(a.model, ctx, seq_len=a.seq_len, config_json=True,
                                  path_settings=a.plan_said or a.config, **a.plan_kw)
        took = a.plan if unset else shared.auto_path(shared.plan_for(
            a.model, ctx, seq_len=a.seq_len, config_json=True,
            path_settings={k: v for k, v in (a.plan_said or a.config).items() if k != "path"
                           and not (k == "store_options" and cached)},
            **a.plan_kw),
            plans.__getitem__)
        out = [Option("", f"auto → {took.path}", " · ".join(x for x in (
            _fit_words(took), "what it trains at" if now == "" else "",
            shared.path_why(took)) if x))]
        for val in shared.paths_for(store):
            p = plans.get(val)
            if p is None:  # the card's settings refuse it: block-major cannot train them
                out.append(Option(val, val, f"refused: block-major cannot train {why}"
                                  if (why := shared.path_why(took)) else "refused"))
                continue
            out.append(Option(val, val, " · ".join(x for x in (
                _fit_words(p), "what it trains at" if val == now else "",
                picks.get(val, "")) if x)))
        return out

    return shown, shared.path_why(a.plan) if unset else "", options


def _fit_facts(p, cert=None) -> List[Tuple[str, str, str]]:
    """Under /train's answer, always the same three rows so nothing moves as a setting changes:
    host memory, speed (a certified run's figures on this machine, else none), and what is
    still to download."""
    if p is None:
        return []
    none = ("—", "no certified run of these settings here")
    speed = (f"{cert.s_per_step:.2f} s/step", "certified") if cert is not None else none
    ram = (f"{ceil1(cert.host_ram_peak_gib):.1f} GiB", "certified") if (
        cert is not None and cert.host_ram_peak_gib is not None) else none
    down = (f"{p.download.value:.1f} GiB", "from the Hub") if p.download.value else ("—", "")
    if p.download is not None and p.download.value == 0:
        down = ("none", "already downloaded")
    if p.gpu is not None and p.gpu.unified:  # one pool: the row above said memory
        return [("speed", *speed), ("download", *down)]
    return [("host ram", *ram), ("speed", *speed), ("download", *down)]


def _billions(size: str) -> float:
    try:
        return float(size.lower().rstrip("b"))
    except ValueError:
        return 0.0


def _gpu_name(gpu) -> str:
    return gpu.name.removesuffix(" (simulated)").removeprefix("NVIDIA ").removeprefix("GeForce ")


def recommended(ctx: Context) -> Optional[Tuple[str, str]]:
    """(model, why) a card opens with: the starter model where this machine trains it, else the
    smallest instruct model it trains. A first run, not a record: no figure decides it."""
    entries = list(mac.trainable(ctx))
    if not entries:
        return None
    start = next((e for e in entries if e.id == shared.START_WITH), None)
    best = start or min(entries, key=lambda e: (e.variant != "instruct", _billions(e.size)))
    return best.short, "recommended here"


def _matched(values: Dict[str, Value], ctx: Context) -> Dict[str, Value]:
    """The opening values the nearest receipt gives: the same ones a run with
    no card takes (train.matched)."""
    from poros.cli.verbs import train

    a, _, _ = _resolved("train", values, ctx)
    return {} if a is None else train.matched(a, ctx)
