"""What several verbs need: the model Question, a plan for a catalog model."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Callable, Optional

from poros.cli.schemas import Choice, Figure, PlanOutput, Question
from poros.cli.verbs.mac import mac_row, trainable
from poros.cli.words import models as model_words
from poros.core.refusal import Refusal
from poros.preflight.disk import disk_text  # noqa: F401 (the one way a size on disk is said)

if TYPE_CHECKING:
    from poros.cli.context import Context

DEFAULT_SEQ = 512  # fit's default: the length every measured run ran
DEFAULT_RANK = 16
# The small model a first run starts on: what `poros agents` says to begin on, `poros machine`
# suggests, and `poros init` writes (poros.cli.commands.train.STARTER_MODEL is the same).
START_WITH = "Qwen/Qwen2.5-0.5B-Instruct"
MAC_PLANNED_STEPS = 200  # what a total time on a Mac is for, until a run says its own


GPU_EXAMPLES = "a card: 4090, \"RTX 4070 Ti\", l4, a100; or its memory: 8GB"


def gpu_name(value) -> Optional[str]:
    """A --gpu value as the card it names (``poros.core.card.named_card``: a name, a memory size,
    or a name with its size), said the way a result's command repeats it. Refuses a name Poros
    does not know, a card sold in two sizes named without one, and a card of a generation Poros
    does not train on (E_GPU_UNSUPPORTED, as `train` refuses it there)."""
    if value is None:
        return None
    from poros.core import card

    if value is True or not str(value).strip():
        raise Refusal("E_USAGE", "Which GPU?", "--gpu needs a card's name or its memory",
                      context=[GPU_EXAMPLES])
    text = " ".join(str(value).split())
    try:
        name, gpu = card.named_card(text)
    except card.CardNeedsSize as exc:
        raise Refusal("E_USAGE", "Which size?", str(exc), context=[
            "say which: " + " or ".join(f'--gpu "{exc.name} {n}GB"' for n in exc.sizes)]
            if exc.sizes else [GPU_EXAMPLES])
    except card.UnknownCard:
        gpu, name = None, text
    cap = card.capability_of_name(name) if gpu is None or gpu.name != "NVIDIA GPU" else None
    if cap is not None and (why := card.bf16_reason(cap)):
        raise Refusal("E_GPU_UNSUPPORTED", "Not a supported GPU", f"{name}: {why}")
    if gpu is None:  # a real card whose memory Poros has no figure for, or a typo
        raise Refusal("E_USAGE", "Unknown GPU", f"Poros has no memory size for {text}",
                      context=['add its memory to the name, as in --gpu "RTX 4060 Ti 16GB", or '
                               "give the memory alone: --gpu 16GB"])
    return name


def size_of(entry: model_words.Entry) -> float:
    return float(re.sub(r"[^\d.]", "", entry.size) or 0)


def support(model_type: Optional[str]) -> tuple[str, Optional[str]]:
    """(support level, entry) exactly as `poros train` will treat the model."""
    from poros.core.arch_registry import user_asserted_arch_types, validated_arch_types

    if model_type is None:
        return "unknown", None
    if model_type in validated_arch_types("moe"):
        return "validated", "moe"
    if model_type in validated_arch_types():
        return "validated", "dense"
    if model_type in user_asserted_arch_types():
        return "user_asserted", "dense"
    return "unsupported", "dense"


def evidence_note(e: model_words.Entry) -> str:
    if e.evidence in ("validated", "measured"):
        return e.evidence
    return f"same shapes as {model_words.by_id(e.via).short}" if e.via else e.evidence


# A card's settings its plan reads, as given: what the plan keys (the loop, the split, the block,
# the tables, the caches, the compute) and what decides the loop a CUDA run takes
# (PorosConfig.apply_cuda_defaults, which make_plan calls: the trainer's own rule).
PLAN_KEYS = ("path", "per_device_batch_size", "gradient_accumulation_steps", "block_size",
             "offload_embeddings_and_head", "moe_expert_cache_gib", "nf4_dequant_cache_mb",
             "fused_compute", "moe_expert_compute", "freeze_embeddings", "freeze_lm_head",
             "lora_target_modules", "preserve_rng_state", "store_options")


def _plan_keys(said: Optional[dict]) -> dict:
    """The settings a train card (or a fit) was given that its plan reads (PLAN_KEYS)."""
    return {k: v for k, v in (said or {}).items() if k in PLAN_KEYS and v not in (None, "")}


def plan_for(
    model_id: str,
    ctx: "Context",
    *,
    seq_len: Optional[int] = None,
    rank: int = DEFAULT_RANK,
    dropout: Optional[float] = None,
    download: bool = False,
    store: str = "ram",
    store_path: Optional[str] = None,
    steps: int = 0,
    model_type: Optional[str] = None,
    path_settings: Optional[dict] = None,
    config_json: bool = False,
    held: bool = False,
) -> PlanOutput:
    """The plan for one model on this machine, at the default rank and length unless given.

    A catalog model's architecture comes from the catalog (no network); any
    other model's from ``model_type``, or else its config.json. ``download`` asks the Hub for the
    size still to fetch (the inspect stage only), and reads config.json; ``config_json`` reads
    that alone (what sizes a block-major path's estimate), as a plan beside a downloading one
    does. On a Mac the table fills what is left unsaid, and ``steps`` is what the total time
    is for.
    """
    from poros.core.config import PorosConfig
    from poros.plan.configs import cached_model_config, download_size, read_model_config
    from poros.plan.make import make_plan

    entry = model_words.by_id(model_id)
    # A same-shape model trains in exactly the memory of the one it matches:
    # plan with that model's receipts, and say so.
    # (When it has receipts of its own, those are used.)
    from poros.plan.cells import CELLS, MAC_CELLS, here

    # (a certification run's cell measures its own machine only: it is no receipt of its own here)
    own = any(c["key"]["model"] == model_id and not c.get("only_its_machine")
              for c in (MAC_CELLS if ctx.unified else CELLS))
    same_shape = entry is not None and entry.evidence == "same-shape" and not own
    planned = entry.via if same_shape else model_id
    if entry is not None:
        model_type = entry.arch
    elif model_type is None:
        from poros.preflight.arch import resolve_model_type as _resolve_model_type

        model_type, _ = _resolve_model_type(model_id)
    level, path = support(model_type)
    from poros.recipes import CUDA_MEASURED, MEASURED

    # at what every recipe trains with (the receipts' settings), so fit's
    # verdict is the one /train will meet
    if ctx.unified:
        measured = {"max_steps": steps or MAC_PLANNED_STEPS,
                    **({"max_seq_length": seq_len} if seq_len else {}),
                    **({"store": "ram"} if held else {})}  # store=ram: every layer held
    else:
        measured = {**CUDA_MEASURED, "max_seq_length": seq_len or DEFAULT_SEQ}  # a CUDA card's
    cfg = PorosConfig(
        model_name_or_path=planned,
        dataset="synthetic",
        **{**measured, "lora_rank": rank,
           "lora_dropout": MEASURED["lora_dropout"] if dropout is None else dropout,
           # the loop, split and tables the card's run takes (a Mac's table says its own)
           **({} if ctx.unified else _plan_keys(path_settings)),
           **({"store": store, "store_path": store_path} if store == "ssd" else {})},
    )
    asked = cfg.model_copy(update={"model_name_or_path": model_id})
    disk, fetched = "", None
    if download and mac_row(model_id, ctx) is not None:
        from poros.metal.store import built, survey, use

        settled, disk = survey(model_id, store_path)
        if settled is not None:  # what is left to fetch is what this disk lacks
            use(settled)
            if built(settled, model_id):
                fetched = Figure(value=0.0, unit="GiB", source="local")
    probe = None
    if store == "ssd" and store_path and not ctx.unified:  # the disk's probe, if one was taken
        from poros.sss.probe import cached

        probe = cached(store_path)
    def plan(cfg: "PorosConfig") -> PlanOutput:
        return make_plan(
            cfg,
            model_type=model_type,
            support=level,
            entry=path,
            gpu=ctx.gpu,
            probe=probe,
            download=fetched or (download_size(asked) if download else None),
            # what decides the loop a CUDA run takes (sequential_reason), sizes what a config
            # holds beyond a receipt, and a MoE's host RAM: read where asked, else from the cache
            model_config=read_model_config(model_id) if download or config_json
            else cached_model_config(model_id),
            host_free_gib=ctx.host_free_gib,
        )

    p = plan(cfg)
    if (same_shape and p.peak_vram_allocated.source != "measured" and ctx.gpu is not None
            and any(c["key"]["model"] == model_id and c.get("only_its_machine")
                    for c in (MAC_CELLS if ctx.unified else here(ctx.gpu.name)))):
        # its base's runs do not measure these settings, and a certification run of this model on
        # this card may: that measurement, where it is this request's (no other card's plan reads
        # it: poros.plan.cells.counts)
        mine = plan(cfg.model_copy(update={"model_name_or_path": model_id}))
        if mine.peak_vram_allocated.source == "measured":
            planned, p = model_id, mine
    if planned != model_id:
        p.model = model_id
        same = f"{entry.via.split('/')[-1]}'s" if ctx.unified else f"{planned}'s receipts"
        p.warnings.insert(0, f"memory and speed are {same}: the same shapes")
    if disk:
        p.warnings.append(disk)
    return p


# The CUDA training loops, as a card lists them (poros.core.config.PorosConfig.path).
PATHS = ("efficient", "fast", "sequential")
# Why a path left unset takes block-major's first choice, where the config records no reason.
# (It holds less than fast; the sequential loop can peak lower: each path's figures say.)
FIRST_CHOICE = {"efficient": "NF4's default"}


# The SSD store's own choice beside the paths: path fast with the store's host cache
# (store_options cache), FAST_CACHE of it. Opt-in, never a default.
CACHE_PATH = "fast+cache"
FAST_CACHE = "4G"


def path_config(path: str) -> dict:
    """The config a path choice sets: ``path``, and for ``fast+cache`` path fast with the SSD
    store's host cache (store_options)."""
    if path == CACHE_PATH:
        return {"path": "fast", "store_options": {"cache": FAST_CACHE}}
    return {"path": path}


def path_choice(config: dict) -> Optional[str]:
    """The path choice a config makes (``path_config``'s inverse): ``fast+cache`` for path fast
    on the SSD store with FAST_CACHE of host cache, else its path (None: unset)."""
    if (config.get("path") == "fast" and config.get("store") == "ssd"
            and (config.get("store_options") or {}) == {"cache": FAST_CACHE}):
        return CACHE_PATH
    return config.get("path") or None


def paths_for(store: str) -> tuple[str, ...]:
    """The path choices a card or fit lists: the SSD store adds ``fast+cache`` after fast."""
    return ("efficient", "fast", CACHE_PATH, "sequential") if store == "ssd" else PATHS


def path_plans(model_id: str, ctx: "Context", **plan: Any) -> dict[str, PlanOutput]:
    """Each CUDA path's plan for ``model_id`` here, at ``plan``'s settings (plan_for's keywords;
    ``path_settings`` keeps the rest of the card's): what each would need, side by side; a path
    the settings refuse (block-major where it cannot train the same update) is left out. On the
    SSD store, fast with its host cache too (``fast+cache``). Empty on a Mac, which trains one
    loop, or with no GPU to plan for."""
    if ctx.unified or ctx.gpu is None:
        return {}
    said = {k: v for k, v in (plan.pop("path_settings", None) or {}).items()
            if k not in ("path", "store_options")}
    out = {}
    for p in paths_for(plan.get("store", "ram")):
        try:
            out[p] = plan_for(model_id, ctx, **plan, path_settings={**said, **path_config(p)})
        except ValueError:  # a path the config refuses (poros.core.config.path_refusal)
            continue
    return out


def auto_path(first: PlanOutput, plan: Callable[[str], PlanOutput]) -> PlanOutput:
    """The plan of the path a run that leaves ``path`` unset takes, from ``first``, its plan with
    path unset; ``plan(path)`` plans a path at the same settings. As the run decides it
    (poros.preflight.fit.path_to_fit): where its path falls short of host RAM, or of the GPU (a
    peak past it, or an estimate), and another path fits, that one, its ``path_reason`` saying
    why (path_why)."""
    from poros.core.config import PorosConfig
    from poros.preflight.fit import path_to_fit

    if first.path is None:  # a Mac: one loop
        return first
    plans = {first.path: first}

    def planned(c: PorosConfig) -> PlanOutput:
        if c.path is None:
            return first
        if c.path not in plans:
            plans[c.path] = plan(c.path)
        return plans[c.path]

    unset = PorosConfig(model_name_or_path=first.model, dataset="synthetic")
    other = path_to_fit(unset, planned)
    if other is None:
        return first
    return plans[other[0]].model_copy(update={"path_reason": other[1]})


def path_why(p: PlanOutput) -> str:
    """Why a run that leaves ``path`` unset takes ``p.path``: the reason the plan records, else
    block-major's first choice in words; "" on a Mac."""
    from poros.cli.sheets.fit import plain

    return plain(p.path_reason) or FIRST_CHOICE.get(p.path or "", "")


def hold_gpu(doing: str, *, queue: bool = False) -> Optional[Any]:
    """The lock jobs queue on, while ``doing`` loads a model: beside a run a card thrashes, and a
    Mac's one pool has no room. The open file holds it, for as long as the caller keeps it; None
    with nothing to hold: no GPU, or no flock here. ``queue``: a person waits their turn, on no
    torch and none of the card. Anything else is refused at once, as a tool call or a script is
    never left hanging behind a run for hours."""
    import sys
    import time

    from poros import runs
    from poros.cli import contract, questions
    from poros.cli.ui import simulate, term
    from poros.cli.ui.screens import Fix
    from poros.core.card import detect_gpu

    if simulate.enabled() or detect_gpu() is None:
        return None
    try:
        from poros import jobs

        lock = jobs.hold_machine()
    except (ImportError, OSError):  # no flock, or no home to keep its file in
        return None
    if lock is not None:
        return lock
    run = next((r for r in runs.all_runs() if r.state in ("running", "stopping")), None)
    who = f"run {run.id}" if run else "another export or eval"
    if queue and questions.interactive():
        with term.get().activity(f"Waiting for {who} to finish") as act:
            while (lock := jobs.hold_machine()) is None:
                time.sleep(1)
            act.done([])
        return lock
    contract.refuse(
        "E_USAGE", "Busy",
        f"{who} is using {'this Mac' if sys.platform == 'darwin' else 'the GPU'}, "
        f"and {doing} needs it to itself",
        context=["try again when it ends"],
        fix=Fix("", f"poros runs {run.id} --wait 30",
                f"returns 30 steps on, when it ends, or after {_wait_max()}") if run else None,
    )


def _wait_max() -> str:
    """How long a `--wait` holds at most, as its hint says it."""
    from poros.cli.verbs.runs_live import WAIT_MAX_S

    return f"{WAIT_MAX_S:.0f} s"


def how_known(peak_source: str, bounded: bool) -> str:
    """How a peak beside a fit is known, in the one word every screen says (`poros machine`,
    `poros models`, a card's fit row): ``measured``, a run of these settings on this card;
    ``tested``, a larger run's peak bounds it (``bounded``: the plan's ``bound``); else
    ``estimated``, a figure from other runs."""
    if peak_source == "measured":
        return "measured"
    return "tested" if bounded or peak_source == "bounded" else "estimated"


def fit_note(p: PlanOutput) -> str:
    if p.fits is True and p.host_fits is not False:
        return "fits"
    if p.fits is False or p.host_fits is False:
        return "won't fit"
    return "unknown"


def fitted_length(seq: int, plan: Callable[[int], PlanOutput]) -> tuple[int, PlanOutput]:
    """The length a run takes when no one said one: ``seq`` from its data, or where a receipt says
    that won't fit, the longest that does."""
    at = plan(seq)
    if at.fits is False:
        for shorter in (2048, 1024, 512):
            if shorter < seq and (there := plan(shorter)).fits:
                return shorter, there
    return seq, at


def model_choices(ctx: "Context", entries=None, **plan: Any) -> list[Choice]:
    """Catalog models as choices, the ones the check before a run would let start first (for
    ``plan``: the asker's length, rank, dropout). The order is the check's; a note says only what
    a model is, never the plan's verdict (what trains where is a certified recipe's)."""
    entries = trainable(ctx, entries)
    marks = {e.id: fit_note(plan_for(e.id, ctx, **plan)) for e in entries}
    order = {"fits": 0, "unknown": 1, "won't fit": 2}
    entries.sort(key=lambda e: order[marks[e.id]])
    return [
        Choice(
            value=e.id,
            label=e.short,
            # what it is; "dense" (the usual) goes unsaid
            note=" · ".join(
                [e.size.upper()]
                + (["moe"] if e.entry == "moe" else [])
                + ([] if ctx.unified else [evidence_note(e)])
                + (["downloaded"] if e.id in ctx.cached else [])
            ),
        )
        for e in entries
    ]


def refuse_planted(name: str) -> None:
    """A Hub name that is a folder here is refused, by every verb that takes a base."""
    from poros.core.config import planted_base

    if planted_base(name):
        raise Refusal("E_USAGE", "Which one", f"a folder here is named {name}",
                      context=[f"./{name} takes that folder", "or work in another folder"])


def resolve_model(words: list[str], ctx: "Context", *, verb: str):
    """(model id, how it was named, None), or (None, words, a Question) when
    the words are part of some catalog names (``72b``, ``qwen 7b``): the
    picker of just those, never a guess. One word otherwise: a catalog name,
    a Hub id or a local path; anything else is refused with the closest names."""
    from pathlib import Path

    said = " ".join(words)
    hits = model_words.matching(said) if words and not model_words.looks_local(said) else []
    hits = trainable(ctx, hits) or hits  # a Mac asks only among what it trains
    if hits and (len(words) != 1 or model_words.by_name(words[0]) is None):
        return None, said, _model_choice(hits, said, ctx)
    if len(words) != 1:
        raise Refusal(
            "E_USAGE",
            "Name one model",
            f"{verb} takes one model, not {' '.join(words)!r}",
            context=["a catalog name (Qwen2.5-7B-Instruct), a Hub id, or a path"],
            fix="poros models",
            note="lists every name",
        )
    word = words[0]
    if model_words.looks_local(word):
        path = Path(word).expanduser()
        if not path.exists():
            raise Refusal(
                "E_NOT_FOUND",
                "Not found",
                f"no model at {word}",
                context=["a path is read from disk, never looked up on the Hub"],
            )
        return os.path.abspath(path), word, None  # a folder stays one, in every run's record of it
    entry = model_words.by_name(word)
    refuse_planted(entry.id if entry is not None else word)
    if entry is not None:
        return entry.id, word, None
    if model_words.looks_like_hub_id(word):
        return word, word, None
    if model_words.is_model_dir(word):
        return os.path.abspath(word), word, None
    near = model_words.closest(word)
    raise Refusal(
        "E_NOT_FOUND",
        "No such model",
        f"nothing in the catalog is called {word}",
        context=[*([f"closest: {', '.join(near)}"] if near else []),
                 "a model is a catalog name, a Hub id (owner/name) or a folder"],
        fix="poros models",
        note="lists every model Poros names",
        docs="docs/cli.md#names-and-flags",
    )


def one_model(word: str, ctx: "Context", *, verb: str) -> str:
    """``resolve_model`` for a command that takes its model as one argument and asks nothing
    (`poros check`): the model it names, or the refusal every command gives for that word; part
    of a name is refused with the names it could be."""
    model, said, ask = resolve_model([word], ctx, verb=verb)
    if ask is not None:
        raise Refusal("E_USAGE", "Which one", f"{said} is part of more than one model's name",
                      context=["name one: " + ", ".join(c.label for c in ask.choices)],
                      fix="poros models", note="lists every name")
    return model


def _model_choice(entries: list, said: str, ctx: "Context") -> Question:
    """The catalog names ``said`` is part of, smallest first."""
    return Question(
        id="model",
        prompt=f"Which {said}?",
        default=None,
        choices=[
            Choice(value=e.id, label=e.short, note=" · ".join(
                [e.size.upper(), *(() if ctx.unified else (e.entry, e.evidence.replace("-", " ")))]
                + (["downloaded"] if e.id in ctx.cached else [])))
            for e in sorted(entries, key=size_of)
        ],
    )


def pick_model_question(ctx: "Context") -> Question:
    """No model named at all: offer the validated catalog, smallest first."""
    entries = sorted((e for e in trainable(ctx)
                      if ctx.unified or e.evidence in ("validated", "measured")),
                     key=size_of)
    return Question(
        id="model",
        prompt="Which model?",
        default=None,
        choices=[
            Choice(
                value=e.id,
                label=e.short,
                note=" · ".join(
                    [e.size.upper(), *(() if ctx.unified else (e.entry,))]
                    + (["downloaded"] if e.id in ctx.cached else [])
                ),
            )
            for e in entries
        ],
    )


def download_confirm(model_id: str, size_gib: Optional[float], ctx: "Context") -> Optional[str]:
    """How much a download would fetch, when there is one to ask about
    (None: nothing to download). Every download needs a yes, whatever its
    size. Refuses offline when the model is not on disk: a run would only
    fail after its preflights. The plan counts it in GiB; the question says decimal GB."""
    if size_gib is not None:
        return disk_text(size_gib * 2**30 / 1e9) if size_gib else None
    if _on_disk(model_id, ctx):
        return None
    try:
        from huggingface_hub import constants

        offline = bool(constants.HF_HUB_OFFLINE)
    except ImportError:
        offline = False
    if offline:
        raise Refusal(
            "E_NETWORK",
            "Offline",
            f"{model_id} is not downloaded and the Hub is off",
            context=["unset HF_HUB_OFFLINE, or download it first: poros models get"],
        )
    return "a size the Hub did not report"


def _on_disk(model_id: str, ctx: "Context") -> bool:
    """The weights are here: a local path, or every weight file in the HF cache."""
    from pathlib import Path

    from poros.cli.context import weights_cached

    if Path(model_id).expanduser().exists() or model_id in ctx.cached:
        return True
    return weights_cached(model_id)


def dataset_on_disk(name: str) -> bool:
    """The Hub's files for this dataset are in the HF cache."""
    from pathlib import Path

    try:
        from huggingface_hub import constants
    except ImportError:
        return False
    return (Path(constants.HF_HUB_CACHE) / f"datasets--{name.replace('/', '--')}").is_dir()


def hf_cache_shown(store: Optional[str] = None) -> str:
    """Where a Hugging Face download lands here, as a person reads it (~ for the home folder),
    by huggingface_hub's own rule (huggingface_hub.constants.HF_HUB_CACHE), read from the
    environment as it is now: HF_HUB_CACHE (or its older name, HUGGINGFACE_HUB_CACHE), else
    HF_HOME/hub, else ``store`` (a Mac's), else XDG_CACHE_HOME/huggingface/hub, else
    ~/.cache/huggingface/hub."""
    from pathlib import Path

    env = os.environ
    xdg = env.get("XDG_CACHE_HOME")
    home_hf = env.get("HF_HOME")
    where = (env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE")
             or (str(Path(home_hf) / "hub") if home_hf else None) or store
             or (str(Path(xdg) / "huggingface" / "hub") if xdg else "~/.cache/huggingface/hub"))
    home = str(Path.home())
    return "~" + where[len(home):] if where.startswith(home + os.sep) else where


def dataset_question(name: str) -> "Question":
    """A Hub dataset is a download of a size no one has said, so it is asked about like a model."""
    from poros.cli.questions import confirmation

    return confirmation(
        "dataset_download",
        f"Download the dataset {name}? From Hugging Face, into {hf_cache_shown()}, at a size "
        "Poros cannot see first. Nothing downloads unless you say yes.",
        "--download", "Yes, download and train", "a dataset from the Hub")


def download_question(model_id: str, what: str, flag: str, label: str,
                      disk: Optional[str] = None) -> "Question":
    """The one way Poros asks before downloading: what, how big, where to."""
    from poros.cli.questions import confirmation

    name = model_id.split("/")[-1]
    where = hf_cache_shown()
    if disk:
        from poros.metal.store import hub_of
        from poros.metal.table import pack_gb

        where = f"{hub_of(disk)}, and {disk_text(pack_gb(model_id))} to prepare it in {disk}"
    return confirmation("download", f"Download {name}? {what} from Hugging Face into {where}. "
                        "Nothing downloads unless you say yes.", flag, label, what)


def refuse_without_a_device(ctx: "Context", *, cuda: bool = True) -> None:
    """What a command that trains checks before any question, check of its disk or download:
    (``cuda``) off a Mac a CUDA GPU, then torch importable (E_NO_CUDA, E_NO_TORCH, exit 5). The
    GPU first, as this machine's card is read without torch: a machine with none hears that,
    not to install torch. A preview (POROS_DEV_SIMULATE) trains nothing and needs
    neither."""
    import importlib.util

    from poros.cli.ui import simulate
    from poros.preflight.command import NO_CUDA, NO_TORCH, no_cuda_before_torch

    if simulate.enabled():
        return
    has_torch = importlib.util.find_spec("torch") is not None
    if cuda and not ctx.unified and ctx.gpu is None:  # this machine's card, as every plan reads
        raise NO_CUDA if has_torch else no_cuda_before_torch()
    if not has_torch:
        raise NO_TORCH
