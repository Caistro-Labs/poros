"""Which measured runs (receipts) a config's plan reads (``poros.plan``): a request's key, how a
receipt's key matches it (the knobs a figure does not depend on, the card's free memory), the
bound a larger run puts on a smaller one (``bound``), the nearest measured run (``nearest``), and
the receipts' own defaults. Torch-free."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

from poros.core.arch_registry import should_run_startup_selfcheck
from poros.core.card import (
    GPU_SPECS_MIB,
    _norm,
    fused_rank_reason,
)
from poros.core.compute import choose
from poros.core.config import train_path
from poros.core.fused_tiles import rope_kernel_head_dim
from poros.core.schemas import PlanBound, PlanGpu, PlanNearest
from poros.plan import cells as _cells  # the receipt set, read at call time

if TYPE_CHECKING:
    from poros.core.config import PorosConfig
    from poros.plan.estimate import _Sizes


# How a figure from the same config on another GPU carries over. VRAM: one
# dense pair measured (docs/validation/README.md). Host RAM: an assumption.
_OTHER_GPU = {
    "peak_vram_allocated": "same config measured on {gpu} (the highest of {n} receipts); "
    "peak VRAM matched within 1% across GPUs in the one dense pair measured "
    "(32B: 10.95 GB on A100, 11.04 GB on B300)",
    "peak_host_ram": "same config measured on {gpu}'s host (the highest of {n} receipts); "
    "assumes host RAM does not depend on the GPU",
}


# Config fields that change memory or speed and that no receipt records:
# assumed to be at their PorosConfig default in every receipt (none of the
# scripts that wrote them set these). Any other value plans as unknown.
_RECEIPT_DEFAULTS = (
    "torch_dtype", "freeze_embeddings", "freeze_lm_head", "lora_target_modules", "lora_bias",
    "enable_prefetch", "offload_suffix_inputs_to_cpu", "adapter_backend",
    "attn_implementation", "debug_memory",
    "moe_expert_grouping",
)


# The values the receipts ran for fields whose PorosConfig default is resolved from other
# fields (``None`` = auto): MoE receipts grouped experts "pow2", what None resolves to.
_RECEIPT_VALUES = {"moe_expert_grouping": (None, "pow2")}


# The loop a receipt ran is its key's `path` (poros.core.config.train_path): sequential when
# absent. A config plans the path a CUDA run of it takes, from receipts of that path where they
# exist. Where no receipt of the model ran it, the plan takes the sequential loop's figures at
# the same settings and adds what the path holds on top, from config.json (path_extra): an
# estimate, labelled as one, that decides the fit as a measured figure would. Without a
# config.json to size it the sequential loop's figures are shown as estimates and decide nothing.
PATH_NOTE = ("a CUDA run of this config takes block-major accumulation's {path} path, which no "
             "receipt of this model measured at these settings: its figures are the sequential "
             "loop's at these settings plus what that path holds on top ({what}), an estimate")


PATH_NOTE_UNSIZED = ("a CUDA run of this config takes block-major accumulation's {path} path, "
                     "which no receipt of this model measured at these settings, and config.json "
                     "does not give its sizes: it holds more than the sequential loop's measured "
                     "peaks, so no memory figure here (path: sequential trains the loop the "
                     "receipts measured)")


# The same, where the memory figures came from the model's other runs (an upper bound from
# receipts at other settings): what the sequential loop's receipts stand in for is named, and
# nothing is said to be missing that the plan shows.
PATH_NOTE_GUIDE = ("a CUDA run of this config takes block-major accumulation's {path} path, "
                   "which no receipt of this model measured at these settings: its {what} here "
                   "is the sequential loop's, a guide (config.json, which sizes the path, was "
                   "not read)")


def _no_path_run(model: str, path: str) -> str:
    """Why no receipt of ``model`` on ``path`` gave a figure: none took it, or none at these
    settings (the others' settings need config.json to size)."""
    if any(c["key"]["model"] == model and _key_path(c["key"]) == path for c in _cells.CELLS):
        return f"no measured run of this model on the {path} path is at these settings"
    return f"no measured run of this model took the {path} path"


def _key_path(key: dict) -> str:
    """The loop a receipt's key ran: its ``path``, else ``sequential``."""
    return key.get("path", "sequential")


def _grows_to_card(key: dict) -> bool:
    """A receipt whose VRAM grew to fit its own card: a MoE run on the fast path with
    moe_expert_cache_gib unset, whose expert cache is sized to what that GPU left free
    (poros.train.compute). Its peaks hold on that GPU only: elsewhere the cache grows to
    another size, so they neither carry over nor bound another card's fit from below. A set
    cap (the key's ``expert_cache``) is the same size on every card."""
    return (key.get("entry") == "moe" and _key_path(key) == "fast"
            and _key_value(key, "expert_cache") is None)


def _grows_request(req: Any) -> bool:
    """A request whose expert cache grows to the card it runs on (``_grows_to_card``'s rule)."""
    return req.entry == "moe" and req.path == "fast" and req.expert_cache is None


# A run's VRAM budget (poros.sss.tune.vram_budget: free plus its own reserved, measured once the
# model has loaded) is below the free VRAM a plan reads before the process starts by at least its
# CUDA context, a few hundred MiB on any card measured: a plan reading at most this much more free
# than a receipt's budget plans a run that grows no larger a cache than the receipt's.
_BUDGET_MATCH = 0.25


def _card_free(gpu: "PlanGpu") -> float:
    return gpu.free_gib if gpu.free_gib is not None else gpu.total_gib


def _card_growth(c: dict, gpu: "PlanGpu", s: Optional["_Sizes"]) -> Optional[float]:
    """For a receipt whose expert cache grew to its card (``_grows_to_card``), on that card: GiB
    of VRAM a run on ``gpu`` may hold beyond the receipt's peak, as its cache grows into the
    free VRAM the receipt's run did not have (0 where it has no more: a smaller cache, its peak
    no higher). At most every expert's bytes (``s``), with what comes with them
    (``_CACHE_OVERHEAD``). None where the cell does not record the budget it grew within."""
    budget = c["fields"].get("vram_budget", {}).get("value")
    if budget is None:
        return None
    more = _card_free(gpu) - budget
    if more <= _BUDGET_MATCH:
        return 0.0
    cache = c["fields"].get("expert_cache_size", {}).get("value")
    if s is not None and s.experts_nf4 and cache is not None:
        more = min(more, max(s.experts_nf4 / 2**30 - cache, 0.0))
    return more * _CACHE_OVERHEAD


def _card_matches(c: dict, gpu: "PlanGpu") -> bool:
    """Whether ``gpu``'s free VRAM is the receipt's run's (``_BUDGET_MATCH``): its cache grew to
    the size this run's would, so its peaks are this run's own."""
    budget = c["fields"].get("vram_budget", {}).get("value")
    return budget is not None and budget <= _card_free(gpu) <= budget + _BUDGET_MATCH


def _key_value(key: dict, name: str) -> Any:
    """A receipt key's field, as the run took it when absent: ``path``, ``moe_compute``,
    ``fused_compute``, ``offload`` (offload_embeddings_and_head: off), ``expert_cache`` and
    ``dequant_cache`` (moe_expert_cache_gib, nf4_dequant_cache_mb: None, the path's default)."""
    if name == "path":
        return _key_path(key)
    if name == "moe_compute":
        return key.get("moe_compute", "reference")
    if name == "fused_compute":  # every receipt before poros.core.fused ran PEFT's modules
        return key.get("fused_compute", False)
    if name == "offload":
        return bool(key.get("offload", False))
    if name in ("expert_cache", "dequant_cache"):
        return key.get(name)
    return key[name]


# Key fields each figure does not depend on (every figure must be listed).
# The startup self-check runs before training resets the peak stats, so it
# does not move the allocated peak; its cached blocks stay reserved, so it
# can move the reserved one.
_IGNORE = {
    "peak_vram_allocated": ("grad_accum", "selfcheck"),
    "peak_vram_reserved": ("grad_accum",),
    "s_per_step": ("selfcheck",),
    "peak_host_ram": ("grad_accum",),
    "load_time": ("grad_accum",),
    "gpu_visible": ("grad_accum",),
}


def _ignored(name: str, path: str) -> tuple:
    """_IGNORE on CUDA, where the split is a dimension of block-major: its peaks hold every
    micro-batch's state between blocks (and efficient's host RAM, their block inputs), so they
    depend on the micro-batches a step takes, where the sequential loop's, one at a time, do
    not."""
    keep = ((path != "sequential" and name in ("peak_vram_allocated", "peak_vram_reserved"))
            or (path == "efficient" and name == "peak_host_ram"))
    return tuple(k for k in _IGNORE[name] if not (keep and k == "grad_accum"))


EXCLUDES = [
    "CUDA context (outside torch's allocator; typically under 1 GiB)",
    "the load/quantization phase: every peak is the training phase's (stats reset "
    "when training starts)",
]


@dataclass
class Request:
    """The cell key a config asks for."""

    model: str
    quantization: str
    seq_len: int
    block_size: Optional[int]
    batch: int
    grad_accum: int
    lora_rank: int
    lora_dropout: float
    load_strategy: Optional[str]
    selfcheck: bool
    entry: str
    path: str = "sequential"  # train_path: sequential, fast or efficient
    # a MoE run's expert compute (poros.core.config.moe_compute); a key without one ran the
    # reference compute, as every receipt before the fused path did. Dense runs: "reference".
    moe_compute: str = "reference"
    # a dense NF4 CUDA run's poros.core.fused compute (on unless fused_compute: false); a key
    # without one ran PEFT's modules, as every dense receipt before the fused compute did.
    fused_compute: bool = False
    # offload_embeddings_and_head: the embedding table and LM head in host RAM, not the GPU's
    offload: bool = False
    # a MoE run's moe_expert_cache_gib, and a dense NF4 block-major run's nf4_dequant_cache_mb;
    # None: the path's default (a key without one ran it)
    expert_cache: Optional[float] = None
    dequant_cache: Optional[int] = None

    @classmethod
    def of(cls, cfg: "PorosConfig", entry: str, *, cuda: bool = True,
           silu_experts: bool = True, triton: Optional[bool] = True,
           cap: Optional[tuple] = None, rope_head_dim: Optional[int] = None) -> "Request":
        """The key ``cfg`` asks for; ``cuda=False`` for a Mac's plan, where no MoE run takes the
        fused compute (it is CUDA's), ``silu_experts=False`` for a model whose experts the fused
        compute cannot run (poros.moe.arch.experts_silu), ``triton`` whether the run's machine
        has Triton (None: this machine's, found out): the trainer's choices. ``cap``: the card's
        compute capability, where known: a LoRA rank whose fused kernels do not fit it takes
        PEFT's modules (dense) or the reference expert compute (MoE), as a run there does
        (poros.core.card.fused_rank_reason; ``rope_head_dim``: the head size of a model
        whose fused layer runs the rotary kernels)."""
        dense = entry == "dense"
        # the compute a run takes (poros.core.compute, the trainer's own decision); "auto" -- the
        # model's layers decide -- is fused wherever silu_experts says the experts qualify
        choice = choose(cfg, entry=entry, cuda=cuda, triton=triton, silu_experts=silu_experts)
        fused = dense and choice.fused
        moe = {"auto": "fused"}.get(choice.moe_expert, choice.moe_expert) \
            if entry == "moe" else "reference"
        if cuda and cap is not None:
            if fused and fused_rank_reason(cap, cfg.lora_rank, head_dim=rope_head_dim):
                fused = False
            if moe == "fused" and fused_rank_reason(cap, cfg.lora_rank, entry="moe"):
                moe = "reference"
        return cls(
            model=cfg.model_name_or_path, quantization=cfg.quantization,
            seq_len=cfg.max_seq_length,
            # poros.moe streams per expert; block_size applies to dense only.
            block_size=(cfg.block_size or 4) if entry == "dense" else None,
            batch=cfg.per_device_batch_size, grad_accum=cfg.gradient_accumulation_steps,
            lora_rank=cfg.lora_rank, lora_dropout=cfg.lora_dropout,
            # "auto" loads block-streamed on CUDA (poros/train/model.py).
            load_strategy={"auto": "block_streamed"}.get(cfg.nf4_load_strategy,
                                                         cfg.nf4_load_strategy)
            if dense else None,
            selfcheck=should_run_startup_selfcheck(cfg.startup_parity_selfcheck),
            entry=entry,
            path=train_path(cfg),
            moe_compute=moe,
            fused_compute=fused,
            offload=bool(cfg.offload_embeddings_and_head),
            expert_cache=cfg.moe_expert_cache_gib if entry == "moe" else None,
            # read by block-major alone (the sequential loop reads nothing of it)
            dequant_cache=cfg.nf4_dequant_cache_mb
            if dense and train_path(cfg) != "sequential" else None,
        )

    def matches(self, key: dict, *, ignore: tuple = ()) -> bool:
        return all(_key_value(key, k) == v for k, v in vars(self).items() if k not in ignore)


# Knobs peak VRAM never decreases in. load_strategy and selfcheck are ordered
# by what they leave cached in torch's allocator: a gpu load and the startup
# self-check only ever add reserved memory, never allocated training memory.
_MONOTONE = ("seq_len", "lora_rank", "batch", "block_size", "lora_dropout")


_LOAD_ORDER = {None: 0, "block_streamed": 0, "gpu": 1}


def _gib(name: str) -> Optional[float]:
    hit = [k for k in GPU_SPECS_MIB if _norm(k) == _norm(name)]
    return GPU_SPECS_MIB[hit[0]] / 1024 if hit else None


# "Peak VRAM matched within 1% across GPUs in the one dense pair measured"
# (_OTHER_GPU): a won't-fit bound must clear free memory by more than that.
_TOL = 0.01


_KEYS = ("seq_len", "lora_rank", "batch", "block_size", "lora_dropout", "grad_accum",
         "load_strategy", "selfcheck")


def _split_le(small: tuple, large: tuple, path: str, entry: str) -> bool:
    """Whether a run split ``small`` (rows a micro-batch, micro-batches a step) holds no more than
    one split ``large``, on ``path``. The sequential loop holds one micro-batch at a time: its rows
    alone. Dense block-major holds every micro-batch's state between blocks and one micro-batch's
    activations: no more rows a micro-batch and no more a step, as every measured pair shows
    (RTX 4090, fused: 7B efficient 1x8 7.63 / 8.54 GiB under 8x1's 8.11 / 10.71, fast 1x8 8.28 /
    11.46 under 11.23 / 13.75, 72B efficient 1x8 8.48 / 10.56 under 8x1's 10.31 / 13.35; 0.5B
    fast 1x8 under 2x4). MoE block-major only at the same split: a layer's experts serve the
    union of the micro-batches' routes, and 1x8 measured above 8x1 (fast, cache 9 GiB: 17.12 /
    19.89 against 16.88 / 18.15; _MOE_SPLIT)."""
    (rs, gs), (rl, gl) = small, large
    if path == "sequential":
        return rs <= rl
    if entry == "moe":
        return (rs, gs) == (rl, gl)
    return rs <= rl and rs * gs <= rl * gl


def _cap_le(small: Any, large: Any) -> bool:
    """A cache cap that holds no more: the same one, or a smaller number."""
    if small == large:
        return True
    return isinstance(small, (int, float)) and isinstance(large, (int, float)) and small <= large


def _holds_no_more(small: dict, large: dict, path: str, entry: str) -> bool:
    """Beyond the ordered knobs (_MONOTONE): a run keyed ``small`` holds no more than one keyed
    ``large`` (both receipt keys or a request's fields): the same compute, its split no larger
    (_split_le), its embedding table and LM head on the GPU only if the other's are, and no larger
    a cache. Across the fused and PEFT-module computes nothing is ordered: each is the larger in
    some measured pair (_FUSED_OVER_PEFT)."""
    def v(k: dict, name: str) -> Any:
        return _key_value(k, name) if name in ("moe_compute", "fused_compute", "offload",
                                               "expert_cache", "dequant_cache") else k[name]

    return (v(small, "fused_compute") == v(large, "fused_compute")
            and v(small, "moe_compute") == v(large, "moe_compute")
            and _split_le((small["batch"], small["grad_accum"]),
                          (large["batch"], large["grad_accum"]), path, entry)
            and (v(small, "offload") or not v(large, "offload"))
            and _cap_le(v(small, "expert_cache"), v(large, "expert_cache"))
            and (path == "sequential" or _cap_le(v(small, "dequant_cache"),
                                                 v(large, "dequant_cache"))))


def bound(req: Request, gpu: PlanGpu) -> Optional[PlanBound]:
    """A fits / won't-fit verdict for ``req`` bounded by a receipt for the same
    model, or None. Never across models, never interpolated (PlanBound).

    Never used when a receipt measured this exact config on this GPU: that is
    the plan's own evidence, whatever it says. A config measured more than
    once (another dataset, loader or GPU) counts by its lowest peak, as the
    plan's figures do. A reserved peak counts only on the GPU it was measured
    on; elsewhere a run proves only that it completed within its card, and
    not under WSL2, where CUDA can spill into system memory. Beyond the ordered
    knobs, a bounding run holds at least as much (_holds_no_more): its compute,
    split, tables and caches.
    """
    free = gpu.free_gib if gpu.free_gib is not None else gpu.total_gib
    same = [c for c in _cells.here(gpu.name) if c["key"]["model"] == req.model
            and c["key"]["quantization"] == req.quantization and c["key"]["entry"] == req.entry
            and _key_path(c["key"]) == req.path and "peak_vram_allocated" in c["fields"]]

    def here(c: dict) -> bool:
        return _norm(c["key"]["gpu"]) == _norm(gpu.name)

    def floor(c: dict) -> bool:  # its allocated peak is at least what any card needs
        # not a cache grown to its card: a run with less free VRAM grows a smaller one, so its
        # peak is no floor of any run's, even on its own card (a refusal takes a fixed cache's)
        return not _grows_to_card(c["key"])

    if any(here(c) and req.matches(c["key"]) for c in same):
        return None

    def le(a: Any, b: Any) -> bool:  # a <= b on a knob; None (moe block_size) only equals None
        return a == b if a is None or b is None else a <= b

    def differs(c: dict) -> dict:
        return {k: [getattr(req, k), c["key"][k]] for k in _KEYS
                if getattr(req, k) != c["key"][k]}

    def holds_more(c: dict) -> bool:
        """The receipt's load path and self-check left at least as much cached."""
        load, asked = c["key"]["load_strategy"], req.load_strategy
        if load not in _LOAD_ORDER or asked not in _LOAD_ORDER:
            same_load = load == asked
        else:
            same_load = _LOAD_ORDER[load] >= _LOAD_ORDER[asked]
        return same_load and (not req.selfcheck or c["key"]["selfcheck"] is True)

    def peak(c: dict) -> float:
        return c["fields"]["peak_vram_allocated"]["value"]

    # A larger block holds more layers' activations, and on the sequential loop and fast fewer
    # saved block inputs: ordered on efficient alone (7B sequential block 1 8x1 9.53 GiB over
    # block 4's 8.71; 0.5B fast 2x4 block 1 3.15 over block 4's 3.06; efficient block 1 under
    # block 4 at every size measured).
    ordered = _MONOTONE if req.path == "efficient" else tuple(
        k for k in _MONOTONE if k != "block_size")

    def le_all(a: dict, b: dict) -> bool:
        return all(le(a[k], b[k]) for k in ordered) and (
            "block_size" in ordered or a["block_size"] == b["block_size"])

    ask = vars(req)
    lowest: dict[tuple, dict] = {}
    for c in same:
        if (floor(c) and le_all(c["key"], ask)
                and _holds_no_more(c["key"], ask, req.path, req.entry)):
            knobs = tuple(c["key"][k] for k in _MONOTONE)
            if knobs not in lowest or peak(c) < peak(lowest[knobs]):
                lowest[knobs] = c
    over = [c for c in lowest.values() if peak(c) > free * (1 + _TOL)]
    if over:
        c = min(over, key=peak)
        return PlanBound(
            fits=False, receipt=c["id"], gpu=c["key"]["gpu"], peak_gib=peak(c),
            differs=differs(c),
            basis=f"a run no larger ({c['id']}) peaked at {peak(c)} GiB allocated, "
                  f"over the {free} GiB here",
        )
    for c in same:
        if not (le_all(ask, c["key"]) and holds_more(c)
                and _holds_no_more(ask, c["key"], req.path, req.entry)):
            continue
        reserved = c["fields"].get("peak_vram_reserved", {}).get("value")
        if here(c) and reserved is not None:
            held, what = reserved, "reserved at most"
        elif "wsl" not in str(c["host"].get("os", "")).lower():
            held, what = _gib(c["key"]["gpu"]), "completed within its"
        else:
            continue
        if held is not None and held <= free:
            return PlanBound(
                fits=True, receipt=c["id"], gpu=c["key"]["gpu"], peak_gib=peak(c),
                differs=differs(c),
                basis=f"a run at least this large ({c['id']}, {c['key']['gpu']}) {what} "
                      f"{round(held, 2)} GiB; {free} GiB here",
            )
    return None


def nearest(req: Request, gpu: Optional[PlanGpu]) -> Optional[PlanNearest]:
    """The receipt for this model closest to ``req``: fewest differing knobs,
    this GPU first. Shown beside an unknown verdict; it decides nothing."""
    keys = (*_MONOTONE, "load_strategy", "selfcheck", "quantization", "entry")
    same = [c for c in _cells.here(gpu.name if gpu is not None else None)
            if c["key"]["model"] == req.model and _key_path(c["key"]) == req.path]
    if not same:
        return None

    def score(c: dict) -> tuple:
        diff = sum(c["key"][k] != getattr(req, k) for k in keys)
        here = gpu is not None and _norm(c["key"]["gpu"]) == _norm(gpu.name)
        return diff, not here, "peak_vram_allocated" not in c["fields"]

    c = min(same, key=score)
    peak = c["fields"].get("peak_vram_allocated", {}).get("value")
    return PlanNearest(receipt=c["id"], gpu=c["key"]["gpu"], peak_gib=peak,
                       differs={k: [getattr(req, k), c["key"][k]] for k in keys
                                if c["key"][k] != getattr(req, k)})


def _no_run(req: Optional[Request], gpu: Optional[PlanGpu]) -> str:
    """Why a figure is unknown, for someone who has not met this module's word for a run's record.
    "This length" only where the length is what no run here had."""
    here = [c for c in _cells.here(gpu.name if gpu else None) if req and gpu
            and c["key"]["model"] == req.model
            and _norm(c["key"]["gpu"]) == _norm(gpu.name)]
    what = "these settings" if any(c["key"]["seq_len"] == req.seq_len for c in here) else (
        "this length")
    return f"no measured run of this model at {what} on this card"


def _highest_of_repeats(cells: list[dict], name: str) -> list[dict]:
    """``cells``, of each key's repeats (the same config on the same GPU, measured more than
    once) only the one with the highest ``name`` figure, as receipt_figure takes it: a config
    bounds others from its highest peak, never from a lower repeat (an older tree's, or a run
    whose rows were shorter). Cells without the figure pass through."""
    top: dict = {}
    for c in cells:
        if name in c["fields"]:
            k = json.dumps(c["key"], sort_keys=True)
            if k not in top or c["fields"][name]["value"] > top[k]["fields"][name]["value"]:
                top[k] = c
    return [c for c in cells
            if name not in c["fields"] or top[json.dumps(c["key"], sort_keys=True)] is c]


# The expert cache's bytes and what comes with them: a 9 GiB cap held 6.09 GiB more than the
# fast path's default, a quarter of the NF4 experts (3.48 GiB): 5.52 GiB of cache, and its
# buffers (Qwen3-30B-A3B, RTX 4090: 16.88 against 10.80 GiB).
_CACHE_OVERHEAD = 1.15


def _text(c: dict) -> dict:
    """A wrapper's config.json holds its sizes in its text config."""
    return c.get("text_config") or c


def rope_head_dim(model_config: Optional[dict], model_type: Optional[str]) -> Optional[int]:
    """The head size of the q/k rotary kernels a fused run of this model launches: Qwen2's fused
    decoder layer runs them (poros.core.fused.FUSED_LAYERS) where the kernels take its head dim
    (poros.core.fused_tiles.rope_kernel_head_dim); None for every other model, or where
    config.json does not say."""
    if not model_config or (model_type or model_config.get("model_type")) != "qwen2":
        return None
    try:
        t = _text(model_config)
        hd = int(t.get("head_dim") or int(t["hidden_size"]) // int(t["num_attention_heads"]))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return 128  # the larger of the two Qwen2 head sizes: the stricter fit
    return rope_kernel_head_dim(hd)
