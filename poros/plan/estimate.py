"""What a plan estimates where no receipt measured the config: a MoE model's host RAM
(``moe_host_ram``) and SSD step time, what a block-major path holds on top of the sequential
loop (``path_extra``, ``token_bytes``), and the upper bound a measured run of the same model
puts on another setting of it (``_upper``, from config.json's sizes). Torch-free."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional, Sequence

from poros.core.arch_registry import registry_type
from poros.core.card import (
    _norm,
)
from poros.core.config import ssd_cache_bytes, train_path, trains_block_major
from poros.core.fused_tiles import dense_partial_bound
from poros.core.fused_tiles import pad as pad_rank
from poros.core.schemas import Figure, PlanGpu
from poros.moe.arch import MOE_FAMILIES, sparse_layers
from poros.plan import cells as _cells  # the receipt set, read at call time
from poros.plan.receipts import (
    _CACHE_OVERHEAD,
    _LOAD_ORDER,
    Request,
    _cap_le,
    _card_free,
    _card_growth,
    _grows_request,
    _grows_to_card,
    _key_path,
    _key_value,
    _text,
)

if TYPE_CHECKING:
    from poros.core.config import PorosConfig


# Streamed MoE load (poros.moe.load): host RSS peaks near the NF4 experts
# plus the dense weights plus one checkpoint shard. NF4 with double quant is
# 33/64 bytes per weight (4-bit codes and an 8-bit absmax code per 64
# weights). A shard is taken at 5 GB (4.66 GiB, what Hub checkpoints are
# usually cut at), or the whole bf16 checkpoint if smaller; the process adds
# 1.5 GiB. Check: the receipt's 30B-A3B load peaked at 22.04 GiB, with 13.93
# GiB of NF4 and 2.87 GiB of bf16 dense weights (a 5.2 GiB rest for its ~3.7
# GiB shards and the process); this estimates 22.9 GiB.
_SHARD_GIB, _PROCESS_GIB = 5e9 / 2**30, 1.5


_MOE_HOST_RECEIPT = "docs/validation/moe/moe_streamed_load_3090.json#/full_streamed_load"


_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}


def _moe_params(c: dict) -> tuple[int, int]:
    """(routed expert, everything else) parameter counts from a MOE_FAMILIES
    config.json, exact to the norms for qwen3_moe, qwen2_moe, olmoe and mixtral.
    Attention is counted as full q/k/v/o projections, which over-counts
    DeepSeek-V2's compressed (MLA) attention."""
    c = _text(c)
    experts = c.get("num_experts") or c.get("num_local_experts") or c["n_routed_experts"]
    sparse = len(sparse_layers(SimpleNamespace(**{**c, "num_experts": experts})))
    h, layers, heads = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"]
    head_dim = ((c.get("qk_nope_head_dim") or 0) + (c.get("qk_rope_head_dim") or 0)
                or c.get("head_dim") or h // heads)
    mid = c[MOE_FAMILIES[registry_type(c)]]
    shared = ((c.get("shared_expert_intermediate_size") or 0)
              + (c.get("n_shared_experts") or 0) * mid)
    dense = (c["vocab_size"] * h * (1 if c.get("tie_word_embeddings") else 2)
             + layers * 2 * h * head_dim * (heads + (c.get("num_key_value_heads") or heads))
             + (layers if c.get("enable_moe_block") else layers - sparse)
             * 3 * h * (c.get("intermediate_size") or 0)
             + sparse * h * (experts + 3 * shared))  # router, shared experts
    return sparse * experts * 3 * h * mid, dense


def _ceil1(v: float) -> float:
    """An estimate to 0.1 GiB, rounded up: the JSON never reads below what it bounds."""
    return math.ceil(round(v * 10, 6)) / 10


def moe_host_ram(model_config: dict, torch_dtype: str, *, store: str = "ram",
                 sss_ram: int = 4 << 30, sss_cache: int = 0,
                 expert_format: Optional[str] = None) -> Figure:
    """Peak host RAM of a streamed MoE run, from config.json alone.

    ``store="ssd"`` keeps the NF4 experts in a ``.sss`` file instead of RAM,
    which then holds the pinned read ring (at most ``sss_ram`` bytes) and the
    host cache (``sss_cache`` bytes, at most the experts'). ``expert_format``: the run's
    ``moe_expert_format``, for a family whose experts it may hold as released (_expert_bytes).
    """
    try:
        experts, dense = _moe_params(model_config)
        per = _expert_bytes(model_config, expert_format)
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as exc:
        return _unknown("GiB", f"config.json does not give the MoE sizes ({type(exc).__name__})")
    shard = min(_SHARD_GIB, (experts + dense) * 2 / 2**30)
    dense_part = (f"{dense / 1e9:.2f}B dense weights in {torch_dtype} + one "
                  f"{shard:.2f} GiB checkpoint shard + {_PROCESS_GIB} GiB for the process")
    if store == "ssd":
        # SsdSource sizes its ring to (max_depth + 2) slabs, capped by sss_ram.
        text = _text(model_config)
        slab = 3 * text["hidden_size"] * text[MOE_FAMILIES[registry_type(text)]] * per
        ring = min(sss_ram, _SSD_RING_SLABS * (-(-slab // 4096) * 4096))
        cache = min(int(sss_cache), int(experts * per) + 4096 * _SSD_RING_SLABS)
        weights = ring + cache + dense * _DTYPE_BYTES[torch_dtype]
        held = f" and the {cache / 2**30:.2f} GiB host cache (sss_cache)" if cache else ""
        return Figure(value=_ceil1(weights / 2**30 + shard + _PROCESS_GIB), unit="GiB",
                      source="estimated", receipt=_MOE_HOST_RECEIPT,
                      method=f"from config.json: the RAM-store estimate with its "
                      f"{experts * per / 2**30:.1f} GiB of experts replaced by the "
                      f"{ring / 2**30:.2f} GiB read ring ({_SSD_RING_SLABS} expert slabs, "
                      f"capped by sss_ram){held}: {dense_part}")
    weights = experts * per + dense * _DTYPE_BYTES[torch_dtype]
    return Figure(value=_ceil1(weights / 2**30 + shard + _PROCESS_GIB), unit="GiB",
                  source="estimated", receipt=_MOE_HOST_RECEIPT,
                  method=f"from config.json: {experts / 1e9:.2f}B expert weights in NF4 "
                  f"+ {dense_part}; "
                  "the receipt is the 30B-A3B load it reproduces within 4%")


_PROBE_RECEIPT = "~/.cache/poros/sss_probe.json"


_SSD_RING_SLABS = 264  # poros.sss.ssd: 4 * max_depth (64) + 8 slabs


def moe_ssd_s_per_step(model_config: dict, cfg: "PorosConfig", probe, ram: Figure, *,
                       vram_cache_gib: Optional[float] = None) -> tuple[Figure, Optional[str]]:
    """Seconds per step with the experts on an SSD: compute + streamed bytes / B.

    Every MoE layer's experts are read by the forward and again by the MoE
    backward (the dense block rematerialization reuses the forward's expert
    output), minus the layer the one-layer cache still holds at the turn:
    ``2 - 1 / layers`` times the NF4 experts -- once an update on a block-major
    path (``efficient``, ``fast``: a layer runs every micro-batch while its
    experts are on the GPU), once a micro-batch on the sequential loop.
    Experts no token picks are not read, so this is an upper bound (RTX 4090,
    Qwen3-30B-A3B, 8 x 512 tokens: 1.95x the experts an update at mb8 x 1 and at
    mb1 x 8 block-major, docs/validation/sss/4090/). On ``path: fast`` the
    expert cache grown into free VRAM after update 1 (``vram_cache_gib``: its cap,
    or what a receipt of this config on this GPU grew) is read by neither pass;
    unsized, the figure is unknown (a guess there would be off by the cache).
    The host cache (``store_options: {cache: ...}``) serves the backward's first
    layers and the next pass's from RAM: up to twice its size fewer bytes from the
    disk each pass pair (measured: 20.5 GB an update against 29.4 at 4 GiB,
    efficient; none once the two caches hold every expert, fast). The reads overlap
    the compute only in part, so the two add, an upper bound (RTX 4090, Qwen3-30B-A3B, 8 x
    512: fast 3.2 s for 1.36 s of compute and 1.94 of reads; efficient 10.0 s for 2.5 and
    9.2 at the probed 3.2 GB/s; fast with a 4 GiB host cache 1.46 against 1.42 in RAM).
    """
    try:
        experts, _ = _moe_params(model_config)
        layers = len(sparse_layers(SimpleNamespace(**{
            **model_config, "num_experts": model_config.get("num_experts")
            or model_config.get("num_local_experts") or model_config["n_routed_experts"]})))
    except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as exc:
        return (_unknown("s", f"config.json does not give the MoE sizes ({type(exc).__name__})"),
                None)
    passes = 2 - 1 / max(layers, 1)
    micro = 1 if trains_block_major(cfg) else cfg.gradient_accumulation_steps
    nf4 = experts * 33 / 64
    held = ""
    if train_path(cfg) == "fast":
        if vram_cache_gib is None:
            return (_unknown("s", "path fast grows its expert cache into this GPU's free VRAM "
                                  "after update 1, and no receipt of this config on this GPU "
                                  "says how large (moe_expert_cache_gib sets it)"), None)
        vram = min(max(float(vram_cache_gib), 0.0) * 2**30, nf4)
        reads, passes = 2 * (nf4 - vram), 2.0  # the cache holds the turn's layer too
        held = f", less the {vram / 2**30:.2f} GiB expert cache on the GPU"
    else:
        reads = passes * nf4
    host = min(ssd_cache_bytes(cfg.store_options), nf4)
    if host:
        held += f", less twice the {host / 2**30:.2f} GiB host cache"
    streamed = max(reads - 2 * host, 0.0) * micro
    disk = streamed / probe.bandwidth
    how = "O_DIRECT" if probe.direct else "buffered (O_DIRECT refused)"
    how_often = "an update, block-major" if micro == 1 else f"x {micro} micro-batches"
    method = (f"RAM-store s/step + {streamed / 1e9:.1f} GB streamed per step "
              f"({passes:.2f}x the NF4 experts {how_often}{held}) / "
              f"{probe.bandwidth / 1e9:.2f} GB/s probed = {disk:.1f} s; disk "
              f"{probe.b_ssd / 1e9:.2f} GB/s {how}, H2D "
              + ("not probed" if probe.b_h2d is None else f"{probe.b_h2d / 1e9:.2f} GB/s"))
    if ram.value is None:
        return Figure(value=round(disk, 1), unit="s", source="estimated",
                      receipt=_PROBE_RECEIPT,
                      method=method + "; no RAM-store receipt, so this is the disk floor"
                      ), "disk"
    bound = "disk" if disk > ram.value else "compute"
    return Figure(value=round(disk + ram.value, 1), unit="s", source="estimated",
                  receipt=ram.receipt or _PROBE_RECEIPT, method=method), bound


@dataclass
class PathExtra:
    """What block-major accumulation's path holds beyond the sequential loop at the same settings,
    in GiB, and how that was sized (path_extra)."""

    vram: float
    host: float
    what: str


# A block's input under NF4 block-major is fp32 (k-bit preparation makes the norms fp32).
_BOUNDARY_BYTES = 4


# fp32 tensors per hidden unit and per MLP unit a decoder layer keeps for its backward (bound).
_ACT_HIDDEN, _ACT_MLP = 8, 4


# Block inputs' worth of state block-major holds per micro-batch beyond the sequential loop: the
# gradient, the final hidden state and its gradient, the block's inputs in hand (bound); and on
# efficient, the next block's inputs on their way up.
_HELD_PER_MICRO_BATCH, _ON_THEIR_WAY = 5, 3


# A host-RAM peak moves a few MB run to run; block-major's errs high by this much on top.
_HOST_SLACK = 1 << 28


# Bytes per element of the dropout walk's scratch draw at a LoRA dropout above 0 on a dense
# model (poros.core.sequential_rng): the input and its output in fp32 at most, and the mask.
_WALK_BYTES = 9


def path_extra(cfg: "PorosConfig", model_config: Optional[dict], gpu: Optional[PlanGpu],
               entry: Optional[str]) -> Optional[PathExtra]:
    """What ``cfg``'s block-major path holds on top of the sequential loop at the same settings,
    from config.json; None where config.json does not give the sizes.

    Block-major keeps state for all G micro-batches of an update where the sequential loop keeps
    one's: each micro-batch's gradient between blocks, its final hidden state and attention
    mask, and the inputs of the block in hand. ``efficient`` also has the next block's inputs on
    their way up from host RAM, where all the others wait (G x blocks x rows x length x hidden x
    4 B); ``fast`` keeps them all in VRAM, with one block's dequantized NF4 weights (at most
    half of the card). Either holds the activations of the block's other layers for one
    micro-batch, where the sequential loop's per-layer checkpointing holds one layer's. On a MoE
    model, one layer's experts dequantized in bf16 too
    (the expert cache block-major keeps for a layer's visit); the expert residency it grows after
    update 1 takes only VRAM left free. Each term errs high, so the estimate does: checked against
    the RTX 4090 receipts in docs/validation/block-major-dropout/ (tests/test_plan_paths.py)."""
    path = train_path(cfg)
    if path == "sequential" or not model_config:
        return None
    try:
        t = _text(model_config)
        h, layers = int(t["hidden_size"]), int(t["num_hidden_layers"])
        heads = int(t["num_attention_heads"])
        kv = int(t.get("num_key_value_heads") or heads)
        hd = int(t.get("head_dim") or h // heads)
        mid = int(t.get("intermediate_size") or 0)
    except (KeyError, TypeError, ValueError):
        return None
    rows, length, g = cfg.per_device_batch_size, cfg.max_seq_length, cfg.gradient_accumulation_steps
    size = (cfg.block_size or 4) if entry == "dense" else 1
    blocks = -(-layers // size)
    one = rows * length * h * _BOUNDARY_BYTES  # one block input of one micro-batch
    held = g * blocks * one  # every block input of an update
    per_update = (_HELD_PER_MICRO_BATCH + (_ON_THEIR_WAY if path == "efficient" else 0)) * g * one
    masks = g * rows * length * length * 4
    vram = float(per_update + masks)
    said = [f"{(per_update + masks) / 2**30:.2f} GiB for {g} micro-batches' state at once "
            f"({per_update // (g * one)} x {g} x {rows} x {length} x {h} x 4 B, and their "
            "attention masks)"]
    if path == "efficient":
        host = float(held) + _HOST_SLACK
        said.append(f"{held / 2**30:.2f} GiB of block inputs in host RAM ({g} x {blocks} blocks "
                    f"x {rows} x {length} x {h} x 4 B)")
    else:
        host = float(_HOST_SLACK)
        vram += held
        said.append(f"{held / 2**30:.2f} GiB of block inputs in VRAM ({g} x {blocks} blocks x "
                    f"{rows} x {length} x {h} x 4 B)")
    cache_mb = cfg.nf4_dequant_cache_mb
    if cache_mb or (cache_mb is None and path == "fast"):
        block = size * (h * hd * (2 * heads + 2 * kv) + 3 * h * mid) * 2  # bf16 copies
        cap = (gpu.total_gib * 2**30 / 2) if gpu is not None else block
        cache = min(block, cap) if cache_mb is None else cache_mb * 2**20
        vram += cache
        said.append(f"{cache / 2**30:.2f} GiB of one block's dequantized weights")
    if entry == "moe":  # one MoE layer's experts, dequantized once a pass for every micro-batch
        try:
            n = int(t.get("num_experts") or t.get("num_local_experts") or t["n_routed_experts"])
            experts = n * 3 * h * int(t[MOE_FAMILIES[registry_type(t)]]) * 2
        except (KeyError, TypeError, ValueError):
            return None
        vram += experts
        said.append(f"{experts / 2**30:.2f} GiB of one MoE layer's experts, dequantized")
    if size > 1:
        act = (size - 1) * _layer_activations(rows, length, h, heads, mid)
        vram += act
        said.append(f"{act / 2**30:.2f} GiB for {size - 1} more layers' activations of one "
                    "micro-batch")
    if entry == "dense" and cfg.lora_dropout > 0:
        # the walk draws each new input size once on a scratch buffer, when a micro-batch is
        # fetched: transient, but counted (its widest input is the MLP's)
        walk = rows * length * max(h, mid) * _WALK_BYTES
        vram += walk
        said.append(f"{walk / 2**30:.2f} GiB for the dropout walk's widest scratch draw")
    return PathExtra(vram=vram / 2**30, host=host / 2**30, what="; ".join(said))


def _layer_activations(rows: int, length: int, h: int, heads: int, mid: int) -> float:
    """Bytes one decoder layer holds for its backward, one micro-batch, fp32: a bound of the
    tensors autograd keeps (the hidden-sized ones, the MLP's, and the math attention's scores
    and probabilities, which deterministic training runs)."""
    return rows * length * (_ACT_HIDDEN * h + _ACT_MLP * mid) * 4 + rows * heads * length**2 * 4 * 2


def token_bytes(text: dict, *, path: str, entry: str, block_size: Optional[int],
                shapes: Sequence[tuple[int, int]]) -> Optional[float]:
    """Bytes of VRAM a block-major update holds that grow with its micro-batches' ``shapes``
    (rows, padded length each): path_extra's per-micro-batch terms (block-major's state and the
    attention masks, and on ``fast`` every block input) plus one layer's activations for the
    largest micro-batch (a MoE layer's at its routed width), each a bound that errs high. The
    difference between two updates' figures is what the larger adds to a measured peak
    (``poros.train.compute``: the expert cache sized before an update at the run's full shape
    is seen). None where the config does not give the sizes."""
    try:
        h, layers = int(text["hidden_size"]), int(text["num_hidden_layers"])
        heads = int(text["num_attention_heads"])
        mid = int(text.get("intermediate_size") or 0)
        if entry == "moe":
            routed = int(text.get("num_experts_per_tok") or text.get("top_k_experts") or 0)
            width = text.get(MOE_FAMILIES.get(registry_type(text), ""), 0)
            mid = max(mid, routed * int(width or 0))
    except (KeyError, TypeError, ValueError):
        return None
    size = (block_size or 4) if entry == "dense" else 1
    tokens = sum(r * n for r, n in shapes)
    per = (_HELD_PER_MICRO_BATCH + (_ON_THEIR_WAY if path == "efficient" else 0)) * tokens
    held = -(-layers // size) * tokens if path == "fast" else 0
    masks = sum(r * n * n * 4 for r, n in shapes)
    act = max((_layer_activations(r, n, h, heads, mid) for r, n in shapes), default=0.0)
    return float((per + held) * h * _BOUNDARY_BYTES + masks + size * act)


# -- upper bounds from a receipt at other settings ----------------------------------------------
# Every figure below is at least what a run uses: a receipt's own peak where it ran the config,
# else the lowest of the upper bounds each receipt of the model gives (_upper): its peak, scaled
# where it ran another compute or split by the largest ratio measured between the two, rounded
# up, plus what config.json says the config holds beyond it (_variable), where it holds more.
#
# The fused NF4 compute (poros.core.fused) against PEFT's modules, in pairs on one RTX 4090
# (block-major-dropout/receipts, block-major-dropout/rtx4090_72b, rtx4090-rc): fused over PEFT
# at most 1.274 allocated on one card (7B sequential, one row a micro-batch: 6.192 -> 7.888 GiB;
# 1.285 against the RTX 3090's 6.14) and 1.054 reserved (72B efficient block 1 1x8, tables
# offloaded: 10.02 -> 10.56); PEFT over fused at most 1.58 allocated and 2.00 reserved (0.5B fast
# 2x4: 3.118 / 9.115 against 1.974 / 4.561, one tree). The 0.5B's cells are now the release
# tree's (docs/validation/fused/rank_tiles/rtx4090), whose pairs sit inside both: fused over PEFT
# 1.24 / 0.78 (1x8), 1.22 / 0.61 (2x4), 1.20 / 0.78 (rank 64), 1.17 / 0.81 (rank 128).
_FUSED_OVER_PEFT = (1.30, 1.10)


_PEFT_OVER_FUSED = (1.60, 2.05)


# The fused MoE expert compute against the reference (Qwen3-30B-A3B, path fast, one tree,
# rtx4090-rc/receipts/q30rc_*: fused at a 9 GiB expert cache cap, the reference at 6, each cache
# full): less its cache, the reference holds 1.35x the fused compute's allocated peak (8x1:
# 10.63 against 7.88 GiB; 4x2: 11.25 against 8.68) and 1.50x its reserved (13.98 against 9.29;
# 14.95 against 10.32). A reference receipt bounds the fused compute as it is.
_MOE_FUSED_OVER_REF = (1.00, 1.00)


_MOE_REF_OVER_FUSED = (1.35, 1.50)


# MoE at another split (_split_le), cache cap 9 GiB, every pair of receipts (rtx4090-rc and
# moe/4090, three trees): at most 1.083 allocated (fast 4x2 17.68 GiB against 8x1 16.33, a later
# tree's; 16.88 on its own) and 1.133 reserved (fast 1x8 19.89 against 8x1 17.55, a later tree's;
# 18.15 on its own).
_MOE_SPLIT = (1.09, 1.14)


def _same_shapes(model: str) -> set[str]:
    """The receipt models whose config.json gives ``model``'s shapes, so the same memory
    (poros/plan/shapes.json, built from each model's Hub config.json): a receipt of one plans
    the other, and says so (as the catalog plans a same-shape entry by the model it matches)."""
    models = _cells.SHAPES["models"]
    sig = models.get(model, {}).get("signature")
    return {m for m, e in models.items() if sig is not None and m != model
            and e["signature"] == sig}


# A reserved peak grows by the allocator's rounding over the bytes it adds.
_RESERVED_SLACK = 1.10


@dataclass(frozen=True)
class _Sizes:
    """What config.json says of a model's sizes, as _variable reads them."""

    h: int
    layers: int
    heads: int
    hd: int
    mid: int  # a token's MLP width: the dense intermediate size, or a MoE's routed experts'
    vocab: int
    tables: int  # bytes of the embedding table and LM head on the GPU, fp32 (one if tied)
    per_rank: int  # LoRA parameters per unit of rank, every adapted projection
    layer_bf16: int  # bytes of one layer's weights dequantized
    experts_nf4: int  # bytes of every MoE expert in NF4 (33/64 bytes a weight); 0 for a dense
    expert_slabs: int = 0  # the SSD store's slabs: one per (layer, expert)
    # decoder-layer calls a forward makes: the layers, plus a looped decoder's extra passes over
    # its loop (Inkling 2 Mini: 24 layers, 48 visits); each saves its input for the backward
    visits: int = 0


def _inkling2_released_bytes(layers: int) -> float:
    """Bytes a weight of Inkling 2 Mini's experts takes kept as released
    (poros.moe.inkling2.released): NVFP4 (codes, a scale per 16, a factor per expert: 9/16) on
    all but its last layers, MXFP8 (a value and a scale per 32: 33/32) on those; the release
    has 4 of 24 layers in MXFP8, counted here as one in six, rounded up."""
    mx = -(-layers // 6)
    return ((layers - mx) * 9 / 16 + mx * 33 / 32) / layers


#: Families whose experts a run may keep in their released format, and its bytes a weight.
_RELEASED_STORE = {"inkling2_loop": _inkling2_released_bytes,
                   "inkling2_loop_text": _inkling2_released_bytes}


def _expert_bytes(model_config: dict, expert_format: Optional[str]) -> float:
    """Bytes a routed expert weight takes as the run holds it: the format the run records
    (poros.moe.inkling2.formats: ``expert_format``, the run's ``moe_expert_format``, else
    POROS_INKLING_EXPERTS, then what config.json says the checkpoint stores; the plan passes
    ``formats.expected``, which reads all that is in hand without the weights), the released bytes
    or NF4's 33/64; NF4's for every other family."""
    import os

    from poros.moe.inkling2.formats import RELEASED, from_config

    t = _text(model_config)
    store = _RELEASED_STORE.get(registry_type(t))
    if store is None:
        return 33 / 64
    asked = expert_format or os.environ.get("POROS_INKLING_EXPERTS")
    if from_config(model_config, asked) == RELEASED:
        return store(int(t["num_hidden_layers"]))
    return 33 / 64


def _sizes(model_config: Optional[dict], expert_format: Optional[str] = None
           ) -> Optional[_Sizes]:
    """config.json's sizes; ``expert_format``: the run's ``moe_expert_format`` (a MoE family
    whose experts a run may hold in another format: _expert_bytes)."""
    if not model_config:
        return None
    try:
        t = _text(model_config)
        h, layers = int(t["hidden_size"]), int(t["num_hidden_layers"])
        heads = int(t["num_attention_heads"])
        kv = int(t.get("num_key_value_heads") or heads)
        hd = int(t.get("head_dim") or h // heads)
        vocab = int(t["vocab_size"])
        tied = bool(model_config.get("tie_word_embeddings", t.get("tie_word_embeddings", False)))
        mid = int(t.get("intermediate_size") or 0)
        attn = h * hd * (2 * heads + 2 * kv)
        attn_lora = (h + heads * hd) + 2 * (h + kv * hd) + (heads * hd + h)
        n = t.get("num_experts") or t.get("num_local_experts") or t.get("n_routed_experts")
        experts = slabs = 0
        if n:  # a MoE: the routed experts' widths, every expert adapted
            emid = int(_expert_width(t))
            top = int(t.get("num_experts_per_tok") or 1)
            width = max(mid, top * emid)
            mlp, mlp_lora = int(n) * 3 * h * emid, int(n) * 3 * (h + emid)
            experts = int(layers * mlp * _expert_bytes(model_config, expert_format))
            slabs = layers * int(n)
        else:
            width, mlp, mlp_lora = mid, 3 * h * mid, 3 * (h + mid)
        visits = _visits(t, layers)  # a loop_config that does not read: unknown, as the rest
    except (KeyError, TypeError, ValueError):
        return None
    return _Sizes(h=h, layers=layers, heads=heads, hd=hd, mid=width, vocab=vocab,
                  tables=(1 if tied else 2) * vocab * h * 4,
                  per_rank=layers * (attn_lora + mlp_lora), layer_bf16=(attn + mlp) * 2,
                  experts_nf4=experts, expert_slabs=slabs, visits=visits)


#: A config saved by the official transformers class (model_type inkling2_loop_text) words
#: Inkling 2 Mini's expert width and loop by these names; the release's config.json uses
#: intermediate_size and loop_config. Absent ones take the class's defaults.
_OFFICIAL_LOOP = (("loop_layer_index_start", 6), ("loop_layer_index_end", 18),
                  ("loop_num_passes", 3))


def _expert_width(t: dict) -> Any:
    """A MoE text config's routed expert width, by its family's name for it (MOE_FAMILIES); an
    Inkling 2 Mini config in the official class's layout names it moe_intermediate_size."""
    kind = registry_type(t)
    width = t.get(MOE_FAMILIES[kind])
    if width is None and kind in _RELEASED_STORE:
        width = t["moe_intermediate_size"]
    if width is None:
        raise KeyError(MOE_FAMILIES[kind])
    return width


def _visits(t: dict, layers: int) -> int:
    """Decoder-layer calls one forward makes, from config.json: a looped decoder's
    ``loop_config`` (core_start, core_end, num_passes; Inkling 2 Mini's release), or the
    official class's ``loop_layer_index_start``, ``loop_layer_index_end`` and
    ``loop_num_passes``, adds a call per extra pass for each loop layer; any other decoder calls
    each layer once. A field that is no number raises (``_sizes``: unknown)."""
    loop = t.get("loop_config")
    if isinstance(loop, dict):
        start, end = int(loop.get("core_start", 0)), int(loop.get("core_end", 0))
        passes = int(loop.get("num_passes", 1))
    elif any(name in t for name, _ in _OFFICIAL_LOOP):
        start, end, passes = (int(t.get(name, default)) for name, default in _OFFICIAL_LOOP)
    else:
        return layers
    return layers + max(passes - 1, 0) * max(end - start, 0)


@dataclass(frozen=True)
class _Knobs:
    """What of a run's settings its peak grows with, each ordered so more holds more."""

    rows: int  # a micro-batch's
    tokens: int  # rows a step: block-major holds every micro-batch's state between blocks
    length: int
    block: int  # layers held at once: their activations and dequantized weights
    blocks: int  # the block count: the sequential loop's saved block inputs, fast's held ones
    rank: int
    dropout: bool
    tables: bool  # the embedding table and LM head on the GPU (not offloaded)
    cache: int  # a MoE block-major run's expert cache, bytes


def _knobs(k: dict, path: str, s: Optional[_Sizes]) -> _Knobs:
    """A receipt's key, or a request's fields, on ``path``; the block count and the expert
    cache only with sizes (else 0: nothing is grown without them)."""
    block = k["block_size"] or 1
    cache = 0
    if s is not None and k["entry"] == "moe" and path != "sequential":
        cap = _key_value(k, "expert_cache")
        # unset: efficient keeps one layer's (path_extra's); fast grows it to what its card
        # leaves (_grows_to_card), compared only with a run of that rule on the same card, so
        # its size cancels; a cap holds at most every expert
        cache = (min(int(cap * 2**30), s.experts_nf4) if cap is not None
                 else s.experts_nf4 if path == "fast" else 0)
    # the sequential loop holds one micro-batch at a time: its rows a step do not count
    tokens = 0 if path == "sequential" else k["batch"] * k["grad_accum"]
    return _Knobs(rows=k["batch"], tokens=tokens, length=k["seq_len"],
                  # the sequential loop saves one input per block visit: a looped decoder's
                  # visits, not its layer count
                  block=block, blocks=-(-(s.visits or s.layers) // block) if s is not None else 0,
                  rank=k["lora_rank"], dropout=k["lora_dropout"] > 0,
                  tables=not _key_value(k, "offload"), cache=cache)


def _join(a: _Knobs, b: _Knobs) -> _Knobs:
    """The larger of each: a run that holds at least what either holds."""
    return _Knobs(*(max(x, y) for x, y in zip(dataclasses.astuple(a), dataclasses.astuple(b))))


def _variable(kn: _Knobs, s: _Sizes, path: str, fused: bool = False) -> float:
    """Bytes of a run's peak that grow with its knobs, a bound of each term (each errs high): a
    block's layers' activations of one micro-batch and their dropout draws, a block's weights
    dequantized, one micro-batch's logits with their gradient and softmax in fp32, the adapters
    with their gradients and Adam state, the adapters' A products a block's layers keep for the
    backward ([tokens, rank] fp32 for each of a layer's seven projections, the rank padded as the
    fused kernels take it), with ``fused`` (a dense model's fused compute) the most one fused call
    holds in partials at that rank or a lower one (poros.core.fused_tiles.dense_partial_bound),
    the tables on the GPU, the expert cache, and the state between blocks (the sequential loop:
    one micro-batch's saved block inputs; block-major: every micro-batch's, path_extra's
    terms)."""
    act = kn.block * _layer_activations(kn.rows, kn.length, s.h, s.heads, s.mid)
    drop = kn.block * kn.rows * kn.length * (2 * s.h + s.heads * s.hd + s.mid) * 4 \
        if kn.dropout else 0
    one = kn.length * s.h * _BOUNDARY_BYTES
    if path == "sequential":
        state = kn.blocks * kn.rows * one
    else:
        state = ((_HELD_PER_MICRO_BATCH + (_ON_THEIR_WAY if path == "efficient" else 0))
                 * kn.tokens * one + kn.tokens * kn.length ** 2 * 4
                 + (kn.blocks * kn.tokens * one if path == "fast" else 0))
    tokens = kn.rows * kn.length
    a_products = kn.block * 7 * tokens * pad_rank(kn.rank) * 4
    partials = (dense_partial_bound(kn.rank, tokens, s.mid, s.h)
                if fused and not s.experts_nf4 else 0)
    return float(act + drop + kn.block * s.layer_bf16 + 3 * kn.rows * kn.length * s.vocab * 4
                 + kn.rank * s.per_rank * 20 + a_products + partials
                 + (s.tables if kn.tables else 0) + _CACHE_OVERHEAD * kn.cache + state)


# Margins on an upper bound from a receipt at other settings: its peak moved up to 1% between
# runs of one config (another dataset, another accumulation on the sequential loop: 32B 10.96 /
# 11.02 GiB, 7B 6.19 / 6.24), and up to 5% across cards (Qwen3-30B-A3B sequential: 8.36 GiB on an
# RTX 3090, 8.78 on an L40S); a reserved peak up to 4% between repeats (72B efficient 8x1: 13.35
# and 13.80 GiB). A reserved peak is bounded only where the activations do not grow: the
# allocator's blocks for more of them grew far past their bytes (0.5B fast 1x8 2.25 GiB reserved,
# 2x4 4.56; with PEFT's modules 4.25 and 9.12).
_SAME_GPU, _CROSS_GPU, _RESERVED = 1.01, 1.06, 1.05


def _upper(name: str, req: Request, c: dict, s: Optional[_Sizes], gpu: Optional[PlanGpu],
           lift: Optional[float]) -> Optional[tuple[float, str, str, bool]]:
    """(an upper bound of ``req``'s ``name`` peak in GiB, how, the receipt, whether it is the
    sequential loop's plus what ``req``'s path holds on top) from cell ``c``, or None where ``c``
    bounds nothing. ``lift``: what ``req``'s block-major path holds over the sequential loop at
    its settings (path_extra, GiB), so a sequential receipt bounds it; None where config.json
    does not size it."""
    k = c["key"]
    if not (name in c["fields"] and k["quantization"] == req.quantization
            and k["entry"] == req.entry
            and (k["model"] == req.model or k["model"] in _same_shapes(req.model))):
        return None
    here = gpu is not None and _norm(k["gpu"]) == _norm(gpu.name)
    reserved = name == "peak_vram_reserved"
    path = _key_path(k)
    lifted = path == "sequential" and req.path != "sequential"
    if (reserved and not here) or (path != req.path and not lifted) or (lifted and lift is None):
        return None
    # an expert cache grown to its card (_grows_to_card) bounds, and is bounded by, only a run
    # that grows it by the same rule on the same card: elsewhere it is another size
    grows = _grows_request(req)
    if (_grows_to_card(k) or grows) and not (_grows_to_card(k) and grows and here):
        return None
    # on its card: a run with more free VRAM than the receipt's grows a larger cache
    growth = _card_growth(c, gpu, s) if _grows_to_card(k) else 0.0
    if growth is None:
        return None
    if (s is None and not lifted and req.path != "sequential"
            and (req.expert_cache, req.dequant_cache) != (_key_value(k, "expert_cache"),
                                                          _key_value(k, "dequant_cache"))):
        return None  # caches sized by config.json alone
    if (req.entry == "dense" and req.path != "sequential" and not lifted
            and not _cap_le(req.dequant_cache, _key_value(k, "dequant_cache"))):
        return None
    if reserved:  # its load path and self-check left at least as much cached
        load, asked = k["load_strategy"], req.load_strategy
        same_load = (_LOAD_ORDER[load] >= _LOAD_ORDER[asked]
                     if load in _LOAD_ORDER and asked in _LOAD_ORDER else load == asked)
        if not same_load or (req.selfcheck and k["selfcheck"] is not True):
            return None
    i, factor, said = int(reserved), 1.0, []
    if req.entry == "dense" and _key_value(k, "fused_compute") != req.fused_compute:
        r = (_FUSED_OVER_PEFT if req.fused_compute else _PEFT_OVER_FUSED)[i]
        factor *= r
        said.append(f"x {r}, its {'PEFT modules' if req.fused_compute else 'fused compute'} "
                    f"as the {'fused compute' if req.fused_compute else 'PEFT modules'}")
    if req.entry == "moe" and _key_value(k, "moe_compute") != req.moe_compute:
        r = (_MOE_FUSED_OVER_REF if req.moe_compute == "fused" else _MOE_REF_OVER_FUSED)[i]
        factor *= r
        said.append(f"x {r}, its {_key_value(k, 'moe_compute')} expert compute as the "
                    f"{req.moe_compute}")
    if req.entry == "moe" and (req.batch, req.grad_accum) != (k["batch"], k["grad_accum"]):
        factor *= _MOE_SPLIT[i]
        said.append(f"x {_MOE_SPLIT[i]}, its split {k['batch']}x{k['grad_accum']} as "
                    f"{req.batch}x{req.grad_accum}")
    have = _knobs(k, req.path if lifted else path, s)
    want = _join(_knobs(vars(req), req.path if lifted else path, s), have)
    if lifted:  # the sequential loop keeps no expert cache: the request's is all to add
        have = dataclasses.replace(have, cache=0)
    grown = 0.0
    if want != have:
        if s is None:
            return None
        moved = [f.name for f in dataclasses.fields(_Knobs)
                 if getattr(want, f.name) != getattr(have, f.name)]
        if reserved and {"rows", "tokens", "length"} & set(moved):
            return None
        fused = req.entry == "dense" and req.fused_compute  # the compute the request runs
        grown = (_variable(want, s, path, fused) - _variable(have, s, path, fused)) / 2**30
        grown *= _RESERVED_SLACK if reserved else 1
        said.append(f"+ {grown:.2f} GiB for {', '.join(moved)} (config.json)")
    margin = _RESERVED if reserved else _SAME_GPU if here else _CROSS_GPU
    value = c["fields"][name]["value"] * factor * margin + grown
    said.append(f"x {margin}" + ("" if here else f", measured on {k['gpu']}"))
    if growth:
        value += growth
        said.append(f"+ {growth:.2f} GiB of expert cache {_card_free(gpu)} GiB free grows "
                    f"beyond its run's {c['fields']['vram_budget']['value']} GiB")
    if lifted:
        value += lift
        said.append(f"+ {lift:.2f} GiB the {req.path} path holds on top")
    whose = f" ({k['model']}'s, the same shapes)" if k["model"] != req.model else ""
    loop = ", the sequential loop's," if lifted else ""
    return (value, " ".join([f"{c['id']}'s peak{whose}{loop}", *said]),
            c["fields"][name]["receipt"], lifted)


def _nf4_host_floor(cfg: "PorosConfig", entry: Optional[str],
                    s: Optional[_Sizes]) -> Optional[float]:
    """A floor of a dense NF4 CUDA run's host RAM (GiB, rounded down): blockwise residency keeps
    the frozen decoder layers in host RAM between their visits, 33/64 bytes a weight in NF4 with
    its double-quantized scales (the layers config.json sizes; the tables and buffers come on
    top). None where it does not apply: a MoE run (its stores size it), blocks kept on the GPU
    (resident_blocks), or no config.json sizes."""
    if entry != "dense" or s is None or s.experts_nf4:
        return None
    if getattr(cfg, "resident_blocks", 0):
        return None
    return math.floor(s.layers * (s.layer_bf16 // 2) * 33 / 64 / 2**30 * 100) / 100


def _unknown(unit: str, why: str) -> Figure:
    return Figure(value=None, unit=unit, source="unknown", why=why)
