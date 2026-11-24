"""The Mac table: what a run on a device class takes, per model. Torch-free.

The rows live in poros/catalog.toml. Every verb picks its row here, so fit, train, models, machine
and prove cannot disagree, and a plan reads the same settled config a run trains with.

A model the catalog does not list, of a family someone asserted (``--unvalidated``,
``POROS_USER_ASSERTED_ARCHS``), has a row derived from its config.json, a Hub id's or a folder's
alike: the lengths, tier and saves of the table's nearest model of its kind, and sizes by
``poros.metal.layout``'s arithmetic. Its evidence is ``none`` and it says it is derived: no run
measured any of it.

Every row in the catalog was measured on one class, ``REFERENCE``. Any other Mac's rows are derived
from those for its own memory (``replan``): the same lengths, steps and saves, plus longer lengths a
larger Mac holds, each from RAM when the pack and the rest of the run fit this Mac's GPU working set
less what macOS keeps (and the reference held it in RAM, or this Mac has more memory), else
from the SSD, and none that cannot fit even from the SSD. What a run holds is what one held on the
reference, where one ran, else the arithmetic with the margin measured runs needed. A Mac with the
reference's memory gets the reference's table. Its evidence is ``derived``: memory is arithmetic,
and speed unknown until a run on that Mac measures it.
"""

from __future__ import annotations

import dataclasses
import json
import platform
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple, Optional

from poros.cli.words.models import Entry, MetalRow, by_id, catalog, looks_like_hub_id
from poros.core.arch_registry import arch_status, registry_type
from poros.core.errors import PorosGuaranteeUnavailableError
from poros.core.schemas import PlanGpu
from poros.metal import (
    DEVICE_CLASSES,
    dev_memory_gb,
    device_class,
    measured,
    names_experts,
    require_class,
)

REFERENCE = DEVICE_CLASSES[0]  # the class every derived row comes from
REFERENCE_MEMORY = 16 * 2**30
# macOS and its services, on the reference class: a planning figure until a receipt replaces it.
# Another Mac keeps the same share of its memory, and never less than the floor.
OS_RESERVE_GB = 3.50
OS_RESERVE_FLOOR_GB = 2.50
WORKING_SET_BYTES = {REFERENCE: 12_713_115_648}  # recommended_max_memory, measured
# The lengths a derived row may take: the reference table's own. A longer one is still a
# --seq-len away, and runs if its memory fits.
LENGTHS = (64, 128, 256, 512)
GIB_GB = 2**30 / 1e9


class Size(NamedTuple):
    """Decimal GB, computed from each model's config.json. No figure here is ever shown as a
    measurement: disk space decides where a model goes, memory only refuses what cannot fit."""

    shards: float
    pack: float
    memory: float  # the least a run holds at any length, process included
    per_token: float
    adapter: float  # at rank 16, measured
    # One layer's packed bytes, for a model whose rows hold every layer: from the SSD a run holds
    # only the slot a layer is read into and the landing it is read ahead into.
    slab: float = 0.0


_SIZES = {
    "Qwen/Qwen2.5-0.5B": Size(0.99, 0.46, 1.72, 0.0019, 0.035, 0.0077),
    "Qwen/Qwen2.5-7B": Size(15.2, 5.58, 6.74, 0.00224, 0.162),
    "Qwen/Qwen2.5-32B": Size(65.5, 19.21, 4.61, 0.003125, 0.537),
    "Qwen/Qwen2.5-72B": Size(145.4, 41.18, 6.83, 0.00443, 0.842),
    "Qwen/Qwen3-0.6B": Size(1.5, 0.54, 1.84, 0.00194, 0.04, 0.0081),
    "Qwen/Qwen3-32B": Size(65.52, 19.21, 4.58, 0.00313, 0.537),
    "Qwen/Qwen3.5-4B": Size(9.32, 3.11, 5.01, 0.00331, 0.13, 0.0582),
    "Qwen/Qwen3.8-27B": Size(55.56, 17.65, 4.11, 0.00429, 0.467),
    "google/gemma-3-1b-pt": Size(2.0, 0.96, 2.34, 0.00327, 0.052, 0.0138),
    "google/gemma-3-27b-pt": Size(54.86, 16.03, 4.18, 0.00448, 0.454),
    "google/gemma-4-31B-it": Size(62.55, 17.93, 4.45, 0.00444, 0.49),
    "Qwen/Qwen3-30B-A3B-Instruct-2507": Size(61.07, 18.04, 6.88, 0.0026, 1.258),
    "Qwen/Qwen1.5-MoE-A2.7B": Size(28.63, 10.14, 3.43, 0.0025, 0.383),
    "allenai/OLMoE-1B-7B-0924": Size(13.84, 4.28, 2.56, 0.0009, 0.235),
    "deepseek-ai/DeepSeek-V2-Lite": Size(31.41, 10.05, 3.65, 0.0017, 0.443),
    "Qwen/Qwen3.6-35B-A3B": Size(71.9, 21.52, 7.68, 0.00314, 1.845),
    "google/gemma-4-26B-A4B-it": Size(51.61, 16.57, 5.14, 0.00331, 0.952),
}


def class_of(gpu_name: str) -> str:
    """``Apple M4 10-core GPU 16GB`` -> ``apple-m4-10gpu-16gb``."""
    return gpu_name.lower().replace("-core gpu", "gpu").replace(" ", "-")


def name_of(device_class_: str) -> str:
    """``apple-m4-10gpu-16gb`` -> ``Apple M4 10-core GPU 16GB``."""
    *chip, cores, memory = device_class_.split("-")
    return f"{' '.join(chip).title()} {cores.replace('gpu', '')}-core GPU {memory.upper()}"


def memory_of(device_class_: str) -> int:
    """A class's memory in bytes: its GB are GiB, as ``hw.memsize`` counts them. 0: unreadable."""
    found = re.search(r"-(\d{1,4})gb$", device_class_)
    return int(found.group(1)) * 2**30 if found else 0


def os_reserve_gb(memory: int) -> float:
    """What a plan sets aside for macOS on a Mac with this memory: 3.5 GB on the reference's
    16 GB, the same share on another, and never under 2.5 GB."""
    return max(OS_RESERVE_FLOOR_GB, OS_RESERVE_GB * memory / REFERENCE_MEMORY)


def estimated_working_set(memory: int) -> int:
    """What macOS was reported to let the GPU hold: two thirds of memory up to 36 GB (11453.25 MB
    on 16 GB M1, M2 and M3 in llama.cpp #9808 and #14527 and ggml #990), and three quarters above,
    as is commonly said and not checked here. The reference measured more than this. Used only
    where no figure is read: a Mac planned for from elsewhere, or one whose GPU did not answer."""
    return memory * 2 // 3 if memory <= 36 * 2**30 else memory * 3 // 4


def working_set(device_class_: str) -> tuple[int, str]:
    """(bytes the GPU may hold on this class, how that is known): ``measured`` on the reference,
    ``read`` from this Mac's GPU, ``budget`` under the developer's budget (the smaller of what a
    Mac with that memory is given and what this GPU has), or ``estimated``."""
    if device_class_ in WORKING_SET_BYTES:
        return WORKING_SET_BYTES[device_class_], "measured"
    memory = memory_of(device_class_)
    if platform.machine() == "arm64" and device_class_ == device_class():
        from poros.metal import machine, selfcheck

        live = machine.gpu_working_set() or selfcheck.recorded_working_set(device_class_)
        if live and dev_memory_gb() is not None:
            return min(live, estimated_working_set(memory)), "budget"
        if live:
            return live, "read"
    return estimated_working_set(memory), "estimated"


def plan_gpu(device_class_: str) -> PlanGpu:
    working = working_set(device_class_)[0] if memory_of(device_class_) else None
    return PlanGpu(
        name=name_of(device_class_),
        total_gib=float(re.search(r"(\d+)gb$", device_class_).group(1)),
        free_gib=None,
        source="spec",
        unified=True,
        working_set_gib=round(working / 2**30, 2) if working else None,
    )


def this_mac() -> Optional[PlanGpu]:
    return plan_gpu(device_class()) if platform.machine() == "arm64" else None


def _weights_gb(checkpoint: str) -> float:
    return sum(p.stat().st_size for p in Path(checkpoint).glob("*.safetensors")) / 1e9


def _same_family(arch: str) -> list[Entry]:
    return [e for e in catalog() if e.id in _SIZES
            and e.arch.removesuffix("_text") == arch.removesuffix("_text")]


def _as_catalog(checkpoint: str) -> Optional[Entry]:
    """What a local checkpoint of a family in this table trains as: the smallest model of its
    family here that holds its weights, else the largest. None: not a checkpoint, or no family in
    this table (a family someone asserted has a row of its own, ``derived``)."""
    folder = Path(checkpoint)
    try:
        arch = registry_type(json.loads((folder / "config.json").read_text())) or ""
    except (OSError, ValueError, AttributeError):
        return None
    same = sorted(_same_family(arch), key=lambda e: _SIZES[e.id].shards)
    weights = _weights_gb(checkpoint)
    return next((e for e in same if weights <= _SIZES[e.id].shards), same[-1] if same else None)


class Derived(NamedTuple):
    """A model's row from its config alone: ``entry`` holds its rows, ``size`` its arithmetic."""

    entry: Entry
    size: Size


def _config(model_id: str) -> Optional[dict]:
    """config.json of a folder, or of a Hub id: from the Hub cache, else as ``poros arch`` reads it,
    that file alone and with no token. A gated repo has none."""
    from poros.plan.configs import read_config

    folder = Path(model_id)
    if folder.is_dir():
        return read_config(folder / "config.json")
    if not looks_like_hub_id(model_id):
        return None
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_id, "config.json")
    except Exception:  # not a name the Hub takes
        return None
    if isinstance(cached, str):
        return read_config(Path(cached))
    from poros.cli.verbs.arch import _from_hub

    seen = _from_hub(model_id)
    return None if seen.gated else seen.raw


@lru_cache(maxsize=16)
def _layout(text: str) -> Any:
    """``poros.metal.layout``'s reading of a config, or None for one no class here builds, or whose
    experts the Mac's path would not find. Built once per config: the skeleton takes seconds."""
    from poros.cli.verbs.arch import _memory_held
    from poros.metal import layout

    raw = json.loads(text)
    if layout.can_build(raw):
        return None
    try:
        with _memory_held():
            found = layout.layout_of(raw)
    except Exception:  # a config no bound foresaw: no row, and train's reading says why
        return None
    return None if names_experts(raw) and not found.moe else found


def derived(model_id: str) -> Optional[Derived]:
    """The row of a model the catalog does not list, of a family someone asserted and this table
    has no model of: the lengths, tier and saves of the nearest model of its kind here (the
    smallest whose pack holds this one's, else the largest), its sizes from its config. A Hub id
    and a folder of the same model get the same row. None for anything else."""
    if by_id(model_id) is not None:
        return None
    raw = _config(model_id)
    arch = registry_type(raw) if raw else None
    if not isinstance(arch, str) or arch_status(arch) != "user_asserted" or _same_family(arch):
        return None
    try:
        found = _layout(json.dumps(raw, sort_keys=True))
    except (ValueError, RecursionError):
        return None
    if found is None:
        return None
    kind, pack = ("moe" if found.moe else "dense"), found.pack_bytes / 1e9
    same = sorted((e for e in catalog() if e.id in _SIZES and e.entry == kind and e.metal),
                  key=lambda e: _SIZES[e.id].pack)
    near = next((e for e in same if pack <= _SIZES[e.id].pack), same[-1])
    rows = tuple(dataclasses.replace(r, evidence="none", method="", receipt="", macos_build="",
                                     derived=True) for r in near.metal)
    memory = found.floor_gb(0, 16)
    if next(r for r in rows if r.default).tier == "ram":  # every layer held, embed and head too
        memory += (found.pack_bytes - found.one_layer_bytes) / 1e9
    size = Size(shards=found.weights_bytes / 1e9, pack=pack, memory=memory,
                per_token=found.per_token_gb, adapter=found.adapter_bytes_16 / 1e9,
                slab=found.one_layer_bytes / 1e9)
    return Derived(Entry(id=model_id, arch=arch, family="", size="", variant="", entry=kind,
                         evidence="none", metal=rows), size)


def _entry(model_id: str) -> Optional[Entry]:
    found = by_id(model_id) or _as_catalog(model_id)
    if found is None and (made := derived(model_id)) is not None:
        found = made.entry
    return found


def _own_rows(model_id: str, device_class_: str) -> tuple[MetalRow, ...]:
    """A model's rows on a class as the catalog (or its config) has them; a same-shape model's are
    its base's."""
    entry = _entry(model_id)
    if entry is None:
        return ()
    own = entry.metal or (by_id(entry.via).metal if entry.via else ())
    return tuple(r for r in own if r.device_class == device_class_)


def rows(model_id: str, device_class_: str) -> tuple[MetalRow, ...]:
    """A model's rows on this class: a measured class's own, any other Mac's derived from the
    reference's for its memory (``replan``)."""
    if measured(device_class_):
        return _own_rows(model_id, device_class_)
    return replan(model_id, device_class_)


# -- Another Mac: the reference's rows, planned again for its memory --------------------------


class Budget(NamedTuple):
    """What a class has for a run, in decimal GB but ``gb``, the class's own: ``pool`` is its
    memory less what macOS keeps, ``ram`` what may hold every layer (the GPU's working set less
    the same), ``gpu`` the GPU's working set, known as ``gpu_how`` says."""

    gb: int
    reserve: float
    pool: float
    ram: float
    gpu: float
    gpu_how: str


def budget(device_class_: str) -> Budget:
    memory = memory_of(device_class_)
    reserve = os_reserve_gb(memory)
    gpu, how = working_set(device_class_)
    return Budget(memory >> 30, reserve, memory / 1e9 - reserve, gpu / 1e9 - reserve, gpu / 1e9,
                  how)


def _basis(model_id: str) -> str:
    """The catalog id whose runs a model's memory comes from: its own, its base's or what a local
    checkpoint trains as. A model only its config describes has none, so it is itself."""
    entry = by_id(model_id) or _as_catalog(model_id)
    return entry.id if entry else model_id


def _cells(model_id: str, tier: str) -> list[tuple[float, Optional[float], int]]:
    """(peak GB, of it on the GPU in GB, length) of each run of this model from this tier on the
    reference, at the table's rank: its own runs, else its base's."""
    from poros.plan.cells import MAC_CELLS

    name, basis = name_of(REFERENCE), _basis(model_id)
    entry = by_id(basis)
    for model in dict.fromkeys((basis, entry.via if entry else "")):
        found = []
        for c in MAC_CELLS:
            key, fields = c["key"], c["fields"]
            if (key["gpu"] == name and key["model"] == model and key.get("lora_rank") == 16
                    and key.get("load_strategy") == f"metal_{tier}"
                    and "peak_vram_allocated" in fields
                    # a recipe's certification run: it answers its own Mac's fit, and sizes
                    # no other Mac's rows
                    and not c.get("only_its_machine")):
                visible = fields.get("gpu_visible", {}).get("value")
                found.append((fields["peak_vram_allocated"]["value"] * GIB_GB,
                              visible * GIB_GB if visible is not None else None,
                              key["seq_len"]))
        if found:
            return found
    return []


def _reference_tier(model_id: str) -> str:
    return next((r.tier for r in _own_rows(model_id, REFERENCE) if r.default), "ssd")


def _measured_held(model_id: str) -> Optional[float]:
    """What runs of this model held on the reference before their tokens, the most. None: none."""
    size = _size(model_id)
    cells = _cells(model_id, _reference_tier(model_id))
    return max(peak - size.per_token * seq for peak, _, seq in cells) if cells else None


def margin(kind: str) -> float:
    """How far past their arithmetic floor runs on the reference held, for models of this kind
    (``dense`` or ``moe``): the most of any. What a model no run measured is taken to need past
    its own floor. 1 when no run of that kind measured anything."""
    seen = [found / _SIZES[e.id].memory for e in catalog()
            if e.entry == kind and e.id in _SIZES and (found := _measured_held(e.id)) is not None]
    return max([1.0, *seen])


def held(model_id: str, tier: str) -> tuple[float, Optional[float], bool]:
    """(GB a run of this model holds from this tier before its tokens, process included; of it on
    the GPU, where a run measured it; whether a run on the reference measured it). From the tier the
    reference ran, its runs' peaks less their tokens, the largest; with none, the arithmetic floor
    times ``margin``. From RAM where the reference read the SSD, its whole pack more; from the SSD
    where the reference held every layer, the pack less the slot and landing a layer is read
    through."""
    size, ref = _size(model_id), _reference_tier(model_id)
    cells = _cells(model_id, ref)
    if cells:
        whole = max(peak - size.per_token * seq for peak, _, seq in cells)
        seen = [gpu - size.per_token * seq for _, gpu, seq in cells if gpu is not None]
        gpu = max(seen) if seen else None
    else:
        entry = _entry(model_id)
        whole, gpu = size.memory * margin(entry.entry if entry else "dense"), None
    if tier == ref:
        return whole, gpu, bool(cells)
    if tier == "ram":
        return whole + size.pack, None if gpu is None else gpu + size.pack, bool(cells)
    return whole - size.pack + 2 * size.slab, gpu, bool(cells)


def need_gb(model_id: str, seq: int, tier: str) -> float:
    """What a run of this model at this length holds from this tier, process included."""
    return held(model_id, tier)[0] + _size(model_id).per_token * seq


def gpu_gb(model_id: str, seq: int, tier: str) -> Optional[float]:
    gpu = held(model_id, tier)[1]
    return None if gpu is None else gpu + _size(model_id).per_token * seq


def _tier_at(model_id: str, seq: int, room: Budget, was: str) -> Optional[str]:
    """RAM when every layer and the rest of the run fit the GPU's working set less what macOS
    keeps, and either the reference held it in RAM (``was``) or this Mac has more memory than the
    reference: the reference's runs chose the SSD where RAM swapped. Else the SSD when what
    that holds fits this Mac's memory less the same, and its GPU part the working set; else None."""
    more = room.gb > REFERENCE_MEMORY >> 30  # memory, not a GPU share macOS may report higher
    if need_gb(model_id, seq, "ram") <= room.ram and (was == "ram" or more):
        return "ram"
    gpu = gpu_gb(model_id, seq, "ssd")
    if need_gb(model_id, seq, "ssd") <= room.pool and (gpu is None or gpu <= room.gpu):
        return "ssd"
    return None


def replan(model_id: str, device_class_: str) -> tuple[MetalRow, ...]:
    """This model's rows on a class no run measured, from the reference's: its lengths, and on a
    Mac with more memory the table's longer ones it holds, each from RAM or the SSD as ``_tier_at``
    says. A length the reference left out stays out on a Mac with no more memory than it. With none
    of those, the table's shorter lengths, the longest that fits. Steps, accumulation and saves are
    the reference row's of the nearest length. The default is the reference's, or the longest
    below it here, or the shortest."""
    reference = sorted(_own_rows(model_id, REFERENCE), key=lambda r: r.seq)
    memory = memory_of(device_class_)
    if not reference or not memory:
        return ()
    room = budget(device_class_)
    top = reference[-1].seq
    longer = {s for s in LENGTHS if s > top} if memory > REFERENCE_MEMORY else set()

    def near(seq: int) -> MetalRow:
        return min(reference, key=lambda r: (abs(r.seq - seq), r.seq))

    tiers = {seq: tier for seq in sorted({r.seq for r in reference} | longer)
             if (tier := _tier_at(model_id, seq, room, near(seq).tier))}
    if not tiers:
        shorter = [s for s in reversed(LENGTHS) if s < reference[0].seq]
        tiers = next(({s: t} for s in shorter
                      if (t := _tier_at(model_id, s, room, near(s).tier))), {})
    if not tiers:
        return ()
    default = next(r.seq for r in reference if r.default)
    picked = (default if default in tiers
              else max((s for s in tiers if s < default), default=min(tiers)))
    out = []
    for seq, tier in tiers.items():
        out.append(dataclasses.replace(
            near(seq), device_class=device_class_, seq=seq, tier=tier, default=seq == picked,
            evidence="derived", method="", receipt="", macos_build="", planned_from=REFERENCE))
    return tuple(out)


def refusal(model_id: str, device_class_: str) -> str:
    """Why a model the reference trains has no row on this class, in a sentence: even from the SSD
    it cannot fit. "" when it has one, or the reference does not train it either."""
    if measured(device_class_) or not memory_of(device_class_) or rows(model_id, device_class_):
        return ""  # memory unread: require_class says so
    reference = sorted(_own_rows(model_id, REFERENCE), key=lambda r: r.seq)
    if not reference:
        return ""
    room = budget(device_class_)
    seq = min(LENGTHS[0], reference[0].seq)
    need, gpu = need_gb(model_id, seq, "ssd"), gpu_gb(model_id, seq, "ssd")
    how = ("by what runs of it held on an M4 with 16 GB" if held(model_id, "ssd")[2]
           else "by arithmetic on its shapes")
    if need > room.pool:
        part = (f"a run of it at {seq} tokens holds about {need:.1f} GB, {how}, and this Mac has "
                f"{room.pool:.1f} GB once macOS has its {room.reserve:.1f} GB")
    else:
        part = (f"at {seq} tokens about {gpu:.1f} GB of it is on the GPU, {how}, past the "
                f"{room.gpu:.1f} GB macOS lets this Mac's GPU hold")
    return (f"{model_id} is too large for this Mac's {room.gb} GB, even reading its weights "
            f"from the SSD: {part}.")


def pick(
    model_id: str, device_class_: str, *, seq: Optional[int] = None, tokens: Optional[int] = None
) -> Optional[MetalRow]:
    """The row for a data scan's ``tokens`` (the smallest that holds them, else the largest), a
    set ``seq`` (its own row, else the default), or neither (the default). None: no row here."""
    table = sorted(rows(model_id, device_class_), key=lambda r: r.seq)
    if not table:
        return None
    if tokens is not None:
        return next((r for r in table if r.seq >= tokens), table[-1])
    return next((r for r in table if r.seq == seq), None) or next(r for r in table if r.default)


def defaults(row: MetalRow) -> dict[str, Any]:
    """What a row sets in a config that leaves it unset."""
    out = {
        "max_seq_length": row.seq,
        "gradient_accumulation_steps": row.grad_accum,
        "save_steps": row.save_steps,
        "store": row.tier,
        "block_size": 1,
    }
    if row.tier == "ssd":  # a held-out row costs a third of a step, and a run scores both ends
        out["eval_split"] = 0.0
    return out


def fill(cfg: Any, device_class_: str) -> Optional[MetalRow]:
    """Set in ``cfg`` what it left unset from its row, which is returned. A set value stays: the
    floor below is the only thing that refuses it. None, and nothing changes: no row here."""
    given = cfg.model_fields_set
    row = pick(
        cfg.model_name_or_path,
        device_class_,
        seq=cfg.max_seq_length if "max_seq_length" in given else None,
    )
    if row is None:
        return None
    for name, value in defaults(row).items():
        if name not in given:
            setattr(cfg, name, value)
    return row


def require_row(model_id: str, device_class_: str) -> None:
    """Refuse a model this Mac does not train, or cannot fit, or a Mac whose class is unreadable."""
    require_class(device_class_)
    if rows(model_id, device_class_):
        return
    if why := refusal(model_id, device_class_):
        raise PorosGuaranteeUnavailableError(
            why, remediation="`poros models` lists what this Mac trains.")
    raise PorosGuaranteeUnavailableError(
        f"{model_id} does not train on a Mac in this release.",
        remediation="`poros models` lists what does.",
    )


def evidence(model_id: str, device_class_: str) -> str:
    """What this class holds for a model: the row's value, ``derived`` for a class no run measured,
    ``same-shape``, or ``absent``."""
    row = pick(model_id, device_class_)
    if row is None:
        return "absent"
    if row.planned_from:
        return row.evidence
    return "same-shape" if _entry(model_id).via else row.evidence


def _size(model_id: str) -> Size:
    entry = by_id(model_id) or _as_catalog(model_id)
    return _SIZES[entry.via or entry.id] if entry else derived(model_id).size


def _local(model_id: str) -> bool:
    return by_id(model_id) is None and Path(model_id).is_dir()


def floor_gb(model_id: str, seq: int) -> float:
    """The least a run holds at this length on the reference, by arithmetic: what refuses there."""
    size = _size(model_id)
    return size.memory + size.per_token * seq


def disk_gb(model_id: str) -> float:
    """Shards plus pack. A local checkpoint's shards are already where they are."""
    return pack_gb(model_id) + (0.0 if _local(model_id) else _size(model_id).shards)


def pack_gb(model_id: str) -> float:
    """A local checkpoint can be far smaller than what it trains as, and packing only shrinks it."""
    pack = _size(model_id).pack
    return min(pack, _weights_gb(model_id)) if _local(model_id) else pack


def run_gb(model_id: str, rank: int, keep: int) -> float:
    """The adapter and Adam's two moments, in a run's folder and each checkpoint it keeps."""
    return 3 * _size(model_id).adapter * rank / 16 * (keep + 1)
