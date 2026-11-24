"""Where a Mac run keeps its weights: the volume, and the Hub cache on it. Torch-free.

A store is laid out like HF_HOME: ``<store>/hub``, with each model's pack beside it. A volume
qualifies if it is a writable local APFS disk that keeps 10% free after what the model still lacks
there; bytes already on it need no room. A model already on a volume stays there. Otherwise the
store this machine remembers (``store`` in preferences.toml) is used, then the one volume that
qualifies; several are asked about once, and none is ``E_DISK``. A Hub cache the user set up is
theirs.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Union

from poros import runs
from poros.core.errors import PorosConfigError, PorosDiskError
from poros.metal import run_tool
from poros.metal.table import disk_gb, pack_gb

KEEP_FREE = 0.10


def nearest_existing(path: Path) -> Path:
    """The deepest part of ``path`` that exists: what a disk question can be asked of."""
    return next(p for p in (path, *path.parents) if p.exists())


def _same_disk(a: Path, b: Path) -> bool:
    return os.stat(nearest_existing(a)).st_dev == os.stat(nearest_existing(b)).st_dev


def internal_store() -> Path:
    return runs.home() / "store"


@dataclass(frozen=True)
class Volume:
    path: Path
    total: int
    free: int

    @property
    def label(self) -> str:
        return "internal" if self.path == Path.home() else self.path.name

    @property
    def store(self) -> Path:
        return internal_store() if self.path == Path.home() else self.path / "poros"

    def qualifies(self, need_gb: float) -> bool:
        return need_gb <= 0 or self.free - need_gb * 1e9 >= KEEP_FREE * self.total

    def __str__(self) -> str:
        return f"{self.label} {self.free / 1e9:.0f} GB free"


def _mounted() -> list[Path]:
    """The disks macOS shows: local APFS, neither read-only nor hidden. That leaves out the sealed
    system and the internal data volume, which home stands for."""
    points = []
    for line in run_tool("/sbin/mount").splitlines():
        hit = re.fullmatch(r"/dev/\S+ on (/Volumes/[^/]+) \((.+)\)", line)
        if hit and {"apfs", "local"} <= (options := set(hit.group(2).split(", "))):
            if not {"read-only", "nobrowse"} & options:
                points.append(Path(hit.group(1)))
    return points


def volumes() -> list[Volume]:
    """The internal disk, then every other one that can be written to."""
    seen, out = set(), []
    for path in (Path.home(), *_mounted()):
        device = os.stat(path).st_dev
        if device not in seen and os.access(path, os.W_OK):
            seen.add(device)
            usage = shutil.disk_usage(path)
            out.append(Volume(path, usage.total, usage.free))
    return out


@lru_cache(maxsize=1)
def _first_hub() -> tuple[str, bool]:
    """(the Hub cache this process started with, whether the user chose it), read once: ``use``
    below rewrites what it comes from."""
    from huggingface_hub import constants

    return constants.HF_HUB_CACHE, "HF_HUB_CACHE" in os.environ or "HF_HOME" in os.environ


def hub_of(store: Union[str, Path]) -> str:
    first, chosen = _first_hub()
    return first if chosen or Path(store) == internal_store() else str(Path(store) / "hub")


def use(store: Union[str, Path]) -> None:
    """Point this process, and any child, at this store's Hub cache."""
    from huggingface_hub import constants

    hub = hub_of(store)
    os.environ["HF_HUB_CACHE"] = constants.HF_HUB_CACHE = hub


def _remembered() -> Optional[Path]:
    """The store this machine remembers, with or without a terminal: it is not a question."""
    from poros.cli.questions import setting

    found = setting("store")
    return Path(found) if found else None


def _stores() -> list[Path]:
    """Every store a model of this Mac may be in, the remembered one first."""
    mine = _remembered()
    every = dict.fromkeys([*([mine] if mine else []), *(v.store for v in volumes())])
    return list(every)


def hubs() -> list[str]:
    """Every Hub cache a model of this Mac may be in."""
    return list(dict.fromkeys(map(hub_of, _stores())))


def _shards(store: Path, model_id: str) -> tuple[list[Path], bool]:
    """(this model's weight files in this store, whether that is all of them). A local checkpoint's
    stay where they are: in the store on their own volume, with nothing to fetch for any other."""
    from huggingface_hub import try_to_load_from_cache

    own = Path(model_id)
    if own.is_dir():
        return [*own.glob("*.safetensors")] if _same_disk(store, own) else [], True

    def cached(name: str) -> Optional[str]:
        found = try_to_load_from_cache(model_id, name, cache_dir=hub_of(store))
        return found if isinstance(found, str) else None

    if one := cached("model.safetensors"):
        return [Path(one)], True
    index = cached("model.safetensors.index.json")
    if not index:
        return [], False
    names = set(json.loads(Path(index).read_text())["weight_map"].values())
    found = [Path(path) for path in map(cached, names) if path]
    return found, len(found) == len(names)


def _holds(store: Path, model_id: str) -> bool:
    """Weights of this model are here, some or all, or its pack: a resumed download stays where it
    began. A config.json alone is only someone having asked about the model."""
    return bool(_shards(store, model_id)[0]) or built(store, model_id)


def ready(model_id: str) -> bool:
    """This Mac holds what a run of this model needs, as `train` reads it: in some store, its
    finished pack, or every weight file in that store's Hub cache. Nothing is fetched to say so."""
    return any(holds(store, model_id) for store in _stores())


def holds(store: Union[str, Path], model_id: str) -> bool:
    """This one store holds what a run of this model needs: its finished pack, or every weight
    file in its Hub cache. Nothing is fetched to say so."""
    store = Path(store)
    try:
        found, whole = _shards(store, model_id)
    except (OSError, ValueError, KeyError):  # an index torn or not JSON: not whole
        found, whole = [], False
    return built(store, model_id) or bool(found and whole)


def built(store: Path, model_id: str) -> bool:
    """A finished pack: its index is what a pack commits last."""
    slug = Path(model_id).name if Path(model_id).is_dir() else model_id.replace("/", "--")
    return any(store.glob(f"{slug}-*/model.sss.idx"))


def missing_gb(store: Path, model_id: str) -> float:
    """What this model would still take in this store: nothing once it is packed."""
    if built(store, model_id):
        return 0.0
    found, whole = _shards(store, model_id)
    shards = disk_gb(model_id) - pack_gb(model_id)
    have = shards if whole else sum(path.stat().st_size for path in found) / 1e9
    return max(0.0, shards - have) + pack_gb(model_id)


def qualifying(model_id: str) -> list[Volume]:
    """The volumes that can take this model, most free space first."""
    fit = (v for v in volumes() if v.qualifies(missing_gb(v.store, model_id)))
    return sorted(fit, key=lambda v: -v.free)


def _no_room(need: float, model_id: str, where: str) -> PorosDiskError:
    return PorosDiskError(
        f"{model_id} needs {need:.1f} GB of disk and 10% left free: {where}.",
        remediation="Free some space or attach a disk, or name one with --store.",
    )


def _checked(store: Path, model_id: str) -> Path:
    """A store named outright: its volume must be mounted and have the room."""
    root = Path(*store.parts[:3])
    if len(store.parts) > 3 and not root.exists():
        raise PorosDiskError(f"{root} is not mounted.", remediation="Attach it and run this again.")
    need = missing_gb(store, model_id)
    existing = nearest_existing(store)
    usage = shutil.disk_usage(existing)
    volume = Volume(existing, usage.total, usage.free)
    if not volume.qualifies(need):
        raise _no_room(need, model_id, f"{store} has {usage.free / 1e9:.0f} GB free")
    return store


def pick(model_id: str, given: Optional[str] = None) -> Optional[Path]:
    """The store for this model: named, or the user's own Hub cache's parent, or the store holding
    the model, or the one remembered from before, or the volume that qualifies. None: ask."""
    first, chosen = _first_hub()
    if given or chosen:
        return _checked(Path(given or Path(first).parent), model_id)
    held = next((s for s in _stores() if _holds(s, model_id)), None)
    if held:
        return _checked(held, model_id)
    mine = _remembered()
    if mine and Path(*mine.parts[:3]).exists():  # its disk gone: a new question
        return _checked(mine, model_id)
    fit = qualifying(model_id)
    if not fit:
        raise _no_room(disk_gb(model_id), model_id, ", ".join(map(str, volumes())))
    return fit[0].store if len(fit) == 1 else None


def run_files(store: Union[str, Path]) -> Optional[Path]:
    """Where a run keeps its files if not in its own folder: beside its model when that is off the
    internal disk, since a 72B run is 5 GB. POROS_HOME, set, says otherwise."""
    store = Path(store)
    internal = _same_disk(store, Path.home())
    return None if internal or os.environ.get("POROS_HOME") else store / "runs"


def settle(config: Any) -> None:
    """Decide ``config.store_path`` for a run with no one to ask, and use it."""
    store = pick(config.model_name_or_path, config.store_path)
    if store is None:
        names = ", ".join(str(v.store) for v in qualifying(config.model_name_or_path))
        raise PorosConfigError(
            f"More than one disk can hold {config.model_name_or_path}: {names}.",
            remediation="Set store_path to one of them.",
        )
    config.store_path = str(store)
    use(store)


def survey(model_id: str, given: Optional[str] = None) -> tuple[Optional[Path], str]:
    """(the store if settled, where the weights go in a line). Never asks or raises: a plan only
    reports, and a read-only answer does not stop at a full disk."""
    need = f"{disk_gb(model_id):.1f} GB of disk"
    try:
        store = pick(model_id, given)
    except PorosDiskError:
        every = ", ".join(map(str, volumes()))
        return None, f"needs {need} and 10% left free; no disk has room ({every})"
    if store is None:
        every = ", ".join(map(str, qualifying(model_id)))
        return None, f"needs {need}; train asks which: {every}"
    if _holds(store, model_id):
        more = "" if built(store, model_id) else f", and {pack_gb(model_id):.1f} GB to prepare it"
        where = model_id if Path(model_id).is_dir() else hub_of(store)
        return store, f"the model is in {where}{more}"
    return store, f"needs {need} at {store}"
