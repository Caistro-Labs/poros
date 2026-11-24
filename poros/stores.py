"""Expert-store plugins: MoE expert stores provided by installed packages.

Poros ships two expert stores: ``ram`` (pinned host RAM, the default) and ``ssd`` (a ``.sss``
file streamed from a local SSD, ``poros.sss``). Any other store comes from a package that
registers it in the ``poros.stores`` entry-point group::

    # the plugin package's pyproject.toml
    [project.entry-points."poros.stores"]
    mystore = "my_package.poros_plugin:STORE"

The entry point resolves to a :class:`StorePlugin`. Users then select it by name everywhere
a store is chosen: ``load_moe_streamed(..., store="mystore", store_path=...)``,
``PorosTrainer.from_pretrained(..., store="mystore")``, ``store: mystore`` in the YAML config
and ``poros train --store mystore[:/path]``. A plugin's store must keep the frozen
``poros.moe.contracts.ExpertStore`` contract (immutable slabs, pure byte-copy
``load_slice``), so the streamed == resident guarantee is the plugin's to prove with
``poros.moe.gates`` on its own bytes.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

__all__ = [
    "BUILTIN_STORES",
    "ENTRY_POINT_GROUP",
    "PorosStoreNotFoundError",
    "StorePlugin",
    "available_stores",
    "get_store_plugin",
    "is_builtin",
    "moved_store_note",
    "recorded_store",
]

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "poros.stores"
BUILTIN_STORES = ("ram", "ssd")


class PorosStoreNotFoundError(ImportError):
    """A non-built-in store was requested and no installed package provides it."""


@dataclass(frozen=True)
class StorePlugin:
    """What a store plugin provides.

    ``load_model(model_name_or_path, dtype, *, store_path, options, trust_remote_code,
    attn_implementation, revision)`` returns the model on CPU, ready for
    ``poros.moe.prepare_moe``: fused expert parameters still on ``meta`` and the store on
    ``model._poros_moe_store`` (exactly what ``poros.moe.load.load_moe_streamed`` returns).
    ``backends(store)``, if set, returns the ``poros.moe.prepare.MoeBackends`` to prepare it
    with (e.g. a store-specific managed block); ``None`` uses the default backends.
    """

    name: str
    load_model: Callable[..., Any]
    backends: Optional[Callable[[Any], Any]] = None
    description: str = ""
    extras: Mapping[str, Any] = field(default_factory=dict)


def is_builtin(name: str) -> bool:
    return name in BUILTIN_STORES


def default_store_dir() -> Path:
    """Where ``store: ssd`` keeps a model's ``.sss`` when no one names a place: ``~/.poros/sss``
    (``POROS_HOME/sss``), as ``poros train --store ssd`` (``poros.cli.verbs.train.sss_dir``)."""
    import os

    return Path(os.environ.get("POROS_HOME") or Path.home() / ".poros") / "sss"


def moved_store_note(recorded: Any, default: Any) -> str:
    """The one line said when a run folder's recorded store is not where the store is built."""
    return (f"No store of this model to reuse at the recorded store_path {recorded}: the store "
            f"is built at the default place, {default}.")


def _a_store_of(path: str, model: str) -> bool:
    """A committed ``.sss`` of ``model``'s experts at ``path`` (a directory, or the file):
    its index and header read, its meta names the model. Read only; nothing is created."""
    try:
        from poros.moe.load import sss_file_path
        from poros.sss import SssFile, SssFormatError
    except ImportError:  # no torch: no store is read here
        return False
    try:
        with SssFile.open(sss_file_path(path, model)) as f:
            return (f.meta.get("kind") == "poros.moe.expert_slabs"
                    and f.meta.get("model_id") == model)
    except (SssFormatError, OSError):
        return False


def recorded_store(fields: dict, given: Collection[str] = (),
                   say: Optional[Callable[[str], None]] = None) -> tuple[dict, Optional[str]]:
    """A config's ``fields`` read from a run folder's records, with the store settings the folder
    may give, and where the SSD store is built if the recorded one cannot be reused.

    A folder can come from anyone: its ``store_path`` is taken only to READ a store of this model
    already there, never to create one. With none there, ``store: ssd`` builds at
    ``default_store_dir()`` (still on the SSD, said in one line), a Mac picks its store as a new
    run does, and ``store: ram`` reads none. A plugin store's ``store_path`` and
    ``store_options`` are the plugin's to read, and say nothing Poros can check: not taken.
    ``given`` names the fields the caller set: those are the caller's and stand as they are.

    Returns ``(fields, build_path)``. ``build_path`` is set where the recorded SSD store is kept
    to read: the load builds there instead if that store turns out not to be reusable (another
    revision, version or dtype), so the recorded place is never written."""
    import sys

    out = dict(fields)
    store, path = out.get("store", "ram"), out.get("store_path")
    model = str(out.get("model_name_or_path") or "")
    say = say or logger.warning
    if not is_builtin(store):
        dropped = [n for n in ("store_path", "store_options") if n not in given and out.get(n)]
        for name in dropped:
            out.pop(name)
        if dropped:
            say(f"store {store}: the recorded {' and '.join(dropped)} are not taken from a run's "
                "folder; pass them yourself to use them.")
        return out, None
    if "store_path" in given or not path:
        return out, None
    if sys.platform == "darwin":
        from poros.metal.store import holds

        if holds(path, model):
            return out, None
        out.pop("store_path")
        if "store" not in given:
            out.pop("store", None)  # the Mac table's, as for a new run
        say(f"No store of this model at the recorded store_path {path}: this Mac picks its "
            "store as a new run does.")
        return out, None
    if _a_store_of(str(path), model):
        return out, (str(default_store_dir()) if store == "ssd" else None)
    if store == "ssd":
        out["store_path"] = str(default_store_dir())
        say(moved_store_note(path, out["store_path"]))
    else:  # ram: a store there is only read, and there is none
        out.pop("store_path")
    return out, None


def _entry_points() -> dict:
    from importlib.metadata import entry_points

    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


def available_stores() -> dict[str, str]:
    """``{store name: providing distribution}`` for the built-in and installed stores."""
    out = {name: "poros-train (built in)" for name in BUILTIN_STORES}
    for name, ep in _entry_points().items():
        dist = getattr(ep, "dist", None)
        out.setdefault(name, getattr(dist, "name", None) or ep.value.split(":")[0])
    return out


def check_store_name(name: str, package: Optional[str] = None) -> str:
    """Raise unless ``name`` is built in or registered by an installed package (no import).

    ``package`` (the config's ``store_package``) names the distribution that provides the
    store, so the error can say exactly what to install."""
    if is_builtin(name) or name in _entry_points():
        return name
    raise PorosStoreNotFoundError(_missing(name, package))


def _missing(name: str, package: Optional[str] = None) -> str:
    others = sorted(set(available_stores()) - set(BUILTIN_STORES))
    have = f" Installed store plugins: {', '.join(others)}." if others else (
        " No store plugins are installed.")
    how = (f"Install {package!r}, which provides it, then retry." if package else
           f"Install the package that provides the {name!r} store (its documentation names "
           "it; set store_package to have this message name it), then retry.")
    return (
        f"store {name!r} is not built in ({', '.join(BUILTIN_STORES)}) and no installed "
        f"package registers it in the {ENTRY_POINT_GROUP!r} entry-point group.{have} {how}"
    )


def get_store_plugin(name: str, package: Optional[str] = None) -> StorePlugin:
    """Load the plugin registered for ``name``; a clear error names what is missing."""
    eps = _entry_points()
    if name not in eps:
        raise PorosStoreNotFoundError(_missing(name, package))
    ep = eps[name]
    dist = getattr(getattr(ep, "dist", None), "name", None) or ep.value.split(":")[0]
    try:
        plugin = ep.load()
    except ImportError as exc:
        raise PorosStoreNotFoundError(
            f"store {name!r} is registered by the package {dist!r}, but it failed to import "
            f"({exc}). Reinstall {dist!r} with its dependencies."
        ) from exc
    if callable(plugin) and not isinstance(plugin, StorePlugin):
        plugin = plugin()
    if not isinstance(plugin, StorePlugin) or plugin.name != name:
        raise TypeError(
            f"the {ENTRY_POINT_GROUP!r} entry point {name!r} of {dist!r} must resolve to a "
            f"poros.stores.StorePlugin named {name!r}, got {plugin!r}"
        )
    return plugin
