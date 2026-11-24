"""Which family a model is, and whether this release runs it: before any weights download.

* ``named_model``: the model a command's word names (a folder, a catalog name, a Hub id): one
  rule for ``check``, ``arch``, ``plan`` and the verbs.
* ``resolve_model_type``: a model's ``model_type`` from its config.json alone (a local file or
  directory, or the Hub's config.json), with why it could not be read. ``cheap_model_type`` is
  the same answer, None on any doubt (the trainer's pre-load gate falls through to the post-load
  one).
* ``support_of``: the support level and the entry (``dense``, ``moe``) a run takes.
* ``no_flag_trains``: why no flag trains a family off a Mac (``poros train`` refuses it,
  ``--unvalidated`` too; ``poros check`` and ``poros agents arch`` say so, ``E_ARCH_UNSUPPORTED``):
  a family or variant the registry marks unsupported, or a mixture of experts (by its config, or
  by its registry entry) whose family is not validated on ``poros.moe``. A dense target
  (gemma2) trains with ``--unvalidated``, as any dense family does.
* ``arch_refusal``: the architecture gate's refusal of a ``model_type`` routed as a run routes
  it (a validated MoE family trains on ``poros.moe``; any other through
  ``poros.core.arch_registry.blockwise_arch_check``), ``E_ARCH_UNSUPPORTED``. ``poros train``
  says it in its own words before the download; PorosTrainer raises it.

Torch-free.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from poros.core.errors import PorosError


def support_of(model_type: Optional[str]) -> tuple[str, Optional[str]]:
    """(support level, entry) exactly as a run treats the model."""
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


def cheap_model_type(model_name_or_path: str,
                      revision: Optional[str] = None) -> Optional[str]:
    """model_type from config.json alone -- no torch weights, no big pull.

    Local dir: read its config.json. Hub id: download just config.json
    (cached by huggingface_hub). Returns None whenever the answer is not
    definitive; callers must fall back to the post-load gate, never block.
    """
    from poros.core.arch_registry import registry_type
    from poros.plan.configs import read_model_config

    try:
        return registry_type(read_model_config(model_name_or_path, revision))
    except Exception:
        return None


def plain_type(model_type: Any) -> str:
    """A config's model_type is anyone's text, and this name reaches an agent: plain, or unknown."""
    plain = isinstance(model_type, str) and re.fullmatch(r"[a-z0-9_-]{1,64}", model_type)
    return model_type if plain else "unknown"


# A catalog model whose repository asks for a license: said wherever the catalog answers for it.
GATED_NOTE = ("its repository is gated: accept its license on huggingface.co and log in (hf auth "
              "login) before a download")


def named_model(name: str) -> str:
    """The model a command's word names, for every command that takes one: a folder or a .json
    here as it is; a catalog model's Hub id from its id or its short name, any case
    (``qwen2.5-7b-instruct``); anything else as given (a Hub id)."""
    if Path(name).exists():
        return name
    row = _catalog_row(name)
    return row["id"] if row else name


def _catalog_row(name: str) -> Optional[dict]:
    from poros import catalog

    low = name.lower()
    return next((m for m in catalog.raw()["model"]
                 if low in (m["id"].lower(), m["id"].split("/", 1)[-1].lower())), None)


def resolve_model_type(model: str) -> tuple[Optional[str], str]:
    """Resolve a HF model_type from a local path or hub id, WITHOUT torch.

    Reads config.json directly: a local file, a local directory, or a
    downloaded hub config. Returns ``(model_type, reason)`` -- on failure
    model_type is None and ``reason`` says specifically why (gated repo,
    unknown id, network, unreadable config), so the caller never has to
    guess. The catalog answers first for a model it names (``named_model``): its type is the
    catalog's record of its config.json, the same online and offline, with ``reason`` saying
    where its repository is gated; the Hub is asked for any other id.
    """
    from poros.core.arch_registry import bounded, registry_type
    from poros.plan.configs import ConfigTooLarge, fetch_config, read_config

    too_large = "unknown", "its config asks for more memory than a model can hold."
    model = named_model(model)
    listed = None if Path(model).exists() else _catalog_row(model)
    if listed is not None:  # the catalog answers first: the same online, offline, gated or not
        return listed["arch"], GATED_NOTE if listed.get("gated") else ""

    def no_hub() -> tuple[Optional[str], str]:
        return None, _no_hub(model)

    p = Path(model)
    cfg_path: Optional[Path] = None
    if p.is_file() and p.suffix == ".json":
        cfg_path = p
    elif p.is_dir() and (p / "config.json").is_file():
        cfg_path = p / "config.json"
    elif p.exists():
        # Never route an existing local path to the Hub: a "not found on the
        # Hub" answer for a file sitting on disk is maximally misleading.
        return None, (
            f"local path {model} exists but is not a model directory "
            "(with config.json) or a .json config file."
        )

    if cfg_path is not None:
        raw = read_config(cfg_path)
        if raw is None:
            return None, f"could not parse {cfg_path} as JSON."
        try:
            bounded(raw)
        except ValueError:
            return too_large
        mt = registry_type(raw)
        if mt is None:
            return None, f"{cfg_path} has no 'model_type' field."
        return plain_type(mt), ""

    # Hub id: fetch just config.json.
    try:
        from huggingface_hub.utils import (
            EntryNotFoundError,
            GatedRepoError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
        )
    except Exception:
        return None, "huggingface-hub is not installed (pip install huggingface-hub)."
    try:
        # an anonymous fetch: the Hub's "unauthenticated requests" notice comes back with it
        # (a response header huggingface_hub logs), and fetch_config keeps it off the screen
        # (poros.core.env.quiet_hub_notice)
        downloaded = fetch_config(model)
    except GatedRepoError:  # subclass of RepositoryNotFoundError: check first
        return None, (
            "this model is gated on the Hub; accept its license on "
            "huggingface.co, authenticate (hf auth login), then retry."
        )
    except ConfigTooLarge:
        return too_large
    except LocalEntryNotFoundError:  # subclass of EntryNotFoundError: no Hub, and not cached
        return no_hub()
    except EntryNotFoundError:
        return None, (
            "the repo exists but has no config.json, so it is not a standard "
            "transformers checkpoint."
        )
    except RepositoryNotFoundError:
        return None, (
            "model id not found on the Hub. If it is private or gated, "
            "authenticate first (hf auth login); unauthenticated requests "
            "report gated repos as not-found. Otherwise check the spelling, "
            "or pass a local model directory / config.json."
        )
    except OSError as exc:  # OfflineModeIsEnabled, a refused or dropped connection
        if _offline():
            return no_hub()
        return None, (
            f"could not reach the Hub to fetch {model}'s config.json ({type(exc).__name__}); "
            "check your network, or pass a local model directory / config.json."
        )
    except Exception as exc:
        return None, (
            f"could not fetch config.json from the Hub ({type(exc).__name__}); "
            "check your network, or pass a local model directory / "
            "config.json."
        )
    raw = read_config(downloaded)
    if raw is None:
        return None, "the model's config.json could not be parsed as JSON."
    try:
        bounded(raw)
    except ValueError:
        return too_large
    mt = registry_type(raw)
    if mt is None:
        return None, "the model's config.json has no 'model_type' field."
    return plain_type(mt), ""


def hub_fault(model: str, revision: Optional[str] = None) -> Optional[str]:
    """Why a Hub id's repo cannot be read, asked before anything is offered or downloaded for
    it: ``"missing"`` (no such repo; a private one answers the same with no login here),
    ``"gated"``, ``"offline"`` (HF_HUB_OFFLINE is set and its config.json is not in the cache)
    or ``"network"`` (the Hub was not reached, and it is not in the cache). None where its
    config.json is in the cache or was fetched, and for anything else (a repo with no
    config.json, a config too large): the family gate says those."""
    try:
        from huggingface_hub.utils import (
            EntryNotFoundError,
            GatedRepoError,
            HFValidationError,
            LocalEntryNotFoundError,
            RepositoryNotFoundError,
        )
    except Exception:  # no huggingface_hub: the load says what it needs
        return None
    from poros.plan.configs import ConfigTooLarge, fetch_config

    try:
        fetch_config(model, revision)
    except GatedRepoError:  # a RepositoryNotFoundError: asked first
        return "gated"
    except (RepositoryNotFoundError, HFValidationError):
        return "missing"
    except LocalEntryNotFoundError:  # an EntryNotFoundError: the Hub was not asked
        return "offline" if _offline() else "network"
    except (EntryNotFoundError, ConfigTooLarge):
        return None
    except OSError:  # OfflineModeIsEnabled, a refused or dropped connection
        return "offline" if _offline() else "network"
    except Exception:
        return None
    return None


def routable_arch_types() -> set[str]:
    """Every ``model_type`` a run routes: the arch gate's runnable set (validated and
    user-asserted, dense) and the families validated on ``poros.moe``."""
    from poros.core.arch_registry import runnable_arch_types, validated_arch_types

    return runnable_arch_types() | validated_arch_types("moe")


def asserted_only(model_type: Optional[str]) -> bool:
    """Whether ``model_type`` runs on its user's assertion alone (no parity guarantee)."""
    from poros.core.arch_registry import user_asserted_arch_types, validated_arch_types

    return (model_type is not None and model_type in user_asserted_arch_types()
            and model_type not in validated_arch_types())


def arch_refusal(model_type: str, mode: str = "error") -> Optional[Exception]:
    """The architecture gate's refusal of ``model_type`` (``PorosArchNotValidatedError``), routed
    as a run routes it: None where it runs (a validated MoE family, or one the dense gate lets
    through in ``mode``)."""
    from poros.core.arch_registry import blockwise_arch_check, validated_arch_types
    from poros.core.errors import PorosArchNotValidatedError

    if model_type in validated_arch_types("moe"):
        return None
    try:
        blockwise_arch_check(model_type, mode=mode)  # type: ignore[arg-type]
    except PorosArchNotValidatedError as exc:
        return exc
    return None


# Why a mixture of experts no one has validated does not train on a GPU, where no flag changes that.
MOE_ON_GPU = ("it is a mixture of experts, and on a GPU only a validated MoE family trains: a new "
              "one would take the dense route, where its fused experts are not made NF4, and no "
              "run has checked that route")
# Why a looped decoder does not train on a Mac: the Mac's path runs each layer once a forward.
# What it says next depends on the family's registry status (mac_looped): only a validated
# family is sent to Linux; a target is said to be not validated anywhere yet.
_RERUNS = ("its decoder reruns its middle layers once per loop pass, which the Mac's path does "
           "not run")
MAC_LOOPED = f"{_RERUNS}: in this release it trains on Linux with an NVIDIA GPU"
MAC_LOOPED_TARGET = (f"{_RERUNS}; Poros is adding this family and has not validated it on any "
                     "machine yet, so no run of it trains in this release")
MAC_LOOPED_WORDS = (MAC_LOOPED, MAC_LOOPED_TARGET)


def _looped_validated(model: Any) -> bool:
    from poros.core.arch_registry import arch_status, registry_type

    kind = registry_type(model) if isinstance(model, dict) else model
    return isinstance(kind, str) and arch_status(kind) == "validated"


def mac_looped(model: Any) -> str:
    """Why a looped decoder (``model``: its model_type or config.json) does not train on a
    Mac, true to the family's registry status: a validated one trains on Linux with an NVIDIA
    GPU; any other is not validated on any machine yet, and nothing is promised of Linux."""
    return MAC_LOOPED if _looped_validated(model) else MAC_LOOPED_TARGET


def mac_looped_fix(model: Any) -> str:
    """The next step after ``mac_looped``, as true as it is."""
    return ("Train it on Linux with an NVIDIA GPU." if _looped_validated(model) else
            "Train a model this Mac trains (poros models lists them).")


class LoopedOnAMacError(PorosError):
    """A looped decoder asked of the Mac's path (MAC_LOOPED): no flag trains it there."""

    code = "E_ARCH_UNSUPPORTED"


# Why a MoE family Poros has a path for, but has not yet validated, does not train.
MOE_TARGET = ("it is a mixture of experts Poros is adding but has not validated, and on a GPU "
              "only a validated MoE family trains")


def no_flag_trains(model_type: Optional[str], raw: Optional[dict], *, unified: bool,
                   moe: Optional[bool] = None) -> str:
    """Why no flag trains ``model_type`` (its config.json ``raw``) here, off a Mac: a variant the
    registry marks unsupported (neither loop trains it), or a mixture of experts whose family is
    not validated on ``poros.moe`` (its registry entry says so, or its config names experts). A
    dense target is no different from any dense family here. "" where ``--unvalidated``
    can train it, and on a Mac, whose own reasons are its own (``poros.cli.verbs.arch``), but
    for a decoder that runs its layers as visits (MAC_LOOPED), which no Mac run trains.
    ``moe``: whether its layers hold experts, where a reading of them says (else its config's
    names)."""
    from poros.core.arch_registry import arch_status, architectures, validated_arch_types
    from poros.moe.carried import declares_visits

    if unified and declares_visits(raw or model_type):  # validated or not: not on a Mac
        return mac_looped(raw or model_type)
    if unified or model_type is None:
        return ""
    if model_type in validated_arch_types() | validated_arch_types("moe"):
        return ""
    status = arch_status(model_type)
    spec = architectures().get(model_type)
    notes = spec.notes if spec is not None and spec.notes else ""
    if status == "unsupported":
        return "neither loop trains this variant" + (f": {notes}" if notes else "")
    if spec is not None and spec.entry == "moe" and status == "target":
        # a MoE family poros.moe is being made to run, not yet checked: say which
        return MOE_TARGET + (f": {notes}" if notes else "")
    if spec is not None and spec.entry == "moe":  # a mixture of experts by its registry entry
        return MOE_ON_GPU
    if moe is None and raw is not None:
        from poros.metal import names_experts

        moe = names_experts(raw)
    return MOE_ON_GPU if moe else ""


def _offline() -> bool:
    """Whether Hub access is turned off here (HF_HUB_OFFLINE)."""
    try:
        from huggingface_hub import constants

        return bool(constants.HF_HUB_OFFLINE)
    except Exception:  # pragma: no cover - an old or broken huggingface_hub
        return False


def offline_reason(model: str) -> Optional[str]:
    """Why a Hub id's config.json cannot be read where Hub access is off (HF_HUB_OFFLINE) and
    the cache does not hold it, in the one message every command says; None with the Hub on."""
    return _no_hub(model) if _offline() else None


def _no_hub(model: str) -> str:
    """Why a Hub id's config.json could not be read when the Hub could not be asked: the repo
    may well have one, so this never says it has none."""
    if _offline():
        return (f"Hub access is off here (HF_HUB_OFFLINE is set) and {model}'s config.json is "
                "not in the local cache; unset HF_HUB_OFFLINE, or pass a local model directory "
                "/ config.json.")
    return (f"could not reach the Hub, and {model}'s config.json is not in the local cache; "
            "check your network (or a proxy / firewall), or pass a local model directory / "
            "config.json.")
