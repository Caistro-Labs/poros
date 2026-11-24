"""arch: what it would take to train a model, from its config alone. No weights, and no token.

    poros arch Qwen/Qwen3-Coder-Next
    poros arch ./my-model
    poros arch try Qwen/Qwen3-Coder-Next

It reads config.json and the Hub's list of files, as anyone could, so a gated repo says it is gated.
It builds the model with nothing allocated: seconds, and small memory. Sizes are arithmetic, not a
run. `try` builds a tiny random model from the real config and checks, on the CPU, the route this
machine trains on. On a Mac: the Mac's path, bit for bit against a dense reference. Elsewhere: the
CUDA route, the blockwise forward a run trains with, against the model's own transformers
forward: the loss and every adapter gradient within 1e-4. Passing says the math holds on tiny
shapes. It says nothing
about Metal, NF4, the GPU, real shapes or the real weights: on CUDA, `poros prove` runs the GPU
gates next. A model no class here can build says so, and Poros never runs a repo's own code. A
config may be anyone's: only numbers and known words from one reach this output.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shlex
import stat
import sys
from pathlib import Path
from typing import Any, ClassVar, Literal, NamedTuple, Optional

from pydantic import BaseModel, Field

from poros.cli.action import Action, Draft
from poros.cli.context import Context
from poros.cli.schemas import Output, Question
from poros.cli.ui.text import Line
from poros.cli.verbs import shared
from poros.cli.words.grammar import Sentence
from poros.core.errors import EXIT_CODES
from poros.core.refusal import Refusal
from poros.plan.configs import CONFIG_CAP, read_config

GIB = 2**30
_SAID = re.compile(r"[A-Za-z0-9._/-]{1,200}")  # what a model may be called
HEADER_CAP = 100 << 20  # safetensors' own limit on a file's header
ROOM_BYTES, ROOM_PER_CORE = 3 << 30, 64 << 20  # what arch and arch try may map beyond their code
_FORMATS = ("BF16", "F16", "F32", "F64", "BOOL", "I16", "I32", "I64", "U16", "U32", "U64")
# What a pack cannot start from: weights already at 8 bits, which NF4 cannot be made from.
_EIGHT_BIT = ("F8_E4M3", "F8_E5M2", "FP8", "I8", "U8", "UI8")


class ArchAction(Action):
    verb: ClassVar[str] = "arch"
    model: str
    do: Literal["see", "try"] = "see"

    def canonical(self) -> list[str]:
        return ["arch", *(["try"] if self.do == "try" else []), self.model]


class Weights(BaseModel):
    format: Optional[str] = None  # a known word, or "unknown"
    usable: Optional[bool] = None
    why: str = ""


class ArchOutput(Output):
    schema_: Literal["poros.arch.v1"] = Field("poros.arch.v1", alias="schema")
    model: str
    model_type: Optional[str]  # a type this install or the registry knows, or "unknown"
    status: str  # validated, user_asserted, target, not-in-registry or gated
    weights: Weights
    layers: Optional[dict[str, Any]] = None  # path, count, linears in the largest
    experts: Optional[dict[str, Any]] = None  # per_layer, per_token
    head: Optional[dict[str, Any]] = None  # tied, gib
    memory: Optional[dict[str, Any]] = None  # adapter_at_rank_16_gib, one_layer_gib, embed_head_gib
    disk: dict[str, Any] = {}  # pack_gib, download_gib, by_range
    cannot_place: list[str] = []  # what could not be found or built, in Poros's words
    next: list[str] = []  # built from the checked model name alone, with no consent in it
    tried: Optional[dict[str, Any]] = None  # try: ok, status, first_difference, why, says
    # no flag trains it (off a Mac): {"code": "E_ARCH_UNSUPPORTED", "reason": ...}, and exit 4
    refused: Optional[dict[str, str]] = None
    exit_code: int = Field(0, exclude=True)  # 1: try ran and found a difference, or could not run


class _Seen(NamedTuple):
    raw: Optional[dict] = None  # config.json, as data
    counts: Optional[dict] = None  # parameters per format
    download_gib: Optional[float] = None
    gated: bool = False


def resolve(s: Sentence, ctx: Context) -> Draft[ArchAction]:
    if not s.subject:
        raise Refusal("E_USAGE", "Which model", "arch needs a model",
                      fix="poros arch Qwen/Qwen2.5-7B", note="a Hub id or a folder")
    word = s.subject[0]
    if not _SAID.fullmatch(word):
        raise Refusal("E_USAGE", "Which model", "the model is not a Hub id or a plain path")
    # a catalog name, a Hub id or a folder, as every command names a model
    model, _, ask = shared.resolve_model([word], ctx, verb="arch")
    if ask is not None:
        return Draft(None, [ask])
    return Draft(ArchAction(model=model, do="try" if s.action == "try" else "see"))


def confirmations(action: ArchAction, ctx: Context) -> list[Question]:
    return []


def _header(shard: Path) -> dict:
    """A safetensors file's table of tensors, from its header alone: no file mapped, however large,
    so the memory limit below is about this config. A pipe cannot hang this, and it is capped."""
    fd = os.open(shard, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as f:
        if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):
            raise ValueError("it is not a regular file")
        size = int.from_bytes(f.read(8), "little")
        if size > HEADER_CAP:
            raise ValueError("its header is too large")
        return json.loads(f.read(size))


def _from_folder(model: str) -> _Seen:
    path = Path(model)
    raw = read_config(path / "config.json" if path.is_dir() else path)
    counts: dict[str, int] = {}
    try:
        for shard in path.glob("*.safetensors") if path.is_dir() else ():
            for name, tensor in _header(shard).items():
                if name != "__metadata__":
                    kind = tensor["dtype"]
                    counts[kind] = counts.get(kind, 0) + math.prod(tensor["shape"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):  # a hostile or torn shard:
        counts = {}  # its format stays unknown
    return _Seen(raw, counts or None)


def _from_hub(model: str) -> _Seen:
    """What the Hub tells anyone: no token is read or sent, and only config.json comes down."""
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import GatedRepoError

    counts, download, sized, gated = None, None, None, False
    try:
        info = HfApi(token=False).model_info(model, files_metadata=True, token=False)
        counts = dict(info.safetensors.parameters) if info.safetensors else None
        files = [s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors")]
        download = round(sum(files) / GIB, 2) if files else None
        sized = next((s.size for s in info.siblings if s.rfilename == "config.json"), None)
        gated = bool(info.gated)
    except GatedRepoError:
        return _Seen(gated=True)
    except Exception:  # offline, or no such repo: the cache may hold its config
        pass
    if gated or (sized or 0) > CONFIG_CAP:
        return _Seen(None, counts, download, gated)
    try:
        path = Path(hf_hub_download(model, "config.json", token=False))
    except GatedRepoError:
        return _Seen(None, counts, download, True)
    except Exception:
        return _Seen(None, counts, download)
    return _Seen(read_config(path), counts, download)


def _weights(counts: Optional[dict]) -> Weights:
    if not counts:
        return Weights(why="its weights' format could not be read")
    most = max(counts, key=counts.get).upper()
    if most in _EIGHT_BIT or most.startswith("F8"):
        word = most if most in _EIGHT_BIT else "unknown"
        return Weights(format=word, usable=False, why=f"this release ships {word}, and Poros needs "
                       "bf16 to make NF4 from: look for a bf16 repo of this model")
    return Weights(format=most, usable=True) if most in _FORMATS else Weights(format="unknown")


def _named(kind: Any) -> tuple[str, str]:
    """(its type, its status), in words someone else's config cannot have written."""
    from poros.core.arch_registry import architectures, validated_arch_types

    known = isinstance(kind, str) and kind in architectures()
    try:
        from poros.metal import layout

        known = known or layout.known_type({"model_type": kind})
    except ImportError:  # no transformers to ask: a type the registry does not know is named
        if not known:  # as `poros check` names it there
            from poros.preflight.arch import plain_type

            return plain_type(kind), "not-in-registry"
    if not known:
        return "unknown", "not-in-registry"
    if kind in validated_arch_types() | validated_arch_types("moe"):
        return kind, "validated"
    spec = architectures().get(kind)
    return kind, spec.status if spec else "not-in-registry"


def _listed(model: str) -> Optional[dict]:
    """The catalog's row for a Hub id it names; None for a folder here or any other model."""
    from poros.preflight.arch import _catalog_row

    return None if Path(model).exists() else _catalog_row(model)


def _unread(model: str) -> str:
    """Why no config.json was read for ``model``: with the Hub off, the words `poros check` says
    it in. Nothing is asked again: arch asks the Hub once, with no token."""
    from poros.preflight.arch import offline_reason

    return (None if Path(model).exists() else offline_reason(model)) or UNREAD


def _gib(nbytes: float) -> float:
    return round(nbytes / GIB, 3)


def see(action: ArchAction, seen: _Seen, ctx: Context, fits: bool = True) -> ArchOutput:
    from poros.core.arch_registry import registry_type

    raw = seen.raw
    unread = seen.gated or raw is None or not raw.get("model_type")
    listed = _listed(action.model) if unread else None
    if listed is not None:  # the catalog answers for a model it names, as `poros check` does
        kind, status = _named(listed["arch"])
        why = ("it is gated, and arch asks the Hub with no token" if seen.gated
               else _unread(action.model))
        return ArchOutput(model=action.model, model_type=kind, status=status,
                          weights=_weights(seen.counts), disk={"download_gib": seen.download_gib},
                          cannot_place=[f"its sizes, which arch reads from its config.json: {why}"])
    if seen.gated:
        return ArchOutput(model=action.model, model_type="unknown", status="gated",
                          weights=_weights(seen.counts), disk={"download_gib": seen.download_gib},
                          cannot_place=["it is gated, and Poros asks the Hub with no token"])
    if unread:
        raise Refusal("E_NOT_FOUND", "No config", _unread(action.model),
                      context=["a catalog name, a Hub id, a folder or a .json file"])
    model_type, status = _named(registry_type(raw))
    out = ArchOutput(model=action.model, model_type=model_type, status=status,
                     weights=_weights(seen.counts), disk={"download_gib": seen.download_gib},
                     next=[action.model_copy(update={"do": "try"}).command()])
    from poros.preflight.arch import no_flag_trains

    if not ctx.unified and no_flag_trains(model_type, raw, unified=False, moe=False):
        out.next = []  # what the registry refuses: its layout is not this release's to read
        return out
    try:
        from poros.metal import layout
    except ImportError:  # `arch try` would refuse for it: no next that says to run it
        out.cannot_place, out.next = [NEEDS_ML], []
        return out
    if why := layout.can_build(raw):
        out.cannot_place, out.next = [why], []
        return out
    try:
        found = layout.layout_of(raw)
    except Exception as exc:  # an unknown family, or a hostile one: what it is, and never its words
        out.cannot_place = [f"it could not be built from its config ({type(exc).__name__})"]
        return out
    out.layers = {"path": found.layers, "count": found.count, "linears": found.linears}
    if found.moe:
        out.experts = {"per_layer": found.experts_per_layer,
                       "per_token": found.experts_per_token}
    out.head = {"tied": found.tied, "gib": _gib(found.head_bytes)}
    out.memory = {"adapter_at_rank_16_gib": _gib(found.adapter_bytes_16),
                  "one_layer_gib": _gib(found.one_layer_bytes),
                  "embed_head_gib": _gib(found.embed_bytes + found.head_bytes)}
    out.disk.update(pack_gib=_gib(found.pack_bytes), by_range=False,
                    by_range_note="a pack fetched by byte range would need only the pack's size; "
                    "poros downloads the shards first today")
    from poros.cli.words.models import by_id

    if (fits and action.do != "try" and ctx.unified and by_id(action.model) is None
            and out.status in _NEW):  # a family no one
        why = (out.weights.why if out.weights.usable is False else "") or _not_on_a_mac(
            seen.raw, out)  # validated this family: why no flag trains it on this Mac
        if why:
            out.cannot_place.append(f"no flag trains it on a Mac: {why}")
    return out


# What a family no one has validated trains through on a Mac, as `train --unvalidated` runs it.
DERIVED_ROW = ("on its own assertion (train asks for --unvalidated): through a row derived from "
               "its config, the nearest model's lengths, its memory and disk arithmetic, none "
               "of it measured")
_NEW = ("user_asserted", "target", "not-in-registry")  # what train asserts


# What `try` proved, by route: in Poros's words alone.
SAYS = {
    "mac": "on a tiny random model of this config, on the CPU: nothing about Metal, real shapes or "
           "the real weights",
    "cuda": "a tiny random model of this config, on the CPU in fp32. Nothing about NF4, the "
            "GPU, real shapes or the real weights: prove runs the GPU gates on this machine",
}


def tried(out: ArchOutput, raw: dict, ctx: Context) -> ArchOutput:
    """Rung one, for whatever the model is, on the route this machine trains on: a Mac's, or the
    CUDA route. A failure is a result, and names where it differs. A pass for a family Poros has
    not validated points at the next rung: on CUDA `prove`, which asks before it runs; on a Mac,
    where the gates do not run, `train` with ``--data <file>`` to fill in, which asks for
    --unvalidated itself."""
    import tempfile

    route = "mac" if ctx.unified else "cuda"
    try:
        from poros.metal import tiny
        from poros.validate import tiny_route
    except ImportError:
        out.tried = {"ok": False, "route": route,
                     "why": "needs torch and transformers (pip install poros-train)"}
        return out
    with tempfile.TemporaryDirectory(prefix="poros-arch-") as folder:
        try:
            out.tried = tiny.prove(raw, Path(folder)) if route == "mac" else tiny_route.prove(raw)
        except Exception as exc:  # its name alone: its words may be the config's
            out.tried = {"ok": False, "why": f"it could not be built or run ({type(exc).__name__})"}
    out.tried["route"] = route
    out.tried["says"] = SAYS[route]
    out.next = []
    if out.tried.get("ok") is False and route == "cuda":  # a failure is a result: the exit says so
        out.exit_code = 1
    if not out.tried.get("ok") or out.status in ("validated", "gated"):
        return out
    if route == "cuda":
        out.next = [shlex.join(["poros", "prove", out.model])]
    elif not (out.weights.usable is False or _not_on_a_mac(raw, out)):  # one a Mac trains
        # its data is a fill-in, as the guide's are: <file>, the agent's own data file
        out.next = [shlex.join(["poros", "train", out.model]) + " --data <file>"]
    return out


@contextlib.contextmanager
def _memory_held():
    """Room for one skeleton or one tiny model, on top of what this process maps already, so a field
    no bound foresaw ends here: a MemoryError, in its name alone, in this process alone. The stack
    and arena a thread maps count, so the room grows with the cores. Linux only: a Mac does not
    enforce RLIMIT_AS, and there the bounds above are the whole of it."""
    if sys.platform != "linux":
        yield
        return
    import resource

    try:  # what it would load comes first: a limit is about the config, not an import
        from poros.metal import layout, tiny  # noqa: F401
    except ImportError:
        pass
    try:  # the CUDA route's: peft and the trainer
        import peft  # noqa: F401

        from poros.train import trainer  # noqa: F401
        from poros.validate import tiny_route  # noqa: F401
    except ImportError:
        pass
    with open("/proc/self/statm") as f:
        mapped = int(f.read().split()[0]) * resource.getpagesize()
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    room = mapped + ROOM_BYTES + ROOM_PER_CORE * (os.cpu_count() or 1)
    resource.setrlimit(resource.RLIMIT_AS, (room if hard < 0 else min(room, hard), hard))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))


def run(action: ArchAction, ctx: Context) -> ArchOutput:
    shared.refuse_planted(action.model)
    if action.do == "try":  # it builds the model's layers and trains them a step: torch's work
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            raise Refusal("E_NO_TORCH", "Needs PyTorch",
                          "arch try builds the model's layers and trains them a step, which "
                          "needs PyTorch and transformers, not installed here",
                          fix="pip install poros-train",
                          note="torch, transformers, peft") from None
    with _memory_held():
        local = Path(action.model).exists()
        seen = _from_folder(action.model) if local else _from_hub(action.model)
        out = see(action, seen, ctx)
        why = refusal(out, seen.raw, ctx) if not ctx.unified and out.status != "validated" else ""
        if ctx.unified and _looped(seen.raw, out.model_type):  # whatever its status, not here
            from poros.preflight.arch import mac_looped

            why = mac_looped(seen.raw or out.model_type)
        if why and why != NEEDS_ML:  # docs/models.md, rung 0: refused, exit 4
            out.refused = {"code": "E_ARCH_UNSUPPORTED", "reason": why}
            out.next, out.exit_code = [], EXIT_CODES["E_ARCH_UNSUPPORTED"]
            return out
        return tried(out, seen.raw, ctx) if action.do == "try" and not out.cannot_place else out


NEEDS_ML = "its layout: needs torch and transformers (pip install poros-train)"
# Why a family no one has validated does not train, where no flag changes that: in Poros's words.
GATED = "it is gated, and Poros asks the Hub with no token, so what it is can't be read"
UNREAD = "no config.json with a model_type could be read for it"
EXPERTS_ON_MAC = ("its config names experts, and the Mac's path finds none in its layers that "
                  "it can place")


def _looped(raw: Optional[dict], model_type: Optional[str]) -> bool:
    """A decoder that declares visits (poros.moe.carried.declares_visits): the Mac's path, which
    runs each layer once a forward, does not train it."""
    from poros.moe.carried import declares_visits

    return declares_visits(raw or model_type)


def _not_on_a_mac(raw: dict, out: "ArchOutput") -> str:
    """Why the Mac's path cannot pack this model, before anything downloads, or "". Its pack holds
    bf16 alone, and a config that names experts it cannot find would train none of them."""
    from poros.metal import names_experts

    said = out.weights.format
    if said and said != "unknown" and said != "BF16":
        return (f"this release ships {said}, and a Mac makes NF4 from bf16 alone: look for a "
                "bf16 repo of this model")
    return EXPERTS_ON_MAC if names_experts(raw) and out.experts is None else ""


class Reading(NamedTuple):
    """What `train` asks of a model outside the catalog: arch's reading of it, in its words."""

    model_type: str  # a type this install or the registry knows, or "unknown"
    status: str  # validated, user_asserted, target, not-in-registry, gated or unread
    refused: str = ""  # why no flag trains it, or ""
    moe: bool = False


def reading(model: str, ctx: Context) -> Reading:
    """``arch``'s reading, as `train` takes it: config, weights' format and layout, read as `arch`
    reads them, with no token and nothing of the repo's run. Sizes are not computed."""
    from poros.core.arch_registry import registry_type

    shared.refuse_planted(model)
    with _memory_held():
        seen = _from_folder(model) if Path(model).exists() else _from_hub(model)
        if seen.raw is not None and seen.raw.get("model_type"):
            kind, status = _named(registry_type(seen.raw))
            if ctx.unified and _looped(seen.raw, kind):  # validated or not: not on a Mac
                from poros.preflight.arch import mac_looped

                return Reading(kind, status, mac_looped(seen.raw), True)
            if status == "validated":  # its layout needs no reading: the registry has checked it
                return Reading(kind, status)
        try:
            out = see(ArchAction(model=model), seen, ctx, fits=False)
        except Refusal:
            return Reading("unknown", "unread", UNREAD)
    if out.status == "gated":
        return Reading("unknown", "gated", GATED)
    kind = out.model_type or "unknown"
    return Reading(kind, out.status, refusal(out, seen.raw, ctx), out.experts is not None)


def refusal(out: "ArchOutput", raw: Optional[dict], ctx: Context) -> str:
    """Why no flag trains what ``out`` read (``train`` refuses it, ``--unvalidated`` too, and
    ``arch`` refuses it, E_ARCH_UNSUPPORTED): gated, weights NF4 cannot be made from, a layout
    that could not be placed, what the registry and the route refuse
    (``poros.preflight.arch.no_flag_trains``), or on a Mac its own reasons; "" where a flag
    trains it."""
    from poros.preflight.arch import no_flag_trains

    if out.status == "gated":
        return GATED
    if ctx.unified and _looped(raw, out.model_type):
        from poros.preflight.arch import mac_looped

        return mac_looped(raw or out.model_type)
    why = (no_flag_trains(out.model_type, raw, unified=False, moe=False)
           if not ctx.unified else "")  # what the registry refuses comes first
    why = why or (out.weights.why if out.weights.usable is False else "")
    why = why or (out.cannot_place[0] if out.cannot_place else "")
    if not why and not ctx.unified:
        why = no_flag_trains(out.model_type, raw, unified=False, moe=out.experts is not None)
    if not why and ctx.unified:
        why = _not_on_a_mac(raw, out)
    return why


def view(r: ArchOutput, w: int) -> list[Line]:
    from poros.cli import views

    return views.arch(r, w)
