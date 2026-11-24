"""`poros check <model>`, `poros agents` and `poros schema`: a model's architecture
(``poros.preflight.arch``), the agents' guide and the JSON Schemas; and the environment checks
`poros machine` runs (``_environment_checks``)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from poros.cli import contract
from poros.cli.contract import refuse
from poros.cli.root import _JSON_OPTION, _ui, app
from poros.core.errors import EXIT_CODES
from poros.preflight import arch as preflight_arch


def _torch_facts(quietly: bool) -> Optional[tuple[str, Optional[str], list, Optional[int]]]:
    """torch's version, its CUDA, its cards as (name, GiB) and its default threads; None when it
    does not import. ``quietly``: from its own files and nvidia-smi, where that can say, as
    importing it takes 750 MB beside a run. Its threads are then None: only it knows that count."""
    if quietly:
        import runpy
        from importlib.metadata import distribution

        from poros.core.card import visible_gpus

        cards = visible_gpus()
        version = distribution("torch").locate_file("torch/version.py")
        if cards is not None and Path(version).is_file():
            build = runpy.run_path(str(version))
            rows = [(c.name, c.total_gib) for c in cards] if build["cuda"] else []
            return build["__version__"], build["cuda"], rows, None
    try:
        import torch
    except ImportError:
        return None
    rows = [(torch.cuda.get_device_name(i),
             torch.cuda.get_device_properties(i).total_memory / 1024**3)
            for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    return torch.__version__, torch.version.cuda or "unknown", rows, torch.get_num_threads()


def _cublas_check(value: str) -> tuple[str, str, str]:
    """The determinism setting. Every way a run starts sets it itself before it loads a model
    (``poros.core.determinism.enforce_determinism``), so unset is nothing to do: a note, never
    a warning, for a person and an agent alike."""
    if value == ":4096:8":
        return "CUBLAS_WORKSPACE_CONFIG", value, "ok"
    return "CUBLAS_WORKSPACE_CONFIG", "poros train sets it as it starts", "info"


def _environment_checks(quietly: bool = False) -> tuple[list[tuple[str, str, str]], bool]:
    """Every environment check `poros machine` shows, as (name, value, status) rows.

    The one source for the card and for its JSON. Works with NO torch/GPU installed. Returns
    ``(checks, torch_available)``. ``quietly``: for a reader beside a run, see _torch_facts;
    asked plainly it imports torch, as whether torch imports is its question.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as _installed

    from poros import __version__

    def installed(dist: str) -> Optional[str]:
        # read from the install's metadata, not by importing: transformers
        # alone takes half a second to import, and /machine is meant to be instant
        try:
            return _installed(dist)
        except PackageNotFoundError:
            return None

    checks: list[tuple[str, str, str]] = []  # (name, value, status)

    # --- Python ---
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 11)
    checks.append((
        "Python",
        py_ver,
        "ok" if py_ok else "warn",
    ))

    # --- Platform ---
    import platform as _platform

    checks.append(("Platform", _platform.platform(), "info"))

    # --- Poros ---
    checks.append(("Poros", __version__, "ok"))

    # --- The Python path ---
    from poros._path_rule import dropped

    if dropped:  # by the time any of this runs, a planted poros/ there has run instead
        checks.append(("PYTHONPATH", "an empty or relative entry reads whatever folder you run in: "
                       "code planted there can run before Poros. Remove the entry.", "warn"))

    # --- PyYAML ---
    v = installed("PyYAML")
    checks.append(("pyyaml", v or "not installed", "ok" if v else "error"))

    # --- Pydantic ---
    v = installed("pydantic")
    checks.append(("pydantic", v or "not installed", "ok" if v else "error"))

    if sys.platform == "darwin":
        from poros.metal import machine

        mac, torch_available = machine.checks()
        return checks + mac, torch_available

    # --- torch ---
    # read only where it is installed
    facts = _torch_facts(quietly) if installed("torch") else None
    torch_available = facts is not None
    checks.append(("torch", facts[0], "ok") if facts else ("torch", "not installed", "warn"))

    # --- CUDA ---
    cuda_available = False
    if facts:
        _, cuda_ver, cards, threads_default = facts
        cuda_available = bool(cards)
        if cards:
            checks.append(("CUDA", cuda_ver, "ok"))
            for i, (name, vram_gb) in enumerate(cards):
                checks.append((
                    f"  GPU {i}",
                    f"{name} ({vram_gb:.1f} GiB)",
                    "ok",
                ))
        else:
            checks.append(("CUDA", "not available", "warn"))

        from poros.core.env import cpu_threads
        from poros.core.errors import PorosConfigError

        try:
            threads, source = cpu_threads(threads_default or 1)
            if threads_default:
                checks.append(("CPU threads", f"{threads} ({source})", "ok"))
        except PorosConfigError as e:
            checks.append(("CPU threads", e.info.message, "error"))

    for dist in ("transformers", "peft", "bitsandbytes", "accelerate"):
        v = installed(dist)
        checks.append((dist, v or "not installed", "ok" if v else "warn"))

    # --- jinja2 (for reports) ---
    v = installed("Jinja2")
    checks.append(("jinja2", v or "not installed (needed for HTML reports)", "ok" if v else "warn"))

    # --- Architecture support ---
    try:
        from poros.core.arch_registry import validated_arch_types

        # Collapse the <family>_text aliases: they are the same family and
        # the full list overflows the row (it was being truncated with an
        # ellipsis, hiding the answer to "is my model supported"). A MoE
        # family trains through poros.moe: its own row, as long.
        for name, entry in (("Validated archs", "dense"), ("Validated MoE archs", "moe")):
            families = sorted({a.removesuffix("_text") for a in validated_arch_types(entry)})
            checks.append((name, ", ".join(families), "info"))
    except ImportError:
        checks.append(("Validated archs", "registry not loaded (torch required)", "warn"))

    # --- Determinism protocol ---
    if torch_available and cuda_available:
        import os

        checks.append(_cublas_check(os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")))

    # --- Disk space ---
    try:
        import shutil

        usage = shutil.disk_usage(".")
        free_gb = usage.free / 1e9  # decimal GB, as a download's size is said
        status = "ok" if free_gb > 20 else ("warn" if free_gb > 5 else "error")
        checks.append(("Disk free", f"{free_gb:.1f} GB", status))
    except Exception:
        checks.append(("Disk free", "unknown", "warn"))

    return checks, torch_available


def _check_verdict(model: str, model_type: Optional[str], status: str,
                   unified: bool) -> tuple[str, list[str]]:
    """``poros check``'s verdict beyond the registry status: why no flag trains the family
    (poros.preflight.arch.no_flag_trains; train and `agents arch` refuse it alike; on a Mac,
    ``unified``, its row says), else, for a family --unvalidated trains, (why, the next
    command) -- ("", []) where neither says."""
    from poros.plan.configs import read_model_config
    from poros.preflight.arch import mac_looped, no_flag_trains

    if unified and model_type is not None:  # a looped decoder, validated or not: not on a Mac
        from poros.moe.carried import declares_visits

        if declares_visits(model_type):
            return mac_looped(model_type), []
    if model_type is None or status in ("validated", "could-not-resolve"):
        return "", []

    refused = no_flag_trains(model_type, read_model_config(model), unified=unified)
    if refused or status == "unsupported":
        return refused, []
    if status in ("not-in-registry", "target"):
        return "", [f"{model_type} is not validated: it trains for one run with --unvalidated, "
                    "with no parity guarantee", f"poros arch try {model}"]
    return "", []


@app.command()
def check(
    model: str = typer.Argument(
        ..., help="A catalog name, a Hub id, a local model folder, or a config.json path."
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Say whether Poros supports a model's family, from its config.json alone.

    Reads the model's model_type (a local folder, or the Hub's config.json) and says whether
    that family is validated. No torch, and no weights are downloaded.
    """
    from poros.cli.context import Context
    from poros.cli.verbs.shared import one_model
    from poros.core.arch_registry import arch_status, architectures, validated_arch_types
    from poros.core.refusal import Refusal

    try:  # named as train and fit name a model: the catalog, a Hub id (owner/name), a folder
        model = one_model(model, Context(), verb="check")
    except Refusal as r:
        contract.refused(r)
    mac = None
    if sys.platform == "darwin":
        from poros.core.card import detect_gpu
        from poros.metal.table import class_of, pick

        mac = detect_gpu()
    if contract.json_mode():
        model_type, resolve_reason = preflight_arch.resolve_model_type(model)
    else:
        with _ui()[0].activity("Resolving model_type") as _act:
            model_type, resolve_reason = preflight_arch.resolve_model_type(model)
            _act.done([])
    # The path its registry entry names (ArchSpec.entry), whatever its status: an unsupported MoE
    # family is a mixture as qwen3_moe is. Not in the registry: the dense route.
    spec = architectures().get(model_type) if model_type is not None else None
    moe = spec is not None and spec.entry == "moe"

    # Exit 0 only for validated; user_asserted is runnable but on the
    # caller's authority, so non-zero keeps scripts from mistaking it for a
    # Poros guarantee. An unresolvable model is a usage problem (2).
    if model_type is None:
        status_str, code = "could-not-resolve", 2
    else:
        raw = arch_status(model_type)
        status_str = (raw if raw in ("validated", "user_asserted", "target", "unsupported")
                      else "not-in-registry")
        if raw == "unsupported":
            resolve_reason = architectures()[model_type].notes
        code = 0 if status_str == "validated" else EXIT_CODES["E_ARCH_UNSUPPORTED"]
    # what no flag trains, and the step a family --unvalidated trains takes next: the rule train
    # and `agents arch` read (poros.preflight.arch.no_flag_trains)
    refused, after = _check_verdict(model, model_type, status_str, mac is not None)
    from poros.preflight.arch import MAC_LOOPED_WORDS

    if mac is not None and refused in MAC_LOOPED_WORDS:  # a looped decoder: not on a Mac
        status_str, code = "not-on-this-device", EXIT_CODES["E_ARCH_UNSUPPORTED"]
    if (mac is not None and model_type is not None and not refused
            and status_str != "unsupported" and pick(model, class_of(mac.name)) is None):
        # on a Mac: a family --unvalidated trains has the row train derives once it is asserted
        import logging

        from poros.core.arch_registry import asserted_inside

        # its layout imports torch, which warns on import that a Mac has no stdout redirects
        logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").disabled = True
        try:
            with asserted_inside(model_type):
                derived = (status_str in ("not-in-registry", "target")
                           and pick(model, class_of(mac.name)) is not None)
        except ImportError:  # its layout needs torch, as its run does: the family decides
            derived = status_str in ("not-in-registry", "target")
        if not derived:
            status_str, code, after = "not-on-this-device", EXIT_CODES["E_ARCH_UNSUPPORTED"], []
    if refused and status_str != "unsupported":
        resolve_reason = refused
    elif after and not resolve_reason:
        resolve_reason = after[0]

    if contract.json_mode():
        from poros.cli.schemas import CheckOutput

        contract.emit(CheckOutput(
            model=model, model_type=model_type, status=status_str,
            entry=None if model_type is None else ("moe" if moe else "dense"),
            reason=resolve_reason, next=after[1:] if after else [],
        ))
        raise typer.Exit(code=code)

    from poros.cli.ui import components as uc
    from poros.cli.ui.live import columns
    from poros.cli.ui.term import out
    from poros.cli.ui.text import S, ln

    w = min(columns(), uc.CAP)
    head = [uc.welcome("poros check", model, max_width=w), []]
    if model_type is None:  # the resolver's own reason: offline, gated, not found, or no field
        from poros.cli.ui.term import err

        err(head + uc.error_event("Could not resolve", str(resolve_reason or
                                                          "no model_type could be read"), w))
        raise typer.Exit(code=code)

    status = arch_status(model_type)
    lines = head + [uc.event("Resolved", f"model_type {model_type}")]
    if resolve_reason == preflight_arch.GATED_NOTE:  # the catalog's answer, and what a run needs
        lines += [uc.rail(S(part, "dm")) for part in uc.wrap(resolve_reason, w - 4)]
    if mac is not None and refused in MAC_LOOPED_WORDS:  # validated or not: not here
        lines += uc.error_event("Not on this Mac", f"{model_type}: {refused}", w)
        out(lines)
        raise typer.Exit(code=code)
    if mac is not None and not refused and status_str != "unsupported":  # those: as off a Mac
        if status_str == "not-on-this-device":
            lines += uc.error_event("Not on this Mac", f"{model} does not train here", w)
            lines += [[], *uc.continue_with("poros models", w)]
            out(lines)
            raise typer.Exit(code=code)
        if status_str != "validated":  # --unvalidated trains it here: said, and exit 4
            lines += [ln(S("○ ", "tx"), S("Trains on this Mac with --unvalidated", "a")),
                      *[uc.rail(S(part, "dm")) for part in uc.wrap(resolve_reason, w - 4)], [],
                      *uc.continue_with(after[1] if len(after) > 1 else "poros models", w)]
            out(lines)
            raise typer.Exit(code=code)
        lines += [ln(S("● ", "tx"), S("Trains on this Mac", "a")), [],
                  *uc.continue_with("poros recipes", w)]
        out(lines)
        return
    if status == "validated":
        lines.append(ln(S("● ", "tx"), S("Supported", "a"),
                        S("  validated · 0.00e+00 parity", "dm")))
        lines.append(uc.kv("entry", "moe" if moe else "dense", lw=9, vw=0))
        lines.append(uc.kv("quant", "nf4", lw=9, vw=0))
        lines.append([])
        lines += uc.continue_with(f"poros init -m {model}", w)
        out(lines)
        return
    if status == "user_asserted":
        lines.append(uc.event("Runs on your assertion", None, "not validated by Poros", mark="○"))
        lines.append(uc.rail(S("registered via register_architecture() "
                               "or POROS_USER_ASSERTED_ARCHS", "dm")))
        lines.append(uc.rail(S("no Poros parity guarantee applies: "
                               "confirm correctness with your own run", "dm")))
        out(lines)
        raise typer.Exit(code=code)
    if status == "unsupported":
        lines += uc.error_event("Unsupported", "neither loop trains this variant", w)
        lines.append(uc.rail(S(resolve_reason, "dm")))
        out(lines)
        raise typer.Exit(code=code)
    if refused:  # no flag trains it here: said, with no flag offered
        lines += uc.error_event("Not trainable", f"{model_type}: no flag trains it", w)
        lines.append(uc.rail(S(refused, "dm")))
        out(lines)
        raise typer.Exit(code=code)
    if status == "target":
        lines.append(uc.event("Target", None, "roadmap · parity not yet confirmed", mark="○"))
        lines.append(uc.rail(S("gated by default; to try it set ", "dm"),
                             S("blockwise_arch_check: warn"),
                             S(" and confirm parity with your own run", "dm")))
        lines.append(uc.rail(S("or train it for one run, with no parity guarantee: ", "dm"),
                             S(f"poros train {model} --unvalidated")))
        out(lines)
        raise typer.Exit(code=code)
    valid = ", ".join(sorted(validated_arch_types() | validated_arch_types("moe")))
    lines += uc.error_event("Not supported", f"{model_type} is not in the registry", w)
    lines.append(uc.rail(S("validated: ", "dm"), S(valid)))
    lines.append(uc.rail(S("a type this install knows trains for one run, with no parity "
                           "guarantee: ", "dm"), S(f"poros train {model} --unvalidated")))
    lines.append(uc.rail(S("how a family gets validated: ", "dm"), S("poros docs models")))
    out(lines)
    raise typer.Exit(code=code)


@app.command(hidden=False)
def agents() -> None:
    """The guide for coding agents; `poros agents <command>` runs one, in JSON for their shell."""
    from importlib import resources

    text = resources.files("poros").joinpath("agents.md").read_text()
    print(text, end="" if text.endswith("\n") else "\n", file=contract.stdout())


@app.command()
def schema(
    kind: Optional[str] = typer.Argument(
        None, help="config, events, artifact, or a command's output (see --list)."
    ),
    list_names: bool = typer.Option(False, "--list", help="List every schema name."),
    json_output: bool = typer.Option(
        False, "--json", help="Accepted, as every command takes it: a schema is JSON already."),
) -> None:
    """Print the JSON Schema for a Poros data model (for agents/tooling).

    config -> PorosConfig, events -> PorosEvent (events.jsonl rows),
    artifact -> RunArtifact (benchmark cells); every other name is the
    `--json` result of that command (poros.<name>.v1). Works without torch.
    """
    import json as _json

    from poros.cli.schemas import result_schema, schema_names

    names = schema_names()
    if list_names:
        print("\n".join(names), file=contract.stdout())
        return
    if kind == "config":
        from poros.core.config import PorosConfig as _M
    elif kind == "events":
        from poros.core.events import PorosEvent as _M
    elif kind == "artifact":
        from poros.bench.artifacts import RunArtifact as _M
    elif kind is not None and (doc := result_schema(kind)) is not None:
        print(_json.dumps(doc, indent=2), file=contract.stdout())
        return
    else:
        import difflib

        if kind is None:
            refuse("E_USAGE", "Which schema?", "name one",
                   fix="poros schema --list", note="lists every name")
        near = difflib.get_close_matches(kind, names, n=1)
        refuse("E_USAGE", "Unknown schema", f"{kind!r} is not one of {', '.join(names)}",
               fix=f"poros schema {near[0]}" if near else "poros schema --list")

    # Plain stdout (not Rich) so the JSON is machine-parseable when piped.
    print(_json.dumps(_M.model_json_schema(), indent=2), file=contract.stdout())
