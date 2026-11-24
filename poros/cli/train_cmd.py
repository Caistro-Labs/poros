"""``poros train``: preflight, load, live training surface, receipt, summary.

The flow, as the terminal shows it (the CLI visual design):

    ▛▜ poros v1.0.0 · Qwen/Qwen2.5-72B-Instruct · RTX 4090 24.0 GB
    ● Checked config  train.yaml          (+ rail: nf4 · block 4 · rank 16 · ./out)
    • Loading Qwen2.5-72B-Instruct 12s    (live)  ->  ● Loaded … 41.2s
    • Training …                          (live surface)
    ● Trained 1,000 steps  2h 44m 10s     (+ summary rails, receipt)
    Continue with: poros eval train.yaml

In the session the run's card is up from the load on: loading and the wait for step 1 are its
states, named on its activity line, with every number "–" until it exists.

The checks before the load run in ``poros.preflight.command``'s order (the output folder, the
data, the recipe, the resume, the model and its family, the GPU, the fit, the disk), each a
refusal in seconds, before any download. Torch is imported only after the torch-free checks. A
recipe run's data and loop are ``poros.cli.train_recipe``.
"""

from __future__ import annotations

import errno
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional

import typer

from poros.cli import contract
from poros.cli.contract import refuse
from poros.cli.train_recipe import recipe_data, train_recipe
from poros.cli.ui import components as c
from poros.cli.ui import term
from poros.cli.ui.screens import Fix, TrainResult, TrainState, error_screen
from poros.cli.ui.term import out
from poros.cli.ui.text import Line, S
from poros.core.errors import PorosCancelledError, PorosDiskError
from poros.core.refusal import Refusal
from poros.preflight import command
from poros.preflight.disk import checkpoint_note as _checkpoint_note
from poros.preflight.disk import download_note as _download_note
from poros.preflight.disk import output_over_start


def _refuse_if(r: Optional[Refusal]) -> None:
    """Show a preflight's refusal and exit with its code; nothing where there is none."""
    if r is not None:
        contract.refused(r)


def _argv_with(config_path: str, **flags: Any) -> str:
    """The exact retry command: the user's own argv with flags replaced."""
    argv = term.current_argv() or ["train", config_path]
    for flag, value in flags.items():
        name = "--" + flag.replace("_", "-")
        if name in argv:
            i = argv.index(name)
            if i + 1 < len(argv):
                argv[i + 1] = str(value)
                continue
        argv += [name, str(value)]
    return "poros " + " ".join(shlex.quote(a) for a in argv)


def _gpu() -> tuple[str, float]:
    if sys.platform == "darwin":
        from poros.core.card import detect_gpu

        mac = detect_gpu()
        return mac.name, mac.total_gib * 2**30 / 1e9
    import torch

    try:
        props = torch.cuda.get_device_properties(0)
        name = props.name.replace("NVIDIA ", "").replace("GeForce ", "")
        return name, props.total_memory / 1e9
    except Exception:
        return "GPU", 0.0


def _arch_label(trainer: Any) -> str:
    bm = getattr(trainer, "block_manager", None)
    if bm is not None and getattr(trainer, "_adapter_backend", None) != "experts":
        blocks = "" if sys.platform == "darwin" else f" · {bm.num_blocks} blocks"
        return f"dense · {bm.num_layers} layers{blocks}"
    cfg = getattr(getattr(trainer, "model", None), "config", None)
    cfg = getattr(cfg, "text_config", None) or cfg
    layers = getattr(cfg, "num_hidden_layers", None)
    experts = (
        getattr(cfg, "num_experts", None)
        or getattr(cfg, "n_routed_experts", None)
        or getattr(cfg, "num_local_experts", None)
    )
    parts = ["MoE"]
    if layers:
        parts.append(f"{layers} layers")
    if experts:
        parts.append(f"{experts} experts")
    return " · ".join(parts)


def write_receipt(path: Path, payload: dict) -> Optional[str]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
        return str(path)
    except OSError:
        return None


def hold_mac() -> Any:
    """One run per Mac: what this returns holds the lock while the caller keeps it."""
    from poros import jobs

    lock = jobs.hold_machine()
    if lock is None:
        refuse("E_USAGE", "Another run is using this Mac",
               "one pool of memory cannot host two runs",
               context=["add --detach to queue this one behind it"])
    return lock


def _mac_gate(cfg: Any, plan_gate, mt, submit, job_id) -> Optional[Any]:
    """Before a run on a Mac: the gate, one run per machine, and no sleep until it ends. A job's
    turn already holds the lock; what this returns holds it for anyone else."""
    from poros.preflight import mac

    mac.card(cfg)  # this Mac's stack and class (poros.metal.require)
    if submit is not None:  # the job's turn checks this Mac, holding it
        submit()
        return None
    lock = None if job_id else hold_mac()
    mac_ready()
    if plan_gate is not None:
        plan_gate(cfg, mt)
    return lock


def mac_ready() -> None:
    """A Mac its caller holds: checked once, and kept awake for as long as this process runs. A
    run's, and an eval's, which streams the same weights."""
    from poros import metal

    metal.check_this_mac()
    subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@dataclass
class _Ready:
    """What the checks before the load decided, for the rest of the flow."""

    choice: Any  # the recipe (poros.recipes.choose)
    mt: Optional[str]  # the model's model_type
    support: Optional[str]  # "user_asserted" where the run trains on an assertion
    sim: bool  # the simulator stands in for a GPU
    mac: bool
    gpu: str
    card: float  # the card's memory, GiB (a Mac: its share)
    path_word: str  # " · fast path" beside the settings, "" for sequential
    lock: Any = None  # a Mac's machine lock: held while the run is


def run(
    cfg: Any,
    *,
    config_path: str,
    resume_from: Optional[str],
    trust_remote_code: bool,
    friendly_error: Callable[[Exception], None],
    plan_gate: Optional[Callable[[Any, Optional[str]], None]] = None,
    submit: Optional[Callable[[], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    job_id: Optional[str] = None,
    checked: Optional[tuple[str, str]] = None,
    shown_run: str = "",
    resume_cmd: Optional[str] = None,
    next_cmd: Optional[str] = None,
    emit_json: bool = True,
    unvalidated: bool = False,
    force: bool = False,
    began: Optional[Callable[[], None]] = None,
    next_unscored: Optional[tuple[str, str]] = None,
    replaces: bool = False,
) -> Optional[dict]:
    """The train flow, shared by `poros train <config>` and the `train` verb.

    ``checked`` is the (what, kind) line naming the source ("run.yaml",
    "config"); ``resume_cmd`` and ``next_cmd`` are the follow-up commands the
    screens offer. Returns the training metrics, or None for a --detach
    submit. ``emit_json=False`` leaves the JSON result to the caller.
    ``unvalidated``: --unvalidated, the model's own type runs as user_asserted for this run.
    ``force``: --force, which turns the refusals a measurement predicts (the fit, a download
    past the free disk) into notes. ``began`` is called once every check before the load has
    passed, as the download and load begin: a refusal before it trained nothing.
    ``next_unscored``: (the command, how to score it) offered in ``next_cmd``'s place by a run
    that held no rows out, where ``next_cmd`` scores the held-out rows. ``replaces``: the
    output folder holds a run the caller agreed to replace and clears in ``began``, once every
    check has passed: its checkpoints are not refused as another run's."""
    w = term.get().width()
    wc = min(w, c.CAP)
    ready = _before_the_load(cfg, resume_from=resume_from, trust_remote_code=trust_remote_code,
                             friendly_error=friendly_error, plan_gate=plan_gate, submit=submit,
                             job_id=job_id, unvalidated=unvalidated, replaces=replaces)
    if ready is None:
        return None
    done = _steps_left_none(cfg, resume_from)
    if done is not None:  # every step it is asked for is trained and saved: nothing loads
        return _finished(cfg, done, cfg.max_steps, config_path=config_path,
                         resume_cmd=resume_cmd, next_cmd=next_cmd, emit_json=emit_json, w=w)
    _show_settings(cfg, ready, config_path=config_path, checked=checked, resume_from=resume_from,
                   trust_remote_code=trust_remote_code, force=force, wc=wc)
    if began is not None:
        began()
    loaded = _load(cfg, ready, shown_run=shown_run, resume_from=resume_from,
                   friendly_error=friendly_error,
                   plan_gate=plan_gate)
    if loaded is None:
        return None
    run_id, st, built, trainer, view = loaded
    return _train_and_save(cfg, ready, trainer, st, built, run_id=run_id,
                           resume_from=resume_from, should_stop=should_stop, job_id=job_id,
                           resume_cmd=resume_cmd, next_cmd=next_cmd, config_path=config_path,
                           emit_json=emit_json, friendly_error=friendly_error, w=w,
                           next_unscored=next_unscored, view=view)


def _steps_left_none(cfg: Any, resume_from: Optional[str]) -> Optional[int]:
    """The steps a resume's checkpoint holds, where they are every step the run is asked for
    and its finished adapter is saved: nothing is left to train. None otherwise, and for a run
    by epochs, whose length is known only once its data is read (``_train_and_save`` says it
    then)."""
    from poros.cli.ui.train_view import steps_in
    from poros.core import checkpoints

    done = steps_in(resume_from)
    if done is None or cfg.max_steps <= 0 or done < cfg.max_steps:
        return None
    return done if checkpoints.has_weights(Path(cfg.output_dir)) else None


def _finished(cfg: Any, done: int, total: int, *, config_path: str, resume_cmd: Optional[str],
              next_cmd: Optional[str], emit_json: bool, w: int, view: Any = None) -> dict:
    """A resume with no step left to train: said (never as a trained run with a loss no step
    computed), with the command that trains it further; exit 0, as a retry loop's last pass.
    ``view``: the live view to close into it, where the model was loaded to find out."""
    from poros.cli.ui.train_view import finished_lines

    if resume_cmd:  # a run by its id
        more = f"{resume_cmd} --more {done}"
    else:  # this command with more steps; a resume, whatever the command line said
        more = _argv_with(config_path, steps=done * 2)
        more += "" if {"--resume-from", "-r"} & set(more.split()) else " --resume-from latest"
    after = next_cmd or f"poros eval {config_path}"
    lines = finished_lines(done, total, cfg.output_dir, more, after, w)
    if view is not None:
        view.close(lines)
    else:
        out(lines)
    metrics = {"total_steps": 0, "steps_done": done, "already_finished": True}
    if contract.json_mode() and emit_json:
        from poros.cli.schemas import TrainOutput

        contract.emit(TrainOutput(output_dir=cfg.output_dir,
                                  events=str(Path(cfg.output_dir) / "events.jsonl"),
                                  metrics=metrics))
    return metrics


def _fill_as_this_mac(cfg: Any) -> None:
    """On a Mac (not the simulator), what the table fills where the config is silent
    (``poros.metal.table.fill``: gradient_accumulation_steps, max_seq_length, save_steps, the
    store), set now rather than at the Mac gate. The run trains with those values and its
    checkpoints record them, so the resume check must compare against them: a classic-loop run
    left its accumulation to the table (1), and checked against the config default (8) its
    continue was refused. ``fill`` sets only unset fields: the gate's own fill changes nothing."""
    if sys.platform != "darwin":
        return
    from poros.cli.ui import simulate

    if simulate.enabled():
        return
    from poros import metal
    from poros.metal import table

    table.fill(cfg, metal.device_class())


def _before_the_load(cfg: Any, *, resume_from: Optional[str], trust_remote_code: bool,
                     friendly_error: Callable[[Exception], None],
                     plan_gate: Optional[Callable[[Any, Optional[str]], None]],
                     submit: Optional[Callable[[], None]], job_id: Optional[str],
                     unvalidated: bool, replaces: bool = False) -> Optional["_Ready"]:
    """The checks before the load (``poros.preflight.command``'s order), and what they decided;
    None where the flow ends here (a --detach submit, or an error already reported)."""
    model = cfg.model_name_or_path
    # ---- torch-free preflight, in poros.preflight.command's order (1-5) ----
    _refuse_if(command.output_refusal(cfg.output_dir))
    _refuse_if(output_over_start(cfg.output_dir, cfg.init_adapter, resume_from=resume_from))
    from poros.preflight.resume import adopt_recorded

    # a resume: what the config leaves unset is what its run recorded, before presets or a Mac's
    # table fill anything (the config's own settings are compared against the checkpoint below)
    adopt_recorded(resume_from, cfg)
    try:
        _refuse_if(command.dataset_refusal(cfg))
    except Exception as exc:  # a dataset spec that cannot be read
        friendly_error(exc)
        return None
    from poros import recipes

    try:
        choice = recipes.choose(cfg)
    except ValueError as exc:
        contract.refused(command.recipe_refusal(exc))
    _refuse_if(command.start_refusal(resume_from, cfg))  # init_adapter: before the presets
    by_recipe = recipes.apply(cfg, choice)  # the recipe's settings, wherever the config says none
    _fill_as_this_mac(cfg)  # before the resume check reads them
    _write_resolved(cfg, choice)
    from poros.preflight.resume import adopt_revision

    # A run records what it trains on by content, and a resume compares it: one read of the
    # data file, and a sample of a local base. Past a second the wait is said, with how far
    # the file's read is; read here, the trainer's own read is a cache hit.
    doing = ("Checking the data matches the checkpoint" if resume_from
             else "Reading the data for the run's record")
    clash = None
    with term.get().activity(doing, delay=1.0) as act:
        def progress(done: int, size: int) -> None:
            act.label = f"{doing}  {done * 100 // max(size, 1)}%"

        if resume_from:
            adopt_revision(resume_from, cfg)  # before the download: the base it trained on
            clash = command.resume_refusal(resume_from, cfg, progress=progress)
        else:
            from poros.preflight.resume import trained_on

            trained_on(cfg.dataset, progress=progress)
        act.done([])
    _refuse_if(clash)
    _refuse_if(None if replaces else command.fresh_run_refusal(resume_from, cfg))

    from poros.cli.ui import simulate
    from poros.preflight import arch as preflight_arch

    with term.get().activity("Checking architecture") as _act:
        gone = command.model_refusal(model, cfg.model_revision, cache_decides=not (
            simulate.enabled() or sys.platform == "darwin"))
        mt, _reason = (None, "") if gone else preflight_arch.resolve_model_type(model)
        if gone is None and mt is None and sys.platform == "darwin" and not simulate.enabled():
            from poros.cli.verbs.mac import unread  # its family unread: no Hub, and no copy

            gone = unread(model, cfg.model_revision)
        _act.done([])
    _refuse_if(gone)  # no such model, a gated one, or no way to ask: before its family is read
    if sys.platform == "darwin" and not simulate.enabled():  # what no flag trains on a Mac
        _refuse_if(command.looped_on_a_mac(model, mt))
    if unvalidated:
        _assert_family(model, mt, trust_remote_code)
    from poros.core.arch_registry import arch_status

    support = "user_asserted" if mt is not None and arch_status(mt) == "user_asserted" else None
    if cfg.blockwise_arch_check == "error" and mt is not None and preflight_arch.asserted_only(mt):
        out(
            [
                c.event("Note", f"{mt} runs on your own assertion", mark="○"),
                c.rail(S("no parity guarantee applies to this run", "dm")),
            ]
        )
    _refuse_if(command.arch_refusal(model, mt, cfg.blockwise_arch_check))

    sim = simulate.enabled()
    lock = None
    if sim and submit is not None:
        refuse("E_USAGE", "Not simulated", "a simulated run is foreground only; drop --detach")
    mac = sys.platform == "darwin" and not sim
    if "block_size" in by_recipe and not mac:
        from poros.core.arch_registry import validated_arch_types

        if mt in validated_arch_types("moe"):  # a dense setting: poros.moe refuses one you say
            cfg.block_size = None
            _write_resolved(cfg, choice)
    if sim:
        gpu, card = simulate.GPU[0], simulate.GPU[1] / c.GIB  # the stand-in is said in GiB
    elif mac:
        from poros.cli import root
        from poros.core.errors import PorosError

        root._require_torch("poros train")  # every Mac trains: only torch can be missing
        try:
            # holds the lock until this returns
            lock = _mac_gate(cfg, plan_gate, mt, submit, job_id)
        except PorosError as exc:
            friendly_error(exc)
            return None
        if submit is not None:
            return None
        _write_resolved(cfg, choice)  # with what the table filled in
        gpu, card = _gpu()
    else:
        if (no_device := command.device_refusal()) is not None:
            contract.refused(no_device)
        import torch

        from poros.core import card
        from poros.core.errors import PorosError

        try:  # what this card cannot run as asked, said before the plan and any download
            card.check(cfg)
        except PorosError as exc:
            friendly_error(exc)
            return None
        if submit is not None:  # --detach: the job runs the fit check when its turn comes
            submit()
            return None
        if plan_gate is not None:
            plan_gate(cfg, mt)
        from poros.train.trainer import apply_expandable_segments

        # the allocator this process runs with, set up before config.resolved.json records it
        apply_expandable_segments(cfg, torch.device("cuda:0"))  # the trainer's device
        gpu, card = _gpu()
    if not mac:  # a CUDA run's own defaults (the simulator stands in for one): block-major.
        # After the plan gate, which plans the config as written and says when CUDA trains this.
        from poros.plan.configs import read_model_config

        said = cfg.path_reason  # a path the fit decided: its note is said already
        cfg.apply_cuda_defaults(read_model_config(cfg.model_name_or_path, cfg.model_revision)
                                or ({"model_type": mt} if mt else None))
        if cfg.path_reason and cfg.path_reason != said:  # decided here, for a reason: said once
            from poros.cli import views

            out(views.note(f"path: {cfg.path}", cfg.path_reason))
        _write_resolved(cfg, choice)
    from poros.core.config import train_path

    ran = train_path(cfg)
    from poros.cli.verbs import shared

    named = shared.path_choice({"path": ran, "store": cfg.store,  # fast+cache, as it was set
                                "store_options": cfg.store_options}) or ran
    path_word = "" if ran == "sequential" else f" · {named} path"
    if trust_remote_code:
        cfg.trust_remote_code = True
    return _Ready(choice, mt, support, sim, mac, gpu, card, path_word, lock)


def _block_word(cfg: Any, mt: Optional[str]) -> str:
    """The start box's block: a dense run's ("block 4 · "); none for a model that routes to
    poros.moe on CUDA, which streams per expert and has no block."""
    from poros.core.arch_registry import validated_arch_types

    return "" if mt in validated_arch_types("moe") else f"block {cfg.block_size or 4} · "


def _show_settings(cfg: Any, ready: "_Ready", *, config_path: str, checked: Optional[tuple],
                   resume_from: Optional[str], trust_remote_code: bool, force: bool,
                   wc: int) -> None:
    """What will train, on what, with which settings, before the load; and the disk check (8),
    once the download's size is known."""
    from poros.data import held_rows

    model = cfg.model_name_or_path
    choice, mac, gpu, card = ready.choice, ready.mac, ready.gpu, ready.card
    path_word = ready.path_word
    if c.in_session:
        # what trains on what, and one dim line: the settings were on the card just
        # confirmed, the run's folder comes at the end, each score in the live card's table
        short = model.rstrip("/").split("/")[-1]
        data = Path(str(cfg.dataset)).name if cfg.dataset else ""
        hint = unscored_hint(cfg, choice)
        if getattr(cfg, "eval_dataset", None):
            held = f"scored on {Path(str(cfg.eval_dataset)).name}"
        elif holds_rows_out(choice) and (n := held_rows(cfg)):
            held = f"{c.num(n)} rows held out"
        else:
            held = "every row trains, none held out" if hint else ""
        facts = [gpu, choice.label if choice is not None else "", held]
        out([c.titled(c.event("Train", short + (f" on {data}" if data else ""))),
             c.rail(S(" · ".join(f for f in facts if f), "dm"))]
            + hint_rows(hint, wc))
    else:
        out([c.welcome("poros", model, gpu if mac else f"{gpu} {c.gib(card):.1f} GiB",
                       max_width=wc), []])
        out(
            [
                c.event("Checked", *(checked or (Path(config_path).name, "config"))),
                c.rail(
                    S(
                        (
                            f"rank {cfg.lora_rank} · {cfg.max_seq_length} tokens" if mac else
                            f"{cfg.quantization} · {_block_word(cfg, ready.mt)}"
                            f"rank {cfg.lora_rank} · seq {cfg.max_seq_length}{path_word}"
                        ),
                        "dm",
                    )
                ),
                c.rail(S(recipe_note(choice, cfg), "dm")),
            ]
        )
    if not c.in_session:
        out(c.rail_path("output", cfg.output_dir, wc))
    if mac and cfg.store_path:
        out(c.rail_path("weights", cfg.store_path, wc))
    elif getattr(cfg, "store", "ram") != "ram":
        out(c.rail_path("store", f"{cfg.store} {cfg.store_path or ''}".strip(), wc))
    if resume_from:
        out(c.rail_path("resume", resume_from, wc))
    elif getattr(cfg, "init_adapter", None):
        out(c.rail_path("from", cfg.init_adapter, wc))
    if c.in_session:
        pass  # said in the line under the title
    elif getattr(cfg, "eval_dataset", None):
        out(c.rail_path("eval", cfg.eval_dataset, wc))
    elif holds_rows_out(choice) and (n := held_rows(cfg)):
        each = cfg.save_steps and getattr(cfg, "eval_checkpoints", True)
        out([c.rail(S("eval".ljust(10), "dm"), S(f"{n} rows held out", "tx"),
                    S("  scored at the start, each checkpoint and the end" if each
                      else "  scored before and after", "dm"))])
    elif hint := unscored_hint(cfg, choice):
        out([c.rail(S("eval".ljust(10), "dm"), S("none held out", "tx"),
                    S("  this loop trains every row", "dm")),
             *hint_rows(hint, wc, indent=10)])
    if trust_remote_code:
        out(
            [
                c.rail(
                    S("--trust-remote-code: ", "dm"),
                    S(model),
                    S(" may run its own Python on load", "dm"),
                )
            ]
        )
    size = _download_note(cfg)
    if not force:  # the download cannot fit the disk (8)
        _refuse_if(command.disk_refusal(size))
    if size and size[2]:  # --force: said, and the download goes ahead
        out([c.rail(S("--force: ", "dm"), S(size[2]))])
    if size:
        out([c.rail(S(size[0], "dm"))] + ([c.rail(S("✕ ", "e"), S(size[1]))] if size[1] else []))
    if not ready.mac:  # the checkpoints it keeps against the output disk (a Mac's: run_room)
        from poros.plan.configs import read_model_config

        room = _checkpoint_note(cfg, read_model_config(cfg.model_name_or_path,
                                                       cfg.model_revision))
        if room:
            out([c.rail(S("✕ ", "e"), S(room))])
    out([[]])


def holds_rows_out(choice: Any) -> bool:
    """Whether a run on ``choice``'s loop holds rows of its data out by ``eval_split``: a
    recipe's run of chats or text does (pairs say their own); the engine's own loop never does,
    and scores only a file of held-out rows it is given (``eval_dataset``)."""
    return choice is not None and choice.name != "classic" and choice.shape != "pairs"


def hint_rows(hint: str, wc: int, indent: int = 0) -> List[Line]:
    """``unscored_hint``'s words as rail rows that fit ``wc`` columns, ``indent`` in from the rail:
    wrapped between words, never cut at the edge."""
    if not hint:
        return []
    pad = [S(" " * indent, None)] if indent else []
    room = max(1, wc - len(c.RAIL) - indent)
    return [c.rail(*pad, S(row, "dm")) for row in c.wrap(hint, room)]


def unscored_hint(cfg: Any, choice: Any) -> str:
    """For a run on the engine's own loop of a local file, with no file of held-out rows: how it
    would score some. "" where that does not apply (a recipe's run, a held-out file given, a
    test run on synthetic rows, data that is not a local file)."""
    if choice is None or choice.name != "classic" or getattr(cfg, "eval_dataset", None):
        return ""
    path = Path(str(cfg.dataset or "")).expanduser()
    if not path.is_file():
        return ""
    said = "to score held-out rows, give a file of them: eval_dataset, or --eval FILE"
    from poros import data

    try:
        text = path.suffix in (".jsonl", ".json", ".csv") and data.detect(path) == "text"
    except (OSError, ValueError):
        text = False
    if text:  # SFT trains text rows on every token too, and holds 5% of them out
        said += "; or recipe sft (--recipe sft), which holds 5% of 40 rows or more out"
    return said


def _load(cfg: Any, ready: "_Ready", *, shown_run: str, resume_from: Optional[str],
          friendly_error: Callable[[Exception], None],
          plan_gate: Optional[Callable[[Any, Optional[str]], None]]) -> Optional[tuple]:
    """A recipe's data, read before the load (a row's fault is said in seconds), then the model
    loaded: (run id, the screen's state, the recipe's data, the trainer, the live view already
    up or None), or None where an error was reported. In the session the run's card is up from
    the load on, the load its first state; elsewhere the load is one line, then "Loaded"."""
    model = cfg.model_name_or_path
    choice, sim, mac = ready.choice, ready.sim, ready.mac
    from poros.cli.ui.train_view import new_run_id

    run_id = new_run_id()
    tab = getattr(term.get(), "tab", None)
    if tab is not None:  # the session's tab knows which run it trains
        tab.run = Path(cfg.output_dir).name
    st = TrainState(model=model, gpu=ready.gpu, card_gb=ready.card, output_dir=cfg.output_dir,
                    run=shown_run or run_id, one_row=cfg.gradient_accumulation_steps == 1)

    # ---- a recipe's data, read before the load: a row's fault is said in seconds -------------
    built = None
    if not sim and not mac and choice.name != "classic":  # a Mac's run keeps its own tokenizer
        try:
            from poros.train.run import _load_tokenizer

            tok = _load_tokenizer(cfg)
            built = (tok, *recipe_data(cfg, choice, tok))
        except Exception as exc:
            friendly_error(exc)
            return None
        _refuse_if(command.data_resume_refusal(resume_from, built[1]))  # the same rows, or none

    # ---- load ----------------------------------------------------------
    from poros.cli.ui import simulate

    short = model.rstrip("/").split("/")[-1]
    doing = f"{'Preparing' if mac else 'Loading'} {short}"
    view = None
    try:
        if c.in_session:  # inside the try: a fault or ctrl-c here reads as one in the load
            view = _card_from_the_load(cfg, ready, st, doing, resume_from)
            view.start()  # once held here, so an exit from now on takes it off
        if sim:
            PorosTrainer = simulate.SimTrainer
        else:
            from poros.train.trainer import PorosTrainer

        def load() -> Any:
            return PorosTrainer.from_pretrained(
                model,
                quantization=cfg.quantization,
                lora_rank=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                lora_target_modules=cfg.lora_target_modules,
                block_size=cfg.block_size,
                config=cfg,
                check_host_ram=plan_gate is None,  # the plan gate already checked it
                check_fit=plan_gate is None,  # and the GPU fit (or --force overrode it)
            )

        if view is not None:  # the card's activity line says the load and its clock
            began = time.monotonic()
            trainer = load()
            st.arch, st.load_s = _arch_label(trainer), time.monotonic() - began
            view.loaded()
        else:
            with term.get().activity(
                doing, statusline=[f"run {shown_run or run_id}", "ctrl-c to stop"],
            ) as act:
                trainer = load()
                st.arch, st.load_s = _arch_label(trainer), act.elapsed
                act.done([c.event("Loaded", short, c.duration(act.elapsed)),
                          c.rail(S(st.arch, "dm"))])
    except KeyboardInterrupt:
        _card_off(view)
        out([c.event("Stopped", "while loading", mark="○")])
        raise typer.Exit(code=130)
    except Exception as exc:
        _card_off(view)
        friendly_error(exc)
        return None
    except BaseException:  # an exit from inside the load
        _card_off(view)
        raise
    return run_id, st, built, trainer, view


def _card_off(view: Any) -> None:
    """The card up since the load, taken off on an exit before training: and the tab's label,
    which it set, back to the command's name, as when no card had been up."""
    if view is None:
        return
    try:
        view.close()
    finally:
        tab = getattr(term.get(), "tab", None)
        if tab is not None:
            tab.label = None


def _card_from_the_load(cfg: Any, ready: "_Ready", st: TrainState, doing: str,
                        resume_from: Optional[str]) -> Any:
    """The run's card for before the load, returned unstarted: the load and the wait for step 1
    are its states. It holds a row for each number the run will show, read off the config, so
    nothing moves as they come: the steps, the checkpoints' table where the base is scored before
    step 1, a pair run's accuracy and margin, and the stream where one runs (a Mac's layers come
    from one pool)."""
    from poros.cli.ui.train_view import TrainView
    from poros.data import held_rows

    choice = ready.choice
    st.label = doing
    st.total = max(cfg.max_steps or 0, 0)  # a run by epochs: when its data is read
    st.recipe = choice.label
    st.pairs = choice.shape == "pairs"
    # the engine's own loop holds no rows out and scores a given file only at its checkpoints
    # (the simulator's scores it before step 1); a recipe scores its held-out rows or file first
    given = bool(getattr(cfg, "eval_dataset", None))
    st.scored = not resume_from and (
        (choice.name != "classic" and (given or held_rows(cfg) > 0))
        or (ready.sim and choice.name == "classic" and given))
    st.stream_soon = not ready.mac
    view = TrainView(st, live=term.get().live, under_blank=True)
    term.get().progress(lambda: (st.step, st.total))  # the tab's label, from the same frame
    return view


def _train_and_save(cfg: Any, ready: "_Ready", trainer: Any, st: TrainState, built: Any, *,
                    run_id: str, resume_from: Optional[str], should_stop, job_id: Optional[str],
                    resume_cmd: Optional[str], next_cmd: Optional[str], config_path: str,
                    emit_json: bool, friendly_error: Callable[[Exception], None],
                    w: int, next_unscored: Optional[tuple[str, str]] = None,
                    view: Any = None) -> Optional[dict]:
    """The live training surface, the adapter saved, and the summary and receipt. ``view``: the
    run's card, already up since the load (the session's)."""
    came_up = view is not None  # the session's card, up since the load
    try:  # an exit before the training starts takes that card off
        from poros.cli.ui.train_view import TrainView, interrupt_lines, saved_checkpoint
        from poros.core.errors import PorosOOMError, PorosStepCheckError

        model = cfg.model_name_or_path
        choice, sim, mt, support = ready.choice, ready.sim, ready.mt, ready.support
        gpu, card = ready.gpu, ready.card
        # ---- train ---------------------------------------------------------
        notice = getattr(getattr(trainer, "optimizer", None), "notice", None)
        if notice:  # the optimizer's slow path, said before the first step and in the records
            from poros.cli import views

            # over the card where it is up: a line printed under a live area is drawn over
            (view.print if view is not None else out)(views.note("optimizer: reference", notice))
            _write_resolved(cfg, choice)
        fresh = view is None
        if fresh:
            view = TrainView(st, live=term.get().live, under_blank=c.in_session)
        term.get().progress(lambda: (st.step, st.total))
        trainer.reporter = view
        st.recipe = choice.label
        t0 = time.monotonic()
        if fresh:
            view.start()
    except BaseException:
        if came_up:
            _card_off(view)
        raise
    try:
        if sim:
            trainer.recipe = choice.name
        if choice.name == "classic" or sim:
            metrics = trainer.train(
                cfg.dataset,
                steps=cfg.max_steps if cfg.max_steps > 0 else None,
                seq_len=cfg.max_seq_length,
                # the file of held-out rows it scores; the engine's loop holds none out itself
                eval_dataset=cfg.eval_dataset or None,
                resume_from=resume_from,
                should_stop=should_stop,
            )
            if not sim and hasattr(trainer, "_save_checkpoint"):
                trainer._save_checkpoint()  # the last step, as a recipe saves it: train more
        else:
            metrics = train_recipe(trainer, cfg, choice, resume_from, should_stop, built)
        if getattr(trainer, "ran_path", None):  # the loop its steps took, beside the setting
            _write_resolved(cfg, choice, ran_path=trainer.ran_path)
    except PorosCancelledError as exc:  # a job's stop request: checkpointed at a step boundary
        resume = f"poros jobs resume {job_id}" if job_id else resume_cmd
        done = st.step or exc.step + 1  # steps done: the error's own is the last one's index
        view.close(interrupt_lines(done, exc.checkpoint, config_path, w, resume))
        if contract.json_mode() and resume:
            from rich.markup import escape

            said = stop_said(done, exc.checkpoint, job_id)
            contract.emit_error(exc.code, escape(said), hint=escape(resume))
        elif contract.json_mode():
            friendly_error(exc)  # the error object carries the checkpoint
        raise typer.Exit(code=contract.EXIT_CODES[exc.code])
    except KeyboardInterrupt:
        # ctrl-c saves nothing: what the run resumes from is the newest checkpoint on the disk,
        # read now (the trainer's own word for it may be older than its last save)
        saved = saved_checkpoint(cfg.output_dir)
        view.close(interrupt_lines(st.step, saved[1] if saved else None, config_path, w,
                                   resume_cmd, held=saved[0] if saved else None))
        raise typer.Exit(code=130)
    except PorosOOMError as exc:
        view.close()
        _oom(exc, cfg, config_path, st, friendly_error)
        return None
    except PorosStepCheckError as exc:
        view.close()
        write_receipt(Path(cfg.output_dir) / "receipt.json", {
            "kind": "poros.train.receipt", "version": 1, "run": run_id, "model": model,
            "failed": exc.code, "witness": exc.witness,
        })
        friendly_error(exc)
        return None
    except Exception as exc:
        view.close()
        friendly_error(disk_error(exc, cfg))
        return None
    wall = time.monotonic() - t0
    if resume_from and not (metrics.get("loss_history") or st.losses):
        # a resume that found no step left (a run by epochs: known once its data is read)
        return _finished(cfg, st.step, st.total or st.step, config_path=config_path,
                         resume_cmd=resume_cmd, next_cmd=next_cmd, emit_json=emit_json, w=w,
                         view=view)

    # ---- save + receipt ------------------------------------------------
    # The live view stays up while the adapter is written (its activity line
    # becomes "Saving adapter"), then turns into the summary in one paint.
    st.label = "Saving adapter"
    try:
        if getattr(trainer, "_metal", False):  # as before a checkpoint: this is one more copy
            trainer._prune_checkpoints(room_for_one=True)
        trainer.save(cfg.output_dir, push_to_hub=cfg.push_to_hub, hub_model_id=cfg.hub_model_id)
    except Exception as exc:
        view.close()
        friendly_error(disk_error(exc, cfg, writing=True))
        return None
    from poros.cli import train_readme

    # the folder's README: what the adapter is, in place of PEFT's unfilled template
    train_readme.write(cfg, choice.label, st.step or None, c.version())

    # `poros eval <run>` scores what the run held out, its --eval file, or a pairs run's own
    # pairs: a run with none of those is offered a step that works
    scored = (choice.shape == "pairs" or bool(cfg.eval_dataset)
              or metrics.get("eval_loss_start") is not None or metrics.get("eval_loss") is not None)
    note = ""
    if next_cmd and next_unscored and not scored:
        next_cmd, note = next_unscored
        note = contract.as_agent(note)  # its command, in the form this caller runs
    try:
        return _finish(view, st, cfg, metrics, wall, sim, run_id, model, gpu, card, config_path,
                next_cmd, emit_json, trainer,
                {"support": support, "model_type": mt} if support else {}, note)
    except Exception as exc:  # the live view goes first: a bug's report never lands inside it
        view.close()
        friendly_error(disk_error(exc, cfg))


def _assert_family(model: str, mt: Optional[str], trust_remote_code: bool) -> None:
    """--unvalidated: the model's own type runs as user_asserted for this run, as
    POROS_USER_ASSERTED_ARCHS would make it, after the refusals no flag gets past
    (``poros.preflight.command.unvalidated_refusal``)."""
    from poros.cli.context import Context
    from poros.cli.verbs import arch
    from poros.core.arch_registry import assert_this_command, validated_arch_types

    _refuse_if(command.unvalidated_refusal(model, mt, trust_remote_code,
                                           lambda m: arch.reading(m, Context())))
    if mt in validated_arch_types() | validated_arch_types("moe"):
        return  # validated: nothing to assert, and nothing is downgraded
    assert_this_command(mt)


def disk_error(exc: Exception, cfg: Any, *, writing: bool = False) -> Exception:
    """A pulled drive or a full disk on a Mac run, whichever write found out first, or ``exc`` as it
    was. torch.save tells of a full disk only in a RuntimeError's text."""
    if sys.platform != "darwin" or not isinstance(exc, (OSError, RuntimeError)):
        return exc
    if not Path(cfg.output_dir).exists():
        return PorosDiskError(f"{cfg.output_dir} is gone, and the run with it.",
                              remediation="Attach its disk again, then `poros runs continue`.")
    if writing or (isinstance(exc, OSError) and exc.errno == errno.ENOSPC):
        return PorosDiskError(f"Could not write to {cfg.output_dir}: the disk is full.",
                              remediation="Free some space, then `poros runs continue`.")
    return exc


def recipe_note(choice: Any, cfg: Any) -> str:
    """The rail under Checked: what trains, on what."""
    from poros.cli.views import recipe_text

    return recipe_text(choice.label, choice.shape,
                       {"train_on": cfg.train_on, "dpo_beta": cfg.dpo_beta},
                       lr=cfg.learning_rate)


def _write_resolved(cfg: Any, choice: Any, ran_path: Optional[str] = None) -> None:
    """``config.resolved.json`` in the run's folder: every setting it trains
    at, after the recipe's presets, and the loop that trains it. warmup_steps is
    the setting; the schedule clamps it to the run, and that is
    warmup_steps_effective (null: a run by epochs, whose length is not known here).
    ``path`` is the loop setting, with ``path_reason`` where it was decided sequential
    (``recorded_config``); once the run has trained, ``ran_path`` is the loop its steps took."""
    from poros.core.config import recorded_config

    out = Path(cfg.output_dir)
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # what output_refusal could not see first: a full disk
        from poros.preflight.disk import output_error

        contract.refused(command.unwritable_output(output_error(exc, cfg.output_dir)))
    ramps = cfg.lr_scheduler in ("cosine", "linear")
    body = {"trained_with": choice.name, "data_shape": choice.shape,
            **recorded_config(cfg, mode="json"),
            "warmup_steps_effective": (cfg.warmup_for(cfg.max_steps) if ramps else 0)
            if cfg.max_steps > 0 else None,
            **({"ran_path": ran_path} if ran_path else {})}
    (out / "config.resolved.json").write_text(json.dumps(body, indent=2) + "\n")



def stop_said(step: int, checkpoint: str, job_id: str = "") -> str:
    """A stopped run in a sentence. An agent's names a checkpoint as its listing does: a recipe
    counts its folders from 0, so checkpoint-21 holds 22 steps and is poros://<run>/weights/22."""
    from poros.core.env import under_agent

    held = f"poros://{job_id}/weights/{step}" if under_agent() and job_id else checkpoint
    return f"stopped after step {step}; {held} holds the run so far"


def card_curve(g: int, losses: list, final: float) -> tuple:
    """The card's curve, its last number and the steps a point stands for. One
    row a step reads by the mean of several; the receipt keeps the last row's."""
    if g != 1 or not losses:
        return losses, final, 1
    points = c.row_means(losses)
    if not points:
        return [], math.fsum(losses) / len(losses), 1
    return [v for _, v in points], points[-1][1], points[0][0]


def _finish(view, st, cfg, metrics, wall, sim, run_id, model, gpu, card, config_path,
            next_cmd, emit_json, trainer, support: Optional[dict] = None,
            next_note: str = "") -> dict:
    """The receipt, then the summary painted over the live view. ``support``: a run on its own
    assertion says so, its receipt and its result. ``next_note``: said under the next step."""
    from poros.train.reporting import host_ram_peak_gb

    mac = sys.platform == "darwin" and not sim
    if mac:  # one pool: the process's own peak, not the allocator's view of it
        metrics = {**metrics, "peak_vram_allocated_gb": st.vram_peak_gb}
    peak_ram = None if mac else max(filter(None, [view.peak_ram, host_ram_peak_gb()]), default=None)
    losses = list(metrics.get("loss_history") or st.losses)
    receipt_path = write_receipt(
        Path(cfg.output_dir) / "receipt.json",
        {
            "kind": "poros.train.receipt",
            "simulated": sim,
            "version": 1,
            "poros": c.version(),
            "run": run_id,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "model": model,
            "arch": st.arch,
            "recipe": st.recipe,
            "gpu": {"name": gpu, "total_gb": round(card, 2)},
            "config": {
                k: getattr(cfg, k, None)
                for k in (
                    "quantization",
                    "block_size",
                    "lora_rank",
                    "lora_alpha",
                    "max_seq_length",
                    "per_device_batch_size",
                    "gradient_accumulation_steps",
                    "max_steps",
                    "learning_rate",
                    "dataset",
                )
            },
            "steps": metrics.get("total_steps"),
            "final_loss": metrics.get("final_loss"),
            "wall_s": round(wall, 2),
            "s_per_step": round(metrics.get("avg_step_time_ms", 0.0) / 1000.0, 4),
            "tokens_per_sec": round(metrics.get("tokens_per_sec", 0.0), 2),
            "peak_vram_allocated_gb": round(metrics.get("peak_vram_allocated_gb", 0.0), 3),
            "peak_vram_reserved_gb": round(metrics.get("peak_vram_reserved_gb", 0.0), 3),
            "peak_host_ram_gb": round(peak_ram, 3) if peak_ram else None,
            "output_dir": cfg.output_dir,
            **(support or {}),
            **(_mac_receipt(cfg, st, trainer) if mac else {}),
            **({} if mac or sim else _staging_receipt(trainer)),
        },
    )
    curve, final, every = card_curve(cfg.gradient_accumulation_steps, losses,
                                     float(metrics.get("final_loss", float("nan"))))
    result = TrainResult(
        steps=int(metrics.get("total_steps") or st.step),
        wall_s=wall,
        final_loss=final,
        first_loss=curve[0] if curve else None,
        losses=curve,
        every=every,
        peak_vram_gb=float(metrics.get("peak_vram_allocated_gb", 0.0)),
        card_gb=card,
        peak_ram_gb=peak_ram,
        ram_total_gb=st.ram_total_gb,
        output_dir=cfg.output_dir,
        receipt=receipt_path,
        tokens_per_sec=float(metrics.get("tokens_per_sec", 0.0)),
        next_cmd=next_cmd or (f"poros export {cfg.output_dir} --format adapter" if mac
                              else f"poros eval {config_path}"),
        # a run started from another's adapter is no recipe (recipes save refuses it): no offer
        save_cmd="poros recipes save last" if next_cmd and not cfg.init_adapter else "",
        next_note=next_note,
        extra=dict(st.extra),
        recipe=st.recipe,
        eval_start=metrics.get("eval_loss_start"),
        eval_end=metrics.get("eval_loss"),
        eval_note=metrics.get("eval_start_note") or "",
        ends=(metrics["eval_ends"], metrics["eval_answer_rows"]) if "eval_ends" in metrics
        else None,
    )
    view.finish(result, term.get().width())
    if contract.json_mode() and emit_json:
        from poros.cli.schemas import TrainOutput

        contract.emit(
            TrainOutput(
                output_dir=cfg.output_dir,
                events=str(Path(cfg.output_dir) / "events.jsonl"),
                metrics=metrics,
                **(support or {}),
            )
        )
    return metrics


def _staging_receipt(trainer: Any = None) -> dict:
    """How a CUDA run staged its host copies: page-locked, partly, or pageable where the host's
    page-locked budget said no (a WSL2 host whose child probe failed pins nothing: same bytes,
    synchronous copies), with that budget and its basis (``poros.sss.pinning.staging``); and on
    path efficient where its block inputs went after update 1 (``block_inputs``)."""
    import torch

    if not torch.cuda.is_available():
        return {}
    from poros.sss.pinning import staging

    held = getattr(trainer, "_boundary_offload_stats", None)
    return {"host_staging": staging(), **({"block_inputs": held} if held else {})}


def _mac_receipt(cfg: Any, st: TrainState, trainer: Any) -> dict:
    """What a Mac run adds: which machine, which build, which row of the table, what its per-step
    checks saw."""
    from poros import metal
    from poros.metal.table import class_of, pick
    from poros.metal.watchdog import spill_bytes, this_memory

    row = pick(cfg.model_name_or_path, class_of(st.gpu), seq=cfg.max_seq_length)
    grew = trainer._watchdog.swap_growth if trainer._watchdog else 0
    return {
        "witness": {"checked": trainer._witness.checked, "mismatches": trainer._witness.mismatches},
        "spilled": grew >= spill_bytes(this_memory()),
        "swap_growth_mb": round(grew / 1e6, 1),
        "lora_dropout": cfg.lora_dropout,
        "load_s": round(st.load_s, 2),
        "platform": sys.platform,
        "device_class": metal.device_class(),
        "memory_budget_gb": metal.dev_memory_gb(),  # a developer's budget: never a measurement
        "macos_build": metal.macos_build(),
        "tier": cfg.store,
        "table_row": row.seq if row and row.seq == cfg.max_seq_length else None,
        "table_build": row.macos_build or None if row else None,
    }


def _oom(exc: Any, cfg: Any, config_path: str, st: TrainState,
         friendly_error: Callable[[Exception], None]) -> None:
    if sys.platform == "darwin":  # the message says what to do; a block size would not help
        friendly_error(exc)
        return
    bs = getattr(exc, "block_size", None) or cfg.block_size or 4
    seq = getattr(exc, "seq_len", None) or cfg.max_seq_length
    fixes: List[Fix] = []
    if bs > 1:
        nb = max(1, bs // 2)
        fixes.append(
            Fix(
                f"Retry with --block-size {nb}",
                _argv_with(config_path, block_size=nb),
                "fewer layers resident, slower steps",
            )
        )
    if seq > 128:
        ns = max(128, seq // 2)
        fixes.append(
            Fix(
                f"Retry with --seq-len {ns}",
                _argv_with(config_path, seq_len=ns),
                "shorter examples",
            )
        )
    where = f"at step {st.step}" if st.step else "before the first step"
    reason = f"the GPU ran out of memory {where}"
    peak = f"{c.ceil1(c.gib(st.vram_peak_gb)):.1f} of {c.gib(st.card_gb):.1f} GiB"
    ctx = [f"block {bs} · seq {seq} · peak {peak}"]
    if contract.json_mode():  # the error object, never a menu
        friendly_error(exc)
    oom_code = contract.EXIT_CODES["E_OOM"]
    screen = error_screen("Out of memory", reason, term.get().width(), context=ctx,
                          fix=fixes[0] if fixes else None, docs="docs/limitations.md")
    out([[]] + screen)
    if not fixes or not (sys.stdin.isatty() and sys.stdout.isatty()):
        raise typer.Exit(code=oom_code)
    out([[]])
    pick = term.get().choose("Out of memory", [(f.label, f.note) for f in fixes])
    if pick is None:
        raise typer.Exit(code=oom_code)
    cmd = shlex.split(fixes[pick].command)
    out([c.event("Retrying", " ".join(cmd[1:]))])
    # A fresh process: the OOM'd allocator state and the half-built model
    # never leak into the retry. Inside the session it runs as a child, so
    # the session survives it.
    argv = [sys.executable, "-P", "-m", "poros.cli.main", *cmd[1:]]
    if c.in_session:
        code = subprocess.call(argv)
        raise typer.Exit(code=code)
    os.execvp(sys.executable, argv)
