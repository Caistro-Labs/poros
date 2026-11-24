"""`poros init` and `poros train <config>`: a config scaffolded, trained
(the flow itself is ``poros.cli.train_cmd``; the gate before the download renders
``poros.preflight.fit``), and planned."""

from __future__ import annotations

import functools
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from poros.cli import contract
from poros.cli.contract import refuse
from poros.cli.errors import _friendly_error
from poros.cli.root import _JSON_OPTION, _load_yaml_config, _no_grants, _ui, app
from poros.core import checkpoints as ckpt_contract
from poros.preflight import arch as preflight_arch
from poros.preflight import command

if TYPE_CHECKING:  # torch-free at runtime; PorosConfig is imported per-command
    from poros.core.config import PorosConfig


def _starter_config_yaml(model: str, dataset: str, output: str) -> str:
    """The YAML `poros init` writes: three required fields, defaults explained."""
    import json

    # json.dumps produces valid double-quoted YAML scalars, so user-supplied
    # values can never break the file.
    q = json.dumps
    if sys.platform == "darwin":
        intro = ("# On a Mac, Poros picks the rest for the model. "
                 "Full reference: poros schema config")
        source = "an HF Hub id"
        knobs = "# lora_rank: 16           # higher = more adapter capacity, more memory\n"
    else:
        intro = """\
# Everything omitted uses the validated defaults (NF4, rank-16 LoRA,
# block size 4). Full reference: poros schema config"""
        source = "an HF Hub id or a local model directory"
        knobs = """\
# max_seq_length: 1024    # tokens per training example
# lora_rank: 16           # higher = more adapter capacity, more VRAM
# block_size: 4           # lower = less VRAM, slower steps
# save_steps: 50          # write checkpoint-<step> dirs during training;
#                         # resume later with: poros train ... --resume-from latest
# path: fast              # CUDA: fastest, more VRAM (default efficient: less VRAM than
#                         # fast; sequential: one micro-batch at a time, slower, and
#                         # can peak lower; same result)
"""
    return f"""\
# Poros starter config -- the three fields below are all Poros needs.
{intro}

# Base model: {source}.
model_name_or_path: {q(model)}

# Training data: a local .jsonl/.json/.csv file, an HF dataset id,
# or "synthetic" (built-in smoke-test data -- runs with no data at all).
dataset: {q(dataset)}

# Where the trained adapter is written.
output_dir: {q(output)}

# Common knobs (uncomment to customize):
# max_steps: 100          # -1 = one full pass over the dataset
{knobs}"""


# The model a first run starts on, everywhere Poros names one: `poros init`, `poros machine`'s
# next step, the agents' guide (poros.cli.verbs.shared.START_WITH).
STARTER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _free_name(path: Path) -> Path:
    """``path`` with the first number after its name that no file has: train-2.yaml."""
    n = 2
    while (free := path.with_name(f"{path.stem}-{n}{path.suffix}")).exists():
        n += 1
    return free


@app.command()
def init(
    path: Path = typer.Argument(
        Path("train.yaml"),
        help="Where to write the starter config.",
    ),
    model: str = typer.Option(
        STARTER_MODEL, "--model", "-m",
        help="Base model: a Hub id or a local model folder. The default is the small model a "
        "first run starts on; for chats, take an Instruct model.",
    ),
    dataset: str = typer.Option(
        "./my_data.jsonl", "--dataset", "-d",
        help="Training data: a local .jsonl/.json/.csv file, or an HF dataset id.",
    ),
    output: str = typer.Option(
        "./my-adapter", "--output", "-o",
        help="Directory the trained adapter is written to.",
    ),
    force: bool = typer.Option(
        False, "--force", help="Replace an existing file with a new starter config: nothing "
        "in the old one is kept."
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Write a ready-to-run starter training config.

    The file contains the three fields Poros needs (model, dataset, output);
    everything else uses the validated defaults. Works without torch -- run
    it right after pip install, then: poros train train.yaml
    """
    from poros.cli.ui import components as uc
    from poros.cli.ui.live import columns
    from poros.cli.ui.term import out
    from poros.cli.ui.text import S

    if path.exists() and not force:
        # the safe next step first: --force would replace a config someone edited with the
        # starter's own model, data and output
        refuse("E_USAGE", "Already exists", f"{path} already exists, and init leaves it as it is",
               context=[f"--force replaces it with a new starter config: nothing in {path.name} "
                        "is kept"],
               fix=f"poros init {_free_name(path)}", note="writes a new starter config beside it")
    if path.is_symlink():  # a link is replaced, whatever it points at
        path.unlink()

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_starter_config_yaml(model, dataset, output))
    if contract.json_mode():
        from poros.cli.schemas import InitOutput

        contract.emit(InitOutput(path=str(path), model=model, dataset=dataset, output_dir=output))
        return

    w = min(columns(), uc.CAP)
    lines = [uc.event("Wrote", str(path)), uc.kv("model", model, lw=9, vw=0)]
    missing = False
    if dataset.strip().lower() != "synthetic":
        ds_path = Path(dataset)
        looks_local = dataset.startswith((".", "/", "~")) or ds_path.suffix in (
            ".jsonl", ".json", ".csv",
        )
        missing = looks_local and not ds_path.expanduser().exists()
    lines.append(uc.kv("dataset", dataset, lw=9, vw=0, style="tx",
                       tail="not there yet" if missing else None))
    lines.append(uc.kv("output", output, lw=9, vw=0))
    if missing:
        lines.append(uc.rail())
        lines.append(uc.rail(S("put your data at ", "dm"), S(dataset),
                             S(": one JSON object per line,", "dm")))
        lines.append(uc.rail(S('with "text", instruction/output, or chat "messages" fields', "dm")))
    lines.append([])
    lines += uc.continue_with(f"poros train {path}", w)
    out(lines)


_STORE_HELP = ("Where MoE experts live: 'ram' (default), 'ssd:/path' (a .sss file on a "
               "local SSD, written on the first run and reused; it does not change what "
               "training computes), or "
               "a store an installed plugin package provides: 'name' or 'name:/path'.")


def _resolve_resume_from(resume_from: str, output_dir: str) -> Path:
    """Resolve --resume-from conveniences to a concrete checkpoint directory.

    "latest" (or "auto") picks the newest checkpoint-<step> under output_dir.
    A run directory that contains checkpoint-<step> subdirs but no adapter
    weights of its own also resolves to its newest checkpoint. Fails fast
    (before any model download) when nothing resumable exists.
    """

    def _newest(root: Path) -> Optional[Path]:
        # Adopt any complete checkpoint stranded under checkpoint-N.tmp/.old
        # by a kill inside the save's promote window, then enumerate through
        # the one shared completeness contract (poros.core.checkpoints,
        # torch-free): the trainer-written sentinel, or for legacy dirs the
        # adapter weights AND rng_state.pt -- weights alone resume from
        # step 0 with fresh optimizer moments, silently discarding progress.
        ckpt_contract.recover_checkpoint_dirs(root)
        complete = dict(ckpt_contract.list_checkpoints(root))
        # newest by step across runs is no run's newest: never guess (a run and the unstamped
        # checkpoints it resumed from are one)
        if clash := ckpt_contract.mixed_runs_clash(root):
            refuse("E_CONFIG", "More than one run", clash,
                   context=["pass the checkpoint to continue: --resume-from "
                            f"{root}/checkpoint-<step>"])
        skipped: list[str] = []
        for d in root.glob("checkpoint-*"):
            step = ckpt_contract.checkpoint_step(d)
            if step is not None and step not in complete:
                skipped.append(d.name)
        for step in sorted(complete, reverse=True):
            if skipped:
                from poros.cli import views

                _ui()[0].print(views.note(
                    f"Resuming from {complete[step].name}",
                    f"skipped incomplete {', '.join(sorted(skipped))} (interrupted mid-save)"))
            return complete[step]
        return None

    if resume_from.strip().lower() in ("latest", "auto"):
        newest = _newest(Path(output_dir))
        if newest is None:
            refuse("E_NOT_FOUND", "No checkpoint",
                   f"--resume-from latest found no complete checkpoint in {output_dir}",
                   context=["a run writes one every save_steps updates (500 unless set) and "
                            "one when it finishes",
                            "pass a checkpoint directory explicitly if it lives elsewhere"])
        return newest

    path = Path(resume_from)
    if not path.exists():
        refuse("E_NOT_FOUND", "Not found", f"--resume-from directory {resume_from} does not exist")
    if not path.is_dir():
        refuse("E_NOT_FOUND", "Not a checkpoint",
               f"--resume-from {resume_from} is a file, not a checkpoint folder",
               context=["name a checkpoint-<N> folder, the run's folder, or latest"])
    if path.is_dir() and not (
        (path / "adapter_config.json").exists()
        or (path / "adapter_weights.pt").exists()
    ):
        newest = _newest(path)
        if newest is not None:
            return newest
    return path


@app.command()
def train(
    config: Path = typer.Argument(
        ...,
        help="Path to a YAML training config.",
        exists=True,
        readable=True,
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Override output directory."
    ),
    steps: Optional[int] = typer.Option(
        None, "--steps", "-s", help="Override max_steps."
    ),
    resume_from: Optional[str] = typer.Option(
        None, "--resume-from", "-r",
        help="Resume from a checkpoint directory written by a prior run "
        "(restores adapter weights, optimizer, RNG state, and step counter). "
        "Pass 'latest' to auto-pick the newest checkpoint in output_dir.",
    ),
    trust_remote_code: bool = typer.Option(
        False, "--trust-remote-code",
        help="Allow the model repo to execute its own Python on load. OFF by "
        "default: only enable it for a base you trust that ships custom "
        "modeling code. Poros's validated architectures never need it.",
    ),
    block_size: Optional[int] = typer.Option(
        None, "--block-size", help="Override block_size (lower = less VRAM, slower steps)."
    ),
    seq_len: Optional[int] = typer.Option(
        None, "--seq-len", help="Override max_seq_length (tokens per example)."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite",
        help="Train into an output_dir that holds a trained adapter: this run overwrites it.",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="The same as --overwrite: its name before."
    ),
    force: bool = typer.Option(
        False, "--force", help="Train even if the check before the download refuses it, or "
        "the download is larger than the free disk."
    ),
    store: Optional[str] = typer.Option(
        None, "--store", help=_STORE_HELP,
    ),
    sss_ram: Optional[str] = typer.Option(
        None, "--sss-ram", help="Cap on the pinned SSD read ring, e.g. 4G (default 4G).",
    ),
    detach: bool = typer.Option(
        False, "--detach", "-d",
        help="Run as a background job and print its id; see `poros jobs`.",
    ),
    stop_file: Optional[Path] = typer.Option(None, "--stop-file", hidden=True),
    allow_unvalidated: bool = typer.Option(False, "--allow-unvalidated", hidden=True),
    unvalidated: bool = typer.Option(
        False, "--unvalidated",
        help="Train a model whose family Poros has not validated: this run only, with no parity "
             "guarantee, and every result about it says so.",
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Train from a YAML config file.

    Reads the config, checks it, and trains. Requires torch.

    New here? Run 'poros init' to write a ready-to-run starter config,
    or 'poros schema config' to see every field.
    """
    from poros.core.config import PorosConfig

    # Validate the config BEFORE the torch gate: a YAML mistake should be
    # reported as such even on a box without torch (a light install).
    raw = _load_yaml_config(config)
    _no_grants(raw, config, flagged=trust_remote_code)
    if output_dir is not None:
        raw["output_dir"] = output_dir
    if steps is not None:
        raw["max_steps"] = steps
    _apply_store_flags(raw, store, sss_ram)
    if block_size is not None:
        raw["block_size"] = block_size
    if seq_len is not None:
        raw["max_seq_length"] = seq_len
    # allow_unvalidated: accepted, and nothing to do here. Every Apple silicon Mac trains on rows
    # derived for it (poros.metal.table), so no Mac needs an override any more.

    try:
        cfg = PorosConfig(**raw)
    except Exception as exc:
        _friendly_error(exc)
        return  # unreachable, _friendly_error raises Exit

    from poros.cli import train_cmd

    if detach and resume_from is not None:
        refuse("E_USAGE", "Can't run that",
               "--detach starts a new job; poros jobs resume <id> continues one.")
    from poros.preflight.disk import output_over_start

    # output_dir on the adapter it starts from (init_adapter): before the overwrite is asked
    if (over := output_over_start(cfg.output_dir, cfg.init_adapter, resume_from=resume_from)):
        contract.refused(over)
    if (not detach and resume_from is None and ckpt_contract.has_weights(Path(cfg.output_dir))
            # a folder that also holds a run's checkpoints is refused whatever is agreed to
            # (the flow's fresh-run check says so): nobody is asked a yes that changes nothing
            and command.fresh_run_refusal(None, cfg) is None):
        _ask_to_overwrite(cfg.output_dir, overwrite or yes)
    # the flow asks at its device step, after the output folder and the data: the GPU first,
    # then torch (poros.preflight.command.device_refusal); on a Mac torch alone (train_cmd)
    if resume_from is not None:
        resume_from = str(_resolve_resume_from(resume_from, cfg.output_dir))

    def submit() -> None:
        # The fit check runs in the job, against the GPU's free memory when
        # its turn comes: now, another job may be holding it.
        from poros.cli.commands.jobs import _jobs

        jobs = _jobs()
        if trust_remote_code:  # the job's runner reads its config, not this flag
            raw["trust_remote_code"] = True
        if cfg.store_path:  # a Mac's, as the store rule settled it
            raw["store_path"] = cfg.store_path
        family = (preflight_arch.resolve_model_type(cfg.model_name_or_path)[0] if unvalidated
                  else None)
        job = jobs.submit(raw, model=cfg.model_name_or_path, gpu=jobs.gpu_uuid(), force=force,
                          unvalidated=family)
        if contract.json_mode():
            contract.emit(job)
            return
        from poros.cli import views

        t, _, w = _ui()
        t.print(views.job_submitted(job.job_id, job.state, job.output_dir, w))

    from poros.core.arch_registry import one_command

    with one_command():  # what --unvalidated asserts ends with this run
        train_cmd.run(cfg, config_path=str(config), resume_from=resume_from,
                      trust_remote_code=trust_remote_code,
                      friendly_error=_friendly_error,
                      plan_gate=lambda c, mt: _plan_gate(c, mt, force),
                      submit=submit if detach else None,
                      should_stop=stop_file.exists if stop_file is not None else None,
                      job_id=stop_file.parent.name if stop_file is not None else None,
                      unvalidated=unvalidated, force=force)


def _ask_to_overwrite(output_dir: str, agreed: bool, said: str = "",
                      verb: str = "train") -> None:
    """A second run into a folder that holds a trained adapter, asked as `poros train <model>
    --output` asks it: a person at a terminal picks, and anyone else gets the question as data
    (``poros.needs_input.v1``, id ``overwrite``) whose one choice is this command with
    ``--overwrite``. ``--yes`` says the same, as it did before the two forms shared a flag.
    ``said`` and ``verb``: another command's question of its own folder (`poros export DIR`)."""
    import shlex

    from rich.markup import escape

    from poros.cli import questions

    if agreed:
        return
    said = said or f"{output_dir} already holds a trained adapter; this run overwrites it."
    if questions.interactive():
        contract.confirm(escape(said), False, flag="--overwrite")
        return
    argv = [a for a in (contract._argv or [verb, "<config.yaml>" if verb == "train" else "DIR"])
            if a != "--json"]
    again = shlex.join(["poros", *argv, "--overwrite",
                        *(["--json"] if contract.json_mode() else [])])
    questions.needs_input(verb, [questions.confirmation(
        "overwrite", said, "--overwrite", "Overwrite it", command=again)])


def _apply_store_flags(raw: dict, store: Optional[str], sss_ram: Optional[str]) -> None:
    """Fold --store / --sss-ram into a raw config dict (validated by PorosConfig)."""
    from poros.core.config import parse_store

    if store is not None:
        try:
            kind, path = parse_store(store)
        except ValueError as exc:
            refuse("E_USAGE", "Can't run that", str(exc))
        raw["store"] = kind
        if path is not None:
            raw["store_path"] = path
    if sss_ram is not None:
        raw["sss_ram"] = sss_ram


def _as_it_runs(cfg: "PorosConfig") -> "PorosConfig":
    """``cfg`` as the plan should read it: a recipe whose rows fit one micro-batch
    (``per_device_batch_size``) runs every step as that one micro-batch."""
    from poros import recipes
    from poros.data import train_rows

    try:
        recipe = recipes.choose(cfg).name != "classic"
    except ValueError:  # the run refuses it with its own words
        return cfg
    rows = train_rows(cfg) if recipe else None
    if rows is not None and rows <= cfg.per_device_batch_size:
        return cfg.model_copy(update={"gradient_accumulation_steps": 1})
    return cfg


LENGTH_NOTE = "planned at max_seq_length"  # a plan's note of a length its config did not set


def _at_data_length(cfg: "PorosConfig") -> tuple["PorosConfig", str]:
    """``cfg`` at the length its CUDA run reaches, and where that length is from, in words. A
    config that sets max_seq_length trains at it. One that sets none cuts no row before the
    default, and each batch is padded to its own longest row: its run reaches its data's longest
    row, read as train reads a data file (about 4 characters a token, each non-ASCII
    character a token, as a power of two from 256). Data that is not a file here (a Hub
    dataset, synthetic) plans at the default.
    The rule `poros train <config>`'s fit check reads."""
    seq = cfg.max_seq_length
    if "max_seq_length" in cfg.model_fields_set:
        return cfg, "max_seq_length in the config"
    scan = _scanned(str(cfg.dataset), _stamp(cfg.dataset)) if cfg.dataset else None
    if scan is None or not scan.chars:
        return cfg, ("the default: the config sets no max_seq_length, and its data's rows are "
                     "not read here. Set max_seq_length to plan the length you train at")
    whole = min(seq, scan.whole_len)
    said = (f"your data: the longest of {scan.rows:,} rows is about {scan.longest_tokens:,} "
            "tokens (an estimate from its characters; the tokenizer counts them in the run), "
            "and the config sets no max_seq_length, so no row is cut before "
            f"{seq:,}")
    if scan.seq_len < whole:
        said += (f". 95% of the rows are no longer than {scan.seq_len:,} tokens: "
                 f"max_seq_length: {scan.seq_len} trains and plans at that")
    # a copy that still reads as unset: the run's own length is the config's
    at = cfg.model_copy(update={"max_seq_length": whole})
    at.model_fields_set.discard("max_seq_length")
    return at, said


def _stamp(dataset: Optional[str]) -> tuple:
    """What changes when a data file does (where it is read from, its size and time), so a scan
    is read once a file."""
    try:
        st = Path(str(dataset)).expanduser().stat()
    except OSError:
        return (str(Path.cwd()),)
    return (str(Path.cwd()), st.st_size, st.st_mtime_ns)


@functools.lru_cache(maxsize=4)
def _scanned(dataset: str, stamp: tuple):
    from poros.cli.verbs import train_data

    return train_data.scan(dataset, Path.cwd())


def _plan_gate(cfg: "PorosConfig", model_type: Optional[str], force: bool) -> None:
    """Refuse (exit 3) a run that cannot fit the GPU or host RAM, before the download."""

    from poros.cli import views
    from poros.plan.configs import read_model_config
    from poros.preflight.fit import check, planner_here

    # the decisions PorosTrainer takes too (poros.preflight.fit.check)
    here = planner_here(model_type, read_model_config(cfg.model_name_or_path,
                                                      cfg.model_revision))
    # planned at the length the run reaches (_at_data_length); a Mac's
    # table sets its own
    mac = sys.platform == "darwin"
    fit = check(cfg, here if mac else (lambda c: here(_at_data_length(c)[0])),
                as_runs=_as_it_runs)
    p = fit.p
    from poros.cli.sheets.fit import mac_memory_note, plain, unfit_refusal

    # the decisions in words: the planner's figures stay inside the check
    if fit.path is not None:  # the path a CUDA run would take is short of host RAM or VRAM
        _ui()[0].print(views.note(f"path: {fit.path[0]}", plain(fit.path[1])))
    if fit.offload:  # the plan decided it as the run does: the run takes it
        _ui()[0].print(views.note("tables: host RAM", "to leave GPU memory for training"))
    if fit.warning:
        from poros.preflight.mac import SHORT

        said = mac_memory_note(p, cfg) if p.gpu is not None and p.gpu.unified else (
            "it may not fit the GPU at these settings, and could stop at its first step"
            + (f"; {fit.advice}, and block_size is never changed for you"
               if fit.advice else ""))
        if said:
            _ui()[0].print(views.note(SHORT if p.gpu and p.gpu.unified else "may not fit", said))
    unfit = unfit_refusal(fit)  # the trainer's refusal, in words without the planner's figures
    if unfit is not None:
        if not force:
            contract.refused(unfit)
        _ui()[0].print(views.note("--force", unfit.reason))  # the reason names the model
    err = fit.host
    if err is not None:
        if not force:
            contract.refused(unfit_refusal(fit, host=True))
        _ui()[0].print(views.note("--force", "not enough host RAM for this run"))
