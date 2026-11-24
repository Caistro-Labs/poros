"""export: turn a run into a model you can serve.

    poros export last
    poros export 7f3a --format adapter
    poros export last --format merged --output ./served
    poros export poros://7f3a/weights/best --format adapter

Flags:
    --format NAME     merged (a standalone model) or adapter (the LoRA weights)
    --output DIR, -o  where it goes (default: the run's folder, export/<format>/)
    --overwrite       replace what is there

A config-file run's folder:
    poros export ./run -o ./merged    merges the adapter in ./run into its base model
    --yes, -y                         replace what is in the output folder
    --trust-remote-code               for a base model that ships its own code

`merged` folds the adapter into the base, rounded once to bf16; `adapter` is
the LoRA weights alone, for a server that loads adapters (vLLM, TGI, PEFT).
The format does not change what training computed. A run exports its finished adapter;
poros://<run>/weights/<step> one of its checkpoints, and .../best the one with
the lowest held-out loss, into export/<format>-<step>/ by default.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import ClassVar, Literal, Optional

from pydantic import Field

from poros import runs
from poros.cli.action import Action, Draft, Source
from poros.cli.context import Context
from poros.cli.schemas import Choice, Output, Question
from poros.cli.ui.text import Line
from poros.cli.verbs import runs_report
from poros.cli.words import runs as run_words
from poros.cli.words.grammar import Sentence
from poros.core import checkpoints
from poros.core.env import under_agent
from poros.core.hub_files import portable_manifest
from poros.core.refusal import Refusal

FORMATS = ("merged", "adapter")
_REVISION = re.compile(r"[\w./-]{1,128}", re.ASCII)
_PLAIN_DIR = re.compile(r"/[\w ./+@-]{1,1023}")  # an absolute path of plain words, no quote or $
# The files that make an adapter loadable on its own (PEFT layout + Poros's record).
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
    "adapter_weights.pt",
    "poros_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


class ExportAction(Action):
    verb: ClassVar[str] = "export"
    run: str
    format: Literal["merged", "adapter"]
    output: Optional[str] = None
    overwrite: bool = False
    download: bool = False  # --download (or --yes): a merge may fetch its base, asked first
    step: Optional[str] = None  # a checkpoint's step; None: the finished adapter

    def canonical(self) -> list[str]:
        which = f"poros://{self.run}/weights/{self.step}" if self.step else self.run
        out = ["export", which, "--format", self.format]
        out += ["--output", self.output] if self.output else []
        return (out + (["--overwrite"] if self.overwrite else [])
                + (["--download"] if self.download else []))


class ExportOut(Output):
    schema_: Literal["poros.export.v1"] = Field("poros.export.v1", alias="schema")
    adapter: str
    output: str
    format: str  # merged | adapter (adds to the v1 values: safetensors = merged)
    run: Optional[str] = None
    fidelity: str = ""
    base_model: Optional[str] = None  # what the adapter goes on
    base_revision: Optional[str] = None  # the commit it trained on, if the run recorded one
    load: Optional[str] = None  # one line of python that loads it; None where none would work
    # why there is no load line, or what is left to set: a native adapter, a local base
    note: Optional[str] = None
    # a run trained with --unvalidated: user_asserted, and the model_type, as the run says
    support: Optional[str] = None
    model_type: Optional[str] = None


def _format_question() -> Question:
    return Question(
        id="format",
        prompt="Export as?",
        flag="--format",
        default="merged",
        remember="export",
        choices=[
            Choice(
                value="merged",
                label="A standalone model",
                note="adapter folded into the base, rounded once to bf16",
            ),
            Choice(
                value="adapter",
                label="The adapter alone",
                note="LoRA weights, for vLLM, TGI or PEFT",
            ),
        ],
    )


def resolve(s: Sentence, ctx: Context) -> Draft[ExportAction]:
    from poros.cli import questions
    from poros.cli.verbs import checkpoints as saved

    all_runs = runs.all_runs()
    ids = [r.id for r in all_runs]
    word, step = s.subject[0] if s.subject else "last", None
    if "/weights/" in word:
        word, step = saved.parse(word)
        step = None if step == saved.FINAL else step
    found = run_words.resolve(word, ids, newest=ids[0] if ids else None,
                              names=[(r.name, r.id) for r in all_runs])
    if isinstance(found, run_words.RunMiss) and (os.sep in word or word.startswith((".", "~"))):
        raise Refusal(  # a folder the form `poros export <folder>` would have read, missing
            "E_NOT_FOUND", "No folder", f"nothing at {word}",
            context=["export takes a run (its id, its name or last) or a run's folder"],
            fix="poros runs", note="lists them")
    if isinstance(found, run_words.RunMiss):
        raise Refusal(
            "E_NOT_FOUND",
            "No such run",
            found.reason,
            context=[", ".join(found.candidates)] if found.candidates else [],
            fix="poros runs",
            note="lists them",
        )
    if s.flags.pop("--merged", None):  # `poros agents export`'s own flag, read here as well
        if str(s.value("format") or "merged").lower() == "adapter":
            raise Refusal("E_USAGE", "One format", "--merged and --format adapter: say one")
        s.flags["--format"] = "merged"
    raw = s.value("format")
    fmt = str(raw).lower() if raw else None
    source = Source(kind="given", detail=f"as {fmt}") if fmt else None
    if ctx.unified and not _merges_here(next(r.model for r in all_runs if r.id == found)):
        if fmt is None:
            fmt, source = "adapter", Source(kind="default")
        elif fmt in ("merged", "safetensors", "model", "standalone"):
            raise Refusal(  # this run's adapter, named: `last` may be another run
                "E_EXPORT_UNSUPPORTED", "Can't merge yet", "a merged model is above 7B on a Mac",
                context=["a merged model is not in this release at this size"],
                fix=f"poros export {found} --format adapter", note="the adapter loads anywhere")
    if fmt is None:
        fmt = questions.preference("export") or None
        source = Source(kind="preference", detail="export") if fmt else None
    if fmt is None:
        return Draft(None, [_format_question()])
    if fmt in ("safetensors", "model", "standalone"):
        fmt = "merged"
    if fmt not in FORMATS:
        raise Refusal("E_USAGE", "Unknown format", f"{fmt} is not merged or adapter")
    if step is not None:  # best as the step it is, so the command says which
        step = saved.step_of(next(r for r in all_runs if r.id == found), step)
    a = ExportAction(
        run=found,
        format=fmt,
        output=str(said) if (said := s.value("output") or s.flags.get("-o")) else None,
        overwrite=bool(s.flags.get("--overwrite") or s.flags.get("--yes")),
        download=bool(s.flags.get("--download") or s.flags.get("--yes")),
        step=step,
    )
    a.given("run", word)
    a.provenance["format"] = source
    return Draft(a)


def _merges_here(model: str) -> bool:
    from poros.metal.export import merges

    return merges(model)


def _target(a: ExportAction, run: runs.Run) -> Path:
    """Where it goes."""
    if a.output:
        return Path(a.output).expanduser()
    # a Mac's run may keep its files on another disk than its folder, and a merge is the largest
    name = f"{a.format}-{a.step}" if a.step else a.format
    return Path(run.output_dir if sys.platform == "darwin" else run.folder) / "export" / name


def _weights(a: ExportAction, r: runs.Run) -> Path:
    """The folder it exports from: the run's finished adapter, or a checkpoint the listing finds."""
    if a.step is None:
        return Path(r.output_dir)
    from poros.cli.verbs import checkpoints as saved

    found = saved._folders(Path(r.output_dir)).get(int(a.step))
    if found is None:
        raise Refusal("E_NOT_FOUND", "No such checkpoint",
                      f"run {r.id} keeps none after step {a.step}",
                      fix=f"poros runs checkpoints {r.id}", note="lists them")
    return found


def confirmations(a: ExportAction, ctx: Context) -> list[Question]:
    run = next(r for r in runs.all_runs() if r.id == a.run)
    if a.step is None and run.state != "succeeded":
        raise Refusal(
            "E_USAGE",
            "Not finished",
            f"run {run.id} is {run.state}",
            context=["export a run that succeeded"],
            fix=f"poros runs {run.id}",
        )
    if a.step is not None and run.state in ("running", "queued", "stopping"):
        raise Refusal("E_USAGE", "Still running",  # it prunes its checkpoints as it goes
                      f"run {run.id} is {run.state}, and its checkpoints change until "
                      "it ends", fix=f"poros runs stop {run.id}",
                      note="stops it at a checkpoint")
    _weights(a, run)
    out = _target(a, run)
    _refuse_place(out, run)  # before anyone is asked to overwrite it
    asks = []
    if out.is_dir() and any(out.iterdir()) and not a.overwrite:
        from poros.cli.questions import confirmation

        asks.append(confirmation("overwrite", f"{out} is not empty; export overwrites it.",
                                 "--overwrite", "Overwrite it"))
    if a.format == "merged" and not a.download and not ctx.unified:
        # a merge reads every shard of its base: one not here is downloaded, and asked first
        from poros.cli.verbs import shared
        from poros.core.config import PorosConfig
        from poros.plan.configs import download_size

        base, revision = _base(run)
        size = download_size(PorosConfig(model_name_or_path=base, model_revision=revision,
                                         dataset="synthetic")).value
        what = shared.download_confirm(base, size, ctx)
        if what:
            asks.append(shared.download_question(base, what, "--download",
                                                 "Yes, download it and merge"))
    return asks


def _base(r: runs.Run) -> tuple[str, Optional[str]]:
    """The model a run trained on, and the commit it pinned: the run's record, as a merge reads it.
    A run's files can sit in a project: a revision is a plain word, or it is none."""
    files = Path(r.output_dir)
    said = (runs_report._read(files / "poros_config.json") or {}).get("model_revision") or (
        runs_report._read(files / "adapter_config.json") or {}).get("revision")
    return r.model, said if isinstance(said, str) and _REVISION.fullmatch(said) else None


def _hub(base: str) -> Optional[str]:
    """The Hub cache this machine holds the base in, when it is not the one a plain
    `from_pretrained` reads (a Mac's store on another disk, or a cache the user set), so the load
    line reads it there and fetches nothing. Only Poros's own paths, and only a plain one: a
    folder named like code or a quote is left out, and the base is fetched as before."""
    if Path(base).is_dir():
        return None
    from huggingface_hub import constants

    from poros.cli.context import weights_cached

    hubs = [constants.HF_HUB_CACHE]
    if sys.platform == "darwin":
        from poros.metal import store

        try:
            hubs = store.hubs()
        except Exception:  # no disk to read: the one this process has
            pass
    try:
        held = next((h for h in hubs if weights_cached(base, h)), None)
    except Exception:  # a base that is no Hub id
        return None
    if not held or not _PLAIN_DIR.fullmatch(held):
        return None
    default = Path("~/.cache/huggingface/hub").expanduser()
    return None if Path(held).expanduser() == default else held


def _load_line(fmt: str, out: Path, base: str, revision: Optional[str]) -> str:
    """Its one line of python, on the base it trained on, as the run trained. Every value
    quoted, so a path or a name is only ever a string."""
    if fmt == "merged":
        return ("from transformers import AutoModelForCausalLM; "
                f"model = AutoModelForCausalLM.from_pretrained({str(out)!r})")
    hub = _hub(base)
    at = (f", revision={revision!r}" if revision else "") + (
        f", cache_dir={hub!r}" if hub else "")
    if sys.platform == "darwin":  # no bitsandbytes here: a full base, so answers differ a little
        return ("from peft import PeftModel; from transformers import AutoModelForCausalLM as M; "
                f"model = PeftModel.from_pretrained(M.from_pretrained({base!r}{at}), {str(out)!r})")
    return ("import torch; from peft import PeftModel; "
            "from transformers import AutoModelForCausalLM as M, BitsAndBytesConfig as B; "
            f"model = PeftModel.from_pretrained(M.from_pretrained({base!r}{at}, device_map='auto', "
            "quantization_config=B(load_in_4bit=True, bnb_4bit_quant_type='nf4', "
            "bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)), "
            f"{str(out)!r})")


# What an exported adapter names as its base when the run's base was a folder on this machine:
# the folder's path never leaves with it (as push_to_hub refuses one), and no reader can mistake
# this for a model id.
LOCAL_BASE = "<the base model's Hub id>"


def local_base(base: str) -> bool:
    """The base is a folder here, by push_to_hub's rule: an absolute, ./, ../ or ~ path, or a
    folder that holds a model's config.json."""
    return bool(base) and (Path(base).is_absolute() or base.startswith(("./", "../", "~"))
                           or (Path(base) / "config.json").exists())


def _with_base(raw: bytes, key: str) -> bytes:
    """A JSON file's bytes with ``key`` set to LOCAL_BASE (absent keys stay absent)."""
    import json

    said = json.loads(raw)
    if key in said:
        said[key] = LOCAL_BASE
    return (json.dumps(said, indent=2) + "\n").encode()


def native_only(folder: Path) -> bool:
    """The adapter is the native backend's alone (a CUDA MoE run's adapter_weights.pt): no PEFT
    adapter_model file, so no PEFT or vLLM line loads it."""
    return ((folder / "adapter_weights.pt").is_file()
            and not any((folder / f).is_file()
                        for f in ("adapter_model.safetensors", "adapter_model.bin")))


def _refuse_place(out: Path, r: runs.Run) -> None:
    """Where an export cannot go: a file, a home folder, or the run's own folder, which it would
    copy onto itself."""
    if os.path.lexists(out) and not out.is_dir():
        raise Refusal("E_USAGE", "Not a folder", f"{out} is a file: --output names a folder")
    if not out.resolve().name:  # "/": a merge is staged beside its folder, and / has none
        raise Refusal("E_USAGE", "Not a folder of its own", f"{out} is the file system's root",
                      context=["name a folder of its own: --output ./adapter"])
    if checkpoints.holds_home(out):
        raise Refusal("E_USAGE", "A home folder", f"{out} holds your home folder",
                      context=["name a folder of its own: --output ./adapter"])
    if out.resolve() in (Path(r.output_dir).resolve(), Path(r.folder).resolve()):
        raise Refusal("E_USAGE", "The run's own folder", f"{out} is run {r.id}'s folder",
                      context=["name a folder of its own: --output ./adapter"])


def run(a: ExportAction, ctx: Context) -> ExportOut:
    r = next(x for x in runs.all_runs() if x.id == a.run)
    if r.unvalidated:  # the family its run asserted, for this export alone: a Mac's merge finds
        from poros.core.arch_registry import assert_this_command  # the base by the run's row

        assert_this_command(r.unvalidated)
    out = _target(a, r)
    if under_agent():
        out = Path(os.path.abspath(out))
    _refuse_place(out, r)
    began = time.time_ns()
    files = _weights(a, r)
    if a.format == "adapter":
        # the run's own poros_config.json holds its machine's paths: only its portable form goes
        copied = [f for f in ADAPTER_FILES if f != "poros_config.json"
                  and (files / f).is_file() and not (files / f).is_symlink()]
        if not any(f.startswith("adapter_") for f in copied):
            raise Refusal(
                "E_NOT_FOUND", "No adapter", f"{files} has no adapter files"
            )
        manifest = portable_manifest(files)  # an adapter gets passed around: no one's paths
        if manifest is None and (files / "poros_config.json").exists():  # before out is touched
            raise Refusal("E_CONFIG", "Unreadable record",
                          f"{files / 'poros_config.json'} cannot be read, so its portable form "
                          "cannot be written")
        out.mkdir(parents=True, exist_ok=True)
        local = local_base(r.model)
        if local and manifest:  # the folder's path stays here (push_to_hub's rule)
            manifest = _with_base(manifest, "model_name_or_path")
        writing = {*copied, *(["poros_config.json"] if manifest else [])}
        for name in ADAPTER_FILES:  # its own names alone: a user's links and files stay
            there = out / name
            if there.is_symlink() or (name not in writing and there.is_file()):
                there.unlink()  # a link is replaced, not followed; an earlier export's file goes
        for f in copied:
            if local and f == "adapter_config.json":  # PEFT names the folder it loaded from
                (out / f).write_bytes(_with_base((files / f).read_bytes(),
                                                 "base_model_name_or_path"))
                continue
            shutil.copyfile(files / f, out / f)  # as a new file's: copy2 would keep its old time
        if manifest:
            (out / "poros_config.json").write_bytes(manifest)
        checkpoints.follow_umask(out, began)
        fidelity = "the adapter exactly as trained, bit for bit"
    else:
        from poros.cli.commands import export as export_command

        export_command._EXPORT_RESULT_TO_CALLER = True
        try:
            export_command.export(
                adapter_dir=files,
                format="safetensors",
                output=str(out),
                trust_remote_code=False,
                yes=True,
                json_output=False,
            )
        finally:
            export_command._EXPORT_RESULT_TO_CALLER = False
        fidelity = "the trained adapter, merged: merging rounds the weights once to bf16"
    base, revision = _base(r)
    load: Optional[str] = _load_line(a.format, Path(os.path.abspath(out)), base, revision)
    note = None
    if a.format == "adapter" and local_base(r.model):
        base, load = LOCAL_BASE, None
        note = (f"its base was a folder on this machine, written as {LOCAL_BASE}: set "
                "base_model_name_or_path in adapter_config.json and model_name_or_path in "
                "poros_config.json to the base's Hub id before you share it")
    elif a.format == "adapter" and native_only(out):
        load = None
        note = (f"this adapter is Poros's own format (adapter_weights.pt): it loads with Poros "
                f"as a new run's start: poros.Run.create({base!r}, init_adapter={str(out)!r}), "
                f"or init_adapter: {out} in a train config. poros export {r.id} --format merged "
                "gives a model any server loads")
    return ExportOut(
        adapter=str(files), output=str(out), format=a.format, run=r.id, fidelity=fidelity,
        base_model=base, base_revision=revision, load=load, note=note,
        **({"support": "user_asserted", "model_type": r.unvalidated} if r.unvalidated else {}),
    )


def view(r: ExportOut, w: int) -> list[Line]:
    from poros.cli import views

    return views.exported(r, w)
