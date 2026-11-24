"""Whether a checkpoint can resume into this run, decided before any model loads.

A resume continues a run on the data and the base it trained on, wherever they are now: a run
moved to another machine names both by other paths. So each checkpoint records what its run read
(INPUTS_RECORD: a local file or folder by content, anything else by what can be said of it), and
a resume compares that, not the location. What cannot be compared is said, once."""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, NamedTuple, Optional

logger = logging.getLogger(__name__)

# What a recipe run trained on, written beside each checkpoint (PorosTrainer._data_record).
DATA_RECORD = "data.json"
# What a run read, written beside each checkpoint (PorosTrainer._inputs): its dataset
# (``trained_on``) and its base model's folder (``base_identity``).
INPUTS_RECORD = "inputs.json"
INPUTS_VERSION = 1  # of the record, and of how a base's folder is sampled
# A weight shard is named by its size and this many evenly spaced pieces of this many bytes: about
# 16 MiB read of each shard, a second or so for 72B, where hashing every byte takes minutes. The
# pieces start 1/255 of the shard apart, so a rewritten stretch that long (16 MB of a 4 GB shard:
# any weight matrix of a fine-tuned or Instruct sibling) always meets one.
_PIECES, _PIECE = 256, 1 << 16
_READ = 1 << 24  # bytes of a dataset file hashed between two calls of a caller's ``progress``
# ``resume_verdict``'s ``dataset``: what this run trains on is what its config names (the CLI
# before the load), or its rows are the checkpoint's own (a recipe's rows, compared by the caller).
CONFIG, SAME_ROWS = "config", "same rows"
_SHARD_INDEX, _ONE_SHARD = "model.safetensors.index.json", "model.safetensors"

# The next step where a setting differs; a clash of the dataset or the base says its own.
SAME_SETTINGS = "resume with the settings the checkpoint trained with"
_NEW_RUN = "or start a new run in another output_dir"
_DATASET_FIX = ("set dataset to a copy of the data the checkpoint trained on (at any path: its "
                f"contents are compared, not where it is), {_NEW_RUN}")
_ROWS_FIX = ("resume with the data and settings (max_seq_length, train_on, the tokenizer) the "
             "checkpoint trained on, or start a new run")


def _base_fix(was: str) -> str:
    """The next step for another base: the one the checkpoint names, a Hub id or a folder."""
    return (f"set model_name_or_path to the base the checkpoint trained on ({was!r}), or to a "
            f"folder holding a copy of its files (at any path), {_NEW_RUN}")


def new_run_from(checkpoint: str) -> str:
    """The next step for someone who wants the other data or setting: not a resume (a resume is
    the same run), a new run that starts from the checkpoint's adapter, with a fresh optimizer
    and schedule. In each form's own words."""
    return ("to train its adapter on other data or settings, start a new run from it: poros train "
            f"<model> --from <run> --data <file>, or init_adapter: {checkpoint} with another "
            "output_dir (a config file, poros.Run.create, PorosTrainer.from_pretrained)")


# what a new run from the adapter may change: the rows and how they are read, not its tensors
_NEW_RUN_MAY = ("gradient_accumulation_steps", "per_device_batch_size", "seed", "dataset",
                "max_seq_length")


class Clash(str):
    """Why a checkpoint cannot resume into this run. ``fix`` is the next step in plain words, or
    None where SAME_SETTINGS says it. ``route`` is ``new_run_from``, where starting a new run
    from the checkpoint's adapter is a way to have what this run asked for."""

    fix: Optional[str]
    route: Optional[str]

    def __new__(cls, why: str, fix: Optional[str] = None, route: Optional[str] = None) -> "Clash":
        said = super().__new__(cls, why)
        said.fix, said.route = fix, route
        return said


def _targets(v: Any) -> Any:
    return None if v is None else sorted([v] if isinstance(v, str) else v)


def start_adapter(path: Any) -> str:
    """The checkpoint a new run starts from (``init_adapter``): ``path`` where it is a
    checkpoint; else, of the run whose folder it is, the later of its newest complete
    checkpoint and the adapter saved in the folder itself (a finished run's; the checkpoint
    where they are the same step, or the folder does not say its step). FileNotFoundError says
    what is missing."""
    from poros.core import checkpoints

    folder = Path(str(path)).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"init_adapter: no folder at {path}")
    if checkpoints.checkpoint_step(folder) is not None and checkpoints.is_complete(folder):
        return str(folder)
    found = checkpoints.list_checkpoints(folder)
    if _unfinished(folder) and not found:  # a run's folder cut short still has its checkpoints
        newest = checkpoints.list_checkpoints(folder.parent)
        raise FileNotFoundError(
            f"init_adapter: {path} is a save that did not finish (a staging folder, or one "
            "with no completion marker), so its adapter may be cut short; "
            + (f"the newest complete checkpoint beside it is {newest[-1][1]}" if newest
               else "no complete checkpoint is beside it"))
    itself = not _unfinished(folder) and any(
        (folder / name).exists()
        for name in ("poros_config.json", "adapter_config.json", "adapter_weights.pt"))
    if found and itself:
        try:
            at = int((folder / checkpoints.COMPLETE_MARKER).read_text().strip())
        except (OSError, ValueError):
            at = None
        return str(folder if at is not None and at > found[-1][0] else found[-1][1])
    if found:
        return str(found[-1][1])
    if itself:
        return str(folder)
    raise FileNotFoundError(f"init_adapter: {path} holds no adapter (no checkpoint-<step> "
                            "folder, no adapter files)")


def _unfinished(folder: Path) -> bool:
    """Whether ``folder`` is a save Poros began and did not finish: a staging folder
    (``checkpoint-N.tmp``, ``.old``), or one stamped as a save with no completion marker
    (``poros.core.checkpoints``). A folder Poros did not write (a PEFT adapter) is neither."""
    from poros.core import checkpoints

    name = folder.name.removeprefix("checkpoint-")
    staged = name != folder.name and name.endswith((".tmp", ".old"))
    return staged or ((folder / checkpoints.FORMAT_MARKER).exists()
                      and not (folder / checkpoints.COMPLETE_MARKER).exists())


# The LoRA settings a new run takes from the adapter it starts from, where it names none.
SHAPE = ("lora_rank", "lora_alpha", "lora_target_modules")
_NAME = re.compile(r"[\w.-]{1,200}", re.ASCII)  # a module's name, as a list of targets holds it


class Start(NamedTuple):
    """``start_from``'s answer: the checkpoint the run starts from (None: it names none), why it
    cannot (None: it can), and what the adapter trained with, as it was read and checked."""

    checkpoint: Optional[str]
    clash: Optional[Clash]
    said: dict


def start_from(cfg: Any, adopt: Any = SHAPE) -> Start:
    """The ``init_adapter`` check, one for every way a new run starts from an earlier adapter
    (``poros train``'s preflight, ``Run.create``, ``PorosTrainer``), torch-free, before the
    download:

    - the checkpoint ``cfg.init_adapter`` names (``start_adapter``; FileNotFoundError says
      what is missing);
    - what it trained with, read from its own files and checked before any of it is used (a
      folder is someone else's data: a rank that is no rank, targets that are no names, a file
      that does not read are refused, naming the file);
    - of ``adopt``, the LoRA settings ``cfg`` names none of, taken from it: its weights load
      only into the same shape, and act as trained only at the same alpha;
    - its base, revision, rank and target modules against ``cfg``'s (``start_clash``)."""
    named = getattr(cfg, "init_adapter", None)
    if not named:
        return Start(None, None, {})
    start = start_adapter(named)
    try:
        said = _said_by(start)
    except ValueError as exc:
        return Start(start, Clash(str(exc), "name an adapter whose records read, or start the "
                                            "run without init_adapter"), {})
    for name in adopt:
        if name not in cfg.model_fields_set and said.get(name) is not None:
            setattr(cfg, name, said[name])
    # Inkling 2 Mini: the experts the adapter trained against (released or nf4) are its base's
    # weights, as a revision is: a new run from it holds them the same way, unless it says
    if (adopt and "moe_expert_format" not in cfg.model_fields_set
            and said.get("moe_expert_format") is not None):
        cfg.moe_expert_format = said["moe_expert_format"]
    return Start(start, _start_clash(start, said, cfg), said)


def require_start(cfg: Any, adopt: Any = ()) -> Start:
    """``start_from`` for the Python API: a clash is a ``PorosConfigError`` with its next step."""
    from poros.core.errors import PorosConfigError

    start = start_from(cfg, adopt)
    if start.clash:
        fix = start.clash.fix or SAME_SETTINGS
        raise PorosConfigError(f"Can't start from this adapter: {start.clash}.",
                               remediation=f"{fix[0].upper()}{fix[1:]}.")
    return start


def start_clash(start: str, cfg: Any) -> Optional[Clash]:
    """Why the adapter at ``start`` cannot begin a run of ``cfg``: another base or another
    revision of it (the same tensors on other weights), another LoRA rank or set of target
    modules (its tensors do not load). Read from what it recorded, before any model loads; the
    data, the length and the batch shape are the new run's own."""
    try:
        return _start_clash(start, _said_by(start), cfg)
    except ValueError as exc:
        return Clash(str(exc), "name an adapter whose records read, or start the run without "
                               "init_adapter")


def _start_clash(start: str, saved: dict, cfg: Any) -> Optional[Clash]:
    base, _ = _base(start, saved, cfg, _recorded(start))
    if base:
        return Clash(str(base), "set model_name_or_path to the base the adapter trained on "
                                f"({saved.get('model_name_or_path')!r})")
    was, now = saved.get("model_revision"), getattr(cfg, "model_revision", None)
    named = str(getattr(cfg, "model_name_or_path", "") or "")
    # a revision is a Hub base's commit; a folder's contents are what ``_base`` compared
    if was and now and str(was) != str(now) and not Path(named).expanduser().is_dir():
        return Clash(f"the adapter at {start} trained on revision {str(was)[:12]} of its base, "
                     f"and this run pins {str(now)[:12]}: the same tensors on other weights",
                     f"set model_revision to {was!r}, or drop it")
    was, now = saved.get("moe_expert_format"), getattr(cfg, "moe_expert_format", None)
    if was and now and was != now:
        return Clash(f"the adapter at {start} trained against Inkling 2 Mini's experts held as "
                     f"{was}, and this run holds them as {now}: the same tensors on other "
                     "weights", f"set moe_expert_format to {was!r}, or drop it")
    for name, norm in (("lora_rank", int), ("lora_target_modules", _targets)):
        was, now = saved.get(name), getattr(cfg, name, None)
        if was is not None and now is not None and norm(was) != norm(now):
            return Clash(f"the adapter at {start} trained with {name} {was!r}, and this run has "
                         f"{now!r}: its weights load only into the same shape",
                         f"set {name} to {was!r}, or drop it to take the adapter's")
    return None


def _said_by(start: str) -> dict:
    """What the adapter at ``start`` trained with: its poros_config.json, or for a PEFT folder
    with none, what its adapter_config.json names (rank, alpha, base, revision). Checked, since
    a folder is data from wherever it came: ValueError, naming the file, for a file that is
    there and does not read as a JSON object, a ``lora_rank`` that is not a whole number within
    the config's bounds, a ``lora_alpha`` that is no such number, target modules that are not
    a name or a list of plain module names, a base or revision that is not text."""
    from poros.core.config import PorosConfig

    saved, where = _read(start, "poros_config.json"), Path(start) / "poros_config.json"
    if saved is None and where.exists():
        raise ValueError(f"{where} does not read as a JSON object")
    if saved is None:
        where = Path(start) / "adapter_config.json"
        peft = _read(start, "adapter_config.json")
        if peft is None and where.exists():
            raise ValueError(f"{where} does not read as a JSON object")
        named = {"lora_rank": "r", "lora_alpha": "lora_alpha",
                 "model_name_or_path": "base_model_name_or_path", "model_revision": "revision"}
        saved = {ours: (peft or {}).get(theirs) for ours, theirs in named.items()}
        saved = {k: v for k, v in saved.items() if v is not None}

    def bad(name: str, why: str) -> ValueError:
        return ValueError(f"{where} names {name} {repr(saved[name])[:60]}, which is {why}")

    for name in ("lora_rank", "lora_alpha"):
        value = saved.get(name)
        if value is None:
            continue
        whole = type(value) is int or (name == "lora_alpha" and type(value) is float
                                       and value.is_integer())
        try:
            if not whole:
                raise ValueError
            PorosConfig(**{name: int(value)})  # the config's own bounds
        except ValueError:
            raise bad(name, "no whole number a LoRA adapter takes (1 or more)") from None
    targets = saved.get("lora_target_modules")
    plain = (targets is None or (isinstance(targets, str) and 0 < len(targets) <= 200
                                 and targets.isprintable())
             or (isinstance(targets, list) and 0 < len(targets) <= 512
                 and all(isinstance(t, str) and _NAME.fullmatch(t) for t in targets)))
    if not plain:
        raise bad("lora_target_modules", "no module name and no list of module names")
    for name in ("model_name_or_path", "model_revision", "resolved_base_model_path"):
        if saved.get(name) is not None and not isinstance(saved[name], str):
            raise bad(name, "not text")
    if saved.get("moe_expert_format") not in (None, "released", "nf4"):
        raise bad("moe_expert_format", "neither released nor nf4")
    return saved


def data_record(data: Any) -> dict:
    """The record of a ``poros.data.Data`` a checkpoint keeps: its fingerprint, rows and kind."""
    return {"fingerprint": data.fingerprint, "rows": data.rows, "kind": data.kind}


def data_clash(checkpoint: str, data: Any) -> Optional[str]:
    """Why ``data`` is not what ``checkpoint`` trained on (its DATA_RECORD): another row list
    (other rows, rows dropped or added, other tokens) puts other rows behind each step, so the
    resumed run would repeat some rows and never train others. None where it is the same, or
    where the checkpoint keeps no record (one written before the record existed)."""
    return rows_clash(checkpoint, data_record(data))


def rows_clash(checkpoint: str, now: dict) -> Optional[Clash]:
    """``data_clash`` for a run that holds its rows' record (``data_record``)."""
    saved = _read(checkpoint, DATA_RECORD)
    if saved is None or saved.get("fingerprint") == now["fingerprint"]:
        return None
    return Clash(f"{checkpoint} trained on {saved.get('rows')} rows of {saved.get('kind')} "
                 f"(fingerprint {saved.get('fingerprint')}), and this run reads {now['rows']} "
                 f"rows of {now['kind']} (fingerprint {now['fingerprint']})", _ROWS_FIX,
                 new_run_from(checkpoint))


def rows_recorded(checkpoint: str) -> bool:
    """Whether ``checkpoint`` keeps a record of its rows (DATA_RECORD) to compare with."""
    return _read(checkpoint, DATA_RECORD) is not None


def trained_on(dataset: Any, progress: Any = None) -> Optional[dict]:
    """What a run trains on, as its checkpoints record it (INPUTS_RECORD's ``dataset``), taken
    when the run reads it:

    - a local file or folder, by content: ``dataset_identity``'s sha256 and size;
    - a Hub id or "synthetic", by name: ``{"name": ...}``;
    - rows handed over in memory (a Dataset, a DataLoader, any iterable), by what can be said
      without reading them: ``{"in_memory": the type's name, "length": len() where it has one}``.
      Reading them is not an option: a DataLoader's sampler draws from the run's random state,
      and an iterable can be endless;
    - None, for None: nothing is known (a loop of the caller's own hands the run its batches).

    ``progress(done, size)`` is called as a file is hashed (``dataset_identity``)."""
    if dataset is None:
        return None
    if isinstance(dataset, (str, os.PathLike)):
        return dataset_identity(dataset, progress) or {"name": str(dataset)}
    try:
        length = len(dataset)
    except TypeError:
        length = None
    return {"in_memory": type(dataset).__name__, "length": length}


def dataset_identity(spec: Any, progress: Any = None) -> Optional[dict]:
    """A local dataset by content: the sha256 and size of the file ``spec`` names, or of the files
    a folder trains on (each by its name and sha256). None for anything with no file to read: a Hub
    id, "synthetic", a dataset handed over in memory, a file that is gone. One read of the file
    (about 2 GB a second), kept while the file's size and time stay the same; ``progress(done,
    size)`` is called every ``_READ`` bytes of it, for a caller that shows the wait."""
    from poros.core.dataset_spec import folder_files, resolve_dataset_spec
    from poros.core.errors import PorosError

    if not isinstance(spec, (str, os.PathLike)):
        return None
    try:
        kind, _, found = resolve_dataset_spec(str(spec))
        if kind != "local":
            return None
        path = Path(found)
        if path.is_file():
            return {"sha256": _file_sha256(path, progress), "bytes": path.stat().st_size}
        files = sorted(Path(f) for f in folder_files(path)[1])
        each = "".join(f"{f.name}\0{_file_sha256(f, progress)}\n" for f in files)
        return {"sha256": hashlib.sha256(each.encode()).hexdigest(),
                "bytes": sum(f.stat().st_size for f in files)}
    except (OSError, PorosError):
        return None


def base_identity(folder: Any) -> Optional[dict]:
    """A base model's folder by content: config.json's bytes, and each safetensors shard's name,
    size and sampled bytes (``_PIECES`` evenly spaced pieces; all of a small shard). The same for
    a Hub cache snapshot and for a copy of its files anywhere; another model of the same shapes
    (an Instruct or fine-tuned sibling, another revision) differs. Sampled: a rewritten tensor at
    least 1/255 of its shard always shows, and a change confined to less (a few rows, a norm) can
    pass. None for a folder that is not a safetensors model's."""
    from poros.core.checkpoints import shard_name

    try:
        root = Path(folder).expanduser()
        if not (root / "config.json").is_file():
            return None
        if (root / _SHARD_INDEX).is_file():
            index = json.loads((root / _SHARD_INDEX).read_text())
            shards = sorted({shard_name(s) for s in index["weight_map"].values()})
        else:
            shards = [_ONE_SHARD]
        files = [root / name for name in ("config.json", *shards)]
        if not shards or not all(f.is_file() for f in files):
            return None
        seen = tuple((f.name, f.stat().st_size, f.stat().st_mtime_ns) for f in files)
        return {"fingerprint": _base_fingerprint(str(root.resolve()), seen),
                "files": len(shards), "bytes": sum(size for _, size, _ in seen[1:])}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def inputs_record(dataset: Optional[dict], base_folder: Any) -> dict:
    """What a checkpoint keeps of what its run read (INPUTS_RECORD): ``dataset``, what the run
    took of its data when it read it (``trained_on``), and the base folder's ``base_identity``;
    each where there is one."""
    said: dict = {"version": INPUTS_VERSION}
    for name, found in (("dataset", dataset),
                        ("base", base_identity(base_folder) if base_folder else None)):
        if found is not None:
            said[name] = found
    return said


def resume_verdict(checkpoint: str, cfg: Any, *, dataset: Any = CONFIG,
                   progress: Any = None) -> tuple[Optional[Clash], list[str]]:
    """(why not, what could not be checked) of resuming ``checkpoint`` into a run of ``cfg``,
    read from its poros_config.json and INPUTS_RECORD before any model loads.

    Why not, naming both values: another quantization, LoRA rank or set of target modules gives
    other tensors (the trainer's key and shape check would refuse it only after the load);
    another base model or base revision gives the same shapes on other weights; another
    gradient_accumulation_steps, per_device_batch_size or seed puts other rows behind each step
    (the resumed loop skips step x accumulation micro-batches of a stream ordered by the seed),
    so the run repeats some rows and never trains others; another dataset or max_seq_length gives
    the skipped and the next steps other rows or other tokens. A dataset file and a base folder
    are compared by content where the checkpoint recorded it, so the same data or model at
    another path resumes.

    What could not be checked, each in words that name both sides: a dataset or a base with
    nothing recorded to compare (an older checkpoint, rows handed over in memory, a loop of the
    caller's own). The run goes on, and its next checkpoint records what it read.

    ``dataset`` is what this run trains on: its ``trained_on`` record as the trainer took it
    (None: not known), CONFIG for what ``cfg.dataset`` names, SAME_ROWS where the caller compared
    a recipe's rows (``rows_clash``). ``progress`` goes to ``trained_on`` for CONFIG."""
    saved, recorded = _read(checkpoint, "poros_config.json"), _recorded(checkpoint)
    if saved is None:
        return None, []
    old = _old_update_rule(checkpoint) or _another_precision(checkpoint, saved, cfg)
    if old:
        return old, []

    clash, unverified = _base(checkpoint, saved, cfg, recorded)
    notes = [unverified] if unverified else []
    if clash:
        return clash, []
    for name, norm in (("quantization", str), ("lora_rank", int), ("lora_target_modules", _targets),
                       ("model_revision", str), ("gradient_accumulation_steps", int),
                       ("per_device_batch_size", int), ("seed", int),
                       ("dataset", None), ("max_seq_length", int),
                       ("moe_expert_format", str)):
        if norm is None:
            if dataset != SAME_ROWS:
                here = (trained_on(getattr(cfg, "dataset", None), progress)
                        if dataset == CONFIG else dataset)
                clash, unverified = _dataset(checkpoint, saved.get("dataset"),
                                             recorded.get("dataset"), here,
                                             getattr(cfg, "dataset", None))
                notes += [unverified] if unverified else []
        elif name not in saved or saved[name] is None or getattr(cfg, name, None) is None:
            continue
        elif norm(saved[name]) != norm(getattr(cfg, name)):
            clash = Clash(f"{checkpoint} trained with {name} {saved[name]!r}, and this run has "
                          f"{getattr(cfg, name)!r}")
        if clash:
            if name in _NEW_RUN_MAY:  # what a new run from its adapter may have
                clash.route = new_run_from(checkpoint)
            return clash, []
    return None, notes


def _old_update_rule(checkpoint: str) -> Optional[Clash]:
    """Why a checkpoint's optimizer state cannot be continued: an older Poros kept bf16 adapters'
    AdamW state in plain bf16 (``poros.train.expert_adamw``). Its adapter can start a new run."""
    from poros.core.checkpoints import (
        OLD_RULE,
        OLD_STATE_FIX,
        RULE,
        CheckpointUnreadable,
        checkpoint_record,
        checkpoint_rule,
        old_state_message,
    )

    try:
        rule = checkpoint_rule(checkpoint)
    except CheckpointUnreadable as exc:
        return Clash(str(exc), fix="resume from a checkpoint Poros wrote")
    if rule == OLD_RULE:
        try:
            experts = (checkpoint_record(checkpoint) or {}).get("experts")
        except CheckpointUnreadable:
            experts = None
        return Clash(old_state_message(checkpoint, experts), fix=OLD_STATE_FIX)
    if rule and rule.startswith("poros.expert-adamw.") and rule != RULE:
        return Clash(f"{checkpoint} was written by the update rule {rule}, a release candidate's, "
                     f"and this Poros updates with {RULE}: its rounding bits are drawn another "
                     "way, so it cannot continue that state as the same run", fix=OLD_STATE_FIX)
    return None


def _another_precision(checkpoint: str, saved: dict, cfg: Any) -> Optional[Clash]:
    """Why a MoE checkpoint cannot resume at another torch_dtype: its expert adapters are made in
    the model's dtype, so another one trains other tensors with another update rule (bf16 ones
    with ExpertAdamW, fp32 ones with torch's AdamW), and neither optimizer's state is the
    other's."""
    from poros.core.checkpoints import CheckpointUnreadable, checkpoint_record

    try:
        record = checkpoint_record(checkpoint) or {}
    except CheckpointUnreadable:
        return None  # _old_update_rule says why, first
    was, now = record.get("torch_dtype") or saved.get("torch_dtype"), getattr(cfg, "torch_dtype",
                                                                               None)
    if not record.get("experts") or not was or not now or was == now:
        return None
    return Clash(f"{checkpoint} trained its expert adapters at torch_dtype {was} (their tensors "
                 f"{record.get('trainable_dtype') or was}), and this run has torch_dtype {now}: "
                 "its adapters would be another precision, updated by another rule",
                 fix=f"resume with torch_dtype: {was}", route=new_run_from(checkpoint))


# What a resume takes from its checkpoint where the config leaves it unset: every setting
# ``resume_verdict`` compares by value, and the adapter's shape a start adopts (SHAPE). The
# dataset is compared by its contents and named by the config; the revision is
# ``adopt_revision``'s, with its own conditions.
RESUMED = ("quantization", "lora_rank", "lora_alpha", "lora_target_modules",
           "gradient_accumulation_steps", "per_device_batch_size", "seed", "max_seq_length",
           "moe_expert_format")


def adopt_recorded(checkpoint: Optional[str], cfg: Any) -> dict:
    """A resume continues its own run: of ``RESUMED``, each setting ``cfg`` was not given (not
    in its ``model_fields_set``) is the one the checkpoint's poros_config.json recorded, the
    run's own earlier resolution of it (an ``init_adapter``'s shape, a recipe's preset, a Mac's
    table, a default). A setting the config gives stays, and ``resume_verdict`` compares it.
    Called before anything else fills the config (presets, a Mac's table), so the same config
    file always resumes its own run, with its init_adapter folder there or not. A recorded
    value the config's own bounds refuse is not taken (the comparison then names it). Returns
    what it changed."""
    from poros.core.config import PorosConfig

    saved = _read(checkpoint, "poros_config.json") if checkpoint else None
    took: dict = {}
    for name in RESUMED if saved else ():
        if name in cfg.model_fields_set or saved.get(name) is None:
            continue
        try:
            value = getattr(PorosConfig(**{name: saved[name]}), name)  # the config's bounds
        except (ValueError, TypeError):
            continue
        if value != getattr(cfg, name):
            took[name] = value
        setattr(cfg, name, value)  # set, so a preset or a table fills nothing over it
    return took


def resume_clash(checkpoint: str, cfg: Any, **how: Any) -> Optional[Clash]:
    """``resume_verdict``'s why not: None where the checkpoint can resume into this run."""
    return resume_verdict(checkpoint, cfg, **how)[0]


def resume_unverified(checkpoint: str, cfg: Any, **how: Any) -> list[str]:
    """``resume_verdict``'s list of what a resume it allows could not check."""
    return resume_verdict(checkpoint, cfg, **how)[1]


_warned: set[str] = set()


def warn_once(said: str) -> None:
    """Log ``said`` as a warning, once in this process."""
    if said not in _warned:
        _warned.add(said)
        logger.warning(said)


def warn_unverified(checkpoint: str, cfg: Any, **how: Any) -> None:
    """Log ``resume_unverified`` as warnings, each once in this process (``poros train`` asks
    before the load and the trainer after it)."""
    for said in resume_unverified(checkpoint, cfg, **how):
        warn_once(said)


def same_base_contents(adapter: Any, folder: Any) -> Optional[bool]:
    """Whether the local ``folder`` holds the base the adapter in ``adapter`` (a checkpoint, a
    run's folder) trained on, by content: its INPUTS_RECORD against the folder's
    ``base_identity``, or for an adapter with no record, the base its poros_config.json names
    where that is another folder still on this machine. None where nothing can be compared. For
    a caller whose own mark of a local base moves when the base is only copied: a Mac's is the
    shards' sizes and times (``poros.metal.same_base``)."""
    here, kept = base_identity(folder), _recorded(str(adapter)).get("base")
    if kept is not None:
        return here is not None and here["fingerprint"] == kept.get("fingerprint")
    was = (_read(str(adapter), "poros_config.json") or {}).get("model_name_or_path")
    if here is None or not was or _norm_base(str(was)) == _norm_base(str(folder)):
        return None
    there = base_identity(was) if Path(str(was)).expanduser().is_dir() else None
    return None if there is None else here["fingerprint"] == there["fingerprint"]


def base_moved(adapter: Any, folder: Any) -> bool:
    """Whether ``folder`` is another place than the base the adapter's poros_config.json names."""
    was = (_read(str(adapter), "poros_config.json") or {}).get("model_name_or_path")
    return not was or _norm_base(str(was)) != _norm_base(str(folder))


def unchecked_base(adapter: Any, folder: Any) -> str:
    """The warning for a base ``same_base_contents`` could not compare, as a resume words it."""
    was = (_read(str(adapter), "poros_config.json") or {}).get("model_name_or_path")
    return _unchecked_base(str(adapter), str(was), str(folder))


def adopt_revision(checkpoint: Optional[str], cfg: Any) -> Optional[str]:
    """A resume whose config pins no model_revision takes the one its checkpoint trained on, so
    the download is that base, not the Hub's head today. Not for a base that could not be checked
    (``resume_unverified``): its next checkpoints would claim a revision nothing showed. Not for
    a run that loads the base's own code (``trust_remote_code``): it says so, once. Returns the
    revision it took, or None."""
    if not checkpoint or getattr(cfg, "model_revision", None):
        return None
    saved, recorded = _read(checkpoint, "poros_config.json"), _recorded(checkpoint)
    if saved is None:
        return None
    was = saved.get("model_revision")
    if not isinstance(was, str) or not was or any(_base(checkpoint, saved, cfg, recorded)):
        return None
    if getattr(cfg, "trust_remote_code", False):
        # the run loads the repo's own code: at which revision is its caller's to say, never a
        # checkpoint's (a folder from elsewhere could name a commit whose code does harm)
        warn_once(
            f"{checkpoint} trained on revision {was[:12]} of its base. This run loads the "
            "base's own code (trust_remote_code), so the revision is not taken from the "
            f"checkpoint. Set model_revision: {was} to resume on the code and weights it "
            "trained on.")
        return None
    cfg.model_revision = was
    return was


def snapshot_sha(path: Optional[str]) -> Optional[str]:
    """The commit a Hub cache snapshot folder is (``.../snapshots/<40 hex>``), or None."""
    parts = Path(path).parts if path else ()
    if "snapshots" in parts and parts.index("snapshots") + 1 < len(parts):
        sha = parts[parts.index("snapshots") + 1]
        if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
            return sha
    return None


def _dataset(checkpoint: str, was: Any, kept: Optional[dict], here: Optional[dict],
             name: Any) -> tuple[Optional[Clash], Optional[str]]:
    """(clash, unverified) of the dataset: what the checkpoint recorded (``kept``, a
    ``trained_on`` record; ``was``, the name its config says) against what this run trains on
    (``here``, its own ``trained_on`` record, or None where it is not known; ``name``, the name
    its config says).

    Two files are compared by content, whatever their paths; two names (a Hub id, "synthetic")
    as names; a file is never taken for a name. Rows in memory have no content to compare: other
    lengths are refused, and anything else is said, not passed in silence. So is a run whose
    data is not known at all, and a checkpoint that recorded nothing. An older checkpoint that
    recorded only a name: the same name is the same data, two files both here are compared, a
    file here standing in for one that is not (a moved run) goes on with a warning, and any
    other name is refused."""
    if here is None:
        return None, (
            f"Data that could not be checked: this run is handed its rows by a loop of its own, "
            f"so nothing shows they are the rows {checkpoint} trained on. If they are not, the "
            "resumed run repeats some rows and never trains others. A loop that names its "
            "data is checked: poros.Run.create(..., dataset=\"chats.jsonl\"), poros train, "
            "PorosTrainer.train on a file.")
    now = _named(here, name)
    before = ("data it did not record" if kept is None and was is None
              else _named(kept or {}, was))
    if "in_memory" in here or (kept is not None and "in_memory" in kept):
        if (kept is not None and "in_memory" in kept and "in_memory" in here
                and None not in (kept.get("length"), here["length"])
                and kept.get("length") != here["length"]):
            return Clash(f"{checkpoint} trained on {before}, and this run is handed {now}",
                         f"hand the run the rows the checkpoint trained on, {_NEW_RUN}"), None
        return None, (
            f"Data that could not be checked: {checkpoint} trained on {before}, and this run "
            f"trains on {now}. Rows handed over in memory have no contents on record to "
            "compare, so nothing shows they are the same rows. If they are not, the resumed run "
            "repeats some rows and never trains others. A dataset given as a file is checked.")
    if kept is None and was is None:
        return None, (
            f"Data that could not be checked: {checkpoint} did not record what it trained on, so "
            f"nothing shows that {now} is the same data. If it is not, stop this run: on other "
            "data it repeats some rows and never trains others. Its next checkpoint records "
            "what it reads.")
    another = Clash(f"{checkpoint} trained with dataset {before}, and this run has {now}",
                    f"set dataset to the one the checkpoint trained on, {_NEW_RUN}")
    if kept is not None and "sha256" in kept:
        if here.get("sha256") == kept["sha256"]:
            return None, None
        if "sha256" not in here:
            return Clash(f"{checkpoint} trained on the data at {before} (sha256 {_short(kept)}), "
                         f"and this run's dataset {now} is no file or folder here to compare "
                         "with it", _DATASET_FIX), None
        return Clash(f"{checkpoint} trained on the data at {before} (sha256 {_short(kept)}), and "
                     f"{now} holds other data (sha256 {_short(here)})", _DATASET_FIX), None
    if kept is not None:  # a name it recorded: a Hub id, "synthetic"
        return (None if here.get("name") == kept.get("name") else another), None
    # an older checkpoint: only the name its config says
    if _norm_dataset(was) == _norm_dataset(name):
        return None, None
    there = dataset_identity(was)
    if "sha256" in here and there is not None:  # both on this machine: compared
        if here["sha256"] == there["sha256"]:
            return None, None
        return Clash(f"{checkpoint} trained on the data at {was!r} (sha256 {_short(there)}), and "
                     f"{now} holds other data (sha256 {_short(here)})", _DATASET_FIX), None
    if "sha256" in here and _a_path(str(was)):
        return None, (
            f"Data that could not be checked: {checkpoint} trained on {was!r} and kept no record "
            "of its contents, and that file is not on this machine to compare, so nothing shows "
            f"that {now} is the same data. If it is not, stop this run: on other data it repeats "
            "some rows and never trains others. Its next checkpoint records the data's contents.")
    return another, None


def _named(record: dict, name: Any) -> str:
    """A ``trained_on`` record in a sentence: what rows in memory are, or ``name``, the file or
    Hub id as the config writes it."""
    if "in_memory" in record:
        length = record.get("length")
        return (f"rows handed over in memory (a {record['in_memory']}"
                f"{'' if length is None else f' of {length}'})")
    return repr(str(record.get("name", name)))


def _base(checkpoint: str, saved: dict, cfg: Any,
          recorded: dict) -> tuple[Optional[Clash], Optional[str]]:
    """(clash, unverified) of the base model. A folder this run names is compared by content with
    what the checkpoint recorded, whatever the paths (the folder it named itself, with no model
    left to read in it, goes by its name, as before). Where there is no record (an older
    checkpoint, a base that is no safetensors folder): the same name, or a Hub id and its
    snapshot folder in the cache, is the same base, as before; a folder named another way is
    compared with the checkpoint's own base if that is still on this machine, and otherwise goes
    on with a warning, since nothing is left to compare. A Hub id is never taken for another."""
    was, now = saved.get("model_name_or_path"), getattr(cfg, "model_name_or_path", None)
    if not was or not now:
        return None, None
    was, now = str(was), str(now)
    folder = Path(now).expanduser()
    here = base_identity(folder) if folder.is_dir() else None
    kept = recorded.get("base")
    other = Clash(f"{checkpoint} trained on base {was!r}, and the folder {now!r} does not hold "
                  "the same model: its config.json or its weights differ", _base_fix(was))
    renamed = _norm_base(was) != _norm_base(now)
    if kept is not None and folder.is_dir() and (here is not None or renamed):
        # another folder with no model to read in it is not that base
        same = here is not None and here["fingerprint"] == kept.get("fingerprint")
        return (None if same else other), None
    resolved = saved.get("resolved_base_model_path")
    if renamed:  # named another way: its own cache path says which
        revision, snapshot = saved.get("model_revision"), snapshot_sha(now)
        if folder.is_dir() and revision and snapshot and revision != snapshot:
            return Clash(f"{checkpoint} trained on base {was!r} at revision {str(revision)[:12]}, "
                         f"and {now!r} is the Hub cache's snapshot of revision {snapshot[:12]}",
                         _base_fix(was)), None
        if not any(a and b and _names(a, b) for a, b in ((now, resolved), (now, was), (was, now))):
            return _another_name(checkpoint, was, now, here, resolved, other)
    return None, None


def _another_name(checkpoint: str, was: str, now: str, here: Optional[dict], resolved: Any,
                  other: Clash) -> tuple[Optional[Clash], Optional[str]]:
    """``_base`` where no recorded contents decide, and this run's base is named another way and
    is not the Hub cache's snapshot of the checkpoint's."""
    if not Path(now).expanduser().is_dir():  # a Hub id, or a folder not here: only a name to go on
        return Clash(f"{checkpoint} trained on base {was!r}, and this run's base is {now!r}",
                     _base_fix(was)), None
    for old in (resolved, was):  # the checkpoint's own base, where it is still on this machine
        there = base_identity(old) if old and Path(str(old)).expanduser().is_dir() else None
        if here is not None and there is not None:
            return (None if here["fingerprint"] == there["fingerprint"] else other), None
    return None, _unchecked_base(checkpoint, was, now)


def _unchecked_base(checkpoint: str, was: str, now: str) -> str:
    return (
        f"A base that could not be checked: {checkpoint} trained on base {was!r} and kept no "
        "record of its contents, and no copy of it on this machine can be compared, so nothing "
        f"shows that the folder {now!r} holds the same model. If it does not, stop: another "
        "model of the same shapes (an Instruct or fine-tuned one) loads without an error, and "
        "the adapter is used on weights it was not trained on. A checkpoint written from here "
        "on records the folder's contents.")


def _read(checkpoint: str, name: str) -> Optional[dict]:
    """A JSON object a checkpoint keeps, or None where it keeps none that can be read."""
    try:
        said = json.loads((Path(checkpoint) / name).read_text())
    except (OSError, ValueError, RecursionError):  # nested past Python's stack: unread
        return None
    return said if isinstance(said, dict) else None


def _recorded(checkpoint: str) -> dict:
    """A checkpoint's INPUTS_RECORD: its ``dataset`` and ``base`` identities where it kept them, in
    a version this code reads. {} for an older checkpoint, or another version's record."""
    said = _read(checkpoint, INPUTS_RECORD) or {}
    if said.get("version") != INPUTS_VERSION:
        return {}
    return {k: v for k, v in said.items() if k in ("dataset", "base") and isinstance(v, dict)}


def _short(identity: dict) -> str:
    return str(identity.get("sha256"))[:12]


_sha256s: dict[tuple, str] = {}  # (path, size, time) -> sha256: one read of each file as seen


def _file_sha256(path: Path, progress: Any = None) -> str:
    """A file's sha256, read once for each (path, size, time) it is seen with, ``_READ`` bytes
    at a time; ``progress(done, size)`` after each."""
    seen = path.stat()
    key = (str(path.resolve()), seen.st_size, seen.st_mtime_ns)
    if key not in _sha256s:
        digest, done = hashlib.sha256(), 0
        with open(path, "rb") as f:
            while chunk := f.read(_READ):
                digest.update(chunk)
                done += len(chunk)
                if progress is not None:
                    progress(done, seen.st_size)
        _sha256s[key] = digest.hexdigest()
    return _sha256s[key]


@functools.lru_cache(maxsize=16)
def _base_fingerprint(root: str, seen: tuple) -> str:
    """``base_identity``'s digest of the files ``seen`` names in ``root`` (config.json first),
    worked out once for each (folder, sizes, times) it is seen with."""
    digest = hashlib.sha256(f"poros base v{INPUTS_VERSION}\n".encode())
    for i, (name, size, _) in enumerate(seen):
        digest.update(f"{name}\0{size}\n".encode())
        with open(os.path.join(root, name), "rb") as f:
            if i == 0 or size <= _PIECES * _PIECE:  # config.json, or a small shard: all of it
                digest.update(hashlib.file_digest(f, "sha256").digest())
                continue
            for k in range(_PIECES):
                f.seek((size - _PIECE) * k // (_PIECES - 1))
                digest.update(f.read(_PIECE))
    return digest.hexdigest()


def _a_path(spec: str) -> bool:
    """Whether an older checkpoint's dataset was a local file's path, by how it is written (the
    file itself may be on another machine): a path prefix, or a data file's suffix. A Hub id such
    as ``org/name`` is not."""
    from poros.core.dataset_spec import _LOCAL_BUILDERS, is_synthetic

    return not is_synthetic(spec) and (spec.startswith(("./", "../", "/", "~"))
                                       or Path(spec).suffix.lower() in _LOCAL_BUILDERS)


def _norm_dataset(v: Any) -> str:
    """A dataset spec as one name: a local file by its resolved path, anything else as said."""
    text = str(v)
    path = Path(text).expanduser()
    return str(path.resolve()) if path.exists() else text


def _norm_base(v: str) -> str:
    p = Path(v).expanduser()
    return str(p.resolve()) if p.exists() else v.strip().rstrip("/")


def _names(a: str, b: str) -> bool:
    """``a`` names the base at ``b``: the same folder, or a Hub id whose cache snapshot ``b`` is
    (``.../models--Org--Name/snapshots/<sha>``)."""
    pa, pb = Path(a).expanduser(), Path(b).expanduser()
    if pa.exists() and pb.exists():
        return pa.resolve() == pb.resolve()
    return ("models--" + a.strip().replace("/", "--")) in pb.parts
