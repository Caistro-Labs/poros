"""Checkpoint-directory contract shared by the trainer and the CLI.

Torch-free on purpose: the CLI resolves ``--resume-from latest`` before any
ML import, so everything here is stdlib-only.

The contract: ``PorosTrainer.save()`` writes :data:`FORMAT_MARKER` as the
FIRST file of a checkpoint and :data:`COMPLETE_MARKER` as the very LAST.
For any directory bearing the format marker, the complete marker is
authoritative: present means every resume-critical file is on disk, absent
means the save was torn — no per-file heuristic may overrule that, because
a kill between any two state files leaves a subset that heuristics can
mistake for complete. Only directories with NEITHER marker are positively
pre-format (an older Poros version), and only those fall back to "adapter
weights AND RNG AND optimizer state present".

Saves stage into ``checkpoint-N.tmp`` and are promoted into place with a
rename-aside sequence (``checkpoint-N`` -> ``checkpoint-N.old``, staging ->
``checkpoint-N``), so at no instant is the newest checkpoint neither at its
final name nor at a recoverable sibling. :func:`recover_checkpoint_dirs`
adopts or discards those siblings after a kill.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import struct
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

COMPLETE_MARKER = ".poros_checkpoint_complete"
FORMAT_MARKER = ".poros_checkpoint_format"  # content: the format version
CHECKPOINT_FORMAT_VERSION = "1"
# Its held-out score, written beside the weights after they are whole, numbers only: no resume
# reads it, so a checkpoint without one is as complete as before.
SCORE_FILE = "eval.json"
# The run a checkpoint belongs to (an id each trainer draws, and a resume adopts): pruning and the
# trainer's own "newest" read only its run's. A checkpoint without one is an older Poros's: another
# run's, unless a run says it resumed from one (PARENT_MARKER).
RUN_MARKER = ".poros_run"
# Beside RUN_MARKER in every checkpoint of a run that resumed from an unstamped checkpoint of its
# own output_dir: that checkpoint's folder name. The unstamped checkpoints there are then this
# run's earlier ones (one lineage, which "latest" reads as one run), not another run's. The
# checkpoint resumed from is never written to.
PARENT_MARKER = ".poros_run_parent"

_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin", "adapter_weights.pt")
# A step as the trainer writes it, f"checkpoint-{step}": "007" or a superscript two is no step.
_STEP = re.compile(r"0|[1-9][0-9]*", re.ASCII)


def shard_name(name: object) -> str:
    """``name`` if it is a shard's: an index is the model repo's word, and what it says is joined on
    to read a file and, in an export, to write one. "evil.pth" in site-packages runs at Python's
    next start. A ValueError that repeats none of it."""
    if not (isinstance(name, str) and re.fullmatch(r"[\w.-]+\.safetensors", name)):
        raise ValueError("unsafe shard name in checkpoint index")
    return name


def regular_only(folder: "str | os.PathLike") -> None:
    """ValueError for a pipe, socket or device directly in a model's folder. Opened by any loader,
    a pipe named model.safetensors hangs with no one to time it out. A link is its target."""
    try:
        entries = list(os.scandir(folder))
    except OSError:  # a folder that cannot be listed cannot be loaded either
        return
    for entry in entries:
        try:
            mode = entry.stat().st_mode
        except OSError:  # a dangling link opens nothing
            continue
        if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError("a model's folder holds something that is not a regular file")


def links_in(folder: Path) -> list[str]:
    """The names of the links directly in ``folder``; a folder not there yet has none."""
    try:
        return sorted(e.name for e in os.scandir(folder) if e.is_symlink())
    except OSError:
        return []


def holds_home(folder: Path) -> bool:
    """A home folder, or anything above one: nobody's output. Its links are the user's own."""
    try:
        return Path.home().resolve().is_relative_to(folder.resolve())
    except RuntimeError:  # no home to know of, as in a bare container
        return False


def drop_links(folder: Path) -> None:
    """Unlink every link directly in ``folder``, an output its user named, so what is written there
    lands there. A project's ``out/config.json`` linked to ``~/.bashrc`` would be overwritten by an
    export. A home folder is refused: ValueError, with nothing touched."""
    if holds_home(folder):
        raise ValueError(f"{folder} holds this user's home folder: name a folder of its own")
    for name in links_in(folder):
        os.unlink(os.path.join(folder, name))


def follow_umask(out: Path, since_ns: int) -> None:
    """An export is for whatever serves it, and safetensors writes a user's own 0600 whatever the
    umask says: what was written in ``out`` since then, and only that, is as readable as anything
    this user writes. A file system's clock is coarse, so two seconds' grace."""
    umask = os.umask(0)
    os.umask(umask)
    for path in out.iterdir() if out.is_dir() else ():
        seen = path.lstat()
        if stat.S_ISREG(seen.st_mode) and seen.st_mtime_ns >= since_ns - 2 * 10**9:
            path.chmod(0o666 & ~umask)


def checkpoint_step(d: Path) -> Optional[int]:
    """The step number of a ``checkpoint-<step>`` dir, else None.

    One parser for every caller (the trainer, the pruner and the CLI
    resolver), so a sibling dir like ``checkpoint-best-2`` cannot crash one
    parser while passing another's filter.
    """
    if not d.is_dir():
        return None
    suffix = d.name[len("checkpoint-"):] if d.name.startswith("checkpoint-") else ""
    return int(suffix) if _STEP.fullmatch(suffix) else None


def has_weights(d: Path) -> bool:
    """True if ``d`` holds adapter weights that can be LOADED.

    Distinct from :func:`is_complete`, which answers the stricter question
    "can training RESUME from here" and additionally requires the RNG/step
    state. Evaluation, export, and inference only need the weights, so
    gating them on is_complete would reject a plain
    ``peft_model.save_pretrained()`` directory or an adapter downloaded from
    the Hub -- both perfectly loadable.
    """
    return any((d / w).exists() for w in _WEIGHT_FILES)


def is_complete(d: Path) -> bool:
    """True iff ``d`` holds a complete, resumable checkpoint."""
    if (d / COMPLETE_MARKER).exists():
        return True
    # Format marker without complete marker: a CURRENT-version save that was
    # killed mid-write. Torn, whatever files it managed to land -- a kill
    # between optimizer and scheduler writes leaves a subset that would pass
    # the legacy heuristic below and then resume with a reset LR schedule.
    if (d / FORMAT_MARKER).exists():
        return False
    # Neither marker: positively pre-format (an older Poros version, which
    # wrote no markers at all). Require every resume-critical file that
    # version could have produced.
    return (
        has_weights(d)
        and (d / "rng_state.pt").exists()
        and (d / "optimizer_state.pt").exists()
    )


def fsync_tree(path: Path) -> None:
    """Every file under ``path`` and every folder (``path`` too) flushed to the disk: what a
    completion marker or a rename may only follow, so a power cut never leaves a marker that
    vouches for bytes still in the page cache."""
    for top, _dirs, files in os.walk(path):
        for name in files:
            fsync_path(Path(top) / name)
        fsync_path(Path(top))


def fsync_path(path: Path) -> None:
    """One file's, or one folder's entries', bytes flushed (a folder: where the OS allows it)."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) if path.is_dir()
                     else os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:  # a folder some file systems do not flush: their files are flushed
        pass
    finally:
        os.close(fd)


def run_of(d: Path) -> Optional[str]:
    """The run id a checkpoint was stamped with; None for an unstamped (older) one."""
    try:
        said = (d / RUN_MARKER).read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return said if re.fullmatch(r"[0-9a-f]{8,64}", said) else None


def parent_of(d: Path) -> Optional[str]:
    """The unstamped checkpoint (its folder name) the run of checkpoint ``d`` resumed from, where
    it says so (PARENT_MARKER); else None."""
    try:
        said = (d / PARENT_MARKER).read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None
    step = said.removeprefix("checkpoint-")
    return said if step != said and _STEP.fullmatch(step) else None


def list_checkpoints(output_dir: Path, *, run: Optional[str] = None,
                     with_unstamped: bool = False) -> List[Tuple[int, Path]]:
    """All complete ``checkpoint-<step>`` dirs in ``output_dir``, ascending; with ``run``, only
    that run's (stamped with its id), and with ``with_unstamped`` the unstamped ones too: the
    earlier checkpoints of a run that resumed from one of them."""
    out: List[Tuple[int, Path]] = []
    for d in output_dir.glob("checkpoint-*"):
        step = checkpoint_step(d)
        if step is None or not is_complete(d):
            continue
        owner = run_of(d)
        if run is None or owner == run or (with_unstamped and owner is None):
            out.append((step, d))
    return sorted(out)


def checkpoint_dirs(output_dir: Path) -> List[str]:
    """Every ``checkpoint-<step>`` folder in ``output_dir``, whole or torn, and a save's whole
    leftover (``.tmp``, ``.old``), by name: what a new run there would mix its own with. A torn
    leftover (a kill mid-save) is none of these: :func:`recover_checkpoint_dirs` drops it at the
    run's start, as a resume does. Read only: a check never deletes."""
    try:
        dirs = [d for d in Path(output_dir).iterdir() if d.is_dir() and not d.is_symlink()]
    except OSError:
        return []
    out = []
    for d in dirs:
        step = d.name.removeprefix("checkpoint-")
        if step == d.name:
            continue
        leftover = step.endswith((".tmp", ".old"))
        for tail in (".tmp", ".old"):
            step = step.removesuffix(tail)
        if _STEP.fullmatch(step) and not (leftover and not is_complete(d)):
            out.append(d.name)
    return sorted(out)


def fresh_run_clash(output_dir: "str | os.PathLike") -> Optional[str]:
    """Why a new run (not a resume) must not start in ``output_dir``: the checkpoint folders an
    earlier run left there, which pruning and ``--resume-from latest`` would mix with its own.
    None when it holds none."""
    found = checkpoint_dirs(Path(output_dir))
    if not found:
        return None
    shown = ", ".join(found[:4]) + (f" and {len(found) - 4} more" if len(found) > 4 else "")
    return (f"{output_dir} holds {shown} from an earlier run; a new run there would mix its "
            "checkpoints with them")


def runs_in(output_dir: Path) -> set:
    """The runs whose complete checkpoints ``output_dir`` holds: their ids, and None for unstamped
    checkpoints (an older Poros's). A run that resumed from an unstamped checkpoint there
    (PARENT_MARKER) continues that run: where exactly one run says so, the unstamped ones are its
    own and are not counted apart. Stamped runs are never merged."""
    found = [d for _, d in list_checkpoints(output_dir)]
    runs = {run_of(d) for d in found}
    if None in runs and len(_heirs(found)) == 1:
        runs.discard(None)
    return runs


def mixed_runs_clash(output_dir: "str | os.PathLike") -> Optional[str]:
    """Why ``output_dir`` has no newest checkpoint to continue: it holds the checkpoints of more
    than one run (``runs_in``), and the newest by step across runs is no run's newest. Says
    which checkpoints are which run's, so one can be named. None for one run's, or none at all.
    ``--resume-from latest`` and ``poros.Run.open`` ask this."""
    found = list_checkpoints(Path(output_dir))
    runs = runs_in(Path(output_dir))
    if len(runs) < 2:
        return None
    # where one run resumed from the unstamped ones they are its own, and listed with it
    heir = next(iter(_heirs([d for _, d in found])), None) if None not in runs else None
    theirs: dict = {}
    for _, d in found:  # ascending by step
        theirs.setdefault(run_of(d) or heir, []).append(d.name)
    said = []
    for run in sorted(theirs, key=lambda r: (r is None, r or "")):
        names = theirs[run]
        shown = ", ".join(names[-4:])  # the newest: what a resume would want
        shown = f"{len(names) - 4} older and {shown}" if len(names) > 4 else shown
        said.append(f"{shown} name no run" if run is None else f"run {run[:8]} has {shown}")
    return (f"{output_dir} holds checkpoints of {len(runs)} runs, so latest names none of them: "
            + "; ".join(said))


def lineage_of(ckpt: Path, output_dir: Path, new_id: str) -> Tuple[str, Optional[str]]:
    """(run id, parent) a run stamps its checkpoints with when it resumes from ``ckpt`` and
    writes into ``output_dir``: RUN_MARKER and, where there is a parent, PARENT_MARKER.

    A stamped checkpoint's run goes on under its id. An unstamped one in the run's own output_dir
    has no id to go on under, so its folder name is recorded as the parent, under the id of the
    run that already continued the unstamped checkpoints there (resuming from any of them goes on
    with the same run, as resuming from any stamped checkpoint of a run does), or ``new_id`` for
    the first to do so. An unstamped one in another folder is no ancestor of anything in
    output_dir: ``new_id``, no parent."""
    here = ckpt.resolve().parent == output_dir.resolve()
    run = run_of(ckpt)
    if run is not None:
        return run, parent_of(ckpt) if here else None
    if not here:
        return new_id, None
    heirs = _heirs([d for _, d in list_checkpoints(output_dir)])
    return (heirs.pop() if len(heirs) == 1 else new_id), ckpt.name


def _heirs(found: List[Path]) -> set:
    """The runs among checkpoints ``found`` that resumed from an unstamped one (PARENT_MARKER)."""
    return {run_of(d) for d in found if run_of(d) is not None and parent_of(d) is not None}


def recover_checkpoint_dirs(output_dir: Path) -> None:
    """Adopt or discard staging leftovers from a killed save.

    For each ``checkpoint-N.tmp`` / ``checkpoint-N.old``: if it is complete
    and the final ``checkpoint-N`` is missing or incomplete, promote it with
    an atomic rename; otherwise delete it. Run this before choosing a
    checkpoint to resume from — a kill inside the promote window leaves the
    only good copy under a sibling name.
    """
    for suffix in (".tmp", ".old"):
        for sib in sorted(output_dir.glob(f"checkpoint-*{suffix}")):
            if not sib.is_dir():
                continue
            base = sib.name[: -len(suffix)]
            stem = base[len("checkpoint-"):] if base.startswith("checkpoint-") else ""
            if not _STEP.fullmatch(stem):
                continue
            final = sib.with_name(base)
            try:
                if is_complete(sib) and not (final.exists() and is_complete(final)):
                    if final.exists():
                        shutil.rmtree(final)
                    os.replace(sib, final)
                    logger.warning(
                        "Recovered checkpoint %s from interrupted save (%s)",
                        final.name, sib.name,
                    )
                else:
                    shutil.rmtree(sib)
            except OSError as exc:  # never let cleanup break training/resume
                logger.warning("Could not recover/clean %s: %s", sib, exc)


# ---- the optimizer's update rule (poros.train.expert_adamw) ------------------------------------

RULE = "poros.expert-adamw.v4"  # ExpertAdamW's update rule, in its param_groups and RULE_FILE
TORCH_RULE = "torch.adamw"  # torch's AdamW on fp32 adapters (a dense run)
OLD_RULE = "torch.adamw.bf16"  # torch's AdamW on bf16 adapters: what no release writes any more
RULE_FILE = "optimizer.json"  # beside optimizer_state.pt, in every checkpoint 1.0.0 writes

# the largest lr ExpertAdamW takes: above it a gradient its rule reads as zero (at most 2**-126)
# would have moved a weight by more than a rounding in fp32 AdamW (poros.train.expert_adamw)
MAX_LR = 1.0


def min_adam_epsilon(beta2: float) -> float:
    """The smallest adam_epsilon under which ExpertAdamW's rule and fp32 AdamW agree to within a
    rounding: 2**-39 / sqrt(1 - beta2) (5.75e-11 at beta2 0.999).

    The rule reads a second moment v1 <= 2**-126 as zero, where fp32 AdamW keeps it (as a
    subnormal), so their denominators sqrt(v) / bc + eps differ by at most 2**-63 / bc, with bc =
    sqrt(1 - beta2**t) at least sqrt(1 - beta2) (t = 1). Relative to the denominator, at least
    eps, that is at most 2**-63 / (bc * eps); it stays under half an fp32 step (2**-24) where
    eps >= 2**-39 / bc. Below it a tiny eps turns a gradient whose square the rule reads as zero
    into an enormous step (eps 1e-30, g 1e-19: a weight of -1e7 where fp32 AdamW gives -1e-4).
    The other values the rule reads as zero (m, g, a remainder at most 2**-126) move a weight by
    at most lr * 2**-126 * 10 / eps, under 1e-25 * lr at this bound."""
    import math

    return 2.0 ** -39 / math.sqrt(1.0 - beta2) if beta2 < 1 else math.inf


_OLD_STATE_BEFORE = (
    "{where} was written by a Poros build before 1.0.0 that kept bf16 adapters' AdamW state in "
    "plain bf16, which loses small updates and freezes the second moment. ")
# what 1.0.0 does instead, by what the checkpoint's adapters were: a MoE model's expert adapters
# are still bf16, updated with a remainder and keyed rounding; a dense run's native adapters are
# fp32 now, updated by torch's AdamW in fp32
_OLD_STATE_NOW = {
    True: "1.0.0 updates them with a remainder and keyed rounding, so it cannot continue that "
          "state as the same run",
    False: "1.0.0 makes those adapters fp32 and updates them with torch's AdamW in fp32, so it "
           "cannot continue that state as the same run",
    None: "1.0.0 updates a MoE model's bf16 expert adapters with a remainder and keyed rounding, "
          "and makes a dense run's adapters fp32 under torch's AdamW, so it cannot continue that "
          "state as the same run"}
OLD_STATE_MESSAGE = _OLD_STATE_BEFORE + _OLD_STATE_NOW[None]  # true whichever the adapters were


def old_state_message(where: Any, experts: Optional[bool]) -> str:
    """OLD_STATE_MESSAGE for a checkpoint whose adapters were a MoE model's expert adapters
    (True), a dense run's (False), or are not known (None): what 1.0.0 does instead, as it is for
    those adapters."""
    return (_OLD_STATE_BEFORE + _OLD_STATE_NOW[experts]).format(where=where)
OLD_STATE_FIX = ("its adapter weights load exactly: start a new run from them (poros train "
                 "<model> --from <checkpoint>, or init_adapter: <checkpoint>), or finish the run "
                 "with the Poros version that wrote it")


def checkpoint_record(checkpoint: Any) -> Optional[dict]:
    """What a checkpoint says of its optimizer state: ``{"rule", "trainable_dtype",
    "torch_dtype", "experts"}`` (a value None where it is not known). ``rule``,
    ``trainable_dtype`` and ``torch_dtype`` come from RULE_FILE, which every checkpoint 1.0.0
    writes carries. A checkpoint from before it has none: its rule is read from its LoRA
    tensors' dtypes (a saved copy of a frozen base table, an embedding or LM head an adapter
    targets, is not one of them), OLD_RULE where they are bf16 and TORCH_RULE where they are
    not. ``experts``: its LoRA tensors are a MoE model's expert adapters, made in the model's
    dtype. None where the folder holds neither a rule file nor adapter weights it can read.
    Reads no tensor: a safetensors header, or the pickle of a torch zip file without torch."""
    folder = Path(checkpoint)
    said: Optional[dict] = None
    try:
        said = json.loads((folder / RULE_FILE).read_text())
        if not (isinstance(said, dict) and isinstance(said.get("rule"), str)):
            said = None
    except (OSError, ValueError):
        said = None
    lora: Optional[dict] = None
    for name in ("adapter_model.safetensors", "adapter_weights.pt", "adapter_model.bin"):
        path = folder / name
        if path.is_file():
            dtypes = (_safetensors_dtypes(path) if name.endswith(".safetensors")
                      else _pickled_dtypes(path))
            lora = {k: d for k, d in (dtypes or {}).items() if "lora_" in k}
            break
    experts = None if not lora else any(
        ".adapter.lora_" in k or "experts." in k for k in lora)
    if said is not None:
        return {"rule": said["rule"], "trainable_dtype": said.get("trainable_dtype"),
                "torch_dtype": said.get("torch_dtype"), "experts": experts}
    if not lora:
        return None
    found = set(lora.values())
    return {"rule": OLD_RULE if "bfloat16" in found else TORCH_RULE,
            "trainable_dtype": found.pop() if len(found) == 1 else "mixed",
            "torch_dtype": None, "experts": experts}


def checkpoint_rule(checkpoint: Any) -> Optional[str]:
    """The update rule a checkpoint's optimizer state was written by (``checkpoint_record``)."""
    record = checkpoint_record(checkpoint)
    return None if record is None else record["rule"]


_SAFETENSORS_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64"}


def _safetensors_dtypes(path: Path) -> Optional[dict]:
    """Each tensor's dtype by name, from a safetensors header; None where it cannot be read."""
    try:
        with open(path, "rb") as f:
            size = struct.unpack("<Q", f.read(8))[0]
            if size > 1 << 28:
                return None
            header = json.loads(f.read(size))
    except (OSError, ValueError, struct.error):
        return None
    return {k: _SAFETENSORS_DTYPES.get(v.get("dtype"), str(v.get("dtype")))
            for k, v in header.items() if isinstance(v, dict) and "dtype" in v}


_STORAGE_DTYPES = {"BFloat16Storage": "bfloat16", "HalfStorage": "float16",
                   "FloatStorage": "float32", "DoubleStorage": "float64"}


class _Stub:
    """Whatever a torch checkpoint's pickle builds that is not a tensor: inert."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        pass


PICKLE_LIMIT = 64 << 20  # the largest data.pkl the torch-free reader reads


class CheckpointUnreadable(ValueError):
    """A checkpoint's file that the torch-free reader refuses to read, and why."""


def _pickled_dtypes(path: Path) -> Optional[dict]:
    """Each top-level tensor's dtype by name in a torch zip checkpoint of a flat state dict,
    read from its pickle without torch: a storage is named by its legacy type
    (``BFloat16Storage``) and a tensor is rebuilt from one by ``torch._utils._rebuild_*``. None
    where it is not such a file."""
    import io
    import pickle

    class Reader(pickle.Unpickler):
        def find_class(self, module: str, name: str) -> Any:
            if module == "torch" and name.endswith("Storage"):
                return _STORAGE_DTYPES.get(name, name)
            if module == "torch._utils" and name.startswith("_rebuild_"):
                return lambda storage, *a, **k: storage
            if module == "collections" and name == "OrderedDict":
                import collections

                return collections.OrderedDict
            return _Stub

        def persistent_load(self, pid: Any) -> Any:
            return pid[1] if isinstance(pid, tuple) and len(pid) > 1 else None

    try:
        with zipfile.ZipFile(path) as z:
            pkl = next((i for i in z.infolist() if i.filename.endswith("data.pkl")), None)
            if pkl is None:
                return None
            if pkl.file_size > PICKLE_LIMIT:  # an adapter's is a few kB: no tensor data in it
                too_big = pkl.file_size
            else:
                too_big = None
                state = Reader(io.BytesIO(z.read(pkl))).load()
    except (OSError, zipfile.BadZipFile, pickle.UnpicklingError, EOFError, ValueError,
            AttributeError, TypeError, IndexError):
        return None
    if too_big is not None:
        raise CheckpointUnreadable(
            f"{path}: its data.pkl is {too_big} bytes, more than the {PICKLE_LIMIT} a "
            "checkpoint's adapter needs, so it is not read")
    if not isinstance(state, dict):
        return None
    return {str(k): v for k, v in state.items() if isinstance(v, str)}
