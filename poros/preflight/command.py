"""``poros train``'s checks before the download: each refusal, worded once, and their order.

``poros train`` (``poros.cli.train_cmd``) takes them in this order; ``plan_exit`` gives the code
the first of 5-7 would give, for a plan:

1. the output folder (``output_refusal``): not a home folder, no links, and one this user can
   create and write;
2. the dataset (``dataset_refusal``): named, a Hub id that exists and is not gated, no rows a
   run would stop at;
3. the recipe (``recipe_refusal``): one that trains this data;
4. the resume (``resume_refusal``): a checkpoint whose adapter loads into this run; once the
   recipe's data is read (still before the load), the rows it trained on (``data_resume_refusal``);
5. the model (``model_refusal``): a folder that is there, or a Hub id whose repo exists and can
   be read; then its architecture (``arch_refusal``; with ``--unvalidated``,
   ``unvalidated_refusal``);
6. the card (``device_refusal``): a CUDA GPU, looked for without torch first, then torch
   (``NO_CUDA``, ``NO_TORCH``), then what the card runs (``poros.core.card.check``);
7. the fit (``fit_refusal``, from ``poros.preflight.fit.check``);
8. the disk (``disk_refusal``): the download fits the free disk; then, a warning and never a
   refusal, the checkpoints the run keeps against its output disk
   (``poros.preflight.disk.checkpoint_note``).

``PorosTrainer.from_pretrained`` takes 6, 5 and 7 (``poros.preflight.start.on_cuda``: the card
before the architecture, as it holds the device already). It has no data, recipe or output folder
until ``train`` and ``save``: those checks are this command's, before a model downloads. Each
function returns a ``poros.core.refusal.Refusal`` (None where the run may go on); the CLI shows it
(``poros.cli.contract.refused``). Torch-free.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import Any, Callable, Optional

from poros.core.refusal import Refusal

BEFORE_THE_LOAD = "refused before the model download"
_REFUSALS = "docs/training.md#refusals"  # the order of these checks, and each one's code
RESUME_DOCS = "docs/training.md#checkpoints-and-resume"  # where a resume's refusal points

NO_CUDA = Refusal("E_NO_CUDA", "No GPU", "training requires a CUDA GPU and none is available here",
                  fix="poros machine", note="inspects this environment")
NO_TORCH = Refusal("E_NO_TORCH", "Needs PyTorch",
                   "training needs PyTorch and the training libraries, which are not installed "
                   "here",
                   fix="pip install poros-train", note="torch, transformers, peft")


def no_cuda_before_torch() -> Refusal:
    """No GPU, found without torch: said before the missing training libraries are, so a machine
    that cannot train is not sent to a multi-GB install first."""
    return Refusal("E_NO_CUDA", "No GPU",
                   "training requires a CUDA GPU and none was found here",
                   context=["nvidia-smi shows none, or is not installed",
                            "on a machine with one, training also needs PyTorch and the "
                            "training libraries, which are not installed here: pip install "
                            "poros-train"],
                   fix="poros machine", note="inspects this environment")


def device_refusal() -> Optional[Refusal]:
    """What every command that trains on CUDA checks at its device step (train, prove): a CUDA
    GPU, then torch. The GPU is looked for without torch first (nvidia-smi, as `poros machine`
    reads it): where there is none, that is said, not the install. None when both are there."""
    import importlib.util

    from poros.core.card import visible_gpus

    has_torch = importlib.util.find_spec("torch") is not None
    cards = visible_gpus()  # None: no nvidia-smi (or a MIG slice), and only torch can say
    if not has_torch:
        if cards:
            return NO_TORCH
        return no_cuda_before_torch()
    import torch

    return None if torch.cuda.is_available() else NO_CUDA


def output_refusal(output_dir: str) -> Optional[Refusal]:
    """All a run writes lands in its folder, and a config file may have named that folder for
    it: a home folder, or a link there, is refused and left as it is."""
    from poros.core import checkpoints

    out_dir = Path(output_dir)
    another = "name a folder of its own: --output DIR, or output_dir in a config"
    if checkpoints.holds_home(out_dir):
        return Refusal("E_CONFIG", "Not an output folder",
                       f"{output_dir} is your home folder, or a folder above it: a run's files "
                       "go in a folder of their own", context=[another])
    links = checkpoints.links_in(out_dir)
    if links:
        named = ", ".join(links[:3]) + (f" and {len(links) - 3} more" if len(links) > 3 else "")
        return Refusal("E_CONFIG", "Not an output folder",
                       f"{output_dir} holds links ({named}), and a run never writes through "
                       "a link", context=["remove them, or " + another])
    from poros.preflight.disk import output_fault

    return unwritable_output(output_fault(output_dir))


def unwritable_output(fault: Optional[tuple], *, own: bool = False) -> Optional[Refusal]:
    """An output folder a run cannot create or write (``poros.preflight.disk.output_fault``, or
    ``output_error`` for what only the write itself found): ``E_CONFIG`` for a path that is no
    folder, ``E_DISK`` for one this user may not write. ``own``: the folder is the run's own,
    under POROS_HOME, and no one named it."""
    if fault is None:
        return None
    where = ("a run keeps its record under ~/.poros: POROS_HOME names another home" if own
             else "name another folder: --output DIR, or output_dir in a config")
    return Refusal(fault[0], "Can't write the output folder", fault[1],
                   context=[where, BEFORE_THE_LOAD])


def dataset_refusal(cfg: Any) -> Optional[Refusal]:
    """No dataset, a Hub id that is missing or gated, or a local file with rows a run would stop
    at. A dataset spec that cannot be read raises as it is (the caller reports the exception)."""
    from poros.core.dataset_spec import resolve_dataset_spec
    from poros.preflight import data

    if cfg.dataset is None:
        return Refusal("E_CONFIG", "No dataset", "the config has no dataset field",
                       context=["add one: a local .jsonl/.json/.csv file, or a Hub dataset id",
                                "(e.g. OpenAssistant/oasst1)"],
                       docs="docs/training.md#data")
    ds_kind, _, ds_path = resolve_dataset_spec(cfg.dataset)
    if ds_kind == "raw" and not ds_path:
        return data.hub_dataset_refusal(cfg.dataset)
    if not data.checked_before_load(Path(str(cfg.dataset))):
        # only a file `data` reads as the trainer does: another format the trainer reads
        # (.parquet, .txt) goes to it as before
        return None
    try:
        rows, stops = data.rows_and_stops(Path(str(cfg.dataset)))
    except Refusal as r:  # a file that cannot be read at all: its own refusal, not a trace
        return Refusal(r.code, r.title, r.reason, context=[*r.context, BEFORE_THE_LOAD])
    if not rows and not stops:  # an empty file: said as the verb form says it
        return Refusal("E_DATASET", "No rows", f"{cfg.dataset} has no rows to train on",
                       context=[BEFORE_THE_LOAD])
    if stops:
        more = f"; and {len(stops) - 3} more" if len(stops) > 3 else ""
        return Refusal("E_DATASET", "Rows that stop a run", "; ".join(stops[:3]) + more,
                       context=["`poros data <file> --fix` writes a copy without them",
                                BEFORE_THE_LOAD])
    return None


def recipe_refusal(exc: ValueError) -> Refusal:
    """No recipe trains this data (``poros.recipes.choose`` said why)."""
    return Refusal("E_CONFIG", "Can't train this", str(exc),
                   fix="poros recipes", note="lists the recipes and the data each one reads")


def resume_refusal(resume_from: Optional[str], cfg: Any,
                   progress: Optional[Callable[[int, int], None]] = None) -> Optional[Refusal]:
    """A checkpoint that is not this run's (other adapter shapes, another base, other data):
    refused now, not after the load, with the next step. What could not be checked (an older
    checkpoint's data or base at another path, with no record of the contents) is warned of.

    The check reads the dataset file once to compare its contents (about 2 GB a second):
    ``progress(done, size)``, in bytes, is called as it goes, for a caller that shows the wait.
    It is not called where there is no file to read or its hash is already known."""
    from poros.preflight.resume import SAME_SETTINGS, resume_verdict, warn_once

    if not resume_from:
        return None
    clash, unverified = resume_verdict(resume_from, cfg, progress=progress)
    if clash:
        return Refusal("E_CONFIG", "Can't resume this", clash,
                       context=[BEFORE_THE_LOAD, clash.fix or SAME_SETTINGS,
                                *([clash.route] if clash.route else [])], docs=RESUME_DOCS)
    for said in unverified:
        warn_once(said)
    return None


def fresh_run_refusal(resume_from: Optional[str], cfg: Any) -> Optional[Refusal]:
    """A new run (no resume) into an output_dir an earlier run left checkpoints in: refused before
    the load, naming them; resume it, or name another folder."""
    from poros.core.checkpoints import fresh_run_clash

    clash = None if resume_from else fresh_run_clash(cfg.output_dir)
    if clash:
        return Refusal("E_CONFIG", "Not a new run's folder", clash,
                       context=[BEFORE_THE_LOAD, "continue that run with --resume-from latest, "
                                "or name another output_dir"], docs=RESUME_DOCS)
    return None


def data_resume_refusal(resume_from: Optional[str], data: Any,
                        before_load: bool = True) -> Optional[Refusal]:
    """A checkpoint whose recorded rows are not the ones this run reads (``data``: the recipe's
    data, read before the load): refused, naming both, before the model loads."""
    from poros.preflight.resume import data_clash

    clash = data_clash(resume_from, data) if resume_from else None
    if clash:
        how = ("resume with the data and settings (max_seq_length, train_on, the tokenizer) the "
               "checkpoint trained on, or start a new run")
        return Refusal("E_CONFIG", "Not the same data", clash,
                       context=[*([BEFORE_THE_LOAD] if before_load else []), how,
                                *([clash.route] if getattr(clash, "route", None) else [])],
                       docs=RESUME_DOCS)
    return None


def start_refusal(resume_from: Optional[str], cfg: Any) -> Optional[Refusal]:
    """A new run that starts from an earlier adapter (``init_adapter``, the verb form's
    ``--from``): the folder holds one, on this base, at this rank and these target modules, or
    it is refused here, before the download. ``cfg.init_adapter`` becomes the checkpoint itself
    (a run's folder names its newest), and where the config says no rank, alpha or target
    modules, the adapter's are taken. A resume reads its own checkpoint: nothing to do."""
    from poros.preflight.resume import start_from

    if resume_from or not getattr(cfg, "init_adapter", None):
        return None
    try:
        start, clash, _ = start_from(cfg)
    except FileNotFoundError as exc:
        return Refusal("E_NOT_FOUND", "No adapter to start from", str(exc),
                       context=["init_adapter takes a checkpoint folder, or a run's folder for "
                                "its newest checkpoint"], docs=RESUME_DOCS)
    cfg.init_adapter = start
    if clash:
        return Refusal("E_CONFIG", "Can't start from this adapter", clash,
                       context=[BEFORE_THE_LOAD, clash.fix or ""], docs=RESUME_DOCS)
    return None


def _near_name(model: str) -> Optional[str]:
    """The catalog id ``model`` is a slip of the keys away from, or None."""
    import difflib

    from poros.catalog import raw

    ids = {m["id"].lower(): m["id"] for m in raw()["model"]}
    near = difflib.get_close_matches(model.lower(), ids, n=1, cutoff=0.85)
    return ids[near[0]] if near else None


def model_refusal(model: str, revision: Optional[str] = None, *,
                  cache_decides: bool = True) -> Optional[Refusal]:
    """A model that is not there to train: a path with nothing at it, or a Hub id whose repo
    does not exist (``E_NOT_FOUND``, with the catalog name it is a slip away from), is gated
    (``E_HUB_AUTH``), or cannot be asked for and is not in the cache (``E_NETWORK``: offline
    mode, or no network). Asked before any consent is offered for the model, and before a
    download. ``cache_decides``: whether a load reads the Hugging Face cache this looks in, so
    that "not in it, and no Hub" means the load cannot start. False for a preview, which loads
    nothing, and on a Mac, whose weights can be in a store of its own (its own checks say what
    is missing there): the Hub's own answers (no such repo, gated) are refused either way.
    Except "gated" where the stored token was withheld (under ``poros agents``, a name the
    catalog does not hold): that is the Hub's answer to a stranger, and the load, which sends
    the token, is left to say whether this user has access."""
    from poros.catalog import names
    from poros.core.env import under_agent
    from poros.preflight.arch import hub_fault

    path = Path(model).expanduser()
    if path.exists():
        return None
    if model.startswith(("./", "../", "/", "~")):
        return Refusal("E_NOT_FOUND", "No such model", f"no model folder at {model}",
                       context=["a path is read from disk, never looked up on the Hub",
                                BEFORE_THE_LOAD])
    fault = hub_fault(model, revision)
    if fault is None or (not cache_decides and fault in ("offline", "network")):
        return None
    # Under `poros agents` no stored token is sent for a name the catalog does not hold (an
    # agent's words may be anyone's), so the Hub answered as it answers a stranger: "gated"
    # says nothing of this user's access. The load sends the token and says what the Hub says.
    withheld = under_agent() and not names(model)
    if fault == "gated" and withheld:
        return None
    if fault == "missing":
        near = None if names(model) else _near_name(model)
        private = ("under poros agents no stored token is sent for a name the catalog does not "
                   f"hold, so a private repo you can read answers the same: hf download {model} "
                   "first, or name its folder")
        return Refusal("E_NOT_FOUND", "No such model",
                       f"{model} is not on the Hugging Face Hub",
                       context=[*([f"did you mean {near}?"] if near else []),
                                private if withheld else
                                "check the name; a private repo answers the same until you log "
                                "in: hf auth login", BEFORE_THE_LOAD],
                       fix="poros models", note="lists every model Poros names",
                       docs=_REFUSALS)
    if fault == "gated":
        return Refusal("E_HUB_AUTH", "Needs access", f"{model} is gated on the Hugging Face Hub",
                       context=["accept its license on its page at huggingface.co, then log in: "
                                "hf auth login", BEFORE_THE_LOAD], docs=_REFUSALS)
    if fault == "offline":
        return Refusal("E_NETWORK", "Offline",
                       f"{model} is not in the local cache, and Hub access is off here "
                       "(HF_HUB_OFFLINE is set)",
                       context=["unset HF_HUB_OFFLINE to download it, or name a local model "
                                "folder", BEFORE_THE_LOAD], docs=_REFUSALS)
    return Refusal("E_NETWORK", "Can't reach the Hub",
                   f"{model} is not in the local cache, and the Hugging Face Hub could not be "
                   "reached",
                   context=["check the network (or a proxy), then run this again; a local "
                            "model folder needs none", BEFORE_THE_LOAD], docs=_REFUSALS)


def arch_refusal(model: str, model_type: Optional[str], arch_check: str) -> Optional[Refusal]:
    """The architecture gate (``poros.preflight.arch.arch_refusal``) in this command's words."""
    from poros.preflight.arch import arch_refusal as gate
    from poros.preflight.arch import routable_arch_types

    if arch_check != "error" or model_type is None or gate(model_type) is None:
        return None
    return Refusal("E_ARCH_UNSUPPORTED", "Not supported",
                   f"architecture {model_type} is not in the validated set",
                   context=[", ".join(sorted(routable_arch_types())), BEFORE_THE_LOAD,
                            "--unvalidated trains a family Poros has not validated, this run "
                            "only, with no parity guarantee"],
                   fix=f"poros arch try {model}", note="checks its math on a tiny copy")


def looped_on_a_mac(model: str, model_type: Optional[str]) -> Optional[Refusal]:
    """On a Mac: a decoder that runs its layers as visits (a looped one, as its config class
    declares) is refused before anything downloads, whatever its status or --unvalidated."""
    from poros.moe.carried import declares_visits
    from poros.preflight.arch import mac_looped

    if model_type is None or not declares_visits(model_type):
        return None
    return Refusal("E_ARCH_UNSUPPORTED", "Not on a Mac", f"{model}: {mac_looped(model_type)}",
                   context=[f"model_type {model_type}: no flag trains it on a Mac"])


def unnamed_family(model: str, reason: str) -> Refusal:
    """``--unvalidated`` for a model whose ``model_type`` cannot be read."""
    return Refusal("E_ARCH_UNSUPPORTED", "Can't name its family", f"{model}: {reason}",
                   context=["--unvalidated trains a family it can read first"])


def not_trainable(model: str, why: str, model_type: Optional[str] = None) -> Refusal:
    """A family no flag trains (``poros.preflight.arch.no_flag_trains``): said in plain words."""
    return Refusal("E_ARCH_UNSUPPORTED", "Not trainable", f"{model}: {why}",
                   context=[f"model_type {model_type}: no flag trains it" if model_type
                            else "no flag trains it"],
                   fix=shlex.join(["poros", "arch", model]), note="what it is, from its config")


def unvalidated_refusal(model: str, model_type: Optional[str], trust_remote_code: bool,
                        reading: Callable[[str], Any]) -> Optional[Refusal]:
    """``--unvalidated``: what no flag gets past, before the model's own type runs as
    ``user_asserted`` for this run. ``reading(model)``: the family as ``poros arch`` reads it
    (its ``refused`` and ``model_type``), asked only where this command has not read it yet."""
    from poros.core.arch_registry import arch_status, validated_arch_types

    if trust_remote_code:
        return Refusal("E_USAGE", "Not with --trust-remote-code",
                       "--unvalidated never runs a repo's own code", context=["drop one of them"])
    if model_type is None:
        return Refusal("E_ARCH_UNSUPPORTED", "Can't name its family",
                       f"{model}: no model_type could be read for it, so --unvalidated has "
                       "nothing to assert")
    if model_type in validated_arch_types() | validated_arch_types("moe"):
        return None  # validated: nothing to assert, and nothing is downgraded
    if arch_status(model_type) != "user_asserted":  # not read yet in this command
        seen = reading(model)
        if seen.refused or seen.model_type != model_type:
            return not_trainable(model, seen.refused or "its family could not be read")
    return None


def fit_refusal(fit: Any) -> Optional[Refusal]:
    """The GPU fit's refusal (``poros.preflight.fit.check``'s ``vram``), word for word the
    trainer's, with what would fit and where the certified runs are listed."""
    unfit = fit.vram
    if unfit is None:
        return None
    msg = unfit.info.message + (f" What to change: {unfit.would}." if unfit.would else "")
    lighter = {"fast": "path: efficient holds less; ", "efficient": "a smaller block_size "
               "or path: sequential may hold less; "}.get(fit.p.path or "", "")
    if fit.p.entry == "moe":  # poros.moe refuses a block_size: only what holds less there
        lighter = "path: efficient holds less; " if fit.p.path == "fast" else ""
    return Refusal("E_OOM_PREDICTED", "Won't fit", msg,
                   fix=("poros recipes lists the runs certified on this Mac; --force trains "
                        "anyway." if unfit.would else "the check found no setting here it does not "
                        "refuse; poros recipes lists the runs certified on this Mac; --force "
                        "trains anyway.")
                   if fit.p.gpu is not None and fit.p.gpu.unified
                   else f"{lighter}poros recipes lists the runs certified on this card; --force "
                   "trains anyway.",
                   docs="docs/training.md#what-fits")


def shorter_that_fits(seq_len: int, floor: int, plan: Callable[[int], Any]) -> Optional[tuple]:
    """What would fit a run its plan says will not: the longest length, halved from
    ``seq_len`` down to ``floor``, whose ``plan(length)`` does not refuse it (``fits`` not
    False): (length, that plan), or None. The device-independent half of every "what fits"
    (``poros train``'s question on CUDA and on a Mac); ``plan`` holds the rest of the run's
    settings."""
    seq = seq_len // 2
    while seq >= floor:
        q = plan(seq)
        if q is not None and q.fits is not False:
            return seq, q
        seq //= 2
    return None


def fits_said(flag: str, value: Any, q: Any) -> str:
    """``shorter_that_fits``'s answer in words: "--seq-len 2048 is not refused", what the check
    knows (never that it fits: only a certified run says that)."""
    return f"{flag} {value} is not refused"


def disk_refusal(size: Optional[tuple]) -> Optional[Refusal]:
    """A download larger than the free disk (``poros.preflight.disk.download_note``'s third
    part): refused now, not when it fills."""
    if not (size and size[2]):
        return None
    return Refusal("E_DISK", "Disk full", size[2],
                   context=[size[0], "refused before the download",
                            "free space at the Hugging Face cache, or point HF_HOME at a larger "
                            "disk", "--force downloads anyway"])


def plan_exit(cfg: Any, plan: Any) -> Optional[str]:
    """The contract code ``poros train`` would exit with for ``plan``,
    in this module's order: the architecture (5), the card (6), the fit (7); None where it
    trains."""
    from poros.core.card import card_refusal

    if plan.support == "unsupported" and cfg.blockwise_arch_check == "error":
        return "E_ARCH_UNSUPPORTED"
    if card_refusal(cfg, plan.gpu) is not None:
        return "E_GPU_UNSUPPORTED"
    if False in (plan.fits, plan.host_fits):
        return "E_OOM_PREDICTED"
    return None
