"""Whether the download fits the disk, decided before it starts (``E_DISK``; ``--force``
downloads anyway), and whether a run can write its output folder (``output_fault``)."""

from __future__ import annotations

import errno
import os
from pathlib import Path
from typing import Any, Optional

# an OSError a disk gives (no room, or none this user may write): E_DISK; any other is the path's
_DISK_ERRNOS = (errno.ENOSPC, errno.EROFS, errno.EACCES, errno.EPERM, errno.EDQUOT)


def output_fault(output_dir: Any, *, read_only_ok: bool = False) -> Optional[tuple[str, str]]:
    """Why a run cannot write ``output_dir``: (its code, a sentence naming the folder), or None.
    ``E_CONFIG`` for a path that is no folder (a file, or under one), ``E_DISK`` for one this
    user may not write. Nothing is created: a folder not there yet is judged by the nearest
    folder that is. ``read_only_ok``: a folder that is there passes as it is (a load that only
    reads a run)."""
    out = Path(str(output_dir)).expanduser()
    try:
        here = next((p for p in (out, *out.parents) if os.path.lexists(p)), None)
        if here is None:  # nothing of the path is there: a relative path's own start
            return None
        if not here.is_dir():
            return "E_CONFIG", (f"{output_dir} is a file, not a folder" if here == out else
                                f"{output_dir} cannot be created: {here} is a file, not a folder")
        if here == out and read_only_ok:
            return None
        if not os.access(here, os.W_OK | os.X_OK):
            why = "permission denied, or a read-only disk"
            return "E_DISK", (f"{output_dir} cannot be written: {why}" if here == out else
                              f"{output_dir} cannot be created: {here} cannot be written ({why})")
    except OSError as exc:  # a name too long, a link that loops
        return output_error(exc, output_dir)
    return None


def output_error(exc: OSError, output_dir: Any) -> tuple[str, str]:
    """An OSError from making or writing ``output_dir``, as ``output_fault`` says one: what the
    check could not see first (a full disk, a file system that refuses the name)."""
    code = "E_DISK" if exc.errno in _DISK_ERRNOS else "E_CONFIG"
    why = os.strerror(exc.errno) if isinstance(exc.errno, int) else type(exc).__name__
    return code, f"{output_dir} cannot be written: {why.lower()}"


def require_output_dir(output_dir: Any, *, read_only_ok: bool = False) -> None:
    """``output_fault`` raised, for the Python API: PorosDiskError or PorosConfigError, before
    the download."""
    from poros.core.errors import PorosConfigError, PorosDiskError

    fault = output_fault(output_dir, read_only_ok=read_only_ok)
    if fault is not None:
        kind = PorosDiskError if fault[0] == "E_DISK" else PorosConfigError
        raise kind(f"Can't write the output folder: {fault[1]}.",
                   remediation="Give another output_dir.")


def output_over_start(output_dir: Any, *starts: Any, resume_from: Any = None) -> Any:
    """A new run whose output folder is the run it starts from: ``output_dir`` is, holds, or
    is inside a folder of ``starts`` (the earlier run's folder, the adapter ``init_adapter``
    names). Training there would replace, or clear, the adapter and checkpoints it starts
    from: a Refusal (E_CONFIG), before anything is asked or removed. None where they are
    apart, where there is nothing to start from, and for a resume (its folder is its own)."""
    from poros.core.refusal import Refusal

    if resume_from or not output_dir:
        return None
    out = Path(str(output_dir)).expanduser().resolve()
    for start in starts:
        if not start:
            continue
        there = Path(str(start)).expanduser().resolve()
        if out == there or out in there.parents or there in out.parents:
            how = ("is the folder of the run this one starts from" if out == there else
                   f"holds the run this one starts from ({start})" if out in there.parents else
                   f"is inside the run this one starts from ({start})")
            return Refusal(
                "E_CONFIG", "Not an output folder",
                f"{output_dir} {how}: training into it would replace the adapter and "
                "checkpoints it starts from",
                context=["nothing was removed",
                         "name another output folder (--output DIR, or output_dir in a config), "
                         "or leave --output out: a new run gets a folder of its own"])
    return None


def disk_text(gb: float) -> str:
    """A download or a size on disk as Poros says one everywhere, in decimal GB, as the Hub, a
    disk and the Mac table count: two places under 1 GB, so a small model's 0.27 GB does not
    read 0.3, then one. (Memory, VRAM and host RAM, is said in GiB.)"""
    if gb < 0.1:
        return "under 0.1 GB"
    return f"{gb:.2f} GB" if gb < 1 else f"{gb:,.1f} GB"


def download_note(cfg: Any) -> Optional[tuple]:
    """(``"140.2 GB download · 412 GB free at the HF cache"``, warning or "", refusal or ""): the
    refusal where the download is larger than the free disk, which no setting changes. The
    download is what a load fetches of ``cfg``'s revision and is not cached yet
    (``poros.plan.configs.download_size``: the loader's own files, never a repo's other weights)."""
    if Path(cfg.model_name_or_path).expanduser().exists():
        return None
    try:
        import shutil

        from huggingface_hub.constants import HF_HUB_CACHE

        from poros.plan.configs import download_size

        fetched = download_size(cfg).value  # GiB; None where the Hub's list does not say
        if not fetched:
            return None
        size = fetched * 2**30
        cache = Path(HF_HUB_CACHE).expanduser()
        probe = next((p for p in (cache, *cache.parents) if p.exists()), Path.home())
        free = shutil.disk_usage(str(probe)).free
        gb = 1e9  # decimal GB, the unit every download and disk is said in (disk_text)
        s = disk_text(size / gb)
        warn = "free disk may not fit this model plus its cache" if free < size * 1.15 else ""
        short = (f"{cfg.model_name_or_path} is a {s} download and {free / gb:.0f} GB is free at "
                 f"the Hugging Face cache ({probe})") if free < size else ""
        return (f"{s} download · {free / gb:.0f} GB free at the HF cache", warn, short)
    except Exception:
        return None


def checkpoint_parts(cfg: Any, model_config: Optional[dict]) -> Optional[tuple[int, int]]:
    """About one checkpoint of ``cfg``'s run and its adapter weights alone, in bytes, from
    config.json's sizes (``poros.plan.estimate``); None where config.json does not size it.

    A dense model's LoRA is fp32 with Adam's two fp32 moments: 12 bytes a weight, the weights 4
    of them. A MoE's expert adapters are bf16, with ExpertAdamW's state beside each weight: its
    two bf16 moments and its bf16 remainder (poros.train.expert_adamw), 8 bytes a weight, the
    weights 2 of them."""
    from poros.plan.estimate import _sizes, _text

    s = _sizes(model_config)
    if s is None:
        return None
    rank = int(cfg.lora_rank)
    dense = rank * s.per_rank
    if not s.experts_nf4:
        return 12 * dense, 4 * dense
    try:  # poros.moe.adapters: per expert r x (2h + 3 x expert width), and the attention's LoRA
        from poros.core.arch_registry import registry_type
        from poros.moe.arch import MOE_FAMILIES

        t = _text(model_config)
        n = int(t.get("num_experts") or t.get("num_local_experts") or t.get("n_routed_experts"))
        emid = int(t[MOE_FAMILIES[registry_type(t)]])
    except (KeyError, TypeError, ValueError):
        return 12 * dense, 4 * dense
    experts = s.layers * n * rank * (2 * s.h + 3 * emid)
    rest = max(dense - s.layers * n * rank * 3 * (s.h + emid), 0)
    return 8 * experts + 12 * rest, 2 * experts + 4 * rest


def checkpoint_bytes(cfg: Any, model_config: Optional[dict]) -> Optional[int]:
    """About one checkpoint of ``cfg``'s run, in bytes (``checkpoint_parts``)."""
    parts = checkpoint_parts(cfg, model_config)
    return None if parts is None else parts[0]


def checkpoint_note(cfg: Any, model_config: Optional[dict]) -> Optional[str]:
    """Why the run's checkpoints may not fit its output disk, or None: the checkpoints it keeps
    (save_total_limit, or every one it saves), one more while a save is written, and its finished
    adapter, against the free disk at output_dir. A warning, never a refusal: the estimate errs
    high, and a run that fills the disk stops with E_DISK at its newest whole checkpoint."""
    if not getattr(cfg, "save_steps", 0):
        keep = 1  # the one a run saves at its end
    elif cfg.save_total_limit and cfg.save_total_limit > 0:
        keep = int(cfg.save_total_limit)
    elif cfg.max_steps and cfg.max_steps > 0:
        keep = -(-int(cfg.max_steps) // int(cfg.save_steps))
    else:
        keep = 1
    parts = checkpoint_parts(cfg, model_config)
    if not parts or not parts[0]:
        return None
    one, adapter = parts  # adapter: the finished run's weights alone
    need = (keep + 1) * one + adapter
    import shutil

    out = Path(str(cfg.output_dir)).expanduser()
    probe = next((p for p in (out, *out.resolve().parents) if p.exists()), Path.cwd())
    try:
        free = shutil.disk_usage(str(probe)).free
    except OSError:
        return None
    if free >= need:
        return None
    gb = 1e9
    return (f"its checkpoints may fill the disk: {keep} of about {one / gb:.1f} GB each, one more "
            f"while a save is written, and the adapter: about {need / gb:.0f} GB, and "
            f"{free / gb:.0f} GB is free at {out} (save_total_limit=1 keeps one; another "
            "output_dir moves them)")
