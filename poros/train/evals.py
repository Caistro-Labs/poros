"""Held-out scores during a run, one set of rules for both loops: the engine's own
(``PorosTrainer.train``) and ``poros.Run``'s (the recipes). When a run scores, the generators an
eval pass must leave as they were, the score a checkpoint keeps beside its weights, and the
first and last scores a run's result carries."""

from __future__ import annotations

import contextlib
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional

from poros.core import checkpoints

if TYPE_CHECKING:  # torch-free to import: the CLI's light install reads these rules too
    import torch


def saves_at(config: Any, step: int) -> bool:
    """Whether a run writes a checkpoint after update ``step`` (its index from 0): every
    ``save_steps``, never after update 0."""
    save = config.save_steps
    return save > 0 and step > 0 and step % save == 0


def scores_at(config: Any, step: int, total: Optional[int]) -> bool:
    """Whether a loop scores its held-out rows after update ``step``: every ``eval_steps``, and
    where it saves a checkpoint (``eval_checkpoints``), so the checkpoint keeps its score. Once a
    step, and not after the last (``total`` updates): the loop scores the finished run then."""
    if step <= 0 or (total is not None and step + 1 >= total):
        return False
    every = config.eval_steps > 0 and step % config.eval_steps == 0
    return every or (bool(config.eval_checkpoints) and saves_at(config, step))


@contextlib.contextmanager
def kept_rng(device: "torch.device") -> Iterator[None]:
    """The generators as they were, whatever runs inside: an eval pass must not move the next
    step's dropout. A loader with no generator of its own draws its seed from the CPU one as it
    starts, and a draw added to a forward later would move every later step."""
    import torch

    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    mps = torch.mps.get_rng_state() if device.type == "mps" else None
    try:
        yield
    finally:
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state(cuda, device)
        if mps is not None:
            torch.mps.set_rng_state(mps)


# Beside a checkpoint: the run's first held-out score, of the model it started from. A resume
# reads it from the checkpoint it continues (``read_start``) and every later checkpoint keeps it,
# so a run resumed any number of times says the start it really had.
START_FILE = "eval_start.json"


def _numbers(said: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in said.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)}


def _write(folder: Path, name: str, said: Dict[str, Any], keep: tuple = ()) -> None:
    part = folder / f".{name}.part"
    words = {k: said[k] for k in keep if isinstance(said.get(k), str)}
    part.write_text(json.dumps({**_numbers(said), **words}) + "\n")
    os.replace(part, folder / name)


def write_score(folder: Path, said: Dict[str, Any]) -> None:
    """The score beside the weights it is of, numbers only, whole or not at all: a reader finds
    the file or none."""
    _write(folder, checkpoints.SCORE_FILE, said)


def write_start(folder: Path, said: Dict[str, Any]) -> None:
    """The run's first score (before its first step) beside a checkpoint, as ``write_score``,
    with ``held``: which rows it was scored on (``held_rows``)."""
    _write(folder, START_FILE, said, keep=("held",))


def held_rows(rows: Any) -> Optional[str]:
    """Which held-out rows a score was taken on, as the run read them: a recipe's rows by their
    tokens (``poros.data.Data.fingerprint``, as the data record names a run's training rows), or
    a hash the engine's loop took of each batch's tokens and labels, in order (``hashlib``
    object). None: not known."""
    if rows is None:
        return None
    if hasattr(rows, "hexdigest"):
        return rows.hexdigest()[:16]
    said = getattr(rows, "fingerprint", None)
    return said if isinstance(said, str) else None


# A resumed run's start, dropped because the rows it is scored on now are not the rows it was
# scored on then (another eval_dataset, or the file edited): said where the start would be.
ROWS_CHANGED = ("the held-out rows changed since the run began, so its first score is not one "
                "to compare with")


def settle_start(engine: Any, rows: int, held: Optional[str]) -> None:
    """After a score of ``rows`` rows that ``held`` names: the start a resumed run carries stays
    only if it was scored on the same rows. Rows that differ drop it and say so
    (``ROWS_CHANGED``); a start that names no rows (a checkpoint from before starts did) cannot
    be checked, and is dropped with nothing to say. The resume itself goes on: the held-out
    rows train nothing, and a resume refuses only what changes what trains."""
    start = getattr(engine, "_held_start", None)
    if start is None:
        return
    if start.get("held") is None:
        engine._held_start = None
    elif start.get("held") != held or start.get("rows") != rows:
        engine._held_start, engine._held_changed = None, True


def read_start(folder: Any) -> Optional[Dict[str, Any]]:
    """The run's first score as the checkpoint in ``folder`` keeps it; None where it keeps none
    (no held-out rows, or a checkpoint written before checkpoints kept it): not known."""
    try:
        said = json.loads((Path(str(folder)) / START_FILE).read_text())
    except (OSError, ValueError):
        return None
    return said if isinstance(said, dict) and isinstance(said.get("eval_loss"), (int, float)) \
        else None


def first_of(history: List[Dict[str, Any]], start: Optional[Dict[str, Any]]
             ) -> Optional[Dict[str, Any]]:
    """The run's first score: ``start`` where the run carries it (a resume reads it from its
    checkpoint), else this process's score before the first step (step 0) if it took one. A
    resumed run's first score here was taken on the way: never a start."""
    if start is not None:
        return start
    return history[0] if history and history[0].get("step") == 0 else None


# What each caller of a held-out pass can change when it runs out of GPU memory.
OOM_ADVICE = {
    "train": ("A resume keeps per_device_batch_size and max_seq_length: free the GPU's memory "
              "and continue, or start the run again with a lower per_device_batch_size or "
              "max_seq_length. A shorter held-out file does not help: it scores fewer batches "
              "of the same size."),
    "run": ("poros eval scores as the run scored, at the batch and length its record says, and "
            "takes no other: free the GPU's memory (another program may hold some), or score it "
            "on a card with more. --rows and a shorter --data file do not help: they score "
            "fewer batches of the same size."),
    "config": ("These are the config's per_device_batch_size and max_seq_length: lower either "
               "there. A shorter max_seq_length cuts each row shorter, so the score is no longer "
               "the one the run took. A shorter file does not help: it scores fewer batches of "
               "the same size."),
}


@contextlib.contextmanager
def scoring_out_of_memory(engine: Any, rows: int, length: int,
                          caller: str = "train") -> Iterator[None]:
    """Around one batch of a held-out pass, in both loops: CUDA running out of memory there is a
    ``PorosOOMError`` that says it happened while scoring and what to change, as a training
    step's is (``engine``: the trainer; ``rows`` and ``length``: the batch's). A Mac's allocator
    error is its own (``_raise_if_mps_oom``), as in training. The advice is what the caller can
    change (``OOM_ADVICE``): a run's scoring (``train``), ``poros eval <run>`` (``run``: the run's
    record sets both, and it takes no other), ``poros eval <config>`` or ``evaluate()``
    (``config``)."""
    import torch

    from poros.core.errors import PorosOOMError

    try:
        yield
    except torch.cuda.OutOfMemoryError as exc:
        engine._raise_if_mps_oom(exc)
        engine._release_retained_forward_blocks()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        said = f"{rows} row{'s' * (rows != 1)}"
        raise PorosOOMError(
            "GPU out of memory while scoring the held-out rows.",
            seq_len=length if rows == 1 and caller != "run" else None,
            remediation=(f"A score runs one batch at a time through the model, here {said} of "
                         f"up to {length} tokens. " + OOM_ADVICE[caller]),
        ) from exc
    except RuntimeError as exc:  # mps: the allocator's out of memory is a plain RuntimeError
        engine._raise_if_mps_oom(exc)
        raise


def with_eval(metrics: dict, history: List[Dict[str, Any]],
              start: Optional[Dict[str, Any]] = None, changed: bool = False) -> dict:
    """The held-out loss first (the model it started from) and last; and last, on chats, how many
    rows end their answers, and on pairs how many prefer the chosen answer, first and last. No
    start where the run's first score is not known (``first_of``): the last alone."""
    if not history:
        return metrics
    first, last = first_of(history, start), history[-1]
    if changed:  # the start was of other rows: never the "from" of these
        first = None
    out = {**metrics, **({"eval_loss_start": first["eval_loss"]} if first is not None else {}),
           "eval_loss": last["eval_loss"],
           **({"eval_start_note": ROWS_CHANGED} if changed else {})}
    if "ends" in last:
        out.update(eval_ends=last["ends"], eval_answer_rows=last["answer_rows"])
    if "prefers" in last:
        out["eval_prefers"] = last["prefers"]
        if first is not None:
            out["eval_prefers_start"] = first.get("prefers")
    return out
