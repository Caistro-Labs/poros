"""JSONL event envelope for Poros training.

Every training step emits one JSON line to {output_dir}/events.jsonl.
The event schema is structured for downstream consumption by the report
and display subsystems. The file is append-only; a resumed run appends a
``resume`` marker, and ``current`` folds out the steps it supersedes.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

# What `poros schema events` states of ``data``. Memory is in decimal MB, as the trainer writes it.
DATA_FIELDS = {name: {"description": said} for name, said in {
    "loss": "step: the mean over this step's rows, so it swings with them",
    "grad_norm": "step: the gradient's norm before clipping",
    "lr": "step: the learning rate the next step trains at",
    "s_per_step": "step: the seconds this step took",
    "tokens": "step: a recipe's, the tokens this step trained on (a chat's answers, a pair's "
              "two); the classic loop's, the count so far",
    "vram_allocated_mb": "step: allocated at its end, its streamed blocks gone: not the peak",
    "vram_reserved_mb": "step: what the allocator holds from the driver at the step's end",
    "vram_peak_mb": "step, CUDA: the allocator's high-water mark, the measure a receipt records",
    "footprint": "step, a Mac: the process's peak in its one pool, in bytes",
    "swap": "step, a Mac: swap in use, in bytes",
    "pressure": "step, a Mac: the system's memory pressure: 1 normal, 2 warning, 4 critical",
    "checkpoint": "resume: the checkpoint the run resumed from; its step and eval events at or "
                  "past the event's step, written before it, are superseded",
    "unverified": "resume: what the resume could not check, in words, where there is any: the "
                  "data or the base at another path than an older checkpoint named, with no "
                  "record of its contents to compare",
    "eval_loss": "eval: the mean over the held-out rows' batches, before training, then after",
    "eval_perplexity": "eval: the exponential of eval_loss",
    "rows": "eval: how many rows it scored; setup: how many rows it trains on",
    "held": "eval: which held-out rows it scored, a fingerprint of them as the run read them; a "
            "run's first score is compared only with scores of the same rows",
    "path": "setup: the loop the run trains with: sequential, or block-major accumulation's "
            "efficient (block inputs in host RAM) or fast (on the GPU)",
    "path_reason": "setup: why the run's loop was decided sequential, where it was",
    "offload_embeddings_and_head": "setup: whether the embedding table and LM head were in host "
                                   "RAM (as set, or decided for the run)",
    "offload_reason": "setup: why offload_embeddings_and_head, left unset, was turned on",
    "fused_compute": "setup: whether the fused compute ran (Triton kernels for the adapter "
                     "layers, attention, norms and loss; the config's ran_compute)",
    "moe_expert_compute": "setup, MoE: the expert compute that ran: fused or reference",
    "compute_reason": "setup: why a default took the slower compute, where it did",
    "expert_optimizer": "setup: what steps bf16 trainable tensors, by device: ExpertAdamW's "
                        "kernel or its reference, with the reference's cost per update measured "
                        "at the start",
    "same_pairs": "setup, pairs: how many pairs train their chosen and rejected as the same "
                  "tokens, after the length cut; they teach nothing",
}.items()}


class PorosEvent(BaseModel):
    """Structured event for the JSONL stream."""

    event_type: Literal[
        "step",
        "eval",
        "checkpoint",
        "oom",
        "error",
        "block_load",
        "block_release",
        "prefetch",
        "parity_check",
        "setup",
        "teardown",
        "resume",
    ]
    timestamp: float = Field(default_factory=time.time)
    step: Optional[int] = None
    block_id: Optional[int] = None
    data: Dict[str, Any] = Field(default_factory=dict,
                                 json_schema_extra={"properties": DATA_FIELDS})

    # For step events:
    #   data = {loss, grad_norm, lr, s_per_step, vram_allocated_mb, vram_reserved_mb, vram_peak_mb,
    #           tokens}
    # For eval events:
    #   data = {step, eval_loss, eval_perplexity, eval_seconds}
    # For parity_check events:
    #   data = {loss_max_diff, weight_max_diff, bitwise_exact, arch}
    # For error events (E_STEP_CHECK): ``step`` is the index, from 0, of the update that stopped
    # the run, as a step event's is; its record counts that update from 1, as its message does:
    #   data = {code, nonfinite: {step (the index + 1), loss | grad_norm, torch_dtype}}
    #   (a Mac's remat that differs: {code, witness: {step (the index + 1), mismatches, first_layer,
    #   nonfinite_loss}}; a non-finite value alone is the nonfinite record on a Mac too)
    # For resume events: ``step`` is the first update the resumed run runs (a step event's
    #   index, from 0: the updates its checkpoint holds), data = {checkpoint, unverified (where
    #   the resume could not check its data or base)}. Step and eval events
    #   written before it at or past that step are superseded: ``current`` folds them out.


def resumes(event: Any) -> bool:
    """Whether ``event`` (a parsed events.jsonl line) is a resume marker."""
    return (isinstance(event, dict) and event.get("event_type") == "resume"
            and type(event.get("step")) is int)


def current(events: Any) -> list:
    """events.jsonl's lines (parsed dicts, in file order) as the run's record stands.

    The file is append-only: a resumed run never rewrites what its interrupted self wrote, so a
    reader tailing it by byte offset keeps its place. It appends a resume marker instead, and the
    step, eval and error events written before the marker at or past its ``step`` (the updates
    the interrupted run took, or stopped at, after its checkpoint, which the resumed run takes
    again) are superseded. Every other line is kept, the markers too.
    """
    out: list = []
    for event in events:
        if resumes(event):
            at = event["step"]
            out = [e for e in out if not (
                isinstance(e, dict) and e.get("event_type") in ("step", "eval", "error")
                and type(e.get("step")) is int and e["step"] >= at)]
        out.append(event)
    return out


class EventWriter:
    """Writes PorosEvent objects as JSONL to a file.

    Args:
        output_dir: Directory to write events.jsonl to.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self._path = Path(output_dir) / "events.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: PorosEvent) -> None:
        """Append an event to the JSONL file.

        Args:
            event: The event to write.
        """
        with open(self._path, "a") as f:
            f.write(event.model_dump_json() + "\n")
