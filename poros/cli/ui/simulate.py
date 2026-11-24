"""A stand-in trainer for previewing the CLI on a machine without a GPU.

Enabled only by ``POROS_DEV_SIMULATE=1`` (scripts/cli/try sets it when
nvidia-smi is missing). ``poros train`` then runs its real path -- preflight,
load activity, live surface, receipt, summary -- against this object, which
drives the same reporter hooks the real trainer does. Every number is
synthetic and the welcome line says "simulated", so a preview can never be
mistaken for a measurement. No adapter is written.
"""

from __future__ import annotations

import functools
import math
import os
import random
import sys
import time
from typing import Any, Callable, Dict, Optional

from poros.train.reporting import StepReport


def _card() -> tuple:
    """The card it stands in for: ``POROS_DEV_SIMULATE_GPU="RTX 5090:31.8"``; else on a Mac the
    Mac itself, its own chip and memory; else an RTX 4090."""
    name, _, gib = os.environ.get("POROS_DEV_SIMULATE_GPU", "").rpartition(":")
    if name.strip():
        try:
            return (f"{name.strip()} (simulated)", float(gib))
        except ValueError:
            return _DEFAULT
    mac = this_mac()
    return (f"{mac.name} (simulated)", mac.total_gib) if mac is not None else _DEFAULT


_DEFAULT = ("RTX 4090 (simulated)", 24.0)


def this_mac() -> Any:
    """The Mac this runs on, as /machine reads it, when a preview stands in for it (no card asked
    for with POROS_DEV_SIMULATE_GPU): a Mac has hardware of its own to show. None elsewhere."""
    if sys.platform != "darwin" or os.environ.get("POROS_DEV_SIMULATE_GPU", "").strip():
        return None
    from poros.core.card import detect_gpu

    return detect_gpu()


@functools.lru_cache(maxsize=None)
def _gpu() -> tuple:
    return _card()


def __getattr__(name: str) -> Any:
    if name == "GPU":  # read when asked, not at import: finding a Mac's chip takes a moment
        return _gpu()
    raise AttributeError(name)


def enabled() -> bool:
    return os.environ.get("POROS_DEV_SIMULATE", "") not in ("", "0")


class _BlockManager:
    def __init__(self, layers: int, block: int) -> None:
        self.num_layers = layers
        self.num_blocks = max(1, layers // block)
        self.active_block_id: Optional[int] = None
        self.h2d_bytes_total = 0


# billions of parameters -> decoder layers, Qwen2.5's (Llama's 8B and 70B agree)
_LAYERS = ((0.6, 24), (1.6, 28), (3.5, 36), (8.5, 28), (15, 48), (33, 64), (80, 80))


def _layers(model: str) -> int:
    """A plausible layer count for the model the run names, from its size."""
    import re

    size = re.search(r"(\d+(?:\.\d+)?)[bB](?![a-z])", str(model).split("/")[-1])
    if not size:
        return 32
    b = float(size.group(1))
    return next((n for top, n in _LAYERS if b <= top), 80)


class SimTrainer:
    """Mimics the parts of PorosTrainer that ``poros train`` touches."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.reporter = None
        self.block_manager = _BlockManager(_layers(cfg.model_name_or_path), cfg.block_size or 4)
        self.model = None
        self.recipe = "classic"  # train_cmd sets the recipe it picked
        # memory follows how much of the base stays on the card: a few blocks
        # streamed, or all of it resident (prove's resident arm keeps 24)
        resident = min(1.0, (cfg.block_size or 4) / 24)
        self.peak_gb = round(4.6 + 13.5 * resident, 1)

    @classmethod
    def from_pretrained(cls, model: str, **kw: Any) -> "SimTrainer":
        time.sleep(float(os.environ.get("POROS_DEV_SIMULATE_LOAD", "2.5")))
        return cls(kw["config"])

    def _event(self, step: int, loss: float, step_t: float, tokens: int,
               extra: Optional[Dict[str, float]] = None) -> None:
        """A step event, as the real trainer writes it, so `poros runs` can read it."""
        import json

        out = os.path.join(self.cfg.output_dir, "events.jsonl")
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        with open(out, "a") as f:
            f.write(json.dumps({"event_type": "step", "timestamp": time.time(), "step": step,
                                "data": {"loss": loss, "s_per_step": step_t,
                                         "vram_allocated_mb": (self.peak_gb - 0.7) * 1000,
                                         "vram_peak_mb": self.peak_gb * 1000,
                                         "tokens": tokens, "simulated": True,
                                **(extra or {})}}) + "\n")

    def _said(self, loss: float, rows: int, done: float) -> dict:
        """A pass's numbers, as a recipe reports them: on chats the answers that end, on pairs
        the rows that prefer the chosen one, rising as the run goes (``done`` 0 to 1)."""
        said: dict = {"eval_loss": loss}
        if not rows or self.recipe == "classic":  # the engine's loop scores the loss alone
            return said
        if self.recipe in ("dpo", "ipo", "kto", "orpo", "simpo"):
            said.update(prefers=round(rows * (0.5 + 0.42 * done)), rows=rows)
        else:
            said.update(ends=round(rows * (0.2 + 0.74 * done)), answer_rows=rows)
        return said

    def _scored(self, step: int, said: dict, rows: int) -> None:
        """An eval event, as a recipe writes it."""
        import json

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        with open(os.path.join(self.cfg.output_dir, "events.jsonl"), "a") as f:
            f.write(json.dumps({"event_type": "eval", "timestamp": time.time(), "step": step,
                                "data": {**said, "rows": rows or None,
                                         "simulated": True}}) + "\n")

    def train(
        self,
        dataset: Any,
        steps: Optional[int] = None,
        seq_len: int = 1024,
        resume_from: Optional[str] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        eval_dataset: Optional[str] = None,
    ) -> Dict[str, Any]:
        total = min(steps or 120, 400)
        dt = float(os.environ.get("POROS_DEV_SIMULATE_STEP", "0.35"))
        rng = random.Random(7)
        # The loss has its own stream: like the real math, it must not depend
        # on how many blocks the stream walks (`poros prove` compares arms).
        loss_rng = random.Random(7)
        bm = self.block_manager
        rep = self.reporter
        start = 0
        if resume_from:  # a checkpoint-<index> folder: carry on from the update after it
            tail = os.path.basename(str(resume_from).rstrip("/")).rpartition("-")[2]
            start = min(int(tail) + 1, total) if tail.isdigit() else 0  # all done: none left
        if rep is not None:
            rep.on_start(self, total, start)
        from poros import data

        held_file = getattr(self.cfg, "eval_dataset", None)
        if self.recipe == "classic":  # the engine's own loop holds no rows out: it scores a file
            rows = _file_rows(held_file)
            held = bool(held_file)
        else:
            rows = data.held_rows(self.cfg)
            held = bool(held_file) or rows > 0
        from poros.train.evals import read_start, saves_at, scores_at, with_eval

        pairs = self.recipe in ("dpo", "ipo", "kto", "orpo", "simpo")
        base = 0.6931 if pairs else 2.91  # a pair loop's loss starts at log 2
        history: list = []  # each score, by step, as the trainer's history keeps them
        # the run's first score: before its first step, or as the checkpoint it resumes keeps it
        first = read_start(resume_from) if resume_from else None
        if held and not start:
            first = self._said(base, rows, 0.0)
            history.append({"step": 0, **first})
            self._scored(0, first, rows)
            if rep is not None:
                rep.on_eval(0, first)
        losses, times = [], []
        tokens = 0
        try:
            for i in range(start, total):
                t0 = time.perf_counter()
                for b in range(bm.num_blocks):  # the stream walks the blocks each step
                    bm.active_block_id = b
                    bm.h2d_bytes_total += int(21.4e9 * dt / bm.num_blocks * rng.uniform(0.9, 1.1))
                    time.sleep(dt / bm.num_blocks)
                bm.active_block_id = None
                loss = 1.1 + 1.75 * math.exp(-i / max(total / 4, 1)) + loss_rng.uniform(-0.06, 0.06)
                extra = {}
                # a pair loop starts at log 2, and the margin opens up
                if self.recipe in ("dpo", "ipo", "kto", "orpo", "simpo"):
                    grown = 1 - math.exp(-i / max(total / 3, 1))
                    margin = 2.4 * grown + loss_rng.uniform(-0.08, 0.08)
                    loss = math.log1p(math.exp(-margin))
                    acc = 0.5 + 0.42 * grown + loss_rng.uniform(-0.04, 0.04)
                    extra = {"accuracy": min(1.0, acc),
                             "margin": margin, "chosen_reward": 0.6 * margin,
                             "rejected_reward": -0.4 * margin}
                losses.append(loss)
                step_t = time.perf_counter() - t0
                times.append(step_t)
                tokens += seq_len * 4
                self._event(i, loss, step_t, tokens, extra)  # 0-based, as the trainer writes
                if rep is not None:
                    rep.on_step(
                        StepReport(
                            step=i + 1,
                            total=total,
                            loss=loss,
                            step_time=step_t,
                            tokens=tokens,
                            tokens_per_sec=tokens / sum(times),
                            lr=2e-4,
                            vram_gb=self.peak_gb - 0.7 + rng.uniform(-0.25, 0.25),
                            vram_peak_gb=self.peak_gb,
                            vram_reserved_gb=self.peak_gb + 0.9,
                            extra=extra,
                        )
                    )
                # where the trainer scores and saves (poros.train.evals, the rules the loops
                # call): scored before the save, every eval_steps, at a save, after the last step
                said = None
                if held and (i + 1 >= total or scores_at(self.cfg, i, total)):
                    said = self._said(round(loss + 0.12, 4), rows, (i + 1) / total)
                    history.append({"step": i, **said})
                    self._scored(i, said, rows)
                    if rep is not None:
                        rep.on_eval(i, said)
                if saves_at(self.cfg, i):
                    saved = self._checkpoint(i, said, first, rows)
                    if rep is not None:
                        rep.on_checkpoint(i, saved)
                if should_stop is not None and i + 1 < total and should_stop():
                    from poros.core.errors import PorosCancelledError

                    raise PorosCancelledError(i, self._checkpoint(i, said, first, rows))
        except KeyboardInterrupt:
            if rep is not None:
                rep.on_interrupt(len(losses), None)
            raise
        if not losses:  # resumed with no step left, as the trainer returns: nothing ran
            return {"total_steps": 0, "final_loss": float("nan"), "loss_history": []}
        # like a recipe's last save: train more starts here, with the last score in it
        self._checkpoint(total - 1, said, first, rows)  # said: the last step's score
        metrics = {
            "total_steps": total,
            "final_loss": losses[-1],
            "loss_history": losses,
            "avg_step_time_ms": sum(times) / len(times) * 1000,
            "tokens_per_sec": tokens / sum(times),
            "peak_vram_allocated_gb": self.peak_gb,
            "peak_vram_reserved_gb": self.peak_gb + 0.9,
        }
        if held:  # first and last, as the trainer says them: no start where none is known
            metrics = with_eval(metrics, history, first)
        if rep is not None:
            rep.on_done(metrics)
        return metrics

    def _checkpoint(self, step: int, said: Optional[dict] = None, first: Optional[dict] = None,
                    rows: int = 0) -> str:
        """An empty but complete checkpoint folder, named by the last update's index
        (``checkpoint-15`` holds 16 steps) as the trainer names it, so stop, continue and
        train more behave as they do on a GPU; with the score of its step (``said``) and the
        run's first (``first``) beside it, as the trainer keeps them."""
        from pathlib import Path

        from poros.core.checkpoints import COMPLETE_MARKER
        from poros.train.evals import write_score, write_start

        path = os.path.join(self.cfg.output_dir, f"checkpoint-{step}")
        os.makedirs(path, exist_ok=True)
        open(os.path.join(path, COMPLETE_MARKER), "w").close()
        if said is not None:
            write_score(Path(path), {**said, **({"rows": rows} if rows else {})})
        if first is not None:
            write_start(Path(path), {**first, **({"rows": rows} if rows else {})})
        return path

    def save(self, output_dir: str, **kw: Any) -> None:
        os.makedirs(output_dir, exist_ok=True)
        time.sleep(0.4)


def _file_rows(path: Optional[str]) -> int:
    """How many rows a held-out file holds (0: none, or not a local file)."""
    from pathlib import Path

    from poros import data

    if not path or not Path(str(path)).is_file():
        return 0
    try:
        return sum(1 for _ in data._rows(Path(str(path))))
    except (OSError, ValueError):
        return 0
