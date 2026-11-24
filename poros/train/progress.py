"""PorosTrainer's events and receipts: the step event (events.jsonl), the run's metrics, the
reporter hooks, and the progress line and final summary on stdout."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from poros.core.events import PorosEvent
from poros.train.reporting import StepReport
from poros.train.state import _TrainerState

logger = logging.getLogger("poros.train.trainer")  # the trainer's own log, as before


@dataclass
class RunHistory:
    """What a run has done so far, one entry per optimizer update: what the metrics, the step
    events and the progress lines read. The loop appends to it; a new run starts a new one
    (``_Loop._reset_runtime_counters``)."""

    step_times: List[float] = field(default_factory=list)  # seconds per update
    vram_alloc_mb: List[float] = field(default_factory=list)  # allocated after each update
    losses: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    evals: List[Dict[str, Any]] = field(default_factory=list)  # each evaluation's metrics
    tokens: int = 0  # tokens trained on


class _Progress(_TrainerState):
    """PorosTrainer's step events, metrics and progress lines (mixed into PorosTrainer)."""

    def _compute_metrics(self) -> Dict[str, Any]:
        """Compute training metrics."""
        total_time = sum(self.history.step_times)
        total_steps = len(self.history.step_times)
        avg_step_time_ms = (
            (total_time / total_steps) * 1000.0 if total_steps else 0.0
        )
        final_loss = self.history.losses[-1] if self.history.losses else float("nan")

        peak_vram_allocated_gb = 0.0
        peak_vram_reserved_gb = 0.0
        if self.device.type == "cuda" and torch.cuda.is_available():
            peak_vram_allocated_gb = (
                torch.cuda.max_memory_allocated(self.device) / 1e9
            )
            peak_vram_reserved_gb = (
                torch.cuda.max_memory_reserved(self.device) / 1e9
            )

        return {
            "device": str(self.device),
            "total_steps": total_steps,
            "final_loss": final_loss,
            "loss_history": list(self.history.losses),
            "grad_norm_history": list(self.history.grad_norms),
            "eval_history": list(self.history.evals),
            "total_time_s": total_time,
            "step_time_history": list(self.history.step_times),
            "vram_alloc_history_mb": list(self.history.vram_alloc_mb),
            "avg_vram_allocated_mb": (
                sum(self.history.vram_alloc_mb) / len(self.history.vram_alloc_mb)
                if self.history.vram_alloc_mb else 0.0
            ),
            "avg_step_time_ms": avg_step_time_ms,
            "steps_per_sec": (
                (total_steps / total_time) if total_time > 0 else 0.0
            ),
            "tokens_processed": self.history.tokens,
            "tokens_per_sec": (
                (self.history.tokens / total_time) if total_time > 0 else 0.0
            ),
            "peak_vram_allocated_gb": peak_vram_allocated_gb,
            "peak_vram_reserved_gb": peak_vram_reserved_gb,
            "torch_num_threads": torch.get_num_threads(),
            "quantization": self.config.quantization,
            "block_size": self.config.block_size,
            "adapter_backend": self._adapter_backend,
            "adapter_summary": (
                dict(self._adapter_summary) if self._adapter_summary else None
            ),
        }

    # an SSD store's counters each step event carries (cumulative): reads, and how each was served
    # (a prefetch that landed, a synchronous read, the host cache), bytes delivered, the bytes
    # of them the host cache served (no disk read), seconds the loop waited on reads
    _SSS_STEP_KEYS = ("reads", "prefetch_hits", "sync_reads", "prefetch_dropped", "bytes_read",
                      "blocked_s", "cache_hits", "cache_bytes")

    def _sss_step_counters(self) -> Dict[str, Any]:
        """The SSD store's counters for a step event (``_SSS_STEP_KEYS``; a source without one,
        a store plugin's, leaves it out)."""
        stats = self._sss_source.stats
        return {k: stats[k] for k in self._SSS_STEP_KEYS if k in stats}

    def _emit_step_event(
        self, loss: float, step_time: float, extra: Optional[Dict[str, Any]] = None
    ) -> None:
        """Emit a JSONL step event (``extra``: a loss's own metrics)."""
        if self._event_writer is None:
            return
        vram_alloc_mb = 0.0
        vram_reserved_mb = 0.0
        device_fields = {}
        if self.device.type == "cuda" and torch.cuda.is_available():
            vram_alloc_mb = torch.cuda.memory_allocated(self.device) / 1e6
            vram_reserved_mb = torch.cuda.memory_reserved(self.device) / 1e6
            # the allocator's high-water mark: what a receipt's peak reads, where allocated alone
            # is the step's end, with its streamed blocks gone
            device_fields = {"vram_peak_mb": torch.cuda.max_memory_allocated(self.device) / 1e6}
        if self.device.type == "mps":  # one pool: the same two fields, and what only a Mac reads
            from poros.metal import machine

            sample = machine.memory_sample(self.device)
            vram_alloc_mb, vram_reserved_mb = sample.allocated / 1e6, sample.driver / 1e6
            device_fields = {
                "footprint": sample.footprint, "swap": sample.swap, "pressure": sample.pressure
            }

        lr = self.config.learning_rate
        if self._lr_scheduler is not None:
            lr = self._lr_scheduler.get_last_lr()[0]

        self._event_writer.emit(
            PorosEvent(
                event_type="step",
                step=self.global_step,
                data={
                    "loss": loss,
                    "grad_norm": (
                        self.history.grad_norms[-1] if self.history.grad_norms else 0.0
                    ),
                    "lr": lr,
                    "s_per_step": step_time,
                    "vram_allocated_mb": vram_alloc_mb,
                    "vram_reserved_mb": vram_reserved_mb,
                    "tokens": self.history.tokens,
                    **({"sss": self._sss_step_counters()}
                       if self._sss_source is not None else {}),
                    **(extra or {}),
                    **device_fields,
                },
            )
        )

    def _report(self, hook: str, *args: Any) -> None:
        """Call a reporter hook; a display bug must never stop training."""
        if self.reporter is None:
            return
        try:
            getattr(self.reporter, hook)(*args)
        except Exception:  # pragma: no cover - defensive
            logger.debug("reporter.%s failed", hook, exc_info=True)

    def _step_report(self, loss: float, step_time: float) -> StepReport:
        use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        lr = self.config.learning_rate
        if self._lr_scheduler is not None:
            lr = self._lr_scheduler.get_last_lr()[0]
        total_time = sum(self.history.step_times)
        report = StepReport(
            step=self.global_step + 1,
            total=self._total_optimizer_steps or (self.global_step + 1),
            loss=float(loss),
            step_time=step_time,
            tokens=self.history.tokens,
            tokens_per_sec=self.history.tokens / total_time if total_time > 0 else 0.0,
            lr=float(lr),
            vram_gb=torch.cuda.memory_allocated(self.device) / 1e9 if use_cuda else 0.0,
            vram_peak_gb=torch.cuda.max_memory_allocated(self.device) / 1e9 if use_cuda else 0.0,
            vram_reserved_gb=torch.cuda.memory_reserved(self.device) / 1e9 if use_cuda else 0.0,
        )
        if self.device.type == "mps":  # one pool: what the allocator holds, and the process's peak
            from poros.metal import machine

            sample = machine.memory_sample(self.device)
            report.vram_gb = sample.driver / 1e9
            report.vram_peak_gb = (sample.footprint or 0) / 1e9
        return report

    def _progress_enabled(self) -> bool:
        """Whether to emit the live stdout progress display.

        Progress goes to stdout, not to ``logger.info``: no example (nor the CLI)
        calls ``logging.basicConfig``, so a logged signal would print nothing.
        ``progress`` turns the stdout display on (default True); the
        ``POROS_PROGRESS`` env var overrides it (``POROS_PROGRESS=0`` forces it off).
        """
        env = os.environ.get("POROS_PROGRESS")
        if self.reporter is not None:  # the reporter owns the terminal
            return False
        if env is not None and env.strip() != "":
            return env.strip().lower() not in ("0", "false", "no", "off")
        return bool(getattr(self.config, "progress", True))

    def _logs_step(self, first: int) -> bool:
        """Whether the update just applied is one the progress line says: the first this run
        trains (``first``: its index; a resumed run's is past 0), then every ``logging_steps``
        steps done, and the run's last step, whatever its count."""
        done = self.global_step + 1
        return (self.global_step == first or done % self.config.logging_steps == 0
                or done == getattr(self, "_total_optimizer_steps", 0))

    @staticmethod
    def _progress_print(message: str) -> None:
        """Print a progress line to stdout, flushed so it shows under piping."""
        print(message, flush=True)

    def _log_step(self, loss: float, step_time: float) -> None:
        """Log a training step.

        When the progress display is on, a compact line goes to stdout and the
        detailed line is demoted to ``logger.debug`` (so a user who DID
        configure ``logging.basicConfig(INFO)`` does not see it twice). When
        off, the detailed line stays on ``logger.info`` -- exactly as before.
        """
        avg_loss = sum(self.history.losses[-100:]) / min(len(self.history.losses), 100)
        steps_per_sec = 1.0 / step_time if step_time > 0 else 0.0
        tokens_per_sec = (
            self.history.tokens / max(sum(self.history.step_times), 1e-12)
            if self.history.step_times
            else 0.0
        )
        use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        # memory is said in GiB on every line a person reads, as the CLI says it (the metrics'
        # and events' own fields keep their names and units)
        vram_used = (
            torch.cuda.memory_allocated(self.device) / 2**30 if use_cuda else 0.0
        )
        vram_reserved = (
            torch.cuda.memory_reserved(self.device) / 2**30 if use_cuda else 0.0
        )
        progress = self._progress_enabled()
        log_fn = logger.debug if progress else logger.info
        done = self.global_step + 1  # steps done, counted from 1, as the CLI's line says them
        log_fn(
            "Step %6d | Loss: %.4f (avg: %.4f) | %.2f steps/s | "
            "%.1f tok/s | VRAM: %.1f/%.1f GiB",
            done,
            loss,
            avg_loss,
            steps_per_sec,
            tokens_per_sec,
            vram_used,
            vram_reserved,
        )
        if progress:
            peak_gib = (
                torch.cuda.max_memory_allocated(self.device) / 2**30
                if use_cuda
                else 0.0
            )
            total = self._total_optimizer_steps or done
            self._progress_print(
                f"step {done:>4}/{total}  "
                f"loss {loss:.4f}  {step_time:.2f} s/step  peak {peak_gib:.1f} GiB"
            )

    def _log_final_summary(self, metrics: Dict[str, Any]) -> None:
        """Log the final training summary.

        The summary ALWAYS reaches stdout when the progress display is on
        (headline metric: peak VRAM), regardless of any logging configuration.
        When progress is off, it is emitted on ``logger.info`` exactly as before.
        """
        peak_vram = ""
        gib = 1e9 / 2**30  # the metrics count in GB; a person reads memory in GiB
        if metrics["peak_vram_reserved_gb"] > 0:
            peak_vram = (
                f"\nPeak VRAM alloc/reserved: "
                f"{metrics['peak_vram_allocated_gb'] * gib:.2f}/"
                f"{metrics['peak_vram_reserved_gb'] * gib:.2f} GiB"
            )
        progress = self._progress_enabled()
        log_fn = logger.debug if progress else logger.info
        log_fn(
            "\n%s\nPoros Training Complete\n%s\n"
            "Total steps: %s\n"
            "Final loss: %.4f\n"
            "Total time: %.1fs\n"
            "Avg step time: %.1fms\n"
            "Tokens processed: %s\n"
            "Tokens/sec: %.1f%s\n"
            "%s",
            "=" * 60,
            "=" * 60,
            metrics["total_steps"],
            metrics["final_loss"],
            metrics["total_time_s"],
            metrics["avg_step_time_ms"],
            metrics["tokens_processed"],
            metrics["tokens_per_sec"],
            peak_vram,
            "=" * 60,
        )
        if progress:
            bar = "=" * 60
            lines = [
                bar,
                "Poros Training Complete",
                bar,
                f"Total steps:      {metrics['total_steps']}",
                f"Final loss:       {metrics['final_loss']:.4f}",
                f"Total time:       {metrics['total_time_s']:.1f}s",
                f"Avg step time:    {metrics['avg_step_time_ms']:.1f}ms",
                f"Tokens processed: {metrics['tokens_processed']}",
                f"Tokens/sec:       {metrics['tokens_per_sec']:.1f}",
            ]
            if metrics["peak_vram_reserved_gb"] > 0:
                lines.append(
                    f"Peak VRAM:        "
                    f"{metrics['peak_vram_allocated_gb'] * gib:.2f}/"
                    f"{metrics['peak_vram_reserved_gb'] * gib:.2f} GiB (alloc/reserved)"
                )
            # where the run's events and checkpoints are: a script's run names no folder else
            lines.append(f"Output folder:    {self.config.output_dir}")
            lines.append(bar)
            self._progress_print("\n".join(lines))


# ── Module-level helpers ───────────────────────────────────────────────


# Canonical, paper-validated block size (every Poros canonical run used 4: the
# parity suite, 3090, Blackwell, and the seq-len sweep). block_size is a
# memory<->speed knob -- larger blocks transfer less often (faster) but keep
# more frozen base resident (more VRAM); smaller blocks save VRAM but are
# slower. It does NOT affect parity (0.00e+00 holds for any block_size). We do
# NOT guess it from free VRAM (an untested per-machine heuristic that could
# select a block too large for the card and OOM); users override explicitly.
