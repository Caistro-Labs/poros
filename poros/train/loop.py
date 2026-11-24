"""PorosTrainer's training loop: an update's micro-batches through the sequential loop or
the block-major driver (``poros.train.block_major``), the passes' random states, the memory
watchdog, the optimizer and its schedule, the non-finite stop, batches and the dataloader."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, Dataset

from poros.core import checkpoints
from poros.core.config import train_path
from poros.core.errors import (
    PorosCancelledError,
    PorosConfigError,
    PorosOOMError,
    PorosStepCheckError,
)
from poros.core.events import PorosEvent
from poros.train import block_major
from poros.train.datasets import (
    _build_synthetic_dataloader,
    _load_hf_dataset_as_dataloader,
    _synthetic_sample_count,
)
from poros.train.evals import (
    held_rows,
    kept_rng,
    saves_at,
    scores_at,
    scoring_out_of_memory,
    settle_start,
    with_eval,
)
from poros.train.progress import RunHistory
from poros.train.state import _TrainerState

logger = logging.getLogger("poros.train.trainer")  # the trainer's own log, as before
# An eval pass's rows of tokens per chunk of logits; 0: half a training head chunk (tests set it)
_EVAL_ROWS = 0


class _Loop(_TrainerState):
    """PorosTrainer's training loop (mixed into PorosTrainer)."""

    def _train_blockwise_adapters(
        self,
        dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> Dict[str, Any]:
        """Run the blockwise adapter training loop."""
        config = self.config
        assert self.optimizer is not None

        self._reset_runtime_counters()
        num_optimizer_steps = config.max_steps if config.max_steps > 0 else (
            self._epoch_steps(len(dataloader))
        )
        # an epoch-based run reads each row once an epoch: its last update takes the micro-batches
        # left (fewer than G where they do not divide), never rows from the start again
        epoch_micro = (len(dataloader) * config.num_epochs if config.max_steps <= 0 else None)
        self._total_optimizer_steps = num_optimizer_steps  # for progress display
        start_step = self._resume_step
        if start_step >= num_optimizer_steps:
            logger.info(
                "Resume step %d >= target %d; nothing to do.",
                start_step, num_optimizer_steps,
            )
            return self._compute_metrics()
        if start_step > 0:
            logger.info(
                "Resuming Poros blockwise training at step %d / %d",
                start_step, num_optimizer_steps,
            )
            # Fast-forward the data stream so resumed steps see fresh batches.
            # Its RNG draws (an iterator over a DataLoader without a
            # generator, such as the synthetic one, seeds itself from the
            # global CPU RNG) happened before the checkpoint saved that RNG,
            # so they must not advance the restored stream.
            with torch.random.fork_rng(devices=[]):
                data_iter = iter(dataloader)
                for _ in range(start_step * config.gradient_accumulation_steps):
                    _, data_iter = self._next_batch(data_iter, dataloader)
        else:
            logger.info(
                "Starting Poros blockwise training: %s optimizer steps",
                num_optimizer_steps,
            )
            data_iter = iter(dataloader)

        # Adopt or discard staging leftovers from a previously killed save
        # BEFORE anything enumerates checkpoints.
        checkpoints.recover_checkpoint_dirs(Path(config.output_dir))

        self.optimizer.zero_grad(set_to_none=True)

        # Interrupt reporting: Ctrl-C says where the last valid checkpoint is,
        # so nobody assumes the work is gone and re-runs from scratch. Handled
        # with try/except rather than a signal handler: no process-wide state
        # to leak if train() raises, and no filesystem I/O inside a signal
        # handler.
        def _interrupt_note() -> None:
            try:
                ckpts = self._own_checkpoints()
            except Exception:  # pragma: no cover - never mask the interrupt
                ckpts = []
            # global_step is the index of the update under way: as many are done
            done = f"{self.global_step} step{'' if self.global_step == 1 else 's'}"
            if ckpts:
                step, path = ckpts[-1]  # named by its last update's index: it holds one more
                msg = (
                    f"\ninterrupted after {done}. {path.name} ({step + 1} steps) is "
                    "complete; resume with:\n"
                    "  poros train <config.yaml> --resume-from latest"
                )
            else:
                msg = (
                    f"\ninterrupted after {done}. No checkpoint was written yet -- set "
                    "save_steps in the config to make long runs resumable."
                )
            if self.reporter is not None:
                self._report("on_interrupt", self.global_step,
                             str(ckpts[-1][1]) if ckpts else None)
                return
            print(msg, flush=True)

        if eval_dataloader is not None and start_step == 0:  # the model it starts from; a
            # resume scored it when it began
            self._score_held_out(eval_dataloader, 0)
        self._report("on_start", self, num_optimizer_steps, start_step)
        self._start_watchdog()
        try:
            for self.global_step in range(start_step, num_optimizer_steps):
                step_start = time.perf_counter()
                accumulated_loss = 0.0

                try:
                    G = config.gradient_accumulation_steps
                    if epoch_micro is not None:  # the last update: what is left of the epochs
                        G = min(G, epoch_micro - self.global_step * G)
                    self.begin_passes(self.global_step)
                    rng = block_major.rng_for(self, G)
                    if rng is not None:
                        batches = []
                        for g in range(G):
                            self.before_fetch(g)
                            batch, data_iter = self._next_batch(
                                data_iter, dataloader
                            )
                            batch = self._move_batch_to_device(batch)
                            self.history.tokens += self._count_batch_tokens(
                                batch
                            )
                            batches.append(batch)
                            rng.add(block_major.positions(batch))  # where its forward draws
                        losses = block_major.update(self, batches, rng)
                        if losses is None:  # stopped before applying anything: sequentially
                            losses = [self._sequential_pass(b, g, G)
                                      for g, b in enumerate(batches)]
                            self.end_passes()
                        # summed as the sequential loop below sums them (in the loss's own
                        # dtype: bf16/fp16 losses round), so the logged loss is the same bits
                        for loss in losses:
                            accumulated_loss = accumulated_loss + loss
                    else:
                        for g in range(G):
                            self.before_fetch(g)
                            batch, data_iter = self._next_batch(
                                data_iter, dataloader
                            )
                            batch = self._move_batch_to_device(batch)
                            self.history.tokens += self._count_batch_tokens(
                                batch
                            )
                            accumulated_loss = accumulated_loss + self._sequential_pass(
                                batch, g, G)
                        self.end_passes()
                except torch.cuda.OutOfMemoryError as exc:
                    if self.device.type == "mps":
                        raise
                    # Turn a raw CUDA OOM into an actionable Poros error with a
                    # concrete block_size/seq_len suggestion.
                    self.optimizer.zero_grad(set_to_none=True)
                    self._release_retained_forward_blocks()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise PorosOOMError(
                        block_size=config.block_size,
                        seq_len=config.max_seq_length,
                    ) from exc

                avg_loss = float(accumulated_loss) / G
                if self._witness is not None:
                    self._witness.check(self.global_step, avg_loss)
                self._stop_if_nonfinite(loss=avg_loss)
                self._adapter_optimizer_step()
                self._release_retained_forward_blocks()

                step_time = time.perf_counter() - step_start
                self.history.step_times.append(step_time)
                if self.device.type == "cuda" and torch.cuda.is_available():
                    self.history.vram_alloc_mb.append(
                        torch.cuda.memory_allocated(self.device) / 1e6
                    )
                self.history.losses.append(avg_loss)

                # Emit step event
                self._emit_step_event(avg_loss, step_time)
                if self.reporter is not None:
                    self._report("on_step", self._step_report(avg_loss, step_time))

                if self._logs_step(start_step):
                    self._log_step(avg_loss, step_time)

                last = self.global_step + 1 >= num_optimizer_steps
                # Scored before the save, so the checkpoint keeps the score of its own weights;
                # once a step (poros.train.evals.scores_at), and after the last step always.
                if eval_dataloader is not None and (
                        last or scores_at(config, self.global_step, num_optimizer_steps)):
                    self._score_held_out(eval_dataloader, self.global_step)

                if saves_at(config, self.global_step):
                    self._report("on_checkpoint", self.global_step, self._save_checkpoint())

                # After this step's eval and save, so this checkpoint holds the state the run
                # continues from. Not after the last step: that run is finished, not cancelled.
                if (
                    should_stop is not None
                    and self.global_step + 1 < num_optimizer_steps
                    and should_stop()
                ):
                    raise PorosCancelledError(self.global_step, self._save_checkpoint())
                if self.global_step + 1 < num_optimizer_steps:
                    self._stop_if_out_of_memory()

        except KeyboardInterrupt:
            _interrupt_note()
            raise
        except RuntimeError as exc:
            # mps: the allocator's OOM is a plain RuntimeError, and a failed command
            # buffer raises at the next sync, which may be the loss read or the
            # optimizer step, past the microstep handler above.
            self._raise_if_mps_oom(exc)
            raise
        finally:
            self._stop_watchdog()

        logger.info("Training complete")
        # the held-out loss before the first step and after the last, as a recipe's run says them
        metrics = with_eval(self._compute_metrics(), self.history.evals, self._held_start,
                            self._held_changed)
        self._log_final_summary(metrics)
        self._report("on_done", metrics)

        # Emit teardown event
        if self._event_writer is not None:
            self._event_writer.emit(
                PorosEvent(event_type="teardown", data=metrics)
            )

        return metrics

    def _start_watchdog(self) -> None:
        if self.device.type == "mps":
            from poros.metal.watchdog import Watchdog

            self._watchdog = Watchdog(guard=self.config.memory_guard)

    def _stop_watchdog(self) -> None:
        if self._watchdog is not None:
            self._watchdog.stop()

    def _stop_if_out_of_memory(self) -> None:
        """A step boundary on a Mac under critical pressure: keep the run, and stop it."""
        if self._watchdog is not None and self._watchdog.fired.is_set():
            from poros.metal.watchdog import checkpoint_first, oom_error

            checkpoint_first(self)
            raise oom_error(self, self._watchdog.why)

    def _raise_if_mps_oom(self, exc: BaseException) -> None:
        if self.device.type == "mps":
            from poros.metal.watchdog import is_mps_oom, oom_error

            if is_mps_oom(exc):
                raise oom_error(self) from exc

    # ── Forward pass ───────────────────────────────────────────────────

    def _sss_after_pass(self) -> None:
        """One forward+backward ended: drive SSD schedule replay (each pass's reads become the
        next pass's schedule); after the first, record how the store was tuned. The device
        expert cache is not sized here: like the RAM store's, it is sized before update 2, on
        the peak through update 1 and its optimizer step (RunCompute.cache_before_update), which
        a pass's end comes before."""
        from poros.sss.pinning import staging
        from poros.sss.pinning import status as pinning_status
        from poros.train.compute import _residency

        source = self._sss_source
        source.mark_step()
        if self._sss_tuned:
            return
        self._sss_tuned = True
        residency = _residency(self.model) if self.model is not None else None
        capacity = None if residency is None else int(getattr(residency, "_capacity", 0))
        stats = source.stats
        info = {
            "store": "ssd", "store_path": self.config.store_path,
            "depth": stats["depth"], "direct": stats["direct"], "pinned": stats["pinned"],
            "ring_bytes": stats["ring_bytes"], "trace_len": stats["trace_len"],
            "compute_s_per_read": stats["compute_s_per_read"],
            "expert_capacity_bytes": capacity,
            "pinned_bytes": stats.get("pinned_bytes"),
            "pinning": pinning_status(),
            "staging": staging(ring_pinned=stats.get("pinned")),
        }
        if self._sss_probe is not None:
            info.update(b_ssd=self._sss_probe.b_ssd, b_h2d=self._sss_probe.b_h2d,
                        probe_direct=self._sss_probe.direct)
        logger.info("SSS tuned after the first pass: %s", info)
        if self._event_writer is not None:
            self._event_writer.emit(PorosEvent(event_type="setup", step=self.global_step,
                                               data={"sss": info}))

    def _remat_scope(self, block_id: int) -> Any:
        """Context for a block's rematerialization whose output only roots ``autograd.grad``: the
        fused compute skips the last frozen GEMM no gradient reads (poros.core.fused)."""
        if not self.compute.fused_ran or self.block_manager is None:
            return contextlib.nullcontext()
        from poros.core import fused

        return fused.output_unused(self.block_manager.get_block_modules(block_id))

    def block_major_run_step(
        self, feeds: List[Dict[str, torch.Tensor]],
        head_fns: List[Callable[[torch.Tensor], torch.Tensor]],
    ) -> bool:
        """One update's micro-batches through the block-major driver for ``poros.Run``
        (poros.train.block_major), between the caller's ``begin_passes`` and its step.

        ``head_fns[g](logits)`` returns micro-batch g's already-weighted loss; its gradient
        flows back exactly as ``(loss).backward()`` per micro-batch would. Returns False where
        the caller runs them sequentially instead (``block_major.rng_for``: the sequential
        path, the Mac, ...; or a draw the walk cannot follow stopped the update before anything
        was applied).
        """
        rng = block_major.rng_for(self, len(feeds))
        if rng is None:
            return False
        for feed in feeds:
            rng.add(block_major.positions(feed))
        return block_major.update(self, feeds, rng, head_fns) is not None

    # -- An update's random states and the loop it ran --------------------------------------

    def begin_passes(self, update: int, done: int = 0) -> None:
        """An update's forward and backward passes begin. A MoE model's random states are keyed
        by (update, micro-batch, layer) from here (``keyed_rng``); a dense model's need nothing.
        ``done``: micro-batches this update already ran (``poros.Run``'s repeated calls)."""
        if self.keyed_rng is not None:
            self.keyed_rng.begin(update, done)

    def before_fetch(self, g: int) -> None:
        """Micro-batch g of the update is fetched next (where a new epoch's shuffle draws)."""
        if self.keyed_rng is not None:
            self.keyed_rng.fetch(g)

    def before_pass(self, g: int) -> None:
        """Micro-batch g's forward and backward run next, sequentially."""
        if self.keyed_rng is not None:
            self.keyed_rng.start(g)

    def end_passes(self) -> None:
        """The update's micro-batches ran sequentially: a MoE model's generators go where the
        update ends, as block-major leaves them."""
        if self.keyed_rng is not None:
            self.keyed_rng.end()
        if self._sss_pass_shapes:  # the SSD store's sequential update, for the expert cache
            shapes, self._sss_pass_shapes = self._sss_pass_shapes, []
            self.compute.cache_after_update(self.model, shapes)
        self._ran_loop(block_major=False)

    def sss_pass_begins(self, batch: Dict[str, torch.Tensor]) -> None:
        """A sequential pass of one micro-batch begins (the classic loop's and ``poros.Run``'s):
        on a MoE model's SSD store the expert cache is sized before it -- before update 2's first
        pass, on update 1 measured through its step -- and its shape kept for ``end_passes``."""
        if self._sss_source is not None and self.block_manager is None:  # a MoE SSD store
            shape = block_major._shape(batch)
            self.compute.cache_before_update(self.model, [shape])
            self._sss_pass_shapes.append(shape)

    def sss_pass_ends(self) -> None:
        """A sequential pass's forward and backward ended: the SSD store's reads of it become
        the next pass's schedule (``_sss_after_pass``)."""
        if self._sss_source is not None:
            self._sss_after_pass()

    def _sequential_pass(self, batch: Dict[str, torch.Tensor], g: int, G: int) -> torch.Tensor:
        """Micro-batch g of G: forward, then ``loss / G`` backward. Returns the loss, detached."""
        self.before_pass(g)
        self.sss_pass_begins(batch)
        loss = self._forward_blockwise_adapter_model(batch)
        (loss / G).backward()
        self.sss_pass_ends()
        out = loss.detach()
        # One pool on a Mac: a loss still bound through the next forward keeps its graph, and
        # every saved boundary with it.
        del loss
        return out

    def _ran_loop(self, *, block_major: bool) -> None:
        """An update's micro-batches ran block-major or sequentially."""
        self._block_major_ran = bool(self._block_major_ran) or block_major

    @property
    def ran_path(self) -> Optional[str]:
        """The loop the run's updates took: ``path`` where any ran block-major, ``sequential``
        where none did (the sequential path, or block-major gave up on the first update), None
        before the first update. The records' ``ran_path``; ``path`` stays as set."""
        if self._block_major_ran is None:
            return None
        return train_path(self.config) if self._block_major_ran else "sequential"

    def _moe_block_manager(self) -> Optional[Any]:
        """The dense engine's BlockManager a ``poros.moe`` model streams its dense blocks with
        (``prepare_moe`` -> ``patch_hf_model``), or None (not MoE / dense parts resident)."""
        if self.model is None or self.block_manager is not None:
            return None
        return getattr(self.model, "_poros_block_manager", None)

    def _get_nf4_dequant_cache(self) -> Optional[Any]:
        """The run's DequantCache (config.nf4_dequant_cache_mb > 0, or auto), built once."""
        raw = getattr(self.config, "nf4_dequant_cache_mb", 0)
        if raw is None:
            if self._auto_cache_mb is None:
                self._auto_cache_mb = self.compute.auto_dequant_cache_mb()
                logger.info("nf4_dequant_cache_mb auto: %d MiB", self._auto_cache_mb)
            raw = self._auto_cache_mb
        mb = int(raw or 0)
        self.compute.record_cache(nf4_dequant_cache_mb=mb)
        if mb <= 0:
            return None
        cache = self._nf4_dequant_cache
        if cache is None:
            from poros.core.nf4_cache import DequantCache

            cache = DequantCache(mb << 20)
            self._nf4_dequant_cache = cache
        return cache if cache.enabled else None

    def _get_adapter_prefetch_stream(self) -> Optional[torch.cuda.Stream]:
        """Return the prefetch CUDA stream, or None if unavailable."""
        if not self.config.enable_prefetch:
            return None
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return None
        return self._prefetch_stream

    def _record_remat_event(self, event: str, block_id: int) -> None:
        """Record a rematerialization event for debugging."""
        if self.config.debug_memory:
            self._remat_event_log.append((event, block_id))

    def _release_retained_forward_blocks(self) -> None:
        """Release blocks retained during forward for suffix path."""
        if self.block_manager is not None:
            for block_id in self._retained_forward_blocks:
                self.block_manager.release_block(block_id)
        self._retained_forward_blocks = []

    def _record_optimizer(self) -> None:
        """What steps the adapters, in the run's records (ran_compute's expert_optimizer, beside
        the expert compute): ExpertAdamW's kernel or reference, by device. Where a GPU runs the
        reference, its cost measured now goes into compute_reason and is said at once: the
        setup event, config.resolved.json and the start screen carry it from there."""
        impl = getattr(self.optimizer, "implementation", None)
        compute = getattr(self, "compute", None)
        if not impl or compute is None:
            return
        notice = getattr(self.optimizer, "notice", None)
        if notice:
            logger.warning("%s", notice)
            compute.reasons.append(notice)
        compute.note(expert_optimizer={
            device: {k: v for k, v in i.items() if k in ("path", "seconds_per_update")}
            for device, i in impl.items()})

    def _initialize_optimizer(self) -> None:
        """Initialize AdamW optimizer and optional LR scheduler."""
        config = self.config
        self.trainable_parameter_names = [
            name
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]
        self.trainable_parameters = [
            param for param in self.model.parameters() if param.requires_grad
        ]
        if not self.trainable_parameters:
            raise RuntimeError(
                "No trainable parameters found after adapter setup"
            )

        # fp32 adapters (a dense run): torch's AdamW, as always. bf16 ones (a MoE model's expert
        # adapters): ExpertAdamW, which keeps their small updates and second moment.
        from poros.train.expert_adamw import build_optimizer

        self.optimizer = build_optimizer(
            [(name, param) for name, param in self.model.named_parameters()
             if param.requires_grad],
            config,
        )
        self._record_optimizer()

        # LR scheduler. max_steps is -1 (placeholder 1000) until train()
        # applies its steps override; train() rebuilds this on the effective
        # step count via _rebuild_lr_scheduler_for_run (see the note there).
        max_steps = config.max_steps if config.max_steps > 0 else 1000
        self._build_lr_scheduler(config.warmup_for(max_steps), max_steps)

        self.model.train()

    def _build_lr_scheduler(self, warmup: int, max_steps: int) -> None:
        """(Re)build ``self._lr_scheduler`` for a warmup/total-step budget.

        Shared by :meth:`_initialize_optimizer` (initial build) and
        :meth:`_rebuild_lr_scheduler_for_run` (the train-time rebuild) so the
        scheduler type and lambda shape are guaranteed identical -- only the
        ``warmup`` and ``max_steps`` the lambda closes over differ.
        """
        config = self.config
        sched_type = config.lr_scheduler

        if sched_type == "constant":
            self._lr_scheduler = None
            return

        from torch.optim.lr_scheduler import LambdaLR

        if sched_type == "cosine":

            def lr_lambda(step: int) -> float:
                if step < warmup:
                    return float(step) / max(1, warmup)
                progress = float(step - warmup) / max(1, max_steps - warmup)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        elif sched_type == "linear":

            def lr_lambda(step: int) -> float:
                if step < warmup:
                    return float(step) / max(1, warmup)
                return max(
                    0.0,
                    1.0 - float(step - warmup) / max(1, max_steps - warmup),
                )

        else:
            self._lr_scheduler = None
            return

        self._lr_scheduler = LambdaLR(self.optimizer, lr_lambda)
        logger.info(
            "LR scheduler: %s (warmup=%d, max_steps=%d)",
            sched_type,
            warmup,
            max_steps,
        )

    def _rebuild_lr_scheduler_for_run(self, effective_steps: int) -> None:
        """Rebuild the LR scheduler for the ACTUAL run length.

        The scheduler is first built at ``from_pretrained`` time (in
        :meth:`_initialize_optimizer`), BEFORE ``train()`` assigns the ``steps``
        override, so it holds ``warmup=warmup_steps`` against the placeholder
        ``max_steps=1000``: a short run on it would spend most (or all) of its
        length inside warmup at a near-zero learning rate. Rebuilding here on the
        effective step count clamps warmup to the run, so a run never spends more
        than ~10% of its steps ramping.
        """
        self._build_lr_scheduler(self.config.warmup_for(effective_steps), effective_steps)

    def _lr_from_schedule(self) -> None:
        """Set each parameter group's rate from the run's schedule at its restored position.

        A resume loads the optimizer's state, rate included, then the schedule's position: a
        run resumed with more steps than its checkpoint planned would train its next update at
        the rate the shorter schedule left (0 at a cosine's end), then follow the longer one.
        Computed as the scheduler computes it (``base * lambda(position)``), so a resume at the
        same length sets the same float it restored."""
        sched = self._lr_scheduler
        lambdas = getattr(sched, "lr_lambdas", None)
        if sched is None or self.optimizer is None or lambdas is None:
            return
        for group, base, fn in zip(self.optimizer.param_groups, sched.base_lrs, lambdas):
            group["lr"] = base * fn(sched.last_epoch)

    def _adapter_optimizer_step(self) -> None:
        """Run one optimizer step with gradient clipping."""
        assert self.optimizer is not None
        if getattr(self, "_unreached_params", None):  # after the first update: fail closed
            reached = [n for n, p in self._unreached_params if p.grad is not None]
            self._unreached_params = []
            if reached:
                raise PorosConfigError(
                    "adapters on a submodule a text-only forward never runs received a "
                    f"gradient: {', '.join(reached[:5])}. Poros counts that submodule out of "
                    "the decoder blocks (ArchSpec.text_unreached), so the update would not be "
                    "one it can vouch for.",
                    remediation="Set lora_target_modules to the text layers' projections, and "
                                "report the model so its registry entry can be fixed.",
                )
        if self.config.max_grad_norm > 0 and self.trainable_parameters:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.trainable_parameters, self.config.max_grad_norm
            )
            self.history.grad_norms.append(float(grad_norm))
            self._stop_if_nonfinite(grad_norm=self.history.grad_norms[-1])
        else:  # no clip: the norm is still read and logged, on CUDA and a Mac alike
            grads = [p.grad for p in self.trainable_parameters if p.grad is not None]
            norm = float(torch.nn.utils.get_total_norm(grads)) if grads else 0.0
            self.history.grad_norms.append(norm)
            if grads:
                self._stop_if_nonfinite(grad_norm=norm)
        self.optimizer.step()
        self.compute.stepped()  # the expert cache sizes on a peak that holds this step
        if self._lr_scheduler is not None:
            self._lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

    def _stop_if_nonfinite(self, *, loss: Optional[float] = None,
                           grad_norm: Optional[float] = None) -> None:
        """Before an update is applied: a non-finite loss or gradient norm stops the run there.

        Nothing of the step is applied (the optimizer has not stepped, its gradients are
        dropped), so the adapter, the optimizer and every checkpoint written before hold finite
        values: the newest checkpoint is the run so far. An ``error`` event in events.jsonl says
        which value went non-finite at which step; ``PorosStepCheckError`` (E_STEP_CHECK) says
        what to change. A finite value does nothing (reads only floats already read). One rule
        on CUDA and a Mac: the Mac runtime's witness (``poros.metal.witness``) stops a remat that
        differs, and leaves a non-finite value to this."""
        bad = {k: v for k, v in (("loss", loss), ("grad_norm", grad_norm))
               if v is not None and not math.isfinite(v)}
        if not bad:
            return
        step = self.global_step + 1  # the update's number, counted from 1
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
        self._release_retained_forward_blocks()
        record = {"step": step, **{k: repr(v) for k, v in bad.items()},
                  "torch_dtype": self.config.torch_dtype}
        if self._event_writer is not None:
            self._event_writer.emit(PorosEvent(  # its index from 0, as a step event's
                event_type="error", step=self.global_step,
                data={"code": PorosStepCheckError.code, "nonfinite": record}))
        saved = self._own_checkpoints()
        kept = (f"{saved[-1][1]} holds the run through update {saved[-1][0] + 1}"
                if saved else "no checkpoint was written yet")
        what = " and ".join(f"the {'loss' if k == 'loss' else 'gradient norm'} is {v!r}"
                            for k, v in bad.items())
        lr = self.config.learning_rate
        if self.config.torch_dtype == "float16":
            fix = ("float16 overflows where bfloat16 does not: on a GPU with bf16 (sm_80 or "
                   "newer) leave torch_dtype unset; on an older one lower learning_rate "
                   f"(now {lr:g}) and keep max_grad_norm on.")
        else:
            fix = (f"Lower learning_rate (now {lr:g}), keep max_grad_norm on, and check the "
                   "data for empty or extreme rows.")
        resume = (" Then `poros train <config.yaml> --resume-from latest` goes on from that "
                  "checkpoint." if saved else " Then start the run again.")
        raise PorosStepCheckError(
            f"At update {step} {what}, so this run stopped before applying it: nothing of "
            f"update {step} was kept, and {kept}.",
            remediation=fix + resume,
            witness=record,
        )

    # ── Helpers ────────────────────────────────────────────────────────

    def _next_batch(
        self,
        data_iter: Iterator[Dict[str, Any]],
        dataloader: DataLoader,
    ) -> Tuple[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """Get the next batch, cycling the dataloader if exhausted."""
        try:
            batch = next(data_iter)
            return batch, data_iter
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
            return batch, data_iter

    def _move_batch_to_device(
        self, batch: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Move a batch of tensors to the compute device."""
        return {
            key: (
                value.to(self.device, non_blocking=self.device.type == "cuda")
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in batch.items()
        }

    def _count_batch_tokens(self, batch: Dict[str, torch.Tensor]) -> int:
        """Count the number of tokens in a batch."""
        attention_mask = batch.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor):
            return int(attention_mask.sum().item())
        for key in ("labels", "input_ids"):
            tensor = batch.get(key)
            if isinstance(tensor, torch.Tensor):
                return int(tensor.numel())
        return 0

    def _reset_runtime_counters(self) -> None:
        """Reset counters at the start of training."""
        self.history = RunHistory()
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            self.compute.measure_budget()  # what the expert cache is sized within

    # ── Evaluation ─────────────────────────────────────────────────────

    def _score_held_out(self, dataloader: DataLoader, step: int) -> None:
        """Score the held-out rows after update ``step`` (0 before the first): an eval event with
        the rows it scored, the run's history, and the score the next checkpoint of these
        weights keeps in eval.json. Timed apart from the step, so a step event's s/step stays
        the training's alone."""
        start = time.perf_counter()
        seen = hashlib.sha256()  # which rows it scored, as it read them
        said, rows = self._eval_pass(dataloader, caller="train", seen=seen)
        score = {**said, "rows": rows}
        self.history.evals.append({"step": step, **score,
                                   "eval_seconds": time.perf_counter() - start})
        # the weights it scored: none trained before the first step, else step + 1 updates
        self._held_score = (step + 1 if self.history.losses else 0, score)
        if not self.history.losses:  # the model it started from: the run's start, kept with it
            self._held_start = {**score, "held": held_rows(seen)}
        else:  # a resumed run's start stays only where it was of these rows
            settle_start(self, rows, held_rows(seen))
        if self._event_writer is not None:  # with the rows it scored, so a report compares a
            # start only with scores of the same rows (runs_report.held_out)
            self._event_writer.emit(PorosEvent(event_type="eval", step=step, data={
                **self.history.evals[-1], **_named(held_rows(seen))}))
        self._report("on_eval", step, dict(self.history.evals[-1]))

    def _evaluate_batches(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate loss across all batches (``_eval_pass``)."""
        return self._eval_pass(dataloader)[0]

    @torch.no_grad()
    def _eval_pass(self, dataloader: DataLoader, row_losses: Optional[List[float]] = None,
                   caller: str = "config", seen: Any = None) -> Tuple[Dict[str, float], int]:
        """(eval_loss, the mean of each batch's loss over every batch, once, in order, and its
        perplexity; the rows scored), with the random generators left as they were
        (``poros.train.evals.kept_rng``): a run that scores trains the same bytes after it as
        one that does not. ``row_losses`` gets each row's own mean loss."""
        assert self.model is not None
        losses: List[float] = []
        rows = 0
        was_training = self.model.training
        self.model.eval()
        try:
            with kept_rng(self.device):
                for batch in dataloader:
                    if seen is not None:  # each batch's tokens and labels, in order
                        for key in ("input_ids", "labels"):
                            if isinstance(batch.get(key), torch.Tensor):
                                seen.update(batch[key].detach().cpu().numpy().tobytes())
                    batch = self._move_batch_to_device(batch)
                    shape = batch["input_ids"].shape
                    with scoring_out_of_memory(self, int(shape[0]), int(shape[-1]), caller):
                        logits, each = None, None
                        if self.block_manager is not None:
                            hidden, labels = self._forward_eval_logits(batch, hidden_only=True)
                            if self.head.fused_dense_applies(hidden, labels):
                                # training's head holds a chunk of rows' logits, never all of
                                # them: so does its score, equal to the whole logits' to fp32
                                # rounding (a GEMM of fewer rows may round a logit otherwise)
                                loss, each = self._eval_loss_in_chunks(
                                    hidden, labels, rows=row_losses is not None)
                            else:
                                logits = self._logits_from_hidden(hidden)
                                loss = self._compute_loss(logits, labels)
                            del hidden
                        else:
                            outputs = self.model(**batch)
                            logits = (outputs.logits if hasattr(outputs, "logits")
                                      else outputs[0])
                            labels = batch.get("labels")
                            loss = getattr(outputs, "loss", None)
                            if loss is None:
                                loss = self._compute_loss(logits, labels)
                        if row_losses is not None:
                            row_losses.extend(each if each is not None
                                              else _row_losses(logits, labels))
                        del logits
                        losses.append(float(loss.item()))
                    rows += int(shape[0])
        finally:
            if was_training:
                self.model.train()

        eval_loss = sum(losses) / len(losses) if losses else float("nan")
        return {
            "eval_loss": eval_loss,
            "eval_perplexity": (
                math.exp(eval_loss) if math.isfinite(eval_loss) else float("inf")
            ),
        }, rows

    def _eval_loss_in_chunks(self, hidden: torch.Tensor, labels: torch.Tensor, *,
                             rows: bool = False) -> Tuple[torch.Tensor, List[float]]:
        """``_compute_loss(_logits_from_hidden(hidden), labels)``, holding the logits of a few
        rows of tokens at a time, never the whole ``[tokens, vocab]``: each position's
        log-softmax is its row's alone, and the mean is ``nll_loss`` over the same positions in
        the same order, as ``cross_entropy`` takes it. Equal to the whole logits' loss to fp32
        rounding, not bit for bit: the head's matrix product over fewer rows may round a logit's
        last bit otherwise (the CPU's does for some row counts; tests/test_eval_memory.py holds
        it to 1e-6 relative). Every score of a run on this path is taken this way, so they agree
        with each other exactly. A chunk is half the rows of
        a training step's head chunk (``poros.core.fused.head.CHUNK_ROWS``, or the batch's
        tokens if fewer): its logits and their log-softmax together hold what one chunk of
        logits does there. ``rows``: each row's own mean loss too (``poros eval``)."""
        from poros.core.fused.head import CHUNK_ROWS

        final_norm = self._get_final_norm()
        normed = final_norm(hidden) if final_norm is not None else hidden
        del hidden
        B, S = labels.shape
        target = labels[:, 1:].to(normed.device)
        picked = torch.zeros((B, S - 1), dtype=torch.float32, device=normed.device)
        step = _EVAL_ROWS or max(1, min(B * S, CHUNK_ROWS) // 2)
        for b in range(B):
            for s0 in range(0, S - 1, step):
                s1 = min(s0 + step, S - 1)
                logits = self._logits_from_hidden(normed[b, s0:s1], normed=True)
                logp = torch.log_softmax(logits, dim=-1)
                del logits
                t = target[b, s0:s1]
                picked[b, s0:s1] = logp.gather(1, t.clamp(min=0)[:, None]).squeeze(1).float()
                del logp
        ignore = torch.where(target == -100, target, torch.zeros_like(target))
        loss = torch.nn.functional.nll_loss(picked.reshape(-1, 1), ignore.reshape(-1))
        each = [float(torch.nn.functional.nll_loss(picked[b][:, None], ignore[b]))
                for b in range(B)] if rows else []
        return loss, each

    def _epoch_steps(self, batches: int) -> int:
        """Optimizer steps in ``num_epochs`` passes over ``batches`` micro-batches: each step
        takes ``gradient_accumulation_steps`` of them, so one pass is ceil(batches / G) steps,
        as a recipe's run counts it (``poros.recipes.sft``: epochs x its data's batches)."""
        config = self.config
        return max(1, -(-batches * config.num_epochs // config.gradient_accumulation_steps))

    def _prepare_dataloader(
        self,
        dataset: Union[str, Dataset, DataLoader],
        max_seq_length: int,
        for_eval: bool = False,
    ) -> DataLoader:
        """Convert a dataset spec into a DataLoader. ``for_eval``: held-out rows, scored every
        one (in order, the last batch short, never dropped: a set smaller than a batch scores).

        Accepts:
        - A string (HuggingFace dataset name) -> loads and tokenizes
        - A torch Dataset -> wraps in DataLoader
        - An existing DataLoader -> returns as-is
        """
        if isinstance(dataset, DataLoader):
            return dataset

        if isinstance(dataset, str):
            name = dataset.strip().lower()
            if name == "synthetic" or name.startswith("synthetic:"):
                return _build_synthetic_dataloader(
                    n_samples=_synthetic_sample_count(name),
                    max_seq_length=max_seq_length,
                    batch_size=self.config.per_device_batch_size,
                    seed=self.config.seed,
                    drop_last=not for_eval,
                )
            return _load_hf_dataset_as_dataloader(
                dataset_name=dataset,
                model_name_or_path=self.config.model_name_or_path,
                max_seq_length=max_seq_length,
                batch_size=self.config.per_device_batch_size,
                seed=self.config.seed,
                trust_remote_code=self.config.trust_remote_code,
                revision=self.config.model_revision,
                # The Mac keeps its rows padded to max_length, the bytes its receipts hold.
                pad_to_longest=not self._metal,
                # a script's own screen: a Hub dataset's download is said and shown there
                say=self._progress_print if self._progress_enabled() else None,
                for_eval=for_eval,
            )

        # Assume it's a torch Dataset. The shuffle generator is seeded from
        # the config, NOT the global RNG: a resumed run restores the global
        # RNG to mid-run state, so an unseeded sampler would draw a different
        # permutation than the original run and break bitwise resume.
        return DataLoader(
            dataset,
            batch_size=self.config.per_device_batch_size,
            shuffle=not for_eval,
            drop_last=not for_eval,
            generator=torch.Generator().manual_seed(self.config.seed),
        )

    # ── Metrics and logging ────────────────────────────────────────────


def _row_losses(logits: torch.Tensor, labels: Optional[torch.Tensor]) -> List[float]:
    """Each row's mean shifted cross-entropy over its scored tokens, as ``_compute_loss`` takes
    it over a batch (a row with none scored: nan)."""
    if labels is None:
        return [float("nan")] * logits.shape[0]
    out = []
    for i in range(logits.shape[0]):  # a row at a time: no float copy of the whole batch's
        target = labels[i, 1:].to(logits.device)
        n = int((target != -100).sum())
        out.append(float(torch.nn.functional.cross_entropy(
            logits[i, :-1].float(), target, ignore_index=-100)) if n else float("nan"))
    return out


def _named(held: Optional[str]) -> Dict[str, str]:
    """An eval event's ``held``: which held-out rows it scored, where known."""
    return {"held": held} if held else {}
