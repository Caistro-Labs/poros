"""``poros.Run``: a training run you drive one step at a time.

Four verbs, named like Tinker's, over the streamed QLoRA engine (dense or
MoE, picked from the architecture registry as ``PorosTrainer`` does):

    run = poros.Run.create("Qwen/Qwen3-8B")
    data = poros.data.chats("chats.jsonl", run.tokenizer)
    for step in run.steps(600):
        run.forward_backward(data.batch(step))
        run.optim_step()
    run.save()

``forward_backward`` streams the base once and accumulates gradients;
``optim_step`` applies them. ``steps`` is the loop's spine: it starts from
the resume point, saves every ``save_steps``, honours a stop request, and
ends the run. A loop written this way is resumable, stoppable and drawn by
the CLI with no extra code.

The run is its folder (``run.dir``): events.jsonl, checkpoint-N/, the
adapter. Everything here goes through the engine's own forward, backward,
optimizer, checkpoint and event code, so the parity guarantees, resume and
the receipts are the engine's.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch

from poros.core import checkpoints
from poros.core.config import PorosConfig, config_fields
from poros.core.errors import PorosCancelledError, PorosConfigError, PorosOOMError
from poros.core.events import PorosEvent
from poros.train.batch import Batch, Output
from poros.train.evals import (
    held_rows,
    read_start,
    saves_at,
    scores_at,
    scoring_out_of_memory,
    settle_start,
)
from poros.train.evals import kept_rng as _kept_rng
from poros.train.trainer import PorosTrainer

LossArg = Union[str, Callable[..., Any]]


def token_logprobs(logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    """``[B, T]`` log p(target_token), computed in float32 one row at a time
    so the ``[B, T, vocab]`` float copy never exists at once. 0 on padding."""
    t = batch.tensors(logits.device)
    target, mask = t["target"], t["mask"]
    rows = [
        -torch.nn.functional.cross_entropy(
            logits[i].float(), target[i], reduction="none"
        )
        for i in range(logits.shape[0])
    ]
    return torch.stack(rows) * mask


class Run:
    """A run you drive: ``forward_backward``, ``optim_step``, ``save``, ``steps``."""

    def __init__(self, engine: PorosTrainer, tokenizer: Any = None) -> None:
        self._engine = engine
        self.tokenizer = tokenizer
        self._next = 0  # the next optimizer step's index
        self._first = 0  # the first step this loop trains: its progress line is always said
        self._micro = 0  # micro-batches run since the last optim_step (keyed MoE states)
        self._total: Optional[int] = None
        self._pending: List[Output] = []
        self._trained = 0.0  # positions the losses scored since the last optim_step
        self._step_started: Optional[float] = None
        self._counters_reset = False
        self._said_saved: Optional[str] = None  # the checkpoint whose place was last printed

    # ── Making one ─────────────────────────────────────────────────────

    @classmethod
    def create(
        cls,
        model: str,
        *,
        rank: int = 32,
        alpha: Optional[int] = None,
        length: int = 2048,
        lr: float = 2e-4,
        output_dir: Optional[str] = None,
        dataset: Optional[str] = None,
        config: Optional[PorosConfig] = None,
        tokenizer: Any = None,
        check_fit: bool = True,
        check_host_ram: bool = True,
        **settings: Any,
    ) -> "Run":
        """Load ``model`` as NF4, attach LoRA, set up streaming.

        ``dataset`` says what your loop trains on: the file (or folder, Hub id) its rows come
        from. The run reads no rows from it. Its checkpoints record it, a file by its contents,
        so ``open`` can check a resume reads the same data, wherever the file is by then. A
        loop that names none still trains, and its resume says it could not be checked.

        ``init_adapter=<checkpoint or run folder>`` starts this new run from an earlier
        run's adapter, with a fresh optimizer and schedule: more or other data after a run.
        It needs the same base, rank and target modules (refused before the download), and
        an ``output_dir`` of its own; ``alpha`` and the target modules, unset, are the adapter's.

        ``settings`` are any other ``PorosConfig`` fields. ``config`` replaces
        all of them (and ``rank``/``alpha``/``length``/``lr``). ``tokenizer``
        defaults to the model's own. ``check_fit`` / ``check_host_ram``: False
        skips that refusal before the download (``PorosTrainer.from_pretrained``'s).
        A new run: an ``output_dir`` another run left checkpoints in is refused
        before the download (``open`` continues that run), and so is one that cannot
        be created or written.
        """
        from poros.preflight.resume import SAME_SETTINGS, start_from

        if config is None:
            config = PorosConfig(
                model_name_or_path=model,
                lora_rank=rank,
                **({"lora_alpha": alpha} if alpha is not None else {}),
                max_seq_length=length,
                learning_rate=lr,
                **({"output_dir": output_dir} if output_dir else {}),
                **({"dataset": dataset} if dataset else {}),
                **settings,
            )
            # init_adapter: the alpha and the target modules it trained with, where none is named
            start = start_from(config)
            if "lora_alpha" not in config.model_fields_set:
                config.lora_alpha = 2 * rank
        else:
            start = start_from(config, adopt=())
        from poros.preflight.disk import require_output_dir

        require_output_dir(config.output_dir)  # a new run writes its folder: before the download
        _refuse_another_runs_folder(config.output_dir)
        if start.clash:  # init_adapter: its base and shape, before the download
            fix = start.clash.fix or SAME_SETTINGS
            raise PorosConfigError(f"Can't start from this adapter: {start.clash}.",
                                   remediation=f"{fix[0].upper()}{fix[1:]}.")
        if start.checkpoint:
            config.init_adapter = start.checkpoint  # a run's folder named its newest
        run = cls._loaded(model, config, tokenizer, check_fit, check_host_ram)
        if config.init_adapter:  # its weights; the optimizer and schedule are this run's, fresh
            run._engine.start_from_adapter()
        return run

    @classmethod
    def _loaded(cls, model: str, config: PorosConfig, tokenizer: Any, check_fit: bool = True,
                check_host_ram: bool = True) -> "Run":
        engine = PorosTrainer.from_pretrained(model, config=config, check_fit=check_fit,
                                              check_host_ram=check_host_ram)
        return cls(engine, tokenizer if tokenizer is not None else _load_tokenizer(config))

    @classmethod
    def open(cls, path: Union[str, Path], *, tokenizer: Any = None, **settings: Any) -> "Run":
        """Reopen a run folder (its newest checkpoint) or one checkpoint, with
        its adapter, optimizer, schedule, RNG and step. ``steps(n)`` with a
        larger ``n`` trains a finished run further.

        A run created with ``dataset=`` is checked against it here: the file's contents now
        against what the checkpoint recorded (another file is refused). ``dataset=`` here says
        where that file is now, if it moved.

        The folder says what the run trained with: its base, its adapter's shape, its data. It
        does not say what only you can: ``trust_remote_code=True`` here, and nothing in the
        folder, lets the base's own code run; a push goes where you say; and the run writes
        into the folder it was opened from, wherever that is now, unless ``output_dir=`` names
        another. Its ``store_path`` is read where a store of this model already is, and never
        written: with none there, ``store: ssd`` builds at the default place
        (``~/.poros/sss``) and says so, unless ``store_path=`` names another."""
        ckpt = _checkpoint_in(Path(path))
        saved = config_fields(json.loads((ckpt / "poros_config.json").read_text()))
        for name in _NOT_THE_FOLDERS:
            saved.pop(name, None)
        saved.update(settings)
        from poros.stores import recorded_store

        # its store_path is read where a store of this model is, and never written
        saved, build_path = recorded_store(saved, given=settings)
        config = PorosConfig(**saved)
        config._store_build_path = build_path
        if "output_dir" not in settings:
            config.output_dir = str(_run_folder(ckpt))  # keep writing into the run's folder
        run = cls._loaded(config.model_name_or_path, config, tokenizer)  # no new run: no clash
        run._declare_data()  # what the loop said it trains on, before the load compares it
        run.load(ckpt)
        return run

    def _declare_data(self) -> None:
        """What this run's loop trains on, where it said (``dataset=``; a recipe's config):
        taken as it is now, a file by its contents, for the checkpoints' record and a resume's
        check. Once: what was taken when the run read it stands."""
        engine = self._engine
        if engine._dataset_record is None and self.config.dataset is not None:
            from poros.preflight.resume import trained_on

            engine._dataset_record = trained_on(self.config.dataset)

    @classmethod
    def from_trainer(cls, trainer: PorosTrainer, tokenizer: Any = None) -> "Run":
        """Drive an engine that is already set up (tests, notebooks)."""
        return cls(trainer, tokenizer)

    # ── What it is ─────────────────────────────────────────────────────

    @property
    def config(self) -> PorosConfig:
        return self._engine.config

    @property
    def dir(self) -> Path:
        return Path(self._engine.config.output_dir)

    @property
    def device(self) -> torch.device:
        return self._engine.device

    @property
    def step(self) -> int:
        """Optimizer steps taken so far."""
        return self._next

    @property
    def metrics(self) -> Dict[str, Any]:
        """The run's summary so far (the receipt's numbers)."""
        return self._engine._compute_metrics()

    # ── The four verbs ─────────────────────────────────────────────────

    def forward_backward(
        self, batch: Batch, loss: LossArg = "cross_entropy", **loss_args: Any
    ) -> Output:
        """Stream the base, apply ``loss``, backpropagate. Gradients add up
        across calls until ``optim_step``.

        The batch goes through the GPU ``per_device_batch_size`` rows (or
        pairs) at a time, each slice's loss weighted by its share of the
        rows, so a large batch costs time, not memory: the same trade the
        fit check plans for. ``Output`` covers the whole batch.
        """
        from poros import losses

        engine = self._engine
        fn = losses.get(loss)
        self._ensure_counters()
        if self._step_started is None:
            self._step_started = time.perf_counter()
        parts = []
        engine.begin_passes(self._next, self._micro)
        try:
            slices = self._slices(batch)
            self._micro += len(slices)
            if self._block_major(slices, batch, fn, loss_args, parts):
                slices = []  # every microstep ran through the block-major driver
            for g, (sub, rows) in enumerate(slices):
                engine.before_pass(g)
                feed = sub.tensors(engine.device)
                engine.sss_pass_begins(feed)
                logits, _ = engine._forward_train_logits(
                    {"input_ids": feed["input_ids"], "attention_mask": feed["attention_mask"]}
                )
                lp = token_logprobs(logits, sub)
                del logits
                out = fn(lp, sub, **loss_args)
                share = len(rows) / len(batch)
                (out.value * share).backward()
                engine.sss_pass_ends()  # an SSD store's schedule, as the classic loop's pass
                parts.append((share, rows, float(out.value.detach()), out.metrics, lp.detach()))
            if slices:
                engine.end_passes()
        except torch.cuda.OutOfMemoryError as exc:
            engine._raise_if_mps_oom(exc)
            self._discard()
            raise PorosOOMError(
                block_size=engine.config.block_size, seq_len=batch.length
            ) from exc
        except RuntimeError as exc:  # mps: the allocator's OOM is a plain RuntimeError
            engine._raise_if_mps_oom(exc)
            raise
        engine.history.tokens += batch.tokens
        self._trained += batch.trained
        result = _join(parts, batch)
        self._pending.append(result)
        return result

    def forward(
        self, batch: Batch, loss: LossArg = "cross_entropy", **loss_args: Any
    ) -> Output:
        """The same pass without gradients, in eval mode: eval loss,
        reference logprobs, scoring."""
        from poros import losses

        return self._scored_pass(batch, losses.get(loss), loss_args)

    @torch.no_grad()
    def _scored_pass(self, batch: Batch, fn: Callable[..., Any], loss_args: Dict[str, Any],
                     look: Optional[Callable[..., None]] = None) -> Output:
        """``forward``'s pass. ``look(logits, logprobs, slice)`` reads each slice's logits before
        they go: a turn's end, from the pass that scores the loss."""
        engine = self._engine
        model = engine.model
        was_training = model.training
        model.eval()
        parts = []
        try:
            for sub, rows in self._slices(batch):
                feed = sub.tensors(engine.device)
                logits, _ = engine._forward_eval_logits(
                    {"input_ids": feed["input_ids"], "attention_mask": feed["attention_mask"]}
                )
                lp = token_logprobs(logits, sub)
                if look is not None:
                    look(logits, lp, sub)
                del logits
                out = fn(lp, sub, **loss_args)
                parts.append((len(rows) / len(batch), rows, float(out.value), out.metrics, lp))
        finally:
            if was_training:
                model.train()
        return _join(parts, batch)

    def _block_major(self, slices, batch, fn, loss_args, parts) -> bool:
        """All of a batch's slices, when the engine runs them block-major
        (``PorosTrainer.block_major_run_step``): each block streamed once for the forward of
        every microstep and once for their backward, instead of once per microstep. The head
        computes exactly this loop's per-microstep loss (``fn`` on ``token_logprobs``,
        weighted by the slice's share)."""
        engine = self._engine
        feeds, heads = [], []
        for sub, rows in slices:
            t = sub.tensors(engine.device)
            feeds.append({"input_ids": t["input_ids"], "attention_mask": t["attention_mask"],
                          "labels": t["input_ids"]})
            share = len(rows) / len(batch)

            def head(logits, sub=sub, rows=rows, share=share):
                lp = token_logprobs(logits, sub)
                out = fn(lp, sub, **loss_args)
                parts.append((share, rows, float(out.value.detach()), out.metrics, lp.detach()))
                return out.value * share

            heads.append(head)
        if engine.block_major_run_step(feeds, heads):
            return True
        parts.clear()  # the slices run sequentially (a stopped update's heads may have run)
        return False

    def _slices(self, batch: Batch) -> List[Any]:
        if self._engine.device.type == "mps":
            # Every new length costs a Mac memory for good, in MPS's graph cache (10 MB a step
            # at 0.5B), and only the table's lengths have been measured.
            batch.pad_to = self._engine.config.max_seq_length
        size = max(1, self._engine.config.per_device_batch_size)
        rows = len(batch) // 2 if batch.pairs else len(batch)
        if rows <= size:
            return [(batch, list(range(len(batch))))]
        return batch.split(size)

    def optim_step(self, lr: Optional[float] = None) -> Dict[str, float]:
        """Apply the accumulated gradients: clip, AdamW, schedule tick.

        ``lr`` sets this step's learning rate; by default the run's schedule
        does. Returns the step's numbers (also written as its step event).
        """
        engine = self._engine
        if not self._pending:
            raise RuntimeError("optim_step() with no forward_backward() since the last step")
        engine.global_step = self._next
        if lr is not None:
            for group in engine.optimizer.param_groups:
                group["lr"] = lr
        mean_loss = sum(o.loss for o in self._pending) / len(self._pending)
        if engine._witness is not None:
            engine._witness.check(engine.global_step, mean_loss)
        engine._stop_if_nonfinite(loss=mean_loss)
        engine._adapter_optimizer_step()
        engine._release_retained_forward_blocks()

        loss = sum(o.loss for o in self._pending) / len(self._pending)
        metrics = _mean_metrics(self._pending)
        # what this step trained on, for a loss that does not count it (dpo, ppo): the step
        # event's `tokens` is a step's, never the run's count so far
        metrics.setdefault("tokens", self._trained)
        self._pending, self._trained = [], 0.0
        step_time = time.perf_counter() - (self._step_started or time.perf_counter())
        self._step_started = None
        engine.history.step_times.append(step_time)
        if engine.device.type == "cuda" and torch.cuda.is_available():
            engine.history.vram_alloc_mb.append(
                torch.cuda.memory_allocated(engine.device) / 1e6
            )
        engine.history.losses.append(loss)
        engine._emit_step_event(loss, step_time, metrics)
        if engine.reporter is not None:
            report = engine._step_report(loss, step_time)
            report.extra = {k: v for k, v in metrics.items() if k != "tokens"}
            engine._report("on_step", report)
        if engine._logs_step(self._first):
            engine._log_step(loss, step_time)
        self._next += 1
        self._micro = 0
        return {"loss": loss, "grad_norm": engine.history.grad_norms[-1], **metrics}

    def save(self, name: Optional[str] = None) -> str:
        """Save the full state: adapter, optimizer, schedule, RNG, step.

        No name: ``checkpoint-<step>`` in the run folder, the form ``open``
        and ``runs continue`` resume from. A name: ``<run dir>/<name>``.
        Scored since its last step, it holds that score too, numbers only, in
        ``eval.json``.

        Before the first ``optim_step`` nothing is trained, so there is no
        ``checkpoint-<step>``: name it (``run.save("start")``) to keep the
        starting state, which ``load``/``open`` resume at step 0.
        """
        engine = self._engine
        if name is None and self._next == 0:
            raise ValueError(
                "run.save() before the first optim_step(): nothing is trained yet, so there is "
                "no checkpoint-<step> to write. Name it (run.save(\"start\")) to keep the "
                "starting state; loading it resumes at step 0.")
        # the last step taken: -1 before any, so loading the save resumes at step 0 (not 1)
        engine.global_step = self._next - 1
        try:
            if name is None:
                path = engine._save_checkpoint()
                # where it is, as a named save says it too; once for a step saved twice (the
                # loop's own save_steps save, then the script's run.save())
                if engine._progress_enabled() and path != self._said_saved:
                    engine._progress_print(f"Saved to:         {path}")
                    self._said_saved = path
            else:
                path = str(self.dir / name)
                engine.save(path, scores=True)  # as a checkpoint-<step> keeps its score
        finally:
            engine.global_step = max(self._next - 1, 0)
        return path

    def load(self, checkpoint: Union[str, Path]) -> None:
        """Restore a checkpoint into this run; the next step follows it."""
        engine = self._engine
        engine.load_checkpoint(str(checkpoint))
        engine._held_start = read_start(checkpoint)  # the start it had, not a score on the way
        engine._held_changed = False
        self._next = engine.global_step + 1
        self._micro = 0

    # ── The loop's spine ───────────────────────────────────────────────

    def steps(
        self, n: int, *, stop: Optional[Callable[[], bool]] = None
    ) -> Iterator[int]:
        """Yield step numbers from the resume point up to ``n``.

        Each step must end with ``optim_step()``. After it: a checkpoint
        every ``save_steps``, and ``stop()`` is checked; True writes a
        checkpoint and raises ``PorosCancelledError`` (resume is bitwise).
        """
        engine = self._engine
        if n <= 0:
            raise ValueError("steps(n) needs n >= 1")
        if self._next == 0:  # create checked before the load; an engine from_trainer had not
            _refuse_another_runs_folder(self.dir)
        if self._total != n:
            self._schedule(n)
        engine._total_optimizer_steps = n
        checkpoints.recover_checkpoint_dirs(self.dir)
        # a recipe's rows were read from what the config names, and a loop of the caller's own
        # may say what its rows come from (create's dataset=): its contents as the run starts,
        # for its checkpoints' record (a recipe's rows themselves are in DATA_RECORD)
        self._declare_data()
        self._ensure_counters()
        engine._report("on_start", engine, n, self._next)
        engine._start_watchdog()
        self._first = self._next
        try:
            for step in range(self._next, n):
                yield step
                if self._next != step + 1:
                    raise RuntimeError(f"step {step} ended without run.optim_step()")
                if saves_at(engine.config, step):
                    engine._report("on_checkpoint", step, self.save())
                if stop is not None and step + 1 < n and stop():
                    raise PorosCancelledError(step, self.save())
                if step + 1 < n:
                    engine._stop_if_out_of_memory()
        finally:
            engine._stop_watchdog()
        metrics = self.metrics
        engine._report("on_done", metrics)
        if engine._event_writer is not None:
            engine._event_writer.emit(PorosEvent(event_type="teardown", data=metrics))

    def evals_at(self, step: int) -> bool:
        """Whether a loop scores its held-out rows after ``step``: every
        ``eval_steps``, and where ``steps`` saves a checkpoint
        (``eval_checkpoints``), so the checkpoint keeps its score. Once a
        step, and not after the last: the loop scores the finished run."""
        return scores_at(self._engine.config, step, self._total)

    # ── Built on the verbs ─────────────────────────────────────────────

    def evaluate(
        self, data: Any, loss: LossArg = "cross_entropy", *, ends: Any = None, **loss_args: Any
    ) -> Dict[str, float]:
        """Mean loss over every batch of ``data``, once, in order; written as
        an eval event, and kept in the checkpoint the next ``save`` writes.

        On pairs it counts ``prefers``: the pairs whose chosen answer scores
        over the rejected one. ``ends`` (a ``poros.ends.Ends``) reads in the
        same pass how many rows end every answer on the end-of-turn token
        (``ends`` of ``answer_rows``). The RNG is left as it was, so a run
        that scores trains the same bytes as one that does not.
        """
        from poros import losses
        from poros.ends import ended, read
        from poros.losses import sequence_logprobs

        fn = losses.get(loss)
        values, prefers, tops = [], 0, []

        def look(logits: Any, lp: Any, sub: Batch) -> None:
            tops.extend(read(logits, lp, sub, ends.at, ends.token))

        with _kept_rng(self.device):
            for b in data.ordered():
                # as large as one slice of a training step: out of memory is said as training's
                size = max(1, self._engine.config.per_device_batch_size)
                with scoring_out_of_memory(self._engine, min(len(b), size), b.length):
                    out = self._scored_pass(b, fn, loss_args,
                                            look if ends is not None else None)
                values.append(out.loss)
                if data.pairs:
                    seq = sequence_logprobs(out.logprobs, b)
                    half = len(b) // 2
                    prefers += int((seq[:half] > seq[half:]).sum())
        mean = sum(values) / len(values) if values else float("nan")
        result: Dict[str, Any] = {"eval_loss": mean}
        if data.pairs:  # a DPO loss has no perplexity
            result["prefers"] = prefers
        else:
            result["eval_perplexity"] = (
                math.exp(mean) if math.isfinite(mean) and mean < 700 else float("inf"))
        if ends is not None:
            result["ends"], result["answer_rows"] = ended(tops)
        engine = self._engine
        step = max(self._next - 1, 0)
        rows = getattr(data, "rows", None)
        engine.history.evals.append({"step": step, **result})
        held = held_rows(data)  # which rows: a report compares a start only with these
        if engine._event_writer is not None:
            engine._event_writer.emit(PorosEvent(event_type="eval", step=step, data={
                **result, "rows": rows, **({"held": held} if held else {})}))
        engine._report("on_eval", step, {**result, "rows": rows})
        engine._held_score = (self._next, {**result, "rows": rows})
        if self._next == 0:  # the model it starts from: the run's start, kept with it
            engine._held_start = {**result, "rows": rows, "held": held}
        else:  # a resumed run's start stays only where it was of these rows
            settle_start(engine, rows, held)
        return result

    def reference(self, data: Any) -> torch.Tensor:
        """``[N, 2]`` sequence logprobs of each pair's (chosen, rejected)
        under the adapter as it is now: the reference DPO compares against.

        At a run's start the LoRA update is zero, so this is the base model;
        after SFT it is the SFT policy. One forward pass over the data,
        cached in the run folder so a resumed run keeps the same reference.
        """
        from poros.losses import sequence_logprobs

        path = self.dir / "reference" / f"{data.fingerprint}.pt"
        if path.exists():
            return torch.load(path, weights_only=True)
        ref = torch.zeros(data.rows, 2)
        for b in data.ordered():
            seq = sequence_logprobs(self.forward(b).logprobs, b).cpu()
            half = len(b) // 2
            ids = torch.tensor(b.ids[:half])
            ref[ids, 0], ref[ids, 1] = seq[:half], seq[half:]
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(ref, path)
        return ref

    # ── Internals ──────────────────────────────────────────────────────

    def _schedule(self, n: int) -> None:
        """Fit the LR schedule to ``n`` steps, keeping a resumed position."""
        engine = self._engine
        old = engine._lr_scheduler
        state = old.state_dict() if old is not None and self._next > 0 else None
        engine._rebuild_lr_scheduler_for_run(n)
        sched = engine._lr_scheduler
        if state is not None and sched is not None:
            sched.load_state_dict(state)
            engine._lr_from_schedule()
        self._total = n

    def _ensure_counters(self) -> None:
        if not self._counters_reset:
            engine = self._engine
            before = engine.history.evals  # the model the run starts from
            engine._reset_runtime_counters()
            engine.history.evals = before
            self._counters_reset = True

    def _discard(self) -> None:
        engine = self._engine
        engine.optimizer.zero_grad(set_to_none=True)
        engine._release_retained_forward_blocks()
        self._pending, self._trained, self._micro = [], 0.0, 0
        if engine.keyed_rng is not None:
            engine.keyed_rng.end(check=False)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _join(parts: List[Any], batch: Batch) -> Output:
    """One ``Output`` from a batch's slices: share-weighted loss and metrics,
    logprobs back in the batch's row order."""
    if len(parts) == 1:
        _, _, value, metrics, lp = parts[0]
        return Output(value, dict(metrics), lp)
    lp = parts[0][4].new_zeros((len(batch), batch.length))
    for _, rows, _, _, part in parts:
        lp[rows, : part.shape[1]] = part
    value = sum(share * v for share, _, v, _, _ in parts)
    keys = {k for p in parts for k in p[3]}
    metrics = {}
    for k in sorted(keys):
        have = [(share, m[k]) for share, _, _, m, _ in parts if k in m]
        total = sum(share for share, _ in have)
        metrics[k] = sum(share * v for share, v in have) / total
    if "tokens" in metrics:  # a count, not a rate: the slices add up
        metrics["tokens"] = sum(p[3].get("tokens", 0.0) for p in parts)
    return Output(value, metrics, lp)


def _mean_metrics(outputs: List[Output]) -> Dict[str, float]:
    keys = {k for o in outputs for k in o.metrics}
    out = {
        k: sum(o.metrics[k] for o in outputs if k in o.metrics)
        / sum(1 for o in outputs if k in o.metrics)
        for k in sorted(keys)
    }
    if "tokens" in out:  # a count, not a rate: a step's calls add up, as its slices do
        out["tokens"] = sum(o.metrics.get("tokens", 0.0) for o in outputs)
    return out




# What a run's folder does not say for whoever opens it (``Run.open``). A folder can come from
# anyone: the base's own code runs, and a push leaves, on the caller's word alone; an opened run
# writes where it was opened from; and it starts from its own checkpoint, not from another
# adapter a path in it names.
_NOT_THE_FOLDERS = ("trust_remote_code", "push_to_hub", "hub_model_id", "output_dir",
                    "init_adapter")


def _run_folder(ckpt: Path) -> Path:
    """The run folder of a checkpoint that was opened: the folder a ``checkpoint-<step>`` or a
    named save (``run.save("start")``) is in, which holds the run's events.jsonl; itself for a
    run's own folder (a finished run's adapter is saved there)."""
    inside = ckpt.name.startswith("checkpoint-") or (
        not (ckpt / "events.jsonl").exists() and (ckpt.parent / "events.jsonl").exists())
    return ckpt.parent if inside else ckpt


def _refuse_another_runs_folder(folder: "str | Path") -> None:
    """A new run never starts among another run's checkpoints (pruning and resume would mix
    them with its own)."""
    if clash := checkpoints.fresh_run_clash(folder):
        raise PorosConfigError(
            f"Can't start a new run: {clash}.",
            remediation="Continue that run with Run.open(<its checkpoint>), or give "
            "this one another output_dir.")


def _checkpoint_in(path: Path) -> Path:
    """``path`` if it is a checkpoint, else the newest checkpoint of the run whose folder it is;
    a folder holding more than one run's has none (as ``--resume-from latest``). First, as
    there, a save a kill left staged (``checkpoint-N.tmp``, complete but not yet renamed) is put
    in its place, so the newest is the newest and a checkpoint named is found."""
    checkpoints.recover_checkpoint_dirs(
        path.absolute().parent if path.name.startswith("checkpoint-") else path)
    if (path / "poros_config.json").exists():
        return path
    found = checkpoints.list_checkpoints(path)
    if not found:
        raise FileNotFoundError(f"{path} holds no checkpoint to open")
    if clash := checkpoints.mixed_runs_clash(path):
        raise PorosConfigError(
            f"Can't open a run here: {clash}.",
            remediation=f"Open the checkpoint to continue: Run.open('{path}/checkpoint-<step>').")
    return found[-1][1]


def _load_tokenizer(config: PorosConfig) -> Any:
    from poros.core.tokenizer import load_tokenizer

    tok = load_tokenizer(
        config.model_name_or_path,
        revision=config.model_revision,
        trust_remote_code=config.trust_remote_code,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
