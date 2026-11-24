"""A recipe run of ``poros train``: its data read as the run reads it, the recipe (or the user's
loop) trained through ``poros.Run``, and the held-out scores it reports (``poros.cli.train_cmd``
calls these, and names them as its own)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from poros.cli import contract
from poros.preflight import command
from poros.train.evals import with_eval  # noqa: F401  (named here as before)


def recipe_data(cfg: Any, choice: Any, tok: Any) -> tuple:
    """A recipe run's data and its held-out file, tokenized as the run reads them; a row's fault
    (a bad row, every answer cut away at max_seq_length) is a PorosDataError."""
    from poros import data as pdata
    from poros.core.errors import PorosDataError

    rows = cfg.per_device_batch_size * cfg.gradient_accumulation_steps
    common = dict(length=cfg.max_seq_length, batch_size=rows, seed=cfg.seed)

    def build(source: str) -> Any:
        """The same builder for the data and its held-out file, so both read the same tokens."""
        if choice.shape == "pairs":
            return pdata.pairs(source, tok, **common)
        if choice.shape == "chats":
            return pdata.chats(source, tok, train_on=cfg.train_on, **common)
        return pdata.text(source, tok, **common)

    try:
        return build(cfg.dataset), (build(cfg.eval_dataset) if cfg.eval_dataset else None)
    except ValueError as exc:  # the file's fault, not a bug: a bad row, or no answer left
        cut = "no rows" in str(exc) and choice.shape in ("pairs", "chats")
        raise PorosDataError(
            f"every row's answer is cut away at {cfg.max_seq_length} tokens" if cut else str(exc),
            remediation="Train at a longer length, or shorten the rows." if cut else "",
        ) from exc


def train_recipe(trainer: Any, cfg: Any, choice: Any, resume_from: Optional[str],
                 should_stop: Optional[Callable[[], bool]], built: Optional[tuple] = None) -> dict:
    """Train ``cfg`` through ``poros.Run`` and a recipe (or the user's loop).

    One optimizer step is ``per_device_batch_size × gradient_accumulation_steps``
    rows, sent through the GPU ``per_device_batch_size`` at a time: the same
    batch and memory the classic loop and the fit check use.
    """
    import torch

    from poros import data as pdata
    from poros import recipes
    from poros.core.config import train_path
    from poros.core.events import PorosEvent
    from poros.ends import Ends, answer_ends, turn_end
    from poros.train.run import Run, _load_tokenizer

    if built is None:  # a Mac's run reads its data here, with its own tokenizer
        tok = getattr(trainer, "_tokenizer", None) or _load_tokenizer(cfg)
        data, held = recipe_data(cfg, choice, tok)
        if (clash := command.data_resume_refusal(resume_from, data, before_load=False)) is not None:
            contract.refused(clash)
    else:  # read before the model loaded (run): a row's fault was said in seconds
        tok, data, held = built
    from poros.preflight.resume import data_record

    trainer._data_record = data_record(data)  # beside each checkpoint: a resume reads the same
    if held is None and not (resume_from and whole_pairs(cfg, data)):
        # by seed: a resume holds out the same rows
        data, held = data.holdout(cfg.eval_split, cfg.eval_rows)
    data.group_by_length = bool(cfg.group_by_length)  # after the split: the rows that train
    noted = []
    if choice.shape == "chats" and (end_id := turn_end(tok)) is not None:
        from poros.preflight.data import degenerate_end_row

        try:
            head_weight = trainer._get_lm_head().weight
        except AttributeError:  # a trainer with no LM head to read (a stand-in)
            head_weight = None
        said = degenerate_end_row(
            head_weight, end_id,
            getattr(tok, "all_special_ids", []), frozen=bool(cfg.freeze_lm_head),
            train_on=cfg.train_on,
            end_name=repr(getattr(tok, "convert_ids_to_tokens", str)(end_id)))
        if said:
            logging.getLogger(__name__).warning(said)
            noted.append(said)
    ends = None
    if held is not None and choice.shape == "chats" and (end := turn_end(tok)) is not None:
        ends = Ends(end, answer_ends(held, tok, end, train_on=cfg.train_on,
                                     source=cfg.eval_dataset or cfg.dataset,
                                     length=cfg.max_seq_length))
    run = Run.from_trainer(trainer, tok)
    if resume_from:
        run.load(resume_from)
    elif cfg.init_adapter:  # --from: the earlier run's adapter, a fresh optimizer
        trainer.start_from_adapter()
    steps = cfg.max_steps if cfg.max_steps > 0 else None
    if trainer._event_writer is not None:
        trainer._event_writer.emit(PorosEvent(event_type="setup", data={
            "model": cfg.model_name_or_path, "quantization": cfg.quantization,
            "block_size": cfg.block_size, "steps": steps or cfg.num_epochs * len(data),
            "torch_num_threads": torch.get_num_threads(),
            "recipe": choice.name, "data": choice.shape, "rows": data.rows,
            "path": train_path(cfg),  # the loop: sequential, efficient or fast
            **({"path_reason": cfg.path_reason} if cfg.path_reason else {}),
            "offload_embeddings_and_head": bool(cfg.offload_embeddings_and_head),
            **({"offload_reason": cfg.offload_reason} if cfg.offload_reason else {}),
            **trainer.compute.record(),  # fused_compute, the MoE expert compute and why
            # pairs whose two sides train as the same tokens: they teach nothing (read, not run)
            **({"same_pairs": pdata.same_pairs(data)} if data.pairs else {}),
            **({"warnings": noted} if noted else {}),
        }))
    if choice.name in recipes.PAIRS:
        out = recipes.get(choice.name).main(run, data, steps=steps, epochs=cfg.num_epochs,
                                            beta=cfg.dpo_beta, eval_data=held, stop=should_stop)
        return with_eval(out, _evals(trainer), _start(trainer), _changed(trainer))
    if choice.name == "sft":
        out = recipes.get("sft").main(run, data, steps=steps, epochs=cfg.num_epochs,
                                      eval_data=held, ends=ends, stop=should_stop)
        return with_eval(out, _evals(trainer), _start(trainer), _changed(trainer))
    import inspect

    main = recipes.load_loop(choice.name)
    # a copied recipe takes these
    extra = {"epochs": cfg.num_epochs, "beta": cfg.dpo_beta, "eval_data": held, "ends": ends}
    params = inspect.signature(main).parameters
    takes_all = any(p.kind is p.VAR_KEYWORD for p in params.values())
    extra = {k: v for k, v in extra.items() if takes_all or k in params}
    out = main(run, data, steps=steps, stop=should_stop, **extra)
    return with_eval(out if isinstance(out, dict) else run.metrics,
                     _evals(trainer), _start(trainer), _changed(trainer))


def whole_pairs(cfg: Any, data: Any) -> bool:
    """A pairs run begun before DPO held pairs out trained on every pair, and its continue must
    too, or it is not the run it was: its first start counted them all."""
    if not data.pairs:
        return False
    from poros.cli.verbs import runs_report

    first = next((ev for ev in runs_report._events(Path(cfg.output_dir))
                  if ev.get("event_type") == "setup"), None)
    return first is not None and (first.get("data") or {}).get("rows") == data.rows


def _changed(trainer: Any) -> bool:
    """Whether the start a resumed run carried was dropped: its held-out rows changed."""
    return bool(getattr(trainer, "_held_changed", False))


def _start(trainer: Any) -> Any:
    """The run's first score as the engine carries it (a resume reads it from its checkpoint)."""
    return getattr(trainer, "_held_start", None)


def _evals(trainer: Any) -> list:
    """The evaluations a run has held so far (``PorosTrainer.history.evals``); none for a
    trainer that keeps no history."""
    return getattr(getattr(trainer, "history", None), "evals", [])
