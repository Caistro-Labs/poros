"""What `eval` scores, and how: a run's own rows through its own streamed forward, the adapter
off, then on.

The rows are the ones the run held out, rebuilt with the builder, tokenizer, length and seed it
trained with, so the split is the run's own. A file whose rebuilt split is not what the run
counted, or that changed after the run began, is refused. The engine is the run's too:
``PorosTrainer`` from its resolved settings, dense or MoE, on CUDA or a Mac, the base streamed
block by block as in training. A fresh adapter is zero, so the first pass is the base's exactly,
the pass the run scored before its first step; then the run's adapter loads and the same rows go
through again. Nothing is written to the run's folder, nothing is fetched, and only numbers leave.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple

from poros import data as pdata
from poros import runs
from poros.cli.verbs import runs_report
from poros.core.refusal import Refusal
from poros.ends import answer_ends, ended, read

logger = logging.getLogger(__name__)

# What a run's settings give the engine: what decides its forward and its rows. Anything else, a
# loop file's code, a push to the Hub, remote code, the folder it writes to, takes its default.
KEPT = ("model_name_or_path", "model_revision", "quantization", "torch_dtype", "block_size",
        "offload_embeddings_and_head",  # where its tables were: the run's, never decided again
        "freeze_embeddings", "freeze_lm_head", "lora_rank", "lora_alpha", "lora_dropout",
        "lora_target_modules", "lora_bias", "adapter_backend",
        "attn_implementation", "deterministic", "enable_prefetch",
        "store", "store_path", "store_options", "sss_ram", "max_seq_length",
        "per_device_batch_size", "gradient_accumulation_steps", "seed", "dataset", "eval_dataset",
        "eval_split", "eval_rows", "train_on", "dpo_beta", "group_by_length")
SHAPES = ("chats", "text", "pairs")


@dataclass
class Rows:
    data: Any  # poros.data.Data: what is scored, in its order
    shape: str  # chats, text or pairs
    source: str  # held_out, eval_file, trained (a pairs run's own pairs), file
    total: int  # the rows there were, before --rows
    path: str
    ends: List[List[int]] = field(default_factory=list)  # per row: where each answer ends


@dataclass
class Side:
    """One pass over the rows: the base's or the run's."""

    loss: float = math.nan
    rows: List[float] = field(default_factory=list)  # each row's loss (chats, text)
    seq: List[Tuple[float, float]] = field(default_factory=list)  # pairs: (chosen, rejected)
    ends: List[List[Tuple[bool, float]]] = field(default_factory=list)  # per row, per answer end


def settings(r: runs.Run, adapter: Path) -> Tuple[Any, Optional[str]]:
    """The run's own settings, the ones its forward and rows depend on, and its data's shape. The
    revision is the one its adapter trained on."""
    from poros.core.config import PorosConfig

    said = runs_report._read(Path(r.output_dir) / "config.resolved.json")
    if not isinstance(said, dict):
        raise Refusal("E_NOT_FOUND", "No settings", f"run {r.id} recorded none to rebuild from",
                      context=["a run writes config.resolved.json as it starts"])
    kept = {k: said[k] for k in KEPT if k in said and k in PorosConfig.model_fields}
    kept["model_revision"] = kept.get("model_revision") or _revision(adapter)
    # A MoE run's expert grouping, as it trained (it moves bits); scoring trains nothing, so the
    # engine takes the sequential loop and sets up nothing of block-major's.
    kept["moe_expert_grouping"] = said.get("moe_expert_grouping")
    # How the run held Inkling 2 Mini's experts (released or nf4): the base it scored and
    # trained against. The record decides, never POROS_INKLING_EXPERTS at eval time.
    kept["moe_expert_format"] = said.get("moe_expert_format")
    kept["path"] = "sequential"
    from poros.stores import recorded_store

    kept, build_path = recorded_store(kept)  # a store_path is read, never written
    try:
        cfg = PorosConfig(**kept)
        cfg._store_build_path = build_path
        cfg._expert_format_recorded = kept["moe_expert_format"] is not None
    except ValueError as exc:
        raise Refusal("E_CONFIG", "Bad settings", f"run {r.id}'s recorded settings do not read",
                      context=[type(exc).__name__]) from None
    shape = said.get("data_shape")
    return cfg, shape if shape in SHAPES else None


def _revision(adapter: Path) -> Optional[str]:
    from poros.cli.verbs.export import _REVISION

    said = (runs_report._read(adapter / "poros_config.json") or {}).get("model_revision")
    return said if isinstance(said, str) and _REVISION.fullmatch(said) else None


@contextmanager
def offline() -> Iterator[None]:
    """No network for what this loads: the base and its tokenizer are where the run left them."""
    from huggingface_hub import constants

    was = constants.HF_HUB_OFFLINE
    constants.HF_HUB_OFFLINE = True
    try:
        yield
    finally:
        constants.HF_HUB_OFFLINE = was


def tokenizer(cfg: Any) -> Any:
    """The tokenizer the run built its rows with."""
    from poros.train.run import _load_tokenizer

    with offline():
        return _load_tokenizer(cfg)


def rows(cfg: Any, shape: Optional[str], r: runs.Run, data: Optional[str], cap: int,
         tok: Any, end: Optional[int]) -> Rows:
    """The rows to score: ``data``'s, else the run's held-out rows or pairs, its --eval file, or
    the pairs of a pairs run that held none out, rebuilt as it built them and checked against what
    it counted."""
    if data:
        path, source = data, "file"
        shape = shape or _shape(data)  # a classic run's: whatever the file holds
    elif cfg.eval_dataset:
        path, source = cfg.eval_dataset, "eval_file"
    elif shape in SHAPES:
        path, source = cfg.dataset, "held_out"
    else:
        raise nothing_held(r)
    if not Path(str(path)).is_file():
        raise Refusal("E_NOT_FOUND", "Data gone", f"{path} is not there to rebuild rows from",
                      fix=f"poros eval {r.id} --data <file>", note="scores a file you name")
    built = _build(shape, str(path), tok, cfg)
    setup, scored = _counted(Path(r.output_dir))
    if source == "held_out":
        trained, held = built.holdout(cfg.eval_split, cfg.eval_rows)
        if built.pairs and (held is None or setup == built.rows):
            source = "trained"  # every pair trained: under 40, or begun before DPO held pairs out
        elif held is None:
            raise Refusal("E_USAGE", "Nothing held out",
                          f"run {r.id} held no rows out: under 40 rows, or eval_split=0",
                          context=["--data <file> scores a file of rows"],
                          fix=f"poros eval {r.id} --data <file>")
        else:
            _same(r, path, (trained.rows, held.rows), (setup, scored))
            built = held
    if source == "eval_file":
        _same(r, path, (built.rows,), (scored,))
    elif source == "trained":
        _same(r, path, (built.rows,), (setup,))
    if source != "file":
        _unchanged(r, Path(str(path)))
    kept = pdata.Data(built.items[:cap], batch_size=built.batch_size, seed=built.seed,
                      pad_id=built.pad_id, pairs=built.pairs, kind=built.kind)
    kept.numbers = built.numbers[:cap] if built.numbers is not None else None
    out = Rows(kept, shape, source, built.rows, str(path))
    if end is not None and shape in ("chats", "pairs"):
        out.ends = answer_ends(out.data, tok, end, train_on=cfg.train_on, source=out.path,
                               length=cfg.max_seq_length)
    return out


def nothing_held(r: runs.Run) -> Refusal:
    """A run on the engine's own loop with no file of held-out rows: it trained every row."""
    return Refusal("E_USAGE", "Nothing held out",
                   f"run {r.id} trained on the classic loop, the engine's own, which trains "
                   "every row and holds none out",
                   context=["--data <file> scores a file of rows",
                            "a run given a file of held-out rows (--eval FILE, eval_dataset) "
                            "scores it as it trains"],
                   fix=f"poros eval {r.id} --data <file>")


def on_engine_loop(r: runs.Run) -> bool:
    """Whether the run trained on the engine's own loop (``trained_with: classic``)."""
    said = runs_report._read(Path(r.output_dir) / "config.resolved.json")
    return isinstance(said, dict) and said.get("trained_with") == "classic"


@dataclass
class LoopRows:
    """What `eval` scores of a run on the engine's own loop: a file, read as that loop reads it
    (its tokenizer and length, every token scored, no end token added), batched as it scored its
    held-out rows. The run's own held-out loss is the mean ``score_loop`` takes of these."""

    path: str
    source: str  # eval_file (its held-out file) or file (--data)
    loader: Any = None
    rows: int = 0  # scored
    total: int = 0  # in the file, as read, before --rows


def loop_rows(cfg: Any, r: runs.Run, data: Optional[str]) -> LoopRows:
    """The file to score for a run on the engine's own loop: ``data``, else its held-out file;
    with neither, refused in words (it held no rows out)."""
    if data:
        out = LoopRows(str(data), "file")
    elif cfg.eval_dataset:
        out = LoopRows(str(cfg.eval_dataset), "eval_file")
    else:
        raise nothing_held(r)
    if not Path(out.path).is_file():
        raise Refusal("E_NOT_FOUND", "Data gone", f"{out.path} is not there to read rows from",
                      fix=f"poros eval {r.id} --data <file>", note="scores a file you name")
    if out.source == "eval_file":
        _unchanged(r, Path(out.path))
    return out


def read_loop_rows(trainer: Any, rows: LoopRows, r: runs.Run, cap: int) -> None:
    """``rows``'s loader, built by the run's own engine as its loop built its held-out rows, at
    most ``cap`` of them; a held-out file that now reads to other rows than the run scored is
    refused."""
    import torch

    with offline():
        loader = trainer._prepare_dataloader(rows.path, trainer.config.max_seq_length,
                                             for_eval=True)
    rows.total = len(loader.dataset)
    if rows.source == "eval_file":
        _same(r, rows.path, (rows.total,), (_counted(Path(r.output_dir))[1],))
    if rows.total > cap:
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(loader.dataset, range(cap)), batch_size=loader.batch_size,
            shuffle=False, drop_last=False, collate_fn=loader.collate_fn)
    rows.loader, rows.rows = loader, min(rows.total, cap)


def score_loop(trainer: Any, rows: LoopRows) -> Side:
    """One pass over the rows through the engine's own eval pass (``_eval_pass``): the loss the
    run's loop reports for them, and each row's own."""
    side = Side()
    trainer._start_watchdog()
    try:
        said, _ = trainer._eval_pass(rows.loader, row_losses=side.rows, caller="run")
    except RuntimeError as exc:  # mps: the allocator's out of memory is a plain RuntimeError
        trainer._raise_if_mps_oom(exc)
        raise
    finally:
        trainer._stop_watchdog()
    side.loss = said["eval_loss"]
    return side


def _shape(path: str) -> str:
    try:
        shape = pdata.detect(Path(path))
    except ValueError as exc:
        raise Refusal("E_DATASET", "Can't read it", str(exc)) from None
    return "chats" if shape == "convertible" else shape


def _common(cfg: Any) -> dict:
    """As the run batched its rows: a step's rows, its length, its seed."""
    return dict(length=cfg.max_seq_length, seed=cfg.seed,
                batch_size=cfg.per_device_batch_size * cfg.gradient_accumulation_steps)


def _build(shape: str, path: str, tok: Any, cfg: Any, train_on: Optional[str] = None) -> Any:
    try:
        if shape == "chats":
            return pdata.chats(path, tok, train_on=train_on or cfg.train_on, **_common(cfg))
        if shape == "pairs":
            return pdata.pairs(path, tok, **_common(cfg))
        return pdata.text(path, tok, **_common(cfg))
    except ValueError as exc:  # the file's fault, in its row numbers and no row's words
        raise Refusal("E_DATASET", "Can't read it", str(exc)) from None


def _counted(out: Path) -> Tuple[Optional[int], Optional[int]]:
    """What the run counted: the rows it trained on (its setup event), and the rows it scored."""
    setup = None
    for ev in runs_report._events(out):
        rows = (ev.get("data") or {}).get("rows")
        if ev.get("event_type") == "setup" and type(rows) is int:
            setup = rows
    return setup, runs_report.held_out(str(out))["eval_rows"]


def _same(r: runs.Run, path: Any, rebuilt: tuple, recorded: tuple) -> None:
    if any(said is not None and said != now for now, said in zip(rebuilt, recorded)):
        raise Refusal("E_DATASET", "Data changed",
                      f"{Path(str(path)).name} now splits as {_rows_said(rebuilt)}, and run "
                      f"{r.id} counted {_rows_said(recorded)}",
                      context=["its rows can't be rebuilt as it scored them"],
                      fix=f"poros eval {r.id} --data {path}", note="scores the file as it is now")


def _rows_said(counts: tuple) -> str:
    if len(counts) == 1:
        return f"{counts[0]} rows"
    return f"{counts[0]} rows trained and {counts[1]} held out"


def _unchanged(r: runs.Run, path: Path) -> None:
    """The file is the one the run read: none written to after it began."""
    try:
        began = dt.datetime.fromisoformat(str(r.started_at)).timestamp()
    except ValueError:
        return
    if path.stat().st_mtime > began + 1:  # its start is said to the second
        raise Refusal("E_DATASET", "Data changed",
                      f"{path.name} was written to after run {r.id} began",
                      context=["its held-out rows may not be the ones it held out"],
                      fix=f"poros eval {r.id} --data {path}", note="scores the file as it is now")


# ── The engine ────────────────────────────────────────────────────────


# Room eval keeps above what the run measured when it trained, before that measurement stands
# for the estimate: eval takes the sequential path, which holds less host RAM than the fast or
# efficient path a run trains on, and this covers what scoring adds (its rows, a second load).
HOST_MARGIN_GIB = 1.0


def measured_host(run_dir: Optional[str], cfg: Any) -> Optional[float]:
    """The host RAM, in GiB, the run in ``run_dir`` measured when it trained: its receipt's
    ``peak_host_ram_gb`` (decimal GB, the process's peak resident set), where the run trained
    this model, at this quantization, on this store, and was not a preview. None otherwise."""
    if not run_dir:
        return None
    receipt = runs_report._read(Path(run_dir) / "receipt.json")
    resolved = runs_report._read(Path(run_dir) / "config.resolved.json")
    if not isinstance(receipt, dict) or not isinstance(resolved, dict) or receipt.get("simulated"):
        return None
    same = (resolved.get("model_name_or_path") == cfg.model_name_or_path
            and resolved.get("quantization", "nf4") == cfg.quantization
            and resolved.get("store", "ram") == cfg.store)
    peak = receipt.get("peak_host_ram_gb")  # the receipt's one host figure
    ok = isinstance(peak, (int, float)) and not isinstance(peak, bool) and peak > 0
    return round(peak * 1e9 / 2**30, 2) if same and ok else None


def frozen_floor(cfg: Any) -> Optional[float]:
    """GiB of host RAM a dense NF4 run's frozen layers take alone (the preflight's hard floor,
    poros.plan.estimate._nf4_host_floor), from its config.json; None where it does not apply.
    No measurement of another run stands below it."""
    from poros.cli.words.models import by_id
    from poros.plan.configs import read_model_config, shipped_config
    from poros.plan.estimate import _nf4_host_floor, _sizes

    e = by_id(cfg.model_name_or_path)
    with offline():  # an eval fetches nothing: the file the run's base left here
        model_config = read_model_config(cfg.model_name_or_path) or shipped_config(
            cfg.model_name_or_path)
    return _nf4_host_floor(cfg, e.entry if e else None, _sizes(model_config, cfg.moe_expert_format))


def engine(cfg: Any, run_dir: Optional[str] = None,
           say: Optional[Callable[[str], None]] = None) -> Any:
    """The run's engine on the run's base, its adapter at zero. Where the host-RAM estimate
    refuses it, but the run in ``run_dir`` measured less when it trained than is available now
    with HOST_MARGIN_GIB to spare, the measurement stands: ``say`` gets one line, and it loads."""
    from poros.core.errors import PorosHostRamError
    from poros.train.trainer import PorosTrainer

    try:
        with offline():
            trainer = PorosTrainer.from_pretrained(cfg.model_name_or_path, config=cfg,
                                                   check_fit=False)  # an eval: no fit check
    except PorosHostRamError as exc:
        measured = measured_host(run_dir, cfg)
        floor = frozen_floor(cfg)  # what the base must hold there, whatever a run measured
        if (measured is None or measured + HOST_MARGIN_GIB > exc.free_gib
                or (floor is not None and floor > exc.free_gib)):
            # the same refusal, its remedy in words eval can act on
            exc.info.remediation = (f"Free host RAM ({exc.free_gib} GiB is available now): close "
                                    "other programs, then run poros eval again.")
            raise
        (say or logger.warning)(
            f"{exc.free_gib:.1f} GiB of host RAM is available, and this run used {measured:.1f} "
            "GiB when it trained, so eval goes ahead.")
        with offline():
            trainer = PorosTrainer.from_pretrained(cfg.model_name_or_path, config=cfg,
                                                   check_fit=False, check_host_ram=False)
    trainer._event_writer = None  # an eval writes no events
    return trainer


def scratch() -> Any:
    """Where the engine may write, for as long as the eval runs: never the run's folder."""
    return tempfile.TemporaryDirectory(prefix="poros-eval-")


def score(trainer: Any, rows: Rows, end: Optional[int]) -> Side:
    """One pass over the rows through the run's own forward, in eval mode and without gradients:
    each slice as ``Run.forward`` sends it, its loss as ``Run.evaluate`` averages it, and at each
    answer's end whether the end-of-turn token is the top choice, and its probability."""
    import torch

    from poros.losses import cross_entropy, sequence_logprobs
    from poros.train.evals import scoring_out_of_memory
    from poros.train.run import Run, token_logprobs

    run, model, side = Run.from_trainer(trainer), trainer.model, Side()
    pairs = rows.data.pairs
    was = model.training
    model.eval()
    trainer._start_watchdog()
    values: List[float] = []
    try:
        with torch.no_grad():
            for batch in rows.data.ordered():
                parts = []
                for sub, at in run._slices(batch):
                    feed = sub.tensors(trainer.device)
                    with scoring_out_of_memory(trainer, len(at), sub.length, "run"):
                        logits, _ = trainer._forward_eval_logits(
                            {"input_ids": feed["input_ids"],
                             "attention_mask": feed["attention_mask"]})
                        lp = token_logprobs(logits, sub)
                        side.ends += read(logits, lp, sub, rows.ends, end)
                        del logits
                    half = len(sub) // 2 if pairs else len(sub)
                    if pairs:
                        seq = sequence_logprobs(lp, sub).tolist()
                        side.seq += list(zip(seq[:half], seq[half:]))
                    else:
                        t = sub.tensors(lp.device)
                        w = t["weights"] * t["mask"] if "weights" in t else t["mask"]
                        side.rows += (-(lp * w).sum(-1) / w.sum(-1).clamp(min=1.0)).tolist()
                        parts.append((len(at) / len(batch), float(cross_entropy(lp, sub).value)))
                    _stop_if_short(trainer)
                if parts:
                    values.append(sum(share * v for share, v in parts))
    except RuntimeError as exc:  # mps: the allocator's out of memory is a plain RuntimeError
        trainer._raise_if_mps_oom(exc)
        raise
    finally:
        trainer._stop_watchdog()
        if was:
            model.train()
    side.loss = sum(values) / len(values) if values else math.nan
    return side


def _stop_if_short(trainer: Any) -> None:
    """A Mac short of memory stops between slices, with nothing to keep: an eval changes nothing."""
    dog = trainer._watchdog
    if dog is not None and dog.fired.is_set():
        from poros.core.errors import PorosOOMError

        raise PorosOOMError(f"{dog.why} while scoring.",
                            remediation="Close other apps, then eval again; --rows N scores "
                                        "fewer rows.")


def trained(trainer: Any, adapter: Path) -> None:
    """The run's adapter on the engine, on the base it trained on."""
    if trainer.device.type == "mps":
        from poros import metal

        metal.same_base(_revision(adapter), trainer.config.model_revision, adapter,
                        trainer.config.model_name_or_path)
    trainer.load_checkpoint(str(adapter), load_optimizer=False, load_rng=False)


def numbers(base: Side, run: Side, rows: Rows, beta: float) -> dict:
    """What the two passes say, as numbers and counts."""
    out: dict = {"base": {}, "trained": {}}
    if rows.data.pairs:
        import torch
        import torch.nn.functional as F

        ref, pol = torch.tensor(base.seq), torch.tensor(run.seq)
        # DPO's margin, with the base as its reference
        margin = beta * ((pol[:, 0] - ref[:, 0]) - (pol[:, 1] - ref[:, 1]))
        for name, said, m in (("base", ref, torch.zeros_like(margin)), ("trained", pol, margin)):
            out[name] = {"loss": float(-F.logsigmoid(m).mean()),
                         "prefers": int((said[:, 0] > said[:, 1]).sum()),
                         "margin": float((said[:, 0] - said[:, 1]).mean())}
        out["improved"] = int((margin > 0).sum())
    else:
        out["base"]["loss"], out["trained"]["loss"] = base.loss, run.loss
        out["improved"] = sum(t < b for b, t in zip(base.rows, run.rows))
    b, t = out["base"]["loss"], out["trained"]["loss"]
    out["change"] = (t - b) / b if b and math.isfinite(b) and math.isfinite(t) else None
    seen = [i for i, ends in enumerate(rows.ends) if ends]
    if rows.ends:
        out["answer_rows"] = len(seen)
    for name, side in (("base", base), ("trained", run)) if seen else ():
        out[name]["ends"] = ended(side.ends)[0]
        out[name]["end_p"] = sum(sum(p for _, p in side.ends[i]) / len(side.ends[i])
                                 for i in seen) / len(seen)
    return out


# ── Answers, for a person to read (--show) ─────────────────────────────

MOST_NEW = 48  # tokens an answer gets: enough to read its start, a forward a token
SAID = 240  # characters of any one text shown


@dataclass
class Ask:
    """One row to answer: its prompt's tokens, and where its words are in its file."""

    prompt: List[int]
    number: Optional[int]  # its 1-based row in the file
    path: str
    pad: int
    stops: Tuple[int, ...] = ()


def prompts(rows: Rows, n: int) -> List[Ask]:
    """The first ``n`` rows' prompts: each token up to its last answer. Chats and pairs only:
    plain text has no question to answer."""
    if rows.shape not in ("chats", "pairs"):
        return []
    out = []
    numbers = rows.data.numbers or []
    for i, item in enumerate(rows.data.items[:n]):
        d = item[0] if rows.data.pairs else item
        w = d.weights or []
        scored = [j for j, x in enumerate(w) if x > 0]
        if not scored:
            continue
        start = scored[-1]
        while start > 0 and w[start - 1] > 0:  # back to where the last answer begins
            start -= 1
        seq = list(d.tokens) + [d.target_tokens[-1]]
        out.append(Ask(seq[: start + 1], numbers[i] if i < len(numbers) else None, rows.path,
                       rows.data.pad_id))
    return out


def answers(trainer: Any, asked: List[Ask], end: Optional[int],
            eos: Optional[int] = None) -> List[List[int]]:
    """Each prompt's greedy answer through the run's own forward, a token at a time, to its
    end-of-turn token, the end of text, ``MOST_NEW`` tokens or the run's length. Padded to the
    run's length on a Mac (the one length its memory was measured at), else to a multiple of 128,
    on the right, so no position moves."""
    import torch

    model, length = trainer.model, int(trainer.config.max_seq_length)
    stops = {t for t in (end, eos) if isinstance(t, int)}
    mac = trainer.device.type == "mps"
    was = model.training
    model.eval()
    trainer._start_watchdog()
    out: List[List[int]] = []
    try:
        with torch.no_grad():
            for ask in asked:
                ids, said = list(ask.prompt), []
                for _ in range(MOST_NEW):
                    n = len(ids)
                    if n >= length:
                        break
                    width = length if mac else min(length, -(-n // 128) * 128)
                    x = torch.full((1, width), ask.pad, dtype=torch.long, device=trainer.device)
                    x[0, :n] = torch.tensor(ids, dtype=torch.long, device=trainer.device)
                    mask = torch.zeros_like(x)
                    mask[0, :n] = 1
                    logits, _ = trainer._forward_eval_logits(
                        {"input_ids": x, "attention_mask": mask})
                    nxt = int(logits[0, n - 1].argmax())
                    del logits
                    _stop_if_short(trainer)
                    if nxt in stops or nxt in ask.stops:
                        break
                    ids.append(nxt)
                    said.append(nxt)
                out.append(said)
    except RuntimeError as exc:
        trainer._raise_if_mps_oom(exc)
        raise
    finally:
        trainer._stop_watchdog()
        if was:
            model.train()
    return out


def plain(text: Any, most: int = SAID) -> str:
    """Words a file or a model wrote, as one plain line: every control and format character (an
    escape, a bidi mark, a zero width) a space, runs of space one, at most ``most`` long."""
    import unicodedata

    said = "".join(" " if unicodedata.category(ch)[0] == "C" else ch for ch in str(text or ""))
    said = " ".join(said.split())
    return said if len(said) <= most else said[: most - 1].rstrip() + "…"


def _words(path: str, numbers: List[Optional[int]]) -> dict:
    """{row number: (its last question, the answer its file gives)} for those rows, in words."""
    want = {n for n in numbers if n is not None}
    found: dict = {}
    if not want:
        return found
    for n, row in pdata._rows(path):
        if n not in want:
            continue
        found[n] = _said(row, n)
        if len(found) == len(want):
            break
    return found


def _said(row: dict, n: int) -> Tuple[str, str]:
    def text(x: Any) -> str:
        if isinstance(x, list):  # messages: the last one's words
            x = x[-1].get("content", "") if x and isinstance(x[-1], dict) else ""
        return plain(x)

    if "chosen" in row and "prompt" in row:
        return text(row["prompt"]), text(row["chosen"])
    try:
        turns = pdata._turns(row, n) or []
    except ValueError:
        return "", ""
    last = max((i for i, t in enumerate(turns) if t.get("role") == "assistant"), default=None)
    if last is None:
        return "", ""
    asked = next((t for t in reversed(turns[:last]) if t.get("role") == "user"), {})
    return plain(asked.get("content", "")), plain(turns[last].get("content", ""))


def samples(tok: Any, asked: List[Ask], base: List[List[int]], ours: List[List[int]]) -> list:
    """What ``--show`` writes: each row's question and its file's answer, then the base's answer
    and the run's, decoded and made plain."""
    from poros.cli.schemas import EvalSample

    words = _words(asked[0].path, [a.number for a in asked]) if asked else {}
    out = []
    for ask, b, o in zip(asked, base, ours):
        q, want = words.get(ask.number, ("", ""))
        out.append(EvalSample(q, want, plain(tok.decode(b, skip_special_tokens=True)),
                              plain(tok.decode(o, skip_special_tokens=True))))
    return out


def file_samples(path: str, n: int) -> list:
    """A preview's: the first ``n`` rows' questions and answers, and no model's, as none ran."""
    from poros.cli.schemas import EvalSample

    out = []
    if not path or not Path(path).is_file():
        return out
    try:
        for number, row in pdata._rows(path):
            q, want = _said(row, number)
            if q or want:
                out.append(EvalSample(q, want))
            if len(out) >= n:
                break
    except (OSError, ValueError):
        return out
    return out
