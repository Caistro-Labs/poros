"""`poros eval <run>` and `poros export <run>`: a trained adapter scored, and merged into a
plain transformers / vLLM / TGI model (or kept as an adapter)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import typer

from poros.cli import contract, root
from poros.cli.contract import refuse
from poros.cli.errors import _friendly_error
from poros.cli.root import (
    _JSON_OPTION,
    _adapter_revision,
    _load_yaml_config,
    _no_grants,
    _not_on_a_mac,
    _ui,
    app,
)
from poros.core import checkpoints as ckpt_contract
from poros.preflight import arch as preflight_arch

if TYPE_CHECKING:  # torch-free at runtime; PorosConfig is imported per-command
    from poros.core.config import PorosConfig

# Set by the `export` verb, which emits its own result (poros.cli.verbs.export).
_EXPORT_RESULT_TO_CALLER = False


# poros/train/export.py's STAGING_MARKER, named here so this check needs no torch
# (tests/test_export_staging.py holds the two the same).
STAGING_MARKER = ".poros-export-staging"


def _foreign_stage(out_path: Path) -> Optional[str]:
    """poros.train.export.foreign_stage, torch-free: why ``<out>.partial`` is in the way."""
    import os

    out = out_path.resolve()
    stage = out.with_name(out.name + ".partial")
    if not os.path.lexists(stage):
        return None
    if stage.is_dir() and not stage.is_symlink() and (stage / STAGING_MARKER).is_file():
        return None
    return (f"{stage} is in the way: a merged export stages its files there, and that folder "
            "was not made by an export, so it is left as it is; move or rename it")


def _resolve_adapter_dir(path: Path) -> Path:
    """Resolve a user-supplied adapter/run directory to loadable weights.

    Gated on ``has_weights``, NOT ``is_complete``: loading needs only the
    adapter files, while is_complete additionally demands the RNG/step state
    that only *resuming* requires -- gating here would reject a plain
    ``peft_model.save_pretrained()`` directory or a Hub download. When the
    directory is a run root with no adapter of its own, fall back to its
    newest checkpoint through the same resolver ``--resume-from latest``
    uses, so the two commands can never disagree about which checkpoint is
    newest.
    """
    if not path.exists():
        refuse("E_NOT_FOUND", "No adapter", f"nothing at {path}",
               context=["train first, or point --adapter at a run or checkpoint directory"])
    if ckpt_contract.has_weights(path):
        return path

    # Adopt anything stranded by a killed save before choosing (identical to
    # the resume path), then take the newest complete checkpoint.
    ckpt_contract.recover_checkpoint_dirs(path)
    complete = ckpt_contract.list_checkpoints(path)
    if not complete:
        refuse("E_NOT_FOUND", "No adapter",
               f"{path} holds no adapter weights and no complete checkpoint",
               context=["point --adapter at a directory with "
                        "adapter_model.safetensors or adapter_weights.pt"])
    newest = complete[-1][1]
    from poros.cli import views

    _ui()[0].print(views.note(f"Using {newest.name}",
                              f"{path.name} has no adapter of its own; "
                              "its newest checkpoint is evaluated"))
    return newest


def _warn_on_adapter_config_mismatch(adapter_dir: Path, cfg: "PorosConfig",
                                     as_trained: Optional[Path] = None) -> None:
    """Loudly flag an eval config that disagrees with how the adapter trained. ``as_trained``:
    a recipe's run's folder, whose engine is built from the settings that folder recorded
    (its config.resolved.json), not the eval config's: the note names them.

    A trained adapter composed onto the wrong base (or a different LoRA
    shape) yields confident, meaningless metrics -- the same class of
    failure as evaluating untrained weights. ``poros export`` already reads
    the adapter's own poros_config.json for this reason; eval must not be
    laxer. Advisory rather than fatal: the loader raises on a genuine
    topology mismatch, and a deliberate cross-base eval stays possible.
    """
    import json as _json

    from poros.cli import views

    manifest = adapter_dir / "poros_config.json"
    if not manifest.exists():
        return
    try:
        trained = _json.loads(manifest.read_text())
    except (OSError, ValueError):
        return
    for field in ("model_name_or_path", "lora_rank", "lora_target_modules"):
        want, got = trained.get(field), getattr(cfg, field, None)
        if want is not None and got is not None and want != got:
            _ui()[0].print(views.note(
                f"Mismatch: {field}",
                f"trained with {want}, the eval config says {got}; " + (
                    # a recipe's run is rebuilt from its own config.resolved.json
                    f"it is scored with the settings {as_trained / 'config.resolved.json'} "
                    "recorded, not the eval config's" if as_trained is not None else
                    "metrics from a mismatched base or LoRA shape are not meaningful")))


@app.command(name="eval")
def evaluate(
    config: Path = typer.Argument(
        ...,
        help="Path to a YAML eval config.",
        exists=True,
        readable=True,
    ),
    adapter: Optional[Path] = typer.Option(
        None,
        "--adapter",
        "-a",
        help=(
            "Trained adapter or checkpoint directory to evaluate. "
            "Defaults to the config's output_dir."
        ),
    ),
    trust_remote_code: bool = typer.Option(
        False, "--trust-remote-code",
        help="Allow the base model's repository to execute custom code. Only this flag grants "
        "that; a config file never can.",
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Evaluate a trained adapter on a config's dataset (CUDA).

    Loads the adapter (from --adapter, else the config's output_dir), then
    runs evaluation and prints metrics. Requires torch.
    A run is scored with `poros eval <run>`, beside its base, on a Mac too.
    """
    _not_on_a_mac("poros eval CONFIG.yaml", hint="poros eval <run> scores a run on this Mac.")
    from poros.core.config import PorosConfig

    raw = _load_yaml_config(config)
    _no_grants(raw, config, flagged=trust_remote_code)
    try:
        cfg = PorosConfig(**raw)
    except Exception as exc:
        _friendly_error(exc)
        return
    cfg.trust_remote_code = trust_remote_code or cfg.trust_remote_code

    # Resolve the adapter BEFORE torch and any model download: evaluating
    # the wrong weights is the failure this command exists to prevent, so a
    # missing adapter must fail in seconds, not after a 100 GB pull.
    adapter_dir = _resolve_adapter_dir(Path(adapter) if adapter else Path(cfg.output_dir))
    folder = _recipe_run(adapter_dir, Path(cfg.output_dir))  # a recipe's run: scored as trained
    _warn_on_adapter_config_mismatch(adapter_dir, cfg, as_trained=folder)
    if cfg.model_revision is None:
        # Evaluate against the SAME frozen base the adapter was trained on.
        import json as _json

        for f, key in (("poros_config.json", "model_revision"),
                       ("adapter_config.json", "revision")):
            try:
                cfg.model_revision = _json.loads(
                    (adapter_dir / f).read_text()).get(key)
            except (OSError, ValueError, AttributeError):
                continue
            if cfg.model_revision:
                break
        _adapter_revision(cfg)

    root._require_torch("poros eval")

    if not cfg.dataset:
        refuse("E_CONFIG", "No eval dataset", "the config has no dataset field",
               context=["set dataset: to a Hub dataset id or a local path"])

    import time as _time

    from poros.cli import views

    t, uc, w = _ui()
    # the engine's own loop: its file of held-out rows where it has one, else its training rows
    scored = str(cfg.eval_dataset or cfg.dataset) if folder is None else str(cfg.dataset)
    t.print(views.eval_start(cfg.model_name_or_path, scored, str(adapter_dir), w))
    if folder is not None:  # scored as it trained: what `poros eval <run>` says of a run
        # the caller's own flag, never a file's word: the run's settings carry no such grant
        _eval_as_trained(folder, adapter_dir, cfg, t, w, trust_remote_code=trust_remote_code)
        return
    short = cfg.model_name_or_path.rstrip("/").split("/")[-1]
    try:
        from poros.train.trainer import PorosTrainer

        # config= is authoritative (from_pretrained overwrites the
        # individual kwargs from it), so pass it alone.
        with t.activity(f"Loading {short}") as act:
            # an eval holds less than the training its config planned: no training-fit check
            trainer = PorosTrainer.from_pretrained(cfg.model_name_or_path, config=cfg,
                                                   check_fit=False)
            # Load the TRAINED adapter before measuring anything. Without this
            # the command evaluated the freshly-initialized LoRA weights that
            # from_pretrained attaches -- every metric it printed described an
            # untrained model while naming the user's run directory.
            trainer.load_checkpoint(
                str(adapter_dir), load_optimizer=False, load_rng=False
            )
            act.done([uc.event("Loaded", short, uc.duration(act.elapsed))])
        t0 = _time.monotonic()
        with t.activity("Evaluating") as act:
            results = trainer.evaluate(scored)
            act.done([])
        # what this number is: the classic loop trains, and so scores, every token of a row; of
        # its held-out file, or of the rows it trained on, which are no held-out measure
        results = {**results, "scored": ("every token of the held-out rows" if cfg.eval_dataset
                                         else "every token of the training rows: none were "
                                              "held out")}
        t.print(views.eval_done(results, _time.monotonic() - t0, w))
        if contract.json_mode():
            from poros.cli.schemas import EvalOutput

            contract.emit(EvalOutput(adapter=str(adapter_dir), dataset=scored,
                                     metrics=results))
    except Exception as exc:
        _friendly_error(exc)


def _recipe_run(adapter_dir: Path, output_dir: Path) -> Optional[Path]:
    """The folder of the run that trained ``adapter_dir``, where a recipe trained it (SFT on
    chats or text, a pair objective, a loop file): its ``config.resolved.json`` says so. None
    for the classic loop's run, and for an adapter with no record of how it trained."""
    from poros.cli.verbs import runs_report
    from poros.cli.verbs.evaluate_pass import SHAPES

    for folder in dict.fromkeys((adapter_dir, adapter_dir.parent, output_dir)):
        said = runs_report._read(folder / "config.resolved.json")
        if isinstance(said, dict):
            recipe = said.get("trained_with") not in (None, "classic")
            return folder if recipe and said.get("data_shape") in SHAPES else None
    return None


def _eval_as_trained(folder: Path, adapter_dir: Path, cfg: "PorosConfig", t, w: int,
                     trust_remote_code: bool = False) -> None:
    """`poros eval <config>` of a recipe's run: its rows as it trained them (a chat's answers,
    a pair's two sides), the base and then the run, as `poros eval <run>` scores a run. The
    rows it held out, or with none held out the config's data, whole, and said so. The classic
    loop's every-token loss is not this run's measure: an SFT run that ends at 0.00 on its
    answers reads 3.6 over its prompts and template, which it never trained on.
    ``trust_remote_code``: the command's own --trust-remote-code, which alone grants it."""
    from poros.cli import views
    from poros.cli.context import Context
    from poros.cli.verbs import evaluate as verb
    from poros.core.refusal import Refusal

    try:
        out = verb.score(verb.folder_run(folder, cfg.model_name_or_path), adapter_dir, Context(),
                         dataset=str(cfg.dataset), trust_remote_code=bool(trust_remote_code))
    except Refusal as r:  # its fix names a run by id, which a config's run has none of
        contract.refuse(r.code, r.title, r.reason, context=r.context)
    except typer.Exit:
        raise
    except Exception as exc:
        _friendly_error(exc)
    out.run = None
    out.verdict, out.warning = verb.verdict(out, cfg.model_name_or_path)
    note = ("it held no rows out: these are rows it trained on"
            if out.rows_from == "file" else "")
    t.print(views.evaluated(out, w, name=folder.name, note=note,
                            then=f"poros export {adapter_dir} -o ./merged"))
    if contract.json_mode():
        contract.emit(out)


@app.command()
def export(
    adapter_dir: Path = typer.Argument(
        ...,
        help="Path to the adapter directory.",
        exists=True,
    ),
    format: str = typer.Option(
        "safetensors",
        "--format",
        "-f",
        help="Export format. Supported: safetensors, or its other name merged (merged HF "
        "weights).",
    ),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory for the merged model."
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help="Allow the base model's repository to execute custom code. "
        "Only this flag grants that; a downloaded adapter's own manifest "
        "never can.",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Write into a non-empty output directory: export overwrites "
        "its files."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Every confirmation given: --overwrite, and the base's "
        "download."
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Merge LoRA adapters into the base model and save deployable weights.

    Writes a merged model in HuggingFace safetensors format that any
    inference stack (vLLM, TGI, transformers) can load. GGUF/llama.cpp
    conversion is a separate step -- run llama.cpp's convert_hf_to_gguf.py
    on the directory this produces. Requires torch.
    """
    import time as _time

    from poros.cli import views

    t, uc, w = _ui()
    supported_formats = {"safetensors", "merged"}  # merged: `poros export <run>`'s name for it
    format = format.lower()
    if format not in supported_formats:
        refuse("E_USAGE", "Unsupported format",
               f"{format!r} is not one; supported: {', '.join(sorted(supported_formats))}",
               context=["for GGUF: export safetensors, then run llama.cpp's convert_hf_to_gguf.py"])

    out_path = Path(output) if output else adapter_dir / "merged"
    if not out_path.resolve().name:  # "/": the merge is staged beside its folder, and / has none
        refuse("E_USAGE", "Not a folder of its own", f"{out_path} is the file system's root",
               context=["name a folder of its own: --output ./merged"])
    # deps first, then the config, the disk and any question (the order every command keeps);
    # found, not imported: beside a run, the GPU turn below refuses with no torch in the process
    root._require_torch("poros export", import_it=False)

    # The saved checkpoint records the base model + LoRA config in
    # poros_config.json. We MUST load the real base from there -- the adapter
    # directory is not itself a base model.
    cfg_path = adapter_dir / "poros_config.json"
    if not cfg_path.exists():
        refuse("E_NOT_FOUND", "No poros_config.json",
               f"{adapter_dir} has no record of its base model",
               context=["export needs the config written by poros train / trainer.save()"])

    try:
        import json as _json

        from poros.core.config import PorosConfig
        from poros.stores import recorded_store

        saved = _json.loads(cfg_path.read_text())
        # Machine-written by an earlier poros version: tolerate fields that
        # version knew and this one does not (strictness is for human YAML).
        # A store_path is read, never written: a Mac's merge fetches its base into the store.
        cfg = PorosConfig(**recorded_store({k: v for k, v in saved.items()
                                            if k in PorosConfig.model_fields},
                                           say=lambda said: None)[0])  # builds none
    except Exception as exc:
        _friendly_error(exc)
        return


    # Remote-code authorization comes from the CLI flag ALONE. The manifest
    # is data found inside the adapter directory -- for a downloaded adapter
    # that is attacker-controlled, and honoring its trust_remote_code would
    # let the artifact authorize executing its own chosen repository's code.
    cfg.trust_remote_code = trust_remote_code

    # Load the SAME frozen base the adapter was trained on: a same-shaped
    # later Hub revision passes every key/shape check while silently
    # changing the weights under the merge. The Poros manifest pins it;
    # older/hub adapters carry it in PEFT's adapter_config.json.
    if cfg.model_revision is None:
        try:
            peft_cfg = _json.loads((adapter_dir / "adapter_config.json").read_text())
            cfg.model_revision = peft_cfg.get("revision")
        except (OSError, ValueError, AttributeError):
            pass
        _adapter_revision(cfg)

    if out_path.is_dir() and any(out_path.iterdir()):
        from poros.cli.commands.train import _ask_to_overwrite

        # asked as `poros export <run>` and the train forms ask it: --overwrite (--yes, its name
        # before), a poros.needs_input.v1 question where no person can answer
        _ask_to_overwrite(str(out_path), overwrite or yes,
                          said=f"{out_path} is not empty; export overwrites its files.",
                          verb="export")
    in_the_way = _foreign_stage(out_path)
    if in_the_way:  # before the download and the card: the merge would stage beside it
        refuse("E_USAGE", "Folder in the way", in_the_way)
    if sys.platform != "darwin":  # a merge reads every shard of its base: a download is asked
        from poros.cli.verbs.shared import disk_text
        from poros.plan.configs import download_size

        fetch = download_size(cfg).value
        if fetch:
            from rich.markup import escape

            contract.confirm(f"The merge downloads {escape(cfg.model_name_or_path)} first: "
                             f"{disk_text(fetch * 2**30 / 1e9)} from Hugging Face.", yes)
    ckpt_contract.drop_links(out_path)  # replaced, not followed

    t.print(views.export_start(str(adapter_dir), cfg.model_name_or_path, format, str(out_path), w))
    if trust_remote_code:
        t.print(views.note("--trust-remote-code",
                           "code from the base model's repository will be executed"))

    from poros.cli.verbs.shared import hold_gpu
    from poros.core.arch_registry import validated_arch_types

    if sys.platform == "darwin":
        # Before anything resolves a backend: a model on a Mac is a skeleton until it trains.
        from poros.metal.export import merged, merges

        if not merges(cfg.model_name_or_path):
            refuse("E_EXPORT_UNSUPPORTED", "Can't merge yet",
                   f"{cfg.model_name_or_path} is above 7B on a Mac",
                   context=["a merged model is not in this release at this size",
                            "the adapter is the folder named: it loads as it is"])
    # Its turn on the card, on no torch: the open file holds it until the merge is written.
    held = hold_gpu("a merged export", queue=True)  # noqa: F841
    root._require_torch("poros export")
    t0, began = _time.monotonic(), _time.time_ns()
    if sys.platform == "darwin":
        try:
            with t.activity("Merging the adapter, shard by shard") as act:
                merged(cfg, adapter_dir, out_path)
                act.done([])
        except Exception as exc:
            _friendly_error(exc)
        _export_done(adapter_dir, out_path, format, began)
        _show_done(t, views.export_done(str(out_path), _time.monotonic() - t0, w))
        return
    model_type = preflight_arch.resolve_model_type(cfg.model_name_or_path)[0]
    if model_type in ("inkling2_loop", "inkling2_loop_text"):
        # Inkling 2 Mini: its experts and attention merged into the released layout, unquantized
        try:
            import torch

            from poros.moe.inkling2.export import export_merged as inkling_merged

            with t.activity("Merging experts and attention, shard by shard") as act:
                inkling_merged(
                    cfg.model_name_or_path,
                    torch.load(adapter_dir / "adapter_weights.pt", weights_only=True),
                    str(out_path),
                    scaling=cfg.lora_alpha / cfg.lora_rank,
                    dtype=getattr(torch, cfg.torch_dtype),
                    revision=cfg.model_revision,
                    expert_codec=cfg.moe_expert_format,  # the run's record, never the environment
                )
                act.done([])
        except Exception as exc:
            _friendly_error(exc)
        _export_done(adapter_dir, out_path, format, began)
        _show_done(t, views.export_done(str(out_path), _time.monotonic() - t0, w))
        return
    if model_type in validated_arch_types("moe"):
        # MoE: merged one expert at a time on CPU, shard by shard, so the
        # model never has to fit anywhere whole.
        try:
            import torch

            from poros.moe.export import export_merged

            with t.activity("Merging experts, shard by shard") as act:
                export_merged(
                    cfg.model_name_or_path,
                    torch.load(adapter_dir / "adapter_weights.pt", weights_only=True),
                    str(out_path),
                    scaling=cfg.lora_alpha / cfg.lora_rank,
                    dtype=getattr(torch, cfg.torch_dtype),
                    revision=cfg.model_revision,
                )
                act.done([])
        except Exception as exc:
            _friendly_error(exc)
        _export_done(adapter_dir, out_path, format, began)
        _show_done(t, views.export_done(str(out_path), _time.monotonic() - t0, w))
        return

    # Dense: the same walk, one tensor at a time on CPU. Each weight is what
    # training saw (an NF4 run: its NF4 round trip, on the GPU when there is
    # one, exactly as the training load quantized it) plus scaling * B @ A,
    # rounded once to bf16 -- never re-quantized, and never whole-model
    # resident, so a model larger than the card merges too.
    tok_ok = True
    try:
        from poros.train.export import export_merged_dense

        with t.activity("Merging adapters, shard by shard") as act:
            report = export_merged_dense(cfg, adapter_dir, str(out_path))
            act.done([])
        if report.nf4_roundtrip == "cpu":
            t.print(views.note("NF4 base rebuilt on the CPU",
                               "no CUDA GPU here; a weight on an NF4 code boundary can "
                               "round to the neighbouring code (bitsandbytes quantizes "
                               "on the GPU with fast math)"))

        # Save the tokenizer alongside the weights so the directory is directly
        # loadable by transformers / vLLM / TGI without a separate fetch.
        from poros.core.tokenizer import TokenizerRefused, load_tokenizer

        try:
            tok = load_tokenizer(
                cfg.model_name_or_path, revision=cfg.model_revision,
                trust_remote_code=cfg.trust_remote_code,
            )
            tok.save_pretrained(str(out_path))
        except TokenizerRefused:
            raise
        except Exception as tok_exc:  # noqa: BLE001
            tok_ok = False
            t.print(views.note("No tokenizer saved",
                               f"{type(tok_exc).__name__}; load it from "
                               f"{cfg.model_name_or_path}"))

        _export_done(adapter_dir, out_path, format, began)
        _show_done(t, views.export_done(str(out_path), _time.monotonic() - t0, w,
                                        tokenizer=tok_ok))
    except typer.Exit:
        raise
    except Exception as exc:
        _friendly_error(exc)


def _show_done(t, lines: list) -> None:
    """The export's own result card: not where the `export` verb called this command, which
    shows its own (one card, one next step)."""
    if not _EXPORT_RESULT_TO_CALLER:
        t.print(lines)


def _export_done(adapter_dir: Path, out_path: Path, format: str, began: int) -> None:
    """What it wrote follows the umask. JSON mode: the export result object (the human screen is
    printed beside it)."""
    ckpt_contract.follow_umask(out_path, began)
    if contract.json_mode() and not _EXPORT_RESULT_TO_CALLER:
        from poros.cli.schemas import ExportOutput

        contract.emit(ExportOutput(adapter=str(adapter_dir), output=str(out_path), format=format))
