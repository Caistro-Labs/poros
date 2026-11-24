"""PorosTrainer's checkpoints and resume: writing a checkpoint (adapter, optimizer, RNG,
schedule) at a step boundary, loading one back bit for bit, pruning old ones, and saving the
finished adapter (PEFT or native, with its tokenizer and a portable base identity)."""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Iterator, Optional

import torch

from poros import metal
from poros.core import checkpoints
from poros.core.config import recorded_config
from poros.core.errors import PorosConfigError, PorosDiskError
from poros.preflight.resume import DATA_RECORD, INPUTS_RECORD, inputs_record, snapshot_sha
from poros.train.adapters import (
    get_trainable_state_dict,
    validate_native_adapter_state,
    validate_peft_adapter_state,
)
from poros.train.state import _TrainerState

logger = logging.getLogger("poros.train.trainer")  # the trainer's own log, as before


# What a disk that refuses a write says: an OSError's errno, or (torch.save, which reports a short
# write only in a RuntimeError's text) its stream writer's words, with the disk nearly full.
_FULL_ERRNOS = frozenset({errno.ENOSPC, errno.EDQUOT, errno.EFBIG})
_SHORT_WRITE = ("No space left", "iostream error", "unexpected pos", "file write failed",
                "PytorchStreamWriter failed writing")
_NEARLY_FULL = 1 << 30  # bytes free at the failure under which a short write is the disk's


def disk_full(exc: BaseException, where: "str | os.PathLike") -> Optional[str]:
    """Why a save failed, where the disk refused it ("the disk is full: 0.0 GB free at ..."), or
    None for any other failure. Reads ``exc``'s whole cause chain."""
    try:
        free = shutil.disk_usage(_nearest(Path(where))).free
    except OSError:
        free = None
    seen, e = set(), exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, OSError) and e.errno in _FULL_ERRNOS:
            break
        if (isinstance(e, RuntimeError) and any(w in str(e) for w in _SHORT_WRITE)
                and free is not None and free < _NEARLY_FULL):
            break
        e = e.__cause__ or e.__context__
    else:
        return None
    if e is None:
        return None
    at = f"{free / 1e9:.1f} GB free at {where}" if free is not None else f"at {where}"
    return f"the disk is full ({at})"


def _nearest(p: Path) -> Path:
    """``p`` or its nearest existing parent (a failed save may have removed ``p``)."""
    for q in (p, *p.parents):
        if q.exists():
            return q
    return Path("/")



def _sorted_config_sets(peft_model: Any, out_path: Path) -> None:
    """Each adapter_config.json PEFT just wrote, with every field its config holds as a set
    (target_modules among them) listed sorted: PEFT lists a set in its iteration order, which a
    string's hash decides per process, so two runs with the same trained bits wrote different
    files. Written as PEFT writes it (indent 2, keys sorted); PEFT reads a list back as the same
    set, so loading is unchanged."""
    import json

    for name, cfg in (getattr(peft_model, "peft_config", None) or {}).items():
        sets = [k for k, v in vars(cfg).items() if isinstance(v, (set, frozenset))]
        path = (out_path if name == "default" else out_path / name) / "adapter_config.json"
        if not sets or not path.is_file():
            continue
        said = json.loads(path.read_text())
        for k in sets:
            if isinstance(said.get(k), list):
                said[k] = sorted(said[k], key=str)
        path.write_text(json.dumps(said, indent=2, sort_keys=True))

class _Checkpointing(_TrainerState):
    """PorosTrainer's checkpoints, resume and saved adapters (mixed into PorosTrainer)."""

    def _same_run(self, ckpt: Path) -> list:
        """Refuse resuming ``ckpt`` into another run: the settings its poros_config.json names
        and the dataset and base it recorded (poros.preflight.resume.resume_verdict, which
        ``poros train`` asks before the load), against what this run trains on as it took it
        when it read it (``_dataset_record``; a recipe's rows by their own record), and the base
        revision it trained on against the one this trainer loaded. What could not be checked is
        warned of, once, and returned (the resume event keeps it)."""
        from poros.preflight.resume import (
            CONFIG,
            SAME_ROWS,
            SAME_SETTINGS,
            resume_verdict,
            rows_clash,
            rows_recorded,
            warn_once,
        )

        try:
            saved = json.loads((ckpt / "poros_config.json").read_text())
        except (OSError, ValueError):
            saved = None
        cfg, unverified = self.config, []
        if self._metal and saved is not None:  # first: it reads the base, and names its revision
            said = metal.same_base(saved.get("model_revision"), cfg.model_revision, ckpt,
                                   cfg.model_name_or_path)
            unverified += [said] if said else []
            # compared there, a local folder's by its contents: not again here by its name
            cfg = cfg.model_copy(update={"model_revision": saved.get("model_revision")})
        here = self._dataset_record  # None: a loop of the caller's own hands it its batches
        clash = None
        if self._data_record is not None:  # a recipe's rows, read from the file the config names
            clash = rows_clash(str(ckpt), self._data_record)
            here = SAME_ROWS if rows_recorded(str(ckpt)) else CONFIG if here is None else here
        notes: list = []
        if clash is None:
            clash, notes = resume_verdict(str(ckpt), cfg, dataset=here)
        if clash:
            fix = clash.fix or f"{SAME_SETTINGS}, or start a new run"
            route = getattr(clash, "route", None)  # other data, on purpose: from its adapter
            raise PorosConfigError(
                f"Can't resume: {clash}.",
                remediation=f"{fix[0].upper()}{fix[1:]}."
                + (f" {route[0].upper()}{route[1:]}." if route else ""))
        for said in notes:
            warn_once(said)
        unverified += [n for n in notes if n not in unverified]
        if saved is None or self._metal:
            return unverified
        was, here = saved.get("model_revision"), snapshot_sha(self._resolved_base_path)
        if was and here and was != here:
            raise PorosConfigError(
                f"Can't resume: {ckpt} trained on base revision {was[:12]}, and the base loaded "
                f"here is {here[:12]}.",
                remediation=f"Set model_revision: {was} (the base it trained on), or start a "
                "new run on this one.")
        return unverified

    def _save_checkpoint(self) -> str:
        """Write ``checkpoint-<global_step>`` under output_dir; return its path."""
        # Stage, then promote with a rename-aside: at no instant is the newest
        # checkpoint neither at its final name nor at a recoverable sibling
        # (checkpoints.recover_checkpoint_dirs adopts .tmp/.old leftovers on
        # the next run). Deleting the final dir before the rename would open
        # a window where the only complete copy sits under a name every
        # resolver skips.
        if self._metal:  # a disk that holds what a run keeps must not need room for one more
            self._prune_checkpoints(room_for_one=True)
        final_ckpt = os.path.join(self.config.output_dir, f"checkpoint-{self.global_step}")
        staging = final_ckpt + ".tmp"
        backup = final_ckpt + ".old"
        if os.path.isdir(staging):
            shutil.rmtree(staging, ignore_errors=True)
        try:
            # its scores in the staged folder, written before its completion marker and flushed
            # with every other file: the checkpoint takes its name with them, and no kill or
            # power cut leaves a complete folder without its start score
            self.save(staging, scores=True)
        except BaseException as exc:
            # A failed save must not leave a half-written staging dir for the
            # recovery sweep to misjudge.
            shutil.rmtree(staging, ignore_errors=True)
            # torch.save tells of a full or missing disk only in a RuntimeError's text.
            if self._metal and isinstance(exc, (OSError, RuntimeError)):
                raise PorosDiskError(
                    f"Could not write a checkpoint to {self.config.output_dir}: the disk is "
                    "full, or it is gone.",
                    remediation="Free some space or attach it again, then `poros runs continue`.",
                ) from exc
            raise
        try:
            if os.path.isdir(final_ckpt):
                shutil.rmtree(backup, ignore_errors=True)
                os.replace(final_ckpt, backup)
            os.replace(staging, final_ckpt)
            checkpoints.fsync_path(Path(final_ckpt).parent)  # the rename itself, on the disk
            shutil.rmtree(backup, ignore_errors=True)
        except OSError as exc:
            # NFS/EBUSY/EACCES: keep training. The complete save survives in
            # the staging dir and the next run's recovery sweep promotes it.
            logger.warning(
                "Could not promote checkpoint %s (%s); the save is preserved at "
                "%s and will be recovered on the next run.", final_ckpt, exc, staging,
            )
        self._prune_checkpoints()
        return final_ckpt

    def _keep_score(self, folder: str) -> None:
        """The held-out score of exactly these weights, if the run scored them, in the folder's
        eval.json (numbers only): a score taken at another step is never written beside them."""
        from poros.train.evals import write_score, write_start

        held = getattr(self, "_held_score", None)
        if held is not None and held[0] == self.global_step + 1 and os.path.isdir(folder):
            write_score(Path(folder), held[1])
        start = getattr(self, "_held_start", None)  # and the run's first, which a resume carries
        if start is not None and os.path.isdir(folder):
            write_start(Path(folder), start)

    def save(
        self,
        output_dir: str,
        *,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        scores: bool = False,
    ) -> None:
        """Save adapter weights (and optionally push to HF Hub).

        Saves only the trainable (adapter) parameters, plus the config
        and RNG state for checkpoint/resume. A write the disk refuses (full, or a quota) is a
        ``PorosDiskError`` (E_DISK) that says so, on CUDA as on a Mac; what was written of this
        save carries no completion marker, so no resume takes it for a checkpoint.

        Args:
            output_dir: Directory to save adapter weights.
            push_to_hub: Whether to push to HuggingFace Hub.
            hub_model_id: Hub model ID for pushing.
            scores: Also the held-out scores of these weights and of the run's start
                (``_keep_score``), written before the completion marker.
        """
        try:
            self._save_files(output_dir, push_to_hub=push_to_hub, hub_model_id=hub_model_id,
                             scores=scores)
        except PorosDiskError:
            raise
        except (OSError, RuntimeError) as exc:
            said = disk_full(exc, output_dir)
            if said is None:
                raise
            raise PorosDiskError(
                f"Could not write {output_dir}: {said}. The checkpoints saved before it are "
                "whole; this save is not one.",
                remediation="Free some space (save_total_limit=1 keeps fewer checkpoints, or "
                "give another output_dir), then `poros train <config> --resume-from latest` "
                "goes on from the newest complete checkpoint.",
            ) from exc
        # where it landed, said once per save a caller asked for (a checkpoint's staging folder
        # is the loop's own): the progress display's line, so a script's run ends by naming it
        if self._progress_enabled() and not str(output_dir).endswith(".tmp"):
            self._progress_print(f"Saved to:         {output_dir}")

    def _save_files(
        self,
        output_dir: str,
        *,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        scores: bool = False,
    ) -> None:
        """``save``'s writes, the completion marker last."""
        assert self.model is not None
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Invalidate any PREVIOUS generation before the first byte lands:
        # saving over an already-complete directory would otherwise leave the
        # old completion sentinel authorizing a torn, mixed-generation state
        # if this save is killed midway. (The staging path uses fresh .tmp
        # dirs; this protects direct saves and the final root save.)
        (out_path / checkpoints.COMPLETE_MARKER).unlink(missing_ok=True)

        # Format marker, written strictly FIRST: it stamps this directory as
        # a current-format save, so if the completion sentinel below never
        # lands, completeness checks call it torn instead of falling back to
        # the legacy per-file heuristic (which a partial save can satisfy).
        (out_path / checkpoints.FORMAT_MARKER).write_text(
            checkpoints.CHECKPOINT_FORMAT_VERSION
        )
        (out_path / checkpoints.RUN_MARKER).write_text(self._run_id, encoding="ascii")
        if self._run_parent:  # resumed from an unstamped checkpoint here: one lineage with it
            (out_path / checkpoints.PARENT_MARKER).write_text(self._run_parent, encoding="ascii")

        # Save adapter weights. The PEFT wrapper writes the standard
        # adapter directory (adapter_model.safetensors + adapter_config.json)
        # loadable by vanilla PEFT / lm-eval; the base HF module would write
        # the full model instead, so it must never be used here.
        peft_saver = self._peft_model
        if peft_saver is None and hasattr(self.model, "peft_config"):
            peft_saver = self.model
        if self._adapter_backend == "experts":
            from poros.metal.experts import save_adapter

            save_adapter(self.model, out_path, self.config)
        elif (
            self._adapter_backend == "peft"
            and peft_saver is not None
            and hasattr(peft_saver, "save_pretrained")
        ):
            with self._portable_base_identity(peft_saver):
                peft_saver.save_pretrained(str(out_path))
            _sorted_config_sets(peft_saver, out_path)
        else:
            state = get_trainable_state_dict(
                self.model, trainable_names=self.trainable_parameter_names
            )
            torch.save(state, out_path / "adapter_weights.pt")

        # Save config. resolved_base_model_path is a DECLARED PorosConfig
        # field (machine-written provenance): the emitted file must stay
        # loadable by PorosConfig itself and valid under `poros schema
        # config` -- an undeclared extra key would make poros reject its
        # own artifact on read-back (extra="forbid"). A sequential run records no
        # ``path``, as config.resolved.json (recorded_config).
        config_dict = recorded_config(self.config)
        if self.ran_path is not None:
            config_dict["ran_path"] = self.ran_path
        if self._resolved_base_path:
            config_dict["resolved_base_model_path"] = self._resolved_base_path
            if config_dict.get("model_revision") is None:
                sha = snapshot_sha(self._resolved_base_path)
                if sha is not None:
                    config_dict["model_revision"] = sha
        with open(out_path / "poros_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        if self._data_record is not None:  # the rows it trained on: a resume reads the same
            with open(out_path / DATA_RECORD, "w") as f:
                json.dump(self._data_record, f, indent=2)
        with open(out_path / INPUTS_RECORD, "w") as f:  # what it read, wherever that is next time
            json.dump(self._inputs(), f, indent=2)

        # Save RNG state for reproducible resume
        rng_state = {
            "torch_rng": torch.get_rng_state(),
            "global_step": self.global_step,
        }
        if torch.cuda.is_available():
            rng_state["cuda_rng"] = torch.cuda.get_rng_state(self.device)
        if self.device.type == "mps":
            rng_state["mps_rng"] = torch.mps.get_rng_state()
            rng_state["device_class"] = metal.device_class()
        torch.save(rng_state, out_path / "rng_state.pt")

        # Save optimizer state
        if self.optimizer is not None:
            torch.save(
                self.optimizer.state_dict(), out_path / "optimizer_state.pt"
            )
            from poros.train.expert_adamw import write_rule

            write_rule(out_path, self.optimizer, self.config.torch_dtype)  # every checkpoint

        # Save LR scheduler state so a resumed run continues the same
        # warmup/decay curve instead of restarting it at step 0.
        if self._lr_scheduler is not None:
            torch.save(
                self._lr_scheduler.state_dict(), out_path / "lr_scheduler_state.pt"
            )

        # Save the tokenizer alongside the adapter so the directory is
        # self-contained for inference / export (no need to re-fetch it).
        self._save_tokenizer(out_path)

        # Completion sentinel, written strictly last: its existence is the
        # contract (poros.core.checkpoints) that every resume-critical file
        # above is already on disk. A kill anywhere earlier leaves no marker
        # and the directory is treated as torn. Every file and folder is
        # flushed before it, and it after: a power cut leaves no marker that
        # vouches for bytes still in the page cache.
        if scores:
            self._keep_score(str(out_path))
        checkpoints.fsync_tree(out_path)
        (out_path / checkpoints.COMPLETE_MARKER).write_text(str(self.global_step))
        checkpoints.fsync_path(out_path / checkpoints.COMPLETE_MARKER)
        checkpoints.fsync_path(out_path)

        logger.info("Saved adapter checkpoint to %s", output_dir)

        # Optionally push to Hub
        if push_to_hub:
            _push_to_hub(out_path, hub_model_id or self.config.hub_model_id)

    def _inputs(self) -> dict:
        """What this run read (poros.preflight.resume.INPUTS_RECORD). Its dataset, as the run
        took it when it read it (``_dataset_record``: ``train``, ``Run.steps``), so a file edited
        later is not what a checkpoint says trained, and no save reads the data. Its base's
        folder by content, sampled once at the run's first save and kept (a PEFT run on a Hub id
        learns its snapshot's folder in that save)."""
        if self._inputs_record is None:
            folder = self._resolved_base_path or self.config.model_name_or_path
            self._inputs_record = inputs_record(None, folder)
        if self._dataset_record is None:
            return self._inputs_record
        return {**self._inputs_record, "dataset": self._dataset_record}

    @contextlib.contextmanager
    def _portable_base_identity(self, peft_saver: Any) -> Iterator[None]:
        """Point PEFT's in-memory config at the portable model id for a save.

        PEFT records whatever path the model was loaded from -- for a cached
        hub model, a machine-local snapshot directory -- into BOTH
        adapter_config.json and the model card's ``base_model:`` metadata,
        so a copied or pushed adapter resolves on exactly one machine.
        Setting the id at the source (instead of patching files afterward)
        fixes both files in one write, atomically, with nothing to re-open.

        The substitution is guarded: it happens only when the recorded value
        is provably the configured repo's own HF-cache snapshot (the
        ``models--Org--Name`` path marker) -- a config id that merely
        *differs* from what was loaded must never be stamped on, because
        base and Instruct variants share shapes and the wrong id composes
        silently. When the snapshot path pins a commit sha, the sha is
        carried into the adapter's ``revision`` field so the substitution
        does not trade an exact reference for a floating one.
        """
        configured = getattr(self.config, "model_name_or_path", None)
        peft_configs = getattr(peft_saver, "peft_config", None) or {}
        saved: dict[str, tuple[str, Any]] = {}
        base_cfg = None
        prev_base_name = None
        try:
            if configured:
                for name, pcfg in peft_configs.items():
                    recorded = getattr(pcfg, "base_model_name_or_path", None)
                    if not recorded or recorded == configured:
                        continue
                    marker = "models--" + configured.replace("/", "--")
                    parts = Path(recorded).parts
                    if marker not in parts:
                        logger.warning(
                            "adapter %r records base %r, which is not a "
                            "cache snapshot of the configured %r; keeping "
                            "the recorded value.", name, recorded, configured,
                        )
                        continue
                    saved[name] = (recorded, getattr(pcfg, "revision", None))
                    pcfg.base_model_name_or_path = configured
                    if getattr(pcfg, "revision", None) is None:
                        sha = snapshot_sha(recorded)
                        if sha is not None:
                            pcfg.revision = sha
                    self._resolved_base_path = recorded
                # The model card reads TWO base-model identities, and they
                # are independent: card.data["base_model"] comes from
                # config._name_or_path, while the base_model:adapter:<...>
                # tag reads the model's name_or_path ATTRIBUTE -- a plain
                # attr captured at load time that later config edits do not
                # touch (patching the config alone leaves the tag holding the
                # snapshot path). Patch both, restore both.
                if saved and hasattr(peft_saver, "get_base_model"):
                    try:
                        base_model = peft_saver.get_base_model()
                        base_cfg = base_model.config
                        prev_base_name = getattr(base_cfg, "_name_or_path", None)
                        base_cfg._name_or_path = configured
                        prev_attr = getattr(base_model, "name_or_path", None)
                        base_model.name_or_path = configured
                    except Exception:  # pragma: no cover
                        base_cfg = None
            yield
        finally:
            for name, (recorded, revision) in saved.items():
                peft_configs[name].base_model_name_or_path = recorded
                peft_configs[name].revision = revision
            if base_cfg is not None:
                base_cfg._name_or_path = prev_base_name
                try:
                    base_model.name_or_path = prev_attr
                except Exception:  # pragma: no cover
                    pass

    def start_from_adapter(self) -> Optional[str]:
        """A new run that starts from an earlier run's adapter (``config.init_adapter``: a
        checkpoint, or a run's folder for its newest): its weights are loaded, and the optimizer,
        schedule, RNG and step are this run's own, fresh. Another base, rank or set of target
        modules is refused (``PorosConfigError``). Returns the checkpoint it loaded, or None
        where the config names none. Not a resume: the data and the length are the new run's."""
        if not getattr(self.config, "init_adapter", None):
            return None
        from poros.preflight.resume import require_start

        start, _, said = require_start(self.config)  # the one check, again now the base is here
        was = said.get("lora_alpha")
        if was is not None and was != self.config.lora_alpha:
            logger.warning(
                "The adapter at %s trained with lora_alpha %s, and this run has %s: its weights "
                "act at another strength from the first step. Set lora_alpha to %s to start "
                "where it stopped.", start, was, self.config.lora_alpha, was)
        self.load_checkpoint(start, load_optimizer=False, load_rng=False)
        return start

    def _stepped_state_kept(self, ckpt: Path, saved: Any) -> None:
        """ExpertAdamW takes an empty state as an optimizer's that has not stepped (a save before
        the first update). A checkpoint whose completion record says updates were taken and
        whose optimizer state is empty is refused here, in words: its moments and step counts
        are gone, and a resume would restart them unsaid. torch's AdamW (every trainable tensor
        fp32) is left as it was."""
        from poros.train.expert_adamw import ExpertAdamW

        if not isinstance(self.optimizer, ExpertAdamW) or not isinstance(saved, dict):
            return
        if saved.get("state"):
            return
        try:  # the marker holds the index of the last update taken: -1 before any
            taken = int((ckpt / checkpoints.COMPLETE_MARKER).read_text().strip()) + 1
        except (OSError, ValueError):
            return
        if taken >= 1:
            raise PorosConfigError(
                f"Can't resume: {ckpt} says {taken} update{'s were' if taken > 1 else ' was'} "
                "taken, and its optimizer state is empty: the moments and step counts are "
                "gone, so a resume would restart them unsaid.",
                remediation="Resume from another checkpoint of the run, or start a new run from "
                "its adapter (--from <checkpoint>).")

    def _same_update_rule(self, ckpt: Path) -> None:
        """Before anything is restored: the checkpoint's optimizer state was written by the rule
        this run updates with, for tensors of this run's dtype; else refused, naming both."""
        from poros.core.checkpoints import (
            OLD_RULE,
            OLD_STATE_FIX,
            CheckpointUnreadable,
            checkpoint_record,
            old_state_message,
        )
        from poros.train.expert_adamw import rule_of, trainable_dtype

        try:
            record = checkpoint_record(ckpt)
        except CheckpointUnreadable as exc:
            raise PorosConfigError(f"Can't resume: {exc}.",
                                   remediation="Resume from a checkpoint Poros wrote.") from exc
        if record is None:
            return
        if record["rule"] == OLD_RULE:
            raise PorosConfigError(
                f"Can't resume: {old_state_message(ckpt, record.get('experts'))}.",
                remediation=f"{OLD_STATE_FIX[0].upper()}{OLD_STATE_FIX[1:]}.")
        rule, dtype = rule_of(self.optimizer), trainable_dtype(self.optimizer)
        was = record.get("trainable_dtype")
        if record["rule"] != rule or (was is not None and was != dtype):
            raise PorosConfigError(
                f"Can't resume: {ckpt}'s optimizer state is {record['rule']}'s for {was or 'its'} "
                f"adapters, and this run updates {dtype} adapters with {rule}: one is not the "
                "other's state.",
                remediation="Resume with the settings it trained with (torch_dtype among them), "
                "or start a new run from its adapter (--from <checkpoint>).")
        from poros.core.checkpoints import RULE_FILE
        from poros.train.expert_adamw import other_implementation

        try:
            recorded = json.loads((ckpt / RULE_FILE).read_text()).get("implementation")
        except (OSError, ValueError, AttributeError):
            recorded = None
        said = other_implementation(recorded, self.optimizer)
        if said is not None:
            raise PorosConfigError(f"Can't resume {ckpt}: {said}.",
                                   remediation="Resume on the machine and implementation it "
                                   "stepped with, or start a new run from its adapter "
                                   "(--from <checkpoint>).")

    def load_checkpoint(
        self,
        checkpoint_dir: str,
        *,
        load_optimizer: bool = True,
        load_rng: bool = True,
    ) -> None:
        """Restore a checkpoint written by :meth:`save` for resumed training.

        Restores the trainable (adapter) weights, and optionally the optimizer
        state, RNG state, and step counter. The base model and architecture
        must match the run that produced the checkpoint -- this loads adapters
        and run state, not the frozen base.

        Args:
            checkpoint_dir: Directory containing the saved checkpoint.
            load_optimizer: Restore optimizer state (set False to fine-tune
                from the saved weights with a fresh optimizer).
            load_rng: Restore torch / CUDA RNG state and ``global_step``.
        """
        assert self.model is not None
        ckpt = Path(checkpoint_dir)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")
        # Completeness is validated HERE, not only in the CLI resolver: this
        # method is public API, and loading a torn directory resumes with
        # silently reset run state.
        if load_optimizer and load_rng and not checkpoints.is_complete(ckpt):
            raise FileNotFoundError(
                f"{ckpt} is not a complete checkpoint (interrupted save). "
                "Resume from a complete one (--resume-from latest skips torn "
                "directories)."
            )
        # a resume: the run the checkpoint was, on its base
        unverified = self._same_run(ckpt) if load_optimizer and load_rng else []
        if load_optimizer and self.optimizer is not None:
            self._same_update_rule(ckpt)

        # 1. Adapter weights (PEFT dir layout or native .pt state dict).
        native_weights = ckpt / "adapter_weights.pt"
        if self._adapter_backend == "peft" and (ckpt / "adapter_config.json").exists():
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file

            st_path = ckpt / "adapter_model.safetensors"
            bin_path = ckpt / "adapter_model.bin"
            if st_path.exists():
                adapter_state = load_file(str(st_path))
            elif bin_path.exists():
                adapter_state = torch.load(bin_path, map_location="cpu", weights_only=True)
            else:
                raise FileNotFoundError(
                    f"No PEFT adapter weights found in {ckpt}"
                )
            # Exact-match validation BEFORE mutation: truncated, foreign, and
            # reshaped adapters are all refused with the model untouched.
            # (Checking unexpected_keys after the load cannot catch
            # truncation -- the surviving keys all land, and the missing
            # tensors silently stay at init. `poros eval` rides this path.)
            validate_peft_adapter_state(
                self._peft_model or self.model, adapter_state, str(ckpt)
            )
            set_peft_model_state_dict(
                self._peft_model or self.model, adapter_state
            )
        elif self._adapter_backend == "experts":
            from poros.metal.experts import read_adapter

            state = read_adapter(ckpt)
            validate_native_adapter_state(self.model, state, str(ckpt))
            self.model.load_state_dict(state, strict=False)
        elif native_weights.exists():
            state = torch.load(native_weights, map_location="cpu", weights_only=True)
            validate_native_adapter_state(self.model, state, str(ckpt))
            self.model.load_state_dict(state, strict=False)
        else:
            raise FileNotFoundError(
                f"No adapter weights (PEFT dir or adapter_weights.pt) in {ckpt}"
            )

        # 2. Optimizer state. Missing state under an active flag is a hard
        # error: silently continuing with fresh Adam moments and step 0 is a
        # full restart wearing a resume's clothes.
        opt_path = ckpt / "optimizer_state.pt"
        if load_optimizer and self.optimizer is not None:
            if not opt_path.exists():
                raise FileNotFoundError(
                    f"{ckpt} has no optimizer_state.pt -- this checkpoint was "
                    "interrupted mid-save and cannot restore the optimizer. "
                    "Resume from an earlier complete checkpoint "
                    "(--resume-from latest skips torn ones)."
                )
            # On mps from the CPU: AdamW's step counters stay there, as in a run that never
            # stopped, so they add no sync and the next save has the same bytes.
            saved_state = torch.load(
                opt_path,
                map_location="cpu" if self.device.type == "mps" else self.device,
                weights_only=True,
                # on a Mac, read from the file as the tensors are moved to the GPU, not copied
                # whole into memory first (ExpertAdamW.load_state_dict moves them one at a time)
                mmap=self.device.type == "mps",
            )
            self._stepped_state_kept(ckpt, saved_state)
            self.optimizer.load_state_dict(saved_state)
            del saved_state

        # 2b. LR scheduler state (keep the warmup/decay curve aligned to the
        # resumed step). Missing state under an active scheduler is the same
        # hard error as missing optimizer state: silently continuing restarts
        # the warmup/decay curve at step 0, breaking bitwise-faithful resume.
        sched_path = ckpt / "lr_scheduler_state.pt"
        if load_optimizer and self._lr_scheduler is not None:
            if not sched_path.exists():
                raise FileNotFoundError(
                    f"{ckpt} has no lr_scheduler_state.pt but this run uses "
                    "an LR scheduler -- resuming would restart the schedule "
                    "from step 0. This happens when the checkpoint was "
                    "written by a run without a scheduler (lr_scheduler: "
                    "constant, or an older Poros); resume with the "
                    "checkpoint's own scheduler config, or pass "
                    "load_optimizer=False to start a fresh schedule "
                    "deliberately."
                )
            self._lr_scheduler.load_state_dict(
                torch.load(sched_path, map_location="cpu", weights_only=True)
            )

        # 3. RNG state + step counter (for bitwise-faithful resume).
        rng_path = ckpt / "rng_state.pt"
        if load_rng and not rng_path.exists():
            raise FileNotFoundError(
                f"{ckpt} has no rng_state.pt -- without it the resumed run "
                "cannot restore its step counter or RNG stream and would "
                "silently restart from step 0. Resume from a complete "
                "checkpoint (--resume-from latest skips torn ones)."
            )
        if load_rng and rng_path.exists():
            rng_state = torch.load(rng_path, map_location="cpu", weights_only=True)
            # Optimizer moments and RNG streams do not cross a device class; the adapter does.
            here = metal.device_class() if self.device.type == "mps" else None
            if rng_state.get("device_class") != here:
                raise PorosConfigError(
                    f"{ckpt} was written on another device class. Resume on the same "
                    "device class; the adapter exports anywhere.",
                    remediation="Start a new run here, or resume where this run was written.",
                )
            if "torch_rng" in rng_state:
                torch.set_rng_state(rng_state["torch_rng"].to(torch.uint8).cpu())
            if (
                "cuda_rng" in rng_state
                and self.device.type == "cuda"
                and torch.cuda.is_available()
            ):
                torch.cuda.set_rng_state(
                    rng_state["cuda_rng"].to(torch.uint8).cpu(), self.device
                )
            if self.device.type == "mps":
                if "mps_rng" not in rng_state:
                    raise PorosConfigError(
                        f"{ckpt} has no MPS generator state, so a resumed run's dropout would "
                        "differ from an uninterrupted one's.",
                        remediation="Resume from a complete checkpoint, or start a new run.",
                    )
                torch.mps.set_rng_state(rng_state["mps_rng"].to(torch.uint8).cpu())
            self.global_step = int(rng_state.get("global_step", self.global_step))

        if load_optimizer and load_rng:  # a resume continues the run the checkpoint was
            self._run_id, self._run_parent = checkpoints.lineage_of(
                ckpt, Path(self.config.output_dir), self._run_id)
        writer = getattr(self, "_event_writer", None)
        if load_optimizer and load_rng and writer is not None:
            # events.jsonl stays append-only (a tail keeps its place): the marker supersedes the
            # steps the interrupted run wrote past its checkpoint, which this run takes again
            from poros.core.events import PorosEvent

            writer.emit(PorosEvent(event_type="resume", step=self.global_step + 1,
                                   data={"checkpoint": str(ckpt),
                                         **({"unverified": unverified} if unverified else {})}))
        logger.info(
            "Restored checkpoint from %s (resuming at step %d)",
            checkpoint_dir, self.global_step,
        )

    def _own_checkpoints(self) -> list:
        """This run's complete checkpoints, (step, folder) ascending: those stamped with its id,
        and where it resumed from an unstamped one, the unstamped ones too (its earlier ones)."""
        return checkpoints.list_checkpoints(
            Path(self.config.output_dir), run=self._run_id,
            with_unstamped=self._run_parent is not None)

    def _prune_checkpoints(self, room_for_one: bool = False) -> None:
        """Keep only the newest ``save_total_limit`` ``checkpoint-*`` dirs, one
        fewer with ``room_for_one`` but never less than the newest.

        A non-positive limit disables pruning (keep everything).
        """
        limit = self.config.save_total_limit
        if limit is None or limit <= 0:
            return
        out = Path(self.config.output_dir)
        if not out.exists():
            return
        for stale in out.glob("checkpoint-*.old"):
            shutil.rmtree(stale, ignore_errors=True)
        # this run's alone: another's are not ours to delete, and an unstamped one (an older
        # Poros's) is never deleted, not even the one this run resumed from
        ckpts = [
            d for d in out.glob("checkpoint-*")
            if checkpoints.checkpoint_step(d) is not None
            and checkpoints.run_of(d) == self._run_id
        ]
        keep = max(limit - 1, 1) if room_for_one else limit
        if len(ckpts) <= keep:
            return
        ckpts.sort(key=lambda d: int(d.name.split("-")[-1]))
        for stale in ckpts[:-keep]:
            shutil.rmtree(stale, ignore_errors=True)
            logger.info("Pruned old checkpoint %s (save_total_limit=%d)", stale, limit)

    def _save_tokenizer(self, out_path: Path) -> None:
        """Save the base tokenizer into a checkpoint dir (best-effort).

        Makes the saved directory self-contained for inference and `poros
        export`. A tokenizer that cannot be loaded (e.g. offline) is skipped
        with a warning rather than failing the whole save.
        """
        if self._tokenizer is not None:
            try:
                self._tokenizer.save_pretrained(str(out_path))
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not save cached tokenizer: %s", exc)
        from poros.core.tokenizer import TokenizerRefused, load_tokenizer

        try:
            tok = load_tokenizer(
                self.config.model_name_or_path,
                revision=self.config.model_revision,
                trust_remote_code=self.config.trust_remote_code,
            )
            tok.save_pretrained(str(out_path))
        except TokenizerRefused:  # no tokenizer that could not be had: the run says so, and stops
            raise
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not save tokenizer for %s: %s (load it from the base "
                "model when running inference).",
                self.config.model_name_or_path, exc,
            )


def _push_to_hub(output_dir: Path, hub_model_id: Optional[str]) -> None:
    """Push adapter weights to HuggingFace Hub."""
    if hub_model_id is None:
        # The push was explicitly requested; publishing NOTHING while the
        # process exits 0 reads as success.
        raise ValueError(
            "push_to_hub=True but no hub_model_id was provided. Set "
            "hub_model_id='org/name' (config or save argument)."
        )
    manifest_src = Path(output_dir) / "poros_config.json"
    if manifest_src.exists():
        try:
            base = json.loads(manifest_src.read_text()).get("model_name_or_path", "")
        except (OSError, ValueError):
            base = ""
        # A machine-local base path would ship an adapter that cannot
        # resolve its base anywhere else -- and leaks the local path. Only
        # UNAMBIGUOUSLY local forms count: absolute / ./ / ~ paths, or a
        # directory that actually holds a model config. A bare `org/name`
        # that merely collides with a same-named CWD directory (e.g. an
        # `hf download --local-dir Org/Name` layout) must not block a push
        # of a genuinely portable Hub id.
        looks_local = base and (
            Path(base).is_absolute()
            or base.startswith(("./", "../", "~"))
            or (Path(base) / "config.json").exists()
        )
        if looks_local:
            raise ValueError(
                f"push_to_hub: this adapter's base model is a local path "
                f"({base}), which no downloader can resolve. Retrain or "
                "re-save with a portable Hub id as model_name_or_path "
                "before pushing."
            )
    # The weights the active backend wrote (PEFT's adapter_model.*, the native backend's
    # adapter_weights.pt for a CUDA MoE run): none in the folder, and nothing is uploaded.
    weights = [n for n in ("adapter_model.safetensors", "adapter_model.bin", "adapter_weights.pt")
               if (Path(output_dir) / n).is_file()]
    if not weights:
        raise ValueError(
            f"push_to_hub: {output_dir} holds no adapter weights (adapter_model.safetensors, "
            "adapter_model.bin or adapter_weights.pt); nothing was uploaded."
        )
    try:
        import io

        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=hub_model_id,
            repo_type="model",
            # Portable adapter + tokenizer files ONLY. The checkpoint dir also
            # holds private run state (optimizer/RNG/scheduler tensors) and
            # the full poros_config.json, which records machine-local paths
            # and a trust_remote_code setting no downloaded artifact may
            # carry. A SANITIZED manifest is uploaded separately below so
            # `poros export` still works on a downloaded adapter.
            allow_patterns=[
                *weights,
                "adapter_config.json", "README.md",
                "tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                "special_tokens_map.json", "vocab.*", "merges.txt",
                "added_tokens.json", "chat_template*",
            ],
        )
        from poros.core.hub_files import portable_manifest

        manifest = portable_manifest(output_dir)
        if manifest is not None:
            api.upload_file(
                path_or_fileobj=io.BytesIO(manifest),
                path_in_repo="poros_config.json",
                repo_id=hub_model_id,
                repo_type="model",
            )
        logger.info("Pushed adapter to HF Hub: %s", hub_model_id)
    except ImportError as exc:
        # The push was explicitly requested; a quiet log line that lets the
        # process exit 0 reads as success. Fail loudly instead.
        raise RuntimeError(
            "push_to_hub=True but huggingface_hub is not installed. "
            "Install with: pip install huggingface-hub"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Failed to push adapter to HF Hub ({hub_model_id}): {type(exc).__name__}. "
            "The local save is complete; fix the Hub issue and push again."
        ) from exc
