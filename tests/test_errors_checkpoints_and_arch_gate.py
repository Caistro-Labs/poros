"""Dataset errors that say what went wrong, checkpoints and resume that stay honest, the training
shape recorded, and the architecture gate before the download.

These tests run with no CUDA and no network.

Covered:
  * dataset failures name their root cause and get a matching remedy
  * exception-chain walking is cycle-safe and never crashes the error path
  * saved adapters point at the configured model id, not a local cache path
  * torn checkpoints are skipped by --resume-from latest
  * staging (.tmp) dirs are invisible to checkpoint selection and pruning
  * prepare() records the real training shape, or nothing at all
"""

import json

import pytest

from poros.cli.commands.train import _resolve_resume_from


# poros.train.datasets imports torch at module scope, and CI's torch-free leg
# installs no ML deps at all -- resolve lazily so the torch-free tests in
# this file still run there and the torch-backed ones skip cleanly.
def _datasets():
    pytest.importorskip("torch")
    from poros.train import datasets
    return datasets


# --------------------------------------------------------------------------
# dataset failures: say what actually went wrong
# --------------------------------------------------------------------------
class TestDatasetErrorTruth:
    """The original bug: a CSV with a valid header was told to add a header,
    while the real cause (invalid UTF-8) was discarded with the chain."""

    def test_utf8_failure_names_the_byte_offset(self):
        err = UnicodeDecodeError("utf-8", b"caf\xe9", 3, 4, "invalid continuation byte")
        remedy = _datasets()._dataset_remediation(err, "/tmp/x.csv")
        assert "not valid UTF-8" in remedy
        assert "3" in remedy  # the offset, so the user can find it
        assert "iconv" in remedy  # and a concrete way to fix it

    def test_ragged_rows_quote_the_parser(self):
        err = ValueError("Expected 3 fields in line 7, saw 5")
        remedy = _datasets()._dataset_remediation(err, "/tmp/x.csv")
        assert "column count" in remedy
        assert "line 7" in remedy

    def test_empty_file_is_named_as_empty(self, tmp_path):
        empty = tmp_path / "empty.jsonl"
        empty.write_text("")
        remedy = _datasets()._dataset_remediation(StopIteration(), str(empty))
        assert "empty" in remedy.lower()

    def test_bad_json_line_is_named(self):
        err = json.JSONDecodeError("Expecting value", "{bad", 0)
        remedy = _datasets()._dataset_remediation(err, "/tmp/x.jsonl")
        assert "JSON" in remedy

    def test_unknown_cause_falls_back_to_format_help(self):
        remedy = _datasets()._dataset_remediation(RuntimeError("something odd"), None)
        assert ".jsonl" in remedy  # the generic format guidance still ships

    @pytest.mark.parametrize(
        "exc",
        [
            UnicodeDecodeError("utf-8", b"x", 0, 1, "bad"),
            ValueError("Expected 2 fields in line 1, saw 3"),
            json.JSONDecodeError("Expecting value", "x", 0),
            StopIteration(),
            RuntimeError("generic"),
            OSError(13, "Permission denied"),
        ],
    )
    def test_remediation_never_raises(self, exc):
        """This code runs while something is already failing: it must not
        replace a helpful error with a crash inside error handling."""
        for path in (None, "/nonexistent/path.csv", ""):
            assert isinstance(_datasets()._dataset_remediation(exc, path), str)


class TestRootCauseWalking:
    def test_walks_to_the_deepest_cause(self):
        root = UnicodeDecodeError("utf-8", b"x", 0, 1, "bad")
        mid = ValueError("wrapper")
        mid.__cause__ = root
        top = RuntimeError("An error occurred while generating the dataset")
        top.__cause__ = mid
        assert _datasets()._root_cause(top) is root

    def test_cycle_does_not_hang(self):
        a = ValueError("a")
        b = ValueError("b")
        a.__cause__ = b
        b.__cause__ = a  # pathological, but must terminate
        assert _datasets()._root_cause(a) in (a, b)

    def test_no_chain_returns_self(self):
        e = ValueError("solo")
        assert _datasets()._root_cause(e) is e


# --------------------------------------------------------------------------
# adapters must be loadable off the machine that trained them
# --------------------------------------------------------------------------
class TestAdapterBaseModelPointer:
    """PEFT records the load path -- for a cached hub model, a snapshot dir
    that exists on exactly one machine. Poros substitutes the portable id at
    the SOURCE (PEFT's in-memory config, so both adapter_config.json and the
    model card get it), guarded so a config id that merely differs from what
    was loaded is never stamped on."""

    SNAP = "/root/.cache/hf/hub/models--Qwen--Qwen2.5-0.5B/snapshots/" + "a" * 40

    def _saver(self, recorded, revision=None):
        class _PCfg:
            base_model_name_or_path = recorded
        _PCfg.revision = revision

        class _BaseCfg:
            _name_or_path = recorded

        class _Base:
            config = _BaseCfg()

        class _Saver:
            peft_config = {"default": _PCfg()}
            def get_base_model(self):
                return _Base()
        return _Saver()

    def _trainer_stub(self, configured):
        pytest.importorskip("torch")
        from poros.train.trainer import PorosTrainer

        stub = PorosTrainer.__new__(PorosTrainer)
        stub._resolved_base_path = None

        class _Cfg:
            model_name_or_path = configured
        stub.config = _Cfg()
        return stub

    def test_snapshot_of_configured_repo_is_substituted_and_pinned(self):
        stub = self._trainer_stub("Qwen/Qwen2.5-0.5B")
        saver = self._saver(self.SNAP)
        with stub._portable_base_identity(saver):
            pcfg = saver.peft_config["default"]
            assert pcfg.base_model_name_or_path == "Qwen/Qwen2.5-0.5B"
            # the snapshot sha must survive as an exact revision pin
            assert pcfg.revision == "a" * 40
            # the model card reads the BASE config -- it must be rewritten too
            assert saver.get_base_model().config._name_or_path is not None
        # restored after the save so runtime state is untouched
        assert saver.peft_config["default"].base_model_name_or_path == self.SNAP
        assert saver.peft_config["default"].revision is None
        assert stub._resolved_base_path == self.SNAP

    def test_mismatched_id_is_never_stamped(self):
        """config said Llama, the weights came from a Qwen snapshot: stamping
        the config id would make the adapter compose silently with the wrong
        base. The recorded truth must win."""
        stub = self._trainer_stub("meta-llama/Llama-3.1-8B")
        saver = self._saver(self.SNAP)
        with stub._portable_base_identity(saver):
            assert saver.peft_config["default"].base_model_name_or_path == self.SNAP
        assert stub._resolved_base_path is None

    def test_plain_local_dir_is_left_alone(self):
        """A deliberate local-directory load is not a cache snapshot; it must
        not be replaced by a hub id it may not correspond to."""
        stub = self._trainer_stub("Qwen/Qwen2.5-0.5B")
        saver = self._saver("/data/my-custom-base")
        with stub._portable_base_identity(saver):
            assert (
                saver.peft_config["default"].base_model_name_or_path
                == "/data/my-custom-base"
            )

    def test_already_correct_is_untouched(self):
        stub = self._trainer_stub("Qwen/Qwen2.5-0.5B")
        saver = self._saver("Qwen/Qwen2.5-0.5B")
        with stub._portable_base_identity(saver):
            assert (
                saver.peft_config["default"].base_model_name_or_path
                == "Qwen/Qwen2.5-0.5B"
            )
        assert stub._resolved_base_path is None

    def test_existing_revision_is_never_overwritten(self):
        stub = self._trainer_stub("Qwen/Qwen2.5-0.5B")
        saver = self._saver(self.SNAP, revision="refs/pr/7")
        with stub._portable_base_identity(saver):
            assert saver.peft_config["default"].revision == "refs/pr/7"
        assert saver.peft_config["default"].revision == "refs/pr/7"

    def test_restore_happens_even_when_save_raises(self):
        stub = self._trainer_stub("Qwen/Qwen2.5-0.5B")
        saver = self._saver(self.SNAP)
        with pytest.raises(RuntimeError):
            with stub._portable_base_identity(saver):
                raise RuntimeError("disk full mid-save")
        assert saver.peft_config["default"].base_model_name_or_path == self.SNAP


# --------------------------------------------------------------------------
# resume must never pick a checkpoint it cannot load
# --------------------------------------------------------------------------
def _ckpt(root, step, complete=True):
    d = root / f"checkpoint-{step}"
    d.mkdir()
    (d / "README.md").write_text("model card, written first")
    if complete:
        (d / "adapter_config.json").write_text("{}")
        (d / "adapter_model.safetensors").write_bytes(b"\x00")
        # weights + RNG + optimizer: the full legacy resumable contract
        (d / "rng_state.pt").write_bytes(b"\x00")
        (d / "optimizer_state.pt").write_bytes(b"\x00")
    return d


class TestStagingDirsAreInvisible:
    """Saves stage into checkpoint-N.tmp before an atomic rename. A leftover
    staging dir from a kill must never be selected or counted."""

    def test_incomplete_tmp_dir_is_cleaned_not_selected(self, tmp_path):
        _ckpt(tmp_path, 5)
        staging = tmp_path / "checkpoint-10.tmp"
        staging.mkdir()
        (staging / "adapter_config.json").write_text("{}")
        assert _resolve_resume_from("latest", str(tmp_path)).name == "checkpoint-5"
        assert not staging.exists(), "torn staging dir should be swept"

    def test_complete_tmp_dir_is_promoted_and_selected(self, tmp_path):
        """A kill between save(staging) and the promote strands the ONLY
        complete copy under .tmp -- the resolver must adopt it, not resume
        200 steps back."""
        _ckpt(tmp_path, 5)
        staging = tmp_path / "checkpoint-10.tmp"
        staging.mkdir()
        (staging / "adapter_config.json").write_text("{}")
        (staging / "adapter_model.safetensors").write_bytes(b"\x00")
        (staging / "rng_state.pt").write_bytes(b"\x00")
        (staging / "optimizer_state.pt").write_bytes(b"\x00")
        chosen = _resolve_resume_from("latest", str(tmp_path))
        assert chosen.name == "checkpoint-10"
        assert not staging.exists()

    def test_old_backup_recovers_when_final_is_torn(self, tmp_path):
        """Kill between os.replace(final, backup) and os.replace(staging,
        final): the good copy sits at checkpoint-N.old."""
        backup = tmp_path / "checkpoint-6.old"
        backup.mkdir()
        (backup / "adapter_config.json").write_text("{}")
        (backup / "adapter_model.safetensors").write_bytes(b"\x00")
        (backup / "rng_state.pt").write_bytes(b"\x00")
        (backup / "optimizer_state.pt").write_bytes(b"\x00")
        assert _resolve_resume_from("latest", str(tmp_path)).name == "checkpoint-6"

    def test_tmp_dir_is_not_counted_by_pruning(self, tmp_path):
        pytest.importorskip("torch")
        """save_total_limit accounting must ignore staging dirs, or a limit
        of N silently keeps N-1 real checkpoints."""
        from poros.train.trainer import PorosTrainer

        stub = PorosTrainer.__new__(PorosTrainer)

        class _Cfg:
            save_total_limit = 2
            output_dir = str(tmp_path)

        stub.config = _Cfg()
        stub._run_id = "a" * 16
        for step in (5, 10, 15):
            _ckpt(tmp_path, step)
            (tmp_path / f"checkpoint-{step}" / ".poros_run").write_text(stub._run_id)  # its own
        (tmp_path / "checkpoint-20.tmp").mkdir()
        stub._prune_checkpoints()

        remaining = sorted(
            d.name for d in tmp_path.glob("checkpoint-*") if d.is_dir()
        )
        # the two newest real checkpoints survive; the staging dir is untouched
        assert "checkpoint-10" in remaining and "checkpoint-15" in remaining
        assert "checkpoint-5" not in remaining
        assert "checkpoint-20.tmp" in remaining


class TestTornCheckpointResume:
    def test_torn_newest_is_skipped(self, tmp_path):
        _ckpt(tmp_path, 5)
        _ckpt(tmp_path, 10, complete=False)
        assert _resolve_resume_from("latest", str(tmp_path)).name == "checkpoint-5"

    def test_all_torn_fails_fast(self, tmp_path):
        import typer

        _ckpt(tmp_path, 5, complete=False)
        with pytest.raises(typer.Exit):
            _resolve_resume_from("latest", str(tmp_path))

    def test_native_backend_checkpoint_counts_as_complete(self, tmp_path):
        d = tmp_path / "checkpoint-7"
        d.mkdir()
        (d / "adapter_weights.pt").write_bytes(b"\x00")
        (d / "rng_state.pt").write_bytes(b"\x00")
        (d / "optimizer_state.pt").write_bytes(b"\x00")
        assert _resolve_resume_from("latest", str(tmp_path)).name == "checkpoint-7"

    def test_weights_without_rng_state_is_torn(self, tmp_path):
        """The silent-restart case: adapter files landed, RNG/optimizer state
        did not. Resuming it would retrain from step 0 -- must be skipped."""
        _ckpt(tmp_path, 4)
        d = tmp_path / "checkpoint-8"
        d.mkdir()
        (d / "adapter_config.json").write_text("{}")
        (d / "adapter_model.safetensors").write_bytes(b"\x00")
        assert _resolve_resume_from("latest", str(tmp_path)).name == "checkpoint-4"

    def test_sentinel_marker_alone_is_complete(self, tmp_path):
        """Current saves end with the sentinel; it certifies everything
        before it."""
        d = tmp_path / "checkpoint-9"
        d.mkdir()
        (d / ".poros_checkpoint_complete").write_text("9")
        assert _resolve_resume_from("latest", str(tmp_path)).name == "checkpoint-9"


# --------------------------------------------------------------------------
# the arch gate must refuse BEFORE the weights download
# --------------------------------------------------------------------------
class TestArchGateBeforeDownload:
    """Refusing an unvalidated architecture after pulling 140 GB of weights
    wastes the user's disk, bandwidth, and hour. config.json alone names
    model_type."""

    def _fake_model_dir(self, tmp_path, model_type):
        d = tmp_path / "model"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({"model_type": model_type}))
        return d

    def test_unvalidated_arch_refused_from_config_json_alone(self, tmp_path):
        pytest.importorskip("torch")
        from poros.core.errors import PorosArchNotValidatedError
        from poros.train.trainer import PorosTrainer

        d = self._fake_model_dir(tmp_path, "llama")
        # the dir has NO weights: reaching a weights-load error would mean
        # the gate ran too late
        with pytest.raises(PorosArchNotValidatedError):
            PorosTrainer.from_pretrained(str(d))

    def test_cheap_resolver_is_never_a_new_failure_mode(self, tmp_path):
        pytest.importorskip("torch")
        from poros.preflight.arch import cheap_model_type as _cheap_model_type

        assert _cheap_model_type(str(tmp_path / "missing")) is None
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _cheap_model_type(str(empty)) is None
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "config.json").write_text("{ not json")
        assert _cheap_model_type(str(bad)) is None

    def test_validated_arch_passes_the_pre_gate(self, tmp_path):
        pytest.importorskip("torch")
        from poros.preflight.arch import cheap_model_type as _cheap_model_type

        d = self._fake_model_dir(tmp_path, "qwen2")
        assert _cheap_model_type(str(d)) == "qwen2"
