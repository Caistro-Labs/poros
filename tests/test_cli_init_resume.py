"""`poros init` and `--resume-from latest` -- the install->init->run path.

Both surfaces must work with NO torch installed (they run before any
training), so these tests import only the CLI module and core config.
"""

import pytest
import typer
import yaml

from poros.cli.commands.train import _resolve_resume_from, _starter_config_yaml, init
from poros.core.config import PorosConfig


class TestStarterConfig:
    def test_roundtrips_through_porosconfig(self):
        text = _starter_config_yaml(
            "Qwen/Qwen2.5-7B", "./my_data.jsonl", "./my-adapter"
        )
        cfg = PorosConfig(**yaml.safe_load(text))
        assert cfg.model_name_or_path == "Qwen/Qwen2.5-7B"
        assert cfg.dataset == "./my_data.jsonl"
        assert cfg.output_dir == "./my-adapter"

    def test_only_three_active_fields(self):
        # The whole point: nothing beyond the required trio is uncommented.
        text = _starter_config_yaml("m", "d", "o")
        assert set(yaml.safe_load(text)) == {
            "model_name_or_path", "dataset", "output_dir",
        }

    def test_hostile_values_cannot_break_the_yaml(self):
        text = _starter_config_yaml(
            'weird"model\nname', "data: [a, b]", "out'dir",
        )
        parsed = yaml.safe_load(text)
        assert parsed["model_name_or_path"] == 'weird"model\nname'
        assert parsed["dataset"] == "data: [a, b]"
        assert parsed["output_dir"] == "out'dir"


class TestInitCommand:
    def test_writes_config(self, tmp_path):
        target = tmp_path / "train.yaml"
        init(path=target, model="m", dataset="synthetic", output="./o",
             force=False)
        cfg = PorosConfig(**yaml.safe_load(target.read_text()))
        assert cfg.dataset == "synthetic"

    def test_refuses_overwrite_without_force(self, tmp_path):
        target = tmp_path / "train.yaml"
        target.write_text("precious: true\n")
        with pytest.raises(typer.Exit):
            init(path=target, model="m", dataset="d", output="o",
                 force=False)
        assert target.read_text() == "precious: true\n"

    def test_force_overwrites(self, tmp_path):
        target = tmp_path / "train.yaml"
        target.write_text("old: true\n")
        init(path=target, model="m", dataset="synthetic", output="o",
             force=True)
        assert "model_name_or_path" in target.read_text()


def _mk_ckpt(root, step, complete=True, marker=False):
    """Create checkpoint-<step>.

    complete=True writes the LEGACY complete layout (weights + rng_state.pt,
    pre-sentinel Poros versions); marker=True writes the sentinel a current
    save() ends with. complete=False simulates a kill mid-save -- note it
    still has adapter files: a dir torn AFTER the weights but BEFORE the
    RNG/optimizer state must count as torn, or resume silently restarts
    from step 0 with fresh optimizer moments.
    """
    d = root / f"checkpoint-{step}"
    d.mkdir()
    (d / "README.md").write_text("model card, written first")
    if complete or marker:
        (d / "adapter_config.json").write_text("{}")
        (d / "adapter_model.safetensors").write_bytes(b"\x00")
    if complete:
        (d / "rng_state.pt").write_bytes(b"\x00")
        (d / "optimizer_state.pt").write_bytes(b"\x00")
    if marker:
        (d / ".poros_checkpoint_complete").write_text(str(step))
    return d


class TestResumeFromResolution:
    def test_latest_picks_highest_step(self, tmp_path):
        for step in (2, 10, 4):
            _mk_ckpt(tmp_path, step)
        resolved = _resolve_resume_from("latest", str(tmp_path))
        assert resolved.name == "checkpoint-10"

    def test_auto_is_an_alias(self, tmp_path):
        _mk_ckpt(tmp_path, 3)
        assert _resolve_resume_from("auto", str(tmp_path)).name == "checkpoint-3"

    def test_latest_with_no_checkpoints_fails_fast(self, tmp_path):
        with pytest.raises(typer.Exit):
            _resolve_resume_from("latest", str(tmp_path))

    def test_explicit_checkpoint_passes_through(self, tmp_path):
        ckpt = tmp_path / "checkpoint-5"
        ckpt.mkdir()
        (ckpt / "adapter_config.json").write_text("{}")
        assert _resolve_resume_from(str(ckpt), str(tmp_path)) == ckpt

    def test_run_dir_descends_to_newest_checkpoint(self, tmp_path):
        _mk_ckpt(tmp_path, 1)
        _mk_ckpt(tmp_path, 12)
        resolved = _resolve_resume_from(str(tmp_path), "ignored-output-dir")
        assert resolved.name == "checkpoint-12"

    def test_torn_checkpoint_skipped_for_intact_earlier_one(self, tmp_path):
        """A kill mid-save must not strand the run: `latest` skips the torn
        newest checkpoint and resumes from the last complete one."""
        _mk_ckpt(tmp_path, 5)
        _mk_ckpt(tmp_path, 10, complete=False)
        resolved = _resolve_resume_from("latest", str(tmp_path))
        assert resolved.name == "checkpoint-5"

    def test_all_checkpoints_torn_fails_fast(self, tmp_path):
        _mk_ckpt(tmp_path, 5, complete=False)
        with pytest.raises(typer.Exit):
            _resolve_resume_from("latest", str(tmp_path))

    def test_missing_path_fails_fast(self, tmp_path):
        with pytest.raises(typer.Exit):
            _resolve_resume_from(str(tmp_path / "nope"), str(tmp_path))

    def test_non_numeric_checkpoint_dirs_ignored(self, tmp_path):
        (tmp_path / "checkpoint-final").mkdir()
        _mk_ckpt(tmp_path, 7)
        resolved = _resolve_resume_from("latest", str(tmp_path))
        assert resolved.name == "checkpoint-7"


class TestConfigStrictness:
    def test_unknown_field_rejected(self):
        import pydantic

        with pytest.raises(pydantic.ValidationError) as exc:
            PorosConfig(per_device_train_batch_size=16)
        assert "per_device_train_batch_size" in str(exc.value)

    def test_known_fields_still_fine(self):
        assert PorosConfig(per_device_batch_size=16).per_device_batch_size == 16


class TestExitCodeContract:
    """Exit codes are agent-facing API; the cli() wrapper must preserve them."""

    def _run(self, *args, cwd=None):
        import os
        import subprocess
        import sys

        # Strip ambient color-forcing (CI/agent shells export these): rich
        # honors FORCE_COLOR even on pipes, and styling between words breaks
        # the plain-substring assertions below.
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("FORCE_COLOR", "CLICOLOR_FORCE")
        }
        return subprocess.run(
            [sys.executable, "-m", "poros.cli.main", *args],
            capture_output=True, text=True, cwd=cwd, env=env,
        )

    def test_help_exits_zero(self):
        assert self._run("--help").returncode == 0

    def test_missing_argument_exits_two(self):
        r = self._run("report")
        assert r.returncode == 2
        assert "needs RUN_DIR" in r.stdout + r.stderr

    def test_config_error_exits_two(self, tmp_path):
        cfg = tmp_path / "t.yaml"
        cfg.write_text(
            "model_name_or_path: x\ndataset: synthetic\nlora_r: 8\n"
        )
        r = self._run("train", str(cfg))
        assert r.returncode == 2
        assert "did you mean" in r.stdout + r.stderr
