"""`poros eval` must measure the TRAINED adapter.

It loads the trained adapter before evaluating, never the freshly-initialized
LoRA weights `from_pretrained` attaches to the base model.

These are torch-free: the CLI function is called directly and PorosTrainer is
stubbed, so the tests assert the *contract* (which directory is resolved, and
that it is loaded before evaluating) without a GPU or a model download.
"""

import json
import sys
import types

import pytest
import typer

from poros.cli import main as cli
from poros.cli import root as cli_root
from poros.cli.commands import export as cli_commands_export
from poros.core import checkpoints
from poros.preflight import arch as preflight_arch
from tests.conftest import as_linux


def _adapter(root, name="adapter", *, weights=True, rng=False, manifest=None):
    """A PEFT-shaped adapter dir. `rng` promotes it to a resumable checkpoint."""
    d = root / name
    d.mkdir(parents=True)
    if weights:
        (d / "adapter_config.json").write_text("{}")
        (d / "adapter_model.safetensors").write_bytes(b"\x00")
    if rng:
        (d / "rng_state.pt").write_bytes(b"\x00")
        (d / "optimizer_state.pt").write_bytes(b"\x00")
    if manifest is not None:
        (d / "poros_config.json").write_text(json.dumps(manifest))
    return d


def _config(root, output_dir, **overrides):
    cfg = {
        "model_name_or_path": "Qwen/Qwen2.5-0.5B",
        "dataset": "synthetic",
        "output_dir": str(output_dir),
        **overrides,
    }
    p = root / "eval.yaml"
    p.write_text(json.dumps(cfg))  # YAML is a JSON superset
    return p


@pytest.fixture
def stub_trainer(monkeypatch):
    """Record what the command loads and evaluates, without torch."""
    calls = {"loaded": [], "evaluated": []}

    class _Stub:
        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

        def load_checkpoint(self, path, **kw):
            calls["loaded"].append((path, kw))

        def evaluate(self, dataset):
            calls["evaluated"].append(dataset)
            return {"eval_loss": 1.23}

    # Inject a fake module rather than monkeypatching the real one: importing
    # poros.train.trainer pulls in torch, and these tests exercise CLI wiring,
    # not ML. Without this they ERROR (not skip) on the torch-free CI lane --
    # which is exactly where they are most valuable.
    fake = types.ModuleType("poros.train.trainer")
    fake.PorosTrainer = _Stub
    monkeypatch.setitem(sys.modules, "poros.train.trainer", fake)
    monkeypatch.setattr(cli_root, "_require_torch", lambda *a, **kw: None)
    # a config file's eval as a CUDA host runs it: a Mac refuses that form (docs/mac.md)
    as_linux(monkeypatch, cli_root)
    return calls


class TestEvalLoadsTheTrainedAdapter:
    def test_adapter_is_loaded_before_evaluating(self, tmp_path, stub_trainer):
        """The adapter is loaded before evaluation."""
        adapter = _adapter(tmp_path)
        cli.evaluate(config=_config(tmp_path, adapter), adapter=None)

        assert stub_trainer["loaded"], "eval evaluated without loading an adapter"
        path, kwargs = stub_trainer["loaded"][0]
        assert path == str(adapter)
        # weights only: optimizer/RNG belong to resuming, not measuring
        assert kwargs == {"load_optimizer": False, "load_rng": False}
        assert stub_trainer["evaluated"] == ["synthetic"]

    def test_explicit_adapter_flag_wins_over_output_dir(self, tmp_path, stub_trainer):
        _adapter(tmp_path, "from_config")
        chosen = _adapter(tmp_path, "explicit")
        cli.evaluate(
            config=_config(tmp_path, tmp_path / "from_config"), adapter=chosen
        )
        assert stub_trainer["loaded"][0][0] == str(chosen)


class TestAdapterResolution:
    def test_plain_peft_directory_is_accepted(self, tmp_path):
        """Weights without rng_state.pt is a normal `save_pretrained()` dir
        and a normal Hub download -- loadable, though not resumable. Gating
        eval on the resume contract rejected every one of them."""
        d = _adapter(tmp_path, rng=False)
        assert not checkpoints.is_complete(d)  # correctly not resumable
        assert cli_commands_export._resolve_adapter_dir(d) == d  # but still evaluable

    def test_run_dir_falls_back_to_newest_checkpoint(self, tmp_path):
        run = tmp_path / "run"
        run.mkdir()
        _adapter(run, "checkpoint-5", rng=True)
        newest = _adapter(run, "checkpoint-10", rng=True)
        assert cli_commands_export._resolve_adapter_dir(run) == newest

    def test_stranded_checkpoint_is_recovered_like_resume(self, tmp_path):
        """A kill inside the save promote window leaves the only complete
        copy under checkpoint-N.tmp. `--resume-from latest` adopts it; eval
        must agree, or the two commands disagree about which weights are
        newest."""
        run = tmp_path / "run"
        run.mkdir()
        _adapter(run, "checkpoint-5", rng=True)
        _adapter(run, "checkpoint-20.tmp", rng=True)
        assert cli_commands_export._resolve_adapter_dir(run).name == "checkpoint-20"

    def test_missing_directory_fails_fast(self, tmp_path):
        with pytest.raises(typer.Exit):
            cli_commands_export._resolve_adapter_dir(tmp_path / "nope")

    def test_directory_without_weights_fails_fast(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(typer.Exit):
            cli_commands_export._resolve_adapter_dir(empty)


class TestConfigMismatchWarning:
    def _cfg(self, **kw):
        from poros.core.config import PorosConfig

        return PorosConfig(**kw)

    def test_different_base_model_warns(self, tmp_path, capsys):
        d = _adapter(
            tmp_path, manifest={"model_name_or_path": "meta-llama/Llama-3.1-8B"}
        )
        cli_commands_export._warn_on_adapter_config_mismatch(
            d, self._cfg(model_name_or_path="Qwen/Qwen2.5-0.5B")
        )
        assert "trained with" in capsys.readouterr().out

    def test_matching_config_is_silent(self, tmp_path, capsys):
        d = _adapter(tmp_path, manifest={"model_name_or_path": "Qwen/Qwen2.5-0.5B"})
        cli_commands_export._warn_on_adapter_config_mismatch(
            d, self._cfg(model_name_or_path="Qwen/Qwen2.5-0.5B")
        )
        assert capsys.readouterr().out == ""

    def test_absent_or_unreadable_manifest_is_silent(self, tmp_path, capsys):
        d = _adapter(tmp_path)  # no poros_config.json
        cli_commands_export._warn_on_adapter_config_mismatch(d, self._cfg())
        (d / "poros_config.json").write_text("{ not json")
        cli_commands_export._warn_on_adapter_config_mismatch(d, self._cfg())
        assert capsys.readouterr().out == ""


class TestSecurityDefault:
    def test_trust_remote_code_is_off_by_default(self):
        """A typo'd or compromised model id must not be able to execute the
        repo's own Python without the user asking for it."""
        from poros.core.config import PorosConfig

        assert PorosConfig().trust_remote_code is False


class TestTheCommandIsWiredToTheRightFunction:
    """The tests above call cli.evaluate() directly, which proves the FUNCTION
    works and says nothing about whether `poros eval` reaches it (a misplaced
    @app.command decorator would bind the command to another function).
    Invoke through Typer, the way a user does."""

    def test_eval_help_describes_evaluation_not_a_helper(self):
        from typer.testing import CliRunner

        from poros.cli.main import app

        res = CliRunner().invoke(app, ["eval", "--help"])
        assert res.exit_code == 0, res.output
        # Assert on the command's own summary and option -- NOT on typer's
        # metavar rendering, which differs across typer versions (0.26 emits
        # CONFIG, 0.27 emits {config}) and would make this test fail for
        # contributors rather than for the bug it guards.
        assert "Evaluate a trained adapter" in res.output
        assert "--adapter" in res.output
        # the helper's docstring must not be the command's help text
        assert "Gated on" not in res.output

    def test_invoking_the_command_loads_then_evaluates(self, tmp_path, stub_trainer):
        from typer.testing import CliRunner

        from poros.cli.main import app

        adapter = _adapter(tmp_path)
        cfg = _config(tmp_path, adapter)
        res = CliRunner().invoke(app, ["eval", str(cfg)])
        assert res.exit_code == 0, res.output
        assert stub_trainer["loaded"], "the COMMAND never loaded an adapter"
        assert stub_trainer["evaluated"] == ["synthetic"]

    def test_every_documented_command_is_registered(self):
        """A misplaced decorator silently rebinds a command name, so assert the
        whole surface the README and docs promise."""
        from typer.testing import CliRunner

        from poros.cli.main import app

        out = CliRunner().invoke(app, ["--help"]).output
        for cmd in ("init", "train", "eval", "export", "report",
                    "leaderboard", "check", "schema", "bench"):
            assert cmd in out, f"{cmd} missing from the CLI surface"


class TestValidationIsNotMintable:
    """`register_architecture()` and POROS_USER_ASSERTED_ARCHS make an architecture runnable
    (user_asserted), never validated: a caller cannot promote their own architecture into
    Caistro's first-party evidence class."""

    def test_runtime_registration_is_user_asserted_not_validated(self):
        from poros.core import arch_registry as reg

        before = reg.validated_arch_types()
        reg.register_architecture("phi3_test_only")
        try:
            assert reg.validated_arch_types() == before, "validation was minted"
            assert "phi3_test_only" in reg.user_asserted_arch_types()
            assert "phi3_test_only" in reg.runnable_arch_types()
            assert reg.architectures()["phi3_test_only"].status == "user_asserted"
        finally:
            reg._ARCHITECTURES.pop("phi3_test_only", None)

    def test_env_override_asserts_but_does_not_validate(self, monkeypatch):
        from poros.core import arch_registry as reg

        monkeypatch.setenv("POROS_USER_ASSERTED_ARCHS", "mamba_test_only")
        assert "mamba_test_only" not in reg.validated_arch_types()
        assert "mamba_test_only" in reg.runnable_arch_types()
        assert reg.arch_status("mamba_test_only") == "user_asserted"

    def test_the_arch_gate_still_runs_user_asserted_archs(self):
        from poros.core import arch_registry as reg

        reg.register_architecture("phi3_gate_test")
        try:
            assert reg.blockwise_arch_check("phi3_gate_test", mode="error") is True
        finally:
            reg._ARCHITECTURES.pop("phi3_gate_test", None)


class TestMoeValidatedForItsOwnPath:
    """The MoE families are validated for poros.moe, which every entry point
    routes them to. The dense engine cannot quantize their fused experts, so
    the dense gate must keep refusing them, and say where to go instead."""

    MOE = {"qwen3_moe", "olmoe", "qwen2_moe", "deepseek_v2", "qwen3_5_moe", "qwen3_5_moe_text",
           "gemma4_moe"}

    def test_validated_for_moe_not_dense(self, monkeypatch):
        from poros.core import arch_registry as reg

        assert reg.validated_arch_types("moe") == self.MOE  # mixtral stays gated
        assert not self.MOE & reg.validated_arch_types()
        assert not self.MOE & reg.runnable_arch_types()
        monkeypatch.setenv("POROS_USER_ASSERTED_ARCHS", ",".join(sorted(self.MOE)))
        assert all(reg.arch_status(t) == "validated" for t in self.MOE)

    def test_dense_gate_points_to_the_moe_route(self):
        from poros.core import arch_registry as reg
        from poros.core.errors import PorosArchNotValidatedError

        for model_type in sorted(self.MOE):
            with pytest.raises(PorosArchNotValidatedError, match="route it to poros.moe"):
                reg.blockwise_arch_check(model_type, mode="error")

    def test_train_preflight_admits_moe(self, monkeypatch, tmp_path):
        from typer.testing import CliRunner

        from poros.cli import main as cli_main

        monkeypatch.setattr(preflight_arch, "resolve_model_type", lambda m: ("qwen3_moe", ""))
        cfg = tmp_path / "train.yaml"
        cfg.write_text('model_name_or_path: "org/moe"\ndataset: "synthetic"\n')
        res = CliRunner().invoke(cli_main.app, ["train", str(cfg)])
        assert "refusing before the model download" not in res.output

    def test_check_json_names_the_moe_entry(self, monkeypatch):
        from typer.testing import CliRunner

        from poros.cli import main as cli_main
        from poros.cli.commands import check

        as_linux(monkeypatch, check)  # the registry's answer: a Mac's says whether it has a row

        monkeypatch.setattr(preflight_arch, "resolve_model_type", lambda m: ("qwen3_moe", ""))
        res = CliRunner().invoke(cli_main.app, ["check", "org/x", "--json"])
        out = json.loads(res.output)
        assert out["status"] == "validated" and out["entry"] == "moe"
        assert res.exit_code == 0


class TestUserAssertedAtTheCliBoundary:
    """The registry split is only real if every entry point agrees: `poros
    check` reports a user-asserted arch as user_asserted, and the train
    preflight admits what the engine gate admits."""

    def test_check_reports_user_asserted_not_missing(self, monkeypatch):
        from poros.core import arch_registry as reg

        monkeypatch.setenv("POROS_USER_ASSERTED_ARCHS", "phi3_cli_test")
        assert reg.arch_status("phi3_cli_test") == "user_asserted"

    def test_builtin_validated_wins_over_env_listing(self, monkeypatch):
        """Listing a first-party arch in the env must not DOWNGRADE it."""
        from poros.core import arch_registry as reg

        builtin = sorted(reg.validated_arch_types())[0]
        monkeypatch.setenv("POROS_USER_ASSERTED_ARCHS", builtin)
        assert reg.arch_status(builtin) == "validated"

    def test_check_json_status_and_exit_code(self, monkeypatch):
        from typer.testing import CliRunner

        from poros.cli import main as cli_main
        from poros.cli.commands import check

        as_linux(monkeypatch, check)  # the registry's answer: a Mac's says whether it has a row

        monkeypatch.setenv("POROS_USER_ASSERTED_ARCHS", "phi3_json_test")
        monkeypatch.setattr(preflight_arch, "resolve_model_type", lambda m: ("phi3_json_test", "")
        )
        res = CliRunner().invoke(cli_main.app, ["check", "org/x", "--json"])
        assert json.loads(res.output)["status"] == "user_asserted"
        # non-zero (4, not validated): a script must not read this as a
        # Poros guarantee
        assert res.exit_code == 4

    def test_train_preflight_admits_what_the_engine_gate_admits(self, monkeypatch):
        """The preflight and the engine gate must agree on the runnable set."""
        from poros.core import arch_registry as reg

        monkeypatch.setenv("POROS_USER_ASSERTED_ARCHS", "phi3_preflight_test")
        assert "phi3_preflight_test" in reg.runnable_arch_types()
        assert reg.blockwise_arch_check("phi3_preflight_test", mode="error") is True
        assert "phi3_preflight_test" not in reg.validated_arch_types()
