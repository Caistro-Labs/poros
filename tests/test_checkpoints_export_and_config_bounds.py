"""Checkpoint completeness, export, reports and config bounds.

Each test covers one failure mode: a torn current-version save
promoted as complete, artifact-controlled remote-code authorization in
export, stored script injection in HTML reports, degenerate config values
crashing mid-run instead of at parse, and Hub pushes shipping private run
state.
"""

import json
import sys
import types

import pytest

from poros.core import checkpoints

# ---------------------------------------------------------------------------
# Checkpoint completeness: the sentinel is authoritative for current saves
# ---------------------------------------------------------------------------


def _ckpt(tmp_path, name, files):
    d = tmp_path / name
    d.mkdir()
    for f in files:
        (d / f).write_text("x")
    return d


_TORN_SUBSET = ["adapter_model.safetensors", "rng_state.pt", "optimizer_state.pt"]


class TestCheckpointCompleteness:
    def test_complete_marker_is_complete(self, tmp_path):
        d = _ckpt(tmp_path, "checkpoint-8",
                  [checkpoints.FORMAT_MARKER, checkpoints.COMPLETE_MARKER])
        assert checkpoints.is_complete(d)

    def test_torn_current_save_is_incomplete(self, tmp_path):
        # kill -9 between the optimizer and scheduler writes leaves exactly
        # the file subset the legacy heuristic accepts. The format marker
        # must make the missing sentinel authoritative.
        d = _ckpt(tmp_path, "checkpoint-8",
                  [checkpoints.FORMAT_MARKER] + _TORN_SUBSET)
        assert not checkpoints.is_complete(d)

    def test_legacy_dir_without_markers_uses_heuristic(self, tmp_path):
        d = _ckpt(tmp_path, "checkpoint-8", _TORN_SUBSET)
        assert checkpoints.is_complete(d)

    def test_legacy_dir_missing_optimizer_is_incomplete(self, tmp_path):
        d = _ckpt(tmp_path, "checkpoint-8",
                  ["adapter_model.safetensors", "rng_state.pt"])
        assert not checkpoints.is_complete(d)

    def test_recover_deletes_torn_current_tmp(self, tmp_path):
        # The recovery sweep must never promote a torn current-format save.
        sib = _ckpt(tmp_path, "checkpoint-8.tmp",
                    [checkpoints.FORMAT_MARKER] + _TORN_SUBSET)
        checkpoints.recover_checkpoint_dirs(tmp_path)
        assert not sib.exists()
        assert not (tmp_path / "checkpoint-8").exists()

    def test_recover_promotes_complete_tmp(self, tmp_path):
        _ckpt(tmp_path, "checkpoint-8.tmp",
              [checkpoints.FORMAT_MARKER, checkpoints.COMPLETE_MARKER])
        checkpoints.recover_checkpoint_dirs(tmp_path)
        assert (tmp_path / "checkpoint-8").exists()

    def test_a_torn_save_leftover_is_no_clash_and_a_whole_one_is(self, tmp_path):
        """A checkpoint-8.tmp a kill tore mid-save is no other run's checkpoint: a new run in
        that folder is not refused for it (recovery drops it at the run's start, and the check
        passes it by, deleting nothing); a whole one still clashes."""
        torn = _ckpt(tmp_path, "checkpoint-8.tmp", [checkpoints.FORMAT_MARKER] + _TORN_SUBSET)
        _ckpt(tmp_path, "checkpoint-3.old", [])  # killed as the rename-aside began
        assert checkpoints.fresh_run_clash(tmp_path) is None and torn.exists()
        _ckpt(tmp_path, "checkpoint-9.tmp",
              [checkpoints.FORMAT_MARKER, checkpoints.COMPLETE_MARKER])
        assert "checkpoint-9.tmp" in (checkpoints.fresh_run_clash(tmp_path) or "")
        assert checkpoints.checkpoint_dirs(tmp_path) == ["checkpoint-9.tmp"]

    @pytest.mark.parametrize("surviving", [
        [],                                            # killed immediately
        ["adapter_model.safetensors"],                 # after weights
        ["adapter_model.safetensors", "rng_state.pt"],
        _TORN_SUBSET,                                  # after optimizer
        _TORN_SUBSET + ["lr_scheduler_state.pt"],      # after scheduler
    ])
    def test_kill_after_any_write_stage_is_torn(self, tmp_path, surviving):
        """Fault injection across the save sequence: whatever subset of state
        files landed before the kill, no sentinel means not complete."""
        d = _ckpt(tmp_path, "checkpoint-8",
                  [checkpoints.FORMAT_MARKER] + surviving)
        assert not checkpoints.is_complete(d)

    def test_latest_falls_back_to_previous_complete_checkpoint(self, tmp_path):
        _ckpt(tmp_path, "checkpoint-4",
              [checkpoints.FORMAT_MARKER, checkpoints.COMPLETE_MARKER])
        _ckpt(tmp_path, "checkpoint-8",
              [checkpoints.FORMAT_MARKER] + _TORN_SUBSET)  # torn newer save
        complete = checkpoints.list_checkpoints(tmp_path)
        assert [step for step, _ in complete] == [4]


# ---------------------------------------------------------------------------
# poros export: remote-code authorization comes from the CLI flag alone
# ---------------------------------------------------------------------------


def test_export_ignores_manifest_trust_remote_code(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    from typer.testing import CliRunner

    import poros.train.export as pe
    from poros.cli.commands import export as cli_commands_export
    from poros.cli.main import app
    from tests.conftest import as_linux

    as_linux(monkeypatch, cli_commands_export)  # the CUDA merger's route: a Mac merges its own way
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "poros_config.json").write_text(json.dumps({
        "model_name_or_path": "attacker/repository",
        "trust_remote_code": True,
        "lora_bias": "all",
        "model_revision": "a" * 40,
    }))

    captured = {}

    def fake_export(cfg, adapter_dir, output_dir, **kw):
        captured["cfg"] = cfg
        raise RuntimeError("stop after capturing the export config")

    monkeypatch.setattr(pe, "export_merged_dense", fake_export)
    CliRunner().invoke(app, ["export", str(adapter)])
    assert captured, "export never reached the merger"
    assert captured["cfg"].trust_remote_code is False
    # The manifest's ADAPTER-DEFINING fields must reach the merger: bias
    # shape and the pinned base revision (same-shaped later revisions pass
    # every key/shape check while silently changing the frozen weights).
    assert captured["cfg"].lora_bias == "all"
    assert captured["cfg"].model_revision == "a" * 40


def test_trainer_forwards_bias_and_revision(monkeypatch):
    pytest.importorskip("torch")
    import poros.train.model as pm
    from poros.core.config import PorosConfig
    from poros.train.trainer import PorosTrainer

    captured = {}

    def fake_from_pretrained(name, **kw):
        captured.update(kw)
        raise RuntimeError("stop after capturing loader kwargs")

    monkeypatch.setattr(pm.PorosModel, "from_pretrained",
                        staticmethod(fake_from_pretrained))
    cfg = PorosConfig(model_name_or_path="org/base",
                      lora_bias="lora_only", model_revision="b" * 40,
                      blockwise_arch_check="off")
    with pytest.raises(RuntimeError, match="capturing"):
        PorosTrainer.from_pretrained("org/base", config=cfg)
    assert captured["lora_bias"] == "lora_only"
    assert captured["revision"] == "b" * 40


# ---------------------------------------------------------------------------
# HTML report: artifact-derived fields are escaped, never executed
# ---------------------------------------------------------------------------


def test_report_escapes_artifact_fields(tmp_path):
    pytest.importorskip("jinja2")
    from poros.report.html import generate_report_from_jsonl

    payload = "32b</div><script>alert(1)</script>"
    row = {
        "tag": payload, "quant": "qlora4",
        "arch": "<svg onload=alert(2)>",
        "hardware": "\"><img src=x onerror=alert(3)>",
        "status": "ok", "steps": 200, "seq_len": 512,
        "poros_vram_mb": 10184.0, "resident_vram_mb": 33966.0,
        "s_per_step": 1.9,
    }
    src = tmp_path / "rows.jsonl"
    src.write_text(json.dumps(row))
    out = generate_report_from_jsonl(src, tmp_path / "report.html")
    html = out.read_text()

    assert "<script>alert(1)</script>" not in html
    assert "<svg onload" not in html
    assert "<img src=x" not in html
    assert "&lt;script&gt;" in html  # escaped, not dropped
    # Defense in depth: the CSP blocks egress (default-src 'none' denies
    # fetch/XHR and non-data images). It cannot block inline script -- the
    # template's own script needs 'unsafe-inline' -- so escaping above is
    # the actual injection fix; the CSP only denies exfiltration.
    assert "default-src 'none'" in html
    # Our own trusted blobs must render as MARKUP, not escaped text: the
    # real stylesheet's content (with CSS combinators) and the generated
    # SVG. These fail if the Markup() wrapping is dropped.
    assert "--accent" in html and "&gt;" not in html.split("</style>")[0]
    assert '<svg class="parity-curve"' in html  # Markup(): escaped = broken
    assert "&lt;path" not in html
    # The footer must not deny that ratios/percentages are derived.
    assert "derived from those measurements" in html
    assert "No numbers derived" not in html


def test_markdown_report_footer_and_skipped_row_warning(tmp_path):
    from poros.report.markdown import _load_jsonl, render_markdown_string

    good = {"tag": "m-32b", "quant": "qlora4", "status": "ok", "steps": 200,
            "seq_len": 512, "poros_vram_mb": 10184.0,
            "resident_vram_mb": 33966.0, "s_per_step": 1.9}
    p = tmp_path / "rows.jsonl"
    p.write_text(json.dumps(good) + "\n{ this line is not json\n")
    with pytest.warns(UserWarning, match="skipped"):
        _load_jsonl(p)
    md = render_markdown_string(str(p))
    assert "derived from them" in md
    assert "none are derived" not in md


@pytest.mark.parametrize("section,kw", [
    ("training", {"steps": 0}),
    ("training", {"gradient_accumulation_steps": 0}),
    ("lora", {"rank": 0}),
    ("lora", {"dropout": 1.0}),
    ("poros", {"block_size": 0}),
])
def test_bench_spec_rejects_degenerate_values(section, kw):
    from poros.bench.specs import BenchmarkSpec

    with pytest.raises(Exception):
        BenchmarkSpec(
            spec_id="x",
            models=[{"model_id": "m", "display_name": "m"}],
            **{section: kw},
        )


# ---------------------------------------------------------------------------
# Native adapter backend: never a fake "unload"
# ---------------------------------------------------------------------------


def test_native_merge_and_unload_raises():
    pytest.importorskip("torch")
    from poros.train.model import PorosModel

    m = PorosModel()
    m._adapter_backend = "native"
    m._model = object()
    with pytest.raises(NotImplementedError, match="native"):
        m.merge_and_unload()


# ---------------------------------------------------------------------------
# Config bounds: degenerate values fail at parse, sentinels stay legal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [
    {"logging_steps": 0},               # step % logging_steps would crash
    {"gradient_accumulation_steps": 0}, # empty accumulation loop
    {"lora_rank": 0},
    {"lora_alpha": 0},
    {"learning_rate": -1.0},
    {"learning_rate": 0.0},
    {"lora_dropout": 1.0},
    {"adam_beta2": 1.0},
    {"adam_epsilon": 0.0},
    {"per_device_batch_size": 0},
    {"max_seq_length": 0},
    {"block_size": 0},
])
def test_config_rejects_degenerate_values(kw):
    from poros.core.config import PorosConfig

    with pytest.raises(Exception):
        PorosConfig(model_name_or_path="x", **kw)


def test_config_disable_sentinels_still_allowed():
    from poros.core.config import PorosConfig

    cfg = PorosConfig(
        model_name_or_path="x", save_steps=0, eval_steps=0,
        max_steps=-1, max_grad_norm=0.0, save_total_limit=0,
        warmup_steps=0,
    )
    assert cfg.save_steps == 0 and cfg.max_steps == -1


# ---------------------------------------------------------------------------
# Removed fields: loud migration for configs, silent tolerance for manifests
# ---------------------------------------------------------------------------


def test_an_unknown_yaml_key_is_refused_by_name(tmp_path):
    from typer.testing import CliRunner

    from poros.cli.main import app

    cfg = tmp_path / "train.yaml"
    cfg.write_text('model_name_or_path: "org/base"\nfused_lora: true\n')
    res = CliRunner().invoke(app, ["train", str(cfg)])
    assert res.exit_code == 2
    assert "unknown field fused_lora" in res.output


def test_old_manifest_with_fused_lora_still_exports(tmp_path, monkeypatch):
    """Two PUBLICLY SHIPPED validation manifests carry fused_lora: the
    model_fields filter must keep them loadable forever."""
    pytest.importorskip("torch")
    from typer.testing import CliRunner

    import poros.train.export as pe
    from poros.cli.commands import export as cli_commands_export
    from poros.cli.main import app
    from tests.conftest import as_linux

    as_linux(monkeypatch, cli_commands_export)  # the CUDA merger's route: a Mac merges its own way
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "poros_config.json").write_text(json.dumps({
        "model_name_or_path": "org/base",
        "fused_lora": True,  # written by 0.2.x / early 0.3.0
    }))
    captured = {}

    def fake_export(cfg, adapter_dir, output_dir, **kw):
        captured["cfg"] = cfg
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(pe, "export_merged_dense", fake_export)
    CliRunner().invoke(app, ["export", str(adapter)])
    assert captured, "old manifest was refused before reaching the merger"
    assert "fused_lora" not in captured["cfg"].model_dump()


# ---------------------------------------------------------------------------
# Adapter loading: completeness, not just key consistency
# ---------------------------------------------------------------------------


def _export_adapter_gate(torch, cfg, path):
    """The merged export's adapter check (the shared exact-match gate), on its own."""
    from poros.core.config import PorosConfig
    from poros.train.export import _adapter_plan, _skeleton

    cfg.save_pretrained(str(path / "base"))
    skeleton = _skeleton(str(path / "base"), torch.float32, False, None)
    _adapter_plan(PorosConfig(model_name_or_path=str(path / "base")), path, skeleton)


def test_truncated_adapter_file_is_refused(tmp_path):
    """A partial adapter save (fewer tensors than the model's trainable set)
    must raise, not silently merge half the adapter."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("safetensors")
    from safetensors.torch import load_file, save_file

    from poros.core.errors import PorosExportError
    from tests.test_train_loop_cpu import _tiny_peft_qwen2

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    model.save_pretrained(str(tmp_path))
    st = tmp_path / "adapter_model.safetensors"
    state = load_file(str(st))
    state.pop(sorted(state)[0])  # truncate: drop one adapter tensor
    save_file(state, str(st))

    with pytest.raises(PorosExportError, match="partial|truncated"):
        _export_adapter_gate(torch, cfg, tmp_path)


def test_complete_adapter_file_loads(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tests.test_train_loop_cpu import _tiny_peft_qwen2

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    model.save_pretrained(str(tmp_path))
    _export_adapter_gate(torch, cfg, tmp_path)  # must not raise


# ---------------------------------------------------------------------------
# Hub push: portable adapter/tokenizer files only, never private run state
# ---------------------------------------------------------------------------


def test_push_to_hub_uploads_only_portable_files(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    folder_calls = {}
    file_calls = {}

    class FakeApi:
        def upload_folder(self, **kw):
            folder_calls.update(kw)

        def upload_file(self, **kw):
            file_calls.update(kw)

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(HfApi=FakeApi))
    (tmp_path / "poros_config.json").write_text(json.dumps({
        "model_name_or_path": "org/base", "lora_rank": 16,
        "trust_remote_code": True,
        "resolved_base_model_path": "/home/someone/.cache/hub/snap",
        "output_dir": "/home/someone/runs/x",
    }))
    from poros.train.checkpointing import _push_to_hub

    (tmp_path / "adapter_model.safetensors").write_bytes(b"w")  # a PEFT save's weights
    _push_to_hub(tmp_path, "org/repo")
    patterns = folder_calls["allow_patterns"]
    assert "adapter_model.safetensors" in patterns
    for private in ("optimizer_state.pt", "rng_state.pt",
                    "lr_scheduler_state.pt", "poros_config.json"):
        assert private not in patterns
    # No wildcard broad enough to sweep the private files back in.
    assert not any(p in ("*", "**", "*.pt", "*.json") for p in patterns)
    # A SANITIZED manifest goes up separately so a downloaded adapter is
    # still exportable -- with the run-private fields stripped.
    assert file_calls["path_in_repo"] == "poros_config.json"
    manifest = json.loads(file_calls["path_or_fileobj"].getvalue())
    assert manifest["model_name_or_path"] == "org/base"
    assert manifest["lora_rank"] == 16
    for stripped in ("trust_remote_code", "resolved_base_model_path",
                     "output_dir"):
        assert stripped not in manifest


def test_push_to_hub_uploads_the_native_backends_weights_and_refuses_none(tmp_path,
                                                                         monkeypatch):
    """A CUDA MoE run's adapter is adapter_weights.pt (the native backend): it goes up. A folder
    with no weights file uploads nothing at all and says so."""
    pytest.importorskip("torch")
    calls = []

    class FakeApi:
        def upload_folder(self, **kw):
            calls.append(("folder", kw))

        def upload_file(self, **kw):
            calls.append(("file", kw))

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    (tmp_path / "poros_config.json").write_text(json.dumps(
        {"model_name_or_path": "Qwen/Qwen3-30B-A3B", "lora_rank": 8,
         "adapter_backend": "native"}))
    from poros.train.checkpointing import _push_to_hub

    with pytest.raises(ValueError, match="holds no adapter weights"):
        _push_to_hub(tmp_path, "org/repo")
    assert calls == []
    for name in ("adapter_weights.pt", "rng_state.pt", "optimizer_state.pt", "README.md"):
        (tmp_path / name).write_bytes(b"x")
    _push_to_hub(tmp_path, "org/repo")
    patterns = calls[0][1]["allow_patterns"]
    monkeypatch.undo()
    from huggingface_hub.utils import filter_repo_objects

    sent = set(filter_repo_objects([p.name for p in tmp_path.iterdir()], allow_patterns=patterns))
    assert sent == {"adapter_weights.pt", "README.md"}


def test_push_to_hub_missing_id_raises(tmp_path):
    pytest.importorskip("torch")
    from poros.train.checkpointing import _push_to_hub

    with pytest.raises(ValueError, match="hub_model_id"):
        _push_to_hub(tmp_path, None)


def test_config_rejects_push_without_hub_id():
    from poros.core.config import PorosConfig

    with pytest.raises(Exception, match="hub_model_id"):
        PorosConfig(model_name_or_path="x", push_to_hub=True)


def test_push_to_hub_local_base_path_raises(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(HfApi=object))
    local_base = tmp_path / "my-local-base"
    local_base.mkdir()
    (tmp_path / "poros_config.json").write_text(json.dumps({
        "model_name_or_path": str(local_base), "lora_rank": 16,
    }))
    from poros.train.checkpointing import _push_to_hub

    with pytest.raises(ValueError, match="local path"):
        _push_to_hub(tmp_path, "org/repo")
