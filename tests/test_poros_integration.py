"""Integration tests for Poros: config, errors, events, the benchmark harness, reports,
leaderboard and architecture registry. Pure Python; the few tests that need jinja2 skip
without it.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# ============================================================================
# Paths to measured benchmark data
# ============================================================================

# Measured B300 results, bundled in-repo so the suite runs in a fresh clone.
_GRID_PRO_JSONL = Path(__file__).parent / "data" / "poros_grid_pro.jsonl"


# ============================================================================
# Pure-python tests (no torch required)
# ============================================================================


class TestPorosConfig:
    """PorosConfig pydantic model tests."""

    def test_defaults(self):
        from poros.core.config import PorosConfig

        cfg = PorosConfig()
        assert cfg.quantization == "nf4"
        assert cfg.lora_rank == 16
        assert cfg.lora_alpha == 32
        assert cfg.blockwise_arch_check == "error"
        assert cfg.preserve_rng_state is True
        assert cfg.seed == 42
        assert cfg.per_device_batch_size == 1
        assert cfg.enable_prefetch is True

    def test_custom_values(self):
        from poros.core.config import PorosConfig

        cfg = PorosConfig(
            model_name_or_path="Qwen/Qwen2.5-32B",
            block_size=8,
            lora_rank=64,
            learning_rate=1e-4,
        )
        assert cfg.model_name_or_path == "Qwen/Qwen2.5-32B"
        assert cfg.quantization == "nf4"
        assert cfg.block_size == 8
        assert cfg.lora_rank == 64
        assert cfg.learning_rate == 1e-4

    def test_invalid_quantization(self):
        from poros.core.config import PorosConfig

        with pytest.raises(Exception):
            PorosConfig(quantization="int32")

    def test_yaml_round_trip(self):
        import yaml

        from poros.core.config import PorosConfig

        cfg = PorosConfig(block_size=4, lora_rank=32)
        data = cfg.model_dump()
        yaml_str = yaml.dump(data)
        reloaded = yaml.safe_load(yaml_str)
        cfg2 = PorosConfig(**reloaded)
        assert cfg2.block_size == cfg.block_size
        assert cfg2.lora_rank == cfg.lora_rank

    def test_config_json_round_trip(self):
        from poros.core.config import PorosConfig

        cfg = PorosConfig(block_size=2, seed=123)
        j = cfg.model_dump_json()
        cfg2 = PorosConfig.model_validate_json(j)
        assert cfg2.block_size == 2
        assert cfg2.seed == 123


class TestErrors:
    """Structured error tests."""

    def test_poros_error_has_info(self):
        from poros.core.errors import PorosError

        e = PorosError("test", remediation="do this", doc_url="https://x.com")
        assert e.info.error_type == "PorosError"
        assert e.info.message == "test"
        assert e.info.remediation == "do this"
        assert e.info.doc_url == "https://x.com"

    def test_arch_not_validated_error(self):
        from poros.core.errors import PorosArchNotValidatedError

        e = PorosArchNotValidatedError("gemma", {"qwen3", "qwen2"})
        assert "gemma" in str(e)
        assert "qwen2" in str(e)
        assert e.info.remediation != ""

    def test_oom_error_with_params(self):
        from poros.core.errors import PorosOOMError

        e = PorosOOMError(block_size=8, seq_len=4096)
        assert "block_size=4" in e.info.remediation
        assert "seq_len=2048" in e.info.remediation

    def test_config_error(self):
        from poros.core.errors import PorosConfigError

        e = PorosConfigError("bad config")
        assert "bad config" in str(e)


class TestEvents:
    """JSONL event system tests."""

    def test_event_creation(self):
        from poros.core.events import PorosEvent

        event = PorosEvent(
            event_type="step",
            step=10,
            data={"loss": 1.5, "grad_norm": 0.3},
        )
        assert event.event_type == "step"
        assert event.step == 10
        assert event.data["loss"] == 1.5

    def test_event_writer_round_trip(self):
        from poros.core.events import EventWriter, PorosEvent

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = EventWriter(tmpdir)
            writer.emit(PorosEvent(event_type="setup", data={"model": "test"}))
            writer.emit(PorosEvent(event_type="step", step=0, data={"loss": 2.5}))
            writer.emit(PorosEvent(event_type="step", step=1, data={"loss": 2.3}))

            lines = (Path(tmpdir) / "events.jsonl").read_text().splitlines()
            events = [PorosEvent.model_validate_json(line) for line in lines]
            assert len(events) == 3
            assert events[0].event_type == "setup"
            assert events[1].step == 0
            assert events[2].data["loss"] == 2.3


class TestBaselineScripts:
    """The generated baseline subprocess scripts must be valid Python.

    Regression guard: a multi-line quant-config block injected into a
    textwrap.dedent template once defeated the dedent and shipped scripts that
    raised IndentationError at subprocess launch (so the baselines never ran).
    """

    def _spec(self):
        from poros.bench.specs import BenchmarkSpec, ModelSpec

        return BenchmarkSpec(
            spec_id="compile-check",
            models=[ModelSpec(model_id="org/model", display_name="Model")],
        )

    def test_all_baseline_scripts_compile(self):
        from poros.bench.baselines import (
            accelerate_offload,
            deepspeed_zero3,
            peft,
            poros,
            resident,
            unsloth,
        )

        spec = self._spec()
        mods = {
            "poros": poros,
            "resident": resident,
            "unsloth": unsloth,
            "peft": peft,
            "accelerate_offload": accelerate_offload,
            "deepspeed_zero3": deepspeed_zero3,
        }
        for name, mod in mods.items():
            for precision in ("nf4", "bf16", "fp16"):
                script = mod._build_script(spec, "org/model", precision)
                # Raises SyntaxError/IndentationError if the template is broken.
                compile(script, f"{name}:{precision}", "exec")

    def test_baselines_never_fabricate_parity(self):
        """No baseline may emit a measured 0.0 parity or fake bitwise_exact."""
        from poros.bench.baselines import (
            accelerate_offload,
            deepspeed_zero3,
            peft,
            poros,
            resident,
            unsloth,
        )

        spec = self._spec()
        # poros included: its OOM/error branches must report null parity, not a
        # fabricated 0.0 (a crashed cell has no measured parity).
        for mod in (resident, unsloth, peft, accelerate_offload, deepspeed_zero3, poros):
            script = mod._build_script(spec, "org/model", "nf4")
            assert '"parity_loss_max_diff": 0.0' not in script
            assert '"parity_weight_max_diff": 0.0' not in script
            # speed/VRAM cells never measure parity -> never claim bitwise
            assert '"bitwise_exact": True' not in script

    def test_baselines_time_per_optimizer_step(self):
        """Every baseline must record ONE optimizer step (GRAD_ACCUM
        micro-batches) per timed entry, matching the Poros trainer's
        step_time_history. Timing a single micro-batch while Poros times a
        full optimizer step inflates slowdown_vs_resident_x by ~GRAD_ACCUM."""
        from poros.bench.baselines import (
            accelerate_offload,
            deepspeed_zero3,
            peft,
            resident,
            unsloth,
        )

        spec = self._spec()
        for mod in (resident, unsloth, peft, accelerate_offload, deepspeed_zero3):
            script = mod._build_script(spec, "org/model", "nf4")
            # Must accumulate GRAD_ACCUM micro-batches inside one timed step.
            assert "for _ in range(GRAD_ACCUM)" in script
            # Must NOT use the old per-micro-batch timing pattern.
            assert "if (step + 1) % GRAD_ACCUM == 0" not in script
            # tok_per_sec must include the GRAD_ACCUM factor.
            assert "SEQ_LEN * BATCH_SIZE * GRAD_ACCUM" in script

    def test_arms_measure_through_shared_helpers(self):
        """Every hand-written loop takes its batch, clock and VRAM split from
        the shared preamble, so a difference between rows is the method's."""
        from poros.bench.baselines import (
            accelerate_offload,
            deepspeed_zero3,
            peft,
            poros,
            resident,
            unsloth,
        )

        spec = self._spec()
        scripts = [
            mod._build_script(spec, "org/model", "nf4")
            for mod in (resident, unsloth, peft, accelerate_offload, deepspeed_zero3)
        ]
        for script in scripts:
            assert "dummy_ids = _synthetic_batch()" in script
            assert "_start_training()\n" in script
            assert "step_times.append(_now() - t_start)" in script
            assert "**_measurements(losses" in script
            assert "**_measurements(oom=True)" in script
            assert "torch.randint(100, 30000, (BATCH_SIZE, SEQ_LEN), device=" not in script
        trainer = poros._build_trainer_script(spec, "org/model", "nf4")
        assert "_start_training()" in trainer and "**_measurements(" in trainer

    def test_vram_cap_and_accelerate_settings_reach_the_script(self):
        from poros.bench.baselines import accelerate_offload
        from poros.bench.specs import AccelerateSpec

        spec = self._spec()
        assert "VRAM_CAP_GB = None" in accelerate_offload._build_script(spec, "o/m", "nf4")
        spec.hardware.vram_cap_gb = 22.76
        spec.accelerate_offload = AccelerateSpec(gpu_max_memory_gb=12, gradient_checkpointing=True)
        script = accelerate_offload._build_script(spec, "o/m", "nf4")
        assert "VRAM_CAP_GB = 22.76" in script
        assert "GPU_MAX_MEMORY_GB = 12.0" in script
        assert "GRADIENT_CHECKPOINTING = True" in script
        assert "llm_int8_enable_fp32_cpu_offload=True" in script
        assert '"cpu": f"{CPU_SHARE_MB}MiB"' in script

    def test_sigkill_is_labeled_host_oom(self, monkeypatch):
        import subprocess

        from poros.bench.baselines import _subprocess

        killed = subprocess.CompletedProcess([], -9, stdout="", stderr="CUDA out of memory")
        monkeypatch.setattr(_subprocess.subprocess, "run", lambda *a, **k: killed)
        art = _subprocess.run_in_subprocess("", "peft", "o/m", "m", "nf4", self._spec())
        assert art.status == "error"
        assert "likely host out of memory" in art.error_message

    def test_wsl_requires_a_vram_cap(self, monkeypatch):
        from poros.bench import runner

        monkeypatch.setattr(runner, "_environment", lambda: {"wsl": True})
        spec = self._spec()
        with pytest.raises(runner.FairnessViolation, match="vram_cap_gb"):
            runner._validate_fairness(spec)
        spec.hardware.vram_cap_gb = 22.0
        runner._validate_fairness(spec)

    def test_zero3_config_follows_spec(self):
        """The ZeRO-3 arm offloads where spec.zero3 says, and user keys win."""
        from poros.bench.baselines.deepspeed_zero3 import _ds_config
        from poros.bench.specs import Zero3Spec

        spec = self._spec()
        zero = _ds_config(spec, "nf4")["zero_optimization"]
        assert zero == {"stage": 3, "offload_param": {"device": "cpu", "pin_memory": True}}

        with pytest.raises(ValueError, match="nvme_path"):
            Zero3Spec(offload_device="nvme")
        spec.zero3 = Zero3Spec(
            offload_device="nvme",
            nvme_path="/nvme",
            zero_optimization={
                "offload_param": {"buffer_size": 10**9},
                "stage3_max_live_parameters": 10**8,
            },
        )
        cfg = _ds_config(spec, "fp16")
        assert cfg["fp16"] == {"enabled": True} and "bf16" not in cfg
        zero = cfg["zero_optimization"]
        # Overriding one offload_param key keeps the device and path.
        assert zero["offload_param"] == {
            "device": "nvme", "pin_memory": True, "nvme_path": "/nvme", "buffer_size": 10**9,
        }
        assert zero["stage3_max_live_parameters"] == 10**8


class TestBenchmarkSpec:
    """BenchmarkSpec validation and I/O."""

    def test_valid_spec(self):
        from poros.bench.specs import BenchmarkSpec, ModelSpec

        spec = BenchmarkSpec(
            spec_id="test-run",
            models=[ModelSpec(model_id="test/model", display_name="Test-7B")],
            methods=["poros", "resident"],
        )
        assert spec.spec_id == "test-run"
        assert len(spec.models) == 1
        assert spec.training.steps == 200
        assert spec.lora.rank == 16

    def test_invalid_method(self):
        from poros.bench.specs import BenchmarkSpec, ModelSpec

        with pytest.raises(Exception):
            BenchmarkSpec(
                spec_id="bad",
                models=[ModelSpec(model_id="x", display_name="X")],
                methods=["poros", "nonexistent_method"],
            )

    def test_spec_hash_deterministic(self):
        from poros.bench.specs import BenchmarkSpec, ModelSpec

        spec = BenchmarkSpec(
            spec_id="hash-test",
            models=[ModelSpec(model_id="m", display_name="M")],
        )
        h1 = spec.spec_hash()
        h2 = spec.spec_hash()
        assert h1 == h2
        assert len(h1) == 16


class TestRunArtifact:
    """RunArtifact schema tests."""

    def test_artifact_creation(self):
        from poros.bench.artifacts import RunArtifact

        art = RunArtifact(
            method="poros",
            model_id="Qwen/Qwen2.5-32B",
            display_name="Qwen2.5-32B",
            precision="nf4",
            vram_peak_mb=11085.0,
            parity_loss_max_diff=0.0,
            parity_weight_max_diff=0.0,
            bitwise_exact=True,
            steps=200,
        )
        assert art.bitwise_exact is True
        assert art.status == "ok"

    def test_artifact_json_round_trip(self):
        from poros.bench.artifacts import RunArtifact

        art = RunArtifact(
            method="poros",
            model_id="test",
            display_name="Test",
            precision="nf4",
            vram_peak_mb=1000.0,
            bitwise_exact=True,
            steps=50,
        )
        j = art.model_dump_json()
        art2 = RunArtifact.model_validate_json(j)
        assert art2.vram_peak_mb == 1000.0
        assert art2.bitwise_exact is True

    def test_artifact_write_and_read(self):
        from poros.bench.artifacts import RunArtifact, read_artifact, write_artifact

        art = RunArtifact(
            method="poros",
            model_id="test",
            display_name="Test",
            precision="nf4",
            arch="qwen2",
            vram_peak_mb=5000.0,
            steps=100,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_artifact(art, tmpdir)
            assert path.exists()
            loaded = read_artifact(path)
            assert loaded.method == "poros"
            assert loaded.vram_peak_mb == 5000.0

    def test_grid_pro_row_loading(self):
        from poros.bench.artifacts import load_grid_pro_row

        row = {
            "arch": "qwen2_5",
            "tag": "qwen32b",
            "quant": "qlora4",
            "status": "ok",
            "bitwise_exact": True,
            "parity_loss_max_diff": 0.0,
            "parity_weight_max_diff": 0.0,
            "resident_vram_mb": 42842.0,
            "poros_vram_mb": 11085.0,
            "unsloth_vram_mb": 20434.0,
            "vram_save_vs_resident_pct": 74.1,
            "vram_save_vs_unsloth_pct": 45.8,
            "slowdown_vs_resident_x": 2.18,
            "steps": 200,
        }
        art = load_grid_pro_row(row)
        assert art.precision == "nf4"  # qlora4 -> nf4
        assert art.bitwise_exact is True
        assert art.vram_poros_mb == 11085.0

    def test_parity_unmeasured_is_none(self):
        """A throughput cell must NOT fabricate a 0.00e+00 parity.

        Parity is measured by the dedicated self-check / GPU test, never by a
        speed/VRAM cell, so an unmeasured cell carries None and survives a
        JSON round-trip as None.
        """
        from poros.bench.artifacts import RunArtifact

        art = RunArtifact(
            method="poros",
            model_id="test",
            display_name="Test",
            precision="nf4",
            vram_peak_mb=1000.0,
        )
        assert art.parity_loss_max_diff is None
        assert art.parity_weight_max_diff is None
        assert art.bitwise_exact is False
        art2 = RunArtifact.model_validate_json(art.model_dump_json())
        assert art2.parity_loss_max_diff is None

    def test_multipass_aggregate_fields(self):
        """Multi-pass official-run aggregates round-trip and stay coherent."""
        from poros.bench.artifacts import RunArtifact

        art = RunArtifact(
            method="poros",
            model_id="test",
            display_name="Test",
            precision="nf4",
            s_per_step=0.42,
            n_passes=3,
            s_per_step_min=0.40,
            s_per_step_max=0.45,
            s_per_step_stddev=0.02,
            pass_s_per_step=[0.45, 0.42, 0.40],
            pass_vram_peaks_mb=[1010.0, 1000.0, 1005.0],
            vram_peak_mb=1010.0,  # worst-case across passes
        )
        art2 = RunArtifact.model_validate_json(art.model_dump_json())
        assert art2.n_passes == 3
        assert art2.s_per_step_min <= art2.s_per_step <= art2.s_per_step_max
        assert len(art2.pass_s_per_step) == 3
        assert art2.vram_peak_mb == max(art2.pass_vram_peaks_mb)

    def test_avg_vram_cross_reference(self):
        """Benchmark records MEASURED average VRAM-used per method and
        cross-references poros vs resident vs unsloth so the three can be
        compared apples-to-apples."""
        from poros.bench.artifacts import RunArtifact
        from poros.bench.runner import _compute_comparisons

        common = dict(precision="nf4", model_id="Qwen/Qwen2.5-32B", status="ok")
        poros = RunArtifact(
            method="poros", display_name="p",
            vram_peak_mb=11740.0, vram_avg_mb=11000.0, **common,
        )
        resident = RunArtifact(
            method="resident", display_name="r",
            vram_peak_mb=42842.0, vram_avg_mb=41000.0, **common,
        )
        unsloth = RunArtifact(
            method="unsloth", display_name="u",
            vram_peak_mb=20434.0, vram_avg_mb=19500.0, **common,
        )
        out = _compute_comparisons([poros, resident, unsloth])
        p = next(a for a in out if a.method == "poros")
        # Average VRAM-used is captured for every method...
        assert p.vram_avg_mb == 11000.0
        # ...and poros carries the resident + unsloth averages for comparison.
        assert p.vram_avg_resident_mb == 41000.0
        assert p.vram_avg_unsloth_mb == 19500.0
        # Survives a JSON round-trip.
        p2 = RunArtifact.model_validate_json(p.model_dump_json())
        assert p2.vram_avg_resident_mb == 41000.0


class TestParityFormatting:
    """The CLI / report parity formatters must never fake 0.00e+00."""

    def test_html_fmt_parity_none_is_dash(self):
        from poros.report.html import _fmt_parity

        assert _fmt_parity(None) == "—"
        assert _fmt_parity(0.0) == "0.00e+00"
        assert _fmt_parity(1.5e-3) == "1.50e-03"

    def test_cli_format_parity_none_is_dash(self):
        from poros.bench.artifacts import RunArtifact
        from poros.cli.commands.report import _format_parity

        unmeasured = RunArtifact(
            method="poros", model_id="t", display_name="T", precision="nf4"
        )
        assert _format_parity(unmeasured) == "—"
        bitwise = RunArtifact(
            method="poros", model_id="t", display_name="T", precision="nf4",
            parity_loss_max_diff=0.0, bitwise_exact=True,
        )
        assert _format_parity(bitwise) == "0.00e+00"


class TestReportMarkdown:
    """Markdown report generation tests."""

    @pytest.fixture
    def sample_artifacts(self):
        from poros.bench.artifacts import RunArtifact

        return [
            RunArtifact(
                method="poros",
                model_id="test/model-7B",
                display_name="Model-32B",
                precision="nf4",
                arch="qwen2",
                hardware="RTX 3090",
                vram_peak_mb=5000.0,
                vram_resident_mb=20000.0,
                vram_unsloth_mb=12000.0,
                vram_poros_mb=5000.0,
                s_per_step=1.5,
                parity_loss_max_diff=0.0,
                parity_weight_max_diff=0.0,
                bitwise_exact=True,
                vram_saved_vs_resident_pct=75.0,
                vram_saved_vs_unsloth_pct=58.3,
                slowdown_vs_resident_x=2.1,
                steps=200,
                seq_len=512,
            ),
        ]

    def test_markdown_from_artifacts(self, sample_artifacts):
        from poros.report.markdown import generate_markdown_report

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_markdown_report(
                sample_artifacts, f"{tmpdir}/report.md"
            )
            assert path.exists()
            content = path.read_text()
            assert "Model-32B" in content
            assert "0.00e+00" in content

    def test_markdown_string_render(self, sample_artifacts):
        from poros.report.markdown import render_markdown_string

        md = render_markdown_string(sample_artifacts)
        assert "VRAM" in md
        assert "Parity" in md

    @pytest.mark.skipif(
        not _GRID_PRO_JSONL.exists(),
        reason="poros_grid_pro.jsonl not available",
    )
    def test_markdown_from_jsonl(self):
        from poros.report.markdown import generate_markdown_report

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_markdown_report(
                str(_GRID_PRO_JSONL),
                f"{tmpdir}/report.md",
                hardware_info={"gpu_name": "NVIDIA B300"},
            )
            assert path.exists()
            content = path.read_text()
            # Verify real numbers appear
            assert "qwen32b" in content or "qwen72b" in content
            assert "0.00e+00" in content


class TestReportHTML:
    """HTML report generation tests."""

    @pytest.fixture
    def sample_artifacts(self):
        from poros.bench.artifacts import RunArtifact

        return [
            RunArtifact(
                method="poros",
                model_id="test/model-32B",
                display_name="Model-32B",
                precision="nf4",
                arch="qwen2_5",
                hardware="NVIDIA B300",
                vram_peak_mb=11085.0,
                vram_resident_mb=42842.0,
                vram_unsloth_mb=20434.0,
                vram_poros_mb=11085.0,
                s_per_step=2.0,
                parity_loss_max_diff=0.0,
                parity_weight_max_diff=0.0,
                bitwise_exact=True,
                vram_saved_vs_resident_pct=74.1,
                vram_saved_vs_unsloth_pct=45.8,
                slowdown_vs_resident_x=2.18,
                steps=200,
                seq_len=512,
            ),
        ]

    def test_html_generation(self, sample_artifacts):
        pytest.importorskip("jinja2")
        from poros.report.html import generate_html_report

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_html_report(
                sample_artifacts, f"{tmpdir}/report.html"
            )
            assert path.exists()
            content = path.read_text()
            # Self-contained: no external CSS/JS links
            assert "<style>" in content
            assert "<script>" in content
            assert "Model-32B" in content
            assert "0.00e+00" in content

    def test_html_self_contained(self, sample_artifacts):
        pytest.importorskip("jinja2")
        from poros.report.html import generate_html_report

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_html_report(
                sample_artifacts, f"{tmpdir}/report.html"
            )
            content = path.read_text()
            # No external resource fetches (CSS/JS/fonts loaded from network).
            # Simple hyperlinks (<a href=...>) in footer text are fine.
            assert 'src="http' not in content
            # No external stylesheet link tags
            assert '<link rel="stylesheet" href="http' not in content

    @pytest.mark.skipif(
        not _GRID_PRO_JSONL.exists(),
        reason="poros_grid_pro.jsonl not available",
    )
    def test_html_from_jsonl(self):
        pytest.importorskip("jinja2")
        from poros.report.html import generate_report_from_jsonl

        with tempfile.TemporaryDirectory() as tmpdir:
            path = generate_report_from_jsonl(
                str(_GRID_PRO_JSONL),
                f"{tmpdir}/report.html",
                hardware_info={"gpu": "NVIDIA B300"},
            )
            assert path.exists()
            size = path.stat().st_size
            assert size > 10000  # should be substantial


class TestLeaderboard:
    """Leaderboard aggregation tests."""

    @pytest.mark.skipif(
        not _GRID_PRO_JSONL.exists(),
        reason="poros_grid_pro.jsonl not available",
    )
    def test_build_leaderboard_from_jsonl(self):
        from poros.report.leaderboard import build_leaderboard

        entries = build_leaderboard(str(_GRID_PRO_JSONL))
        assert len(entries) > 0
        # First entry should have the best VRAM savings
        assert entries[0]["rank"] == 1
        assert "vram_save_vs_resident_pct" in entries[0]

    @pytest.mark.skipif(
        not _GRID_PRO_JSONL.exists(),
        reason="poros_grid_pro.jsonl not available",
    )
    def test_leaderboard_excludes_not_bitwise(self):
        from poros.report.leaderboard import build_leaderboard

        validated_only = build_leaderboard(
            str(_GRID_PRO_JSONL), include_not_bitwise=False
        )
        for e in validated_only:
            assert e["bitwise_exact"] is True

    def test_leaderboard_markdown_render(self):
        from poros.report.leaderboard import leaderboard_to_markdown

        entries = [
            {
                "rank": 1,
                "display_name": "Test",
                "arch": "qwen2",
                "precision": "nf4",
                "precision_display": "NF4",
                "poros_vram_mb": 11000.0,
                "vram_save_vs_resident_pct": 74.0,
                "vram_save_vs_unsloth_pct": 45.0,
                "slowdown_vs_resident_x": 2.1,
                "bitwise_exact": True,
                "status_tag": "VALIDATED",
            },
        ]
        md = leaderboard_to_markdown(entries)
        assert "Test" in md
        assert "VALIDATED" in md

    def test_leaderboard_json_output(self):
        from poros.report.leaderboard import leaderboard_to_json

        entries = [{"rank": 1, "display_name": "Test"}]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = leaderboard_to_json(entries, f"{tmpdir}/lb.json")
            assert path.exists()
            loaded = json.loads(path.read_text())
            assert loaded[0]["display_name"] == "Test"


class TestArchRegistry:
    """Architecture registry tests (pure-python, no torch needed for basic checks)."""

    def test_validated_archs(self):
        from poros.core.arch_registry import validated_arch_types

        validated = validated_arch_types()
        assert "qwen2" in validated
        assert "qwen3" in validated
        assert "llama" not in validated
        assert "mistral" not in validated

    def test_blockwise_arch_check_off(self):
        from poros.core.arch_registry import blockwise_arch_check

        result = blockwise_arch_check("gemma", mode="off")
        assert result is False

    def test_blockwise_arch_check_error(self):
        from poros.core.arch_registry import blockwise_arch_check
        from poros.core.errors import PorosArchNotValidatedError

        with pytest.raises(PorosArchNotValidatedError):
            blockwise_arch_check("gemma", mode="error")

    def test_blockwise_arch_check_valid(self):
        from poros.core.arch_registry import blockwise_arch_check

        assert blockwise_arch_check("qwen2", mode="error") is True

    def test_root_and_layer_paths_defined(self):
        from poros.core.arch_registry import LAYER_PATHS, ROOT_CANDIDATES

        assert len(ROOT_CANDIDATES) >= 5
        assert "" in ROOT_CANDIDATES
        assert "model" in ROOT_CANDIDATES
        assert "language_model" in ROOT_CANDIDATES
        assert len(LAYER_PATHS) >= 4
        assert "model.layers" in LAYER_PATHS

    def test_discover_layers_across_families(self):
        """Architecture auto-detection must find the decoder stack across
        every supported layout (the 'works among all domains' guarantee).

        Pure-python: a layer container only needs ``__len__``/indexing, so
        stub namespaces with a list of fake layers exercise the real search.
        """
        from types import SimpleNamespace

        from poros.core.arch_registry import discover_layers

        ns = SimpleNamespace
        fake_layers = ["L0", "L1", "L2", "L3"]

        cases = {
            # family                         # stub model              # expected path tail
            "qwen/llama (model.layers)": (ns(model=ns(layers=fake_layers)), "model.layers"),
            "gpt-neox/pythia": (ns(gpt_neox=ns(layers=fake_layers)), "gpt_neox.layers"),
            "gpt-2/gpt-j": (ns(transformer=ns(h=fake_layers)), "transformer.h"),
            "opt": (ns(model=ns(decoder=ns(layers=fake_layers))), "model.decoder.layers"),
            "vlm-nested": (ns(language_model=ns(layers=fake_layers)), "language_model.layers"),
            # PEFT wraps base as base_model.model.<base>; Qwen base -> .model.layers
            "peft-wrapped": (
                ns(base_model=ns(model=ns(model=ns(layers=fake_layers)))),
                "model.layers",
            ),
        }
        for family, (model, expected_tail) in cases.items():
            layers, path = discover_layers(model)
            assert list(layers) == fake_layers, f"{family}: wrong container"
            assert path.endswith(expected_tail), f"{family}: path={path!r}"


class TestMatrixExpansion:
    """Bench matrix expansion tests."""

    def test_basic_expansion(self):
        from poros.bench.matrix import expand_matrix
        from poros.bench.specs import BenchmarkSpec, ModelSpec

        spec = BenchmarkSpec(
            spec_id="test",
            models=[
                ModelSpec(model_id="m1", display_name="M1"),
                ModelSpec(model_id="m2", display_name="M2"),
            ],
            methods=["poros", "resident"],
            precisions=["nf4"],
        )
        cells = expand_matrix(spec)
        assert len(cells) == 4  # 2 models x 2 methods x 1 precision


@pytest.mark.usefixtures("cpu_only")
class TestDeviceSampler:
    """DeviceSampler works on CPU (no torch needed)."""

    def test_cpu_sample(self):
        from poros.bench.device import DeviceSampler

        sampler = DeviceSampler()
        s = sampler.sample()
        assert s["allocated_mb"] == 0.0
        assert "timestamp" in s

    def test_gpu_info_cpu(self):
        from poros.bench.device import DeviceSampler

        info = DeviceSampler.gpu_info()
        assert info["gpu_name"] == "cpu"
        assert info["vram_total_mb"] == 0.0
