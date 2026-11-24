"""The report pipeline: Poros rows only, rendered honestly (markdown and HTML).

With several methods per model (poros, resident, unsloth), each model renders
once, as its Poros row; the baseline rows are never table rows.
"""

import json

import pytest

from poros.report.leaderboard import build_leaderboard
from poros.report.markdown import render_markdown_string


def _multi_method_rows(model="m-7b", hw="RTX 3090"):
    """One model x three methods, the shape the runner actually emits."""
    common = dict(
        model_id=model, display_name=model, precision="nf4", arch="qwen2",
        status="ok", hardware=hw, steps=200, seq_len=512,
        resident_vram_mb=33966.0, unsloth_vram_mb=22592.0,
        poros_vram_mb=10184.0,
    )
    poros = dict(common, method="poros", s_per_step=1.893, n_passes=3,
                 s_per_step_stddev=0.005, vram_peak_mb=10184.0,
                 vram_save_vs_resident_pct=70.0, vram_save_vs_unsloth_pct=54.9,
                 slowdown_vs_resident_x=2.42)
    resident = dict(common, method="resident", s_per_step=0.783,
                    vram_peak_mb=33966.0, vram_save_vs_resident_pct=0.0,
                    vram_save_vs_unsloth_pct=-50.3, slowdown_vs_resident_x=1.0)
    unsloth = dict(common, method="unsloth", s_per_step=0.94,
                   vram_peak_mb=22592.0, vram_save_vs_resident_pct=33.5,
                   vram_save_vs_unsloth_pct=0.0, slowdown_vs_resident_x=1.2)
    return [poros, resident, unsloth]


def _write_jsonl(tmp_path, rows, name="a.jsonl"):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


class TestPorosRowsOnly:
    def test_each_model_renders_exactly_once(self, tmp_path):
        md = render_markdown_string(_write_jsonl(tmp_path, _multi_method_rows()))
        vram = [ln for ln in md.splitlines() if ln.startswith("| m-7b")]
        assert len([ln for ln in vram if "10,184" in ln]) >= 1
        # the baseline rows must NOT appear as table rows
        assert not any("-50.3%" in ln for ln in md.splitlines())
        assert md.count("| m-7b") == sum(
            1 for s in ("## VRAM", "## Speed", "## Parity") if s in md
        )

    def test_speed_row_is_the_poros_measurement(self, tmp_path):
        md = render_markdown_string(_write_jsonl(tmp_path, _multi_method_rows()))
        speed = [ln for ln in md.splitlines() if ln.startswith("| m-7b") and "x" in ln]
        assert any("1.893" in ln and "2.42x" in ln for ln in speed)
        assert not any("0.783" in ln for ln in speed), "resident baseline leaked"

    def test_leaderboard_one_row_per_model(self, tmp_path):
        entries = build_leaderboard(
            _write_jsonl(tmp_path, _multi_method_rows()),
            include_not_bitwise=True,
        )
        assert len(entries) == 1
        assert entries[0]["poros_vram_mb"] == 10184.0

    def test_legacy_rows_without_method_still_render(self, tmp_path):
        rows = _multi_method_rows()[:1]
        del rows[0]["method"]  # pre-method-field artifact = a poros run
        md = render_markdown_string(_write_jsonl(tmp_path, rows))
        assert "| m-7b" in md


class TestHonestRendering:
    def test_zero_sstep_dashes_instead_of_fabricating(self, tmp_path):
        rows = _multi_method_rows()[:1]
        rows[0]["s_per_step"] = 0.0  # legacy-loader schema default
        rows[0]["slowdown_vs_resident_x"] = 2.67
        md = render_markdown_string(_write_jsonl(tmp_path, rows))
        assert "0.000" not in md

    def test_single_method_run_still_publishes_speed(self, tmp_path):
        rows = _multi_method_rows()[:1]
        rows[0]["slowdown_vs_resident_x"] = None  # no baseline ran
        md = render_markdown_string(_write_jsonl(tmp_path, rows))
        assert "## Speed" in md and "1.893" in md

    def test_mixed_hardware_gets_a_column(self, tmp_path):
        a = _multi_method_rows("m-7b", "NVIDIA A100")[:1]
        b = _multi_method_rows("m-7b", "RTX 5090")[:1]
        md = render_markdown_string(_write_jsonl(tmp_path, a + b))
        assert "Hardware" in md
        assert "A100" in md and "5090" in md

    def test_vram_peak_mb_primary_field_is_read(self, tmp_path):
        rows = _multi_method_rows()[:1]
        rows[0]["poros_vram_mb"] = None  # only the schema's primary field set
        md = render_markdown_string(_write_jsonl(tmp_path, rows))
        assert "10,184" in md

    def test_none_quant_and_mixed_steps_do_not_crash(self, tmp_path):
        rows = _multi_method_rows()[:1] + _multi_method_rows("m-13b")[:1]
        rows[0]["precision"] = None
        rows[0]["quant"] = None
        del rows[1]["steps"]
        md = render_markdown_string(_write_jsonl(tmp_path, rows))
        assert "m-13b" in md


class TestHtmlRendererPorosRowsOnly:
    def test_html_hero_and_rows_are_poros_cells(self, tmp_path):
        """html.py had the same missing method filter as markdown.py: on a
        multi-method file the hero pick scanned ALL ok artifacts, so with a
        baseline row first the hero stat card showed the resident row's
        0.0% saved instead of the poros row's real number."""
        pytest.importorskip("jinja2")
        from poros.bench.artifacts import RunArtifact
        from poros.report.html import generate_html_report

        common = dict(
            model_id="Qwen/Qwen2.5-32B", display_name="Qwen2.5-32B",
            precision="nf4", arch="qwen2", status="ok",
            hardware="RTX 3090", steps=200, seq_len=512,
            vram_resident_mb=33966.0, vram_unsloth_mb=22592.0,
            vram_poros_mb=10184.0,
        )
        resident = RunArtifact(
            **common, method="resident", s_per_step=0.783,
            vram_peak_mb=33966.0, vram_saved_vs_resident_pct=0.0,
            vram_saved_vs_unsloth_pct=-50.3, slowdown_vs_resident_x=1.0,
        )
        poros_row = RunArtifact(
            **common, method="poros", s_per_step=1.893,
            vram_peak_mb=10184.0, vram_saved_vs_resident_pct=70.0,
            vram_saved_vs_unsloth_pct=54.9, slowdown_vs_resident_x=2.42,
        )
        out = tmp_path / "r.html"
        # resident first: the hero must still pick the Poros row
        generate_html_report([resident, poros_row], str(out))
        html = out.read_text()
        assert "70.0%" in html, "hero must carry the poros row's savings"
        assert "-50.3%" not in html, "baseline row rendered as a Poros cell"


def test_an_artifact_with_no_hardware_is_on_an_unknown_card_not_a_b300(tmp_path):
    """The HTML report says unknown for an artifact that recorded no hardware; it names no
    card."""
    pytest.importorskip("jinja2")
    from poros.bench.artifacts import RunArtifact
    from poros.report.html import generate_html_report

    row = RunArtifact(model_id="Qwen/Qwen2.5-32B", display_name="Qwen2.5-32B", precision="nf4",
                      arch="qwen2", status="ok", hardware="", steps=200, seq_len=512,
                      method="poros", s_per_step=1.0, vram_peak_mb=10184.0)
    out = tmp_path / "r.html"
    generate_html_report([row], str(out))
    html = out.read_text()
    assert "B300" not in html and "unknown" in html


@pytest.mark.cuda_host  # poros bench does not run on a Mac in this release
def test_bench_compare_rows_name_their_model(tmp_path, monkeypatch):
    """compare's row label names the model, so two models' rows of one method read apart."""
    from typer.testing import CliRunner

    from poros.bench.artifacts import RunArtifact
    from poros.cli.main import app

    monkeypatch.setenv("COLUMNS", "200")
    for d, speed in (("a", 1.0), ("b", 0.9)):
        (tmp_path / d).mkdir()
        rows = [RunArtifact(model_id=m, display_name=m.split("/")[1], precision="nf4",
                            arch="qwen2", status="ok", hardware="RTX 4090", steps=10,
                            seq_len=512, method="poros", s_per_step=speed, vram_peak_mb=9000.0)
                for m in ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-32B")]
        (tmp_path / d / "runs.jsonl").write_text(
            "".join(r.model_dump_json() + "\n" for r in rows))
    out = CliRunner().invoke(app, ["bench", "compare", str(tmp_path / "a"), str(tmp_path / "b")])
    assert out.exit_code == 0, out.output
    assert "poros / Qwen2.5-7B / nf4" in out.output and "poros / Qwen2.5-32B / nf4" in out.output
