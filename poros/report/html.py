"""Self-contained HTML report generator for Poros benchmarks.

Generates a single portable HTML file with ALL CSS, JS, and font references
inlined. No external network requests. The report includes:

    Summary -> Hardware -> Methods -> VRAM chart (HERO) -> Speed chart ->
    Loss/parity curves -> Parity table (0.00e+00) -> OOM table ->
    Raw config -> Artifact links.

Design: warm-dark theme (#0C0C0B ink background, #F4F1EC cream text, #FF6B35 accent),
halftone texture, Geist Mono, staggered bar animations with
prefers-reduced-motion support throughout.

This module must work WITHOUT torch installed. It depends only on jinja2,
pydantic (for RunArtifact), and the Python standard library.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

try:
    from jinja2 import Environment
    from markupsafe import Markup
except ImportError:
    Environment = None  # type: ignore[assignment, misc]
    Markup = None  # type: ignore[assignment, misc]

from poros.bench.artifacts import RunArtifact, load_grid_pro_row

# ---------------------------------------------------------------------------
# CSS — read from the assets file at import time for embedding
# ---------------------------------------------------------------------------

_CSS_PATH = Path(__file__).parent / "assets" / "poros.css"


def _read_css() -> str:
    """Read the CSS file. Falls back to empty string if not found."""
    if _CSS_PATH.exists():
        return _CSS_PATH.read_text(encoding="utf-8")
    return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_mb(mb: float) -> str:
    """Format megabytes as 'XX.X GB'."""
    return f"{mb / 1024:.1f}"


def _fmt_pct(pct: float) -> str:
    """Format percentage."""
    return f"{pct:.1f}"


def _fmt_parity(val: Optional[float]) -> str:
    """Format parity value with scientific notation.

    ``None`` means parity was NOT measured in this cell (parity comes from the
    dedicated GPU parity test, not a throughput cell) -- render an em dash
    rather than a fabricated 0.00e+00.
    """
    if val is None:
        return "—"
    if val == 0.0:
        return "0.00e+00"
    return f"{val:.2e}"


def _generate_run_id(artifacts: Sequence[RunArtifact]) -> str:
    """Generate a short run ID from the artifact contents."""
    h = hashlib.sha256()
    for a in artifacts:
        h.update(f"{a.arch}:{a.precision}:{a.display_name}".encode())
    return h.hexdigest()[:12]


def _generate_loss_curve_svg(
    steps: int,
    *,
    width: int = 800,
    height: int = 200,
) -> str:
    """Generate an illustrative loss curve SVG depicting parity overlap.

    This is a schematic diagram (not plotted from measured step data).
    Both lines follow the same synthetic curve to visually convey that
    the resident and poros loss traces are identical at NF4 parity.
    """
    if steps <= 0:
        steps = 200
    n_points = min(steps, 80)
    points_resident = []
    points_poros = []

    margin_left = 50
    margin_right = 20
    margin_top = 20
    margin_bottom = 30
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    for i in range(n_points):
        t = i / max(n_points - 1, 1)
        x = margin_left + t * plot_w
        # smooth exponential decay from ~2.8 to ~1.1
        loss = 2.8 * math.exp(-2.5 * t) + 1.1
        y = margin_top + (1 - (loss - 1.0) / 2.0) * plot_h
        y = max(margin_top, min(margin_top + plot_h, y))
        points_resident.append(f"{x:.1f},{y:.1f}")
        points_poros.append(f"{x:.1f},{y:.1f}")

    resident_d = "M " + " L ".join(points_resident)
    poros_d = "M " + " L ".join(points_poros)

    # Calculate path length for stroke-dashoffset animation
    path_length = sum(
        math.sqrt(
            (float(points_resident[i + 1].split(",")[0]) - float(points_resident[i].split(",")[0])) ** 2
            + (float(points_resident[i + 1].split(",")[1]) - float(points_resident[i].split(",")[1])) ** 2
        )
        for i in range(len(points_resident) - 1)
    )
    path_len_int = int(path_length) + 10

    return f"""<svg class="parity-curve" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Illustrative schematic: identical loss curves for Resident and Poros at NF4 parity">
  <!-- grid lines -->
  <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#2A2622" stroke-width="1"/>
  <line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#2A2622" stroke-width="1"/>
  <!-- y-axis labels -->
  <text x="{margin_left - 8}" y="{margin_top + 4}" fill="#6B6560" font-family="'Geist Mono', monospace" font-size="10" text-anchor="end">3.0</text>
  <text x="{margin_left - 8}" y="{margin_top + plot_h // 2 + 4}" fill="#6B6560" font-family="'Geist Mono', monospace" font-size="10" text-anchor="end">2.0</text>
  <text x="{margin_left - 8}" y="{margin_top + plot_h + 4}" fill="#6B6560" font-family="'Geist Mono', monospace" font-size="10" text-anchor="end">1.0</text>
  <!-- x-axis label -->
  <text x="{margin_left + plot_w // 2}" y="{height - 4}" fill="#6B6560" font-family="'Geist Mono', monospace" font-size="10" text-anchor="middle">Step</text>
  <!-- resident line (drawn first, slightly thicker, below poros) -->
  <path d="{resident_d}" class="parity-curve__line parity-curve__line--resident" data-animate="true" style="--line-length:{path_len_int};--delay:400ms" stroke-dasharray="{path_len_int}" stroke-dashoffset="{path_len_int}" stroke-width="3"/>
  <!-- poros line (on top, thinner, identical path = perfect overlap) -->
  <path d="{poros_d}" class="parity-curve__line parity-curve__line--poros" data-animate="true" style="--line-length:{path_len_int};--delay:700ms" stroke-dasharray="{path_len_int}" stroke-dashoffset="{path_len_int}" stroke-width="1.5"/>
</svg>"""


# ---------------------------------------------------------------------------
# JINJA2 TEMPLATE — the full self-contained HTML
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{ title }}</title>
<meta name="description" content="Poros benchmark report: blockwise QLoRA residency results with bitwise parity verification.">

<!-- Fonts: prefers locally-installed Geist Mono (data) and Instrument Sans
     (display) when present, otherwise falls through to the robust system stacks
     in --font-mono / --font-display. No @font-face with empty local() sources
     (those error in the console and never render), and no external fetch:
     the report is fully self-contained and renders cleanly on any machine. -->
<style>
/* === INLINED CSS === */
{{ css }}
</style>
</head>
<body>

<div class="poros-report" role="main">

  <!-- ====== HEADER ====== -->
  <header class="poros-header" style="--card-delay:0ms">
    <div class="poros-header__badge">Poros Benchmark Report</div>
    <h1>{{ title }}</h1>
    <p class="poros-header__subtitle">{{ subtitle }}</p>
  </header>

  <!-- ====== SUMMARY STATS ====== -->
  <section class="poros-card" style="--card-delay:100ms" aria-label="Summary statistics">
    <div class="poros-card__title">Summary</div>
    <div class="stat-row">
      <div class="stat-cell">
        <div class="stat-cell__label">Model</div>
        <div class="stat-cell__value">{{ hero.display_name }}</div>
      </div>
      <div class="stat-cell">
        <div class="stat-cell__label">Precision</div>
        <div class="stat-cell__value">{{ hero.precision | upper }}</div>
      </div>
      <div class="stat-cell">
        <div class="stat-cell__label">Poros VRAM</div>
        <div class="stat-cell__value stat-cell__value--accent">{{ hero.poros_gb }}<span class="stat-cell__unit">GB</span></div>
      </div>
      <div class="stat-cell">
        <div class="stat-cell__label">VRAM Saved</div>
        <div class="stat-cell__value stat-cell__value--accent">{{ hero.saved_pct }}%</div>
      </div>
      <div class="stat-cell">
        <div class="stat-cell__label">Parity</div>
        <div class="stat-cell__value stat-cell__value--success">{{ hero.parity_str }}</div>
      </div>
      <div class="stat-cell">
        <div class="stat-cell__label">Slowdown</div>
        <div class="stat-cell__value">{{ hero.slowdown }}x</div>
      </div>
    </div>
  </section>

  <!-- ====== HERO: VRAM CHART (the headline chart, kept above the fold) ====== -->
  <section class="poros-card poros-card--hero" style="--card-delay:150ms" aria-label="VRAM comparison chart" id="vram-hero">
    <div class="poros-card__title">Peak VRAM</div>

    <div class="vram-chart" id="vram-chart">
      {% for bar in vram_bars %}
      <div class="vram-chart__row">
        <div class="vram-chart__label">{{ bar.label }}</div>
        <div class="vram-chart__track">
          <div class="vram-bar vram-bar--{{ bar.css_class }}"
               style="--target-width:{{ bar.pct }}%;--delay:{{ bar.delay }}ms"
               data-animate="false"
               role="meter"
               aria-label="{{ bar.label }}: {{ bar.gb }} GB"
               aria-valuenow="{{ bar.mb }}"
               aria-valuemin="0"
               aria-valuemax="{{ vram_max_mb }}">
          </div>
          {% if bar.is_poros and budget_lines %}
          {% for bl in budget_lines %}
          <div class="budget-line{% if bl.gb == 24 %} budget-line--24{% endif %}"
               style="left:{{ bl.pct }}%">
            <span class="budget-line__label">{{ bl.gb }} GB</span>
          </div>
          {% endfor %}
          {% endif %}
        </div>
        <div class="vram-chart__value{% if bar.is_poros %} vram-chart__value--accent{% endif %}">{{ bar.gb }} GB</div>
      </div>
      {% endfor %}
    </div>

    <div class="saved-hero">
      <!-- Static fallback = the real value; the count-up JS animates from 0 over it. -->
      <div class="saved-hero__pct" id="saved-pct" data-target="{{ hero.saved_pct }}">{{ hero.saved_pct }}%</div>
      <div class="saved-hero__label">VRAM saved vs resident</div>
    </div>
    <div class="saved-hero__detail">
      {{ hero.poros_gb }} GB Poros vs {{ hero.resident_gb }} GB resident
      {% if hero.unsloth_gb %} / {{ hero.unsloth_gb }} GB Unsloth ({{ hero.saved_vs_unsloth_pct }}% less){% endif %}
    </div>
  </section>

  <!-- ====== HARDWARE ====== -->
  <section class="poros-card" style="--card-delay:200ms" aria-label="Hardware information">
    <div class="poros-card__title">Hardware</div>
    <div class="stat-row">
      <div class="stat-cell">
        <div class="stat-cell__label">GPU</div>
        <div class="stat-cell__value" style="font-size:var(--font-size-md)">{{ hardware.gpu }}</div>
      </div>
      {% if hardware.vram_gb %}
      <div class="stat-cell">
        <div class="stat-cell__label">VRAM</div>
        <div class="stat-cell__value">{{ hardware.vram_gb }}<span class="stat-cell__unit">GB</span></div>
      </div>
      {% endif %}
      <div class="stat-cell">
        <div class="stat-cell__label">Steps</div>
        <div class="stat-cell__value tabnum">{{ hero.steps }}</div>
      </div>
      <div class="stat-cell">
        <div class="stat-cell__label">Seq Length</div>
        <div class="stat-cell__value tabnum">{{ hero.seq_len }}</div>
      </div>
    </div>
  </section>

  <!-- ====== METHODS ====== -->
  <section class="poros-card" style="--card-delay:300ms" aria-label="Methods compared">
    <div class="poros-card__title">Methods</div>
    <div class="legend">
      <div class="legend__item">
        <div class="legend__swatch" style="background:var(--bar-resident)"></div>
        <span>Resident QLoRA</span>
      </div>
      <div class="legend__item">
        <div class="legend__swatch" style="background:var(--bar-unsloth)"></div>
        <span>Unsloth</span>
      </div>
      <div class="legend__item">
        <div class="legend__swatch" style="background:var(--bar-poros)"></div>
        <span>Poros</span>
      </div>
    </div>
    <p>All methods share one configuration: same model, LoRA rank/alpha, sequence length, batch size, gradient accumulation, and steps, each measured in a fresh process. Data pipelines differ per method (baselines train on a fixed synthetic batch), so treat s/step as indicative; VRAM is insensitive to this.</p>
  </section>

  <!-- ====== SPEED CHART ====== -->
  <section class="poros-card" style="--card-delay:500ms" aria-label="Speed comparison">
    <div class="poros-card__title">Throughput</div>
    <p>Poros trades speed for memory. The {{ hero.slowdown }}x slowdown is the cost of streaming frozen blocks across PCIe.</p>

    {% for bar in speed_bars %}
    <div class="speed-chart__row">
      <div class="vram-chart__label">{{ bar.label }}</div>
      <div class="vram-chart__track">
        <div class="speed-bar speed-bar--{{ bar.css_class }}"
             style="--target-width:{{ bar.pct }}%;--delay:{{ bar.delay }}ms"
             data-animate="false"
             role="meter"
             aria-label="{{ bar.label }}: {{ bar.slowdown }}"
             aria-valuenow="{{ bar.slowdown_raw }}">
        </div>
      </div>
      <div class="vram-chart__value">{{ bar.slowdown }}</div>
    </div>
    {% endfor %}
  </section>

  <!-- ====== LOSS / PARITY CURVES ====== -->
  <section class="poros-card" style="--card-delay:600ms" aria-label="Loss curve parity">
    <div class="poros-card__title">Loss Curve Parity</div>
    <p>Illustrative schematic: at NF4, Poros and the resident reference produce bitwise-identical loss at every step (0.00e+00). The two curves overlap perfectly by construction. This diagram depicts that overlap; it is not plotted from a specific run's step-level data.</p>
    <div class="legend" style="margin-top:var(--space-md)">
      <div class="legend__item">
        <div class="legend__swatch" style="background:var(--bar-resident)"></div>
        <span>Resident</span>
      </div>
      <div class="legend__item">
        <div class="legend__swatch" style="background:var(--accent)"></div>
        <span>Poros (overlapping)</span>
      </div>
    </div>
    <div class="parity-curve-container">
      {{ loss_curve_svg }}
    </div>
  </section>

  <!-- ====== PARITY TABLE ====== -->
  <section class="poros-card" style="--card-delay:700ms" aria-label="Parity verification table">
    <div class="poros-card__title">Parity Verification</div>
    <div style="overflow-x:auto">
    <table class="parity-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Arch</th>
          <th>Precision</th>
          <th>Loss max|diff|</th>
          <th>Weight max|diff|</th>
          <th>Status</th>
          <th>Steps</th>
        </tr>
      </thead>
      <tbody>
        {% for row in parity_rows %}
        <tr>
          <td>{{ row.display_name }}</td>
          <td>{{ row.arch }}</td>
          <td>{{ row.precision | upper }}</td>
          <td class="tabnum">{{ row.loss_diff }}</td>
          <td class="tabnum">{{ row.weight_diff }}</td>
          <td>
            {% if row.bitwise %}
            <span class="parity-badge parity-badge--exact">BITWISE</span>
            {% elif row.status == 'oom' %}
            <span class="parity-badge parity-badge--oom">OOM</span>
            {% elif not row.parity_measured %}
            <span class="parity-badge parity-badge--na">NOT MEASURED</span>
            {% else %}
            <span class="parity-badge parity-badge--drift">DRIFT</span>
            {% endif %}
          </td>
          <td class="tabnum">{{ row.steps }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    </div>
  </section>

  <!-- ====== VRAM COMPARISON TABLE ====== -->
  <section class="poros-card" style="--card-delay:800ms" aria-label="VRAM comparison table">
    <div class="poros-card__title">VRAM Comparison</div>
    <div style="overflow-x:auto">
    <table class="parity-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Precision</th>
          <th>Resident</th>
          <th>Unsloth</th>
          <th>Poros</th>
          <th>vs Resident</th>
          <th>vs Unsloth</th>
          <th>Slowdown</th>
        </tr>
      </thead>
      <tbody>
        {% for row in vram_rows %}
        <tr>
          <td>{{ row.display_name }}</td>
          <td>{{ row.precision | upper }}</td>
          <td class="tabnum">{{ row.resident_gb }} GB</td>
          <td class="tabnum">{{ row.unsloth_gb }} GB</td>
          <td class="tabnum" style="color:var(--accent);font-weight:600">{{ row.poros_gb }} GB</td>
          <td class="tabnum">{{ row.saved_resident }}%</td>
          <td class="tabnum">{{ row.saved_unsloth }}%</td>
          <td class="tabnum">{{ row.slowdown }}x</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    </div>
  </section>

  {% if oom_rows %}
  <!-- ====== OOM TABLE ====== -->
  <section class="poros-card" style="--card-delay:900ms" aria-label="OOM status">
    <div class="poros-card__title">Configurations Not Run</div>
    <div style="overflow-x:auto">
    <table class="oom-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Precision</th>
          <th>Status</th>
          <th>Reason</th>
        </tr>
      </thead>
      <tbody>
        {% for row in oom_rows %}
        <tr>
          <td>{{ row.display_name }}</td>
          <td>{{ row.precision | upper }}</td>
          <td><span class="parity-badge parity-badge--oom">{{ row.status | upper }}</span></td>
          <td>{{ row.reason }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    </div>
  </section>
  {% endif %}

  <!-- ====== RAW CONFIG ====== -->
  <section class="poros-card" style="--card-delay:1000ms" aria-label="Run configuration">
    <div class="poros-card__title">Run Configuration</div>
    <div class="config-block">{{ config_json }}</div>
  </section>

  <!-- ====== FOOTER ====== -->
  <footer class="poros-footer">
    <p>Generated by <a href="https://github.com/Caistro-Labs/poros">Poros</a> v{{ poros_version }} on {{ generated_at }}.</p>
    <p>Source measurements come from the supplied run artifacts. Ratios and percentages are derived from those measurements; the loss-overlap figure is an illustrative schematic, not plotted step data.</p>
    <p style="margin-top:var(--space-sm)">Run ID: <code>{{ run_id }}</code></p>
  </footer>

</div>

<!-- ====== JAVASCRIPT: intersection observer for animations ====== -->
<script>
(function() {
  'use strict';

  // Check reduced-motion preference
  var prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // Animate bars on intersection
  function animateBars() {
    var bars = document.querySelectorAll('.vram-bar, .speed-bar');
    if (prefersReducedMotion) {
      bars.forEach(function(bar) {
        bar.style.width = bar.style.getPropertyValue('--target-width');
        bar.style.opacity = '1';
      });
      return;
    }

    var observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          entry.target.setAttribute('data-animate', 'true');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.2 });

    bars.forEach(function(bar) { observer.observe(bar); });
  }

  // Animate SVG curve draw-on
  function animateCurves() {
    var lines = document.querySelectorAll('.parity-curve__line');
    if (prefersReducedMotion) {
      lines.forEach(function(line) {
        line.style.strokeDashoffset = '0';
      });
      return;
    }

    var observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          var paths = entry.target.querySelectorAll('.parity-curve__line');
          paths.forEach(function(p) { p.setAttribute('data-animate', 'true'); });
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.3 });

    document.querySelectorAll('.parity-curve-container').forEach(function(c) {
      observer.observe(c);
    });
  }

  // Count-up animation for saved percentage
  function animateCountUp() {
    var el = document.getElementById('saved-pct');
    if (!el) return;
    var target = parseFloat(el.getAttribute('data-target') || '0');

    if (prefersReducedMotion) {
      el.textContent = target.toFixed(1) + '%';
      return;
    }

    var observer = new IntersectionObserver(function(entries) {
      entries.forEach(function(entry) {
        if (entry.isIntersecting) {
          observer.unobserve(entry.target);
          var start = 0;
          var duration = 1800;
          var startTime = null;

          function step(timestamp) {
            if (!startTime) startTime = timestamp;
            var elapsed = timestamp - startTime;
            var progress = Math.min(elapsed / duration, 1);
            // ease-out-quart
            var eased = 1 - Math.pow(1 - progress, 4);
            var current = start + (target - start) * eased;
            el.textContent = current.toFixed(1) + '%';
            if (progress < 1) {
              requestAnimationFrame(step);
            } else {
              el.textContent = target.toFixed(1) + '%';
            }
          }
          requestAnimationFrame(step);
        }
      });
    }, { threshold: 0.5 });

    observer.observe(el);
  }

  // Initialize on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
      animateBars();
      animateCurves();
      animateCountUp();
    });
  } else {
    animateBars();
    animateCurves();
    animateCountUp();
  }
})();
</script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _load_jsonl_raw(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load raw JSONL rows as dicts."""
    rows: List[Dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def generate_html_report(
    artifacts: List[RunArtifact],
    output_path: Union[str, Path],
    *,
    title: str = "Poros Benchmark Report",
    subtitle: str = "Fine-tune large models on small GPUs by streaming a frozen quantized base, matching a fully resident run bit for bit.",
    hardware_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate a self-contained HTML benchmark report.

    Args:
        artifacts: List of RunArtifact results to include.
        output_path: Path for the output HTML file.
        title: Report title.
        subtitle: Report subtitle.
        hardware_info: Optional hardware details dict with 'gpu', 'vram_gb'.

    Returns:
        Path to the generated HTML file.
    """
    if Environment is None:
        raise ImportError(
            "jinja2 is required for HTML report generation. "
            "Install it with: pip install poros-train[bench]"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Separate ok artifacts from blocked/oom. POROS cells only, same as the
    # markdown renderer: the runner copies the cross-method comparisons onto
    # every method's artifact, so without this filter resident/unsloth
    # baseline rows render as Poros cards -- and the hero stat itself could
    # be a baseline row (its vs-columns then read 0.0% / negative). Legacy
    # artifacts without a method field are poros runs.
    ok_artifacts = [
        a for a in artifacts
        if a.status == "ok" and (getattr(a, "method", None) or "poros") == "poros"
    ]
    blocked_artifacts = [a for a in artifacts if a.status not in ("ok",)]

    # Pick the hero artifact: prefer qlora4/nf4 qwen32b
    hero_artifact = None
    for a in ok_artifacts:
        if a.precision in ("nf4", "qlora4") and "32b" in a.display_name.lower():
            hero_artifact = a
            break
    if hero_artifact is None and ok_artifacts:
        hero_artifact = ok_artifacts[0]
    if hero_artifact is None:
        raise ValueError("No valid artifacts to generate report from.")

    # --- Build hero dict ---
    hero_poros_mb = hero_artifact.vram_poros_mb or hero_artifact.vram_peak_mb
    hero_resident_mb = hero_artifact.vram_resident_mb or 0.0
    hero_unsloth_mb = hero_artifact.vram_unsloth_mb
    hero_saved_pct = hero_artifact.vram_saved_vs_resident_pct or 0.0
    hero_saved_vs_unsloth = hero_artifact.vram_saved_vs_unsloth_pct

    hero = {
        "display_name": hero_artifact.display_name,
        "precision": hero_artifact.precision,
        "poros_gb": _fmt_mb(hero_poros_mb),
        "resident_gb": _fmt_mb(hero_resident_mb),
        "unsloth_gb": _fmt_mb(hero_unsloth_mb) if hero_unsloth_mb else None,
        "saved_pct": _fmt_pct(hero_saved_pct),
        "saved_vs_unsloth_pct": _fmt_pct(hero_saved_vs_unsloth) if hero_saved_vs_unsloth is not None else None,
        "parity_str": _fmt_parity(hero_artifact.parity_loss_max_diff),
        "slowdown": f"{hero_artifact.slowdown_vs_resident_x:.2f}" if hero_artifact.slowdown_vs_resident_x else "N/A",
        "steps": hero_artifact.steps,
        "seq_len": hero_artifact.seq_len,
    }

    # --- Hardware ---
    hw = hardware_info or {}
    hardware = {
        # what the artifact recorded; a report names no card it was not measured on
        "gpu": hw.get("gpu") or hero_artifact.hardware or "unknown",
        "vram_gb": hw.get("vram_gb", None),
    }

    # --- VRAM bars (hero chart) ---
    # Build bars for the hero artifact's model
    vram_entries = []
    if hero_resident_mb > 0:
        vram_entries.append(("Resident", hero_resident_mb, "resident", False))
    if hero_unsloth_mb and hero_unsloth_mb > 0:
        vram_entries.append(("Unsloth", hero_unsloth_mb, "unsloth", False))
    vram_entries.append(("Poros", hero_poros_mb, "poros", True))

    # Sort by VRAM descending (largest first), but poros always last
    non_poros = [(lbl, v, c, p) for lbl, v, c, p in vram_entries if not p]
    poros_entry = [(lbl, v, c, p) for lbl, v, c, p in vram_entries if p]
    non_poros.sort(key=lambda x: x[1], reverse=True)
    vram_entries_sorted = non_poros + poros_entry

    vram_max_mb = max(v for _, v, _, _ in vram_entries_sorted) * 1.1
    vram_bars = []
    for i, (label, mb, css_class, is_poros) in enumerate(vram_entries_sorted):
        pct = (mb / vram_max_mb) * 100 if vram_max_mb else 0.0
        vram_bars.append({
            "label": label,
            "mb": mb,
            "gb": _fmt_mb(mb),
            "pct": f"{pct:.1f}",
            "css_class": css_class,
            "is_poros": is_poros,
            "delay": 200 + i * 200,
        })

    # Budget lines (relative to the chart)
    budget_lines = []
    for gb in [16, 24]:
        mb = gb * 1024
        if mb < vram_max_mb:
            pct = (mb / vram_max_mb) * 100
            budget_lines.append({"gb": gb, "pct": f"{pct:.1f}"})

    # --- Speed bars ---
    speed_entries = []
    # Resident is 1.0x (the reference)
    if hero_artifact.slowdown_vs_resident_x:
        speed_entries.append(("Resident", 1.0, "resident"))
        if hero_artifact.slowdown_vs_unsloth_x and hero_unsloth_mb:
            # Unsloth is between resident and poros
            unsloth_vs_resident = hero_artifact.slowdown_vs_resident_x / hero_artifact.slowdown_vs_unsloth_x
            speed_entries.append(("Unsloth", unsloth_vs_resident, "unsloth"))
        speed_entries.append(("Poros", hero_artifact.slowdown_vs_resident_x, "poros"))

    speed_max = max((s for _, s, _ in speed_entries), default=1.0) * 1.1
    speed_bars = []
    for i, (label, slowdown_val, css_class) in enumerate(speed_entries):
        pct = (slowdown_val / speed_max) * 100
        speed_bars.append({
            "label": label,
            "slowdown": f"{slowdown_val:.2f}x",
            "slowdown_raw": slowdown_val,
            "pct": f"{pct:.1f}",
            "css_class": css_class,
            "delay": 200 + i * 150,
        })

    # --- Parity rows ---
    parity_rows = []
    for a in ok_artifacts:
        parity_rows.append({
            "display_name": a.display_name,
            "arch": a.arch,
            "precision": a.precision,
            "loss_diff": _fmt_parity(a.parity_loss_max_diff),
            "weight_diff": _fmt_parity(a.parity_weight_max_diff),
            "bitwise": a.bitwise_exact,
            # Speed/VRAM cells record no parity by design; render them as
            # "not measured", never as drift.
            "parity_measured": (
                a.bitwise_exact
                or a.parity_loss_max_diff is not None
                or a.parity_weight_max_diff is not None
            ),
            "status": a.status,
            "steps": a.steps,
        })

    # --- VRAM comparison rows ---
    vram_rows = []
    for a in ok_artifacts:
        poros_mb = a.vram_poros_mb or a.vram_peak_mb
        resident_mb = a.vram_resident_mb or 0.0
        unsloth_mb = a.vram_unsloth_mb or 0.0
        vram_rows.append({
            "display_name": a.display_name,
            "precision": a.precision,
            "resident_gb": _fmt_mb(resident_mb),
            "unsloth_gb": _fmt_mb(unsloth_mb),
            "poros_gb": _fmt_mb(poros_mb),
            "saved_resident": _fmt_pct(a.vram_saved_vs_resident_pct or 0.0),
            "saved_unsloth": _fmt_pct(a.vram_saved_vs_unsloth_pct or 0.0),
            "slowdown": f"{a.slowdown_vs_resident_x:.2f}" if a.slowdown_vs_resident_x else "N/A",
        })

    # --- OOM / blocked rows ---
    oom_rows = []
    for a in blocked_artifacts:
        oom_rows.append({
            "display_name": a.display_name,
            "precision": a.precision,
            "status": a.status,
            "reason": a.error_message or "Unknown",
        })

    # --- Loss curve SVG ---
    loss_curve_svg = _generate_loss_curve_svg(hero_artifact.steps)

    # --- Config JSON ---
    config_data = {
        "hero_model": hero_artifact.display_name,
        "arch": hero_artifact.arch,
        "precision": hero_artifact.precision,
        "steps": hero_artifact.steps,
        "seq_len": hero_artifact.seq_len,
        "hardware": hardware["gpu"],
        "total_artifacts": len(artifacts),
        "bitwise_exact_count": sum(1 for a in ok_artifacts if a.bitwise_exact),
    }
    config_json = json.dumps(config_data, indent=2)

    # --- Run ID & timestamp ---
    run_id = _generate_run_id(artifacts)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # --- Render ---
    # Autoescape everything: artifact fields come from JSONL the user may not
    # have produced, and unescaped values in HTML are stored script injection.
    # Only the two internally generated blobs (our CSS, our SVG) are Markup.
    css = _read_css()
    template = Environment(autoescape=True).from_string(_HTML_TEMPLATE)
    from poros import __version__

    html = template.render(
        title=title,
        subtitle=subtitle,
        css=Markup(css),
        poros_version=__version__,
        hero=hero,
        hardware=hardware,
        vram_bars=vram_bars,
        vram_max_mb=vram_max_mb,
        budget_lines=budget_lines,
        speed_bars=speed_bars,
        loss_curve_svg=Markup(loss_curve_svg),
        parity_rows=parity_rows,
        vram_rows=vram_rows,
        oom_rows=oom_rows,
        config_json=config_json,
        run_id=run_id,
        generated_at=generated_at,
    )

    output.write_text(html, encoding="utf-8")
    return output


def generate_report_from_jsonl(
    jsonl_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    title: str = "Poros Benchmark Report",
    subtitle: str = "Fine-tune large models on small GPUs by streaming a frozen quantized base, matching a fully resident run bit for bit.",
    hardware_info: Optional[Dict[str, Any]] = None,
) -> Path:
    """Generate an HTML report directly from a poros_grid_pro.jsonl file.

    This is the convenience entry point for the CLI and for generating
    the bundled showcase report.

    Args:
        jsonl_path: Path to the JSONL file with benchmark results.
        output_path: Path for the output HTML file.
        title: Report title.
        subtitle: Report subtitle.
        hardware_info: Optional hardware details.

    Returns:
        Path to the generated HTML file.
    """
    raw_rows = _load_jsonl_raw(jsonl_path)
    artifacts = []
    n_skipped = 0
    for row in raw_rows:
        try:
            artifacts.append(load_grid_pro_row(row))
        except Exception:
            n_skipped += 1

    if not artifacts:
        raise ValueError(f"No valid rows found in {jsonl_path}")
    if n_skipped:
        # A dropped row is a missing benchmark cell; a report that renders
        # without it reads as complete. Make the gap loud.
        import warnings

        warnings.warn(
            f"{n_skipped} row(s) in {jsonl_path} were not valid benchmark "
            "artifacts and were skipped; the report may be missing cells.",
            stacklevel=2,
        )

    return generate_html_report(
        artifacts,
        output_path,
        title=title,
        subtitle=subtitle,
        hardware_info=hardware_info,
    )


# ---------------------------------------------------------------------------
# CLI entry point: python -m poros.report.html <jsonl> <output>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m poros.report.html <jsonl_path> <output_path>")
        sys.exit(1)

    result = generate_report_from_jsonl(sys.argv[1], sys.argv[2])
    print(f"Report generated: {result}")
