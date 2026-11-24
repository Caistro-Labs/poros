"""`poros plan`: numbers only from receipts, each labeled with where it came from."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import typer

from poros.cli.schemas import Figure, PlanGpu
from poros.core.config import PorosConfig
from poros.plan.make import make_plan
from tests import benchmark_logs
from tests.conftest import checkout_env

ROOT = Path(__file__).resolve().parent.parent
# These read the benchmark logs (docs/validation/, which the plan's cells name) and
# scripts/build_plan_cells.py: a wheel's test run has neither beside it.
needs_checkout = pytest.mark.skipif(
    not (ROOT / "scripts/build_plan_cells.py").is_file() or not benchmark_logs.available(),
    reason="needs scripts/build_plan_cells.py (a source checkout) and " + benchmark_logs.REASON)


def _missing_logs(*paths: str) -> list[str]:
    """Which of these logs the benchmark logs at hand lack (none without the logs at all)."""
    base = benchmark_logs.root()
    return [] if base is None else [p for p in paths if not (base / p).is_file()]


def _plan_source_logs() -> list[str]:
    """Every log docs/validation's plan_sources.yaml files name, the run settings' as well as the
    figures'."""
    import yaml

    base, found = benchmark_logs.root(), set()

    def walk(node):
        if isinstance(node, dict):
            if isinstance(node.get("receipt"), str):
                found.add(node["receipt"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for rel in ("docs/validation/plan_sources.yaml",
                "docs/validation/metal/m4_16gb/plan_sources.yaml",
                "docs/validation/recipes/plan_sources.yaml"):
        if base is not None and (base / rel).is_file():
            walk(yaml.safe_load((base / rel).read_text()))
    return sorted(found)


# The public benchmark logs hold the log behind every figure of the cells, not every log their
# sources read for a run's settings: rebuilding the table needs a checkout that has them all.
needs_every_plan_log = pytest.mark.skipif(
    bool(_missing_logs(*_plan_source_logs())),
    reason="rebuilding the cells needs every log plan_sources.yaml names; these benchmark logs "
           "hold the logs behind the cells' figures only")
MOE = "Qwen/Qwen3-30B-A3B-Instruct-2507"
# What the dense receipts ran.
DENSE_RECEIPTS = {"lora_dropout": 0.0, "nf4_load_strategy": "gpu",
                  "path": "sequential", "fused_compute": False}
# The MoE receipts' loop: sequential (a MoE run on CUDA takes block-major accumulation by default:
# tests/core/test_block_major_default.py), and their expert compute: reference (a MoE run on CUDA
# takes the fused compute by default: poros.core.config.moe_compute).
SEQUENTIAL = {"path": "sequential", "moe_expert_compute": "reference"}
RTX3090 = PlanGpu(name="NVIDIA GeForce RTX 3090", total_gib=24.0, free_gib=23.5, source="torch")


@pytest.fixture(autouse=True)
def _no_hub(monkeypatch):
    """The gate reads config.json; these tests never reach the Hub for it."""
    monkeypatch.setattr("poros.plan.configs.read_model_config", lambda model, revision=None: None)


def _cfg(model=MOE, **kw):
    base = {"model_name_or_path": model, "dataset": "synthetic", "max_seq_length": 512}
    return PorosConfig(**{**base, **kw})


def _plan(cfg, entry="moe", gpu=RTX3090):
    return make_plan(cfg, model_type="qwen3_moe", support="validated", entry=entry, gpu=gpu)


@needs_checkout
@needs_every_plan_log
def test_committed_cells_match_the_receipts():
    proc = subprocess.run([sys.executable, str(ROOT / "scripts/build_plan_cells.py"), "--check"],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_no_cell_holds_a_host_path():
    """cells.json ships in the wheel: a run host's directory layout (/home/ubuntu/...) must never
    reach it."""
    import re

    from poros.plan.cells import CELLS

    looks_like_path = re.compile(r"^(?:/|~|[A-Za-z]:[\\/])|(?:^|[\s\"'(=])/(?:home|Users|root|"
                                 r"run/media|workspace|data|mnt|tmp|Volumes|private|var)/")

    def strings(x):
        if isinstance(x, dict):
            for k, v in x.items():
                yield from strings(k)
                yield from strings(v)
        elif isinstance(x, list):
            for v in x:
                yield from strings(v)
        elif isinstance(x, str):
            yield x

    for cell in CELLS:
        for s in strings({k: v for k, v in cell.items() if k != "fields"}):
            assert not looks_like_path.search(s), (cell["id"], s)
        for field in cell["fields"].values():  # receipt#ptr: a repo-relative path and a pointer
            path, _, _ = field["receipt"].partition("#")
            assert not looks_like_path.search(path), (cell["id"], path)


@needs_checkout
def test_every_cell_value_cites_a_committed_receipt():
    from poros.plan.cells import CELLS

    for cell in CELLS:
        for field in cell["fields"].values():
            path, ptr = field["receipt"].split("#")
            assert benchmark_logs.benchmark_log(path).is_file() and ptr.startswith("/"), field


def test_same_config_same_gpu_is_measured_and_fits_on_its_reserved_peak():
    # the receipts' loop: a CUDA run of one micro-batch a step takes block-major unless told
    p = _plan(_cfg(max_steps=20, gradient_accumulation_steps=1, **SEQUENTIAL))
    assert p.peak_vram_allocated.source == p.peak_vram_reserved.source == "measured"
    assert p.peak_vram_reserved.receipt.startswith("docs/validation/moe/3090/table6")
    assert p.s_per_step.source == "measured" and p.total_time.source == "estimated"
    assert p.fits is True and "reserved peak" in p.fit_basis


def test_speed_needs_the_same_accumulation_but_memory_does_not():
    p = _plan(_cfg(gradient_accumulation_steps=8, **SEQUENTIAL))
    assert p.s_per_step.source == "unknown"
    assert p.peak_vram_allocated.source == "measured"


def test_another_gpu_estimates_memory_with_its_margin_and_never_speed():
    """No receipt of the 32B on an L40S: the same config on another card, plus the 6% a peak
    moved across cards, is at least what every card measured (11.02 GiB at most)."""
    l40s = PlanGpu(name="NVIDIA L40S", total_gib=44.4, free_gib=44.0, source="torch")
    p = _plan(_cfg("Qwen/Qwen2.5-32B", max_steps=10, **DENSE_RECEIPTS), entry="dense", gpu=l40s)
    assert p.peak_vram_allocated.source == "estimated"
    assert p.peak_vram_allocated.value == round(10.96 * 1.06, 3)  # PRO 6000's, plus the margin
    assert p.peak_vram_allocated.value >= 11.023  # the highest card's measure of it
    assert "RTX PRO 6000" in p.peak_vram_allocated.method
    assert p.peak_vram_reserved.source == p.s_per_step.source == "unknown"


def test_settings_no_receipt_ran_are_unknown_not_measured():
    p = _plan(_cfg("Qwen/Qwen2.5-7B", lora_bias="all", **DENSE_RECEIPTS), entry="dense")
    assert p.peak_vram_allocated.source == "unknown" and p.fits is None
    assert "lora_bias" in p.peak_vram_allocated.why
    # rank 8: none ran it, but a rank-16 run holds more: an upper bound, said as an estimate
    ranked = _plan(_cfg("Qwen/Qwen2.5-7B", lora_rank=8), entry="dense")
    assert ranked.peak_vram_allocated.source == "estimated"
    assert "peak" in ranked.peak_vram_allocated.method
    # the default (block-major efficient, dropout 0.05, 512 tokens) ran on an RTX 4090
    default = make_plan(_cfg("Qwen/Qwen2.5-7B"), model_type="qwen2", support="validated",
                        entry="dense", gpu=RTX3090)
    assert (default.path, default.peak_vram_allocated.source) == ("efficient", "estimated")
    assert "RTX 4090" in default.peak_vram_allocated.method


def test_the_startup_self_check_moves_only_the_load_phase(monkeypatch):
    monkeypatch.delenv("POROS_STARTUP_SELFCHECK", raising=False)
    rtx5090 = PlanGpu(name="NVIDIA GeForce RTX 5090", total_gib=31.84, free_gib=31.0,
                      source="torch")
    off = _plan(_cfg("Qwen/Qwen2.5-32B", **DENSE_RECEIPTS), entry="dense", gpu=rtx5090)
    assert off.peak_vram_allocated.source == "measured" and off.load_time.source == "unknown"
    on = _plan(_cfg("Qwen/Qwen2.5-32B", startup_parity_selfcheck="always", **DENSE_RECEIPTS),
               entry="dense", gpu=rtx5090)
    assert on.load_time.source == "measured"


def test_the_fused_default_plans_from_peft_module_receipts_as_an_estimate():
    """A dense NF4 CUDA run takes the fused compute unless fused_compute: false. A receipt of
    PEFT's modules plans the fused default only as an estimate, scaled by the largest fused /
    PEFT ratio measured in pairs, rounded up (7B sequential 1x8 on an RTX 4090: 1.263)."""
    from poros.plan.receipts import Request

    # a named card, planned away from it: its machine has Triton (a plan for this machine's own
    # GPU checks; this one may have none)
    rtx5090 = PlanGpu(name="NVIDIA GeForce RTX 5090", total_gib=31.84, free_gib=31.0,
                      source="spec")
    peft = _plan(_cfg("Qwen/Qwen2.5-32B", **DENSE_RECEIPTS), entry="dense", gpu=rtx5090)
    fused = _plan(_cfg("Qwen/Qwen2.5-32B", **{**DENSE_RECEIPTS, "fused_compute": None}),
                  entry="dense", gpu=rtx5090)
    assert peft.peak_vram_allocated.source == "measured"
    assert fused.peak_vram_allocated.source == "estimated"
    assert fused.peak_vram_allocated.value >= peft.peak_vram_allocated.value * 1.263
    assert "PEFT modules as the fused compute" in fused.peak_vram_allocated.method
    cfg = _cfg("Qwen/Qwen2.5-32B", **{**DENSE_RECEIPTS, "fused_compute": None})
    assert Request.of(cfg, "dense").fused_compute is True
    assert Request.of(cfg, "dense", cuda=False).fused_compute is False  # a Mac: never fused


def test_the_load_path_is_keyed():
    rtx5090 = PlanGpu(name="NVIDIA GeForce RTX 5090", total_gib=31.84, free_gib=31.0,
                      source="torch")
    streamed = _plan(_cfg("Qwen/Qwen2.5-32B", lora_dropout=0.0, path="sequential",
                          fused_compute=False), entry="dense", gpu=rtx5090)
    assert streamed.peak_vram_allocated.source == "measured"
    assert "v32b_rss_refix" in streamed.peak_vram_allocated.receipt


@needs_checkout
def test_the_teardown_gb_is_the_summary_gib():
    from poros.plan.cells import CELLS

    cell = next(c for c in CELLS if c["id"] == "qwen2.5-32b-rtx5090-synthetic")
    summary = json.loads(benchmark_logs.benchmark_log(
        cell["fields"]["s_per_step"]["receipt"].split("#")[0]).read_text())
    assert round(cell["fields"]["peak_vram_allocated"]["value"], 2) == summary["peak_vram_gb"]


def test_allocated_peak_over_free_memory_does_not_fit():
    small = PlanGpu(name="NVIDIA GeForce RTX 3090", total_gib=24.0, free_gib=4.0, source="torch")
    p = _plan(_cfg(**SEQUENTIAL), gpu=small)
    assert p.fits is False and "> 4.0 GiB free" in p.fit_basis


def test_an_unmeasured_model_is_unknown_everywhere_and_never_refused():
    p = _plan(_cfg("org/never-measured"), entry="dense")
    figures = [p.peak_vram_allocated, p.peak_vram_reserved, p.peak_host_ram, p.s_per_step]
    assert all(f.source == "unknown" and f.value is None and f.why for f in figures)
    assert p.fits is None


@pytest.mark.parametrize("kwargs", [
    {"source": "measured"},
    {"source": "estimated", "receipt": "r#/x"},
    {"source": "unknown", "value": 1.0, "why": "w"},
])
def test_a_figure_cannot_hide_where_it_came_from(kwargs):
    with pytest.raises(ValueError):
        Figure(**{"value": 1.0, "unit": "GiB", **kwargs})


def test_train_refuses_a_predicted_oom_before_the_download(monkeypatch):
    from poros.cli.commands import train as cli_commands_train
    from poros.core.card import detect_gpu  # noqa: F401  (the name the gate imports)

    small = PlanGpu(name="NVIDIA GeForce RTX 3090", total_gib=24.0, free_gib=4.0, source="torch")
    monkeypatch.setattr("poros.core.card.detect_gpu", lambda name=None: small)
    with pytest.raises(typer.Exit) as exc:
        cli_commands_train._plan_gate(_cfg(**SEQUENTIAL), "qwen3_moe", force=False)
    assert exc.value.exit_code == 3
    cli_commands_train._plan_gate(_cfg(**SEQUENTIAL), "qwen3_moe", force=True)  # --force proceeds
    # the default, block-major, which no receipt measured and no config.json sizes: no verdict
    cli_commands_train._plan_gate(_cfg(), "qwen3_moe", force=False)


def test_poros_train_runs_the_gate_before_loading_anything(monkeypatch, tmp_path,
                                                           recipe_configs_cached):
    pytest.importorskip("torch")
    from typer.testing import CliRunner

    from poros.cli import main
    from poros.preflight import arch as preflight_arch
    from poros.train.trainer import PorosTrainer

    small = PlanGpu(name="NVIDIA GeForce RTX 3090", total_gib=24.0, free_gib=4.0, source="torch")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    # the card poros.core.card.check reads (the plan's RTX 3090) and a torch built for it
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=None: (8, 6))
    monkeypatch.setattr("torch.cuda.get_device_name", lambda device=None: small.name)
    monkeypatch.setattr("torch.cuda.get_arch_list", lambda: ["sm_80", "sm_86"])
    monkeypatch.setattr("poros.core.card.detect_gpu", lambda name=None: small)
    monkeypatch.setattr(preflight_arch, "resolve_model_type", lambda m: ("qwen3_moe", ""))
    monkeypatch.setattr(PorosTrainer, "from_pretrained", lambda *a, **k: pytest.fail("loaded"))
    cfg = tmp_path / "run.yaml"
    cfg.write_text(f"model_name_or_path: {MOE}\ndataset: synthetic\nmax_seq_length: 512\n"
                   f"output_dir: {tmp_path / 'out'}\npath: sequential\n"
                   "moe_expert_compute: reference\n")  # the receipts' (SEQUENTIAL): any machine
    res = CliRunner().invoke(main.app, ["train", str(cfg), "--json"])
    assert res.exit_code == 3 and "E_OOM_PREDICTED" in res.output


@pytest.mark.parametrize("cap, name, arch_list, code, exit_code", [
    ((7, 5), "Tesla T4", ["sm_75", "sm_80"], "E_GPU_UNSUPPORTED", 4),  # no bf16: not supported
    ((12, 0), "NVIDIA GeForce RTX 5090", ["sm_80", "sm_86"], "E_STACK_UNSUPPORTED", 5),  # env
])
def test_poros_train_refuses_a_card_with_the_contracts_exit_code(monkeypatch, tmp_path, cap,
                                                                  name, arch_list, code,
                                                                  exit_code, recipe_configs_cached):
    """cli-contract.md: a GPU Poros does not support is exit 4, a torch build without kernels for
    the card an environment problem, exit 5; neither is a config to edit (exit 2)."""
    pytest.importorskip("torch")
    from typer.testing import CliRunner

    from poros.cli import main
    from poros.preflight import arch as preflight_arch
    from poros.train.trainer import PorosTrainer

    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=None: cap)
    monkeypatch.setattr("torch.cuda.get_device_name", lambda device=None: name)
    monkeypatch.setattr("torch.cuda.get_arch_list", lambda: list(arch_list))
    monkeypatch.setattr("poros.core.card.detect_gpu", lambda name=None: pytest.fail("planned"))
    monkeypatch.setattr(preflight_arch, "resolve_model_type", lambda m: ("qwen2", ""))
    monkeypatch.setattr(PorosTrainer, "from_pretrained", lambda *a, **k: pytest.fail("loaded"))
    cfg = tmp_path / "run.yaml"
    cfg.write_text("model_name_or_path: Qwen/Qwen2.5-0.5B\ndataset: synthetic\n"
                   f"output_dir: {tmp_path / 'out'}\n")
    res = CliRunner().invoke(main.app, ["train", str(cfg), "--json"])
    assert res.exit_code == exit_code and f'"code":"{code}"' in res.output, res.output


def _poros(*args, cwd):
    env = checkout_env({**os.environ, "HF_HUB_OFFLINE": "1"})
    env.pop("POROS_OUTPUT", None)
    return subprocess.run([sys.executable, "-P", "-m", "poros.cli.main", *args],
                          capture_output=True, text=True, cwd=cwd, env=env)


def test_the_plan_works_offline_from_the_receipts(tmp_path, monkeypatch):
    """The check a run makes, with the Hub off: a catalog MoE model plans from its receipts, a
    model no catalog names plans with no entry, and a card no list names is refused."""
    from poros.cli.context import Context
    from poros.cli.verbs import shared
    from poros.core.refusal import Refusal

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    ctx = Context(cwd=tmp_path, gpu_name="RTX 3090", fixed_cached=frozenset())
    p = shared.plan_for(MOE, ctx, seq_len=512, path_settings={"path": "sequential"})
    assert p.entry == "moe" and p.fits is True
    assert shared.plan_for("org/not-cached", ctx, model_type="unknown").support == "unsupported"
    with pytest.raises(Refusal):
        shared.gpu_name("RTX 9999")


# Qwen/Qwen3-30B-A3B-Instruct-2507's config.json, the fields the estimate reads.
QWEN3_30B = {"model_type": "qwen3_moe", "hidden_size": 2048, "num_hidden_layers": 48,
             "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
             "vocab_size": 151936, "tie_word_embeddings": False, "intermediate_size": 6144,
             "moe_intermediate_size": 768, "num_experts": 128, "num_experts_per_tok": 8,
             "decoder_sparse_step": 1, "mlp_only_layers": []}


def _moe_plan(host_free_gib, **kw):
    return make_plan(_cfg(**kw), model_type="qwen3_moe", support="validated", entry="moe",
                     gpu=RTX3090, model_config=QWEN3_30B, host_free_gib=host_free_gib)


@needs_checkout
def test_the_moe_host_ram_estimate_reproduces_the_30b_a3b_load():
    from poros.plan.estimate import moe_host_ram

    f = moe_host_ram(QWEN3_30B, "bfloat16")
    path, _ = f.receipt.split("#")
    measured = json.loads(benchmark_logs.benchmark_log(path).read_text())["full_streamed_load"][
        "peak_host_rss_bytes"]
    assert f.source == "estimated" and measured / 2**30 <= f.value <= 1.05 * measured / 2**30
    assert moe_host_ram({"model_type": "qwen3_moe"}, "bfloat16").source == "unknown"


def test_the_moe_host_ram_estimates_round_up(monkeypatch):
    """The JSON's figure is a bound: rounded to 0.1 GiB upward, never to nearest (a 12.04 GiB
    estimate read 12.0, below what it bounds)."""
    import math

    from poros.plan import estimate as E
    from poros.plan import estimate as plan_estimate

    monkeypatch.setattr(E, "_PROCESS_GIB", 0.0)
    for store in ("ram", "ssd"):
        exact = []
        real = E._ceil1
        monkeypatch.setattr(E, "_ceil1", lambda v: exact.append(v) or real(v))
        f = plan_estimate.moe_host_ram(QWEN3_30B, "bfloat16", store=store)
        monkeypatch.setattr(E, "_ceil1", real)
        assert f.value >= exact[0] and f.value == math.ceil(round(exact[0] * 10, 6)) / 10
    assert plan_estimate._ceil1(12.04) == 12.1 and plan_estimate._ceil1(12.0) == 12.0


def test_moe_host_ram_is_measured_when_a_receipt_matches_and_estimated_otherwise():
    measured = _moe_plan(46.6, **SEQUENTIAL)
    assert measured.peak_host_ram.source == "measured" and measured.host_fits is True
    longer = _plan(_cfg(max_seq_length=1024))  # no receipt, no config.json
    assert longer.peak_host_ram.source == "unknown" and longer.host_fits is None
    estimated = _moe_plan(14.0, max_seq_length=1024, **SEQUENTIAL)
    assert estimated.peak_host_ram.source == "estimated" and estimated.host_fits is False
    assert "23.0 GiB > 14.0 GiB available" in estimated.host_fit_basis


def test_dense_host_ram_is_checked_only_against_a_receipt():
    dense = make_plan(_cfg("org/never-measured"), model_type="qwen2", support="validated",
                      entry="dense", gpu=RTX3090, model_config={"model_type": "qwen2"},
                      host_free_gib=1.0)
    assert dense.peak_host_ram.source == "unknown" and dense.host_fits is None
    streamed = _cfg("Qwen/Qwen2.5-7B", lora_dropout=0.0, startup_parity_selfcheck="always",
               path="sequential")
    measured = make_plan(streamed, model_type="qwen2", support="validated", entry="dense",
                         gpu=RTX3090, host_free_gib=8.0)
    assert measured.peak_host_ram.value == 12.06 and measured.host_fits is False


def test_the_host_ram_refusal_names_the_numbers_and_a_way_out():
    from poros.core.errors import EXIT_CODES
    from poros.preflight.fit import host_ram_error

    err = host_ram_error(_moe_plan(14.0, max_seq_length=1024, **SEQUENTIAL))
    assert err.code == "E_HOST_OOM_PREDICTED" and EXIT_CODES[err.code] == 3
    # what is available here, measured; the plan's 23.0 GiB stays in its reason, never said
    assert "14.0 GiB available" in err.info.message and "23.0" not in err.info.message
    assert "--force" in err.info.remediation and "check_host_ram=False" in err.info.remediation
    assert host_ram_error(_moe_plan(None)) is None  # RAM not read: never a refusal


GIB = 2**30


def _host_ram_free(monkeypatch, root, files):
    """host_ram_free() with /proc and /sys/fs/cgroup read from ``root``: Linux's reader, on any
    host (a Mac's reads vm_stat)."""
    import pathlib

    import poros.core.env as env
    from tests.conftest import as_linux

    as_linux(monkeypatch, env)
    for name, text in files.items():
        (root / name[1:]).parent.mkdir(parents=True, exist_ok=True)
        (root / name[1:]).write_text(text)
    monkeypatch.setattr(env, "Path", lambda p: pathlib.Path(f"{root}{p}"))
    return env.host_ram_free()


@pytest.mark.parametrize("files", [
    {"/proc/self/cgroup": "0::/job\n",
     "/sys/fs/cgroup/job/memory.max": f"{50 * GIB}\n",
     "/sys/fs/cgroup/job/memory.current": f"{10 * GIB}\n",
     "/sys/fs/cgroup/job/memory.stat": f"anon 1\nactive_file {GIB}\ninactive_file {GIB}\n",
     "/sys/fs/cgroup/memory.max": "max\n"},
    # v1, with the limit on the parent cgroup
    {"/proc/self/cgroup": "4:memory:/docker/x\n0::/\n",
     "/sys/fs/cgroup/memory/docker/x/memory.limit_in_bytes": f"{2**63 - 4096}\n",
     "/sys/fs/cgroup/memory/docker/x/memory.usage_in_bytes": "0\n",
     "/sys/fs/cgroup/memory/docker/x/memory.stat": "total_inactive_file 0\n",
     "/sys/fs/cgroup/memory/docker/memory.limit_in_bytes": f"{50 * GIB}\n",
     "/sys/fs/cgroup/memory/docker/memory.usage_in_bytes": f"{10 * GIB}\n",
     "/sys/fs/cgroup/memory/docker/memory.stat":
         f"total_active_file {GIB}\ntotal_inactive_file {GIB}\n"},
])
def test_free_host_ram_is_capped_by_the_containers_memory_limit(monkeypatch, tmp_path, files):
    meminfo = f"MemTotal: {1024 * 2**20} kB\nMemAvailable: {900 * 2**20} kB\n"
    free = _host_ram_free(monkeypatch, tmp_path, {"/proc/meminfo": meminfo, **files})
    assert free == 42.0  # 50 GiB limit - 10 GiB used + 2 GiB page cache, not 900


def test_free_host_ram_without_a_limit_is_memavailable_and_none_off_linux(monkeypatch, tmp_path):
    files = {"/proc/meminfo": f"MemAvailable: {12 * 2**20} kB\n", "/proc/self/cgroup": "0::/\n"}
    assert _host_ram_free(monkeypatch, tmp_path, files) == 12.0
    assert _host_ram_free(monkeypatch, tmp_path / "none", {}) is None


def test_train_refuses_short_host_ram_before_the_download(monkeypatch):
    from poros.cli.commands import train as cli_commands_train

    monkeypatch.setattr("poros.core.card.detect_gpu", lambda name=None: RTX3090)
    monkeypatch.setattr("poros.plan.configs.read_model_config", lambda m, r=None: QWEN3_30B)
    monkeypatch.setattr("poros.core.env.host_ram_free", lambda: 14.0)
    cfg = _cfg(max_seq_length=1024)
    with pytest.raises(typer.Exit) as exc:
        cli_commands_train._plan_gate(cfg, "qwen3_moe", force=False)
    assert exc.value.exit_code == 3
    cli_commands_train._plan_gate(cfg, "qwen3_moe", force=True)  # --force proceeds
    monkeypatch.setattr("poros.core.env.host_ram_free", lambda: 46.6)
    cli_commands_train._plan_gate(cfg, "qwen3_moe", force=False)


@pytest.mark.parametrize("kw", [SEQUENTIAL, {}], ids=["sequential", "block-major"])
def test_a_moe_run_short_of_host_ram_is_refused_on_either_loop(monkeypatch, kw):
    """By default a MoE run takes block-major's efficient path, whose block inputs wait in host
    RAM: the plan adds them to the estimate and the trainer's check refuses it before the
    download, as it does the sequential loop's."""
    from poros.core.errors import PorosHostRamError
    from poros.preflight.fit import host_ram_error

    cfg = _cfg(max_seq_length=1024, **kw)
    p = make_plan(cfg, model_type="qwen3_moe", support="validated", entry="moe", gpu=None,
                  model_config=QWEN3_30B, host_free_gib=14.0)
    assert p.path == ("sequential" if kw else "efficient")
    assert p.host_fits is False and isinstance(host_ram_error(p), PorosHostRamError)
    if not kw:  # what efficient adds: every block input of an update, in host RAM
        seq = make_plan(_cfg(max_seq_length=1024, **SEQUENTIAL),
                        model_type="qwen3_moe", support="validated", entry="moe", gpu=None,
                        model_config=QWEN3_30B, host_free_gib=14.0)
        assert p.peak_host_ram.value > seq.peak_host_ram.value


def test_a_config_run_short_of_host_ram_is_refused_before_the_download(monkeypatch, tmp_path):

    from poros.preflight import arch as preflight_arch

    monkeypatch.setattr("poros.core.card.detect_gpu", lambda name=None: RTX3090)
    monkeypatch.setattr("poros.plan.configs.read_model_config", lambda m, r=None: QWEN3_30B)
    monkeypatch.setattr("poros.plan.configs.download_size", lambda cfg: Figure(
        value=None, unit="GiB", source="unknown", why="test"))
    monkeypatch.setattr("poros.core.env.host_ram_free", lambda: 14.0)
    monkeypatch.setattr(preflight_arch, "resolve_model_type", lambda m: ("qwen3_moe", ""))
    cfg = tmp_path / "run.yaml"
    cfg.write_text(f"model_name_or_path: {MOE}\ndataset: synthetic\nmax_seq_length: 1024\n")
    from poros.cli.sheets.fit import unfit_refusal
    from poros.preflight import fit as preflight_fit

    run = PorosConfig(model_name_or_path=MOE, dataset="synthetic", max_seq_length=1024)
    run.apply_cuda_defaults(QWEN3_30B)
    found = preflight_fit.check(run, preflight_fit.planner_here("qwen3_moe", QWEN3_30B))
    said = unfit_refusal(found, host=True)  # `poros train <config>`'s refusal, in words
    assert found.host is not None and said.code == "E_HOST_OOM_PREDICTED"
    # the decision in words: the host RAM measured here, never the plan's need
    assert re.findall(r"([\d.]+) GiB", said.reason) == ["14.0"], said.reason


def test_block_major_without_config_json_shows_no_memory_figure_and_decides_nothing():
    """No receipt of the path at these settings and no config.json to size what it adds: it holds
    more than the sequential loop's measured peaks, so no memory figure, no fit drawn, and the
    plan says so (the 72B's block-major receipts ran block 1 with its tables offloaded:
    config.json sizes the rest)."""
    from poros.plan.receipts import PATH_NOTE_UNSIZED as PATH_NOTE

    cfg = _cfg("Qwen/Qwen2.5-72B", lora_dropout=0.0, path="efficient")
    p = _plan(cfg, entry="dense")
    seq = _plan(_cfg("Qwen/Qwen2.5-72B", lora_dropout=0.05, path="sequential"), entry="dense")
    assert seq.peak_vram_allocated.value is not None
    assert p.peak_vram_allocated.source == p.peak_vram_reserved.source == "unknown"
    assert "config.json" in p.peak_vram_allocated.why
    assert p.fits is None and p.bound is None and p.fit_basis.startswith(
        "no measured run of this model on the efficient path is at these settings")
    assert PATH_NOTE.format(path="efficient") in p.warnings
    assert p.peak_host_ram.source == "unknown" and p.host_fits is None


def test_a_decided_path_short_of_host_ram_gives_way_to_one_that_fits():
    """The efficient path a MoE run would take keeps its block inputs in host RAM; where that
    alone tips it over, the run takes fast (inputs on the GPU) instead and says why. A path you
    set is yours: refused, not changed."""
    from poros.preflight.fit import host_ram_error, path_that_fits_host

    def plan(c, free):
        return make_plan(c, model_type="qwen3_moe", support="validated", entry="moe",
                         gpu=RTX3090, model_config=QWEN3_30B, host_free_gib=free)

    eff = plan(_cfg(max_seq_length=1024), None)
    fast = plan(_cfg(max_seq_length=1024, path="fast"), None)
    assert (eff.path, fast.path) == ("efficient", "fast")
    free = round((eff.peak_host_ram.value + fast.peak_host_ram.value) / 2, 2)
    cfg = _cfg(max_seq_length=1024)
    assert host_ram_error(plan(cfg, free)) is not None  # efficient: short
    path, why = path_that_fits_host(cfg, lambda c: plan(c, free))
    assert path == "fast" and why == ("the check turned down efficient: its block inputs are "
                                      "past the host RAM available")
    cfg.decide_path(path, why)
    assert (cfg.path, cfg.path_reason) == ("fast", why)
    assert host_ram_error(plan(cfg, free)) is None
    assert path_that_fits_host(_cfg(max_seq_length=1024, path="efficient"),
                               lambda c: plan(c, free)) is None  # yours
    from poros.preflight.fit import open_paths

    mine = _cfg(max_seq_length=1024, path="efficient")  # set by hand: another path may help
    opened = open_paths(mine, lambda c: plan(c, free), plan(mine, free))
    assert "fast" in opened
    err = host_ram_error(plan(mine, free), None, opened)
    for alt in opened:  # each path the check does not refuse, by name, and only those
        assert f"path: {alt}" in err.info.remediation
    assert open_paths(cfg, lambda c: plan(c, free), plan(cfg, free)) == ()  # decided: tried


# A plan for a named card (`--gpu`), away from it: the run's machine is assumed to have Triton.
RTX4090 = PlanGpu(name="NVIDIA GeForce RTX 4090", total_gib=24.0, free_gib=23.5, source="spec")
# What the 4090 fused receipts ran (docs/validation/moe/4090/receipts/q30k_def_*): every MoE default
# but the head-to-head's batch, accumulation and dropout.
FUSED_4090 = {"per_device_batch_size": 8, "gradient_accumulation_steps": 1, "lora_dropout": 0.0}


@pytest.mark.parametrize("path,alloc", [("efficient", 7.488), ("fast", 17.574)])
def test_a_default_moe_config_plans_from_the_fused_receipts(path, alloc):
    """moe_expert_compute unset plans the fused compute (a CUDA MoE run takes it): the fused
    receipts of its path measure it; pinned to the reference compute, they do not. (fast grows
    its expert cache to the card: measured at its run's free VRAM, 22.95 GiB.)"""
    gpu = RTX4090.model_copy(update={"free_gib": 22.95}) if path == "fast" else RTX4090
    fused = make_plan(_cfg("Qwen/Qwen3-30B-A3B", path=path, **FUSED_4090),
                      model_type="qwen3_moe", support="validated", entry="moe", gpu=gpu)
    assert fused.peak_vram_allocated.source == "measured"
    assert fused.peak_vram_allocated.value == alloc
    assert "moe/4090/receipts/q30k_def_" in fused.peak_vram_allocated.receipt
    ref = make_plan(_cfg("Qwen/Qwen3-30B-A3B", path=path, moe_expert_compute="reference",
                         **FUSED_4090),
                    model_type="qwen3_moe", support="validated", entry="moe", gpu=RTX4090)
    assert ref.peak_vram_allocated.source != "measured"


def test_a_local_gpu_without_triton_plans_the_reference_compute(monkeypatch):
    """The trainer takes the reference compute where Triton is not installed; a plan for this
    machine's GPU keys the same (m7): no fused receipt stands for it."""
    import importlib.util

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec",
                        lambda name, *a: None if name == "triton" else real(name, *a))
    local = RTX4090.model_copy(update={"source": "torch"})
    plan = make_plan(_cfg("Qwen/Qwen3-30B-A3B", path="fast", **FUSED_4090), model_type="qwen3_moe",
                     support="validated", entry="moe", gpu=local)
    assert plan.peak_vram_allocated.source != "measured"


def test_a_moe_fast_receipt_is_its_own_card_s(monkeypatch):
    """MoE fast sizes its expert cache to what the GPU leaves free (poros.train.compute): the
    4090 receipt's VRAM plans that card only. On a 16 GB card it neither carries over as an
    estimate nor bounds the fit from below (the cache would grow less there). On its own card it
    is measured at its run's free VRAM (22.95 GiB) only: tests/test_plan_upper.py."""
    cfg = _cfg("Qwen/Qwen3-30B-A3B", path="fast", **FUSED_4090)
    here = make_plan(cfg, model_type="qwen3_moe", support="validated", entry="moe",
                     gpu=RTX4090.model_copy(update={"free_gib": 22.95}))
    assert here.peak_vram_allocated.source == "measured"
    small = RTX4090.model_copy(update={"name": "RTX 4060 Ti 16GB", "total_gib": 16.0,
                                       "free_gib": 15.5})
    there = make_plan(cfg, model_type="qwen3_moe", support="validated", entry="moe", gpu=small)
    assert there.peak_vram_allocated.receipt is None or \
        "fused-fast" not in str(there.peak_vram_allocated.receipt)
    assert there.bound is None or there.bound.fits is not False


def test_the_plan_refuses_a_card_poros_train_refuses(monkeypatch, tmp_path):
    """A card with no native bf16: the plan says the run cannot fit it, as `poros train` refuses
    it (E_GPU_UNSUPPORTED)."""
    from poros.cli.context import Context
    from poros.cli.verbs import shared

    t4 = PlanGpu(name="Tesla T4", total_gib=14.58, free_gib=14.5, source="nvidia-smi",
                 capability=(7, 5))
    p = shared.plan_for("Qwen/Qwen2.5-0.5B", Context(cwd=tmp_path, fixed_gpu=t4,
                                                     fixed_cached=frozenset()))
    assert p.fits is False and p.fit_basis.startswith("refused on this card")
