"""The plan's estimate of block-major's paths (poros.plan.estimate.path_extra) errs high: on top of
the sequential loop's measured peaks it is at least what each path measured on the RTX 4090
(docs/validation/block-major-dropout/)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from poros.cli.schemas import PlanGpu
from poros.core.config import PorosConfig
from tests.benchmark_logs import benchmark_log, needs_logs

# config.json's sizes of the models the RTX 4090 receipts trained (Qwen2.5, from the Hub)
QWEN25 = {
    "Qwen/Qwen2.5-0.5B": dict(hidden_size=896, num_hidden_layers=24, num_attention_heads=14,
                              num_key_value_heads=2, intermediate_size=4864),
    "Qwen/Qwen2.5-7B": dict(hidden_size=3584, num_hidden_layers=28, num_attention_heads=28,
                            num_key_value_heads=4, intermediate_size=18944),
    "Qwen/Qwen2.5-32B": dict(hidden_size=5120, num_hidden_layers=64, num_attention_heads=40,
                             num_key_value_heads=8, intermediate_size=27648),
    "Qwen/Qwen2.5-72B": dict(hidden_size=8192, num_hidden_layers=80, num_attention_heads=64,
                             num_key_value_heads=8, intermediate_size=29568),
}
RTX4090 = PlanGpu(name="NVIDIA GeForce RTX 4090", total_gib=23.99, free_gib=None, source="spec")
RECEIPTS = "docs/validation/block-major-dropout"


def _run(path: Path) -> tuple[PorosConfig, dict]:
    said = json.loads((path.parent / f"{path.name}.config.resolved.json").read_text())
    cfg = PorosConfig(**{k: v for k, v in said.items() if k in PorosConfig.model_fields})
    return cfg, json.loads((path.parent / f"{path.name}.receipt.json").read_text())


GB = 1e9 / 2**30


@needs_logs
@pytest.mark.parametrize("seq, other", [
    ("receipts/m7_seq", "receipts/m7_eff"), ("receipts/m7_seq", "receipts/m7_fast"),
    ("receipts/m7_seq", "receipts/m7_eff_b1"),
    ("receipts/m7_seq_2k", "receipts/m7_eff_2k"), ("receipts/m7_seq_2k", "receipts/m7_fast_2k"),
    ("receipts/m7_seq_2k", "receipts/m7_eff_b1_2k"),
    ("receipts/m32_seq", "receipts/m32_eff"), ("receipts/m32_seq", "receipts/m32_fast"),
    ("receipts/m32_seq", "receipts/m32_eff_b1"),
    ("rtx4090_72b/q72_seq_b4", "rtx4090_72b/q72_fast_b4"),
    ("rtx4090_72b/q72_seq_b4", "rtx4090_72b/q72_effc0_b4"),
    ("rtx4090_72b/q72_seq_b1", "rtx4090_72b/q72_effc0_b1"),
])
def test_the_path_estimate_is_never_below_what_the_rtx4090_measured(seq, other):
    """path_extra on top of the sequential loop's measured peaks is at least the path's own
    measured peaks, allocated, reserved and host RAM (docs/validation/block-major-dropout/)."""
    from poros.plan.estimate import path_extra

    base = _run(benchmark_log(f"{RECEIPTS}/{seq}"))
    measured = _run(benchmark_log(f"{RECEIPTS}/{other}"))
    cfg, got = measured
    extra = path_extra(cfg, QWEN25[cfg.model_name_or_path], RTX4090, "dense")
    assert extra is not None
    seq_said = base[1]
    for key, add in (("peak_vram_allocated_gb", extra.vram), ("peak_vram_reserved_gb", extra.vram),
                     ("peak_host_ram_gb", extra.host)):
        assert seq_said[key] * GB + add >= got[key] * GB, (key, seq_said[key], add, got[key])
