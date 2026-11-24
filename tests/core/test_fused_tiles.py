"""The fused compute's tiles by rank (poros.core.fused_tiles), and what the plan takes from them.

Ranks up to 32 keep the configuration every receipt ran (their bits are pinned on the GPU); the
tiles depend on the rank alone. A plan for a named card takes the compute a run there takes at
the requested rank (fused, or PEFT's modules / the reference experts where the kernels do not fit
the card), and grows the peak with the rank's partial buffers and A products, never below.
"""
from __future__ import annotations

import pytest

from poros.core import card
from poros.core import fused_tiles as F
from poros.core.card import capability_of_name, shared_memory_limit


def test_ranks_up_to_32_keep_the_receipts_tiles():
    for R in (16, 32):
        t = F.tiles(R)
        assert t.out_fwd == t.swiglu_fwd == (64, 128, 4)
        assert t.out_bwd == (32, 128, 8, 2) and t.swiglu_bwd == (32, 64, 4, 1)
        assert t.rope_fwd == (64, 4) and t.rope_bwd == (32, 8, 2) and not t.chunk
    for rp in (16, 32, 64):
        assert F.moe_tiles(rp) == (64, 256, 8)


def test_larger_ranks_take_smaller_tiles_and_chunk_their_rows():
    assert F.tiles(64) != F.tiles(32) and F.tiles(64).chunk
    assert F.tiles(128) == F.tiles(256) == F.tiles(512) and F.tiles(128).chunk
    assert F.moe_tiles(128) == F.moe_tiles(256) == (32, 64, 4)
    assert [F.pad(r) for r in (1, 8, 16, 17, 33, 64, 65, 128, 200)] == \
        [16, 16, 16, 32, 64, 64, 128, 128, 256]


def test_row_chunks_are_whole_blocks_under_the_partial_budget():
    t = F.tiles(128)
    for M, n_tiles in ((4096, 1184), (8192, 1848), (100, 1184), (5000, 3)):
        rows = F.row_chunk(t, M, n_tiles, 128, 16)
        assert rows == M or (rows % 16 == 0 and n_tiles * rows * 128 <= F.PARTIAL_ELEMS)
    assert F.row_chunk(F.tiles(16), 4096, 1184, 16, 32) == 4096  # rank <= 32: one launch


def test_kernels_import_the_same_tiles():
    pytest.importorskip("triton")
    pytest.importorskip("torch")
    from poros.core.fused import kernels as K
    from poros.moe import kernels as MK

    assert (K.FWD_BM, K.FWD_BN, K.BWD_BM, K.BWD_BN, K.BWD_STAGES) == (64, 128, 32, 128, 2)
    assert K.tiles is F.tiles and K.splits is F.splits
    assert MK.tiles(64) == (MK.GEMM_BN, MK.WIDE_BN, MK.WIDE_WARPS)


@pytest.mark.parametrize("name, cap", [
    ("NVIDIA GeForce RTX 4090", (8, 9)), ("NVIDIA GeForce RTX 3090 Laptop GPU", (8, 6)),
    ("NVIDIA A10G", (8, 6)), ("NVIDIA A100-SXM4-80GB", (8, 0)), ("NVIDIA L4", (8, 9)),
    ("NVIDIA L40S", (8, 9)), ("H100 80GB HBM3", (9, 0)), ("Tesla T4", (7, 5)),
    ("Tesla V100-SXM2-16GB", (7, 0)), ("NVIDIA GeForce RTX 5090", (12, 0)),
    ("RTX PRO 6000 Blackwell Workstation Edition", (12, 0)), ("NVIDIA B300", (10, 3)),
    ("NVIDIA RTX A6000", (8, 6)), ("NVIDIA RTX 6000 Ada Generation", (8, 9)),
    ("NVIDIA GeForce GTX 1080", (6, 1)), ("Tesla P100-PCIE-16GB", (6, 0)),
    ("NVIDIA GeForce GTX 980", None), ("", None)])
def test_a_card_name_says_its_capability(name, cap):
    assert capability_of_name(name) == cap


@pytest.mark.parametrize("name, cap", [
    # Ada workstation cards: their model numbers share an RTX 20 / 40 / 50 series' digits
    ("NVIDIA RTX 2000 Ada Generation", (8, 9)), ("NVIDIA RTX 2000E Ada Generation", (8, 9)),
    ("NVIDIA RTX 4000 SFF Ada Generation", (8, 9)), ("NVIDIA RTX 4000 Ada Generation", (8, 9)),
    ("NVIDIA RTX 5000 Ada Generation", (8, 9)), ("NVIDIA RTX 5880 Ada Generation", (8, 9)),
    ("NVIDIA RTX 3500 Ada Generation Laptop GPU", (8, 9)),
    # Turing Quadro cards and the Turing / Volta prosumer ones: an RTX 40 / 50 series' digits
    ("Quadro RTX 4000", (7, 5)), ("Quadro RTX 5000", (7, 5)), ("Quadro RTX 6000", (7, 5)),
    ("Quadro RTX 8000", (7, 5)), ("NVIDIA TITAN RTX", (7, 5)), ("NVIDIA TITAN V", (7, 0)),
    # the series those digits belong to still read as theirs
    ("NVIDIA GeForce RTX 2080 Ti", (7, 5)), ("NVIDIA GeForce RTX 4060", (8, 9)),
    ("NVIDIA GeForce RTX 5070 Ti", (12, 0)), ("NVIDIA RTX PRO 4000 Blackwell", (12, 0))])
def test_a_workstation_name_is_not_read_as_the_series_its_digits_share(name, cap):
    assert capability_of_name(name) == cap


def test_a_plan_of_a_card_here_reads_its_capability_not_its_name():
    """The device's own capability decides a plan of a local card; the name only a named one."""
    from poros.cli.schemas import PlanGpu
    from poros.core.card import card_capability

    here = PlanGpu(name="NVIDIA GeForce RTX 4090", total_gib=23.5, free_gib=23.0,
                   source="nvidia-smi", capability=(12, 0))
    assert card_capability(here) == (12, 0)
    assert card_capability(here.model_copy(update={"capability": None})) == (8, 9)
    assert card_capability(PlanGpu(name="RTX 4090", total_gib=24.0, free_gib=None,
                                   source="spec")) == (8, 9)
    assert "capability" not in here.model_dump()  # not part of the plan's JSON


def test_the_moe_reason_on_a_card_without_bf16_experts_names_no_rank():
    why = card.fused_rank_reason((7, 5), 16, entry="moe")
    assert why is not None and "None" not in why and "do not run on sm_75" in why


def test_the_fit_table_follows_the_shared_memory():
    """Up to 128 the fused kernels, dense and MoE, fit every supported card (sm_80 and newer);
    above, only the large-shared-memory ones. (Pre-Ampere is not supported: sm_75 fits 64.)"""
    for cap in (c for c in card.FUSED_FIT if c >= (8, 0)):
        for hd in (None, 64, 128):
            assert card.fused_rank_reason(cap, 128, head_dim=hd) is None
        assert card.fused_rank_reason(cap, 128, entry="moe") is None
    assert card.fused_rank_reason((7, 5), 64, head_dim=128) is None
    assert card.fused_rank_reason((7, 5), 128) is not None
    assert card.fused_rank_reason((8, 9), 128, entry="moe") is None
    assert card.fused_rank_reason((7, 5), 16, entry="moe") is not None  # no bf16 experts there
    for cap in ((8, 6), (8, 9), (12, 0)):
        why = card.fused_rank_reason(cap, 256)
        assert why is not None and "rank 256" in why
    assert card.fused_rank_reason((9, 0), 256) is None
    # a capability the table does not list takes the closest listed one at most its memory
    assert card.fused_rank_reason((10, 3), 256) is None
    assert shared_memory_limit((8, 7)) >= shared_memory_limit((8, 6))


def test_partials_grow_with_rank_and_stay_bounded():
    """The fused compute's partials grow with the rank, and above 32 the row chunks bound them
    (whatever the rows: 16,384 rows at rank 32 hold 1.2 GiB in one launch)."""
    for M in (512, 2048, 4096, 16384):
        mid, h = 18944, 3584
        sizes = [F.dense_partial_bytes(r, M, mid, h) for r in (8, 16, 32, 64, 128, 256)]
        assert sizes[0] == sizes[1] <= sizes[2]  # one launch: the partials double with R
        assert max(sizes[3:]) < 700 * 2**20  # in row chunks: 64 / 256 MiB a buffer, whatever M
        bound = [F.dense_partial_bound(r, M, mid, h) for r in (8, 16, 32, 64, 128, 256)]
        assert all(b >= s for b, s in zip(bound, sizes))
        assert all(a <= b for a, b in zip(bound, bound[1:]))  # the plan's term never falls


def _plan_request(rank, gpu, model="Qwen/Qwen2.5-7B", **kw):
    from poros.core.config import PorosConfig
    from poros.plan import receipts as plan_receipts

    cfg = PorosConfig(model_name_or_path=model, dataset="synthetic", lora_rank=rank, **kw)
    return plan_receipts.Request.of(cfg, "dense", cap=capability_of_name(gpu), rope_head_dim=128)


@pytest.mark.parametrize("rank, gpu, fused", [
    (16, "RTX 4090", True), (64, "RTX 4090", True), (128, "RTX 4090", True),
    (256, "RTX 4090", False), (256, "H100 80GB HBM3", True), (128, "Tesla T4", False),
    (64, "Tesla T4", True), (256, None, True)])
def test_a_plan_takes_the_compute_the_rank_runs_on_the_card(rank, gpu, fused):
    kw = {"torch_dtype": "float16"} if gpu == "Tesla T4" else {}
    assert _plan_request(rank, gpu, **kw).fused_compute is fused


def test_a_moe_plan_takes_the_reference_experts_where_they_do_not_fit():
    from poros.core.config import PorosConfig
    from poros.plan import receipts as plan_receipts

    def moe(rank, gpu):
        cfg = PorosConfig(model_name_or_path="Qwen/Qwen3-30B-A3B", dataset="synthetic",
                          lora_rank=rank)
        return plan_receipts.Request.of(cfg, "moe", cap=capability_of_name(gpu)).moe_compute

    assert moe(128, "RTX 4090") == "fused" and moe(256, "RTX 4090") == "reference"
    assert moe(256, "H100 80GB HBM3") == "fused"


def test_a_higher_rank_plans_at_least_the_rank_16_peak_plus_its_partials():
    """From rank-16 receipts, a rank-64 / 128 fused plan grows by at least the fused partials and
    A products the larger rank holds (never below what a run at that rank holds)."""
    from poros.core.config import PorosConfig
    from tests.test_plan_upper import CONFIGS, RTX4090, _plan

    def peak(rank):
        cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-7B", dataset="synthetic",
                            lora_rank=rank, lora_alpha=2 * rank, max_seq_length=512,
                            per_device_batch_size=1, gradient_accumulation_steps=8)
        return _plan(cfg, "dense", RTX4090).peak_vram_allocated.value

    p16, p64, p128 = peak(16), peak(64), peak(128)
    assert CONFIGS["Qwen/Qwen2.5-7B"]
    if p16 is None:
        pytest.skip("no 7B receipt plans this config")
    grown = (F.dense_partial_bound(128, 512, 18944, 3584)
             - F.dense_partial_bound(16, 512, 18944, 3584)) / 2**30
    assert p16 <= p64 <= p128 and p128 - p16 >= grown


def test_a_plan_says_when_the_rank_takes_peft_modules_on_the_card():
    from poros.core.config import PorosConfig
    from tests.test_plan_upper import RTX4090, _plan

    def warned(rank):
        cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-7B", dataset="synthetic",
                          lora_rank=rank, lora_alpha=2 * rank)
        return [w for w in _plan(cfg, "dense", RTX4090).warnings if "PEFT's modules" in w]

    assert warned(128) == []
    said = warned(256)
    assert len(said) == 1 and "rank 256" in said[0] and "RTX 4090" in said[0]
