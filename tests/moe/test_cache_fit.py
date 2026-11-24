"""The MoE device expert cache is sized to this GPU (poros.train.compute.RunCompute): before update
2, from the VRAM the process could hold at setup, the run's peak through update 1 and its optimizer
step, what an update at the run's full shape would add (poros.plan.estimate.token_bytes) and a
margin; again after the first full-shape update; dropped to one layer by an update beyond the full
shape; one layer on path efficient.
CPU: grow_expert_cache and the residency are stand-ins (the CUDA sizing is tests/gpu/)."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from poros.plan.estimate import token_bytes  # noqa: E402
from poros.sss.tune import CacheFit, grown_capacity  # noqa: E402

GIB = 1 << 30
QWEN3_30B = {"model_type": "qwen3_moe", "hidden_size": 2048, "num_hidden_layers": 48,
             "num_attention_heads": 32, "intermediate_size": 6144, "num_experts": 128,
             "num_experts_per_tok": 8, "moe_intermediate_size": 768}


def test_token_bytes_grows_with_the_update_and_errs_high_against_the_4090_receipt():
    full = token_bytes(QWEN3_30B, path="fast", entry="moe", block_size=None,
                       shapes=[(8, 512)])
    half = token_bytes(QWEN3_30B, path="fast", entry="moe", block_size=None,
                       shapes=[(4, 512)])
    short = token_bytes(QWEN3_30B, path="fast", entry="moe", block_size=None,
                        shapes=[(8, 256)])
    assert full > half > 0 and full > short > 0
    # fast holds every block input (48 layers x 4096 tokens x 2048 x 4 B) on top of efficient
    efficient = token_bytes(QWEN3_30B, path="efficient", entry="moe", block_size=None,
                            shapes=[(8, 512)])
    assert full - efficient == pytest.approx(48 * 4096 * 2048 * 4 - 3 * 4096 * 2048 * 4)
    # the real 30B at mb8 x 512 on the RTX 4090 (docs/validation/moe/4090, q30g_c9): update 1's
    # allocated peak 7.48 GiB over 5.10 GiB at rest, 2.38 GiB that grow with the tokens; the
    # bound is above it
    assert full / GIB > 2.38
    assert token_bytes({"hidden_size": 8}, path="fast", entry="moe", block_size=None,
                       shapes=[(1, 4)]) is None


def test_the_fit_and_its_reason():
    assert grown_capacity(GIB, 23 * GIB, 8 * GIB, 100 * GIB, headroom=7 * GIB) == 9 * GIB
    fit = CacheFit(capacity=9 * GIB, budget=23 * GIB, peak=8 * GIB, longer=0,
                   headroom=7 * GIB, needed=15 * GIB, cap=None)
    assert fit.reason().startswith("sized to this GPU: 23.00 GiB this process may hold - 8.00 GiB")
    whole = CacheFit(capacity=15 * GIB, budget=48 * GIB, peak=8 * GIB, longer=GIB,
                     headroom=14 * GIB, needed=15 * GIB, cap=None)
    assert whole.reason().startswith("every expert (15.00 GiB) is cached") and "1.00 GiB for" in \
        whole.reason()
    capped = CacheFit(capacity=4 * GIB, budget=23 * GIB, peak=8 * GIB, longer=0,
                      headroom=7 * GIB, needed=15 * GIB, cap=4 * GIB)
    assert capped.reason().startswith("moe_expert_cache_gib caps it")


class _Residency:
    def __init__(self):
        self._capacity = self._floor = GIB
        self.shrunk = 0

    def shrink_to_floor(self):
        self.shrunk += 1
        self._capacity = self._floor
        return 0


def _compute(monkeypatch, **cfg):
    """A RunCompute over a stand-in trainer and model; grow_expert_cache records its calls."""
    from poros.moe import modules
    from poros.train import compute as compute_mod

    residency = _Residency()

    class Block(torch.nn.Module):
        pass

    block = Block()
    block.residency = residency
    monkeypatch.setattr(modules, "ManagedMoeBlock", Block)
    model = torch.nn.Sequential(block)
    model.config = types.SimpleNamespace(to_dict=lambda: dict(QWEN3_30B))
    calls = []

    def grow(model, device, max_bytes=None, *, budget=None, longer=0):
        calls.append({"max_bytes": max_bytes, "budget": budget, "longer": longer})
        residency._capacity = 9 * GIB
        return CacheFit(capacity=9 * GIB, budget=budget or 0, peak=8 * GIB, longer=longer,
                        headroom=7 * GIB, needed=15 * GIB, cap=max_bytes)

    import poros.sss.tune as tune

    monkeypatch.setattr(tune, "grow_expert_cache", grow)
    from poros.core.config import PorosConfig

    config = PorosConfig(model_name_or_path="x", dataset="synthetic", max_seq_length=512,
                         per_device_batch_size=8, gradient_accumulation_steps=1, **cfg)
    trainer = types.SimpleNamespace(config=config, device=torch.device("cuda"))
    rc = compute_mod.RunCompute(trainer)
    rc.vram_budget = 23 * GIB
    return rc, model, residency, calls


def _update(rc, model, shapes):
    """One block-major update's hooks: before it runs, after it ran, and its optimizer step."""
    rc.cache_before_update(model, shapes)
    rc.cache_after_update(model, shapes)
    rc.stepped()


def test_fast_sizes_before_update_2_on_update_1_at_the_full_shape_once(monkeypatch):
    rc, model, residency, calls = _compute(monkeypatch, path="fast")
    rc.cache_prepared(GIB)
    assert rc.ran["moe_expert_cache_reason"] == "one layer until update 1 is measured"
    _update(rc, model, [(8, 512)])
    assert calls == []  # update 1's optimizer step has not run yet: sized before update 2
    for _ in range(3):
        _update(rc, model, [(8, 512)])
    assert calls == [{"max_bytes": None, "budget": 23 * GIB, "longer": 0}]
    assert rc.ran["moe_expert_cache_gib"] == 9.0
    assert rc.ran["moe_expert_cache_reason"].startswith("sized to this GPU")
    assert rc._t.config.ran_compute["moe_expert_cache_gib"] == 9.0
    assert residency.shrunk == 0


def test_a_short_update_1_leaves_room_for_the_full_shape_then_sizes_again(monkeypatch):
    rc, model, _, calls = _compute(monkeypatch, path="fast")
    _update(rc, model, [(8, 200)])
    _update(rc, model, [(8, 300)])  # sized on update 1, with room for the full shape
    full = token_bytes(QWEN3_30B, path="fast", entry="moe", block_size=None, shapes=[(8, 512)])
    short = token_bytes(QWEN3_30B, path="fast", entry="moe", block_size=None, shapes=[(8, 200)])
    assert len(calls) == 1 and calls[0]["longer"] == int(full - short) > 0
    _update(rc, model, [(8, 512)])  # update 2 was short too: nothing new to size on
    assert len(calls) == 1
    _update(rc, model, [(8, 512)])  # update 3, at the full shape, measured: sized on it
    assert len(calls) == 2 and calls[1]["longer"] == 0
    _update(rc, model, [(8, 512)])
    assert len(calls) == 2


def test_an_update_beyond_the_full_shape_drops_the_cache_for_the_run(monkeypatch):
    rc, model, residency, calls = _compute(monkeypatch, path="fast")
    _update(rc, model, [(8, 512)])
    _update(rc, model, [(8, 512)])
    assert len(calls) == 1 and residency.shrunk == 0
    _update(rc, model, [(8, 700)])  # a caller's loader past max_seq_length
    assert residency.shrunk == 1 and rc.ran["moe_expert_cache_gib"] == 1.0
    assert "beyond the run's full shape" in rc.ran["moe_expert_cache_reason"]
    _update(rc, model, [(8, 512)])
    assert len(calls) == 1  # never sized again


def test_efficient_keeps_one_layer_and_a_set_cap_is_passed(monkeypatch):
    rc, model, _, calls = _compute(monkeypatch, path="efficient")
    rc.cache_prepared(GIB)
    for _ in range(3):
        _update(rc, model, [(8, 512)])
    assert calls == [] and rc.ran["moe_expert_cache_reason"].startswith("one layer: path effic")
    rc, model, _, calls = _compute(monkeypatch, path="efficient", moe_expert_cache_gib=4)
    for _ in range(2):
        _update(rc, model, [(8, 512)])
    assert calls[0]["max_bytes"] == 4 * GIB


def test_an_update_whose_step_has_not_run_waits_for_it(monkeypatch):
    """poros.Run: two forward_backward calls before optim_step. The second must not size on the
    first, whose peak does not yet hold the optimizer step's state (Adam's, made at step 1)."""
    rc, model, _, calls = _compute(monkeypatch, path="fast")
    rc.cache_before_update(model, [(8, 512)])
    rc.cache_after_update(model, [(8, 512)])
    rc.cache_before_update(model, [(8, 512)])  # no step between
    rc.cache_after_update(model, [(8, 512)])
    assert calls == []
    rc.stepped()
    rc.cache_before_update(model, [(8, 512)])
    assert len(calls) == 1


def test_the_ssd_stores_sequential_loop_sizes_after_update_1s_step(monkeypatch, tmp_path):
    """The SSD store sized its cache at the end of the first pass, before update 1's optimizer
    step made Adam's state, with no room for it; now it sizes as the RAM store does: on the first
    pass of update 2, the peak through update 1 and its step measured."""
    from poros.train.trainer import PorosTrainer

    rc, model, _, calls = _compute(monkeypatch, path="sequential", store="ssd",
                                   store_path=str(tmp_path))
    order = []
    param = torch.nn.Parameter(torch.ones(2))
    fake = types.SimpleNamespace(
        _sss_source=object(), block_manager=None, model=model, compute=rc, keyed_rng=None,
        _sss_pass_shapes=[], before_pass=lambda g: None,
        _forward_blockwise_adapter_model=lambda batch: (param * 2).sum(),
        _sss_after_pass=lambda: order.append("pass"), _ran_loop=lambda block_major: None,
        optimizer=torch.optim.SGD([param], lr=0.1), config=rc._t.config,
        trainable_parameters=[], history=types.SimpleNamespace(grad_norms=[]), _witness=None,
        _lr_scheduler=None,
        global_step=0)
    # the pass hooks the classic loop shares with poros.Run, bound to the fake
    fake.sss_pass_begins = lambda b: PorosTrainer.sss_pass_begins(fake, b)
    fake.sss_pass_ends = lambda: PorosTrainer.sss_pass_ends(fake)
    batch = {"input_ids": torch.ones(8, 512, dtype=torch.long)}
    for update in range(2):
        for g in range(2):
            PorosTrainer._sequential_pass(fake, batch, g, 2)
            order.append(("sized", len(calls)))
        PorosTrainer.end_passes(fake)
        PorosTrainer._adapter_optimizer_step(fake)
        order.append("step")
    # nothing sized through update 1 and its step; sized once, at update 2's first pass
    assert order == ["pass", ("sized", 0), "pass", ("sized", 0), "step",
                     "pass", ("sized", 1), "pass", ("sized", 1), "step"]
    assert calls == [{"max_bytes": None, "budget": 23 * GIB, "longer": 0}]
    assert rc.ran["moe_expert_cache_reason"].startswith("sized to this GPU")
