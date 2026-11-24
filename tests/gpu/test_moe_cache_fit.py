"""The MoE expert cache sized to this GPU (CUDA): path fast, moe_expert_cache_gib unset,
grows the cache after update 1 within the VRAM the process could hold at setup and records how
(ran_compute); the adapters are bit for bit those of a run whose cache stays at one layer, and
of one whose first update is shorter than the rest (sized with room for the full shape, then
again at it). Cache size decides when bytes move, never which."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from tests.moe.test_families import _model  # noqa: E402


def _rows(lengths, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for n in lengths:
        ids = torch.randint(1, 96, (1, n), generator=g)
        out.append({"input_ids": ids, "labels": ids.clone()})
    return out


def _train(checkpoint, out, lengths, **knobs):
    from poros.train.trainer import PorosTrainer

    tr = PorosTrainer.from_pretrained(
        str(checkpoint), output_dir=str(out), max_steps=len(lengths), save_steps=0,
        progress=False, gradient_accumulation_steps=1, per_device_batch_size=1,
        max_seq_length=32, lora_dropout=0.0, path="fast", **knobs)
    loader = torch.utils.data.DataLoader(
        [{k: v[0] for k, v in r.items()} for r in _rows(lengths)], batch_size=1)
    tr.train(loader, steps=len(lengths), seq_len=32)
    params = {n: p.detach().clone() for n, p in tr.model.named_parameters() if p.requires_grad}
    return tr, params


def _capacity(tr) -> int:
    """The trainer's expert cache (bytes; the record rounds GiB to 3 places, 0.0 for these
    models' few KiB)."""
    from poros.train.compute import _residency

    return int(_residency(tr.model)._capacity)


def test_fast_fits_the_cache_and_trains_the_same_bits(tmp_path):
    base = tmp_path / "base"
    _model("olmoe").to(torch.bfloat16).save_pretrained(base)
    fit, a = _train(base, tmp_path / "fit", [32, 32, 32])
    ran = fit.config.ran_compute
    assert ran["moe_expert_cache_reason"].startswith(("sized to this GPU", "every expert"))
    one, b = _train(base, tmp_path / "one", [32, 32, 32], moe_expert_cache_gib=0)
    assert one.config.ran_compute["moe_expert_cache_reason"].startswith(
        "moe_expert_cache_gib caps it")
    assert _capacity(one) < _capacity(fit)  # one layer of two, against both
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_a_short_first_update_then_the_full_shape_trains_the_same_bits(tmp_path):
    base = tmp_path / "base"
    _model("olmoe").to(torch.bfloat16).save_pretrained(base)
    short, a = _train(base, tmp_path / "short", [8, 32, 16])
    assert short.config.ran_compute["moe_expert_cache_reason"].startswith(
        ("sized to this GPU", "every expert"))
    one, b = _train(base, tmp_path / "one", [8, 32, 16], moe_expert_cache_gib=0)
    assert all(torch.equal(a[k], b[k]) for k in a)
