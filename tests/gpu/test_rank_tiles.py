"""The dense fused kernels at every rank's own tiles (poros.core.fused_tiles), against fp64.

Ranks up to 32 run the configuration every receipt ran; 64 and 128 run smaller tiles that fit
every card's shared memory, with a call's rows in chunks so the partial buffers stay small. Each
kernel's outputs are checked against the same math in fp64 (the frozen outputs and grads in the
compute dtype, everything else fp32), at shapes with partial row and column tiles; the chunked
path (forced to many chunks) is checked the same way, and a call repeated gives the same bits.
``check_all(device)`` runs every check: on CUDA here, and on the CPU under Triton's interpreter
(tests/core/test_rank_tiles_interp.py).
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

RANKS = (16, 32, 64, 128)
F64 = torch.float64


def _rand(device, *shape, dtype=torch.float32, seed=0, scale=0.1):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale).to(dtype).to(device)


def _rel(x, ref):
    """max |x - ref| / max |ref|."""
    ref = ref.to(F64)
    return float((x.to(F64) - ref).abs().max() / ref.abs().max().clamp_min(1e-30))


# fp32 sums of at most a few hundred terms: a relative error well under 1e-5; the compute-dtype
# outputs round once to bf16 (2^-8 relative)
TOL32, TOL_BF16 = 2e-5, 1e-2


def check_out(device, R, M=300, N=640, s=0.7, dt=torch.bfloat16):
    from poros.core.fused import kernels as K
    from poros.core.fused import lora

    yb = _rand(device, M, N, dtype=dt, seed=1)
    a, B = _rand(device, M, R, seed=2), _rand(device, N, R, seed=3)
    res = _rand(device, M, N, seed=4)
    out = lora._out_fwd(K, yb, a, B, s, res)
    ref = res.to(F64) + yb.to(F64) + s * (a.to(F64) @ B.to(F64).T)
    assert _rel(out, ref) < TOL32
    g = _rand(device, M, N, seed=5)
    gb, da, db = lora._out_bwd(K, g, a, B, s, dt)
    if device == "cuda":  # the kernel's cast is torch's; the interpreter rounds its own way
        assert torch.equal(gb, g.to(dt))
    assert _rel(gb, g.to(F64)) < TOL_BF16
    assert _rel(da, s * (g.to(F64) @ B.to(F64))) < TOL32
    assert _rel(db, s * (g.to(F64).T @ a.to(F64))) < TOL32
    again = lora._out_bwd(K, g, a, B, s, dt)
    assert all(torch.equal(x, y) for x, y in zip((gb, da, db), again))


def check_rope(device, R, D=128, heads=3, M=200, s=0.5, dt=torch.bfloat16):
    from poros.core.fused import kernels as K
    from poros.core.fused import lora

    N, H = heads * D, D // 2
    yb = _rand(device, M, N, dtype=dt, seed=6)
    a, B = _rand(device, M, R, seed=7), _rand(device, N, R, seed=8)
    ang = _rand(device, 1, M, H, seed=9, scale=3.0)
    cos = torch.cat([ang.cos(), ang.cos()], -1).contiguous()
    sin = torch.cat([ang.sin(), ang.sin()], -1).contiguous()
    rope = lora.Rope(cos, sin, 1)
    out = lora._rope_fwd(K, yb, a, B, s, rope)
    y = (yb.to(F64) + s * (a.to(F64) @ B.to(F64).T)).view(M, heads, D)
    c, sn = cos[0].to(F64)[:, None, :], sin[0].to(F64)[:, None, :]
    rot = torch.cat([-y[..., H:], y[..., :H]], -1)
    ref = (y * c + rot * sn).view(M, N)
    assert _rel(out, ref) < TOL32
    g = _rand(device, M, N, seed=10)
    gb, da, db = lora._rope_bwd(K, g, a, B, s, dt, rope)
    g3 = g.to(F64).view(M, heads, D)
    glo, ghi = g3[..., :H], g3[..., H:]
    dlo = glo * c[..., :H] + ghi * sn[..., H:]
    dhi = ghi * c[..., H:] - glo * sn[..., :H]
    dl = torch.cat([dlo, dhi], -1).view(M, N)
    assert _rel(gb, dl) < TOL_BF16
    assert _rel(da, s * (dl @ B.to(F64))) < TOL32
    assert _rel(db, s * (dl.T @ a.to(F64))) < TOL32


def _swiglu_ref(bg, bu, ag, au, Bg, Bu, Ad, s):
    g = bg.to(F64) + s * (ag.to(F64) @ Bg.to(F64).T)
    u = bu.to(F64) + s * (au.to(F64) @ Bu.to(F64).T)
    sig = torch.sigmoid(g)
    return g, u, sig, g * sig * u


def check_swiglu(device, R, drop, M=300, N=704, s=0.5, p=0.1, dt=torch.bfloat16):
    from poros.core.fused import kernels as K
    from poros.core.fused import lora

    bf = dt
    bg, bu = _rand(device, M, N, dtype=bf, seed=11), _rand(device, M, N, dtype=bf, seed=12)
    ag, au = _rand(device, M, R, seed=13), _rand(device, M, R, seed=14)
    Bg, Bu, Ad = _rand(device, N, R, seed=15), _rand(device, N, R, seed=16), \
        _rand(device, R, N, seed=17)
    g, u, sig, h = _swiglu_ref(bg, bu, ag, au, Bg, Bu, Ad, s)
    hb = torch.empty(M, N, dtype=bf, device=device)
    if drop:
        hf = lora.swiglu_fwd(K, bg, bu, ag, au, Bg, Bu, Ad, s, hb, None)
        assert _rel(hf, h) < TOL32
    else:
        ad = lora.swiglu_fwd(K, bg, bu, ag, au, Bg, Bu, Ad, s, hb, Ad)
        assert _rel(ad, h @ Ad.to(F64).T) < TOL32
    assert _rel(hb, h) < TOL_BF16
    dzb, dad = _rand(device, M, N, dtype=bf, seed=18), _rand(device, M, R, seed=19)
    md = (_rand(device, M, N, seed=20) > -0.12).contiguous() if drop else None
    pscale = 1.0 / (1.0 - p) if drop else 1.0
    out = lora.swiglu_bwd(K, bg, bu, ag, au, Bg, Bu, dzb, dad, Ad, md, s, pscale, bf)
    dl = dad.to(F64) @ Ad.to(F64)
    keep = md.to(F64) * pscale if drop else 1.0
    hd = h * keep
    dh = dzb.to(F64) + dl * keep
    dg = dh * u * (sig * (1 + g * (1 - sig)))
    du = dh * g * sig
    refs = (dg, du, dg @ Bg.to(F64), du @ Bu.to(F64), dg.T @ ag.to(F64), du.T @ au.to(F64),
            dad.to(F64).T @ hd)
    for i, (x, ref) in enumerate(zip(out, refs)):
        assert _rel(x, ref) < (TOL_BF16 if i < 2 else TOL32), (R, drop, i, _rel(x, ref))
    again = lora.swiglu_bwd(K, bg, bu, ag, au, Bg, Bu, dzb, dad, Ad, md, s, pscale, bf)
    assert all(torch.equal(x, y) for x, y in zip(out, again))


def check_all(device):
    for dt in (torch.bfloat16, torch.float16):  # fp16: the compute dtype on sm_70 / sm_75
        for R in RANKS:
            check_out(device, R, dt=dt)
            for D in (64, 128):
                check_rope(device, R, D=D, dt=dt)
            for drop in (False, True):
                check_swiglu(device, R, drop, dt=dt)


@pytest.fixture
def few_rows(monkeypatch):
    """Rows in many chunks above rank 32: a partial buffer of at most 2^16 elements a launch."""
    from poros.core import fused_tiles

    monkeypatch.setattr(fused_tiles, "PARTIAL_ELEMS", 1 << 16)


def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA")
    pytest.importorskip("bitsandbytes")
    return "cuda"


DTYPES = pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])


@pytest.mark.gpu
@DTYPES
@pytest.mark.parametrize("R", RANKS)
def test_lora_out_matches_fp64(R, dt):
    check_out(_cuda(), R, dt=dt)


@pytest.mark.gpu
@DTYPES
@pytest.mark.parametrize("D", [64, 128])
@pytest.mark.parametrize("R", RANKS)
def test_rope_matches_fp64(R, D, dt):
    check_rope(_cuda(), R, D=D, dt=dt)


@pytest.mark.gpu
@DTYPES
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.parametrize("R", RANKS)
def test_swiglu_matches_fp64(R, drop, dt):
    check_swiglu(_cuda(), R, drop, dt=dt)


@pytest.mark.gpu
@pytest.mark.parametrize("R", [64, 128])
def test_row_chunks_match_fp64(R, few_rows):
    """The chunked rows (here dozens of chunks) compute the same sums, to fp32 rounding."""
    from poros.core import fused_tiles

    assert fused_tiles.row_chunk(fused_tiles.tiles(R), 300, 44, R, 16) < 300
    device = _cuda()
    check_out(device, R)
    for drop in (False, True):
        check_swiglu(device, R, drop)


@pytest.mark.gpu
@pytest.mark.parametrize("R", [64, 128])
def test_ranks_64_and_128_fit_this_card(R):
    """Every kernel launches at these ranks on this card (fit.reason probes them)."""
    from poros.core.fused import fit

    _cuda()
    for hd in (None, 64):
        assert fit.reason(R, torch.bfloat16, hd) is None
    cap = torch.cuda.get_device_capability()
    if cap != (7, 5) or R <= 64:  # sm_75: the rotary kernels at head dim 128 fit up to 64
        assert fit.reason(R, torch.bfloat16, 128) is None

