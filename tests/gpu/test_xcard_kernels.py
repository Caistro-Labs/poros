"""The fused kernels on the card at hand, at every LoRA rank: they fit its shared memory or say
so before training, and a cut in pipeline stages changes no bit (poros.core.fused.launch).

A rank's padded width R sizes the kernels' adapter tiles ([BLOCK, R] fp32), so each padded rank
has its own launch configuration (poros.core.fused_tiles.tiles: from R = 64 smaller blocks and
fewer stages, fitting the 99 KB a block may take on sm_86 / sm_89 / sm_120). Where a configured
stage count still does not fit a card, ``launch`` runs the kernel at fewer stages; these tests
check that the values are the configured stages' bit for bit, and that every rank Poros accepts
trains on this card.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the fused kernels are CUDA-only", allow_module_level=True)
pytest.importorskip("bitsandbytes")
pytest.importorskip("triton")
transformers = pytest.importorskip("transformers")

from poros.core import fused  # noqa: E402
from poros.core.fused import kernels as K  # noqa: E402
from poros.core.fused import launch as L  # noqa: E402

pytestmark = pytest.mark.gpu


def _rand(*shape, dtype=torch.float32, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(*shape, generator=g, device="cuda") * 0.1).to(dtype)


def _lora_out_bwd(R, stages, M=200, N=384):
    g, a, b = _rand(M, N, seed=1), _rand(M, R, seed=2), _rand(N, R, seed=3)
    nt, ns, rows = K.splits(M, N, K.BWD_BM, K.BWD_BN)
    gb = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    da = torch.empty(nt, M, R, device="cuda")
    db = torch.empty(ns, N, R, device="cuda")
    L.launch(K._lora_out_bwd, (nt, ns), g, a, b, gb, da, db, M, N, rows, R=R, BLOCK_M=K.BWD_BM,
             BLOCK_N=K.BWD_BN, enable_fp_fusion=True, num_warps=K.BWD_WARPS, num_stages=stages)
    return gb, da, db


def _swiglu_bwd(R, stages, drop, M=200, N=320):
    bf = torch.bfloat16
    bg, bu, dzb = _rand(M, N, dtype=bf, seed=1), _rand(M, N, dtype=bf, seed=2), \
        _rand(M, N, dtype=bf, seed=3)
    ag, au, dad = _rand(M, R, seed=4), _rand(M, R, seed=5), _rand(M, R, seed=6)
    wbg, wbu, wad = _rand(N, R, seed=7), _rand(N, R, seed=8), _rand(R, N, seed=9)
    keep = (_rand(M, N, seed=10) > 0).contiguous() if drop else dzb
    bm, bn = K.SWIGLU_BWD_BM, K.SWIGLU_BWD_BN
    nt, ns, rows = K.splits(M, N, bm, bn)
    out = [torch.empty(M, N, dtype=bf, device="cuda") for _ in range(2)] + \
        [torch.empty(nt, M, R, device="cuda") for _ in range(2)] + \
        [torch.empty(ns, N, R, device="cuda") for _ in range(2)] + \
        [torch.empty(ns, R, N, device="cuda")]
    L.launch(K._swiglu_bwd, (nt, ns), bg, bu, ag, au, wbg, wbu, dzb, dad, wad, keep, *out,
             M, N, 0.5, 1.0 / 0.95, rows, R=R, BLOCK_M=bm, BLOCK_N=bn, HAS_DROP=drop,
             enable_fp_fusion=True, num_warps=K.SWIGLU_BWD_WARPS, num_stages=stages)
    return out


@pytest.mark.parametrize("R", [16, 32, 64])
def test_fewer_stages_change_no_bit(R, monkeypatch):
    """The stages a cut configuration runs at give the configured stages' values exactly."""
    monkeypatch.setattr(L, "_STAGES", {})
    ref = _lora_out_bwd(R, K.BWD_STAGES)
    monkeypatch.setattr(L, "_STAGES", {})
    one = _lora_out_bwd(R, 1)
    assert all(torch.equal(x, y) for x, y in zip(ref, one))
    for drop in (False, True):
        monkeypatch.setattr(L, "_STAGES", {})
        ref = _swiglu_bwd(R, 2, drop)
        monkeypatch.setattr(L, "_STAGES", {})
        one = _swiglu_bwd(R, 1, drop)
        assert all(torch.equal(x, y) for x, y in zip(ref, one))


def _model(tmp_path, rank):
    from poros.train.model import PorosModel

    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=256, hidden_size=256, intermediate_size=512,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2)
    m = transformers.AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05)
    m.to(torch.bfloat16).save_pretrained(tmp_path)
    pm = PorosModel.from_pretrained(str(tmp_path), quantization="nf4", lora_rank=rank,
                                    lora_alpha=2 * rank, lora_dropout=0.05,
                                    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                         "gate_proj", "up_proj", "down_proj"],
                                    adapter_backend="peft")
    return pm.model.cuda().train()


@pytest.mark.parametrize("rank", [8, 16, 32, 64, 128, 256])
def test_every_rank_trains_on_this_card(tmp_path, rank):
    """Fused compute installed, a forward and backward at this rank: no Triton resource error,
    finite gradients on every adapter. Ranks whose kernels cannot fit even at one stage keep
    PEFT's modules (fused.install refuses them, saying why)."""
    model = _model(tmp_path, rank)
    L._STAGES.clear()
    counts = fused.install(model)
    refused = fused.refusals(model)
    ids = torch.randint(3, 250, (2, 64), device="cuda")
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
    if refused:  # this card cannot run the rank: nothing routed, the reason given
        assert counts["linears"] == 0 and "shared memory" in refused[0]
    else:  # the fused kernels ran, forward and backward
        assert counts["linears"] == 14
        assert any(k[0] is K._lora_out_bwd for k in L._STAGES)
    print(f"XCARD rank {rank}: counts {counts} refusals {refused} stages {L.stages_used()}")
    fused.uninstall(model)


@pytest.mark.parametrize("rp", [16, 32, 64, 128])
def test_moe_expert_kernels_fit_or_say_why(rp):
    """The MoE expert kernels at each padded rank on this card: they launch, or fit_reason says
    why not (prepare_moe then takes the reference compute)."""
    from poros.moe import fused as mf

    why = mf.fit_reason(rp)
    assert why is None or "shared memory" in why
    print(f"XCARD moe rp {rp}: {why}")
