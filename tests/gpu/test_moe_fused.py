"""The fused expert compute (``poros.moe.fused``) on CUDA.

* against the reference compute (``_experts_compute``): the same math up to bf16 rounding --
  output, input grad, routing-weight grad and every adapter grad;
* at LoRA dropout: the adapters' dropout with the masks the call drew, against fp64 math;
* deterministic: two runs, the same bits;
* segment-independent, bit for bit: micro-batches computed in one call get exactly the output,
  input grad, routing-weight grad and adapter grads each gets in a call of its own (what
  block-major batching relies on), at a toy width and at Qwen3-30B-A3B's.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the fused expert compute runs on CUDA", allow_module_level=True)
pytest.importorskip("triton")

from tests.moe import require_tested_transformers  # noqa: E402

require_tested_transformers()

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

from poros.moe.fused import (  # noqa: E402
    FusedExpertFunction,
    Plan,
    experts_backward,
    experts_forward,
)
from poros.moe.modules import (  # noqa: E402
    ManagedMoeBlock,
    _experts_compute,
    _rng_state,
    _set_rng_state,
)
from poros.moe.prepare import MoeConfig, prepare_moe  # noqa: E402

DIMS = {
    "toy": dict(hidden_size=256, moe_intermediate_size=128, num_experts=16,
                num_experts_per_tok=4),
    "q30": dict(hidden_size=2048, moe_intermediate_size=768, num_experts=128,
                num_experts_per_tok=8),
}


def _block(dims: str, dropout: float = 0.0, streamed: bool = False) -> ManagedMoeBlock:
    torch.manual_seed(0)
    d = DIMS[dims]
    config = Qwen3MoeConfig(vocab_size=64, intermediate_size=512, num_hidden_layers=1,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                            norm_topk_prob=True, **d)
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Qwen3MoeForCausalLM(config)
    finally:
        torch.set_default_dtype(torch.float32)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" not in name:
                p.normal_(0.0, 0.02)
    knobs = {}
    if streamed:  # one layer's slabs at a time: every call moves its experts to the device
        from poros.moe.prepare import streaming_config

        knobs = vars(streaming_config(model.config))
        knobs = {k: knobs[k] for k in ("capacity_bytes", "look_ahead")}
    prepare_moe(model, MoeConfig(streamed=streamed, device="cuda", lora_dropout=dropout,
                                 **({"look_ahead": 0} | knobs)))
    block = next(m for m in model.modules() if isinstance(m, ManagedMoeBlock))
    torch.manual_seed(1)
    with torch.no_grad():  # a trained adapter: nonzero B
        block.adapter.lora_B_gate_up.normal_(0, 0.02)
        block.adapter.lora_B_down.normal_(0, 0.02)
    return block


def _inputs(block, tokens: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, block.hidden_size, generator=g, device="cuda").to(torch.bfloat16)
    with torch.no_grad():
        _, w, idx = block.gate(x)
    gy = (torch.randn(tokens, block.hidden_size, generator=g, device="cuda") * 0.01).to(
        torch.bfloat16)
    return x, idx, w, gy


def _autograd(fn, block, x, idx, w, gy):
    block.adapter.zero_grad(set_to_none=True)
    xi = x.clone().requires_grad_(True)
    wi = w.clone().requires_grad_(True)
    out = fn(xi, idx, wi, block)
    out.backward(gy)
    return [out.detach(), xi.grad, wi.grad, *(p.grad.clone() for p in block.adapter.parameters())]


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def test_fused_matches_reference():
    block = _block("toy")
    x, idx, w, gy = _inputs(block, 384, 1)
    want = _autograd(_experts_compute, block, x, idx, w, gy)
    got = _autograd(lambda *a: FusedExpertFunction.apply(*a), block, x, idx, w, gy)
    names = ["out", "dx", "dw", "dA_gu", "dB_gu", "dA_d", "dB_d"]
    for name, a, b in zip(names, got, want, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert _rel(a, b) < 2e-2, (name, _rel(a, b))


def _fp64(block, x, idx, w, gy, keep_x=None, keep_h=None, p=0.0):
    """The experts' math in fp64 over the NF4-decoded weights the arms use: per routed (token,
    slot), ``gate_up = x Wgu^T + s B(A(dropout(x)))``, ``h = silu(gate) * up``, ``y = h Wd^T +
    s B(A(dropout(h)))``, summed with the routing weights; ``dropout`` multiplies by the
    (token * top_k + slot)-indexed mask and 1 / (1 - p) (none without masks). Returns the output,
    the input and routing-weight grads, and the four adapter grads."""
    import torch.nn.functional as F

    ad = block.adapter
    s = float(ad.scaling)
    params = [p_.detach().double().clone().requires_grad_(True)
              for p_ in (ad.lora_A_gate_up, ad.lora_B_gate_up, ad.lora_A_down, ad.lora_B_down)]
    x64 = x.double().clone().requires_grad_(True)
    w64 = w.double().clone().requires_grad_(True)
    out = torch.zeros_like(x64)
    hit = sorted(set(idx.reshape(-1).tolist()))
    block.residency.ensure([(block.layer_idx, e) for e in hit])
    for e in hit:
        key = (block.layer_idx, e)
        slab = block.residency.resident_qbytes(key)
        wgu = block.store.dequantize_tensor(key, "gate_up_proj", slab).double()
        wd = block.store.dequantize_tensor(key, "down_proj", slab).double()
        tok, slot = torch.where(idx == e)
        row = tok * idx.shape[1] + slot
        xe = x64[tok]
        xa = xe if keep_x is None else xe * keep_x[row].double() / (1.0 - p)
        gate, up = (xe @ wgu.T + s * ((xa @ params[0][e].T) @ params[1][e].T)).chunk(2, dim=-1)
        h = F.silu(gate) * up
        ha = h if keep_h is None else h * keep_h[row].double() / (1.0 - p)
        y = h @ wd.T + s * ((ha @ params[2][e].T) @ params[3][e].T)
        out = out.index_add(0, tok, y * w64[tok, slot, None])
    out.backward(gy.double())
    return [out.detach(), x64.grad, w64.grad, *(p_.grad for p_ in params)]


def test_fused_dropout_is_the_adapters_dropout_with_its_masks(monkeypatch):
    """At lora_dropout 0.25 the fused compute (the default on CUDA) is the experts' math with
    dropout on each adapter's input, ``B(A(dropout(x)))`` as ``poros.moe.adapters`` defines it
    for the reference compute: with the masks its call drew (the gate/up adapter's input and the
    down adapter's, kept at 1 / (1 - p)), an fp64 computation of that math gives its output,
    input grad, routing-weight grad and adapter grads; its backward draws its forward's masks
    again; about p of each mask drops; and the same math without the masks is far off (the
    control: the masks are applied where they act)."""
    from poros.moe import fused

    p = 0.25
    block = _block("toy", dropout=p)
    x, idx, w, gy = _inputs(block, 384, 3)
    drawn = []
    real = fused._masks

    def probe(*args):
        masks = real(*args)
        drawn.append(masks)
        return masks

    monkeypatch.setattr(fused, "_masks", probe)
    got = _autograd(lambda *a: FusedExpertFunction.apply(*a), block, x, idx, w, gy)
    assert len(drawn) == 2 and all(m is not None for m in drawn)  # the forward's, the backward's
    fwd, bwd = drawn
    assert torch.equal(fwd.x, bwd.x) and torch.equal(fwd.h, bwd.h)
    for keep in (fwd.x, fwd.h):
        assert abs(1.0 - keep.float().mean().item() - p) < 0.01
    dest = Plan.build(idx, block.num_experts).dest.long()  # sorted row -> token * top_k + slot
    keep_x, keep_h = torch.empty_like(fwd.x), torch.empty_like(fwd.h)
    keep_x[dest], keep_h[dest] = fwd.x, fwd.h
    want = _fp64(block, x, idx, w, gy, keep_x, keep_h, p)
    names = ["out", "dx", "dw", "dA_gu", "dB_gu", "dA_d", "dB_d"]
    for name, a, b in zip(names, got, want, strict=True):
        assert a.shape == b.shape, name
        assert _rel(a, b) < 1e-2, (name, _rel(a, b))
    unmasked = _fp64(block, x, idx, w, gy)
    for name, a, b in list(zip(names, got, unmasked, strict=True))[3:]:
        assert _rel(a, b) > 0.1, (name, _rel(a, b))


def test_fused_is_deterministic():
    block = _block("toy")
    inputs = _inputs(block, 300, 2)
    first = _autograd(lambda *a: FusedExpertFunction.apply(*a), block, *inputs)
    second = _autograd(lambda *a: FusedExpertFunction.apply(*a), block, *inputs)
    for a, b in zip(first, second, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("dims,lengths,dropout", [
    ("toy", (64, 31, 97), 0.0), ("q30", (512, 512, 300), 0.0), ("toy", (64, 31, 97), 0.1),
])
def test_segments_compute_as_alone(dims, lengths, dropout):
    block = _block(dims, dropout)
    parts = [_inputs(block, n, 10 + i) for i, n in enumerate(lengths)]
    states = []
    for i in range(len(lengths)):  # each micro-batch's own random state (keyed, as in training)
        torch.manual_seed(100 + i)
        states.append(_rng_state(torch.device("cuda")))
    alone = []
    for (x, idx, w, gy), state in zip(parts, states, strict=True):
        plan = Plan.build(idx, block.num_experts)
        _set_rng_state(torch.device("cuda"), state)  # its own call draws from the live state
        out = experts_forward(x, w, plan, block)
        dx, dw, grads = experts_backward(x, w, gy, plan, block, [state] if dropout else None)
        alone.append((out, dx, dw, grads[0]))
    x, idx, w, gy = (torch.cat([p[i] for p in parts]) for i in range(4))
    plan = Plan.build(idx, block.num_experts, segments=list(lengths))
    keyed = states if dropout else None
    out = experts_forward(x, w, plan, block, keyed)
    dx, dw, grads = experts_backward(x, w, gy, plan, block, keyed)
    assert len(grads) == len(lengths)
    start = 0
    for (a_out, a_dx, a_dw, a_grads), n, seg_grads in zip(alone, lengths, grads, strict=True):
        assert torch.equal(out[start:start + n], a_out)
        assert torch.equal(dx[start:start + n], a_dx)
        assert torch.equal(dw[start:start + n], a_dw)
        for a, b in zip(seg_grads, a_grads, strict=True):
            assert torch.equal(a, b)
        start += n


def test_streamed_experts_compute_as_resident():
    """Experts streamed into a one-layer budget give the resident model's values, bit for bit
    (the slabs are the same bytes wherever they come from)."""
    runs = []
    for streamed in (False, True):
        block = _block("q30", streamed=streamed)
        inputs = _inputs(block, 700, 9)
        runs.append(_autograd(lambda *a: FusedExpertFunction.apply(*a), block, *inputs))
    for a, b in zip(*runs, strict=True):
        assert torch.equal(a, b)
