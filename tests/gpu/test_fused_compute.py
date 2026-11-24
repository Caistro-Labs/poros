"""poros.core.fused on CUDA: each fused piece against PEFT's / transformers' own module,
determinism, the skipped frozen GEMM, the dropout draws, and the training paths bitwise-equal to
each other with the fused compute on.

A tiny random Qwen2 (NF4, PEFT LoRA r16 on the seven projections, LoRA A and B randomized so the
adapter path is not zero). Closeness to the stock modules is checked where the stock computation
is itself stable: a frozen GEMM takes its input rounded to bf16, so a one-ulp change in an fp32
input already moves the stock output by up to a bf16 step (~0.4%); the pieces whose inputs the
fused path forms in another fp32 order (the MLP, the whole layer) are held to that bound.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("the fused compute is CUDA-only", allow_module_level=True)
pytest.importorskip("bitsandbytes")
pytest.importorskip("triton")
transformers = pytest.importorskip("transformers")

from poros.core import fused  # noqa: E402
from poros.core.determinism import enforce_determinism  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(vocab_size=256, hidden_size=256, intermediate_size=512,
                                   num_hidden_layers=3, num_attention_heads=8,
                                   num_key_value_heads=2)
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.05)
    path = tmp_path_factory.mktemp("qwen2")
    model.to(torch.bfloat16).save_pretrained(path)
    return path


def _layer(checkpoint, dropout=0.0, rank=16):
    from poros.train.model import PorosModel

    enforce_determinism(fill_uninitialized=False)
    pm = PorosModel.from_pretrained(str(checkpoint), quantization="nf4", lora_dropout=dropout,
                                    lora_target_modules=TARGETS, adapter_backend="peft",
                                    lora_rank=rank)
    model = pm.model
    for m in model.modules():  # the engine recomputes per block itself (HF's checkpointing off)
        if getattr(m, "gradient_checkpointing", False) is True:
            m.gradient_checkpointing = False
    base = model
    while not hasattr(base, "layers"):
        base = base.model if hasattr(base, "model") else base.base_model
    layer = base.layers[1].cuda().train()
    g = torch.Generator().manual_seed(1)
    for n, p in layer.named_parameters():
        if p.requires_grad:
            p.data = (torch.randn(p.shape, generator=g) * (0.05 if "lora_A" in n else 0.02)).to(
                "cuda", p.dtype)
    return model, base, layer


def _inputs(base, b=2, s=64):
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(b, s, base.config.hidden_size, generator=g, device="cuda")
    pos = torch.arange(s, device="cuda").unsqueeze(0).expand(b, s)
    gy = torch.randn(b, s, base.config.hidden_size, generator=g, device="cuda") * 1e-2
    return x, pos, base.rotary_emb.cuda()(x, pos), gy


def _run(layer, x, pos, pe, gy, unused=False):
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    params = [p for p in layer.parameters() if p.requires_grad]
    xi = x.detach().requires_grad_(True)
    ctx = fused.output_unused([layer]) if unused else torch.enable_grad()
    with ctx, torch.enable_grad():
        out = layer(xi, attention_mask=None, position_ids=pos, position_embeddings=pe,
                    use_cache=False)
        out = out[0] if isinstance(out, tuple) else out
    state = torch.cuda.get_rng_state()
    grads = torch.autograd.grad(out, [xi, *params], grad_outputs=gy)
    return out.detach(), [t.detach() for t in grads], state


def _rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()


def test_pieces_match_the_stock_modules(checkpoint):
    from poros.core.fused import lora as L
    from poros.core.fused.attention import sdpa_efficient

    model, base, layer = _layer(checkpoint)
    x, _, _, _ = _inputs(base)
    attn = layer.self_attn
    g = torch.Generator(device="cuda").manual_seed(3)

    def both(stock, ours, inputs, params, gys):
        res = []
        for fn in (stock, ours):
            ins = [t.detach().requires_grad_(True) for t in inputs]
            with torch.enable_grad():
                ys = fn(*ins)
            ys = ys if isinstance(ys, tuple) else (ys,)
            res.append((ys, torch.autograd.grad(ys, ins + params, gys[:len(ys)])))
        (y0, g0), (y1, g1) = res
        return max(_rel(a, b) for a, b in zip(y1, y0)), max(_rel(a, b) for a, b in zip(g1, g0))

    mods = [attn.q_proj, attn.k_proj, attn.v_proj]
    lins = [fused._lin(m) for m in mods]
    ps = [p for lin in lins for p in (lin.A, lin.B)]
    gys = [torch.randn(*x.shape[:-1], m.base_layer.out_features, generator=g, device="cuda")
           for m in mods]
    out, grad = both(lambda t: tuple(m(t) for m in mods),
                     lambda t: L.LoraLinears.apply(t, None, lins, -1, None, *ps), [x], ps, gys)
    assert out < 1e-5 and grad < 1e-5

    lo = fused._lin(attn.o_proj)
    r = torch.randn(x.shape, generator=g, device="cuda")
    out, grad = both(lambda t, rr: rr + attn.o_proj(t),
                     lambda t, rr: L.LoraLinears.apply(t, rr, [lo], -1, None, lo.A, lo.B)[0],
                     [x, r], [lo.A, lo.B], [gys[0].new_ones(x.shape) * 1e-2])
    assert out < 1e-5 and grad < 1e-5

    ln = layer.input_layernorm
    out, grad = both(ln, lambda t: L.RMSNorm.apply(t, ln.weight, ln.variance_epsilon), [x], [],
                     [torch.randn(x.shape, generator=g, device="cuda")])
    assert out < 1e-5 and grad < 1e-5

    q = torch.randn(2, 64, 8, 32, generator=g, device="cuda").transpose(1, 2)
    k = torch.randn(2, 64, 2, 32, generator=g, device="cuda").transpose(1, 2)
    v = torch.randn(2, 64, 2, 32, generator=g, device="cuda").transpose(1, 2)
    do = torch.randn(2, 64, 8, 32, generator=g, device="cuda")

    def math_sdpa(qq, kk, vv):
        o = torch.nn.functional.scaled_dot_product_attention(
            qq, kk, vv, is_causal=True, scale=32 ** -0.5, enable_gqa=True)
        return o.transpose(1, 2)

    out, grad = both(math_sdpa, lambda qq, kk, vv: sdpa_efficient(qq, kk, vv, 32 ** -0.5),
                     [q, k, v], [], [do])
    assert out < 1e-5 and grad < 1e-5
    runs = []
    for _ in range(2):  # the memory-efficient backward under deterministic mode: same bits
        ins = [t.detach().requires_grad_(True) for t in (q, k, v)]
        with torch.enable_grad():
            o = sdpa_efficient(*ins, 32 ** -0.5)
        runs.append([o.detach(), *torch.autograd.grad(o, ins, do)])
    assert all(torch.equal(a, b) for a, b in zip(*runs))

    # the bf16 flash kernel: q/k/v and the grad rounded to bf16 (8 bits: 2^-9 relative each),
    # fp32 accumulation; deterministic too
    from poros.core.fused.attention import flash_usable, sdpa_bf16

    assert flash_usable(q)
    out, grad = both(math_sdpa, lambda qq, kk, vv: sdpa_bf16(qq, kk, vv, 32 ** -0.5),
                     [q, k, v], [], [do])
    assert out < 1e-2 and grad < 1e-2
    runs = []
    for _ in range(2):
        ins = [t.detach().requires_grad_(True) for t in (q, k, v)]
        with torch.enable_grad():
            o = sdpa_bf16(*ins, 32 ** -0.5)
        runs.append([o.detach(), *torch.autograd.grad(o, ins, do)])
    assert all(torch.equal(a, b) for a, b in zip(*runs))


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_head_cross_entropy_matches_pytorch(weight_dtype):
    """The chunked head + cross-entropy against F.cross_entropy on the full fp32 logits, padded
    targets included; a bf16 weight (a compact head, exactly its fp32 values) through the split
    product. Deterministic."""
    from poros.core.fused.head import CHUNK_ROWS, HeadCrossEntropy

    g = torch.Generator(device="cuda").manual_seed(4)
    B, S, H, V = 3, CHUNK_ROWS // 2 + 7, 64, 1000  # rows span chunk boundaries
    h = torch.randn(B, S, H, generator=g, device="cuda")
    W = (torch.randn(V, H, generator=g, device="cuda") * 0.1).to(weight_dtype)
    labels = torch.randint(0, V, (B, S), generator=g, device="cuda")
    labels[1, S // 2:] = -100

    def stock(hh):
        logits = hh @ W.float().t()
        return torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, V), labels[:, 1:].reshape(-1))

    def ours(hh):
        return HeadCrossEntropy.apply(hh, W, labels)

    res = []
    for fn in (stock, ours, ours):
        hi = h.detach().requires_grad_(True)
        with torch.enable_grad():
            loss = fn(hi)
        res.append((loss.detach(), torch.autograd.grad(loss * 0.25, hi)[0]))
    (l0, g0), (l1, g1), (l2, g2) = res
    assert torch.equal(l1, l2) and torch.equal(g1, g2)
    assert abs(l1.item() - l0.item()) <= 1e-5 * abs(l0.item())
    assert _rel(g1, g0) < 1e-4


@pytest.mark.parametrize("bf16_attention", [False, True])
def test_layer_matches_stock_within_bf16_rounding_and_is_deterministic(checkpoint, monkeypatch,
                                                                        bf16_attention):
    """With fp32 attention, a fused layer is within 4x of how far stock moves for a 1-ulp change of
    its input (bf16 rounding of the frozen GEMMs' inputs). bf16 attention also rounds q/k/v and the
    attention's grad to bf16: within 3% (measured 1.1%; its training-loss refcheck is in
    poros.core.fused.attention's docstring)."""
    from poros.core.fused import attention

    monkeypatch.setattr(attention, "_BF16", bf16_attention)
    model, base, layer = _layer(checkpoint)
    x, pos, pe, gy = _inputs(base)
    o0, g0, s0 = _run(layer, x, pos, pe, gy)
    o_ulp, g_ulp, _ = _run(layer, x * (1 + 2 ** -23), pos, pe, gy)
    noise = max(_rel(o_ulp, o0), *(_rel(a, b) for a, b in zip(g_ulp, g0)))
    try:
        counts = fused.install(model)
        assert counts["layers"] == 3
        o1, g1, s1 = _run(layer, x, pos, pe, gy)
        o2, g2, _ = _run(layer, x, pos, pe, gy)
        o3, g3, _ = _run(layer, x, pos, pe, gy, unused=True)
    finally:
        fused.uninstall(model)
    assert torch.equal(o1, o2) and all(torch.equal(a, b) for a, b in zip(g1, g2))
    assert all(torch.equal(a, b) for a, b in zip(g1, g3))  # the skipped GEMM changes no grad
    assert torch.equal(s0, s1)  # no random draw (dropout 0)
    worst = max(_rel(o1, o0), *(_rel(a, b) for a, b in zip(g1, g0)))
    assert worst <= (0.03 if bf16_attention else max(4 * noise, 1e-5)), (worst, noise)


def test_padded_rank_grads_own_their_memory(checkpoint):
    """At a padded rank (r8 runs as r16, poros.core.fused_tiles.pad) every LoRA grad the fused
    backward returns owns its memory: a view into the padded stack kept its padding rows alive as
    long as the grad (``param.grad``, until the grads are cleared)."""
    model, base, layer = _layer(checkpoint, rank=8)
    x, pos, pe, gy = _inputs(base)
    try:
        assert fused.install(model)["layers"] == 3
        _, grads, _ = _run(layer, x, pos, pe, gy)
    finally:
        fused.uninstall(model)
    names = [n for n, p in layer.named_parameters() if p.requires_grad]
    assert len(names) == 14
    for name, g in zip(names, grads[1:], strict=True):
        assert g.untyped_storage().nbytes() == g.numel() * g.element_size(), name


def test_dropout_draws_as_peft_and_hooks_see_the_calls(checkpoint):
    model, base, layer = _layer(checkpoint, dropout=0.1)
    x, pos, pe, gy = _inputs(base)
    seen = {"stock": [], "fused": []}
    arm = ["stock"]
    hooks = [m.register_forward_hook(lambda mod, i, o: seen[arm[0]].append(o.count_nonzero()))
             for m in layer.modules() if isinstance(m, torch.nn.Dropout) and m.p > 0]
    _, _, s0 = _run(layer, x, pos, pe, gy)
    try:
        fused.install(model)
        arm[0] = "fused"
        _, _, s1 = _run(layer, x, pos, pe, gy)
    finally:
        fused.uninstall(model)
        for h in hooks:
            h.remove()
    assert torch.equal(s0, s1)  # the generator moved exactly as PEFT's draws move it
    assert len(seen["stock"]) == len(seen["fused"]) == 7
    assert [int(t) for t in seen["stock"]] == [int(t) for t in seen["fused"]]  # the same masks


@pytest.mark.parametrize("path", ["fast", "efficient"])
def test_block_major_equals_sequential_and_streamed_equals_resident(checkpoint, tmp_path, path):
    """Fused compute on, lora_dropout 0.05, padded rows: block-major (fast / efficient, block
    sizes 1 and 3, the dequantization cache on) == the sequential loop, and a fully resident run
    == the streamed one, bit for bit."""
    from poros.train.trainer import PorosTrainer

    g = torch.Generator().manual_seed(5)
    rows = []
    for n in (40, 64, 23, 64, 57, 31, 64, 12):
        ids = torch.randint(3, 250, (64,), generator=g)
        mask = torch.zeros(64, dtype=torch.long)
        mask[:n] = 1
        rows.append({"input_ids": ids * mask, "attention_mask": mask,
                     "labels": torch.where(mask.bool(), ids, torch.full_like(ids, -100))})

    def train(name, **knobs):
        tr = PorosTrainer.from_pretrained(
            str(checkpoint), quantization="nf4", lora_target_modules=TARGETS, lora_dropout=0.05,
            output_dir=str(tmp_path / name), max_steps=2, save_steps=0, progress=False,
            gradient_accumulation_steps=4, per_device_batch_size=1, learning_rate=1e-3,
            warmup_steps=0, max_seq_length=64, blockwise_arch_check="off", **knobs)
        assert tr.config.ran_compute["fused_compute"] is True  # what ran (the setting: None)
        tr.train(torch.utils.data.DataLoader(rows, batch_size=1, shuffle=False), steps=2,
                 seq_len=64)
        params = {n: p.detach().clone() for n, p in tr.model.named_parameters()
                  if p.requires_grad}
        return list(tr.history.losses), list(tr.history.grad_norms), params

    def same(a, b):
        assert a[0] == b[0] and a[1] == b[1]
        assert all(torch.equal(p, b[2][n]) for n, p in a[2].items())

    ref = train("seq", path="sequential", block_size=1)
    for bs in (1, 3):  # with the dequantization cache on
        same(ref, train(f"bm{bs}", path=path, block_size=bs, nf4_dequant_cache_mb=64))
    same(ref, train("resident", path="sequential", block_size=1, resident_blocks=3))


def test_offsets_past_two_to_the_31_elements():
    """The kernels' element offsets are int64: a LoRA output of M x N > 2^31 elements (a wide MLP
    at a long micro-batch) gets its last rows right, where int32 ``row * N`` wraps (12.9 GB)."""
    from poros.core.fused.lora import _out_fwd, kernels

    K = kernels()
    N, R = 4096, 16
    M = 2 ** 31 // N + K.FWD_BM + 3  # the last row tiles lie past 2^31 elements
    if torch.cuda.mem_get_info()[0] < (M * N * 6 + (2 << 30)):
        pytest.skip("needs ~13 GB of free VRAM")
    g = torch.Generator(device="cuda").manual_seed(7)
    base = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    tail = slice(M - 2 * K.FWD_BM, M)
    base[tail] = torch.randn(2 * K.FWD_BM, N, generator=g, device="cuda").to(torch.bfloat16)
    a = torch.zeros(M, R, device="cuda")
    a[tail] = torch.randn(2 * K.FWD_BM, R, generator=g, device="cuda")
    B = torch.randn(N, R, generator=g, device="cuda")
    out = _out_fwd(K, base, a, B, 0.5)
    want = (base[tail].double() + 0.5 * (a[tail].double() @ B.t().double())).float()
    assert _rel(out[tail], want) < 1e-6
    head = M - 2 * K.FWD_BM
    for r0 in range(0, head, 1 << 16):  # nothing landed elsewhere (a chunk at a time: 256 MB)
        assert int(torch.count_nonzero(out[r0:min(r0 + (1 << 16), head)])) == 0
