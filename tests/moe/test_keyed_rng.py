"""A MoE model's dropout masks are keyed by (seed, update, micro-batch, layer).

poros.core.sequential_rng.KeyedRng: every decoder layer's forward, and every recompute of it,
starts from a generator state that is a pure function of those, so no draw (an expert's dropout
over the rows routed to it) moves another layer's masks. Streamed and fully resident runs, and
a run stopped and resumed, train the same update at the default dropout: on the CPU, where a MoE
model's dense layers stay resident and only its experts stream, and on CUDA, where its dense
blocks stream too and the run takes block-major accumulation (more of that:
tests/moe/test_block_major_families.py).
"""
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from tests.moe import fused_mask_digests, require_tested_transformers

torch = pytest.importorskip("torch")
require_tested_transformers()

from poros.core.errors import PorosCancelledError  # noqa: E402
from poros.core.sequential_rng import KeyedRng  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402
from tests.moe.test_native_entry import VOCAB, _model  # noqa: E402


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen3_moe_keyed")
    _model().save_pretrained(path)
    return path


def _padded(n=20, rows=2):
    """Rows of 5 to 16 tokens, ``rows`` to a micro-batch, each padded to its longest."""
    g = torch.Generator().manual_seed(1)
    seqs = [torch.randint(3, VOCAB, (int(k),), generator=g)
            for k in torch.randint(5, 17, (n,), generator=g)]

    def collate(items):
        width = max(len(x) for x in items)
        ids = torch.zeros(len(items), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, x in enumerate(items):
            ids[i, :len(x)], mask[i, :len(x)] = x, 1
        return {"input_ids": ids, "attention_mask": mask,
                "labels": ids.masked_fill(mask == 0, -100)}

    # shuffled by its own seeded generator, as the trainer's own loader is: resumable
    return torch.utils.data.DataLoader(seqs, batch_size=rows, shuffle=True, collate_fn=collate,
                                       generator=torch.Generator().manual_seed(0))


def _resident(monkeypatch):
    """The MoE model fully resident: experts too (MoeConfig.streamed=False)."""
    from poros.moe import prepare

    real = prepare.streaming_config
    monkeypatch.setattr(prepare, "streaming_config", lambda *a, **k: dataclasses.replace(
        real(*a, **k), streamed=False, look_ahead=0, capacity_bytes=0))


def _trainer(checkpoint, out, **kw):
    knobs = dict(output_dir=str(out), max_steps=4, save_steps=0, progress=False,
                 gradient_accumulation_steps=4, per_device_batch_size=2, learning_rate=1e-2,
                 warmup_steps=0, deterministic=False)
    return PorosTrainer.from_pretrained(str(checkpoint), **{**knobs, **kw})


def _rng():
    return [torch.get_rng_state(), *([torch.cuda.get_rng_state()] if torch.cuda.is_available()
                                     else [])]


def _state(tr):
    params = {n: p.detach().clone() for n, p in tr.model.named_parameters() if p.requires_grad}
    return list(tr.history.losses), list(tr.history.grad_norms), params, _rng()


def _same(a, b):
    assert a[0] == b[0] and a[1] == b[1]  # every step's loss and grad norm
    assert a[2].keys() == b[2].keys() and a[2]
    for name, p in a[2].items():
        assert torch.equal(p, b[2][name]), name
    assert all(map(torch.equal, a[3], b[3]))  # the generators, where what follows starts


def test_a_layer_s_state_is_a_pure_function_of_its_key():
    keyed = KeyedRng(7, torch.device("cpu"))
    layers = [torch.nn.Identity() for _ in range(3)]
    keyed.install(layers)
    x = torch.zeros(1)

    def drawn(update, micro, layer):
        keyed.begin(update)
        keyed.start(micro)
        layers[layer](x)
        return torch.rand(4)

    first = drawn(2, 1, 2)
    torch.rand(100)  # whatever ran in between
    assert torch.equal(drawn(2, 1, 2), first)
    assert not torch.equal(drawn(2, 1, 1), first)
    assert not torch.equal(drawn(2, 0, 2), first)
    assert not torch.equal(drawn(3, 1, 2), first)
    keyed.end(check=False)  # no pass under way: a forward leaves the generator alone
    before = torch.get_rng_state()
    layers[0](x)
    assert torch.equal(torch.get_rng_state(), before)

    def whole(update):
        keyed.begin(update)
        for g in range(2):
            keyed.start(g)
            for layer in layers:
                layer(x)
        keyed.end()
        return torch.get_rng_state()

    assert torch.equal(whole(2), whole(2))  # an update's end, keyed too


def test_a_layer_the_forward_never_keyed_stops_the_update():
    """Fail closed: each micro-batch must key every layer (hooks on modules the forward never
    runs leave the masks unkeyed)."""
    keyed = KeyedRng(7, torch.device("cpu"))
    layers = [torch.nn.Identity() for _ in range(3)]
    keyed.install(layers)
    keyed.begin(0)
    keyed.start(0)
    layers[0](torch.zeros(1))
    layers[1](torch.zeros(1))  # layer 2 never runs
    with pytest.raises(RuntimeError, match="micro-batch 0, layer 2"):
        keyed.end()


def test_a_draw_between_layers_moves_no_mask(checkpoint, tmp_path):
    """Every layer starts from its key, whatever drew before it: a draw after each layer (and
    before the first) leaves the update as it was. Without the keys it would move every mask
    after it."""
    runs = []
    for name in ("plain", "drawn"):
        tr = _trainer(checkpoint, tmp_path / name)
        if name == "drawn":
            def draw(*args):
                torch.rand(3)
                if torch.cuda.is_available():
                    torch.rand(3, device="cuda")
            for layer in KeyedRng.layers_of(tr.model):
                layer.register_forward_hook(draw)
            tr.model.get_input_embeddings().register_forward_hook(draw)
        tr.train(_padded(), steps=2, seq_len=16)
        runs.append(_state(tr))
    _same(*runs)


@pytest.mark.skipif(torch.cuda.is_available(), reason="a CPU model (on CUDA: the tests below)")
def test_the_keys_sit_on_the_layers_a_streamed_model_runs():
    """A model whose dense blocks stream (poros.patch.hf.patch_hf_model) yields collector proxies
    when its layer list is iterated; the keys go on the layers it runs, which fire in the forward
    and again in each block's recompute."""
    from poros.patch.hf import patch_hf_model
    from tests.test_train_loop_cpu import _tiny_peft_qwen2

    model, _ = _tiny_peft_qwen2(torch, dropout=0.1)
    patch_hf_model(model, block_size=2)
    layers = KeyedRng.layers_of(model)
    assert [type(m).__name__ for m in layers] == ["Qwen2DecoderLayer"] * 4
    fired = []
    for i, m in enumerate(layers):
        m.register_forward_pre_hook(lambda mod, args, i=i: fired.append(i))
    model.train()
    ids = torch.randint(0, 256, (1, 8))
    model(input_ids=ids, labels=ids).loss.backward()
    assert fired == [0, 1, 2, 3, 2, 3, 0, 1]


@pytest.mark.parametrize("G", [4, 1])
def test_streamed_and_fully_resident_train_the_same_update_at_dropout(
        checkpoint, tmp_path, monkeypatch, G):
    """On CUDA the streamed run is block-major, at four micro-batches a step and at one."""
    runs = []
    for name in ("streamed", "resident"):
        if name == "resident":
            _resident(monkeypatch)
        tr = _trainer(checkpoint, tmp_path / name, gradient_accumulation_steps=G)
        assert tr.config.lora_dropout == 0.05  # the default
        tr.train(_padded(), steps=4, seq_len=16)
        if name == "streamed" and torch.cuda.is_available():
            assert tr.ran_path == "efficient"
        runs.append(_state(tr))
    _same(*runs)


def test_without_the_keys_the_masks_differ(checkpoint, tmp_path, monkeypatch):
    """The control: an expert dropout's masks matter, and the keys set them (here the hooks run
    but set nothing, so the masks come from the stream as it runs)."""
    runs = []
    for name in ("keyed", "loose"):
        if name == "loose":
            monkeypatch.setattr(KeyedRng, "_put", lambda self, *parts: None)
        tr = _trainer(checkpoint, tmp_path / name)
        tr.train(_padded(), steps=2, seq_len=16)
        runs.append(_state(tr))
    assert runs[0][0] != runs[1][0]


@pytest.mark.parametrize("rows", [20, 16], ids=["mid-epoch", "at-an-epoch-start"])
def test_a_resumed_run_is_the_uninterrupted_one(checkpoint, tmp_path, rows):
    """Stopped after its second update and resumed: the uninterrupted run's steps, adapters and
    generator. 20 rows are 10 micro-batches: a new epoch, and its shuffle, starts inside the
    third update."""
    whole = _trainer(checkpoint, tmp_path / "whole")
    whole.train(_padded(n=rows), steps=4, seq_len=16)
    part = _trainer(checkpoint, tmp_path / "part")
    with pytest.raises(PorosCancelledError) as stopped:
        part.train(_padded(n=rows), steps=4, seq_len=16, should_stop=lambda: part.global_step == 1)
    resumed = _trainer(checkpoint, tmp_path / "part")
    resumed.train(_padded(n=rows), steps=4, seq_len=16, resume_from=str(stopped.value.checkpoint))
    got, want = _state(resumed), _state(whole)
    assert got[0] == want[0][2:] and got[1] == want[1][2:]
    for name, p in want[2].items():
        assert torch.equal(got[2][name], p), name
    assert all(map(torch.equal, got[3], want[3]))


@pytest.mark.parametrize("compute", [None, "reference"])
def test_every_expert_recompute_draws_its_forward_s_masks(checkpoint, tmp_path, monkeypatch,
                                                         compute):
    """An expert's adapter dropout (the MoE path's adapters are the experts') runs in the
    forward and again in the expert function's rematerializing backward: the same mask both
    times, and the same masks resident. The default compute (fused on CUDA: its masks are drawn
    in poros.moe.fused.Masks, probed there a micro-batch at a time) and the reference compute
    (its masks are nn.Dropout calls a hook can see). Under fused, the streamed run batches each
    layer's micro-batches (poros.moe.batching), whose first visit checks the decoder layer
    against its own forward and so draws its first micro-batch's masks once more, from the same
    state: the same masks, once more often than resident, which does not batch. So there the
    masks are compared as sets (how often each is drawn: at least twice)."""
    masks = {}
    for name in ("streamed", "resident"):
        if name == "resident":
            _resident(monkeypatch)
        tr = _trainer(checkpoint, tmp_path / name,
                      **({} if compute is None else {"moe_expert_compute": compute}))
        fused = tr.config.ran_compute["moe_expert_compute"] == "fused"
        if compute is None and not fused:
            pytest.skip("the default is the reference compute here: its arm covers it")
        got: dict = {}
        with pytest.MonkeyPatch.context() as mp:
            if fused:
                got = fused_mask_digests(mp)
            else:
                for module_name, m in tr.model.named_modules():
                    if isinstance(m, torch.nn.Dropout) and m.p > 0:
                        def hook(mod, inputs, out, module_name=module_name):
                            kept = ((out != 0) | (inputs[0] == 0)).cpu().numpy().tobytes()
                            got.setdefault(module_name, []).append(
                                hashlib.sha256(kept).hexdigest())
                        m.register_forward_hook(hook)
            tr.train(_padded(), steps=2, seq_len=16)
        assert len(got) == 2  # one expert adapter per layer
        for module_name, calls in got.items():
            assert min(calls.count(d) for d in calls) >= 2, module_name
        masks[name] = {n: set(c) if fused else sorted(c) for n, c in got.items()}
    assert masks["streamed"] == masks["resident"]


def test_pass_zero_keys_exactly_as_before():
    """A layer's key gained an optional loop pass (Inkling 2 runs layers several times a forward).
    Every other model's layers run once, in pass 0: their key, so their certified masks, must be
    the one Poros always used, the seed of (layer tag, update, micro-batch, layer index)."""
    import struct

    from poros.core.sequential_rng import _LAYER

    keyed = KeyedRng(7, torch.device("cpu"))
    layer = torch.nn.Identity()
    keyed.install([torch.nn.Identity(), layer])
    keyed.begin(3)
    keyed.start(1)
    layer(torch.zeros(1))
    data = struct.pack("<5q", 7, _LAYER, 3, 1, 1)  # the key as it was written before passes
    before = int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little") >> 1
    assert keyed.key(*keyed.layer_key(1)) == before
    assert keyed.layer_key(1, 0) == (_LAYER, 3, 1, 1)
    drawn = torch.rand(4)
    torch.manual_seed(before)
    assert torch.equal(drawn, torch.rand(4))


def test_each_loop_pass_draws_its_own_masks():
    keyed = KeyedRng(7, torch.device("cpu"))
    layer = torch.nn.Identity()
    keyed.install([layer])
    keyed.begin(0)
    keyed.start(0)
    draws = []
    for visit_pass in range(3):
        layer.poros_visit_pass = visit_pass
        layer(torch.zeros(1))
        draws.append(torch.rand(4))
    assert not torch.equal(draws[0], draws[1]) and not torch.equal(draws[1], draws[2])
    layer.poros_visit_pass = 1
    layer(torch.zeros(1))
    assert torch.equal(torch.rand(4), draws[1])  # still a pure function of its key
    keyed.end()
