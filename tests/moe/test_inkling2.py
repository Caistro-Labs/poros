"""Inkling 2 Mini on the CPU: its decoder as visits, streamed and resident, and its checkpoint.

- The decoder's visit schedule and config come from the released config.json.
- A tiny random model trains the same, bit for bit, with its weights streamed or resident and at
  any block size (fp32 and bf16, dropout on, three AdamW steps), and to rounding as the same
  decoder run by plain autograd; so do two whose loop spans two residual blocks, as the
  release's does (keyed dropout 0.1, ExpertAdamW in bf16, blocks of 1 and 3).
- Sabotage: without the gradient through the block attention residuals, or through the loop's
  messages, the visits no longer agree with plain autograd: the comparison above would catch it.
- Every loop layer's adapters take a gradient from every pass.
- The managed experts compute the decoder's own experts (the softcapped SwiGLU on act(gate, up)).
- The released checkpoint's every text tensor maps onto the decoder; both expert formats decode
  exactly on hand-built tensors.

The official implementation (transformers 5.24) as an oracle: tests/moe/test_inkling2_oracle.py.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from tests.moe import require_tested_transformers  # noqa: E402

require_tested_transformers()

from poros.moe.arch import MoeArch  # noqa: E402
from poros.moe.inkling2 import codec  # noqa: E402
from poros.moe.inkling2.config import text_config_of  # noqa: E402
from poros.moe.inkling2.modeling import Inkling2LoopForCausalLM  # noqa: E402
from poros.moe.inkling2.visits import VisitRunner, install_visit_streaming  # noqa: E402
from poros.moe.prepare import (  # noqa: E402
    MoeConfig,
    dense_managed_filter,
    prepare_moe,
    streaming_config,
)
from tests.moe import inkling2_tiny as tiny  # noqa: E402

HERE = Path(__file__).parent
RELEASED = json.loads((HERE.parent / "metal" / "real_configs" / "inkling_2_mini.json").read_text())


# -- config and schedule -----------------------------------------------------------------------


def test_the_released_config_reads_as_the_official_one():
    text = text_config_of(RELEASED)
    assert (text.loop_layer_index_start, text.loop_layer_index_end, text.loop_num_passes) == (
        6,
        18,
        3,
    )
    assert text.loop_gate_rank == 64 and text.moe_intermediate_size == 1152
    assert text.route_scale == 8.0 and text.conv_kernel_size == 4
    full = [i for i, t in enumerate(text.layer_types) if t == "hybrid"]
    assert full == [5, 11, 17, 23]  # local_layer_ids, read the way transformers derives them
    assert text.visits_per_forward == 48
    arch = MoeArch.from_config(text)
    assert arch.moe_layer_indices == tuple(range(24)) and arch.intermediate_size == 1152
    assert (arch.layout.experts_per_layer, arch.layout.top_k) == (128, 8)


def test_the_pass_count_and_bounds_come_from_the_config():
    loop = dict(RELEASED["text_config"]["loop_config"], num_passes=5, is_varloop=True)
    text = text_config_of(
        {**RELEASED, "text_config": {**RELEASED["text_config"], "loop_config": loop}}
    )
    assert text.loop_num_passes == 5 and text.visits_per_forward == 24 + 4 * 12
    with pytest.raises(ValueError, match="multiples"):
        tiny.config(loop_layer_index_start=1)


def test_the_schedule_visits_the_loop_once_per_pass():
    m = tiny.model()
    visits = [(layer, p) for layers, p in m.model.segments() for layer in layers]
    assert visits == [
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (2, 1),
        (3, 1),
        (2, 2),
        (3, 2),
        (4, 0),
        (5, 0),
    ]
    assert len(visits) == m.config.visits_per_forward


# -- training: streamed, resident, plain -------------------------------------------------------


def _prepared(dtype, streamed, block_size=1, dropout=0.05, shape="one_block"):
    m = tiny.model(dtype, **tiny.SHAPES[shape])
    cfg = (
        streaming_config(m.config, lora_dropout=dropout)
        if streamed
        else MoeConfig(streamed=False, look_ahead=0, lora_dropout=dropout)
    )
    prepare_moe(m, cfg)
    tiny.randomize_lora_b(m)
    if streamed:  # on the CPU prepare_moe leaves dense weights resident: stream them here
        m.get_decoder().poros_visit_runner = None
        install_visit_streaming(m, block_size, dense_managed_filter(MoeArch.from_config(m.config)))
    return m


def _train(m, steps=3, ids=None, keyed=False):
    """``keyed``: the trainer's KeyedRng keys every layer visit's dropout (a loop layer's by its
    pass too), as a MoE run does; otherwise one seed, the same in both arms."""
    from poros.core.sequential_rng import KeyedRng

    ids = tiny.ids() if ids is None else ids
    m.train()
    torch.manual_seed(2)  # both arms draw the same dropout masks
    rng = None
    if keyed:
        rng = KeyedRng(1234, torch.device("cpu"))
        rng.install(KeyedRng.layers_of(m))
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    trainable = [p for _, p in named]
    opt = _optimizer(named)
    losses, first = [], None
    for step in range(steps):
        if rng is not None:
            rng.begin(step + 1)
            rng.start(0)
        loss = m(input_ids=ids, labels=ids).loss
        loss.backward()
        if rng is not None:
            rng.end()  # fails closed where a visit went unkeyed
        if step == 0:
            first = [p.grad.clone() for p in trainable]
        opt.step()
        opt.zero_grad()
        losses.append(loss.detach())
    state = [
        t.clone() for p in trainable for t in opt.state[p].values() if isinstance(t, torch.Tensor)
    ]
    return losses, first, [p.detach().clone() for p in trainable] + state


def _optimizer(named):
    """The trainer's optimizer (poros.train.expert_adamw.build_optimizer): ExpertAdamW where any
    adapter is bf16 (the experts' in a bf16 run; the attention's are fp32), torch's AdamW where
    all are fp32."""
    from types import SimpleNamespace

    from poros.train.expert_adamw import build_optimizer

    config = SimpleNamespace(
        learning_rate=1e-2,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        weight_decay=0.01,
        seed=1234,
    )
    return build_optimizer(named, config)


@pytest.mark.parametrize("block_size", [1, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_streamed_trains_bitwise_equal_to_resident(dtype, block_size, cpu_bf16_backward):
    streamed = _prepared(dtype, True, block_size)
    runner = streamed.get_decoder().poros_visit_runner
    s = _train(streamed)
    r = _train(_prepared(dtype, False))
    assert runner.block_manager is not None and runner.loads > 0
    assert runner.visits == 3 * streamed.config.visits_per_forward
    q = streamed.model.layers[2].self_attn.q_proj.linear.base_layer.weight
    assert q.numel() == 0  # between visits the streamed weights are not on the device
    trained = [n for n, p in streamed.named_parameters() if p.requires_grad]
    assert any(".adapter.lora_" in n for n in trained) and any(
        "q_proj.linear.lora_A" in n for n in trained
    )
    for got, want in zip(s, r, strict=True):
        assert len(got) == len(want) > 0
        assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))
    assert all(g.abs().sum() > 0 for g in s[1])  # every adapter tensor took a gradient


@pytest.mark.parametrize("shape", ["two_blocks_8", "two_blocks_9"])
@pytest.mark.parametrize("block_size", [1, 3])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_a_loop_of_two_residual_blocks_trains_bitwise_streamed_or_resident(
    shape, dtype, block_size, cpu_bf16_backward
):
    """The release's loop spans two residual blocks, so a later pass reads a block state the
    pass before appended (8 layers looping 2-5 in blocks of 2; 9 looping 3-8 in blocks of 3, no
    layer after the loop). Streamed and resident train bit for bit: keyed dropout 0.1, the
    trainer's optimizer (ExpertAdamW in bf16, where the expert adapters are bf16), three steps."""
    from poros.train import expert_adamw

    streamed = _prepared(dtype, True, block_size, dropout=0.1, shape=shape)
    resident = _prepared(dtype, False, dropout=0.1, shape=shape)
    named = [(n, p) for n, p in streamed.named_parameters() if p.requires_grad]
    if dtype == torch.bfloat16:
        assert isinstance(_optimizer(named), expert_adamw.ExpertAdamW)
    runner = streamed.get_decoder().poros_visit_runner
    s = _train(streamed, keyed=True)
    r = _train(resident, keyed=True)
    assert runner.visits == 3 * streamed.config.visits_per_forward
    for got, want in zip(s, r, strict=True):
        assert len(got) == len(want) > 0
        assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))
    assert all(g.abs().sum() > 0 for g in s[1])
    undropped = _train(_prepared(dtype, False, dropout=0.0, shape=shape), steps=1, keyed=True)
    assert not torch.equal(undropped[0][0], r[0][0])  # the dropout drew


def _plain(m):
    """The same prepared model with its decoder run by plain autograd (no visit boundaries)."""
    m.get_decoder().poros_visit_runner = None
    return m


def _rel(a, b):
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-30)).item()


def _visits_vs_plain(monkeypatch=None, sabotage=None, shape="one_block"):
    visits = _prepared(torch.float32, False, dropout=0.0, shape=shape)
    plain = _plain(copy.deepcopy(visits))
    if sabotage is not None:
        monkeypatch.setattr(VisitRunner, "run_visit", sabotage(VisitRunner.run_visit))
    v = _train(visits, steps=1)
    p = _train(plain, steps=1)
    return max(_rel(a, b) for a, b in zip(v[1], p[1], strict=True)), _rel(v[0][0], p[0][0])


#: Visits and plain autograd sum the gradients that fan out across visits in another order.
PLAIN_RTOL = 1e-5


@pytest.mark.parametrize("shape", list(tiny.SHAPES))
def test_visits_agree_with_plain_autograd(shape):
    grads, loss = _visits_vs_plain(shape=shape)
    assert loss == 0.0  # the forward is the same computation
    assert grads < PLAIN_RTOL


def _detach(which):
    def sabotage(run_visit):
        def wrapped(self, key, tensors):
            tensors = list(tensors)
            for i in which(key, len(tensors)):
                tensors[i] = tensors[i].detach()
            return run_visit(self, key, tensors)

        return wrapped

    return sabotage


@pytest.mark.parametrize("shape", list(tiny.SHAPES))
def test_losing_the_block_state_gradient_is_caught(monkeypatch, shape):
    grads, _ = _visits_vs_plain(
        monkeypatch, _detach(lambda key, n: range(1, 1 + key.n_blocks)), shape=shape
    )
    assert grads > 1e3 * PLAIN_RTOL


@pytest.mark.parametrize("shape", list(tiny.SHAPES))
def test_losing_the_message_gradient_is_caught(monkeypatch, shape):
    def messages(key, n):
        return range(1 + key.n_blocks, n) if key.n_msg_blocks is not None else ()

    grads, _ = _visits_vs_plain(monkeypatch, _detach(messages), shape=shape)
    assert grads > 1e3 * PLAIN_RTOL


def test_every_pass_adds_to_a_loop_layer_s_adapters(monkeypatch):
    calls: dict = {}
    real = VisitRunner.visit_params

    def counting(self, key):
        calls[key.layer] = calls.get(key.layer, 0) + 1  # asked once a visit, in its backward
        return real(self, key)

    monkeypatch.setattr(VisitRunner, "visit_params", counting)
    _train(_prepared(torch.float32, False), steps=1)
    assert calls == {0: 1, 1: 1, 2: 3, 3: 3, 4: 1, 5: 1}


def test_a_looped_adapter_is_keyed_once_an_update(monkeypatch, cpu_bf16_backward):
    """A bf16 run: the attention adapters are fp32, the experts' bf16, so the trainer's optimizer
    is ExpertAdamW over both. Its rounding key is (run seed, update, parameter name): an adapter
    on a looped layer, though its gradient comes from three visits, is one parameter updated once
    an update, keyed once, the same in a streamed and a resident run."""
    from poros.train import expert_adamw

    keys: list = []
    real = expert_adamw.update_key
    monkeypatch.setattr(
        expert_adamw,
        "update_key",
        lambda seed, update, name: keys.append((seed, update, name)) or real(seed, update, name),
    )
    runs = []
    for streamed in (True, False):
        m = _prepared(torch.bfloat16, streamed)
        named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
        dtypes = {("expert" if ".adapter." in n else "attention"): p.dtype for n, p in named}
        assert dtypes == {"expert": torch.bfloat16, "attention": torch.float32}
        assert isinstance(_optimizer(named), expert_adamw.ExpertAdamW)
        keys.clear()
        runs.append((_train(m, steps=2), list(keys)))
    (streamed_run, streamed_keys), (resident_run, resident_keys) = runs
    assert streamed_keys == resident_keys
    looped = [k for k in streamed_keys if k[2].startswith("model.layers.2.mlp.adapter.")]
    assert sorted(looped) == sorted(
        (1234, u, f"model.layers.2.mlp.adapter.{f}")
        for u in (1, 2)
        for f in ("lora_A_gate_up", "lora_B_gate_up", "lora_A_down", "lora_B_down")
    )
    assert len(streamed_keys) == len(set(streamed_keys))  # each (update, name) once
    for got, want in zip(streamed_run, resident_run, strict=True):
        assert all(torch.equal(a, b) for a, b in zip(got, want, strict=True))


# -- experts --------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_managed_experts_compute_the_decoder_s_experts(dtype):
    from poros.moe.modules import ManagedMoeBlock

    m = tiny.model(dtype)
    blocks = {i: copy.deepcopy(layer.mlp) for i, layer in enumerate(m.model.layers)}
    prepare_moe(m, MoeConfig(streamed=False, look_ahead=0))
    hidden = torch.randn(2, 24, 64, dtype=dtype)
    for ordinal, reference in blocks.items():
        managed = m.model.layers[ordinal].mlp
        assert isinstance(managed, ManagedMoeBlock) and managed.act_takes_up
        managed.residency.ensure([(ordinal, e) for e in range(managed.num_experts)])
        with torch.no_grad():  # the weights training sees: the NF4 round trip
            for tensor in ("gate_up_proj", "down_proj"):
                for e in range(managed.num_experts):
                    slab = managed.residency.resident_qbytes((ordinal, e))
                    weight = managed.store.dequantize_tensor((ordinal, e), tensor, slab)
                    getattr(reference.experts, tensor)[e] = weight
            tol = dict(rtol=1e-5, atol=1e-6) if dtype == torch.float32 else {}
            torch.testing.assert_close(managed(hidden), reference(hidden), **tol)


def test_the_grouped_compute_is_refused():
    m = tiny.model()
    with pytest.raises(ValueError, match="act\\(gate, up\\)"):
        prepare_moe(m, MoeConfig(streamed=False, look_ahead=0, expert_compute="grouped"))


# -- the released checkpoint --------------------------------------------------------------------


def test_every_released_text_tensor_maps_onto_the_decoder():
    names = [
        line
        for line in (HERE / "data" / "inkling_2_mini_tensors.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    targets = codec.checkpoint_targets(names)
    with torch.device("meta"):
        m = Inkling2LoopForCausalLM(text_config_of(RELEASED))
    held = {**dict(m.named_parameters()), **dict(m.named_buffers())}
    assert set(targets) == set(held)
    experts = [t for t in targets if t.endswith((".gate_up_proj", ".down_proj"))]
    assert len(experts) == 48
    nvfp4 = targets["model.layers.0.mlp.experts.gate_up_proj"]
    assert sorted(nvfp4) == sorted(
        f"model.llm.layers.0.mlp.experts.w13_{p}"
        for p in ("weight", "weight_scale", "weight_amax", "weight_outer_scale_denominator")
    )
    assert len(targets["model.layers.23.mlp.experts.down_proj"]) == 2  # MXFP8: values and scales
    assert targets["model.layers.1.attention_input.query"] == ["model.llm.attn_res.attn_queries.0"]
    assert targets["model.layers.6.q_message_gate.message_gate_proj.weight"] == [
        "model.llm.layers.6.controllers.attn.gain_weight"
    ]
    assert targets["model.layers.5.self_attn.k_proj.sconv.static_weight"] == [
        "model.llm.layers.5.attn.k_sconv.static_w"
    ]
    assert targets["model.layers.4.self_attn.k_proj.sconv.conv1d.weight"] == [
        "model.llm.layers.4.attn.k_sconv.weight"
    ]
    assert targets["lm_head.weight"] == ["model.llm.unembed.weight"]


def test_nvfp4_decodes_hand_built_bytes():
    # two experts, two rows, 32 columns: two 16-value blocks per row, each its own scale
    codes = torch.randint(0, 16, (2, 2, 32), generator=torch.Generator().manual_seed(0))
    values = torch.tensor(codec.E2M1_VALUES)[codes]
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)  # even element low
    scales = torch.tensor([[[0.5, 2.0], [1.5, 0.25]], [[4.0, 1.0], [0.75, 3.0]]])
    amax = torch.tensor([6.0, 12.0])
    denominator = torch.tensor([3.0])
    want = values.view(2, 2, 2, 16) * scales[..., None] * (amax / denominator)[:, None, None, None]
    got = codec.decode_nvfp4(
        packed, scales.to(torch.float8_e4m3fn), amax, denominator, torch.float32
    )
    assert torch.equal(got, want.view(2, 2, 32))


def test_mxfp8_decodes_hand_built_bytes():
    values = torch.tensor([0.5, -1.0, 1.75, 448.0, -0.015625, 0.0, 3.0, -6.0]).repeat(1, 2, 8)
    values = values.view(1, 2, 64).to(torch.float8_e4m3fn)
    exponents = torch.tensor([[[127, 130], [120, 127]]], dtype=torch.uint8)
    want = (
        values.float().view(1, 2, 2, 32) * torch.tensor([[[1.0, 8.0], [2.0**-7, 1.0]]])[..., None]
    )
    got = codec.decode_mxfp8(values, exponents, torch.float32)
    assert torch.equal(got, want.view(1, 2, 64))


def test_the_gate_up_rows_deinterleave():
    rows = torch.arange(8.0).view(1, 8, 1)  # g0 u0 g1 u1 g2 u2 g3 u3
    assert codec.deinterleave_gate_up(rows).view(-1).tolist() == [0, 2, 4, 6, 1, 3, 5, 7]


def test_the_message_queries_merge_as_q_k_v_mlp():
    q_mlp = torch.arange(4.0).view(4, 1)  # layer 0: q 0, mlp 1; layer 1: q 2, mlp 3
    k_v = torch.arange(10.0, 14.0).view(4, 1)
    merged = codec.merge_message_queries(q_mlp, k_v).squeeze(-1)
    assert merged.tolist() == [[0, 10, 11, 1], [2, 12, 13, 3]]


def _official_layout(**overrides) -> dict:
    """The tiny model's config as the official transformers 5.24 class saves it
    (``save_pretrained``): model_type inkling2_loop_text, moe_intermediate_size and the
    loop_layer_index_* / loop_num_passes fields, no intermediate_size, no loop_config. Built
    from the field names; the official class is not importable under the pinned transformers."""
    cfg = {**tiny.CONFIG, **overrides, "model_type": "inkling2_loop_text"}
    assert "intermediate_size" not in cfg and "loop_config" not in cfg
    return cfg


def test_a_config_in_the_official_layout_is_sized_and_counted():
    """The size check and the visit count read the official class's layout as they read the
    release's: the same sizes, and the loop's visits (the tiny model: 6 layers, 10 visits; the
    release's shape, 24 layers looping 6-17 three times: 48)."""
    from poros.plan.estimate import _sizes

    official = _sizes(_official_layout())
    assert official is not None and (official.layers, official.visits) == (6, 10)
    release = {
        k: v
        for k, v in tiny.CONFIG.items()
        if k != "moe_intermediate_size" and not k.startswith("loop_")
    }
    release.update(
        intermediate_size=tiny.CONFIG["moe_intermediate_size"],
        model_type="inkling2_loop_text",
        loop_config={"core_start": 2, "core_end": 4, "num_passes": 3},
    )
    assert _sizes({"model_type": "inkling2_loop", "text_config": release}) == official
    full = _sizes(
        _official_layout(
            num_hidden_layers=24,
            layer_types=["hybrid_sliding", "hybrid"] * 12,
            loop_layer_index_start=6,
            loop_layer_index_end=18,
        )
    )
    assert (full.layers, full.visits) == (24, 48)
    assert _sizes(_official_layout(loop_num_passes=None)) is None  # unknown, never an error


@pytest.mark.parametrize(
    "loop",
    [
        {"num_passes": None, "core_start": 2, "core_end": 4},
        {"num_passes": 3, "core_start": "two", "core_end": 4},
        {"num_passes": 3, "core_start": 2, "core_end": [4]},
    ],
)
def test_a_loop_config_that_does_not_read_sizes_as_unknown(loop):
    """Any model's config.json: a loop_config whose passes or bounds are no numbers makes the
    sizes unknown (None), as any other field that does not read, never an error before the
    download."""
    from poros.plan.estimate import _sizes

    qwen2 = {
        "model_type": "qwen2",
        "hidden_size": 64,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "vocab_size": 128,
        "intermediate_size": 128,
    }
    assert _sizes(qwen2) is not None
    assert _sizes({**qwen2, "loop_config": loop}) is None


def test_the_pre_run_checks_count_a_forward_s_visits():
    """The sequential loop saves one input per visit: a looped decoder's saved state grows with
    its visits (the tiny model: 6 layers, 10 visits), not its layer count; the same config
    without its loop saves one per layer, as every other model."""
    from poros.plan.estimate import _BOUNDARY_BYTES, _knobs, _sizes, _variable

    cfg = tiny.config()  # as config.json words it: the experts' width is intermediate_size
    text = {
        **cfg.to_dict(),
        "intermediate_size": cfg.moe_intermediate_size,
        "model_type": "inkling2_loop_text",
        "loop_config": {"core_start": 2, "core_end": 4, "num_passes": 3},
    }
    looped = _sizes({"model_type": "inkling2_loop", "text_config": text})
    plain = _sizes(
        {
            "model_type": "inkling2_loop",
            "text_config": {k: v for k, v in text.items() if not k.startswith("loop_")},
        }
    )
    assert (looped.layers, looped.visits) == (6, 10) and plain.visits == 6
    key = {
        "block_size": None,
        "entry": "moe",
        "batch": 2,
        "grad_accum": 1,
        "seq_len": 16,
        "lora_rank": 16,
        "lora_dropout": 0.0,
        "offload": False,
    }
    kl, kp = _knobs(key, "sequential", looped), _knobs(key, "sequential", plain)
    assert (kl.blocks, kp.blocks) == (10, 6)
    one = 2 * 16 * looped.h * _BOUNDARY_BYTES  # one visit's saved input, one micro-batch
    grown = _variable(kl, looped, "sequential") - _variable(kp, plain, "sequential")
    assert grown == 4 * one  # the 4 extra visits' inputs, and nothing else


def test_the_dense_route_leaves_it_alone_before_shrinking_it():
    """tests/test_validate_gates walks every real config through the dense route's CPU rung: a
    mixture of experts by its registry entry takes the MoE exit, in words, before anything
    shrinks it (a looped one does not shrink)."""
    from poros.preflight.arch import MOE_TARGET
    from poros.validate import tiny_route

    got = tiny_route.prove(RELEASED)
    assert got["moe"] and got["ok"] is None and not got["routes"]
    assert got["why"] == MOE_TARGET


def test_shrink_refuses_a_looped_config_in_words():
    from poros.metal import tiny as metal_tiny

    with pytest.raises(ValueError, match="looped decoder .* does not shrink"):
        metal_tiny.shrink(RELEASED, hidden=32)


def test_its_config_reads_in_a_fresh_interpreter():
    """Whatever builds a transformers config from its config.json registers the family itself:
    no import order makes "Unrecognized model identifier: inkling2_loop"."""
    import subprocess
    import sys

    path = HERE.parent / "metal" / "real_configs" / "inkling_2_mini.json"
    code = (
        "import json, sys\n"
        "from poros.metal.layout import config_of\n"
        "cfg = config_of(json.load(open(sys.argv[1])))\n"
        "print(type(cfg).__name__, cfg.text_config.loop_num_passes)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", code, str(path)],
        capture_output=True,
        text=True,
        cwd=HERE.parents[1],
        timeout=600,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.split() == ["Inkling2LoopConfig", "3"]


def test_the_size_checks_size_the_format_the_run_records(monkeypatch):
    """The checks before a run size the experts as the run will hold them (the one decision,
    poros.moe.inkling2.formats): released on the release, nf4 when the run says so, and nf4 for a
    merged export (its config.json names no quantization), whatever the variable is unset to."""
    from poros.plan.estimate import _sizes, moe_host_ram

    monkeypatch.delenv("POROS_INKLING_EXPERTS", raising=False)
    weights = 24 * 128 * 3 * 2304 * 1152
    released = _sizes(RELEASED, "released").experts_nf4
    nf4 = _sizes(RELEASED, "nf4").experts_nf4
    assert nf4 == weights * 33 // 64  # 12.61 GB
    assert released == int(weights * (20 * 9 / 16 + 4 * 33 / 32) / 24)  # 15.67 GB
    assert _sizes(RELEASED).experts_nf4 == released  # unset: the release keeps its bytes
    merged = {k: v for k, v in RELEASED.items() if k != "quantization_config"}
    assert _sizes(merged).experts_nf4 == nf4  # an unquantized save is held in NF4
    host = {
        f: moe_host_ram(RELEASED, "bfloat16", expert_format=f).value for f in ("released", "nf4")
    }
    assert host["released"] > host["nf4"]
    monkeypatch.setenv("POROS_INKLING_EXPERTS", "nf4")
    assert _sizes(RELEASED).experts_nf4 == nf4  # the variable, where the run sets nothing
