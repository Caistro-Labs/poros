"""The ``oracle`` gate: engine grads byte-equal to plain ``loss.backward()``, tiny Qwen2, CPU.

Every arm runs the real ``PorosTrainer.train`` over a real pack, so this covers the checkpoint
to adapter path, on CPU. It says nothing about Metal or about real shapes. Each mutation below is
a fault the gate exists to catch: a gate that cannot fail proves nothing.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("accelerate")

import peft  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from peft.tuners.lora import Linear as LoraLinear  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from torch.autograd.function import once_differentiable  # noqa: E402
from transformers import Qwen2ForCausalLM  # noqa: E402

from poros.core import blockwise  # noqa: E402
from poros.metal import head as head_mod  # noqa: E402
from poros.metal import nf4  # noqa: E402
from poros.metal.nf4 import Nf4Linear  # noqa: E402
from tests.metal import nf4_tiny as tiny  # noqa: E402

G = 2
STEPS = 5
MICROSTEPS = G * STEPS
DIVERGED = {"microstep_grads", "adapter", "adam"}


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture(scope="module", params=[False, True], ids=["untied", "tied"])
def checkpoint(request, tmp_path_factory):
    cfg = tiny.tiny_config(tied=request.param)
    source, pack_path = tiny.packed(tmp_path_factory.mktemp("oracle"), cfg)
    return SimpleNamespace(cfg=cfg, source=source, pack=pack_path)


def _differing(a, b) -> set[str]:
    """The surfaces that differ between two records, in any entry or in length."""
    return {name for name, value in a.surfaces().items() if value != b.surfaces()[name]}


@pytest.mark.parametrize("dropout", [0.05, 0.0])
@pytest.mark.parametrize("arm", ["mono", "plain", "hf"])
def test_engine_is_byte_equal_to_each_arm(checkpoint, tmp_path, arm, dropout):
    cfg, pack_path = checkpoint.cfg, checkpoint.pack
    loop, _ = tiny.run_arm(pack_path, cfg, "loop", tmp_path / "a", dropout=dropout)
    other, _ = tiny.run_arm(pack_path, cfg, arm, tmp_path / "b", dropout=dropout)
    assert _differing(loop, other) == set()
    for rec in (loop, other):
        assert len(rec.losses) == len(rec.logits) == len(rec.microstep_grads) == MICROSTEPS
        assert rec.nonfinite == 0


def test_single_microstep_accumulation_is_byte_equal_too(checkpoint, tmp_path):
    cfg, pack_path = checkpoint.cfg, checkpoint.pack
    loop, _ = tiny.run_arm(pack_path, cfg, "loop", tmp_path / "a", dropout=0.05, grad_accum=1)
    other, _ = tiny.run_arm(pack_path, cfg, "plain", tmp_path / "b", dropout=0.05, grad_accum=1)
    assert _differing(loop, other) == set()


def test_the_file_tier_changes_nothing(checkpoint, tmp_path):
    cfg, pack_path = checkpoint.cfg, checkpoint.pack
    ram, _ = tiny.run_arm(pack_path, cfg, "loop", tmp_path / "a", dropout=0.05)
    disk, _ = tiny.run_arm(pack_path, cfg, "loop", tmp_path / "b", dropout=0.05, file_rows=True)
    assert _differing(ram, disk) == set()


def test_declared_decode_counts(checkpoint, tmp_path):
    cfg, pack_path = checkpoint.cfg, checkpoint.pack
    matrices = cfg.num_hidden_layers * 7
    loop, _ = tiny.run_arm(pack_path, cfg, "loop", tmp_path / "a", dropout=0.05)
    mono, _ = tiny.run_arm(pack_path, cfg, "mono", tmp_path / "b", dropout=0.05)
    plain, _ = tiny.run_arm(pack_path, cfg, "plain", tmp_path / "c", dropout=0.05)
    assert loop.decodes == 3 * matrices * MICROSTEPS  # forward, remat, VJP
    assert mono.decodes == 2 * matrices * MICROSTEPS  # forward, VJP
    assert plain.decodes == matrices * MICROSTEPS  # forward; autograd holds the copy


def test_runs_repeat_exactly(checkpoint, tmp_path):
    cfg, pack_path = checkpoint.cfg, checkpoint.pack
    a, _ = tiny.run_arm(pack_path, cfg, "loop", tmp_path / "a", dropout=0.05)
    b, _ = tiny.run_arm(pack_path, cfg, "loop", tmp_path / "b", dropout=0.05)
    assert _differing(a, b) == set()


# -- The gate can fail ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def untied(tmp_path_factory):
    cfg = tiny.tiny_config()
    _, pack_path = tiny.packed(tmp_path_factory.mktemp("mutate"), cfg)
    return cfg, pack_path


_references = {}


@pytest.fixture
def references(untied, tmp_path_factory):
    """Once per module, but not a module fixture: on a Mac, conftest's CPU pick is per test."""
    cfg, pack_path = untied
    if not _references:
        out = tmp_path_factory.mktemp("references")
        for dropout in (0.05, 0.0):
            _references[dropout] = tiny.run_arm(pack_path, cfg, "plain", out, dropout=dropout)[0]
    return _references


@staticmethod
@once_differentiable
def _ascending_backward(ctx, g):
    acc = None
    for a, b in ctx.head.spans:
        p = torch.mm(g[:, a:b].contiguous(), ctx.head.chunk(a, b, g.device))
        acc = p if acc is None else acc + p
    return acc, None


@pytest.mark.parametrize("dropout", [0.05, 0.0])
@pytest.mark.parametrize("arm", ["loop", "mono"])
def test_ascending_head_sum_is_caught(untied, references, tmp_path, monkeypatch, arm, dropout):
    cfg, pack_path = untied
    monkeypatch.setattr(head_mod._ChunkedHeadFn, "backward", _ascending_backward)
    rec, _ = tiny.run_arm(pack_path, cfg, arm, tmp_path, dropout=dropout)
    assert DIVERGED <= _differing(references[dropout], rec)


def test_remat_with_no_rng_restore_is_caught_only_with_dropout(
    untied, references, tmp_path, monkeypatch
):
    cfg, pack_path = untied

    class NoRestore:
        """torch, except a restore does nothing. The fork around it stays."""

        def __getattr__(self, name):
            return getattr(torch, name)

        @staticmethod
        def set_rng_state(state):
            pass

    monkeypatch.setattr(blockwise, "torch", NoRestore())
    rec, _ = tiny.run_arm(pack_path, cfg, "loop", tmp_path / "a", dropout=0.05)
    assert DIVERGED <= _differing(references[0.05], rec)
    # lora_B is zero until the first update, so the forward cannot see it yet; the grads can.
    assert rec.microstep_grads[0] != references[0.05].microstep_grads[0]
    assert rec.losses[:G] == references[0.05].losses[:G]
    # With no dropout there is nothing to restore: the predicted miss.
    rec, _ = tiny.run_arm(pack_path, cfg, "loop", tmp_path / "b", dropout=0.0)
    assert _differing(references[0.0], rec) == set()


def test_dropped_grad_is_caught(untied, references, tmp_path, monkeypatch):
    cfg, pack_path = untied
    collect = blockwise._collect_block_trainable_params
    monkeypatch.setattr(blockwise, "_collect_block_trainable_params",
                        lambda trainer, block: collect(trainer, block)[1:])
    rec, _ = tiny.run_arm(pack_path, cfg, "loop", tmp_path, dropout=0.05)
    assert DIVERGED <= _differing(references[0.05], rec)


@staticmethod
@once_differentiable
def _wrong_vjp(ctx, g):
    return (torch.mm(g, ctx.lin.decoded()) * 1.0078125).to(g.dtype), None


@pytest.mark.parametrize("arm", ["loop", "mono"])
def test_wrong_vjp_is_caught(untied, references, tmp_path, monkeypatch, arm):
    cfg, pack_path = untied
    monkeypatch.setattr(nf4.Nf4Matmul, "backward", _wrong_vjp)
    rec, _ = tiny.run_arm(pack_path, cfg, arm, tmp_path, dropout=0.05)
    assert DIVERGED <= _differing(references[0.05], rec)


def test_wrong_forward_math_is_caught(untied, references, tmp_path, monkeypatch):
    cfg, pack_path = untied
    real = nf4.Nf4Linear.forward

    def bias_dropped(self, x):
        bias, self.bias = self.bias, None
        try:
            return real(self, x)
        finally:
            self.bias = bias

    monkeypatch.setattr(nf4.Nf4Linear, "forward", bias_dropped)
    rec, _ = tiny.run_arm(pack_path, cfg, "loop", tmp_path, dropout=0.05)
    assert "losses" in _differing(references[0.05], rec)


def test_a_one_gemm_head_is_not_the_chunked_head(untied, references, tmp_path, monkeypatch):
    """Why the plain arm chunks too: a single GEMM differs by rounding."""
    cfg, pack_path = untied

    def one_gemm(self, hidden):
        whole = self.rows[0 : self.out_features].to(torch.float32)
        logits = F.linear(hidden.reshape(-1, hidden.shape[-1]), whole)
        return logits.reshape(*hidden.shape[:-1], -1)

    monkeypatch.setattr(tiny.arms.PlainHead, "forward", one_gemm)
    rec, _ = tiny.run_arm(pack_path, cfg, "plain", tmp_path, dropout=0.05)
    assert DIVERGED <= _differing(references[0.05], rec)


# -- On Metal, where there is one. One process, so this is not the ``direct`` gate ------------


metal = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Metal")


@metal
@pytest.mark.parametrize("dropout", [0.05, 0.0])
def test_the_arms_agree_on_metal(untied, tmp_path, dropout):
    cfg, pack_path = untied
    loop, _ = tiny.run_arm(pack_path, cfg, "loop", tmp_path / "a", dropout=dropout, device="mps")
    assert loop.nonfinite == 0
    for name, kwargs in (("mono", {}), ("plain", {}), ("hf", {}), ("loop", {"file_rows": True})):
        other, _ = tiny.run_arm(pack_path, cfg, name, tmp_path / "b", dropout=dropout, device="mps",
                                **kwargs)
        assert _differing(loop, other) == set(), name


@metal
def test_no_mps_rng_restore_is_caught_on_metal(untied, tmp_path, monkeypatch):
    cfg, pack_path = untied
    plain, _ = tiny.run_arm(pack_path, cfg, "plain", tmp_path / "a", dropout=0.05, device="mps")
    monkeypatch.setattr(torch.mps, "set_rng_state", lambda *args: None)
    loop, _ = tiny.run_arm(pack_path, cfg, "loop", tmp_path / "b", dropout=0.05, device="mps")
    assert DIVERGED <= _differing(plain, loop)


# -- PEFT over the zero-size linear -----------------------------------------------------------


def test_peft_targets_exactly_the_seven_projections(checkpoint):
    model = tiny.build_model(checkpoint.pack, checkpoint.cfg)
    peft_model = tiny.attach_lora(model, dropout=0.05)
    assert set(peft_model.peft_config["default"].target_modules) == tiny.TARGETS
    wrapped = [m for m in model.modules() if isinstance(m, LoraLinear)]
    assert len(wrapped) == 7 * checkpoint.cfg.num_hidden_layers
    for layer in wrapped:
        base = layer.get_base_layer()
        assert isinstance(base, Nf4Linear) and base.weight.numel() == 0
    assert isinstance(model.lm_head, head_mod.ChunkedHead)
    assert isinstance(model.model.embed_tokens, head_mod.EmbedRows)


def test_lora_takes_the_bnb_init_path(checkpoint):
    """Init is rounded through bf16 by ``compute_dtype``, as over a bnb layer, then upcast."""
    model = tiny.build_model(checkpoint.pack, checkpoint.cfg)
    tiny.attach_lora(model, dropout=0.05)
    params = tiny.lora_params(model)
    assert len(params) == 2 * 7 * checkpoint.cfg.num_hidden_layers
    for name, param in params.items():
        assert param.dtype == torch.float32, name
        round_trip = param.detach().to(torch.bfloat16).to(torch.float32)
        assert torch.equal(round_trip, param.detach()), name


def test_every_lora_param_gets_a_gradient(checkpoint, tmp_path):
    cfg, pack_path = checkpoint.cfg, checkpoint.pack
    seen = {}

    def after_setup(trainer):
        step = trainer._adapter_optimizer_step

        def record():
            for name, param in tiny.lora_params(trainer.model).items():
                seen[name] = seen.get(name, False) or bool(param.grad.abs().sum() > 0)
            step()

        trainer._adapter_optimizer_step = record

    tiny.run_arm(pack_path, cfg, "loop", tmp_path, dropout=0.05, after_setup=after_setup)
    assert len(seen) == 2 * 7 * cfg.num_hidden_layers
    assert all(seen.values()), [n for n, v in seen.items() if not v][:3]


def test_adapter_loads_on_vanilla_weights(checkpoint, tmp_path):
    cfg, pack_path = checkpoint.cfg, checkpoint.pack
    _, trainer = tiny.run_arm(pack_path, cfg, "loop", tmp_path / "run", dropout=0.05)
    saved = tmp_path / "adapter"
    trainer.save(str(saved))

    config = json.loads((saved / "adapter_config.json").read_text())
    assert set(config["target_modules"]) == tiny.TARGETS
    on_disk = load_file(saved / "adapter_model.safetensors")
    assert len(on_disk) == 2 * 7 * cfg.num_hidden_layers
    assert all(".lora_A.weight" in k or ".lora_B.weight" in k for k in on_disk)

    vanilla = Qwen2ForCausalLM.from_pretrained(checkpoint.source, dtype=torch.bfloat16)
    loaded = peft.PeftModel.from_pretrained(vanilla, saved)
    trained = tiny.lora_params(trainer.model)
    loaded_params = dict(loaded.named_parameters())
    for name, param in trained.items():
        other = loaded_params[f"base_model.model.{name}"]
        assert torch.equal(other.detach().float(), param.detach().float()), name
    ids = torch.arange(100, 116).unsqueeze(0)
    assert torch.isfinite(loaded(input_ids=ids).logits).all()
