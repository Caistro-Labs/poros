"""A patched model (``poros.patch.hf``) hands its loop what the real decoder layer returns, not
what its annotation says: transformers annotates ``Gemma3DecoderLayer.forward`` as a tuple and
returns a bare tensor, and the next block must get the tensor.
No ``from __future__ import annotations`` here: the annotations under test must stay real objects.
"""

import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from torch import nn  # noqa: E402

DEPTH = 4
WIDTH = 8


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(WIDTH, WIDTH)
        self.base.requires_grad_(False)
        self.lora_A = nn.Linear(WIDTH, WIDTH, bias=False)

    def step(self, x):
        return torch.tanh(self.base(x)) + self.lora_A(x)


class AnnotatedTupleReturnsTensor(_Layer):  # Gemma 3
    def forward(self, hidden_states) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.step(hidden_states)


class AnnotatedTupleReturnsPair(_Layer):  # (hidden, extra), unpacked by the loop
    def forward(self, hidden_states) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.step(hidden_states), None


class AnnotatedTensorReturnsTensor(_Layer):  # Qwen
    def forward(self, hidden_states) -> torch.Tensor:
        return self.step(hidden_states)


class AnnotatedTupleReturnsOne(_Layer):  # transformers before the bare-tensor contract
    def forward(self, hidden_states) -> tuple[torch.Tensor, ...]:
        return (self.step(hidden_states),)


class NotAnnotatedReturnsTensor(_Layer):
    def forward(self, hidden_states):
        return self.step(hidden_states)


class AnnotatedTensorReturnsOne(_Layer):
    def forward(self, hidden_states) -> torch.Tensor:
        return (self.step(hidden_states),)


def _assign(layer, x):
    return layer(x)


def _first(layer, x):
    return layer(x)[0]


def _unpack(layer, x):
    x, _ = layer(x)
    return x


class _Backbone(nn.Module):
    def __init__(self, layer_cls, loop):
        super().__init__()
        self.layers = nn.ModuleList(layer_cls() for _ in range(DEPTH))
        self.loop = loop

    def forward(self, x):
        for layer in self.layers:
            x = self.loop(layer, x)
        return x


class _Model(nn.Module):
    def __init__(self, layer_cls, loop):
        super().__init__()
        self.model = _Backbone(layer_cls, loop)

    def forward(self, x):
        return self.model(x).pow(2).mean()


def _run(model, x):
    model.zero_grad(set_to_none=True)
    loss = model(x)
    loss.backward()
    return loss.detach(), {
        n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None
    }


def _assert_same(a, b):
    assert torch.equal(a[0], b[0]), (a[0].item(), b[0].item())
    assert a[1].keys() == b[1].keys() and a[1]
    for name in a[1]:
        assert torch.equal(a[1][name], b[1][name]), name


def _resident_and_patched(layer_cls, loop, block_size):
    from poros.patch.hf import patch_hf_model

    torch.manual_seed(0)
    model = _Model(layer_cls, loop)
    x = torch.randn(2, 5, WIDTH)
    resident = _run(model, x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch_hf_model(model, block_size=block_size, arch_check="off", deterministic=False)
    # Twice: the first pass runs before anything real has, the second knows what the layer returns.
    return resident, _run(model, x), _run(model, x)


@pytest.mark.usefixtures("cpu_only")
@pytest.mark.parametrize("block_size", [1, 2, 3, DEPTH])
@pytest.mark.parametrize(
    "layer_cls, loop",
    [
        (AnnotatedTupleReturnsTensor, _assign),
        (AnnotatedTupleReturnsPair, _unpack),
        (AnnotatedTupleReturnsOne, _first),
        (AnnotatedTensorReturnsTensor, _assign),
        (NotAnnotatedReturnsTensor, _assign),
    ],
    ids=lambda v: getattr(v, "__name__", ""),
)
def test_every_shape_runs_at_every_block_size(layer_cls, loop, block_size):
    resident, first, second = _resident_and_patched(layer_cls, loop, block_size)
    _assert_same(resident, first)
    _assert_same(resident, second)


@pytest.mark.usefixtures("cpu_only")
def test_a_tuple_annotated_as_a_tensor_is_returned_as_the_tuple_it_is():
    resident, first, second = _resident_and_patched(AnnotatedTensorReturnsOne, _first, 1)
    _assert_same(resident, first)
    _assert_same(resident, second)


@pytest.mark.usefixtures("cpu_only")
def test_no_real_layer_runs_more_than_the_blocks_need():
    """Learning the shape costs nothing: patch_hf_model is also the MoE dense engine, whose layers
    keep the no-grad pass's output for the backward, so an extra forward would be a change."""
    from poros.patch.hf import patch_hf_model

    torch.manual_seed(0)
    model = _Model(AnnotatedTupleReturnsTensor, _assign)
    x = torch.randn(2, 5, WIDTH)
    calls = []
    for layer in model.model.layers:
        layer.register_forward_hook(lambda m, i, o: calls.append(torch.is_grad_enabled()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch_hf_model(model, block_size=2, arch_check="off", deterministic=False)
    model(x)
    assert calls == [False] * DEPTH


@pytest.mark.usefixtures("cpu_only")
@pytest.mark.parametrize("block_size", [1, 2, 3])
def test_real_gemma3_text_runs_and_matches_resident(block_size):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from poros.patch.hf import patch_hf_model
    from tests.metal import families as fam

    torch.manual_seed(3)
    base = AutoModelForCausalLM.from_config(fam.config("gemma3_text", vocab=64))
    model = get_peft_model(base, LoraConfig(r=4, target_modules=["q_proj", "v_proj"]))
    ids = torch.randint(0, 64, (2, 12))

    def run():
        model.zero_grad(set_to_none=True)
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        return out.loss.detach(), {
            n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None
        }

    resident = run()
    patch_hf_model(model, block_size=block_size, deterministic=False)
    _assert_same(resident, run())
