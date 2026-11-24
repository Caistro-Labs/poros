"""``prepare_moe``'s guarantee label: ``validated_bitwise`` only on the receipted path."""

from __future__ import annotations

import warnings

import pytest

from tests.moe import require_tested_transformers

torch = pytest.importorskip("torch")
require_tested_transformers()

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

from poros.moe.prepare import (  # noqa: E402
    _EXPERIMENTAL_WARNING,
    MoeConfig,
    default_backends,
    prepare_moe,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="dense streaming needs CUDA")


def _model(dtype=torch.bfloat16):
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=4,
    )
    return Qwen3MoeForCausalLM(config).to(dtype)


def test_resident_is_the_reference():
    report = prepare_moe(_model(), MoeConfig(streamed=False, look_ahead=0))
    assert report.guarantee == "resident_reference"


def test_cpu_streamed_is_experimental():
    report = prepare_moe(_model(), MoeConfig(streamed=True, device="cpu"))
    assert report.guarantee == "experimental_moe_streamed"
    assert report.warnings[0] == _EXPERIMENTAL_WARNING


@pytest.mark.gpu
@cuda
def test_cuda_streamed_bf16_is_validated():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no engine warning contradicting the label
        report = prepare_moe(_model(), MoeConfig(streamed=True, device="cuda"))
    assert report.guarantee == "validated_bitwise"
    assert report.warnings == []


@pytest.mark.gpu
@cuda
def test_fp32_weights_are_experimental():
    report = prepare_moe(_model(torch.float32), MoeConfig(streamed=True, device="cuda"))
    assert report.guarantee == "experimental_moe_streamed"


@pytest.mark.gpu
@cuda
def test_explicit_default_backends_are_validated():
    # The receipt scripts pass default_backends() explicitly.
    config = MoeConfig(streamed=True, device="cuda")
    report = prepare_moe(_model(), config, backends=default_backends())
    assert report.guarantee == "validated_bitwise"
