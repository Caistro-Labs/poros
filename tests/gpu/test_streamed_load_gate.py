"""GPU byte-identity gate for the block-streamed NF4 load (EXPERIMENTAL).

The streamed loader must produce a base byte-identical to the standard
loader's: packed 4-bit data, quant-state tensors, and every unquantized
parameter/buffer compared with ``torch.equal``. A pass certifies the
streamed path feeds the parity-gated runtime the exact same numbers.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.gpu

DEFAULT_MODEL = os.environ.get("POROS_GPU_TEST_MODEL", "Qwen/Qwen2.5-0.5B")


def test_streamed_load_is_byte_identical():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    pytest.importorskip("bitsandbytes")

    import bitsandbytes as bnb

    from poros.train.model import _load_model

    standard = _load_model(DEFAULT_MODEL, torch.bfloat16).to("cpu")
    streamed = _load_model(
        DEFAULT_MODEL, torch.bfloat16,
        nf4_load_strategy="block_streamed",
    )

    ref = dict(standard.named_parameters())
    got = dict(streamed.named_parameters())
    assert sorted(ref) == sorted(got)

    quantized_checked = 0
    for name, a in ref.items():
        b = got[name]
        assert a.dtype == b.dtype, name
        assert a.shape == b.shape, name
        assert torch.equal(a.detach().cpu(), b.detach().cpu()), name
        if isinstance(a, bnb.nn.Params4bit):
            qa = a.quant_state.as_dict(packed=True)
            qb = b.quant_state.as_dict(packed=True)
            assert sorted(qa) == sorted(qb), name
            for key in qa:
                va, vb = qa[key], qb[key]
                if isinstance(va, torch.Tensor):
                    assert torch.equal(va.cpu(), vb.cpu()), f"{name}::{key}"
                else:
                    assert va == vb, f"{name}::{key}"
            quantized_checked += 1
    assert quantized_checked > 0, "no quantized parameters compared"
