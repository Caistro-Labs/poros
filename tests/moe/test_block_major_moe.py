"""Block-major accumulation on the MoE path is the sequential MoE update, bit for bit (CUDA):
losses, adapters, and ExpertAdamW's whole state (both moments, the remainder, the step)."""

from __future__ import annotations

import pytest

from tests.moe import require_tested_transformers

pytestmark = pytest.mark.gpu  # CUDA only: the GPU lane (-m gpu) runs it
torch = pytest.importorskip("torch")
require_tested_transformers()
if not torch.cuda.is_available():
    pytest.skip("block-major MoE composes with the streamed dense engine (CUDA)",
                allow_module_level=True)

from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM  # noqa: E402

from poros.moe.modules import ManagedMoeBlock  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402

VOCAB = 64


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        num_experts=8, num_experts_per_tok=4,
    )
    path = tmp_path_factory.mktemp("qwen3_moe_bm")
    Qwen3MoeForCausalLM(config).to(torch.bfloat16).save_pretrained(path)
    return path


def _loader():
    g = torch.Generator().manual_seed(1)
    rows = []
    # varied lengths: masks and routes differ per microstep
    for seq in (16, 11, 16, 7, 13, 16, 9, 16):
        x = torch.randint(0, VOCAB, (seq,), generator=g)
        rows.append({"input_ids": x, "labels": x.clone(), "attention_mask": torch.ones_like(x)})
    return torch.utils.data.DataLoader(rows, batch_size=1)


def _run(checkpoint, tmp_path, **overrides):
    knobs = dict(output_dir=str(tmp_path), max_steps=2, save_steps=0, progress=False,
                 gradient_accumulation_steps=4, lora_dropout=0.0, learning_rate=1e-3,
                 warmup_steps=0)
    trainer = PorosTrainer.from_pretrained(str(checkpoint), **{**knobs, **overrides})
    trainer.train(_loader())
    named = [(n, p) for n, p in trainer.model.named_parameters() if p.requires_grad]
    params = {n: p.detach().clone() for n, p in named}
    state = {f"{n}.{k}": v.detach().clone() for n, p in named
             for k, v in trainer.optimizer.state[p].items()}
    hits = sum(m.remat_hits for m in trainer.model.modules() if isinstance(m, ManagedMoeBlock))
    return trainer.history.losses, params, hits, state


@pytest.mark.parametrize("store", ["ram", "ssd"])
def test_block_major_moe_equals_sequential(checkpoint, tmp_path, store):
    extra = {}
    if store == "ssd":
        extra.update(store="ssd", store_path=str(tmp_path / "sss"))
    # sequential pinned too: on CUDA block-major accumulation is the default
    seq_losses, seq_params, _, seq_state = _run(checkpoint, tmp_path / "seq", path="sequential",
                                                **extra)
    bm_losses, bm_params, hits, bm_state = _run(checkpoint, tmp_path / "bm", path="efficient",
                                                **extra)
    assert seq_losses == bm_losses
    assert seq_params.keys() == bm_params.keys() and seq_params
    for name, p in seq_params.items():
        assert torch.equal(p, bm_params[name]), name
    assert seq_state.keys() == bm_state.keys()
    assert any(k.endswith(".remainder") for k in seq_state)
    for key, value in seq_state.items():
        assert torch.equal(value, bm_state[key]), key
    assert hits > 0  # the block forward's expert output is reused by the remat
