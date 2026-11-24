"""``poros.Run`` on the MoE path: the same verbs, the experts streamed by ``poros.moe``."""

from __future__ import annotations

import math

import pytest

from tests.conftest import as_linux
from tests.moe import require_tested_transformers

torch = pytest.importorskip("torch")
require_tested_transformers()

from poros import data as pdata  # noqa: E402
from poros.core.errors import PorosCancelledError  # noqa: E402
from poros.recipes import dpo, sft  # noqa: E402
from poros.train.run import Run  # noqa: E402
from tests.moe.test_native_entry import _model  # noqa: E402


class Tok:
    """Byte-level fake with a chat template; ids fit the tiny model's 64."""

    chat_template = "fake"
    eos_token_id, pad_token_id, bos_token_id = 1, 0, None

    def enc(self, s):
        return [8 + ord(c) % 50 for c in s]

    def __call__(self, s, add_special_tokens=True):
        return {"input_ids": self.enc(s)}

    def apply_chat_template(self, turns, tokenize=True, add_generation_prompt=False,
                            return_dict=False):
        ids = []
        for t in turns:
            ids += [2, 3 if t["role"] == "user" else 4] + self.enc(t["content"]) + [1]
        return ids + ([2, 4] if add_generation_prompt else [])


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("qwen3_moe_run")
    _model().save_pretrained(path)
    return path


@pytest.fixture(autouse=True)
def _a_cuda_hosts_planner(monkeypatch):
    """The CPU suite's trainer is the CUDA route's (tests/conftest.py): its planner reads this host
    as that route does. A Mac's one pool would plan the Mac's store, which that route has not."""
    import poros.core.card

    as_linux(monkeypatch, poros.core.card)


def _run(checkpoint, out):
    return Run.create(str(checkpoint), rank=8, lr=1e-2, output_dir=str(out), save_steps=0,
                      progress=False, warmup_steps=0, tokenizer=Tok())


def _chats():
    rows = [{"messages": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": f"a{i} a{i}"}]} for i in range(4)]
    return pdata.chats(rows, Tok(), batch_size=2)


def test_sft_on_moe_learns_and_resumes_bitwise(checkpoint, tmp_path):
    data = _chats()
    whole = _run(checkpoint, tmp_path / "a")
    assert whole._engine.block_manager is None  # the MoE path, not the dense one
    first = whole.forward(data.batch(0)).loss
    sft.main(whole, data, steps=4)
    assert whole.forward(data.batch(0)).loss < first

    part = _run(checkpoint, tmp_path / "b")
    with pytest.raises(PorosCancelledError):
        sft.main(part, data, steps=4, stop=lambda: part.step == 2)
    resumed = Run.open(tmp_path / "b", tokenizer=Tok())
    assert resumed.step == 2
    sft.main(resumed, data, steps=4)
    a = dict(whole._engine.model.named_parameters())
    for name, p in resumed._engine.model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p, a[name]), name


def test_dpo_on_moe_starts_at_log2(checkpoint, tmp_path):
    run = _run(checkpoint, tmp_path)
    rows = [{"prompt": f"p{i}", "chosen": "yes", "rejected": "no no"} for i in range(2)]
    data = pdata.pairs(rows, Tok(), batch_size=2)
    ref = run.reference(data)
    assert math.isclose(run.forward(data.batch(0), loss="dpo", ref=ref).loss, math.log(2),
                        rel_tol=1e-3)
    dpo.main(run, data, steps=2)
