"""``poros.Run`` on real batches against a reference built with no Poros forward: bit for bit.

The reference is vanilla ``Qwen2ForCausalLM`` holding the decoded weights in plain modules, a LoRA
written out here, and plain autograd, and it reads the rows themselves, not a ``Batch``. So what
every Mac arm shares has something to differ from: which rows the head reads, each row's targets
and logits, the weighted mean, and each slice's share of the step. The rows are of unequal length
with part of each scored, and they cross a slice, which is what makes each of those a real question.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("accelerate")

import torch.nn.functional as F  # noqa: E402
from torch import nn  # noqa: E402
from transformers import Qwen2ForCausalLM  # noqa: E402

from poros.core.config import PorosConfig  # noqa: E402
from poros.moe import quant  # noqa: E402
from poros.train.batch import Batch, Datum  # noqa: E402
from poros.train.run import Run  # noqa: E402
from tests.metal import nf4_tiny as tiny  # noqa: E402

SEQ = 16
PAD = 7
PER_SLICE = 2  # three rows: two through the model together, then one
ROWS = (6, 14, 10)  # tokens per row
PROMPT = 3  # tokens per row that are read and not scored
CHUNK = 16384  # the head's own rows at a time: GEMM rounding depends on the shape
LINEARS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class DenseLora(nn.Module):
    """A frozen linear, in bf16, and fp32 LoRA on top. No dropout: this test runs at 0."""

    def __init__(self, weight, bias, a, b, scaling):
        super().__init__()
        self.weight, self.bias, self.scaling = weight, bias, scaling
        self.a, self.b = nn.Parameter(a.detach().clone()), nn.Parameter(b.detach().clone())

    def forward(self, x):
        y = F.linear(x.to(torch.bfloat16).reshape(-1, x.shape[-1]), self.weight)
        y = y if self.bias is None else y + self.bias
        lora = F.linear(F.linear(x, self.a), self.b)
        return y.reshape(*x.shape[:-1], -1).to(x.dtype) + lora * self.scaling


class DenseHead(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = weight

    def forward(self, hidden):
        spans = range(0, self.weight.shape[0], CHUNK)
        return torch.cat([F.linear(hidden, self.weight[a : a + CHUNK]) for a in spans], dim=-1)


def _reference(source, trainer):
    """The model, and each of its LoRA factors beside the trainer's parameter it copies."""
    bf16 = Qwen2ForCausalLM.from_pretrained(source, dtype=torch.bfloat16).state_dict()
    model = Qwen2ForCausalLM.from_pretrained(source, dtype=torch.float32,
                                             attn_implementation="eager")
    params = dict(trainer.model.named_parameters())
    scaling = trainer.config.lora_alpha / trainer.config.lora_rank
    factors = []
    for index, layer in enumerate(model.model.layers):
        for name, module in [*layer.named_modules()]:
            if name.rpartition(".")[2] not in LINEARS:
                continue
            key = f"model.layers.{index}.{name}"
            w = bf16[f"{key}.weight"]
            weight = quant.dequantize(quant.quantize_tensor(w), w.shape, torch.bfloat16)
            a, b = (next(v for k, v in params.items() if f"{key}.lora_{f}." in k) for f in "AB")
            dense = DenseLora(weight, bf16.get(f"{key}.bias"), a, b, scaling)
            parent, _, leaf = name.rpartition(".")
            setattr(layer.get_submodule(parent), leaf, dense)
            factors += [(a, dense.a), (b, dense.b)]
    model.lm_head = DenseHead(model.lm_head.weight.detach())
    for param in model.parameters():
        param.requires_grad_(any(param is mine for _, mine in factors))
    return model, factors


def _rows(vocab):
    ids = torch.randint(0, vocab, (len(ROWS), max(ROWS)),
                        generator=torch.Generator().manual_seed(3)).tolist()
    return [row[:n] for row, n in zip(ids, ROWS)]


def _dense_step(model, rows):
    """The whole batch's mean loss, and its gradients on the reference's factors."""
    total = 0.0
    for start in range(0, len(rows), PER_SLICE):
        part = rows[start : start + PER_SLICE]
        ids = torch.full((len(part), SEQ), PAD)
        mask = torch.zeros(len(part), SEQ, dtype=torch.long)
        targets = torch.zeros(len(part), SEQ, dtype=torch.long)
        weights = torch.zeros(len(part), SEQ)
        for i, row in enumerate(part):
            ids[i, : len(row) - 1] = torch.tensor(row[:-1])
            mask[i, : len(row) - 1] = 1
            targets[i, : len(row) - 1] = torch.tensor(row[1:])
            weights[i, PROMPT - 1 : len(row) - 1] = 1.0
        logits = model(input_ids=ids, attention_mask=mask).logits
        logprobs = torch.stack([
            -F.cross_entropy(logits[i].float(), targets[i], reduction="none")
            for i in range(len(part))
        ])
        value = -(logprobs * mask.float() * weights).sum() / weights.sum()
        share = len(part) / len(rows)
        (value * share).backward()
        total += share * float(value.detach())
    return total


def _bits(t):
    return t.view(torch.int32)


@pytest.fixture
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def test_run_on_real_batches_is_bit_equal_to_a_dense_reference(tmp_path, one_thread):
    cfg = tiny.tiny_config()  # untied: its head's rows are not its embeddings'
    source = tmp_path / "source"
    tiny.write_checkpoint(source, cfg)
    config = PorosConfig(model_name_or_path=str(source), store="ram",
                         store_path=str(tmp_path / "store"), max_seq_length=SEQ,
                         per_device_batch_size=PER_SLICE, lora_dropout=0.0, warmup_steps=0,
                         save_steps=0, eval_steps=0, output_dir=str(tmp_path / "run"),
                         progress=False)
    trainer = tiny.arms.CpuDouble.from_pretrained(str(source), config=config)
    torch.set_num_threads(1)  # a trainer picks its own
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():  # lora_B starts at zero, and then lora_A's gradients are too
        for param in trainer.model.parameters():
            if param.requires_grad:
                param.copy_(0.03 * torch.randn(param.shape, generator=generator))
    model, factors = _reference(source, trainer)
    assert factors and cfg.tie_word_embeddings is False

    rows = _rows(cfg.vocab_size)
    batch = Batch(
        [Datum.from_sequence(row, [0.0] * PROMPT + [1.0] * (len(row) - PROMPT)) for row in rows],
        pad_id=PAD, pad_to=SEQ,
    )
    loss = Run.from_trainer(trainer).forward_backward(batch).loss
    want = _dense_step(model, rows)

    assert loss == want
    for mine, dense in factors:
        assert mine.grad.abs().max() > 0
        assert torch.equal(_bits(mine.grad), _bits(dense.grad))
