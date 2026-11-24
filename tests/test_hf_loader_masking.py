"""HF dataset loader label-masking tests.

The loader pads each batch to its longest row with pad_token = eos_token when
the tokenizer defines no pad token; the labels must mask padding positions with
-100 or the loss is dominated by trivially-predictable padding on short rows.

CPU-only, no downloads: load_dataset and AutoTokenizer are monkeypatched.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")


class _FakeTokenizer:
    """Whitespace tokenizer with the HF call surface the loader uses."""

    def __init__(self):
        self.pad_token = None
        self.eos_token = "<|endoftext|>"
        self.chat_template = None

    pad_token_id = 0

    def __call__(self, texts, truncation, max_length, padding=None, return_tensors=None):
        input_ids = [[(hash(w) % 250) + 2 for w in t.split()][:max_length] for t in texts]
        if padding is None:
            return {"input_ids": input_ids, "attention_mask": [[1] * len(i) for i in input_ids]}
        assert padding == "max_length" and return_tensors == "pt"
        mask = [[1] * len(i) + [0] * (max_length - len(i)) for i in input_ids]
        input_ids = [i + [0] * (max_length - len(i)) for i in input_ids]
        return {"input_ids": torch.tensor(input_ids), "attention_mask": torch.tensor(mask)}


def _build_loader(monkeypatch, rows, max_seq_length=16, batch_size=2, **how):
    import datasets

    from poros.core import dataset_spec
    from poros.train.datasets import _load_hf_dataset_as_dataloader

    fake_tok = _FakeTokenizer()
    monkeypatch.setattr("poros.core.tokenizer.load_tokenizer", lambda *a, **k: fake_tok)
    monkeypatch.setattr(
        dataset_spec,
        "load_hub_dataset",
        lambda name: datasets.Dataset.from_list(rows),
    )
    loader = _load_hf_dataset_as_dataloader(
        dataset_name="fake/dataset",
        model_name_or_path="fake/model",
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        seed=0,
        trust_remote_code=False,
        **how,
    )
    return loader, fake_tok


def test_pad_positions_masked_to_minus_100(monkeypatch):
    rows = [{"text": " ".join(["w"] * (2 + i))} for i in range(8)]
    loader, _ = _build_loader(monkeypatch, rows)

    saw_padding = False
    for batch in loader:
        pad = batch["attention_mask"] == 0
        saw_padding = saw_padding or bool(pad.any())
        assert (batch["labels"][pad] == -100).all(), (
            "padding positions must be masked out of the loss"
        )
    assert saw_padding, "test rows must actually produce padding"


def test_non_pad_labels_preserved(monkeypatch):
    rows = [{"text": f"some longer example row number {i} here"} for i in range(8)]
    loader, _ = _build_loader(monkeypatch, rows)

    for batch in loader:
        real = batch["attention_mask"] == 1
        assert (batch["labels"][real] == batch["input_ids"][real]).all(), (
            "non-padding labels must equal input_ids"
        )
        assert (batch["labels"][real] != -100).all()


def test_pad_token_falls_back_to_eos(monkeypatch):
    rows = [{"text": "a b c"} for _ in range(4)]
    _, fake_tok = _build_loader(monkeypatch, rows)
    assert fake_tok.pad_token == fake_tok.eos_token


def test_single_token_rows_dropped(monkeypatch):
    # 1-token rows lose their only label to the shift (all -100 -> NaN loss,
    # zero gradient); the loader must drop them. 2-token rows survive.
    rows = [{"text": "hi"}] * 3 + [{"text": "two tokens here"}] * 5
    loader, _ = _build_loader(monkeypatch, rows, batch_size=1)
    batches = list(loader)
    assert len(batches) == 5
    for batch in batches:
        assert int(batch["attention_mask"].sum()) >= 2


def test_batches_pad_to_longest_row_not_max_length(monkeypatch):
    rows = [{"text": " ".join(["w"] * (2 + i))} for i in range(8)]
    loader, _ = _build_loader(monkeypatch, rows, max_seq_length=64)
    for batch in loader:
        width = batch["input_ids"].shape[1]
        assert width == int(batch["attention_mask"].sum(1).max()), "width must be the longest row"
        assert width < 64


def test_the_mac_pads_every_row_to_max_length(monkeypatch):
    # pad_to_longest=False is the Mac's loader (the trainer passes it there): every row padded to
    # max_seq_length as before, the padding masked out of the loss.
    rows = [{"text": " ".join(["w"] * (2 + i))} for i in range(8)]
    loader, _ = _build_loader(monkeypatch, rows, max_seq_length=12, pad_to_longest=False)
    for batch in loader:
        assert batch["input_ids"].shape[1] == 12
        pad = batch["attention_mask"] == 0
        assert pad.any() and (batch["labels"][pad] == -100).all()
        assert (batch["labels"][~pad] == batch["input_ids"][~pad]).all()
