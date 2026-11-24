"""A resume refuses another row list, the offloaded head is copied once for evaluation, a frozen
head that cannot reach the chat's end token is said before training, and a saved checkpoint's
line names its folder."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from poros.preflight import command
from poros.preflight.resume import DATA_RECORD, data_clash, data_record


def _data(fp, rows, kind="chats:answers:4096"):
    return SimpleNamespace(fingerprint=fp, rows=rows, kind=kind)


def test_a_resume_refuses_another_row_list_naming_both(tmp_path):
    (tmp_path / DATA_RECORD).write_text(json.dumps(data_record(_data("aaaa", 50478))))
    assert data_clash(str(tmp_path), _data("aaaa", 50478)) is None
    why = data_clash(str(tmp_path), _data("bbbb", 50374))
    assert "50478" in why and "50374" in why and "aaaa" in why and "bbbb" in why
    r = command.data_resume_refusal(str(tmp_path), _data("bbbb", 50374))
    assert r.code == "E_CONFIG" and r.title == "Not the same data"
    assert command.BEFORE_THE_LOAD in r.context


def test_a_checkpoint_with_no_record_resumes_as_before(tmp_path):
    assert data_clash(str(tmp_path), _data("bbbb", 1)) is None
    assert command.data_resume_refusal(None, _data("bbbb", 1)) is None


def test_the_offloaded_heads_eval_copy_is_made_once_and_dropped_for_training():
    torch = pytest.importorskip("torch")
    from poros.train.head import DeviceHead

    head = DeviceHead(SimpleNamespace())
    w = torch.randn(8, 4)
    a = head.eval_weight(w, torch.device("cpu"), torch.float32)
    assert head.eval_weight(w, torch.device("cpu"), torch.float32) is a
    w.add_(1.0)  # the weight changed: a fresh copy
    b = head.eval_weight(w, torch.device("cpu"), torch.float32)
    assert torch.equal(b, w)
    head.drop_eval_weight()
    assert head._eval_weight is None


def test_a_frozen_head_whose_end_row_is_another_tokens_is_said():
    torch = pytest.importorskip("torch")
    from poros.preflight.data import degenerate_end_row

    w = torch.randn(10, 16)
    w[7] = w[5] * 1.0000001  # the end token (7) has token 5's row
    said = degenerate_end_row(w, 7, [5, 6, 7], frozen=True, train_on="answers")
    assert said and "token 5" in said and "lm_head in lora_target_modules" in said
    assert degenerate_end_row(w, 7, [5, 6, 7], frozen=False, train_on="answers") is None
    assert degenerate_end_row(w, 7, [5, 6, 7], frozen=True, train_on="all") is None
    assert degenerate_end_row(torch.randn(10, 16), 7, [5, 6, 7], frozen=True,
                              train_on="answers") is None


def test_a_saved_checkpoints_line_names_its_folder():
    from poros.cli.ui.train_view import checkpoint_said

    assert checkpoint_said("/r/checkpoint-25", 25) == "checkpoint-25"
    assert checkpoint_said("/r/checkpoint-24", 25) == "checkpoint-24 (25 steps)"


def test_a_checkpoint_keeps_the_record_of_the_rows_a_recipe_trained_on(tmp_path, cpu_only):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from tests.test_train_loop_cpu import _make_trainer, _tiny_peft_qwen2

    model, _ = _tiny_peft_qwen2(torch)
    trainer = _make_trainer(torch, model, str(tmp_path))
    trainer._data_record = data_record(_data("cafe", 12))
    trainer.save(str(tmp_path / "ckpt"))
    assert json.loads((tmp_path / "ckpt" / DATA_RECORD).read_text())["fingerprint"] == "cafe"
    assert "12 rows" in data_clash(str(tmp_path / "ckpt"), _data("beef", 11))
