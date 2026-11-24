"""One run, one set of numbers: the live card, the finished card and /runs say the same loss."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from poros.cli import sheets
from poros.cli.train_cmd import card_curve
from poros.cli.ui import components as c
from poros.cli.ui import screens as sc


def _noisy(n: int = 284) -> list[float]:
    rng = random.Random(7)
    return [2.6 * 0.995 ** i + rng.uniform(-0.6, 1.2) * (i % 9 == 0) + 0.3 for i in range(n)]


def _text(line) -> str:
    return "".join(s for s, _ in line)


@pytest.mark.parametrize("total", [284, 40, 100, 31])
def test_one_row_runs_start_where_the_finished_card_starts(total):
    """The live card's "from" never changes at the finish: from the first rows the finished card
    averages on, it is their mean; before them, the first step."""
    losses = _noisy(total)
    curve, _, _ = card_curve(1, losses, losses[-1])
    st = sc.TrainState(model="m", total=total, one_row=True)
    parts = round(total / 8)
    first_rows = total // parts if parts >= 4 else total + 1
    for n in range(1, total + 1):
        st.losses = losses[:n]
        if n >= first_rows:
            assert st.first() == pytest.approx(curve[0], abs=0), n
        else:
            assert st.first() == losses[0], n
    if not curve:  # too short for row means: the card starts on the first step, as live
        assert st.first() == losses[0]


def test_a_run_of_several_rows_a_step_starts_on_its_first_step():
    st = sc.TrainState(model="m", total=284, losses=_noisy(), one_row=False)
    assert st.first() == st.losses[0]
    assert sc.TrainState(model="m").first() is None


def test_a_finished_run_lists_the_loss_its_card_ends_on(tmp_path, monkeypatch):
    losses = _noisy()
    (tmp_path / "events.jsonl").write_text("{}\n")
    end = c.row_means(losses)[-1][1]
    asked = []

    def facts(folder, output_dir):
        asked.append(output_dir)
        return {"losses": c.row_means(losses)}  # (step, mean of its rows), as facts gives them

    from poros.cli.verbs import runs_report

    monkeypatch.setattr(runs_report, "facts", facts)
    sheets.listings._curve_end.cache_clear()
    done = SimpleNamespace(state="succeeded", loss=losses[-1], folder=str(tmp_path),
                           output_dir=str(tmp_path), step=284)
    assert sheets.listings.card_loss(done) == end != losses[-1]
    assert sheets.listings.card_loss(done) == end and len(asked) == 1  # read once: its events stand
    (tmp_path / "events.jsonl").write_text("{}\n{}\n")
    assert sheets.listings.card_loss(done) == end and len(asked) == 2  # and again when they change
    assert f"loss {end:.2f}" in sheets.export._progress_fact(done)
    going = SimpleNamespace(**{**vars(done), "state": "running"})
    assert sheets.listings.card_loss(going) == losses[-1]  # still going: its newest step's


@pytest.mark.parametrize("session", [True, False])
def test_the_session_ends_on_where_it_is_and_one_next_row(session, monkeypatch):
    monkeypatch.setattr(c, "in_session", session)
    monkeypatch.setattr(c, "next_step", "/eval x" if session else "")
    losses = _noisy(400)
    r = sc.TrainResult(steps=400, wall_s=312.0, final_loss=losses[-1], first_loss=losses[0],
                       losses=losses, peak_vram_gb=2.3, card_gb=16.0, peak_ram_gb=None,
                       ram_total_gb=None, output_dir="./runs/x", receipt="./runs/x/receipt.json",
                       tokens_per_sec=120.0, next_cmd="poros eval x", recipe="SFT",
                       save_cmd="poros recipes save last")
    rows = [_text(line) for line in sc.train_done(r, 100)]
    if session:  # two rows under the card: where it is, then every next step on one row
        nxt = next(t for t in rows if t.lstrip().startswith("Next"))
        assert nxt.index("Next") == 2 and nxt.index("/eval") < nxt.index("/recipes save last")
        assert any("saved" in t and "runs/x" in t for t in rows)
        assert not any(w in t for t in rows for w in ("speed", "receipt", "adapter", "Keep"))
    else:
        kept = next(t for t in rows if "Keep it as a recipe" in t)
        assert kept.startswith("Keep it as a recipe: ")


def test_attaching_the_adapter_prints_nothing(capsys):
    """Its line landed inside the live run card, between the header and "Loaded"."""
    pytest.importorskip("peft")
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from poros.train.model import _attach_peft_lora

    base = Qwen2ForCausalLM(Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                                        num_hidden_layers=1, num_attention_heads=2,
                                        num_key_value_heads=1, max_position_embeddings=16))
    summary, _ = _attach_peft_lora(base, lora_rank=2, lora_alpha=4, lora_dropout=0.0,
                                   lora_bias="none", target_modules=["q_proj"])
    assert capsys.readouterr().out == "" and summary["trainable_param_count"] > 0


def test_checkpoints_read_as_the_run_card_did_at_their_step(tmp_path, monkeypatch):
    """One row a step: a checkpoint's loss is the card's curve at its step, so "final" says the
    card's final loss, not its last row's."""
    import json

    from poros import runs

    monkeypatch.setenv("POROS_HOME", str(tmp_path / "poros"))
    losses = _noisy(80)
    folder = runs.create("1108-1849-b81d", {"model": "Qwen/Qwen2.5-0.5B", "steps": 80})
    runs.update(folder, state="succeeded")
    with open(folder / "events.jsonl", "w") as f:
        for step, loss in enumerate(losses):
            print(json.dumps({"event_type": "step", "step": step, "data": {"loss": loss}}), file=f)
    (folder / "receipt.json").write_text(json.dumps({"config": {"gradient_accumulation_steps": 1}}))
    for step in (40, 80):  # as a trainer names them: by the last step counted from 0
        (folder / f"checkpoint-{step - 1}").mkdir()
        (folder / f"checkpoint-{step - 1}" / ".poros_checkpoint_complete").touch()
        (folder / f"checkpoint-{step - 1}" / "adapter_model.safetensors").write_bytes(b"x")
    (folder / "adapter_model.safetensors").write_bytes(b"f")
    (folder / "adapter_config.json").write_text("{}")
    curve, final, _ = card_curve(1, losses, losses[-1])
    windows = c.row_means(losses)
    cells = {_text(i.cells[:1]).strip(): float(_text(i.cells[1:2]))
             for i in sheets.listings.checkpoints_listing("1108-1849-b81d").items}
    assert cells["final"] == pytest.approx(final, abs=5e-5) != pytest.approx(losses[-1], abs=5e-5)
    assert cells["step 80"] == cells["final"]
    assert cells["step 40"] == pytest.approx(next(v for end, v in windows if end >= 40), abs=5e-5)


def test_a_question_in_a_tab_keeps_every_option_with_it():
    """The card's floor is its command, rule and keys: a menu's last option is not its command."""
    from poros.cli.ui import tui

    asking = "Delete for good: poros://r/weights/80."
    rows = [_text(r).strip() for r in tui.card_canvas(
        lambda w: sc.choice_menu(asking, [("Delete it", "")], 0, w), 100, 40)]
    asked = next(i for i, t in enumerate(rows) if t.startswith("Delete for good"))
    assert rows[asked] == "Delete for good: poros://r/weights/80 · Choose one"
    assert rows[asked + 1].endswith("Delete it") and rows[asked + 2].endswith("Cancel")
    assert "choose" in rows[-1] and not rows[-3]  # the floor: no command, the rule, the keys


class _QwenLike:
    """Qwen2.5's chat template, by hand: <|im_start|>=1, <|im_end|>=2, <|endoftext|>=3, "\\n"=4."""
    chat_template = "qwen"
    eos_token_id = 3
    all_special_ids = [1, 2, 3]

    def encode(self, text, add_special_tokens=False):
        return [100 + ord(ch) for ch in text]

    def apply_chat_template(self, turns, tokenize=True, add_generation_prompt=False,
                            return_dict=False):
        out = []
        for t in turns:
            out += [1, *self.encode(t["role"]), 4, *self.encode(t["content"]), 2, 4]
        return out + ([1, *self.encode("assistant"), 4] if add_generation_prompt else [])


def test_eval_reads_the_turn_end_its_template_puts_after_an_answer():
    """A base model's generation config ends on <|endoftext|> alone; a reply stops on the
    <|im_end|> the template puts after an answer, and that is the token eval looks for."""
    from poros import ends

    assert ends.turn_end(_QwenLike()) == 2
    plain = _QwenLike()
    plain.chat_template = None
    assert ends.turn_end(plain) is None
    texty = _QwenLike()
    texty.all_special_ids = [1, 3]  # the turn ends on text the tokenizer does not hold special
    assert ends.turn_end(texty) is None


def test_an_agent_is_told_when_chats_train_a_base_model():
    from poros.cli.verbs.train_run import plan_said

    def action(model, shape):
        return SimpleNamespace(model=model, shape=shape, config={}, preset={}, rows=None, steps=80,
                               seq_len=256, lr=2e-5, recipe="SFT", plan=None)

    said = plan_said(action("Qwen/Qwen2.5-0.5B", "chats"))
    assert "Qwen2.5-0.5B-Instruct ends them" in said["warning"]
    assert "warning" not in plan_said(action("Qwen/Qwen2.5-0.5B-Instruct", "chats"))
    assert "warning" not in plan_said(action("Qwen/Qwen2.5-0.5B", "text"))
