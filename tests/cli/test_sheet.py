"""The settings sheet: layout at every width, the keys, and what it hands back."""

from __future__ import annotations

import warnings

import pytest

from poros.cli import sheets
from poros.cli.ui.sheet import Field, Form, Option, Sheet, View
from poros.cli.ui.text import width
from poros.cli.words.grammar import parse
from poros.core.refusal import Refusal


def _form(values=None) -> Form:
    def build(v):
        fields = [
            Field("subject", "model", value=v.get("subject"), required=True, free=True,
                  shown=str(v.get("subject") or ""), source="given" if v.get("subject") else "",
                  options=lambda: [Option("Qwen2.5-7B", "Qwen2.5-7B", "7B · fits"),
                                   Option("Qwen2.5-72B", "Qwen2.5-72B", "72B · won't fit")]),
            Field("--steps", "steps", kind="text", value=v.get("--steps"),
                  check=lambda t: None if t.isdigit() else "steps is a whole number"),
            Field("--detach", "background", kind="toggle", value=bool(v.get("--detach"))),
        ]
        if v.get("subject") == "Qwen2.5-72B":
            fields.append(Field("--force", "won't fit", kind="toggle", warn=True,
                                value=bool(v.get("--force")), source="predicted to OOM"))
        view = View("Train  test", fields, result=("fit", "fits", "6.2 of 24.0 GiB"),
                    go="Start training")
        view.ready = sheets.common._ready(view)
        return view

    return Form("train", build, dict(values or {}))


def _text(lines) -> str:
    return "\n".join("".join(t for t, _ in line) for line in lines)


def keys(sheet: Sheet, *seq):
    out = None
    for k in seq:
        kind = k if k in ("enter", "esc", "up", "down", "tab", "backspace", "right", "left",
                         "ctrl-s") else "char"
        if kind == "char":
            for ch in k:
                out = sheet.key("char", ch)
        else:
            out = sheet.key(kind, "")
    return out


@pytest.mark.parametrize("w", [60, 80, 100])
def test_every_line_fits_the_width(w):
    for values in ({}, {"subject": "Qwen2.5-72B"}, {"subject": "Qwen2.5-7B", "--steps": "300"}):
        sheet = Sheet(_form(values))
        assert all(width(line) <= w for line in sheet.render(w)), _text(sheet.render(w))


def test_it_opens_on_the_first_thing_to_choose_and_start_waits():
    sheet = Sheet(_form())
    assert sheet.mode == "pick" and sheet.field.key == "subject"
    assert "choose model first" in _text(sheet.render(80))
    keys(sheet, "esc")  # back to the rows; the action row still refuses
    sheet.sel = len(sheet.view.fields)
    assert keys(sheet, "enter") is None and sheet.field.key == "subject"


def test_typing_filters_and_enter_picks_then_start_runs():
    sheet = Sheet(_form())
    keys(sheet, "7b", "enter")
    assert sheet.form.values["subject"] == "Qwen2.5-7B"
    sheet.sel = len(sheet.view.fields)
    assert keys(sheet, "enter") == "go"
    assert sheets.common.form_argv("train", sheet.form.values) == ["train", "Qwen2.5-7B"]


def test_a_free_pick_takes_what_was_typed():
    sheet = Sheet(_form())
    keys(sheet, "./my-model")
    options = [o.label for o in sheet.options()]
    assert any("./my-model" in o for o in options)
    keys(sheet, "enter")
    assert sheet.form.values["subject"] == "./my-model"


def test_a_typed_value_is_checked_on_its_row():
    sheet = Sheet(_form({"subject": "Qwen2.5-7B"}))
    sheet.sel = 1
    keys(sheet, "enter", "abc", "enter")
    assert sheet.mode == "text" and "whole number" in _text(sheet.render(80))
    keys(sheet, "backspace", "backspace", "backspace", "300", "enter")
    assert sheet.form.values["--steps"] == "300"


def test_a_needed_yes_is_its_own_row_and_never_preset():
    sheet = Sheet(_form({"subject": "Qwen2.5-72B"}))
    warn = [f for f in sheet.view.fields if f.warn]
    assert [f.key for f in warn] == ["--force"] and not warn[0].value
    assert not sheet.view.ready and "not yet: ⏎ for yes" in _text(sheet.render(80))
    sheet.sel = [f.key for f in sheet.view.fields].index("--force")
    keys(sheet, "enter")
    assert sheet.form.values["--force"] is True and sheet.view.ready


def test_esc_cancels():
    sheet = Sheet(_form({"subject": "Qwen2.5-7B"}))
    assert keys(sheet, "esc") == "cancel"


def test_which_commands_open_a_sheet():
    assert sheets.wanted("train", parse("train Qwen2.5-7B --data x.jsonl"))
    assert not sheets.wanted("train", parse("train Qwen2.5-7B --data x.jsonl --yes"))
    assert sheets.wanted("export", parse("export last"))
    assert not sheets.wanted("export", parse("export last --format adapter"))
    assert sheets.wanted("runs", parse("runs")) and not sheets.wanted("runs", parse("runs last"))
    assert not sheets.wanted("machine", parse("machine"))


def test_the_train_sheet_reads_the_real_resolve(tmp_path, monkeypatch):
    import poros.plan.configs
    from poros.cli.context import Context
    from poros.cli.schemas import Figure, PlanGpu

    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        poros.plan.configs, "download_size",
        lambda cfg: Figure(value=None, unit="GiB", source="unknown", why="offline in tests"),
    )
    (tmp_path / "notes.jsonl").write_text('{"text": "hello"}\n' * 40)
    gpu = PlanGpu(name="RTX 4090", total_gib=24.0, free_gib=None, source="spec")
    ctx = Context(cwd=tmp_path, fixed_gpu=gpu, fixed_cached=frozenset({"Qwen/Qwen2.5-7B"}),
                  fixed_interactive=True)
    s = parse("train Qwen2.5-7B --data notes.jsonl")
    sheet = Sheet(sheets.train_form(s, ctx))
    text = _text(sheet.render(80))
    assert "40 rows" in text and "1 epoch" in text and "Start training" in text
    assert "inferred" not in text and "given" not in text  # the reason, not the bookkeeping
    assert sheet.view.result is not None and sheet.view.result[0] in ("fit", "fits")
    assert sheet.view.ready and sheet.field is None  # filled in: it opens on its action


def test_a_filled_card_opens_on_its_action_and_esc_goes_back_to_it():
    sheet = Sheet(_form({"subject": "Qwen2.5-7B"}))
    assert sheet.field is None  # nothing to choose: enter runs it
    keys(sheet, "up", "up")
    assert sheet.field is not None and sheet.key("esc") is None and sheet.field is None
    assert sheet.key("esc") == "cancel"  # on the action, esc leaves


def test_a_question_waits_for_the_action_and_its_yes_returns_there():
    sheet = Sheet(_form({"subject": "Qwen2.5-72B"}))
    assert sheet.field is None and not sheet.view.ready  # the warning never steals focus
    assert sheet.key("enter") is None and sheet.field.warn  # enter: to the question
    sheet.key("enter")  # yes
    assert sheet.field is None and sheet.view.ready


def test_a_value_somebody_set_is_bold_and_one_worked_out_is_not():
    sheet = Sheet(_form({"subject": "Qwen2.5-7B"}))
    rows = {line[1][0].strip(): line for line in sheet.render(80) if len(line) > 2}
    assert rows["model"][2][1] == "hi"
    assert "given" not in _text(sheet.render(80))  # bookkeeping goes unsaid


def test_the_keys_line_puts_enter_in_the_accent():
    sheet = Sheet(_form({"subject": "Qwen2.5-7B"}))
    last = sheet.render(80)[-1]
    assert ("⏎", "a") in last and ("start training" in "".join(t for t, _ in last))


def test_typing_on_the_train_card_searches_every_setting(tmp_path, monkeypatch, path_receipts):
    import poros.plan.configs
    from poros.cli.context import Context
    from poros.cli.schemas import Figure, PlanGpu

    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        poros.plan.configs, "download_size",
        lambda cfg: Figure(value=None, unit="GiB", source="unknown", why="offline in tests"),
    )
    (tmp_path / "notes.jsonl").write_text('{"text": "hello"}\n' * 40)
    gpu = PlanGpu(name="RTX 5090", total_gib=31.8, free_gib=None, source="spec")
    ctx = Context(cwd=tmp_path, fixed_gpu=gpu,
                  fixed_cached=frozenset({"Qwen/Qwen2.5-7B-Instruct"}), fixed_interactive=True)
    sheet = Sheet(sheets.train_form(parse(["train"]), ctx))
    text = _text(sheet.render(90))
    # opened filled in: the starter model, the newest data file here
    assert "Train Qwen2.5-0.5B-Instruct on notes.jsonl" in text
    assert "recommended here" in text and "newest file here" in text
    assert "measured" not in text and "estimated" not in text
    assert sheet.field is None
    for ch in "warm":
        sheet.key("char", ch)
    assert sheet.mode == "pick" and sheet.field.adds
    assert any("warmup" in o.label for o in sheet.options())
    sheet.key("esc")
    # changing a row that already had a value keeps the focus on that row
    for key, keys in (("--seq-len", ["enter", "down", "enter"]),
                      ("length", ["enter", "ctrl-u", "3", "0", "enter"])):
        sheet.sel = [f.key for f in sheet.view.fields].index(key)
        for k in keys:
            sheet.key("char", k) if k.isdigit() else sheet.key(k)
        assert sheet.mode == "nav" and sheet.field.key == key


def test_the_fit_row_says_the_decision_and_a_figure_only_from_a_certified_run():
    """No planner figure: the decision in words; a certified run here gives its own figure,
    rounded up to 0.1 (8.539 as 8.6, never 8.5)."""
    from types import SimpleNamespace as NS

    from poros.cli.ui.components import ceil1, floor1

    assert (ceil1(8.539), ceil1(8.5), ceil1(8.501), ceil1(0.0)) == (8.6, 8.5, 8.6, 0.0)
    assert (floor1(9.19), floor1(9.1)) == (9.1, 9.1)
    measured = NS(value=8.539, source="measured")
    p = NS(fits=True, host_fits=None, peak_vram_reserved=measured, peak_vram_allocated=measured,
           gpu=NS(total_gib=24.0, unified=False), bound=None)
    assert sheets.fit._fit_view(p)["result"][:2] == ("fit", "checked at the start")
    cert = NS(fit_gib=lambda: 8.539, memory_peak_gib=None)
    assert sheets.fit._fit_view(p, cert)["result"] == ("fits", "8.6 of 24.0 GiB", "certified")
    over = NS(fits=False, host_fits=None, gpu=NS(total_gib=8.0, unified=False))
    assert sheets.fit._fit_view(over)["result"][1] == "won't fit"
    assert sheets.fit._fit_words(over) == "won't fit"


def test_data_lists_the_files_here_with_what_each_trains(tmp_path):
    from poros.cli.context import Context

    (tmp_path / "t.csv").write_text("question,answer\nq,a\nr,b\n")
    (tmp_path / "p.jsonl").write_text('{"prompt": "q", "chosen": "a", "rejected": "b"}\n')
    L = sheets.listings.data_listing(Context(cwd=tmp_path, fixed_interactive=True))
    rows = {i.text: "".join(t for t, _ in i.cells) for i in L.items}
    assert "2 question/answer rows" in rows["t.csv"] and "SFT" in rows["t.csv"]
    assert "1 preference pair " in rows["p.jsonl"] and "DPO" in rows["p.jsonl"]
    assert L.items[0].actions[1][2][:2] == ["train", "--data"]
    assert sheets.wanted("data", parse("data")) and not sheets.wanted("data", parse("data t.csv"))


def test_a_live_form_keeps_each_change_and_esc_closes_it():
    kept = []

    def build(v):
        return View("Settings", [Field("theme", "theme", value=v.get("theme"), quiet=True,
                                       shown=str(v.get("theme") or ""),
                                       options=lambda: [Option("dark", "dark"),
                                                        Option("light", "light")])],
                    go="Save settings")

    s = Sheet(Form("settings", build, {"theme": "dark"}, browse=True, live=kept.append))
    assert keys(s, "right") is None and kept[-1] == {"theme": "light"}
    text = _text(s.render(80))
    assert "Save settings" not in text and "esc close" in text
    assert keys(s, "esc") == "go"  # one esc: closed, nothing left to save


def test_two_escs_leave_a_card_that_opened_on_a_list():
    sheet = Sheet(_form())
    assert sheet.mode == "pick"  # opened on the model to choose
    assert keys(sheet, "esc") is None and sheet.field is None  # out to the action
    assert keys(sheet, "esc") == "cancel"
    sheet = Sheet(_form({"subject": "Qwen2.5-7B"}))
    keys(sheet, "up", "up", "up", "enter")  # a list you opened yourself
    keys(sheet, "esc")
    assert sheet.field is not None and sheet.field.key == "subject"  # back on its row


def test_a_slash_on_the_rows_is_a_command_but_a_path_being_typed_keeps_it():
    sheet = Sheet(_form({"subject": "Qwen2.5-7B"}))
    assert sheet.mode == "nav" and sheet.takes_slash()
    keys(sheet, "up", "up", "up", "enter")  # the model list takes a typed path
    assert sheet.mode == "pick" and not sheet.takes_slash()
    keys(sheet, "esc", "down", "enter")  # steps: typing a value
    assert sheet.mode == "text" and not sheet.takes_slash()


def test_the_esc_hint_says_what_esc_does():
    sheet = Sheet(_form({"subject": "Qwen2.5-7B"}))
    keys(sheet, "up", "up")
    line = _text(sheet.render(100)).splitlines()[-1]
    assert "esc done" in line and "to start" not in line


def test_an_empty_list_says_why_and_offers_only_the_way_out():
    from poros.cli.ui.sheet import Lister, Listing

    head = [("file".ljust(30), "rl"), ("holds", "rl")]
    text = _text(Lister(Listing("data", "Data", head, [], empty="no data here",
                                aside="newest first")).render(70))
    assert "no data here" in text and "file" not in text and "newest first" not in text
    assert text.splitlines()[-1].strip() == "esc close"


def test_a_narrow_key_line_keeps_the_way_out():
    from poros.cli.ui import components as c

    keys = [("⏎", "start training"), ("↑", "change"), ("ctrl+s", "save as defaults"),
            ("ctrl+r", "save as recipe"), ("esc", "cancel")]
    line = _text(c.keyline(keys, 60)[1:])
    assert line.strip().startswith("⏎ start training") and line.rstrip().endswith("esc cancel")


QWEN3_30B = {"model_type": "qwen3_moe", "hidden_size": 2048, "num_hidden_layers": 48,
             "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
             "vocab_size": 151936, "tie_word_embeddings": False, "intermediate_size": 6144,
             "moe_intermediate_size": 768, "num_experts": 128, "num_experts_per_tok": 8,
             "decoder_sparse_step": 1, "mlp_only_layers": []}


def _store_ctx(tmp_path, monkeypatch):
    from huggingface_hub import constants

    import poros.plan.configs
    from poros.cli.context import Context
    from poros.cli.schemas import Figure, PlanGpu

    # the card refuses a model it cannot fetch when the Hub is off; these tests fetch nothing. The
    # Hub reads the variable once, at its import, which an earlier file's collection may have done
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(constants, "HF_HUB_OFFLINE", False)

    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(
        poros.plan.configs, "download_size",
        lambda cfg: Figure(value=None, unit="GiB", source="unknown", why="offline in tests"),
    )
    monkeypatch.setattr(poros.plan.configs, "read_model_config",
                        lambda model, revision=None: QWEN3_30B if "A3B" in model else None)
    (tmp_path / "notes.jsonl").write_text('{"text": "hello"}\n' * 40)
    gpu = PlanGpu(name="RTX 4090", total_gib=24.0, free_gib=None, source="spec")
    return Context(cwd=tmp_path, fixed_gpu=gpu, fixed_interactive=True)


def _row(sheet, key):
    return next(f for f in sheet.view.fields if f.key == key)


def test_a_moe_card_picks_ram_or_ssd_streaming_in_one_row(tmp_path, monkeypatch):
    ctx = _store_ctx(tmp_path, monkeypatch)
    s = parse("train Qwen3-30B-A3B-Instruct-2507 --data notes.jsonl")
    sheet = Sheet(sheets.train_form(s, ctx))
    row = _row(sheet, "--store")
    assert row.label == "experts" and row.shown == "in RAM"
    ram, ssd = row.options()
    assert ram.label == "in RAM" and ssd.label == "from SSD"
    assert "host RAM" in ram.note and ".sss" in ssd.note and "GiB" not in ram.note + ssd.note
    sheet._set("--store", "ssd")
    row = _row(sheet, "--store")
    # the store under POROS_HOME, as the sheet writes a path (~ for the home folder)
    assert row.shown == "from SSD"
    assert row.source.endswith(" · " + sheets.export._home(str(tmp_path / "home" / "sss")))
    assert "--store ssd" in sheet.view.command
    a, _, _ = sheets.common._resolved("train", sheet.form.values, ctx)
    assert a.config["store"] == "ssd"
    assert a.config["store_path"] == str(tmp_path / "home" / "sss")
    # a typed folder is ssd there; the raw byte settings are not offered under "more"
    sheet._set("--store", str(tmp_path / "nvme"))
    a, _, _ = sheets.common._resolved("train", sheet.form.values, ctx)
    assert (a.config["store"], a.config["store_path"]) == ("ssd", str(tmp_path / "nvme"))
    more = _row(sheet, "more").options()
    assert not any(o.label in ("store", "store_path", "sss_ram") for o in more)


def test_a_dense_card_says_it_trains_from_ram_and_refuses_ssd(tmp_path, monkeypatch):

    ctx = _store_ctx(tmp_path, monkeypatch)
    sheet = Sheet(sheets.train_form(parse("train Qwen2.5-7B --data notes.jsonl"), ctx))
    row = _row(sheet, "--store")
    assert row.shown == "in RAM" and "MoE" in row.source
    assert [o.label for o in row.options()] == ["in RAM"]
    import poros.cli.verbs.train as train

    s = parse("train Qwen2.5-7B --data notes.jsonl --store ssd")
    with pytest.raises(Refusal, match="dense"):
        train.inspect(train.resolve(s, ctx).action, ctx)


@pytest.mark.cuda_host(reason="the CUDA presets' SFT rate (2e-4) and the rates offered around it: "
                               "a Mac's recipes train SFT at 2e-5")
def test_more_lists_settings_in_words_under_groups_at_what_the_run_uses(tmp_path, monkeypatch):
    ctx = _store_ctx(tmp_path, monkeypatch)
    (tmp_path / "chats.jsonl").write_text(
        '{"messages": [{"role": "user", "content": "hi"}, '
        '{"role": "assistant", "content": "yo"}]}\n' * 40)
    sheet = Sheet(sheets.train_form(parse("train Qwen2.5-7B-Instruct --data chats.jsonl"), ctx))
    sheet.sel = [f.key for f in sheet.view.fields].index("more")
    sheet.key("enter", "")
    text = _text(sheet.render(90))
    assert "learning" in text and "learning rate" in text and "how big each update is" in text
    lr = next(o for o in sheet.options() if o.value == "learning_rate=")
    assert lr.now == "2e-4"  # SFT's rate, not the config's 2e-05
    assert "learning_rate" not in text and "sss_ram" not in text  # words, not field names
    assert next(o for o in sheet.options() if o.value == "dpo_beta=").group == "advanced"
    for ch in "lr":  # what people call it finds it
        sheet.key("char", ch)
    assert sheet.options()[0].value == "learning_rate="
    sheet.key("enter", "")
    row = _row(sheet, "learning_rate=")
    assert row.label == "learning rate" and [o.label for o in row.options()][:2] == [
        "5e-5", "1e-4"]


def _fits_only_at_the_receipt(monkeypatch, tmp_path):
    """Short rows on a measured card: the data asks 256, the receipts ran 512."""
    from poros.cli.context import Context
    from poros.cli.schemas import PlanGpu
    from poros.cli.verbs import train

    (tmp_path / "notes.jsonl").write_text('{"text": "hello"}\n' * 40)
    gpu = PlanGpu(name="RTX 5090", total_gib=31.8, free_gib=None, source="spec")
    ctx = Context(cwd=tmp_path, fixed_gpu=gpu,
                  fixed_cached=frozenset({"Qwen/Qwen2.5-7B-Instruct"}), fixed_interactive=False)
    return train, ctx


def test_a_run_with_no_card_opens_where_the_card_does(monkeypatch, tmp_path, path_receipts):
    train, ctx = _fits_only_at_the_receipt(monkeypatch, tmp_path)
    argv = ["train", "qwen2.5-7b-instruct", "--data", "notes.jsonl", "--steps", "10"]
    a = train.inspect(train.resolve(parse(argv), ctx).action, ctx)
    card = sheets.fit._matched({"subject": "qwen2.5-7b-instruct", "--data": "notes.jsonl",
                            "--steps": "10"}, ctx)
    assert card, "this fixture must need matching"
    a = train.settle(a, ctx)
    assert str(a.seq_len) == card["--seq-len"][0]
    assert a.config["lora_dropout"] == card["lora_dropout="][0]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a warning here would print on the run's screen
        a.model_dump(mode="json")
    assert a.provenance["seq_len"].detail == card["--seq-len"][1]
    assert sheets.fit._matched({"subject": "qwen2.5-7b-instruct", "--data": "notes.jsonl",
                            "--steps": "10", "--seq-len": "256"}, ctx) == {}


def test_matching_a_receipt_never_cuts_the_rows(monkeypatch, tmp_path):
    train, ctx = _fits_only_at_the_receipt(monkeypatch, tmp_path)
    argv = ["train", "qwen2.5-7b-instruct", "--data", "notes.jsonl", "--steps", "10"]
    a = train.inspect(train.resolve(parse(argv), ctx).action, ctx)
    a.seq_len = 1024
    a.plan = train.shared.plan_for(a.model, ctx, seq_len=1024, rank=a.rank, download=True)
    assert train.matched(a, ctx) == {}
