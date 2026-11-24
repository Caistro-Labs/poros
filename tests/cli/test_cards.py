"""The cards and the flow between them: the train card's extra rows, the list
cards, /settings, the "Continue with" chain, and the /machine card."""

from __future__ import annotations

import pytest

from poros.cli import questions, sheets
from poros.cli.ui import components as c
from poros.cli.ui import sheet as sh
from poros.cli.ui.sheet import Field, Form, Option, Sheet, View
from poros.cli.ui.text import width
from poros.cli.words.grammar import parse


def _text(lines) -> str:
    return "\n".join("".join(t for t, _ in line) for line in lines)


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """A preferences file of our own, read as if from a terminal."""
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.setattr(questions, "interactive", lambda: True)
    return tmp_path


def _pick_form() -> Form:
    def build(v):
        fields = [
            Field("--seq-len", "length", value=v.get("--seq-len"), shown=str(v.get("--seq-len")),
                  options=lambda: [Option(x, x) for x in ("512", "1024", "2048")],
                  hint="tokens per example"),
            Field("more", "more", adds=True, shown="all settings",
                  options=lambda: [Option("lr_scheduler=", "lr_scheduler", sets="cosine"),
                                   Option("hub_model_id=", "hub_model_id")]),
        ]
        for key in v:
            if key.endswith("="):
                fields.insert(1, Field(key, key[:-1], kind="text", value=v[key],
                                       shown=str(v[key])))
        return View("Train  test", fields, go="Start", command="/train x --seq-len 1024",
                    meter=(0.4, "a"), result=("fit", "fits", "9.6 of 24.0 GiB"))

    return Form("train", build, {"--seq-len": "1024"})


# -- the train card ------------------------------------------------------------


def test_left_and_right_cycle_a_pick_row_without_opening_it():
    sheet = Sheet(_pick_form())
    assert sheet.mode == "nav" and sheet.field is None  # every row set: on the action
    sheet.sel = 0
    sheet.key("right")
    assert sheet.form.values["--seq-len"] == "2048" and sheet.mode == "nav"
    sheet.key("right")
    assert sheet.form.values["--seq-len"] == "512"
    sheet.key("left")
    assert sheet.form.values["--seq-len"] == "2048"


def test_the_more_row_adds_a_setting_and_opens_it():
    sheet = Sheet(_pick_form())
    sheet.sel = 1
    sheet.key("enter")
    assert sheet.mode == "pick"
    for ch in "sched":
        sheet.key("char", ch)
    sheet.key("enter")
    assert sheet.field.key == "lr_scheduler=" and sheet.mode == "text"
    assert sheet.buf == "cosine"
    # a setting with no default still gets its row, empty until typed
    sheet.key("esc")
    sheet.sel = [f.key for f in sheet.view.fields].index("more")
    sheet.key("enter")
    for ch in "hub":
        sheet.key("char", ch)
    sheet.key("enter")
    assert sheet.field.key == "hub_model_id=" and sheet.mode == "text" and sheet.buf == ""


@pytest.mark.parametrize("w", [60, 80, 100])
def test_the_card_shows_the_meter_and_the_command_and_fits(w):
    sheet = Sheet(_pick_form())
    sheet.sel = 0
    lines = sheet.render(w)
    assert all(width(line) <= w for line in lines), _text(lines)
    text = _text(lines)
    assert "fits" in text and "←→ cycle" in text
    assert "tokens per example" in text  # the focused row says what it is for
    sheet.sel = sheet.rows - 1  # on the action row: the command it runs
    assert "/train x --seq-len 1024" in _text(sheet.render(w))
    if w >= 80:
        assert "━" in text  # the fit as a share of the card


def test_the_config_rows_never_offer_repo_code():
    fields = sheets.config_fields()
    assert "trust_remote_code" not in fields and "lr_scheduler" in fields
    assert "max_seq_length" not in fields  # it has its own row
    assert sheets._choices(fields["lr_scheduler"]) == ["constant", "cosine", "linear"]
    assert sheets._choices(fields["enable_prefetch"]) == ["true", "false"]
    assert sheets._typed("learning_rate", fields["learning_rate"])("fast") is not None
    assert sheets._typed("learning_rate", fields["learning_rate"])("1e-4") is None


def test_a_setting_row_becomes_a_clause_on_the_command_line():
    argv = sheets.common.form_argv("train", {"subject": "Qwen2.5-7B", "--data": "x.jsonl",
                                      "lr_scheduler=": "linear"})
    assert argv == ["train", "Qwen2.5-7B", "--data", "x.jsonl", "lr_scheduler=linear"]
    assert parse(argv).clauses["lr_scheduler"].value == "linear"


def test_ctrl_s_saves_train_defaults_and_the_next_card_starts_from_them(prefs):
    values = {"subject": "Qwen2.5-7B", "--data": "x.jsonl", "--rank": "32",
              "lr_scheduler=": "linear", "--force": True}
    note = sheets.save_defaults(values)
    assert note == "saved 2 settings as your /train defaults"
    assert sheets.saved_defaults() == {"--rank": "32", "lr_scheduler=": "linear"}
    form = Form("train", lambda v: View("t", [Field("--rank", "rank", value=v.get("--rank"))]),
                {"--rank": "16"}, keys={"ctrl-s": ("save defaults", sheets.save_defaults)})
    sheet = Sheet(form)
    sheet.key("ctrl-s")
    assert "saved 1 setting" in _text(sheet.render(80))
    assert "ctrl+s save defaults" in _text(sheet.render(80))



def test_ctrl_s_asks_defaults_or_recipe_on_the_line_under_the_action():
    done = []
    form = Form("train", lambda v: View("t", [Field("--rank", "rank", value=v.get("--rank"))]),
                {"--rank": "16"}, keys={"ctrl-s": ("save", [
                    ("defaults", lambda v: done.append("d") or "saved as defaults"),
                    ("recipe", lambda v: done.append("r") or "saved as a recipe")])})
    sheet = Sheet(form)
    assert "ctrl+s save" in _text(sheet.render(80))
    sheet.key("ctrl-s")
    assert "save as" in _text(sheet.render(80)) and "❯ defaults" in _text(sheet.render(80))
    sheet.key("right")
    sheet.key("enter")
    assert done == ["r"] and "saved as a recipe" in _text(sheet.render(80))
    sheet.key("ctrl-s")
    sheet.key("char", "d")  # the first letter picks it
    assert done == ["r", "d"]
    sheet.key("ctrl-s")
    sheet.key("esc")  # asks nothing, saves nothing
    assert done == ["r", "d"] and "save as" not in _text(sheet.render(80))

# -- the list cards --------------------------------------------------------------


def _listing(n=3) -> sh.Listing:
    items = [
        sh.Item(cells=[(f"model-{i}".ljust(12), "hi"), ("fits", "a")], text=f"model-{i}",
                actions=[("enter", "train", ["train", f"model-{i}"]),
                         ("ctrl-f", "fit", ["fit", f"model-{i}"])])
        for i in range(n)
    ]
    return sh.Listing("models", "Models  test", [("model", "rl")], items, window=4)


def test_a_list_card_runs_the_key_picked_for_the_focused_row():
    lister = sh.Lister(_listing())
    lister.key("down")
    assert lister.key("enter") == "go" and lister.argv == ["train", "model-1"]
    lister = sh.Lister(_listing())
    assert lister.key("ctrl-f") == "go" and lister.argv == ["fit", "model-0"]
    assert sh.Lister(_listing()).key("ctrl-d") is None  # not a key this row has


def test_typing_narrows_the_list_and_the_card_keeps_its_height():
    lister = sh.Lister(_listing(6))
    full = len(lister.render(80))
    for ch in "l-5":
        lister.key("char", ch)
    lines = lister.render(80)
    assert len(lines) == full and "1 of 6" in _text(lines)
    assert lister.key("enter") == "go" and lister.argv == ["train", "model-5"]


def test_esc_clears_the_filter_first_then_leaves():
    lister = sh.Lister(_listing())
    lister.key("char", "x")
    assert "nothing matches" in _text(lister.render(80))
    assert lister.key("esc") is None and lister.buf == ""
    assert lister.key("esc") == "cancel"


@pytest.mark.parametrize("w", [60, 80])
def test_a_list_card_fits_the_width(w):
    lister = sh.Lister(_listing(8))
    assert all(width(line) <= w for line in lister.render(w))
    assert "1–4 of 8" in _text(lister.render(w))


def test_a_run_offers_the_next_thing_for_its_state():
    from types import SimpleNamespace as NS

    def acts(state):
        return [(k, argv) for k, _, argv in sheets.listings._run_actions(NS(id="7f3a",
                                                                            state=state))]

    assert acts("running")[0] == ("enter", ["runs", "7f3a"])
    # enter opens the run's page whatever its state; the keys do the rest, its checkpoints last
    saved = ("ctrl-p", ["runs", "checkpoints", "7f3a"])
    assert acts("succeeded") == [("enter", ["runs", "7f3a"]),
                                 ("ctrl-e", ["export", "7f3a"]),
                                 ("ctrl-r", ["runs", "continue", "7f3a", "--more"]),
                                 ("ctrl-t", ["train"]), saved]
    assert acts("failed") == [("enter", ["runs", "7f3a"]),
                              ("ctrl-r", ["runs", "continue", "7f3a"]), ("ctrl-t", ["train"]),
                              saved]


def test_report_without_a_run_opens_its_card():
    assert sheets.wanted("runs", parse("runs report"))
    assert not sheets.wanted("runs", parse("runs report 7f3a"))
    assert not sheets.wanted("runs", parse("runs stop"))


# -- the chain between commands ----------------------------------------------------


def test_continue_with_becomes_the_next_step_in_the_session(monkeypatch):
    monkeypatch.setattr(c, "in_session", True)
    monkeypatch.setattr(c, "next_step", None)
    c.continue_with("poros export last", 80)
    assert c.next_step == "/export last"
    c.continue_with("runs 7f3a", 80)
    assert c.next_step == "/runs 7f3a"
    c.continue_with("pip install poros-train", 80)  # not a slash command: kept as it was
    assert c.next_step == "/runs 7f3a"
    monkeypatch.setattr(c, "next_step", None)
    c.continue_with("pip install poros-train", 80)  # alone: no other step to suggest
    assert c.next_step == ""


def test_outside_the_session_report_stays_runs_report(monkeypatch):
    monkeypatch.setattr(c, "in_session", False)
    monkeypatch.setattr(c, "next_step", None)
    assert c.command("runs report 7f3a") == "poros runs report 7f3a"
    c.continue_with("runs report 7f3a", 80)
    assert c.next_step is None


# -- /settings ---------------------------------------------------------------------


def test_settings_saves_what_the_commands_read(prefs, monkeypatch):
    from poros.cli import settings, theme

    monkeypatch.delenv("POROS_NO_ANIM", raising=False)
    sheet = Sheet(settings.settings_form())
    text = _text(sheet.render(80))
    assert "asks first" in text and "never runs repo code" in text
    values = dict(sheet.form.values, motion="off", export="adapter", theme="dark")
    settings.save(values)
    assert questions.preference("motion") == "off" and questions.preference("export") == "adapter"
    assert theme.current_mode() == "dark"
    import os

    assert os.environ.get("POROS_NO_ANIM") == "1"
    settings.save(dict(values, motion="on", export="ask", theme="auto"))
    assert questions.preference("export") == "" and "POROS_NO_ANIM" not in os.environ


# -- /machine ----------------------------------------------------------------------


def _machine(**kw):
    from poros.cli.schemas import DoctorCheck, PlanGpu
    from poros.cli.verbs.machine import MachineOutput

    checks = [DoctorCheck(name=n, value="1.0", status="ok") for n in ("Python", "torch", "peft")]
    base = dict(
        verdict="ready", version="0.3.0",
        gpu=PlanGpu(name="RTX 5090", total_gib=31.8, free_gib=30.9, source="nvidia-smi"),
        host_ram_free_gib=58.2,
        checks=checks + [DoctorCheck(name="Disk free", value="412 GB", status="ok")],
    )
    base.update(kw)
    return MachineOutput(**base)


@pytest.mark.parametrize("w", [60, 80, 100])
def test_the_machine_card_leads_with_the_verdict_and_names_no_model(w, monkeypatch):
    from poros.cli import views

    monkeypatch.setattr(c, "in_session", True)
    lines = views.machine(_machine(), w)
    text = _text(lines)
    assert "Ready to train" in text and "to fix" not in text
    assert "torch   1.0" in text and "peft" not in text  # torch speaks for the stack
    if w >= 94:  # the card drawn: what it is doing now, with gauges
        assert "GeForce RTX 5090" in text and "vram" in text and "idle" in text
    if w >= 76:  # room for Showcase: the same words, large and centred
        monkeypatch.setattr(views.machine_screens, "_room", lambda: 60)
        tall = _text(views.machine(_machine(), w))
        assert "GeForce RTX 5090" in tall and "vram" in tall and "idle" in tall
    # it checks the machine and names no model as fitting: what trains where is `poros recipes`
    assert "Qwen2.5-32B" not in text and "fits" not in text
    # the next step is a first run on the starter model
    assert c.next_step == "/train Qwen2.5-0.5B-Instruct --data"  # heads the home list
    assert all(width(line) <= w for line in lines), text


def test_the_machine_card_lists_only_what_needs_fixing_once(monkeypatch):
    from poros.cli import views
    from poros.cli.schemas import DoctorCheck

    monkeypatch.setattr(c, "in_session", True)
    checks = [DoctorCheck(name="Python", value="3.12", status="ok"),
              DoctorCheck(name="torch", value="not installed", status="warn"),
              DoctorCheck(name="peft", value="not installed", status="warn"),
              DoctorCheck(name="Disk free", value="9.6 GB", status="warn")]
    text = _text(views.machine(_machine(verdict="ready with warnings", checks=checks), 80))
    # the stack is one thing to fix however many of its packages are missing
    assert "needs the training stack and disk space" in text.splitlines()[0]
    assert "torch   not installed" in text and "peft" not in text
    assert "big downloads may not fit" in text
    assert "pip install poros-train" in text  # "In a shell:" in the session


def test_the_machine_json_names_no_model():
    obj = _machine().model_dump(by_alias=True, mode="json")
    assert obj["schema"] == "poros.machine.v1" and "can_train" not in obj


# -- the export card -----------------------------------------------------------


def _inline_form() -> Form:
    def build(v):
        return View("Export  test", [
            Field("--format", "as", value=v.get("--format"), inline=True,
                  options=lambda: [Option("merged", "merged"), Option("adapter", "adapter")]),
            Field("--output", "to", kind="text", value=None,
                  shown="/Users/someone/.poros/runs/1024-2136-2bfa/export/merged"),
        ], go="Export", result=("gives", "a standalone model", "bf16"))

    return Form("export", build, {"--format": "merged"})


def test_an_inline_row_shows_every_choice_and_enter_steps_through_them():
    sheet = Sheet(_inline_form())
    assert sheet.field is None  # everything set: enter exports at once
    sheet.sel = 0
    assert "● merged   ○ adapter" in _text(sheet.render(80))
    sheet.key("enter")
    assert sheet.form.values["--format"] == "adapter" and sheet.mode == "nav"
    sheet.key("char", "x")  # typing does not open a list on a two-choice row
    assert sheet.mode == "nav"


@pytest.mark.parametrize("w", [46, 57, 80])
def test_a_path_keeps_both_ends_when_narrow(w):
    text = _text(Sheet(_inline_form()).render(w))
    assert "export/merged" in text and all(width([(r, None)]) <= w for r in text.split("\n"))


def test_showcase_is_centred_and_only_where_it_fits(monkeypatch):
    from poros.cli import views
    from poros.cli.ui import gpu_image
    from poros.cli.verbs.machine import GpuNow

    monkeypatch.setattr(c, "in_session", True)
    monkeypatch.setattr(gpu_image, "protocol", lambda stream=None: "kitty")
    r = _machine()
    r.now = GpuNow(source="nvidia-smi", driver="570.86", used_gib=14.2, total_gib=24.0,
                   util_pct=97, temp_c=71, power_w=402, power_limit_w=450, apps=[])
    monkeypatch.setattr(views.machine_screens, "_room", lambda: 60)
    lines = views.machine(r, 100)
    text = _text(lines)
    assert all(word in text for word in ("load", "temp", "power", "97%", "71°C", "402 W"))
    # the card's own total (31.8) less what is in use now: one total on every row
    assert "17.6 GiB free" in text and "driver 570.86" in text
    assert all(width(line) <= 100 for line in lines)
    name = next(_text([line]) for line in lines if "GeForce RTX 5090" in _text([line]))
    pad = len(name) - len(name.lstrip())
    assert abs(pad - (100 - len(name.strip())) // 2) <= 1  # centred
    # shrinks to fit, then gives way to the side card, and never overflows
    for room in (60, 34, 28, 24, 18):
        monkeypatch.setattr(views.machine_screens, "_room", lambda room=room: room)
        assert len(views.machine(r, 100)) <= room


def test_showcase_never_scrolls_its_verdict_away_outside_the_session(monkeypatch):
    import os
    import shutil

    from poros.cli import views

    monkeypatch.setattr(c, "in_session", False)
    monkeypatch.setattr(shutil, "get_terminal_size",
                        lambda fallback=None: os.terminal_size((120, 24)))
    lines = views.machine(_machine(), 100)
    assert len(lines) <= 24 and "Ready to train" in _text(lines[:1])


def test_the_machine_card_says_who_is_on_the_card(monkeypatch):
    from poros.cli import views
    from poros.cli.verbs.machine import App, Busy, GpuNow

    monkeypatch.setattr(c, "in_session", True)
    r = _machine()
    r.now = GpuNow(source="nvidia-smi", driver="570.86", used_gib=14.2, total_gib=24.0,
                   util_pct=97, temp_c=71, power_w=402, power_limit_w=450,
                   apps=[App(pid=4411, name="python3", used_gib=14.0)])
    r.busy = [Busy(id="x", name="qwen-notes", model="Qwen/Qwen2.5-7B", state="running",
                   step=120, total=400)]
    text = _text(views.machine(r, 100))
    assert "driver 570.86" in text
    assert "97%" in text and "71°C" in text and "402 W of 450" in text
    assert "qwen-notes" in text and "step 120 of 400" in text
    assert c.next_step == "/runs"
    r.busy = []
    text = _text(views.machine(r, 100))
    assert "python3 (pid 4411)" in text and "not Poros" in text


def test_the_run_page_reads_the_same_as_the_runs_finished_card():
    from poros.cli import views

    x = {"id": "1105-2235-5069", "state": "succeeded", "name": "n", "model": "m", "total": 300,
         "output_dir": "/tmp/r",
         "step": 300, "started_at": "2026-11-06T04:35:00+00:00",
         "finished_at": "2026-11-06T04:35:12+00:00"}
    f = {"losses": [], "speeds": [], "vram_gib": [], "extra": {},
         "receipt": {"wall_s": 8.4, "peak_vram_allocated_gb": 6.8, "peak_vram_reserved_gb": 7.7,
                     "tokens_per_sec": 40938.0, "s_per_step": 0.03, "simulated": True,
                     "gpu": {"name": "RTX 4090 (simulated)", "total_gb": 25.8}}}
    x["receipt"] = f["receipt"]
    text = "\n".join("".join(seg[0] for seg in row) for row in views.run_page(x, f, 100))
    assert "in 8.4s" in text and "6.4 GiB peak" in text  # 6.8 GB = 6.33 GiB, rounded up
    assert "40,938 tok/s · 0.03 s/step" in text
    assert text.count("simulated") == 1
