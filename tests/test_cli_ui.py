"""The CLI's visual layer (poros/cli/ui, the CLI visual design).

Torch-free except the reporter-seam tests at the bottom, which reuse the tiny
CPU Qwen2 from test_progress_and_scheduler and skip without torch.
"""

from __future__ import annotations

import io
import re
import tempfile

import pytest

from poros.cli import session as sm
from poros.cli.ui import components as c
from poros.cli.ui import screens as sc
from poros.cli.ui.style import ACCENT, Palette, _nearest_256
from poros.cli.ui.text import plain, render, width

WIDTHS = (140, 100, 80, 72, 56, 40)
SGR = re.compile(r"\x1b\[[0-9;]*m")


def _state(**kw):
    st = sc.TrainState(
        model="Qwen/Qwen2.5-72B-Instruct",
        arch="dense · 80 layers · 20 blocks",
        gpu="RTX 4090",
        card_gb=24.0,
        output_dir="./runs/qwen2.5-72b-instruct-lora-7f3a91c2",
        run="7f3a91c2",
        step=412,
        total=1000,
        losses=[2.9 - i * 0.002 for i in range(412)],
        step_time=9.84,
        step_times=[9.84] * 20,
        tokens_per_sec=1665.0,
        vram_gb=17.6,
        vram_peak_gb=18.1,
        ram_gb=96.4,
        ram_total_gb=128.0,
        stream_unit="block",
        stream_pos=7,
        stream_count=20,
        stream_gbs=21.4,
        elapsed=4032.0,
    )
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def _result():
    return sc.TrainResult(
        steps=1000,
        wall_s=9850.0,
        final_loss=1.1832,
        first_loss=2.8683,
        losses=[2.9 - i * 0.0017 for i in range(1000)],
        peak_vram_gb=18.1,
        card_gb=24.0,
        peak_ram_gb=101.7,
        ram_total_gb=128.0,
        output_dir="./runs/qwen2.5-72b-instruct-lora-7f3a91c2",
        receipt="./runs/qwen2.5-72b-instruct-lora-7f3a91c2/receipt.json",
        tokens_per_sec=1665.0,
        next_cmd="poros eval train.yaml",
    )


SCREENS = {
    "live": lambda w: sc.train_surface(_state(), 3, w),
    "done": lambda w: sc.train_done(_result(), w),
    "error": lambda w: sc.error_screen(
        "Out of memory",
        "the GPU ran out of memory at step 412",
        w,
        context=["block 4 · seq 8192 · peak 23.9 of 24.0 GB"],
        fix=sc.Fix("Retry", "poros train train.yaml --block-size 2", "slower"),
    ),
    "menu": lambda w: sc.choice_menu(
        "Out of memory", [("Retry with --block-size 2", "slower")], 0, w
    ),
}


@pytest.mark.parametrize("name", sorted(SCREENS))
@pytest.mark.parametrize("w", WIDTHS)
def test_no_line_exceeds_the_terminal(name, w):
    """Every row fits its width; a bare path is the one thing allowed to wrap."""
    for line in SCREENS[name](w):
        text = plain(line)
        if width(line) > w:
            assert text.strip(" │").startswith("./"), (name, w, text)


@pytest.mark.parametrize("w", (80, 100, 140))
def test_content_caps_at_100_columns(w):
    for line in sc.train_done(_result(), w):
        if not plain(line).strip(" │").startswith("./"):
            assert width(line) <= 100


def test_sparkline_drops_below_80_and_plain_below_60():
    wide = "\n".join(plain(line) for line in sc.train_surface(_state(), 0, 100))
    narrow = "\n".join(plain(line) for line in sc.train_surface(_state(), 0, 72))
    tiny = "\n".join(plain(line) for line in sc.train_surface(_state(), 0, 56))
    assert "▁" in wide or "█" in wide
    assert not any(ch in narrow.split("vram")[0] for ch in "▁▂▃▄▅▆▇")  # loss row has no chart
    assert "│" not in tiny and "loss: " in tiny  # plain key: value lines


def test_a_wide_terminal_gets_a_taller_chart_not_a_wider_one():
    def loss_rows(w):
        return [
            row for row in sc.train_surface(_state(), 0, w) if any(ch in plain(row) for ch in "█▇")
        ]

    # the drawn curve from 76 up, the same height at any width; a row of sparks under that
    assert len(loss_rows(140)) == len(loss_rows(100)) > 3 and len(loss_rows(70)) <= 1


def test_accent_is_the_brand_orange_and_never_index_203():
    p256 = Palette(depth="256", dark=True)
    assert p256.sgr("a") == "38;5;208"
    assert _nearest_256(ACCENT) == 208
    assert Palette(depth="truecolor", dark=True).sgr("a") == "38;2;255;107;53"


def test_no_color_keeps_bold_drops_color():
    pal = Palette(depth="none", dark=True)
    out = render(SCREENS["live"](100), pal)
    codes = set(SGR.findall(out))
    assert codes <= {"\x1b[1m", "\x1b[0m"}, codes


def test_greys_are_the_site_tiers_and_flip_on_light():
    dark, light = Palette("truecolor", True), Palette("truecolor", False)
    assert dark.sgr("hi") == "1;38;2;252;252;252"  # paper on ink
    assert light.sgr("hi") == "1;38;2;22;22;22"  # ink on paper
    assert dark.sgr("dm") == "38;2;137;137;137"  # 50% mix


def test_spectrum_ends_gold_at_the_current_point():
    rows = c.spark([3.0, 2.0, 1.5, 1.2], 10)
    last = rows[0][-1][1]
    assert last == ("sp", 1.0)
    bar = c.flow_bar(0.4, 20)
    assert bar[0][1] == ("sp", 0.0) and bar[7][1] == ("sp", 1.0)  # leading edge = gold


def test_statusline_drops_items_from_the_right():
    rows = c.statusline(
        ["run 7f3a91c2", "dense", "./runs/a-very-long-output-directory-name", "ctrl-c to stop"],
        48,
    )
    text = plain(rows[1])
    assert text.startswith(" run 7f3a91c2") and "run 7f3a91c2" in text and "ctrl-c" not in text
    assert plain(rows[0]) == "─" * 48


def test_paths_are_never_truncated():
    path = "./runs/" + "x" * 120 + "/receipt.json"
    lines = c.rail_path("receipt", path, 80)
    assert path in "\n".join(plain(line) for line in lines)


def test_welcome_drops_items_to_fit():
    line = c.welcome("poros", "Qwen/Qwen3-30B-A3B-Instruct-2507", "RTX 4090 24.0 GB", max_width=50)
    assert width(line) <= 50 and plain(line).startswith("▛▜ poros v")


# --------------------------------------------------------------------------
# live surface
# --------------------------------------------------------------------------


def test_live_without_a_tty_prints_plain_lines_and_heartbeats():
    from poros.cli.ui.live import Live

    buf = io.StringIO()
    beats = []
    live = Live(
        lambda t, w: [[("surface", "tx")]],
        stream=buf,
        heartbeat=lambda t: beats.append(t) or "step 1/3  loss 2.0000",
        heartbeat_s=0.0,
    )
    assert live.live is False
    live.start()
    live.print([c.event("Loaded", "tiny", "1.0s")])
    live.update()
    live.close([c.event("Trained", "3 steps")])
    out = buf.getvalue()
    assert "\x1b[" not in out  # no escape codes into a pipe
    assert "● Loaded tiny  1.0s" in out and "step 1/3" in out and "● Trained 3 steps" in out
    assert "surface" not in out  # the live region is never drawn without a tty


def test_live_clip_keeps_rows_from_wrapping():
    from poros.cli.ui.live import clip

    assert width(clip([("a" * 50, "tx"), ("b" * 50, "dm")], 60)) == 60


# --------------------------------------------------------------------------
# completion
# --------------------------------------------------------------------------


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "train.yaml").write_text(
        "model_name_or_path: Qwen/Qwen2.5-7B\ndataset: synthetic\n"
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "chat.jsonl").write_text("{}\n")
    ck = tmp_path / "runs" / "run-a" / "checkpoint-50"
    ck.mkdir(parents=True)
    (tmp_path / "runs" / "run-a" / "events.jsonl").write_text("")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    from poros.cli.ui import complete as cm

    cm._CACHE.clear()
    return cm


def test_completes_commands_from_the_real_cli(project):
    comp = project.complete("/tr")
    assert comp.best.text == "/train" and comp.ghost == "ain "
    assert {x.text for x in project.complete("bench ").candidates} >= {
        "run",
        "matrix",
        "compare",
        "report",
    }


def test_completes_flags_that_exist_and_skips_used_ones(project):
    flags = [x.text for x in project.complete("train train.yaml --").candidates]
    assert "--steps" in flags and "--block-size" in flags and "--seq-len" in flags
    flags = [x.text for x in project.complete("train train.yaml --steps 5 --").candidates]
    assert "--steps" not in flags


def test_completes_values_from_the_machine(project):
    assert project.complete("train ").best.text == "train.yaml"
    assert project.complete("train ").best.desc == "Qwen/Qwen2.5-7B"  # the model it trains
    ds = [x.text for x in project.complete("init -d ").candidates]
    assert "data/chat.jsonl" in ds and not any(t.startswith("synthetic") for t in ds)
    ck = [x.text for x in project.complete("train train.yaml -r ").candidates]
    assert ck[0] == "latest" and any(t.endswith("checkpoint-50") for t in ck)
    assert project.complete("schema ").best.text == "config"
    assert any("Qwen/" in x.text for x in project.complete("check Qwen").candidates)


def test_history_autosuggest_wins_like_fish(project):
    comp = project.complete("train tr", history=["train train.yaml --steps 100"])
    assert comp.source == "history" and comp.ghost == "ain.yaml --steps 100"


# --------------------------------------------------------------------------
# the trainer's reporter seam (CPU, tiny model)
# --------------------------------------------------------------------------


def test_reporter_receives_every_step_and_silences_plain_prints(capsys):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    from poros.train.reporting import Reporter
    from tests.test_progress_and_scheduler import _loader, _make_trainer, _tiny_peft_qwen2

    class Rec(Reporter):
        def __init__(self):
            self.calls = []

        def on_start(self, trainer, total, start_step):
            self.calls.append(("start", total, start_step))

        def on_step(self, r):
            self.calls.append(("step", r.step, r.total, round(r.loss, 6)))

        def on_done(self, metrics):
            self.calls.append(("done", metrics["total_steps"]))

    torch.manual_seed(0)
    model, cfg = _tiny_peft_qwen2(torch)
    rec = Rec()
    with tempfile.TemporaryDirectory() as d:
        tr = _make_trainer(torch, model, d, max_steps=3, progress=True)
        tr.reporter = rec
        metrics = tr.train(_loader(torch, cfg.vocab_size), steps=3, seq_len=16)
    out = capsys.readouterr().out
    assert rec.calls[0] == ("start", 3, 0)
    steps = [c for c in rec.calls if c[0] == "step"]
    assert [s[1] for s in steps] == [1, 2, 3] and all(s[2] == 3 for s in steps)
    assert [s[3] for s in steps] == [round(x, 6) for x in metrics["loss_history"]]
    assert rec.calls[-1] == ("done", 3)
    assert "s/step" not in out and "Poros Training Complete" not in out  # the reporter owns stdout


def test_train_view_tracks_steps_from_the_seam():
    from poros.cli.ui.train_view import TrainView
    from poros.train.reporting import StepReport

    st = sc.TrainState(model="m")
    view = TrainView(st, stream=io.StringIO())
    view.on_start(object(), 10, 0)
    view.on_step(
        StepReport(
            step=1,
            total=10,
            loss=2.0,
            step_time=0.5,
            tokens=100,
            tokens_per_sec=200,
            lr=1e-4,
            vram_gb=1.0,
            vram_peak_gb=1.5,
        )
    )
    assert st.step == 1 and st.losses == [2.0] and st.eta() == "4.5s"
    rows = view._surface(0, 100)
    assert any("Step 1 of 10" in plain(r) for r in rows)
    view.close()


# --------------------------------------------------------------------------
# the interactive session's footer (fx model), no tty needed
# --------------------------------------------------------------------------


@pytest.fixture
def session(project, monkeypatch):
    from poros.cli import session as sm

    monkeypatch.setattr(c, "next_step", None)  # another test's next step heads no list here
    s = sm.Session(animate=False)
    s._runs = False
    s.pal = Palette("truecolor", True)
    s._gpu = "RTX 4090 24.0 GB"
    return s


def _plain_rows(rows):
    return [plain(r) for r in rows]


def test_idle_footer_is_composer_blank_statusline(session):
    session.s.buf, session.s.caret = "train", 5
    rows, crow, ccol = session.footer(100)
    text = _plain_rows(rows)
    assert text[0] == text[2] == "─" * 100 and text[1].startswith("▛▜ ❯ train")
    assert text[-1].startswith("RTX 4090 24.0 GB · train.yaml · Qwen2.5-7B")
    assert (crow, ccol) == (1, 10)


def test_home_lists_the_commands_with_the_next_step_on_top(session, monkeypatch):
    text = _plain_rows(session.footer(100)[0])
    assert [r.split()[0] for r in text[3:5]] == ["/recipes", "/train"]
    assert "/eval" not in "".join(text)  # no run yet
    monkeypatch.setattr(c, "in_session", True)
    monkeypatch.setattr(c, "next_step", None)
    assert c.continue_with("poros train <model> --data notes.csv", 100) == []
    assert c.next_step == "/train --data notes.csv"  # the sheet asks for the model
    text = _plain_rows(session.footer(100)[0])
    assert text[3].startswith("  /train --data notes.csv") and "next" in text[3]


def test_slash_opens_the_command_picker(session):
    session.s.buf, session.s.caret = "/", 1
    text = _plain_rows(session.footer(100)[0])
    assert text[2] == "─" * 100
    names = [r.split()[0] for r in text[3:3 + len(sm.MENU)]]
    assert names == ["/" + n for n, _ in sm.MENU]  # every one, dropped down under the box
    assert "a whole setup" in text[3] and text[-1].startswith("⏎ run")


def test_the_footer_never_shrinks_while_typing(session):
    home = len(session.footer(100)[0])  # the home list, first
    session.s.buf, session.s.caret = "/", 1
    tall = len(session.footer(100)[0])  # every command: it may grow, once
    assert tall >= home
    session.s.buf, session.s.caret = "/mach", 5
    rows = _plain_rows(session.footer(100)[0])
    assert len(rows) == tall and rows[3].split()[0] == "/machine"
    session.s.buf, session.s.caret = "", 0
    assert len(session.footer(100)[0]) == tall  # closing leaves room, never a jump


def test_arguments_open_a_results_picker(session):
    session.s.buf, session.s.caret = "train train.yaml --st", 21
    text = _plain_rows(session.footer(100)[0])
    assert any(r.strip().startswith("--steps") for r in text)
    assert any("tab use" in r and "⏎ run" in r for r in text)


def test_the_machine_card_draws_the_gpu():
    from poros.cli.ui import gpu_art

    rows = [plain(r) for r in gpu_art.card("RTX 5090", 0.0)]
    cw, ch = gpu_art.size(gpu_art.form("RTX 5090"))
    assert len(rows) == ch and all(len(r) <= cw <= gpu_art.W for r in rows)
    assert any(set(r) - {" "} <= {chr(c) for c in range(0x2800, 0x2900)} and r.strip()
               for r in rows)  # drawn in braille
    assert gpu_art.card("RTX 5090", 0.0) != gpu_art.card("RTX 5090", 0.7)  # they turn
    assert gpu_art.passive("H100") and not gpu_art.passive("RTX 4090")
    assert gpu_art.vendor("Radeon RX 7900") == "AMD" and gpu_art.vendor("mystery") == ""


def test_each_card_is_drawn_in_its_own_form():
    from poros.cli.ui import gpu_art

    form = gpu_art.form
    assert form("NVIDIA GeForce GTX 1080").style == "blower"
    assert form("NVIDIA GeForce RTX 2080 Ti").fans == 2
    assert form("NVIDIA GeForce RTX 4090").style == "flow"  # the Founders Edition
    assert form("NVIDIA GeForce RTX 4090", "ASUS").fans == 3  # a partner card
    assert form("NVIDIA GeForce RTX 5090").style == "dual"
    assert form("NVIDIA GeForce RTX 3060").edition == ""  # no Founders Edition
    assert form("NVIDIA RTX A6000").style == "blower"
    assert form("NVIDIA H100 PCIe").style == "passive" and form("NVIDIA L4").slots == 1
    assert form("NVIDIA H100 80GB HBM3").edition == "module"
    assert form("AMD Radeon RX 7900 XTX").fans == 3
    assert form("Intel(R) Arc(TM) B580 Graphics").fans == 2
    assert form("NVIDIA RTX 6000 Ada Generation").style == "blower"  # not a "60 series"
    # sizes follow the card: a 1060 is drawn shorter than a 4090
    assert gpu_art.size(form("GTX 1060"))[0] < gpu_art.size(form("RTX 4090"))[0]
    assert gpu_art.board("0x16101043") == "ASUS" and gpu_art.board("") == ""
    assert gpu_art.label("RTX 4090") == "Founders Edition"
    assert gpu_art.label("RTX 4090", "MSI") == "MSI"
    assert gpu_art.brand("RTX 4090") == "GEFORCE RTX" and gpu_art.brand("RTX 4090", "MSI") == "MSI"
    assert gpu_art.brand("Radeon RX 7900 XTX") == "RADEON"


def test_the_gpu_image_takes_the_terminals_colours(tmp_path, monkeypatch):
    from poros.cli.ui import gpu_image
    from poros.cli.ui.style import Palette

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    solar = Palette(depth="truecolor", dark=True, ink=(131, 148, 150), ground=(0, 43, 54))
    paper = Palette(depth="truecolor", dark=False, ink=(22, 22, 22), ground=(252, 252, 252))
    assert gpu_image.render("L4", solar) != gpu_image.render("L4", paper)
    assert len(list((tmp_path / "poros" / "gpu").iterdir())) == 2  # drawn once each, kept
    orange = Palette(depth="truecolor", dark=False, ink=(20, 20, 20), ground=(255, 120, 60))
    assert gpu_image._tones(orange)["accent"][0] == (20, 20, 20)  # the accent gives way


def test_the_gpu_image_is_a_png_in_a_kitty_escape(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from poros.cli.ui import gpu_image, text
    from poros.cli.ui.style import Palette

    png = gpu_image.render("RTX 5090", Palette(depth="truecolor", dark=True))
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    esc = gpu_image.escape(png, 46)
    assert esc.startswith("\x1b_Ga=T,f=100") and "c=46," in esc and esc.endswith("\x1b\\")
    line = [("  ", None), (esc, "raw"), ("x", "hi")]
    assert text.width(line) == 3 and text.plain(line) == "  x"
    assert 4 <= gpu_image.rows("RTX 5090") < gpu_image.cols("RTX 5090")
    monkeypatch.setenv("POROS_IMAGES", "0")
    assert not gpu_image.supported()


def test_the_terminal_says_how_it_shows_images(monkeypatch):
    from poros.cli import theme

    def answer(resp):
        theme._reset_cache()
        monkeypatch.setattr(theme, "_ask_terminal", lambda: resp)
        monkeypatch.setattr(theme.sys.stdin, "isatty", lambda: True, raising=False)
        monkeypatch.setattr(theme.sys.stdout, "isatty", lambda: True, raising=False)
        return theme.terminal_graphics()

    kitty = answer(b"\x1b]11;rgb:0000/0000/0000\x1b\\\x1b_Gi=31;OK\x1b\\"
                   b"\x1b[6;36;16t\x1b[?62;22c")
    assert kitty == theme.Graphics(kitty=True, sixel=False, cell=(16, 36))
    assert theme._query_terminal_colors()[1] == (0, 0, 0)  # the same one answer
    assert answer(b"\x1b[?64;4;22c") == theme.Graphics(sixel=True)
    assert answer(b"\x1b[?1;2c") == theme.Graphics()
    theme._reset_cache()


def test_each_terminal_gets_the_image_it_can_show(monkeypatch):
    from poros.cli import theme
    from poros.cli.ui import gpu_image

    class Tty:
        def isatty(self):
            return True

    for k in ("POROS_IMAGES", "NO_COLOR", "TMUX", "KITTY_WINDOW_ID", "LC_TERMINAL",
              "TERM_PROGRAM"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    seen = theme.Graphics()
    monkeypatch.setattr(theme, "terminal_graphics", lambda: seen)

    def pick(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        try:
            return gpu_image.protocol(Tty())
        finally:
            for k in env:
                monkeypatch.delenv(k)

    assert pick() is None
    assert pick(TERM_PROGRAM="ghostty") == "kitty"
    assert pick(TERM_PROGRAM="iTerm.app") == "iterm"
    assert pick(TERM_PROGRAM="WezTerm") == "iterm"
    assert pick(LC_TERMINAL="iTerm2") == "iterm"
    assert pick(TERM_PROGRAM="Apple_Terminal") is None
    assert pick(TERM_PROGRAM="ghostty", TMUX="/tmp/tmux-1/default,1,0") is None
    assert pick(POROS_IMAGES="sixel") == "sixel"
    seen = theme.Graphics(sixel=True)
    assert pick() == "sixel"
    seen = theme.Graphics(kitty=True, sixel=True)
    assert pick(TERM_PROGRAM="WezTerm") == "kitty"  # it said it speaks kitty
    assert gpu_image.protocol(io.StringIO()) is None  # a pipe


def test_the_gpu_image_as_iterm2_and_sixel(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    from poros.cli.ui import gpu_image, text
    from poros.cli.ui.style import Palette

    pal = Palette(depth="truecolor", dark=True, ink=(200, 200, 200), ground=(10, 20, 30))
    seg, aw, ah, at = gpu_image.placed("L4", pal, "", "iterm")
    assert at == ah - 1 and seg.startswith(f"\x1b7\x1b[{ah - 1}A\x1b]1337;File=inline=1;")
    assert f"width={aw};" in seg and seg.endswith("\x07\x1b8")
    assert text.width([(seg, "raw")]) == 0
    six = gpu_image.render("L4", pal, "", "sixel", 200)
    assert six.startswith(b'\x1bP0;1;0q"1;1;200;') and six.endswith(b"\x1b\\")
    head = six.split(b"#", 1)[1]
    assert head.startswith(b"0;2;")  # a palette, in percent
    assert 0 < six.count(b";2;") <= 255  # at most 255 colours
    assert gpu_image.render("L4", pal, "", "sixel", 200) is six  # drawn once


def test_esc_and_ctrl_c_arm_their_hints(session):
    import time as _t

    session.s.armed = ("esc", _t.monotonic())
    assert plain(session._status_row(100)).endswith("esc again to clear")
    session.s.armed = ("ctrl-c", _t.monotonic())
    assert plain(session._status_row(100)) == "press ctrl+c again to exit"


def test_busy_footer_tells_how_to_stop(session):
    rows = session.busy_footer([], 100, "")
    assert plain(rows[1]) == "▛▜ ❯" and plain(rows[-1]).endswith("ctrl+c to stop")
    assert plain(rows[0]) == plain(rows[2]) == "─" * 100  # the idle footer's box
    assert rows[1][0][1] != "dm"  # the mark keeps the brand while it breathes


def test_loaders_keep_a_fixed_slot_and_have_a_still_frame():
    from poros.cli.ui import loaders

    for kind in loaders.LOADERS:
        for t in range(40):
            assert width(loaders.frame(kind, t)) == loaders.WIDTH, (kind, t)
        assert width(loaders.still(kind)) == loaders.WIDTH
    assert loaders.for_label("Loading Qwen2.5-7B") == "river"
    assert loaders.for_label("Training") == "wave"
    assert loaders.for_label("Saving adapter") == "dots"


def test_activity_clock_is_fixed_width_so_the_bar_never_moves():
    a = c.activity("Training", "9.9s", 0, frac=0.5, w=100)
    b = c.activity("Training", "1h 07m 12s", 0, frac=0.5, w=100)
    assert plain(a).index("━") == plain(b).index("━")


def test_a_row_that_did_not_change_is_not_written_again(monkeypatch):
    from poros.cli.ui import live as lv
    from poros.cli.ui.style import Palette

    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("POROS_NO_ANIM", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    out = Tty()
    tick = {"n": 0}

    def surface(t, w):
        return [[(f"frame {tick['n']}", None)], [("\x1b_Gimage\x1b\\", "raw")]]

    live = lv.Live(surface, stream=out, palette=Palette(depth="truecolor", dark=True))
    live._draw()
    tick["n"] += 1
    live._draw()
    assert out.getvalue().count("\x1b_Gimage") == 1  # sent once, then stepped over
    assert "frame 1" in out.getvalue()
    live._closed = True


def test_json_clears_a_closed_live_area_first(monkeypatch, capsys):
    from poros.cli import contract
    from poros.cli.schemas import Output
    from poros.cli.ui import live as lv

    out = io.StringIO()
    lv.defer(out, h=5)
    monkeypatch.setattr(contract, "stdout", lambda: out)

    class R(Output):
        pass

    contract.emit(R())
    text = out.getvalue()
    assert text.startswith("\x1b[5A\r\x1b[J") and text.rstrip().endswith("}")
    assert lv.take_pending() == ""
    contract.reset()


def test_work_quicker_than_the_delay_draws_no_loading_frame(monkeypatch):
    from poros.cli.ui import live as lv
    from poros.cli.ui.style import Palette

    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.delenv("POROS_NO_ANIM", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    out = Tty()
    live = lv.Live(lambda t, w: [[("Checking", None)]], stream=out, delay=5.0,
                   palette=Palette(depth="truecolor", dark=True))
    live.start()
    live.close([[("Checked", None)]])
    assert "Checking" not in out.getvalue() and "Checked" in out.getvalue()


class _Tty(io.StringIO):
    def isatty(self):
        return True


def test_a_card_opens_full_screen_once_however_many_cards_chain(monkeypatch):
    from poros.cli.ui import sheet as sh

    monkeypatch.delenv("POROS_INLINE", raising=False)
    monkeypatch.delenv("ZELLIJ", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    out = _Tty()
    with sh.fullscreen(out):
        with sh.fullscreen(out):  # a recipe opening the train card
            pass
    assert out.getvalue().count("\x1b[?1049h") == 1 and out.getvalue().endswith("\x1b[?1049l")
    monkeypatch.setenv("POROS_INLINE", "1")
    out = _Tty()
    with sh.fullscreen(out):
        pass
    assert out.getvalue() == ""  # asked to stay inline: scrollback, as before


def test_the_canvas_centres_a_card_and_never_scrolls():
    from poros.cli.ui import sheet as sh

    rows = sh.canvas([[("x" * 100, None)]] * 50, 160, 30)
    assert len(rows) == 30 and rows[sh.TOP][0] == (" " * 30, None)


def test_esc_on_a_full_screen_card_leaves_no_line_and_go_leaves_the_command(monkeypatch):
    from poros.cli import app
    from poros.cli.ui import live

    live.defer(io.StringIO(), lead="typed", climb="UP")
    assert app._tell(None) and live.take_pending() == "\rUP"  # the composer goes, nothing else
    live.defer(io.StringIO(), lead="typed", climb="UP")
    assert app._tell(["train", "my model"])
    shown = live.take_pending()
    assert shown.startswith("\rUP") and "/train 'my model'" in shown
    assert not app._tell(["train"])  # nothing waiting: one-shot keeps its note


def test_the_start_box_never_leaves_a_word_alone():
    from poros.cli.ui import intro

    lines = intro.balanced("Fine-tune models bigger than your GPU.", 30)
    assert len(lines) == 2 and min(map(len, lines)) > len("your GPU.")
    assert intro.balanced("short", 40) == ["short"]
