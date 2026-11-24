"""A run's card from the load to the finish: one layout, its values "–" until they exist."""

from __future__ import annotations

import re

import pytest

from poros.cli.ui import components as c
from poros.cli.ui import run_card as rc
from poros.cli.ui import screens as sc
from poros.cli.ui.text import plain


def _rows(st: sc.TrainState, w: int, t: int = 0, animate: bool = False) -> list[str]:
    return [plain(r) for r in sc.train_surface(st, t, w, animate=animate)]


def _moments(pairs: bool = False) -> list[tuple[str, sc.TrainState]]:
    """The live states of one run, in order: loading, the wait for step 1, step 1, step 2. Each a
    state of its own, as the reporter leaves it at that moment."""
    out = []
    st = sc.TrainState(model="Qwen/Qwen2.5-7B-Instruct", card_gb=24.0, total=16, run="x",
                       label="Loading Qwen2.5-7B-Instruct", scored=True, pairs=pairs,
                       stream_soon=True, ram_gb=0.1, ram_total_gb=44.0, elapsed=190.0)
    out.append(("loading", st))
    st = sc.TrainState(**{**vars(st), "label": "Training", "elapsed": 12.0, "stream_soon": False,
                          "stream_unit": "block", "stream_count": 7, "stream_pos": 3,
                          "evals": [(0, 2.91)], "scores": [rc.Score(0, 2.91, 0, 2, "ends")]})
    out.append(("warm-up", st))
    extra = {"accuracy": 0.5, "margin": 0.1, "chosen_reward": 0.06, "rejected_reward": -0.04}
    step = dict(step_time=2.5, step_times=[2.5], tokens_per_sec=410.0, vram_gb=5.7,
                vram_peak_gb=6.4, extra=extra if pairs else {})
    st = sc.TrainState(**{**vars(st), **step, "step": 1, "losses": [2.83], "series": None})
    out.append(("step 1", st))
    st = sc.TrainState(**{**vars(st), "step": 2, "losses": [2.83, 2.42], "step_times": [2.5] * 2,
                          "series": None, "elapsed": 17.0})
    out.append(("step 2", st))
    losses = [2.83, 2.42, 2.2, 2.05, 1.9, 1.8, 1.72, 1.66]
    st = sc.TrainState(**{**vars(st), "step": 8, "losses": losses, "step_times": [2.5] * 8,
                          "series": None, "evals": [(0, 2.91), (8, 1.79)],
                          "scores": [rc.Score(0, 2.91, 0, 2, "ends"),
                                     rc.Score(8, 1.79, 1, 2, "ends")]})
    out.append(("a checkpoint's score", st))
    return out


def _labels(row: str) -> list[tuple[int, str]]:
    """The words on a row that are not values: no digit, no dash, no plotted cell, and not the
    checkpoints' step column ("base" gives way to the best score where there is room for one)."""
    return [(m.start(), m.group()) for m in re.finditer(r"\S+", row)
            if not re.search(r"[\d–▁▂▃▄▅▆▇█●○━─╸╺]", m.group()) and m.group() != "base"]


@pytest.fixture
def room(monkeypatch):
    def at(rows: int) -> None:
        monkeypatch.setattr(sc, "_room", lambda: rows)
    return at


@pytest.mark.parametrize("w", [100, 80, 76])
@pytest.mark.parametrize("rows", [34, 20])  # a tall curve, and a short one
@pytest.mark.parametrize("pairs", [False, True])
def test_nothing_moves_between_the_phases(w, rows, pairs, room):
    """Loading, the wait for step 1, step 1 and step 2 draw one layout: the same rows, every
    label and rule in the same cell. Only values, the clock, the loader, the progress and the
    plotted cells change (and the phase, on the activity line)."""
    room(rows)
    frames = [(name, _rows(st, w)) for name, st in _moments(pairs)]
    first = frames[0][1]
    act = next(i for i, r in enumerate(first) if "Loading" in r)
    for name, rows_now in frames[1:]:
        assert len(rows_now) == len(first), name
        assert "Training" in rows_now[act], name
        for i, (was, now) in enumerate(zip(first, rows_now)):
            if i == act:
                continue
            gone = [x for x in _labels(was) if x not in _labels(now)]
            assert not gone, (name, i, was, now)
            rails = [k for k, ch in enumerate(was) if ch == "│"]
            assert [k for k, ch in enumerate(now) if ch == "│"][:len(rails)] == rails, (name, i)


@pytest.mark.parametrize("w", [100, 80, 76])
def test_a_card_with_no_loss_yet_reads_dashes(w, room):
    room(34)
    st = _moments()[0][1]
    text = "\n".join(_rows(st, w))
    assert "● Step – of 16  –" in text
    rows = text.split("\n")
    loss = next(i for i, r in enumerate(rows) if r.rstrip().endswith("loss"))
    assert rows[loss + 1].rstrip().endswith("–")
    assert re.search(r"│\s+1\s+8\s+16", text)  # the axis, 1 to the run's steps, from the start
    for name in ("vram –", "speed      –", "stream     –", "base         –"):
        assert name in text, name
    assert "Loading Qwen2.5-7B-Instruct" in rows[-2] or "Loading Qwen2.5-7B-Instruct" in text
    assert "%" not in text and "GiB/s" not in text


def test_before_the_run_knows_its_steps_the_axis_holds_their_place(room):
    """A run by epochs learns its step count only once its data is read: the count reads "–"
    where it will stand, and the axis keeps its width."""
    room(34)
    st = sc.TrainState(model="m", card_gb=24.0, label="Loading m", ram_gb=0.1)
    text = "\n".join(_rows(st, 100))
    assert "● Step – of –  –" in text
    axis = next(r for r in _rows(st, 100) if re.search(r"│\s+1\s+–\s*$", r))
    known = sc.TrainState(model="m", card_gb=24.0, label="Loading m", ram_gb=0.1, total=16)
    axis16 = next(r for r in _rows(known, 100) if re.search(r"│\s+1\s+8\s+16\s*$", r))
    assert len(axis.rstrip()) == len(axis16.rstrip())


@pytest.mark.parametrize("w", [100, 80, 76])
def test_one_loss_is_one_dot_on_the_card(w, room):
    room(34)
    st = _moments()[2][1]
    rows = _rows(st, w)
    text = "\n".join(rows)
    assert "● Step 1 of 16  6%" in text
    chart = [r for r in rows if r.startswith("  │")]
    dots = sum(r[3:].count("●") for r in chart)
    assert dots == 1 and not any(ch in text for ch in "▁▂▃▄▅▆▇█")
    assert "2.8300" in text and "↓" not in text and "↑" not in text  # one step falls from nothing
    assert "speed      410 tok/s · 2.50 s/step" in text


def test_two_losses_draw_the_first_segment(room):
    room(34)
    text = "\n".join(_rows(_moments()[3][1], 100))
    assert "● Step 2 of 16  12%" in text and "↓ 14%" in text
    assert any(ch in text for ch in "▁▂▃▄▅▆▇█")


def _done(losses, **kw):
    base = dict(steps=len(losses), wall_s=3.0, final_loss=losses[-1] if losses else float("nan"),
                first_loss=losses[0] if losses else None, losses=losses, peak_vram_gb=6.4,
                card_gb=24.0, peak_ram_gb=0.1, ram_total_gb=44.0, output_dir="./runs/x",
                receipt=None, tokens_per_sec=410.0, recipe="sft")
    return sc.TrainResult(**{**base, **kw})


@pytest.mark.parametrize("w", [100, 76])
def test_in_the_session_a_one_step_run_finishes_into_its_card_with_one_dot(w, room, monkeypatch):
    """The finished card is drawn from the live card's line once it has a point: one step is one
    dot in the first column, the axis says "1" once, and the loss is the run's own result."""
    from poros.cli.ui.curve import Series

    room(34)
    monkeypatch.setattr(c, "in_session", True)
    r = _done([], steps=1, final_loss=2.83)  # one row a step: the card curve has no point
    r.series = Series(1).extend([2.83])
    rows = [plain(x) for x in sc.train_done(r, w)]
    text = "\n".join(rows)
    assert "╭─ Trained" in text and "2.8300" in text and "↓" not in text and "–" not in text
    chart = [x for x in rows if x.startswith("  │") and "final loss" not in x]
    assert sum(x[3:].count("●") for x in chart) == 1
    axis = next(x for x in rows if re.match(r"^  │\s+1\s", x))
    assert re.findall(r"\d+", axis.strip(" │")) == ["1"]


def test_with_no_point_drawn_the_finish_is_the_summary_and_keeps_its_loss(room, monkeypatch):
    room(34)
    monkeypatch.setattr(c, "in_session", True)
    text = "\n".join(plain(x) for x in sc.train_done(_done([], steps=12, final_loss=1.8765), 100))
    assert "● Trained 12 steps" in text and "1.8765" in text and "╭─" not in text


def test_a_finite_result_is_never_a_dash_on_the_card():
    from poros.cli.ui.curve import Series

    k = rc.Card(series=Series(12), last=11, final=1.8765, pct=100, meta="12 steps")
    text = "\n".join(plain(x) for x in rc.card(k, 100, rc.FINAL))
    assert "1.8765" in text


def test_one_step_of_one_draws_one_dot_live(room):
    room(34)
    st = sc.TrainState(model="m", card_gb=24.0, total=1, step=1, losses=[2.0], label="Training",
                       vram_gb=5.0, vram_peak_gb=5.5, ram_gb=0.1, step_time=1.0, step_times=[1.0],
                       tokens_per_sec=100.0)
    rows = _rows(st, 100)
    chart = [x for x in rows if x.startswith("  │")]
    assert sum(x[3:].count("●") for x in chart) == 1
    axis = next(x for x in rows if re.match(r"^  │\s+1\s*$", x))
    assert axis.strip(" │") == "1"


def test_the_finished_card_keeps_the_live_cards_geometry(room):
    """The card's rows, its axis and the readout's column stay where the live card drew them."""
    room(34)
    st = _moments()[3][1]
    live = _rows(st, 100)
    r = _done(st.losses, steps=2)
    r.series = st.series
    done = [plain(x) for x in sc.train_done(r, 100)]
    axis = next(i for i, x in enumerate(live) if re.search(r"│\s+1\s+8\s+16", x))
    head = next(i for i, x in enumerate(live) if x.startswith("● Step"))
    top = next(i for i, x in enumerate(done) if "╭─ Trained" in x)
    assert done[top + axis - head].rstrip(" │") == live[axis].rstrip()
    assert live[head + 1].index("loss") == done[top + 1].index("final loss")


def test_the_activity_line_holds_its_label_still():
    """No sweep across the label: frame to frame only the loader in its 3-cell slot and the
    clock change."""
    lines = [c.activity("Loading Qwen2.5-72B-Instruct", "3m 10s", t) for t in range(40)]
    texts = [plain(x) for x in lines]
    assert len({t[3:] for t in texts}) == 1
    assert all(st == "hi" for x in lines for text, st in x if "Loading" in text)
    assert not any(isinstance(st, tuple) and st[0] == "sh" for x in lines for _, st in x)
    assert not hasattr(c, "shimmer")


def test_the_loading_card_changes_only_its_clock_and_loader_frame_to_frame(room):
    room(34)
    st = _moments()[0][1]
    a, b = _rows(st, 100, t=0, animate=True), _rows(st, 100, t=7, animate=True)
    changed = {(i, k) for i, (x, y) in enumerate(zip(a, b)) for k in range(max(len(x), len(y)))
               if (x + " " * 200)[k] != (y + " " * 200)[k]}
    assert changed and all(k < 3 for _, k in changed)  # the loader's slot, nothing else


_HINT = ("to score held-out rows, give a file of them: eval_dataset, or --eval FILE; or recipe sft "
         "(--recipe sft), which holds 5% of 40 rows or more out")


@pytest.mark.parametrize("w", [100, 80])
@pytest.mark.parametrize("indent", [0, 10])  # the session's header, the plain terminal's eval row
def test_the_unscored_hint_wraps_between_words_and_keeps_them(w, indent):
    from poros.cli.train_cmd import hint_rows

    rows = [plain(r) for r in hint_rows(_HINT, min(w, c.CAP), indent)]
    assert len(rows) >= 2 and all(len(r) <= w for r in rows)
    assert all(r.startswith(c.RAIL + " " * indent) for r in rows)
    assert " ".join(r[len(c.RAIL) + indent:] for r in rows) == _HINT


@pytest.mark.parametrize("w", [100, 80])
def test_a_classic_run_on_a_text_file_keeps_its_header_and_card_still(w, room, tmp_path):
    """The engine's loop on a local text file: its header carries the hint, wrapped, printed
    before the card's first frame; under it the card's rows stand still from loading to step 2."""
    import json
    from types import SimpleNamespace

    from poros.cli.train_cmd import hint_rows, unscored_hint

    data = tmp_path / "notes.jsonl"
    data.write_text("".join(json.dumps({"text": f"word {i}"}) + "\n" for i in range(40)))
    cfg = SimpleNamespace(dataset=str(data), eval_dataset=None)
    hint = unscored_hint(cfg, SimpleNamespace(name="classic"))
    assert hint == _HINT
    head = [plain(r) for r in hint_rows(hint, min(w, c.CAP))]
    room(34)
    frames = []
    for _, st in _moments():
        st.scored, st.evals, st.scores = False, [], []  # this loop holds no rows out
        frames.append(head + _rows(st, w))
    act = next(i for i, r in enumerate(frames[0]) if "Loading" in r)
    for now in frames[1:]:
        assert len(now) == len(frames[0]) and now[:len(head)] == head
        for i, (was, row) in enumerate(zip(frames[0], now)):
            if i != act:
                assert [x for x in _labels(was) if x not in _labels(row)] == [], (i, was, row)


# train_done outside the session, as the tree before the card from the first frame prints it
# (generated from that tree; the run card's change must leave every row of it as it is)
_PLAIN_DONE = {
    'none-100': [
        '● Trained 12 steps  33s',
        '  │ final loss 1.8765',
        '  │',
        '  │ peak vram  ━━━─────────────  4.7 GiB of 22.4 GiB',
        '  │ peak ram   ━━━─────────────  11.2 GiB of 60 GiB',
        '  │ speed      410 tok/s',
        '  │',
        '  │ adapter    runs/x',
        '  │ receipt    runs/x/receipt.json',
        '',
        'Continue with: poros eval train.yaml',
    ],
    'none-80': [
        '● Trained 12 steps  33s',
        '  │ final loss 1.8765',
        '  │',
        '  │ peak vram  ━━━─────────────  4.7 GiB of 22.4 GiB',
        '  │ peak ram   ━━━─────────────  11.2 GiB of 60 GiB',
        '  │ speed      410 tok/s',
        '  │',
        '  │ adapter    runs/x',
        '  │ receipt    runs/x/receipt.json',
        '',
        'Continue with: poros eval train.yaml',
    ],
    'one-100': [
        '● Trained 1 step  33s',
        '  │ final loss 1.8765  from 2.1000  ·  ↓ 11%',
        '  │',
        '  │ peak vram  ━━━─────────────  4.7 GiB of 22.4 GiB',
        '  │ peak ram   ━━━─────────────  11.2 GiB of 60 GiB',
        '  │ speed      410 tok/s',
        '  │',
        '  │ adapter    runs/x',
        '  │ receipt    runs/x/receipt.json',
        '',
        'Continue with: poros eval train.yaml',
    ],
    'one-80': [
        '● Trained 1 step  33s',
        '  │ final loss 1.8765  from 2.1000  ·  ↓ 11%',
        '  │',
        '  │ peak vram  ━━━─────────────  4.7 GiB of 22.4 GiB',
        '  │ peak ram   ━━━─────────────  11.2 GiB of 60 GiB',
        '  │ speed      410 tok/s',
        '  │',
        '  │ adapter    runs/x',
        '  │ receipt    runs/x/receipt.json',
        '',
        'Continue with: poros eval train.yaml',
    ],
    'long-100': [
        '  ╭─ Trained ─────────────────────────────────────────────────────────────────── 60 steps · 33s ─╮',  # noqa: E501
        '  │                                                                      final loss              │',  # noqa: E501
        '  │      ▆▅▄▃▂▁                                                          1.8765  ↓ 31%           │',  # noqa: E501
        '  │  2.5 ███████▆▅▄▃▂▁                                                   from 2.7000             │',  # noqa: E501
        '  │      ██████████████▇▆▅▄▄▃▂▁                                                                  │',  # noqa: E501
        '  │      ███████████████████████▇▆▅▄▃▃▂▁                                 vram 4.7 GiB peak       │',  # noqa: E501
        '  │  2.0 ████████████████████████████████▇▆▆▅▄▃▂▂▁                       ram  11.2 GiB peak      │',  # noqa: E501
        '  │      ██████████████████████████████████████████▇▇▆▅▄▄▃▂▂▁                                    │',  # noqa: E501
        '  │      ██████████████████████████████████████████████████████▇▆▆▅▄▄▃▂                          │',  # noqa: E501
        '  │  1.5 ██████████████████████████████████████████████████████████████                          │',  # noqa: E501
        '  │      ██████████████████████████████████████████████████████████████                          │',  # noqa: E501
        '  │      1                             30                            60                          │',  # noqa: E501
        '  ╰──────────────────────────────────────────────────────────────────────────────────────────────╯',  # noqa: E501
        '',
        '  │ speed      410 tok/s',
        '  │',
        '  │ adapter    runs/x',
        '  │ receipt    runs/x/receipt.json',
        '',
        'Continue with: poros eval train.yaml',
    ],
    'long-80': [
        '  ╭─ Trained ─────────────────────────────────────────────── 60 steps · 33s ─╮',
        '  │                                                  final loss              │',
        '  │      ▆▅▃▂                                        1.8765  ↓ 31%           │',
        '  │  2.5 ████▇▆▄▃▂                                   from 2.7000             │',
        '  │      ██████████▇▅▄▂▁                                                     │',
        '  │      ████████████████▆▅▃▃▁                       vram 4.7 GiB peak       │',
        '  │  2.0 ██████████████████████▇▅▅▃▂▁                ram  11.2 GiB peak      │',
        '  │      █████████████████████████████▇▆▅▃▃▁                                 │',
        '  │      █████████████████████████████████████▆▆▄▄▂                          │',
        '  │  1.5 ██████████████████████████████████████████                          │',
        '  │      ██████████████████████████████████████████                          │',
        '  │      1                   30                  60                          │',
        '  ╰──────────────────────────────────────────────────────────────────────────╯',
        '',
        '  │ speed      410 tok/s',
        '  │',
        '  │ adapter    runs/x',
        '  │ receipt    runs/x/receipt.json',
        '',
        'Continue with: poros eval train.yaml',
    ],
}


@pytest.mark.parametrize("case", sorted(_PLAIN_DONE))
def test_outside_the_session_the_finished_summary_is_as_it_was(case, room, monkeypatch):
    """One-shot, piped, NO_COLOR, TERM=dumb: a short run (no curve point at one row a step), a
    one-step run and a long one print the summary they printed before, row for row."""
    room(38)
    monkeypatch.setattr(c, "in_session", False)
    name, w = case.rsplit("-", 1)
    long = [round(2.4 * 0.99 ** i + 0.3, 6) for i in range(60)]
    losses, steps = {"none": ([], 12), "one": ([2.1], 1), "long": (long, 60)}[name]
    r = sc.TrainResult(steps=steps, wall_s=33.0, final_loss=1.8765,
                       first_loss=losses[0] if losses else None, losses=losses,
                       peak_vram_gb=5.0, card_gb=24.0, peak_ram_gb=12.0, ram_total_gb=64.0,
                       output_dir="runs/x", receipt="runs/x/receipt.json",
                       next_cmd="poros eval train.yaml", tokens_per_sec=410.0)
    assert [plain(x) for x in sc.train_done(r, int(w))] == _PLAIN_DONE[case]


def test_a_reported_metric_prints_as_reported_and_only_a_missing_one_waits():
    reported = sc.metric_rows({"accuracy": float("nan"), "margin": float("nan")}, 11, 16)
    waiting = sc.metric_rows({"accuracy": None, "margin": None}, 11, 16)
    # as the tree before the card's placeholders prints a reported NaN
    assert [plain(x) for x in reported] == [
        "  │ acc        ━━━━━━━━━━━━━━━━  100%  chosen ranked first", "  │ margin     nan"]
    assert plain(waiting[0]).rstrip().endswith("–") and plain(waiting[1]).rstrip().endswith("–")


class _Live:
    """A live area that counts what is asked of it."""

    live = False

    def __init__(self, surface, **_kw):
        self.surface, self.started, self.closed, self.printed = surface, 0, 0, []

    def start(self):
        self.started += 1

    def close(self, final=None):
        self.closed += 1

    def print(self, lines):
        self.printed += [plain(x) for x in lines]

    def update(self):
        pass


class _Term:
    def __init__(self):
        from types import SimpleNamespace

        self.lives, self.printed = [], []
        self.tab = SimpleNamespace(run="", label=None)  # a session tab's name and label

    def live(self, surface, **kw):
        self.lives.append(_Live(surface, **kw))
        return self.lives[-1]

    def print(self, lines):
        self.printed += [plain(x) for x in lines]

    def progress(self, fn):
        self.tab.label = fn

    def width(self):
        return 100


@pytest.fixture
def session_term(monkeypatch):
    from poros.cli.ui import term

    t = _Term()
    term.bind(t)
    monkeypatch.setattr(c, "in_session", True)
    yield t
    term.bind(None)


def _load_args(tmp_path):
    from types import SimpleNamespace

    from poros.cli import train_cmd

    (tmp_path / "notes.jsonl").write_text("".join('{"text": "w"}\n' for _ in range(40)))
    cfg = SimpleNamespace(model_name_or_path="Qwen/Qwen2.5-7B-Instruct", output_dir=str(tmp_path),
                          gradient_accumulation_steps=1, max_steps=6, eval_dataset=None,
                          dataset=str(tmp_path / "notes.jsonl"), eval_split=0.05, eval_rows=0,
                          quantization="nf4", lora_rank=16, lora_alpha=32, lora_dropout=0.0,
                          lora_target_modules=None, block_size=4)
    choice = SimpleNamespace(name="classic", label="classic", shape="text")
    ready = train_cmd._Ready(choice, "qwen2", None, True, False, "RTX 4090", 24.0, "")
    return cfg, ready


@pytest.mark.parametrize("fault", [None, KeyboardInterrupt, RuntimeError, SystemExit])
def test_in_the_session_the_card_is_up_before_the_load_and_comes_off_once(fault, session_term,
                                                                          tmp_path, monkeypatch):
    import typer

    from poros.cli import train_cmd
    from poros.cli.ui import simulate

    cfg, ready = _load_args(tmp_path)
    seen = []

    def load(model, **kw):
        live = session_term.lives[-1]
        seen.append((live.started, live.closed, live.surface(0, 100)))
        if fault is not None:
            raise fault("stop")
        return simulate.SimTrainer(kw["config"])

    monkeypatch.setattr(simulate.SimTrainer, "from_pretrained", staticmethod(load))
    errors = []
    kw = dict(shown_run="", resume_from=None, friendly_error=errors.append, plan_gate=None)
    if fault is KeyboardInterrupt:
        with pytest.raises(typer.Exit):
            train_cmd._load(cfg, ready, **kw)
        assert "○ Stopped while loading" in session_term.printed
    elif fault is SystemExit:
        with pytest.raises(SystemExit):
            train_cmd._load(cfg, ready, **kw)
    else:
        got = train_cmd._load(cfg, ready, **kw)
        assert (got is None) == (fault is not None) and len(errors) == (fault is not None)
    (started, closed, rows), = seen
    assert started == 1 and closed == 0
    assert any("Loading Qwen2.5-7B-Instruct" in plain(x) for x in rows)
    assert any(plain(x).startswith("● Step – of 6") for x in rows)
    live = session_term.lives[-1]
    assert live.closed == (0 if fault is None else 1) and len(session_term.lives) == 1
    # a load that ends the command leaves the tab named as it was, with no progress on it
    assert (session_term.tab.label is None) == (fault is not None)
    if fault is None:
        assert got[4].st.label == "Training" and got[4].live is live


@pytest.mark.parametrize("fault", ["exit", KeyboardInterrupt])
def test_an_exit_before_the_first_step_takes_the_card_off_and_the_notice_goes_over_it(
        fault, session_term, tmp_path, monkeypatch):
    import typer

    from poros.cli import train_cmd
    from poros.cli.ui.train_view import TrainView

    cfg, ready = _load_args(tmp_path)
    st = sc.TrainState(model="m", card_gb=24.0, total=6, label="Training")
    view = TrainView(st, live=session_term.live)
    view.start()
    trainer = type("T", (), {"optimizer": type("O", (), {"notice": "the slow path"})()})()

    def refuse(*a, **k):
        raise typer.Exit(2) if fault == "exit" else KeyboardInterrupt()

    monkeypatch.setattr(train_cmd, "_write_resolved", refuse)
    with pytest.raises((typer.Exit, KeyboardInterrupt)):
        train_cmd._train_and_save(cfg, ready, trainer, st, None, run_id="r", resume_from=None,
                                  should_stop=None, job_id=None, resume_cmd=None, next_cmd=None,
                                  config_path="c.yaml", emit_json=False, friendly_error=print,
                                  w=100, view=view)
    live = session_term.lives[-1]
    assert live.closed == 1
    assert any("optimizer: reference" in x for x in live.printed)  # over the card, not under it
    assert not any("optimizer" in x for x in session_term.printed)


def test_the_notice_lands_above_the_card_inline_and_in_a_tab():
    """Inline, a line printed through the live area is written before the card is drawn again
    in the same write; in a tab it joins the tab's lines above the card."""
    import io

    from poros.cli.ui import tui
    from poros.cli.ui.live import Live
    from poros.cli.ui.style import resolve
    from poros.cli.ui.train_view import TrainView

    st = sc.TrainState(model="m", card_gb=24.0, total=6, label="Training")
    note = [[("○ optimizer: reference  the slow path", "dm")]]
    out = io.StringIO()
    inline = TrainView(st, live=lambda fn, **kw: Live(fn, stream=out, **kw))
    inline.live.live = True
    inline.print(note)
    written = out.getvalue()
    assert written.index("optimizer: reference") < written.index("Training")
    screen = tui.Screen(io.StringIO(), resolve(io.StringIO()))
    tab = tui.Tab("train")
    tabbed = TrainView(st, live=lambda fn, **kw: tui.TabLive(screen, tab, fn))
    tabbed.start()
    tabbed.print(note)
    assert plain(tab.lines[-1]).startswith("○ optimizer: reference") and tab.surface is not None


def test_a_resumed_run_says_so_above_its_card():
    from poros.cli.ui.train_view import TrainView

    st = sc.TrainState(model="m", card_gb=24.0, label="Training")
    seen = []
    view = TrainView(st, live=lambda fn, **kw: seen.append(_Live(fn)) or seen[-1])
    view.on_start(object(), 10, 4)
    assert seen[-1].printed == ["● Resumed after step 4/10"]



def test_host_ram_peak_counts_from_the_end_of_the_load(monkeypatch):
    """The card samples host RAM while the model loads; the peak a run records counts only from
    the load's end, as when the view started after it."""
    from poros.cli.ui import train_view

    reading = [50.0]
    monkeypatch.setattr(train_view, "host_ram_gb", lambda: reading[0])
    monkeypatch.setattr(train_view.sys, "platform", "linux")
    st = sc.TrainState(model="m", card_gb=24.0, total=6, label="Loading m")
    view = train_view.TrainView(st, live=_Live)
    view._surface(0, 100)
    assert view.peak_ram == 50.0  # the load's own, while it shows
    view.loaded()
    assert view.peak_ram is None and st.elapsed == 0.0 and st.label == "Training"
    reading[0] = 3.0
    view._surface(1, 100)
    assert view.peak_ram == 3.0


@pytest.mark.parametrize("losses,step", [([2.0], 0), ([], 1)])
def test_a_frame_between_a_step_and_its_loss_draws_sanely(losses, step, room):
    room(34)
    st = sc.TrainState(model="m", card_gb=24.0, total=16, step=step, losses=list(losses),
                       label="Training", ram_gb=0.1)
    rows = _rows(st, 100)
    head = next(x for x in rows if x.startswith("● Step"))
    assert head.startswith("● Step 1 of 16")
    dots = sum(x[3:].count("●") for x in rows if x.startswith("  │"))
    assert dots == len(losses)


def _painter():
    import io

    from poros.cli.ui import curve
    from poros.cli.ui.style import resolve

    return curve.Painter(resolve(io.StringIO()), io.StringIO())


def test_one_point_is_the_cells_dot_even_where_the_curve_is_an_image(room, monkeypatch):
    """The image painter (kitty's graphics: kitty, Ghostty, Konsole) draws a line from its
    second point, live and finished; one point is the cell-drawn dot."""
    from poros.cli.ui.curve import Series

    room(34)
    st = _moments()[2][1]
    st.painter = _painter()
    rows = sc.train_surface(st, 0, 100)
    assert not any(seg == "raw" for r in rows for _, seg in r)
    assert sum(plain(x)[3:].count("●") for x in rows if plain(x).startswith("  │")) == 1
    st2 = _moments()[3][1]
    st2.painter = _painter()
    assert any(seg == "raw" for r in sc.train_surface(st2, 0, 100) for _, seg in r)
    monkeypatch.setattr(c, "in_session", True)
    r = _done([], steps=1, final_loss=2.83)
    r.series, r.painter = Series(1).extend([2.83]), _painter()
    done = sc.train_done(r, 100)
    assert not any(seg == "raw" for x in done for _, seg in x)
    assert sum(plain(x)[3:].count("●") for x in done if plain(x).startswith("  │")) == 1


def test_a_lone_dot_stays_inside_the_chart():
    from poros.cli.ui import curve

    s = curve.Series(16).extend([9.0])
    scale = curve.fit_scale([1.0, 2.0], 10)  # a scale that does not reach the point
    cells = curve.block_rows(s, scale, 40, 0)
    assert sum(ch == "●" for row in cells for ch, _ in row) == 1


def test_memory_a_step_measured_as_zero_is_a_number():
    from poros.cli.ui.curve import Series

    s = Series(4).extend([2.0, 1.9])
    k = rc.Card(series=s, last=1, memory=True, measured=True, pct=50)
    assert any(plain(x).rstrip().endswith("vram 0.0 GiB") for x in rc.card(k, 100))
    k = rc.Card(series=Series(4), last=-1, memory=True, measured=False)
    assert any(plain(x).rstrip().endswith("vram –") for x in rc.card(k, 100))


def test_a_pair_runs_rows_stay_when_only_one_of_its_numbers_has_come(room):
    room(34)
    st = _moments(pairs=True)[2][1]
    st.extra = {"accuracy": 0.5}
    rows = _rows(st, 100)
    assert any(x.startswith("  │ acc ") for x in rows)
    assert any(x.rstrip() == "  │ margin     –" for x in rows)



def test_a_config_without_a_step_count_reads_dashes(session_term, tmp_path, room):
    from poros.cli import train_cmd

    room(34)
    cfg, ready = _load_args(tmp_path)
    cfg.max_steps = None
    st = sc.TrainState(model=cfg.model_name_or_path, card_gb=24.0)
    view = train_cmd._card_from_the_load(cfg, ready, st, "Loading m", None)
    assert st.total == 0
    assert any(plain(x).startswith("● Step – of –") for x in view.live.surface(0, 100))


@pytest.mark.parametrize("w", [100, 80])
def test_a_long_runs_checkpoint_table_never_narrows_the_curve(w, room):
    """At 1,000 steps the table's step column is a cell wider: the room is held from the first
    frame, so the axis and the readout stay where they were when the score comes."""
    room(34)
    st = sc.TrainState(model="m", card_gb=24.0, total=2000, label="Loading m", scored=True,
                       ram_gb=0.1)
    before = _rows(st, w)
    losses = [2.0 - i * 0.0005 for i in range(1000)]
    later = sc.TrainState(**{**vars(st), "label": "Training", "step": 1000, "losses": losses,
                             "series": None, "evals": [(0, 2.9), (1000, 1.26)],
                             "scores": [rc.Score(0, 2.9, 12, 62, "ends"),
                                        rc.Score(1000, 1.263, 58, 62, "ends")]})
    after = _rows(later, w)
    axis = [i for i, x in enumerate(before) if x.rstrip().endswith("2,000")]
    assert axis and [i for i, x in enumerate(after) if x.rstrip().endswith("2,000")] == axis
    end = before[axis[0]].rstrip()
    assert after[axis[0]].rstrip().endswith("2,000") and end.endswith("2,000")
    assert len(after[axis[0]].rstrip()) == len(end)
    at = next(i for i, x in enumerate(before) if x.rstrip().endswith("loss"))
    assert after[at].rstrip().endswith("loss")
    assert after[at].index("loss") == before[at].index("loss")


@pytest.mark.parametrize("w", [59, 70])
def test_where_the_card_does_not_fit_the_load_is_its_one_line(w, room):
    """Under the card's width the load shows the activity line alone, as before the card: no
    plain rows of zeros for numbers no step has measured."""
    room(34)
    st = _moments()[0][1]
    rows = [x for x in _rows(st, w) if x.strip() and not x.startswith("─")]
    assert rows[0].startswith("─") or "Loading Qwen2.5-7B-Instruct" in rows[0]
    assert not any("0.0 GiB" in x or "0/16" in x or "Step" in x for x in rows)


class _Behind(list):
    """Losses the trainer's thread has appended to after the frame took its reading."""

    def __iter__(self):
        return iter([])


def test_a_frame_draws_from_its_own_reading_of_the_losses(room):
    room(34)
    st = sc.TrainState(model="m", card_gb=24.0, total=16, label="Training", ram_gb=0.1)
    st.losses = _Behind([2.0])
    rows = _rows(st, 100)
    text = "\n".join(rows)
    loss = next(i for i, r in enumerate(rows) if r.rstrip().endswith("loss"))
    assert rows[loss + 1].rstrip().endswith("–") and "from –" in text
    assert "● Step – of 16" in text and "Step – of 0" not in text
    assert sum(r[3:].count("●") for r in rows if r.startswith("  │")) == 0


def test_the_load_card_is_handed_back_unstarted(session_term, tmp_path):
    from poros.cli import train_cmd

    cfg, ready = _load_args(tmp_path)
    st = sc.TrainState(model=cfg.model_name_or_path, card_gb=24.0)
    view = train_cmd._card_from_the_load(cfg, ready, st, "Loading m", None)
    assert view.live.started == 0 and view.live.closed == 0


def test_an_exit_anywhere_before_training_takes_the_card_off(session_term, tmp_path):
    from poros.cli import train_cmd
    from poros.cli.ui.train_view import TrainView

    cfg, ready = _load_args(tmp_path)
    st = sc.TrainState(model="m", card_gb=24.0, total=6, label="Training")
    view = TrainView(st, live=session_term.live)
    view.start()

    class Trainer:
        @property
        def optimizer(self):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        train_cmd._train_and_save(cfg, ready, Trainer(), st, None, run_id="r", resume_from=None,
                                  should_stop=None, job_id=None, resume_cmd=None, next_cmd=None,
                                  config_path="c.yaml", emit_json=False, friendly_error=print,
                                  w=100, view=view)
    assert session_term.lives[-1].closed == 1 and session_term.tab.label is None


def test_the_tab_gets_its_name_back_even_when_the_card_fails_to_close(session_term):
    from poros.cli import train_cmd

    class Broken:
        def close(self):
            raise RuntimeError("draw")

    session_term.tab.label = lambda: "train 0%"
    with pytest.raises(RuntimeError):
        train_cmd._card_off(Broken())
    assert session_term.tab.label is None


def test_the_hint_rows_take_no_hint_and_any_width():
    from poros.cli.train_cmd import hint_rows

    assert hint_rows("", 100) == []
    assert hint_rows("to score held-out rows", 2, indent=10)  # one word a row, never an error


def test_a_run_that_ends_before_any_frame_still_finishes_into_the_card(monkeypatch, room):
    """The redraw thread may not have drawn once before a one-step run ends: the card is chosen
    from the run, not from whether a frame happened."""
    from poros.cli.ui.train_view import TrainView
    from poros.train.reporting import StepReport

    room(34)
    monkeypatch.setattr(c, "in_session", True)
    finals = []

    class Kept(_Live):
        def close(self, final=None):
            finals.append(final)

    st = sc.TrainState(model="m", card_gb=24.0, total=1, label="Training")
    view = TrainView(st, live=Kept)
    view.on_step(StepReport(step=1, total=1, loss=2.83, step_time=1.0, tokens=10,
                            tokens_per_sec=10.0, lr=1e-4, vram_gb=5.0, vram_peak_gb=5.5))
    assert st.series is None  # nothing has drawn
    view.finish(_done([], steps=1, final_loss=2.83), 100)
    text = "\n".join(plain(x) for x in finals[-1])
    assert "╭─ Trained" in text and "2.8300" in text
