"""The start box's turning entrance (intro_turn): frames, hand-off, fallback."""

from poros.cli.ui import intro, intro_turn, tui

WORDS = ("Fine-tune models bigger than your GPU.", "What are we training today?")


def plain(line):
    return "".join(s for s, st in line if st != "raw")


def test_the_turn_hands_over_to_the_settled_box_exactly():
    turn = intro_turn.Turn((10, 20), (255, 107, 53))
    right = intro.box_column(*WORDS, intro.box_room(100))
    draw = intro_turn.frames(100, right, turn)
    settled = draw(draw.duration + 1.0)
    assert [plain(r) for r in settled] == [plain(r) for r in intro.final(100, *WORDS)]
    assert not any(st == "raw" for r in settled for _, st in r)  # the image is gone


def test_while_turning_the_image_rides_on_the_marks_top_line_only():
    turn = intro_turn.Turn((10, 20), (255, 107, 53))
    right = intro.box_column(*WORDS, intro.box_room(100))
    draw = intro_turn.frames(100, right, turn)
    lines = draw(0.4)
    raws = [i for i, r in enumerate(lines) if any(st == "raw" for _, st in r)]
    assert raws == [intro.box_mark_row(right)]
    raw = lines[raws[0]][0][0]
    assert f"c={intro_turn.COLS}" in raw and f"r={intro_turn.ROWS}" in raw and "z=1" in raw
    assert raw.startswith("\x1b7") and raw.endswith("\x1b8")  # the cursor comes back
    # the same id every frame: the screen replaces it in place
    assert tui.image_key(raw) == tui.image_key(draw(0.6)[raws[0]][0][0])
    # every frame has as many lines as the settled box
    assert {len(draw(t)) for t in (0.0, 0.3, 0.9, 1.0, 2.0)} == {len(intro.final(100, *WORDS))}


def test_the_turn_ends_facing_you_full_size_and_in_the_accent():
    turn = intro_turn.Turn((10, 20), (255, 107, 53))
    px = turn.rgba(intro_turn.DUR - 1e-6)
    W = turn.W
    x, y = 2 * 10 + 5, 2 * 20 + 10  # the middle of the mark's top-left cell
    i = (y * W + x) * 4
    assert px[i + 3] == 255 and abs(px[i] - 255) <= 2 and abs(px[i + 1] - 107) <= 3
    assert px[(5 * W + 5) * 4 + 3] == 0  # outside the mark: clear


def test_in_p3_the_accent_is_converted():
    assert intro_turn.to_p3((255, 107, 53)) == (237, 116, 69)
    assert intro_turn.to_p3((255, 255, 255)) == (255, 255, 255)


def test_off_the_full_screen_the_river_plays(monkeypatch):
    monkeypatch.setattr(tui, "current", None)
    assert intro_turn.maybe(100, *WORDS) is None
    assert not hasattr(intro.frames(100, *WORDS), "duration")
