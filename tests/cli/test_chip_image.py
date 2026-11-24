"""A Mac's chip on /machine: every chip drawn, its tones fitted to any
terminal ground, the scan entrance, and the view that lays it out."""

import pytest

from poros.cli.ui import chip_image, chip_plans
from poros.cli.ui import components as c
from poros.cli.ui.style import Palette


def _pal(bg, fg):
    from poros.cli import theme

    return Palette(depth="truecolor", dark=theme.luminance(bg) <= 0.18, ink=fg, ground=bg)


DARK = _pal((20, 20, 20), (230, 230, 230))


def test_every_chip_in_the_lineup_has_a_plan_and_no_more_cores_than_it_ships():
    assert set(chip_plans.CHIPS) == set(chip_plans.PLANS)
    for name in chip_plans.CHIPS:
        assert 0 < chip_plans.gpu_max(name) <= chip_plans.CHIPS[name][1][0]
        assert chip_plans.gpu_max(name) <= chip_plans.gpu_tiles(name)


@pytest.mark.parametrize("gpu, want", [
    ("Apple M4 10-core GPU 16GB", ("M4", 10)),
    ("Apple M4 Pro 20-core GPU 48GB", ("M4 Pro", 20)),
    ("Apple M3 Ultra 80-core GPU 512GB", ("M3 Ultra", 80)),
    ("Apple M1 (8-core GPU)", ("M1", None)),
    ("RTX 5090", None),
    ("Apple M9 10-core GPU 16GB", None),
])
def test_the_chip_is_read_off_the_gpu_name(gpu, want):
    assert chip_image.parse(gpu) == want


def test_the_name_row_says_the_chip_as_apple_does():
    assert chip_image.subtitle("M4", 10, 16.0) == "10-core GPU · 16 GB LPDDR5X · 120 GB/s"
    assert chip_image.subtitle("M4 Max", 32, None) == "32-core GPU · LPDDR5X · 410 GB/s"  # binned
    assert chip_image.subtitle("M2 Pro", None, 32.0).startswith("19-core GPU")  # one tile dark


def test_tones_fit_any_ground(monkeypatch):
    monkeypatch.setattr(chip_image.intro_turn, "images_in_p3", lambda: False)
    plain = chip_image.tones(DARK)
    soft = chip_image.tones(_pal((0, 43, 54), (131, 148, 150)))  # Solarized: low contrast
    from poros.cli.ui import gpu_image

    assert soft["line"] != gpu_image._tones(_pal((0, 43, 54), (131, 148, 150)))["line"]
    assert plain["accent"][0] == (255, 107, 53)
    red = _pal((122, 37, 30), (215, 201, 167))  # Red Sands: the accent's own hue
    assert chip_image.tones(red)["accent"][0] == (215, 201, 167)
    assert chip_image.tones(_pal((34, 79, 188), (255, 255, 255)))["accent"][0] == (255, 107, 53)


def test_where_images_are_p3_every_tone_is_converted(monkeypatch):
    monkeypatch.setattr(chip_image.intro_turn, "images_in_p3", lambda: True)
    t = chip_image.tones(_pal((0, 43, 54), (131, 148, 150)))
    monkeypatch.setattr(chip_image.intro_turn, "images_in_p3", lambda: False)
    s = chip_image.tones(_pal((0, 43, 54), (131, 148, 150)))
    assert t["face"][0] == chip_image.intro_turn.to_p3(s["face"][0]) != s["face"][0]


def test_the_cache_keeps_the_drawing(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    chip_image.picture.cache_clear()
    pic = chip_image.picture("M4", 10, DARK, 160)
    assert list(tmp_path.rglob("*.rgba"))
    chip_image.picture.cache_clear()
    again = chip_image.picture("M4", 10, DARK, 160)
    assert again.px == pic.px and (again.w, again.h) == (pic.w, pic.h)


def _kitty(seg):
    """Each kitty command in ``seg``: its keys, and how much data it carried."""
    import re

    out = []
    for head, data in re.findall(r"\x1b_G([^;\x1b]*);?([^\x1b]*)\x1b\\", seg):
        keys = dict(kv.split("=", 1) for kv in head.split(",") if "=" in kv)
        out.append((keys, len(data)))
    return out


def test_the_scan_sends_the_chip_once_then_only_moves_crops_of_it(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    pic = chip_image.picture("M4", 10, DARK, 160)
    ids, cell, cols = (7, 8, 9), (4, 8), 40
    rows = chip_image.size("M4", cols, cell)[1]
    first = _kitty(chip_image.entrance(pic, cols, rows, cell, ids, 0.0, send=True))
    sent = {k["i"] for k, _ in first if k.get("a") == "t"}
    assert sent == {"8", "9"}  # the chip and its wash, kept and not shown
    crop = []
    for t in (0.1, 0.3, 0.5):
        cmds = _kitty(chip_image.entrance(pic, cols, rows, cell, ids, t))
        assert all(n == 0 for _, n in cmds)  # a frame carries no pixels
        assert not any(k.get("a") in ("t", "T") for k, _ in cmds)
        crop.append(int(next(k["h"] for k, _ in cmds if k.get("a") == "p" and k["i"] == "8")))
    assert crop == sorted(crop) and crop[0] < crop[-1] <= pic.h  # the chip fills in as it goes
    done = _kitty(chip_image.entrance(pic, cols, rows, cell, ids, chip_image.DUR))
    whole = next(k for k, _ in done if k.get("a") == "p")
    assert (whole["i"], int(whole["h"])) == ("8", pic.h)
    assert any(k.get("a") == "d" and k.get("d") == "I" and k["i"] == "9" for k, _ in done)


def test_the_scan_moves_fast_then_settles():
    pic = chip_image.Picture(100, 200, b"", 10, 190, 5, 95, (0, 0, 0), (9, 9, 9))
    d = chip_image.DUR / 4
    steps = [pic.at(t) - pic.at(t - d) for t in (d, 2 * d, 3 * d, 4 * d)]
    assert steps == sorted(steps, reverse=True) and pic.at(0) == 10
    assert pic.at(chip_image.DUR) == 190 + pic.band  # its light just off the dies' foot


def test_the_wash_lights_the_band_and_draws_the_line():
    pic = chip_image.Picture(10, 40, b"", 2, 38, 2, 8, (200, 200, 200), (50, 60, 70))
    wash = pic.wash
    row = lambda i: wash[i * 40:(i + 1) * 40]  # noqa: E731
    line = row(pic.band)
    assert line[8:12] == bytes((50, 60, 70, 255)) and line[:8] == bytes(8)  # across the dies only
    alphas = [row(i)[11] for i in range(pic.band)]
    assert alphas == sorted(alphas) and alphas[0] == 0 and alphas[-1] < 255 * 0.22 + 1


def test_the_still_follows_the_scan_without_clearing_it_in_the_session(monkeypatch, tmp_path):
    """The session takes every image off when a row's image changes
    (tui.image_key); the scan's frames and the still must read as one."""
    from poros.cli import views
    from poros.cli.ui import gpu_image, tui

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(c, "in_session", True)
    monkeypatch.setattr(views.machine_screens, "_room", lambda: 60)
    monkeypatch.setattr(gpu_image, "_cell", lambda stream=None: (4, 8))
    monkeypatch.setenv("POROS_IMAGES", "kitty")
    r = _mac()

    def keys(lines):
        return [(i, tui.image_key(t))
                for i, line in enumerate(lines) for t, s in line if s == "raw"]

    scan = views.machine_screens.machine_chip(r, 100, 0.2, send=True)
    late = views.machine_screens.machine_chip(r, 100, 0.5)
    still = views.machine_screens.machine_chip(r, 100)
    assert keys(scan) == keys(late) == keys(still) and len(keys(still)) == 1
    raw = next(t for line in still for t, s in line if s == "raw")
    assert _kitty(raw)[0][0].get("a") == "T" and _kitty(raw)[0][0].get("z") == "3"
    # the scan's chip and light go with it: left under the still, their
    # see-through shadow would darken the still's as it lands
    ids = views.machine_screens.chip_ident(r)
    gone = {g[0].get("i") for g in _kitty(raw) if g[0].get("a") == "d" and g[0].get("d") == "I"}
    assert gone == {str(ids[1]), str(ids[2])}


def _mac(name="Apple M4 10-core GPU 16GB"):
    from poros.cli.schemas import DoctorCheck, PlanGpu
    from poros.cli.verbs.machine import MachineOutput

    checks = [DoctorCheck(name=n, value="1.0", status="ok") for n in ("Python", "torch")]
    return MachineOutput(
        verdict="ready", version="0.3.0",
        gpu=PlanGpu(name=name, total_gib=16.0, free_gib=None, source="spec", unified=True),
        host_ram_free_gib=6.0, checks=checks,
    )


def test_a_mac_shows_its_chip_where_images_show_and_rows_where_not(monkeypatch, tmp_path):
    from poros.cli import views
    from poros.cli.ui import gpu_image

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(c, "in_session", True)
    monkeypatch.setattr(views.machine_screens, "_room", lambda: 60)
    monkeypatch.setattr(gpu_image, "_cell", lambda stream=None: (4, 8))  # a small drawing
    monkeypatch.setenv("POROS_IMAGES", "kitty")
    lines = views.machine(_mac(), 100)
    text = "".join(t for line in lines for t, s in line if s != "raw")
    assert any(s == "raw" for line in lines for _, s in line)  # the chip, as an image
    assert "Apple M4" in text and "10-core GPU · 16 GB LPDDR5X · 120 GB/s" in text
    assert "Ready to train" in text and "Qwen2.5-7B" not in text  # it names no model
    monkeypatch.setenv("POROS_IMAGES", "0")
    rows = views.machine(_mac(), 100)
    assert not any(s == "raw" for line in rows for _, s in line)
    assert "Apple M4" in "".join(t for line in rows for t, _ in line)


def test_a_chip_too_tall_for_the_window_is_drawn_smaller_or_not_at_all(monkeypatch, tmp_path):
    from poros.cli import views
    from poros.cli.ui import gpu_image

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(c, "in_session", True)
    monkeypatch.setattr(gpu_image, "_cell", lambda stream=None: (4, 8))
    monkeypatch.setenv("POROS_IMAGES", "kitty")
    ultra = _mac("Apple M3 Ultra 80-core GPU 96GB")
    monkeypatch.setattr(views.machine_screens, "_room", lambda: 60)
    lines = views.machine(ultra, 100)
    assert len(lines) <= 60 and any(s == "raw" for line in lines for _, s in line)
    monkeypatch.setattr(views.machine_screens, "_room", lambda: 24)
    short = views.machine(ultra, 100)
    assert not any(s == "raw" for line in short for _, s in line)  # no room: the rows
