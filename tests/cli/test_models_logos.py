"""/models' marks: the lab logos and the two paths (Mac, CUDA), the lab
filter, and image escapes that take no columns."""

from __future__ import annotations

from poros.cli.ui import logos
from poros.cli.ui.sheet import Item, Lister, Listing, whole
from poros.cli.ui.text import width
from poros.cli.words.models import catalog, on_mac


def test_every_mark_rasterizes_inside_its_box():
    for mark in logos.names():
        cov = logos.coverage(mark, 24, 24)
        ink = sum(map(sum, cov))
        assert 0.05 * 24 * 24 < ink < 0.95 * 24 * 24, mark  # a shape, not a blank or a block


def test_png_is_a_png():
    assert logos.png("nvidia", 32, (200, 200, 200)).startswith(b"\x89PNG\r\n\x1a\n")


def test_every_catalog_lab_has_a_mark():
    for e in catalog():
        name, mark = logos.lab(e.id)
        assert name and mark in logos.names(), e.id


def test_on_mac_is_a_subset_of_the_catalog():
    assert on_mac() <= {e.id for e in catalog()}


def _listing():
    items = [Item(cells=[(n, "tx")], text=n, actions=[], group=g)
             for n, g in (("a1", "A"), ("b1", "B"), ("a2", "A"))]
    return Listing(tag="t", title="T", head=[], items=items,
                   groups=[("A", [("A", None)]), ("B", [("B", None)])])


def test_left_right_steps_through_the_labs():
    L = Lister(_listing())
    assert [i.text for i in L.shown()] == ["a1", "b1", "a2"]
    L.key("right")
    assert [i.text for i in L.shown()] == ["a1", "a2"]
    L.key("right")
    assert [i.text for i in L.shown()] == ["b1"]
    L.key("right")  # back round to All
    assert len(L.shown()) == 3
    L.key("left")
    assert [i.text for i in L.shown()] == ["b1"]


def test_the_strip_names_every_lab_with_its_count():
    text = "".join(t for t, _ in Lister(_listing())._strip(100)[0])
    assert "All 3" in text and "A 2" in text and "B 1" in text


def test_an_image_takes_no_columns():
    line = [("\x1b_Ga=T;AAAA\x1b\\", "raw"), ("  ", None), ("name", "hi")]
    assert width(line) == 6
    assert whole(line, 6)[0][1] == "raw"  # kept whole, not cut as text


def test_a_mark_is_drawn_in_the_inks_of_the_moment(monkeypatch):
    from poros.cli.ui import gpu_image
    from poros.cli.ui.style import Palette
    from poros.cli.ui.text import render_line

    monkeypatch.setattr(gpu_image, "_cell", lambda: (8, 16))
    monkeypatch.setattr(logos, "protocol", lambda stream=None: "kitty")
    line = [*logos.cells("apple", "tx", image=True, slot=3), ("Qwen", "hi")]
    assert width(line) == 6  # the placeholder takes no columns
    dark = Palette(depth="truecolor", dark=True)
    light = Palette(depth="truecolor", dark=False)
    assert logos.ink(dark, "tx") != logos.ink(light, "tx")
    a, b = render_line(line, dark), render_line(line, light)
    assert "\x1b_G" in a and a != b  # light or dark, the image follows
    ident = lambda s: s.split("i=", 1)[1].split(",", 1)[0]  # noqa: E731
    assert ident(a) == ident(b)  # one place, one id: the new ink replaces the old
    other = render_line(logos.cells("apple", "tx", image=True, slot=4), dark)
    assert ident(other) != ident(a)  # one id would leave the mark on the last row only
    assert "\x00" not in render_line(line, Palette(depth="plain", dark=True))


def test_sixel_and_iterm_marks(monkeypatch):
    from poros.cli.ui import gpu_image
    from poros.cli.ui.style import Palette
    from poros.cli.ui.text import render_line

    monkeypatch.setattr(gpu_image, "_cell", lambda: (8, 16))
    line = logos.cells("nvidia", "tx", image=True)
    pal = Palette(depth="truecolor", dark=False)
    monkeypatch.setattr(logos, "protocol", lambda stream=None: "sixel")
    assert "\x1bP" in render_line(line, pal)
    monkeypatch.setattr(logos, "protocol", lambda stream=None: "iterm")
    assert "\x1b]1337;File=" in render_line(line, pal)


def test_without_images_the_mac_is_apples_glyph():
    assert logos.cells("apple", "tx", image=False, apple_glyph=True) == [("\uf8ff ", "tx")]
    assert logos.cells("nvidia", "tx", image=False) == [("  ", None)]
