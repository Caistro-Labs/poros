"""/settings: how the session looks and what it remembers, on one card.

Each row is a preference the commands already read (questions.preference),
so nothing here is a second source of truth. Each change is kept as it is
made (clearing the /train defaults waits for the card to close); esc closes.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from poros.cli import questions, theme
from poros.cli.ui import sheet as sh
from poros.cli.ui.sheet import Field, Form, Option, Value, View
from poros.cli.ui.text import S

MOTION = "motion"
TRAIN_WHERE = "train_where"


def apply_saved() -> None:
    """The saved theme and motion, for this process. The environment wins:
    POROS_THEME, and POROS_NO_ANIM (what loaders and live views read)."""
    mode = questions.preference("theme")
    if mode and not os.environ.get("POROS_THEME"):
        theme.set_mode(mode)
    if questions.preference(MOTION) == "off":
        os.environ["POROS_NO_ANIM"] = "1"


def _pick(key: str, label: str, value: Value, opts, hint: str) -> Field:
    hit = next((o for o in opts if o.value == value), None)
    return Field(key, label, value=value, shown=hit.label if hit else str(value or ""),
                 options=lambda: list(opts), hint=hint, quiet=True)


def _short(path) -> str:
    home = str(Path.home())
    return "~" + str(path)[len(home):] if str(path).startswith(home) else str(path)


def settings_form(live=None) -> Form:
    from poros.cli.context import home
    from poros.cli.sheets import saved_defaults
    from poros.cli.verbs.mac import run_files

    def build(values: Dict[str, Value]) -> View:
        from poros.cli.ui.components import key_name

        n = len(saved_defaults())
        from poros.cli.verbs.shared import hf_cache_shown

        store = questions.setting("store")  # a Mac's, when it has one
        cache = hf_cache_shown(store)
        fields = [
            _pick("train_where", "train", values.get("train_where"), [
                Option("here", "watch here", "in this terminal"),
                Option("background", "in the background", "come back with /runs"),
            ], "where /train runs unless you say"),
            *([_pick("defaults", "defaults", values.get("defaults"), [
                Option("keep", f"{n} saved setting{'s' if n != 1 else ''}"),
                Option("clear", "clear them", "back to Poros' own defaults"),
            ], "settings every /train starts from")] if n else []),  # nothing to choose: a fact
            _pick("export", "export", values.get("export"), [
                Option("ask", "ask each time"),
                Option("merged", "a standalone model", "ready to serve · bf16"),
                Option("adapter", "the adapter alone", "loads on its base · PEFT, vLLM"),
            ], "what /export makes when you don't say"),
            _pick("motion", "motion", values.get("motion"), [
                Option("on", "loaders move"),
                Option("off", "loaders still", "also when your OS asks for less motion"),
            ], "loaders and the live training view"),
            _pick("theme", "theme", values.get("theme"), [
                Option("auto", "auto", "follow the terminal"),
                Option("dark", "dark"), Option("light", "light"),
            ], "the palette for every screen"),
        ]
        return View(
            title=[S("Settings", "hi"), S("  what every command opens with", "dm")],
            fields=fields,
            facts=[*([] if n else [("defaults", "none saved",
                                    f"{key_name('ctrl-s')} on /train saves yours")]),
                   ("runs", _short(run_files(store) or home() / "runs"),
                    "checkpoints, receipts, reports"),
                   ("models", _short(cache), "shared with other Hugging Face tools"),
                   ("downloads", "asks first", "weights only · never runs repo code")],
            go="Save settings",
            command=_short(questions._prefs_path()),
        )

    values0: Dict[str, Value] = {
        "theme": theme.current_mode(),
        "motion": questions.preference(MOTION) or "on",
        "export": questions.preference("export") or "ask",
        "train_where": questions.preference(TRAIN_WHERE) or "here",
        "defaults": "keep",
    }
    return Form("settings", build, values0, browse=True, live=live)


def save(values: Dict[str, Value], final: bool = True) -> None:
    from poros.cli.sheets import SAVED

    questions.remember("theme", theme.set_mode(str(values.get("theme") or "auto")))
    questions.remember(MOTION, str(values.get("motion") or "on"))
    export = str(values.get("export") or "ask")
    questions.remember("export", "" if export == "ask" else export)
    questions.remember(TRAIN_WHERE, str(values.get("train_where") or "here"))
    if final and values.get("defaults") == "clear":  # not while cycling past it
        questions.remember(SAVED, "")
    if values.get("motion") == "off":
        os.environ["POROS_NO_ANIM"] = "1"
    else:
        os.environ.pop("POROS_NO_ANIM", None)


def open_settings(session=None) -> Optional[Dict[str, Value]]:
    from poros.cli.ui import components as c
    from poros.cli.ui import term

    def kept(values: Dict[str, Value]) -> None:
        save(values, final=False)
        if session is not None and hasattr(session, "_retheme"):
            session._retheme()  # a new theme shows at once, the card in it

    values = sh.run(settings_form(live=kept))
    if values is None:
        term.get().print([c.event("Cancelled", None, "nothing ran", "○", "dm")])
        return None
    save(values)
    items = [f"{k} {values.get(k)}" for k in ("theme", "motion", "export") if values.get(k)]
    # one line, whole words: what does not fit is dropped from the right
    while items and len("● Saved " + " · ".join(items)) > term.get().width():
        items.pop()
    term.get().print([c.event("Saved", " · ".join(items) or None)])
    return values
