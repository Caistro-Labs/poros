"""/recipes: whole training setups, as /models lists models. A recipe is the exact settings a model
needs -- most often a large one on one card, or a model on the Mac path -- with who wrote it, what
it is for, and its peak against this machine. Yours first, then this project's, then the shared
ones (a GitHub repo, synced on ⌃R), then Poros's own.

The card is /models' list card with a calmer foot: the tabs underlined, the chosen one in the
accent, each row led by its lab's mark; under the rows, the focused recipe in two lines on the
rows' own edge (what it does, then its settings), and the keys on that edge too.
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional, Tuple

from poros.cli.context import Context
from poros.cli.ui import components as c
from poros.cli.ui import sheet as sh
from poros.cli.ui.text import Line, S, banded, ln, width

# the tabs, in the order the strip shows them: what "recipe" most often means first
TABS = (("big", "Big models"), ("mac", "Mac"), ("start", "Start"), ("support", "Support"),
        ("writing", "Writing"), ("code", "Code"), ("extract", "Extract"), ("prefer", "Prefer"))
LOOP = {"sft": "SFT", "dpo": "DPO", "ipo": "IPO", "kto": "KTO", "orpo": "ORPO", "simpo": "SimPO"}
# a file's shape (poros.cli.verbs.data) as the data a recipe names: what earns the ✓
KIND = {"chats": "chats", "convertible": "qa", "pairs": "pairs", "text": "text"}
EDGE = "    "  # where every line under the title starts: the rail's text
TITLE, BY, MODEL, METER = 30, 18, 17, 6


def _model_short(model: str) -> str:
    from poros.cli.words.models import by_id

    e = by_id(model)
    return e.short if e else model.rsplit("/", 1)[-1]  # its own name: an Instruct is not its base


def _who(r) -> str:
    return {"yours": "you", "poros": r.by}.get(r.source) or (
        "this project" if r.source.startswith("recipes") else r.by)


def plan_of(r, ctx: Context):
    """The planner's plan for recipe ``r`` on this machine (never shown: the recipes print only
    certified figures), at the recipe's own settings (its
    length, rank, dropout, store, path, batch and block); None for a model outside the catalog:
    a file's model is anyone's repo, and only the catalog's are planned."""
    from poros.cli.verbs import shared
    from poros.cli.words.models import by_id

    e = by_id(r.model)
    if e is None:
        return None
    st = r.settings
    said = {k: st[k] for k in ("per_device_batch_size", "gradient_accumulation_steps",
                               "block_size", "offload_embeddings_and_head",
                               "moe_expert_cache_gib") if k in st}
    if st.get("path"):
        said.update(shared.path_config(st["path"]))
    store = st.get("store", "ram")
    if store == "ssd":  # where train keeps the store unless told (--store ssd)
        from poros.cli.verbs.train import sss_dir

        where = str(sss_dir())
    kw = dict(seq_len=int(st.get("max_seq_length", 512)), rank=int(st.get("lora_rank", 16)),
              dropout=float(st.get("lora_dropout", 0.0)), model_type=e.arch, store=store,
              store_path=where if store == "ssd" else None)
    p = shared.plan_for(r.model, ctx, path_settings=said or None, **kw)
    if st.get("path"):
        return p
    # a path left to the run: the one its start takes (train's own auto_path), so the list and
    # show read the plan the train card reads
    return shared.auto_path(p, lambda path: shared.plan_for(
        r.model, ctx, path_settings={**said, "path": path}, **kw))


def certified_here(r, ctx: Context):
    """(the certified run that speaks for this machine, ``certified`` or ``same_memory_class``)
    (poros.recipes.certified.match), or None: not one of Poros's, not certified, or this machine
    has less memory than every machine it ran on."""
    from poros.recipes import certified

    if r.source != "poros":
        return None
    from poros.cli.verbs.recipes import machine_check

    hit, _, planned = machine_check(r, ctx, certified.entries(r.key), certified.here(ctx))
    return None if hit is None else (*hit, planned)


def _when(d: Optional[date], today: Optional[date] = None) -> str:
    if d is None:
        return ""
    days = ((today or date.today()) - d).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    if days < 60:
        return f"{days // 7} weeks ago"
    return f"{days // 30} months ago"


def _fit(r, ctx: Context) -> Tuple[Line, str]:
    """Its memory on this machine: for Poros's own, what its certified run used (on its own
    machine, or another with at least that memory), or that none is certified here; for
    another's, what its author reported (said as a report), and where it ran. No planner figure.
    The cells, and the word."""
    gpu = ctx.gpu
    card = gpu.total_gib if gpu else None
    hit = certified_here(r, ctx)
    if hit is not None and hit[1] != "certified":
        # certified on another machine or class: where is the facts' line; no figure here, as
        # that run was not run on this one (the owner's rule: a figure only on its own class)
        return c.chips(None, METER) + [("certified".rjust(13), "rl"),
                                       (" elsewhere".ljust(10), "rl")], "elsewhere"
    if hit is not None and card:
        e = hit[0]
        used = e.memory_peak_gib if e.machine.platform == "mac" else e.fit_gib()
        if used:
            peak = c.ceil1(used)
            return c.chips(peak / card, METER) + [(f"{peak:.1f} GiB".rjust(11) + "  ", "dm"),
                                                  ("certified".ljust(10), "a")], "certified"
    if r.source == "poros":  # Poros's own: what a certified run measured, and nothing else
        return c.chips(None, METER) + [("not certified".rjust(13), "rl"),
                                       (" here".ljust(10), "rl")], "unknown"
    rep = r.measured.get("peak_gib")
    machine = str(r.measured.get("machine", ""))
    if rep and card:
        if machine.startswith("apple-") != bool(ctx.unified):
            where = "on a Mac" if machine.startswith("apple-") else "on NVIDIA"
            return c.chips(None, METER) + [(f"{rep:.1f} GiB".rjust(11) + "  ", "dm"),
                                           (where.ljust(10), "rl")], "elsewhere"
        return c.chips(rep / card, METER, "dm") + [
            (f"{rep:.1f} GiB".rjust(11) + "  ", "dm"), ("reported".ljust(10), "rl")], "reported"
    return c.chips(None, METER) + [("  not tested ", "rl"), ("yet".ljust(10), "rl")], "unknown"


def _certified_facts(r, ctx: Context) -> List[str]:
    """``certified by Caistro Labs on RTX 4090 · Apple M4 ...``, and whether this machine is at
    least one of them; a built-in whose runs are not in yet says so."""
    from poros.recipes import certified

    if r.source != "poros":
        return []
    said = certified.status(r.key)
    if said == "pending":
        return ["certification pending"]
    if said != "certified":
        return []
    out = [f"certified by {certified.CERTIFIER} on {' and '.join(certified.cards(r.key))}"]
    hit = certified_here(r, ctx)
    if hit is not None and hit[1] == "certified":
        out.append("certified on this machine")
    elif hit is not None:  # another machine, or another Mac class: where, and no more
        out.append(f"certified on the {hit[0].machine.gpu}; not run on this machine")
    return out


def listing(ctx: Context) -> sh.Listing:
    from poros.cli.sheets.fit import _gpu_name
    from poros.cli.sheets.listings import _cell
    from poros.cli.ui import logos
    from poros.cli.verbs import mac
    from poros.cli.verbs.data import shape_of
    from poros.cli.verbs.train_data import local_files
    from poros.cli.words.models import by_id
    from poros.recipes import hub, presets

    image = logos.images()
    files = [(f, shape_of(ctx.cwd / f)) for f in local_files(ctx.cwd)[:20]]
    kept, refused = presets.local(ctx.cwd)
    shared_, said = hub.cached()
    recipes = [r for r in kept + shared_ + presets.BUILT_IN if mac.trains_here(r, ctx)]
    items, start, generic = [], None, False
    for n, r in enumerate(recipes):
        mine = next((f for f, shape in files if r.takes(shape)), None)
        made = next((f for f, shape in files if KIND.get(shape) == r.data), None)
        if mine and (start is None or (generic and r.data != "any")):
            start, generic = n, r.data == "any"  # one made for your data beats "any data"
        e = by_id(r.model)
        meter, _ = _fit(r, ctx)
        facts = [(r.model, "dm")] + [(" · " + x, "dm") for x in (
            LOOP.get(r.loop, r.loop),
            *r.summary().split(" · "),  # each setting its own: the narrow card drops whole ones
            f"made for {made}" if made else f"takes {mine}" if mine else
            f"needs {presets.DATA[r.data]}",
            "numbers reported, not checked by Poros" if r.measured else "",
            "no data yet: --data synthetic tries it" if r.source == "poros" and r.data == "any"
            else "",
            *_certified_facts(r, ctx),
            f"updated {_when(r.updated)}" if r.updated else "",
        ) if x]
        if r.unvalidated:
            facts = [("not validated by Poros", "e"), (" · ", "dm")] + facts
        about = r.about or ("saved from one of your runs" if r.source == "yours" else
                            "from this project's recipes folder" if _who(r) == "this project"
                            else f"{LOOP.get(r.loop, r.loop)} on {presets.DATA[r.data]}")
        mark = ([("✓ ", "a")] if made else [("! ", "e")] if r.unvalidated else [("  ", None)])
        lab = logos.lab(r.model)[1] if image else ""
        lead = (logos.cells(lab, "tx", image=True, slot=2000 + n) if lab else [("  ", None)]
                ) + [(" ", None)] if image else []
        items.append(sh.Item(
            group=r.tab(),
            cells=[*lead, *mark,
                   _cell(r.name, TITLE, "hi"),
                   _cell(_who(r), BY, "dm"),
                   _cell(_model_short(r.model), MODEL, "tx"),
                   *meter],
            text="  ".join([r.name, r.by, r.about, r.model, r.data, r.loop, r.tab()]),
            actions=[("enter", "use it", r.argv(mine, model=e.short if e else "")),
                     ("ctrl-r", "sync", ["recipes", "sync"])],
            detail=facts,
            panel=[[(about[:1].upper() + about[1:], "tx")], facts],
        ))
    groups = [(key, [(chip, None)]) for key, chip in TABS if any(i.group == key for i in items)]
    gpu = ctx.gpu
    head = [("   " * bool(image) + "  ", "rl"), ("recipe".ljust(TITLE), "rl"),
            ("by".ljust(BY), "rl"), ("model".ljust(MODEL), "rl"),
            (f"peak of {gpu.total_gib:.0f} GiB" if ctx.unified and gpu else
             f"peak on {_gpu_name(gpu)} · {gpu.total_gib:.1f} GiB" if gpu
             else "peak memory", "rl")]
    if refused:
        aside = f"{len(refused)} file{'s' * (len(refused) != 1)} refused: poros recipes"
    else:
        try:
            where = hub.shown(said.get("repo") or hub.repo())
        except ValueError:
            where = "POROS_RECIPES_REPO is not a repo"
        when = ("demo" if said.get("demo") else
                f"synced {hub.ago(said)}" if hub.ago(said) else "not synced yet")
        aside = f"{where} · {when}"
    return sh.Listing(tag="recipes", title="Recipes", aside=aside, head=head, items=items,
                      groups=groups, window=12, start=start or 0,
                      empty="no recipes: ~/.poros/recipes/*.toml holds yours")


class Lister(sh.Lister):
    """/models' list card, its foot two lines on the rows' edge and its tabs underlined."""

    def render(self, w: int) -> List[Line]:
        from poros.cli.ui.sheet.frame import _fit
        from poros.cli.ui.sheet.lister import _dim, _Narrow, whole

        L = self.listing
        items = self.shown()
        n = min(L.window, max(len(L.items), 1))
        w = min(w, c.CAP)
        aside = L.aside if w - 4 - len(L.title) - len(L.aside) >= 2 else L.aside.split(" · ")[-1]
        gap = w - 2 - len(L.title) - len(aside)
        title = c.titled(ln(S("● ", "tx"), S(L.title, "hi"),
                            S(" " * gap + aside, "rl") if gap > 1 else None))
        page = items[self.top:self.top + n]
        where = (f"{self.top + 1}–{self.top + len(page)} of {len(items)}"
                 if len(items) > len(page) else "")
        search = ([S("search  ", "dm"), S(self.buf, "hi"), S("▏", "a")] if self.buf
                  else [S("type to search", "rl")])
        room = w - len(c.RAIL) - width(search)
        out = [title, c.rail(), *self._strip(w), c.rail(*search, S(where.rjust(room), "rl"))]
        over = max((width(i.cells) for i in L.items), default=0) - (w - len(c.RAIL))
        fit = _Narrow(L.items, over, L.head)  # one set of widths: the columns stay in line
        if L.items:
            out.append(c.rail(*whole(fit.head(L.head), w - len(c.RAIL))))
        for i, item in enumerate(page):
            on = self.top + i == self.sel
            gutter = S("  ❯ ", "a") if on else S(c.RAIL, "rl")
            cells = fit.row(item.cells)
            cells = c.light(cells) if on else [(t, _dim(st)) for t, st in cells]
            row = whole(ln(gutter, *cells), w)
            out.append(banded(row, w) if on else row)
        if not items:
            out.append(c.rail(S(f"nothing matches {self.buf!r}" if self.buf else L.empty, "dm")))
            page = [None]
        out += [c.rail()] * (n - len(page))  # the card keeps its height as typing narrows it
        sel = items[min(self.sel, len(items) - 1)] if items else None
        about, facts = (list(sel.panel) + [[], []])[:2] if sel is not None else ([], [])
        room = w - len(c.RAIL) - 2
        out += [c.rail(),
                c.rail(S("  "), *[(_fit(t, room), st) for t, st in about[:1]]),
                c.rail(S("  "), *whole(list(facts), room)),  # whole facts, dropped from the right
                c.rail()]
        keys = [("⏎" if k == "enter" else c.key_name(k), h)
                for k, h, _ in (sel.actions if sel is not None else [])]
        keys += [("←→", "use"), ("esc", "clear") if self.buf else ("esc", "close")]
        _, segs = c.keyline(keys, w - len(EDGE) + 1)
        out.append([(EDGE, None), *segs[1:]])
        return [c.inked(line) for line in out]

    def _strip(self, w: int) -> List[Line]:
        """The tabs and, under them, a rule: the chosen one's stretch of it in the accent. The
        strip scrolls so the chosen tab always shows, ‹ where some went left, › right."""
        L = self.listing
        if not L.groups:
            return []
        chips = [[("All", None)]] + [list(chip) for _, chip in L.groups]
        counts = [len(L.items)] + [sum(1 for i in L.items if i.group == k) for k, _ in L.groups]
        bodies = [[(t, "hi" if n == self.group else "dm") for t, _ in chip]
                  + [S(f" {count}", "a" if n == self.group else "rl")]
                  for n, (chip, count) in enumerate(zip(chips, counts))]
        room = w - len(c.RAIL)
        start = 0
        more = 2 * (self.group < len(bodies) - 1)  # room for the › after it
        while start < self.group and (
                sum(width(b) + 4 for b in bodies[start:self.group + 1]) + 2 + more > room):
            start += 1
        row: Line = [S(c.RAIL, "rl")] + ([S("‹ ", "rl")] if start else [])
        under: Line = [("──", "tk")] if start else []
        left = room - 2 * bool(start)
        for n, body in enumerate(bodies[start:]):
            last = start + n == len(bodies) - 1
            if width(body) + 4 * bool(n) + 2 * (not last) > left:
                row.append(S(" ›", "rl"))
                break
            row += ([S("    ")] if n else []) + body
            under += ([("─" * 4, "tk")] if n else []) + [
                ("━" * width(body), "a") if start + n == self.group else ("─" * width(body), "tk")]
            left -= width(body) + 4 * bool(n)
        rest = room - sum(len(t) for t, _ in under)
        return [row, [S(c.RAIL, "rl"), *under, ("─" * max(0, rest), "tk")]]
