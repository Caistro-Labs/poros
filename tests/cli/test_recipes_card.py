"""/recipes as /models lists models: big-model and Mac setups first, the shared repo synced only on
the user's word, and its words for people kept to the card."""

from __future__ import annotations

import gzip
import io
import json
import os
import sys
import tarfile
from datetime import date

import pytest

from poros.cli import recipes_card, sheets
from poros.cli.context import Context
from poros.cli.ui.style import resolve
from poros.cli.ui.text import render_line
from poros.recipes import demo, hub, presets, rules

SHARED = '''title = "gpt-oss-120b on one 5090"
about = "OpenAI's 120B MoE, streamed two blocks at a time"
by = "kestrel-dev"
use = "big"
updated = 2026-11-08
model = "openai/gpt-oss-120b"
data = "chats"
[settings]
max_seq_length = 2048
block_size = 2
lora_rank = 16
[measured]
machine = "RTX 5090"
peak_gib = 29.6
'''


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("POROS_DEV_SIMULATE", raising=False)
    monkeypatch.delenv("POROS_RECIPES_REPO", raising=False)
    (tmp_path / "home").mkdir()
    return tmp_path


# -- the file ---------------------------------------------------------------------------------


def test_words_for_people_come_only_from_the_shared_repo():
    fields = rules.parse(SHARED, shared=True)
    assert fields["title"] == "gpt-oss-120b on one 5090" and fields["updated"] == date(2026, 11, 8)
    again = rules.parse(rules.dumps(fields, shared=True), shared=True)
    assert again == fields  # a date writes back as one
    with pytest.raises(ValueError, match="field no recipe"):
        rules.parse(SHARED)  # a file in a folder here holds no one's words
    assert rules.parse('model = "a/b"\nuse = "mac"\n')["use"] == "mac"  # a use is a word of ours


@pytest.mark.parametrize("line", [
    'title = "red\\u001b[31m"', 'title = "a\\u202eevil"', 'title = "zero\\u200bwidth"',
    'title = "line\\nbreak"', 'title = " leading"', f'title = "{"x" * 49}"',
    f'about = "{"x" * 121}"', 'by = "-dash"', 'by = "a--b"', 'by = "you are"',
    'use = "everything"', "updated = 2026-11-08T10:00:00", "updated = 2019-01-01",
])
def test_a_shared_files_words_cannot_steer_a_terminal(line):
    with pytest.raises(ValueError) as err:
        rules.parse(f'model = "a/b"\n{line}\n', shared=True)
    for word in ("\x1b", "evil", "​", "dash", "everything"):
        assert word not in str(err.value)


def test_the_demo_set_reads_through_the_parser_and_leads_with_big_models_and_macs():
    recipes = [presets.from_fields(k, rules.parse(t, shared=True), "shared", "")
               for k, t in demo.FILES]
    tabs = [r.tab() for r in recipes]
    assert tabs.count("big") >= 4 and tabs.count("mac") >= 3
    assert any((presets.params(r.model) or 0) >= 100 for r in recipes)
    assert all(str(r.measured.get("machine", "")).startswith("apple-")
               for r in recipes if r.tab() == "mac" and r.measured)


def test_a_recipe_naming_no_use_takes_its_tab_from_what_it_is():
    def tab(model, **kw):
        return presets.Recipe("k", "n", "you", "", model, **kw).tab()

    assert tab("Qwen/Qwen3-235B-A22B-Instruct-2507") == "big"
    m4 = {"machine": "apple-m4-max-40gpu-128gb"}
    assert tab("Qwen/Qwen2.5-7B-Instruct", measured=m4) == "mac"
    assert tab("Qwen/Qwen2.5-7B-Instruct", data="pairs") == "prefer"
    assert tab("Qwen/Qwen3-0.6B") == "start"
    assert presets.params("Qwen/Qwen3-30B-A3B") == 30
    assert presets.params("openai/gpt-oss-120b") == 120
    assert presets.params("mistralai/Mistral-Large-Instruct-2411") is None


# -- syncing ----------------------------------------------------------------------------------


def _tarball(members) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, body, kind in members:
            info = tarfile.TarInfo(f"Caistro-Labs-poros-recipes-abc123/{name}")
            if kind == "link":
                info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
                tar.addfile(info)
                continue
            data = body.encode()
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_sync_keeps_what_passes_and_writes_nothing_where_the_tarball_points(home, monkeypatch):
    evil = SHARED.replace("gpt-oss-120b on one 5090", "x\\u001b]0;pwned\\u0007")
    members = [
        ("recipes/big/gpt-oss.toml", SHARED, "file"),
        ("recipes/big/evil.toml", evil, "file"),
        ("recipes/../../escape.toml", SHARED, "file"),
        ("/abs.toml", SHARED, "file"),
        ("recipes/a/b/c/deep.toml", SHARED, "file"),
        ("recipes/linked.toml", "", "link"),
        ("recipes/huge.toml", SHARED + "#" * rules.CAP_BYTES, "file"),
        ("recipes/first-run.toml", SHARED, "file"),  # Poros's own name
        ("README.md", "# hi", "file"),
    ]
    monkeypatch.setattr(hub, "_download", lambda name: _tarball(members))
    said = hub.sync()
    assert said["count"] == 1 and said["repo"] == hub.REPO
    refused = dict(said["refused"])
    assert set(refused) == {"recipes/big/evil.toml", "recipes/huge.toml", "recipes/first-run.toml"}
    assert "pwned" not in json.dumps(said)
    kept, meta = hub.cached()
    assert [r.name for r in kept] == ["gpt-oss-120b on one 5090"] and kept[0].by == "kestrel-dev"
    assert kept[0].source == "shared" and meta["count"] == 1
    assert not (home / "escape.toml").exists() and not (home / "home" / "escape.toml").exists()
    assert sorted(os.listdir(hub.folder())) == ["gpt-oss.toml"]
    assert hub.ago(meta) == "just now"


def test_a_failed_sync_leaves_the_last_copy(home, monkeypatch):
    one = _tarball([("recipes/a.toml", SHARED, "file")])
    monkeypatch.setattr(hub, "_download", lambda name: one)
    hub.sync()

    def offline(name):
        raise ValueError("GitHub could not be reached")

    monkeypatch.setattr(hub, "_download", offline)
    with pytest.raises(ValueError, match="reached"):
        hub.sync()
    assert [r.key for r in hub.cached()[0]] == ["a"]


def test_a_repo_can_be_a_folder_and_its_links_are_not_followed(home, monkeypatch):
    repo = home / "repo"
    (repo / "recipes" / "mac").mkdir(parents=True)
    (repo / "recipes" / "mac" / "m4.toml").write_text(SHARED.replace('use = "big"', 'use = "mac"'))
    outside = home / "outside"
    outside.mkdir()
    (outside / "x.toml").write_text(SHARED)
    (repo / "recipes" / "linked").symlink_to(outside)
    monkeypatch.setenv("POROS_RECIPES_REPO", str(repo))
    assert hub.sync()["count"] == 1
    assert [r.tab() for r in hub.cached()[0]] == ["mac"]


def test_the_repo_named_must_be_one(monkeypatch):
    monkeypatch.setenv("POROS_RECIPES_REPO", "owner/name; rm -rf ~")
    with pytest.raises(ValueError):
        hub.repo()
    monkeypatch.setenv("POROS_RECIPES_REPO", "someone/their-recipes")
    assert hub.shown(hub.repo()) == "github.com/someone/their-recipes"


# -- the card ---------------------------------------------------------------------------------


def test_listing_never_syncs(home, monkeypatch):
    def nowhere(*a, **k):
        raise AssertionError("it reached for the network")

    monkeypatch.setattr(hub, "_download", nowhere)
    monkeypatch.setattr("socket.socket.connect", nowhere)
    L = sheets.recipes_listing(Context(cwd=home, fixed_interactive=True))
    assert "not synced yet" in L.aside and "github.com/Caistro-Labs/poros-recipes" in L.aside
    assert all(("ctrl-r", "sync", ["recipes", "sync"]) in i.actions for i in L.items)


# Each platform's card: its tabs in order, and Poros's built-ins under each (the
# demo's shared recipes fill the rest). On a Mac each built-in is its CUDA counterpart's Mac twin,
# under that counterpart's tab; mac-7b, which had none, is under Mac with the Mac-measured ones.
CARD = {
    False: (["big", "mac", "start", "support", "writing", "code", "extract", "prefer"],
            {"big": {"big-32b", "big-72b", "big-72b-fast", "moe-30b", "moe-30b-ssd",
                     "moe-30b-8gb"},
             "start": {"first-run"}, "support": {"support"}, "writing": {"voice"},
             "prefer": {"prefer"}}),
    True: (["big", "mac", "start", "support", "writing", "code", "prefer"],
           {"big": {"mac-72b", "mac-32b", "mac-moe"}, "mac": {"mac-7b"},
            "start": {"mac-first-run"}, "support": {"mac-support"}, "writing": {"mac-voice"},
            "prefer": {"mac-prefer"}}),
}


@pytest.mark.parametrize("mac", [False, True])
def test_the_card_leads_with_big_models_and_the_mac(home, monkeypatch, mac):
    """On either platform Big models leads and each built-in is under its purpose's tab: on a
    Mac its Mac twin, under the CUDA one's tab. The Mac tab keeps what a Mac measured,
    and every recipe is in some tab."""
    from poros.metal import table

    monkeypatch.setenv("POROS_DEV_SIMULATE", "1")
    if mac:
        ctx = Context(cwd=home, fixed_gpu=table.plan_gpu("apple-m4-10gpu-16gb"),
                      fixed_interactive=True)
    else:
        monkeypatch.setattr("sys.platform", "linux")  # no card here, and not a Mac
        ctx = Context(cwd=home, fixed_interactive=True)
    L = recipes_card.listing(ctx)
    assert L.aside.endswith("· demo")
    keys = [k for k, _ in L.groups]
    tabs, built_ins = CARD[mac]
    assert keys == tabs and keys == [k for k, _ in recipes_card.TABS if k in keys]
    by_name = {r.name: r.key for r in presets.DEFINED}
    shown = {k: {by_name[i.text.split("  ")[0]] for i in L.items
                 if i.group == k and i.text.split("  ")[0] in by_name} for k in keys}
    assert {k: v for k, v in shown.items() if v} == built_ins
    assert all(i.group in keys for i in L.items)  # nothing in no tab
    mac_measured = [i for i in L.items if i.group == "mac" and i.text.split("  ")[0] not in by_name]
    assert mac_measured  # the Mac tab still holds the recipes measured on a Mac
    lister = recipes_card.Lister(L)
    lister.key("right")  # the first tab
    assert {i.group for i in lister.shown()} == {"big"}
    rows = [render_line(r, resolve(depth="plain")) for r in lister.render(100)]
    strip, rule = rows[2], rows[3]
    assert "Big models" in strip and "━" in rule  # the chosen tab underlined
    assert rule.index("━") == strip.index("Big models")
    assert any("sync" in r and "esc close" in r for r in rows[-1:])
    assert all(len(r) <= 100 for r in rows)


def test_a_tab_is_display_only_and_no_file_sets_one(home):
    """A Mac built-in's tab changes nothing it trains: its command and settings are the same
    without it. It is the built-ins' own: a recipe file's words cannot set it."""
    import dataclasses

    for r in presets.DEFINED:
        plain = dataclasses.replace(r, shelf="")
        assert r.argv("d.jsonl") == plain.argv("d.jsonl") and r.settings == plain.settings
        assert (r.shelf != "") is (r.use == "mac"), r.key
    with pytest.raises(ValueError):
        rules.parse('model = "Qwen/Qwen2.5-7B"\nshelf = "big"\n')


def test_the_card_keeps_its_height_and_its_foot_stays_whole_when_narrow(home, monkeypatch):
    monkeypatch.setenv("POROS_DEV_SIMULATE", "1")
    lister = recipes_card.Lister(recipes_card.listing(Context(cwd=home, fixed_interactive=True)))
    heights = set()
    for _ in range(len(lister.listing.groups) + 1):
        for w in (100, 84, 72):
            rows = [render_line(r, resolve(depth="plain")) for r in lister.render(w)]
            heights.add((w, len(rows)))
            assert all(len(r) <= w for r in rows)
            assert not any(r.rstrip().endswith((" ·", "·")) for r in rows)  # no dangling fact
        lister.key("right")
    assert len({h for _, h in heights}) == 1


def test_an_agent_gets_no_shared_recipes_words(home, monkeypatch):
    from poros.cli.verbs.recipes import _row

    r = presets.from_fields("gpt-oss", rules.parse(SHARED, shared=True), "shared", "")
    row = _row(r)
    assert row.name == r.title() and "5090" not in row.name
    assert "streamed" not in row.model_dump_json()


@pytest.mark.parametrize(("named", "error", "said"), [
    (None, 404, "github.com/Caistro-Labs/poros-recipes is not published yet"),
    ("someone/recipes", 404, "there is no public repo github.com/someone/recipes"),
    (None, None, "GitHub could not be reached for github.com/Caistro-Labs/poros-recipes"),
])
def test_a_sync_that_cannot_fetch_names_the_repo_and_how_to_name_another(home, monkeypatch,
                                                                         named, error, said):
    """A sync of a shared-recipes repo that is not on GitHub names the repo and
    POROS_RECIPES_REPO, not only "no such repo". No request leaves this test."""
    import io
    import urllib.error
    import urllib.request

    if named:
        monkeypatch.setenv("POROS_RECIPES_REPO", named)
    else:
        monkeypatch.delenv("POROS_RECIPES_REPO", raising=False)
    monkeypatch.delenv("POROS_DEV_SIMULATE", raising=False)

    def refuse(req, timeout=None):
        if error is None:
            raise urllib.error.URLError("no route")
        raise urllib.error.HTTPError(req.full_url, error, "Not Found", {}, io.BytesIO())

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(ValueError) as failed:
        hub.sync()
    assert said in str(failed.value) and "POROS_RECIPES_REPO" in str(failed.value)


@pytest.mark.parametrize("body", ["cut", "not gzip", "incomplete read"])
def test_a_broken_answer_is_a_clean_refusal_and_leaves_nothing_half_written(home, monkeypatch,
                                                                           body):
    """A tarball cut short, a body that is not gzip and a dropped connection (IncompleteRead)
    each end the sync as a ValueError in one line, never a tarfile or zlib traceback: the copy
    synced last stays, and no .new folder is left."""
    import http.client
    import urllib.request

    real = hub._download
    one = _tarball([("recipes/a.toml", SHARED, "file")])
    monkeypatch.setattr(hub, "_download", lambda name: one)
    hub.sync()
    if body == "incomplete read":
        def dropped(req, timeout=None):
            raise http.client.IncompleteRead(b"partial", 100)

        monkeypatch.setattr(hub, "_download", real)
        monkeypatch.setattr(urllib.request, "urlopen", dropped)
    else:
        broken = one[: len(one) // 2] if body == "cut" else b"<html>rate limited</html>"
        monkeypatch.setattr(hub, "_download", lambda name: broken)
    with pytest.raises(ValueError) as failed:
        hub.sync()
    assert "\n" not in str(failed.value)
    assert not hub.folder().with_name(hub.folder().name + ".new").exists()
    assert [r.key for r in hub.cached()[0]] == ["a"]


def _bomb(unpacked: int) -> bytes:
    """A small gzip tarball whose one pax header unpacks to ``unpacked`` bytes of zeros."""
    hdr = tarfile.TarInfo("top/PaxHeader")
    hdr.type, hdr.size = tarfile.XHDTYPE, unpacked
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=9) as gz:
        gz.write(hdr.tobuf(format=tarfile.USTAR_FORMAT))
        zeros = bytes(1 << 20)
        for _ in range(unpacked >> 20):
            gz.write(zeros)
    return buf.getvalue()


def test_a_bomb_is_refused_before_the_tar_parser_reads_past_the_cap(home, monkeypatch):
    """A tarball that unpacks past MAX_READ (here an extended header, read whole by tarfile) is
    refused having unpacked no more than the cap, and the copy synced last stays."""
    import resource

    monkeypatch.setattr(hub, "MAX_READ", 4 << 20)
    one = _tarball([("recipes/a.toml", SHARED, "file")])
    monkeypatch.setattr(hub, "_download", lambda name: one)
    hub.sync()
    bomb = _bomb(256 << 20)
    assert len(bomb) < hub.MAX_BYTES
    monkeypatch.setattr(hub, "_download", lambda name: bomb)
    unit = 1 if sys.platform == "darwin" else 1024  # ru_maxrss: bytes on macOS, KiB on Linux
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit
    with pytest.raises(ValueError, match="unpacks to over 4 MB"):
        hub.sync()
    grew = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * unit - before
    assert grew < 64 << 20  # bytes: 256 MiB if the parser had read the header
    assert [r.key for r in hub.cached()[0]] == ["a"]
    assert not hub.folder().with_name(hub.folder().name + ".new").exists()


def test_owner_name_is_always_github_even_with_a_folder_of_that_name_here(home, monkeypatch):
    """A folder in the current one named like the default repo is not the repo: owner/name goes
    to GitHub, and only a path (/, ./, ../, ~) is a folder, shown as that folder."""
    look_alike = home / "Caistro-Labs" / "poros-recipes" / "recipes"
    look_alike.mkdir(parents=True)
    (look_alike / "planted.toml").write_text(SHARED)
    monkeypatch.chdir(home)
    fetched = []
    one = _tarball([("recipes/a.toml", SHARED, "file")])
    monkeypatch.setattr(hub, "_download", lambda name: fetched.append(name) or one)
    said = hub.sync()
    assert fetched == ["Caistro-Labs/poros-recipes"] and said["count"] == 1
    assert [r.key for r in hub.cached()[0]] == ["a"]
    monkeypatch.setenv("POROS_RECIPES_REPO", "Caistro-Labs/poros-recipes")
    assert hub.shown(hub.repo()) == "github.com/Caistro-Labs/poros-recipes"
    monkeypatch.setenv("POROS_RECIPES_REPO", "./Caistro-Labs/poros-recipes")
    assert hub.shown(hub.repo()) == "./Caistro-Labs/poros-recipes"  # the folder, not GitHub
    assert hub.sync()["count"] == 1 and fetched == ["Caistro-Labs/poros-recipes"]
    assert [r.key for r in hub.cached()[0]] == ["planted"]
    monkeypatch.setenv("POROS_RECIPES_REPO", "./nowhere")
    with pytest.raises(ValueError, match="names no folder"):
        hub.repo()
