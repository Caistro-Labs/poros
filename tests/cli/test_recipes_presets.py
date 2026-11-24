"""/recipes: whole training setups, yours first, opening the train card."""

from __future__ import annotations

import sys

import pytest

from poros.cli import sheets
from poros.cli.context import Context
from poros.cli.words.grammar import parse
from poros.recipes import presets


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    return tmp_path


@pytest.mark.parametrize("key, lr", [("support", "2e-4"), ("mac-support", "2e-5")])
def test_a_recipe_opens_train_with_its_model_data_and_settings(key, lr):
    """support is the CUDA recipe (SFT's CUDA rate, on any host); mac-support, which a Mac
    asking for support is pointed at, opens what support opens on a Mac: SFT's
    Mac rate (poros.recipes.presets.SFT_LR on darwin)."""
    r = presets.DEFINED_BY_KEY[key]
    argv = r.argv("tickets.csv")
    assert argv[:4] == ["train", "Qwen/Qwen2.5-7B-Instruct", "--data", "tickets.csv"]
    assert ["--seq-len", "512"] == argv[argv.index("--seq-len"):argv.index("--seq-len") + 2]
    assert f"learning_rate={lr}" in argv and "lora_dropout=0" in argv and "--recipe" in argv
    assert argv.index("--rank") < argv.index(f"learning_rate={lr}")  # flags, then key=value
    assert r.argv(model="qwen2.5-7b-instruct")[1] == "qwen2.5-7b-instruct"
    assert r.summary().startswith(f"512 tokens · rank 16 · lr {lr} · 2 epochs")
    assert r.takes("convertible") and r.takes("chats") and not r.takes("pairs")
    if sys.platform == "darwin" and key == "mac-support":
        assert float(lr) == presets.SFT_LR  # the Mac's own SFT rate
    assert presets.twin("support", on_mac=True) == "mac-support"
    parse(argv)  # what the train card reads


def test_yours_and_the_projects_come_first(home):
    (home / "recipes").mkdir()
    (home / "recipes" / "team.toml").write_text(
        'name = "team-bot"\nmodel = "Qwen/Qwen2.5-7B-Instruct"\ndata = "qa"\n'
        "[settings]\nmax_seq_length = 1024\n")
    (home / "recipes" / "broken.toml").write_text("name = 'no model'\n")
    path = presets.save(presets.Recipe("mine", "Mine", "you", "", "Qwen/Qwen2.5-0.5B-Instruct",
                                       settings={"lora_rank": 8, "learning_rate": 1e-4}))
    every = presets.all_recipes(home)
    assert [r.name for r in every[:2]] == ["Mine", "team-bot"]  # broken.toml is skipped
    assert every[0].settings == {"lora_rank": 8, "learning_rate": 1e-4} and path.exists()
    assert every[1].by == "someone else" and every[1].source == "recipes/team.toml"


def test_the_card_starts_on_the_first_recipe_that_takes_your_data(home):
    (home / "prefs.jsonl").write_text('{"prompt": "q", "chosen": "a", "rejected": "b"}\n')
    L = sheets.listings.recipes_listing(Context(cwd=home, fixed_interactive=True))
    pick = L.items[L.start]
    assert pick.text.startswith("Better answers")  # the DPO one: pairs are all there is
    assert pick.actions[0][2][2:4] == ["--data", "prefs.jsonl"]
    assert len({len(i.panel) for i in L.items}) == 1  # the panel never changes height


def test_the_train_card_saves_its_setup_as_your_recipe(home):
    chat = '{"messages": [{"role": "user", "content": "hi"}, '
    (home / "c.jsonl").write_text(chat + '{"role": "assistant", "content": "yo"}]}\n')
    note = sheets.save_recipe({"subject": "Qwen2.5-7B-Instruct", "--data": str(home / "c.jsonl"),
                               "--rank": "32", "learning_rate=": "1e-4"})
    assert "saved as your recipe" in note
    r = presets.all_recipes(home)[0]
    assert (r.model, r.data, r.loop, r.by) == ("Qwen/Qwen2.5-7B-Instruct", "chats", "sft", "you")
    assert r.settings == {"lora_rank": 32, "learning_rate": 1e-4}


def test_no_built_in_for_chats_names_a_base_model_that_cannot_end_its_answers():
    """A base model with an Instruct twin never learned its chat end token, and a frozen head
    cannot learn it (the trainer says so mid-run): every built-in on chats names a model that
    ends its answers."""
    from poros.cli.words.models import base_on_chats
    from poros.recipes import presets

    for r in presets.DEFINED:
        if r.data in ("chats", "qa"):
            assert not base_on_chats(r.model), (r.key, r.model)
