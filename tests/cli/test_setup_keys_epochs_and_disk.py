"""A setup's key under --recipe, a recipe's own epochs, a full disk, and the start and fit
screens' store words."""

from __future__ import annotations

import errno
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

from poros.cli.context import Context
from poros.core.refusal import Refusal
from poros.recipes import presets
from tests.cli.test_train_runs import home  # noqa: F401 - the simulated CUDA card's fixture

M4 = None  # the Mac's card, read lazily: poros.metal.table is torch-free but not needed above


def _mac_gpu():
    from poros.metal import table

    return table.plan_gpu("apple-m4-10gpu-16gb")


def _ctx(tmp_path, mac: bool) -> Context:
    from poros.core.schemas import PlanGpu

    gpu = _mac_gpu() if mac else PlanGpu(name="NVIDIA GeForce RTX 4090", total_gib=23.99,
                                         free_gib=23.5, source="nvidia-smi")
    return Context(cwd=tmp_path, fixed_gpu=gpu, fixed_cached=frozenset(),
                   fixed_interactive=False)


# ── a setup's key is told its own command at once ──────────────────────────────────────────


@pytest.mark.parametrize("mac", [False, True], ids=["cuda", "mac"])
@pytest.mark.parametrize("key", sorted(presets.DEFINED_BY_KEY))
@pytest.mark.parametrize("with_model", [False, True], ids=["no-model", "a-model"])
def test_a_setups_key_is_refused_at_once_with_its_own_command(tmp_path, key, mac, with_model):
    """No "Which model?" whose every answer is refused: the setup's own train command, the
    user's --data in it (a key of the other platform: its twin here, where it has one)."""
    from poros.cli.verbs import train as tv
    from poros.cli.words.grammar import parse

    told = presets.pointer(key, mac)
    setup = presets.DEFINED_BY_KEY[told[1] if told else key]
    words = ["train", *(["Qwen/Qwen2.5-7B"] if with_model else []), "--recipe", key,
             "--data", "my rows.jsonl"]
    with pytest.raises(Refusal) as found:
        tv.resolve(parse(words), _ctx(tmp_path, mac))
    assert found.value.title == "A setup, not a loop" and found.value.code == "E_USAGE"
    assert found.value.fix == shlex.join(["poros", *setup.argv("my rows.jsonl")])


def test_a_setups_key_without_data_names_where_the_data_goes(tmp_path):
    from poros.cli.verbs import train as tv
    from poros.cli.words.grammar import parse

    with pytest.raises(Refusal) as found:
        tv.resolve(parse(["train", "--recipe", "voice"]), _ctx(tmp_path, False))
    assert found.value.fix == shlex.join(["poros", *presets.DEFINED_BY_KEY["voice"].argv(
        "<data>")])


# ── the steps a recipe's printed command trains are its own epochs ─────────────────────────


def _rows(path, shape: str, n: int = 60) -> str:
    rows = ([{"prompt": f"q{i}", "chosen": "yes " * 9, "rejected": "no " * 9} for i in range(n)]
            if shape == "pairs" else
            [{"messages": [{"role": "user", "content": f"q{i}"},
                           {"role": "assistant", "content": "an answer " * 9}]}
             for i in range(n)])
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def _start(words, ctx):
    from poros.cli.verbs import train as tv
    from poros.cli.words.grammar import parse

    return tv.inspect(tv.resolve(parse(words), ctx).action, ctx)


@pytest.mark.parametrize("key", sorted(presets.DEFINED_BY_KEY))
def test_a_recipes_printed_command_trains_its_own_epochs(home, key):  # noqa: F811
    r = presets.DEFINED_BY_KEY[key]
    data = _rows(home / f"{key}.jsonl", "pairs" if r.data == "pairs" else "chats")
    from poros.core.errors import PorosConfigError

    try:
        a = _start(r.argv(data), _ctx(home, False))
    except PorosConfigError as refused:
        # a Mac MoE recipe's command says the Mac's block size, which the MoE path refuses on
        # a GPU before the card (the test below); no other recipe's command is refused here
        assert key.startswith("mac-") and "block_size" in str(refused), (key, refused)
        return
    if "max_steps" in r.settings:
        assert a.steps == r.settings["max_steps"]
        return
    want = r.settings.get("num_epochs") or (1 if r.loop == "dpo" else 2)
    assert a.steps == a.per_epoch * want, (a.steps, a.per_epoch, want)


def test_a_mac_moe_recipes_command_is_refused_on_a_gpu_before_the_card(home):  # noqa: F811
    """mac-moe's printed command names the Mac's block size; on a GPU the MoE path does not
    take one, and the refusal comes before the card, naming the setting."""
    from poros.core.errors import PorosConfigError

    r = presets.DEFINED_BY_KEY["mac-moe"]
    data = _rows(home / "mac-moe.jsonl", "chats")
    with pytest.raises(PorosConfigError, match="block_size"):
        _start(r.argv(data), _ctx(home, False))


def test_steps_and_epochs_given_still_win_over_the_recipes(home):  # noqa: F811
    r = presets.DEFINED_BY_KEY["voice"]
    data = _rows(home / "voice.jsonl", "chats")
    ctx = _ctx(home, False)
    a = _start(r.argv(data) + ["--epochs", "1"], ctx)
    assert a.steps == a.per_epoch
    assert _start(r.argv(data) + ["--steps", "5"], ctx).steps == 5


def test_a_run_that_says_no_epochs_keeps_the_loops(home):  # noqa: F811
    data = _rows(home / "plain.jsonl", "chats")
    a = _start(["train", "Qwen/Qwen2.5-7B-Instruct", "--data", data, "--recipe", "sft"],
               _ctx(home, False))
    assert a.steps == a.per_epoch * 2 and a.provenance["steps"].detail.startswith("2 epochs")


# ── a full disk is the disk's refusal, never a bug ─────────────────────────────────────────


def _hub_xet_error() -> RuntimeError:
    """hf_xet's words for a full disk, raised from huggingface_hub's own download code."""
    code = compile("raise RuntimeError('Task error: File reconstruction error: IO Error: No "
                   "space left on device (os error 28)')",
                   "/venv/site-packages/huggingface_hub/file_download.py", "exec")
    try:
        exec(code, {})
    except RuntimeError as exc:
        return exc
    raise AssertionError("not raised")


def _said(exc, capsys, monkeypatch, json_mode: bool):
    from poros.cli import contract, errors

    monkeypatch.setattr(contract, "json_mode", lambda: json_mode)
    with pytest.raises(typer.Exit) as done:
        errors._friendly_error(exc)
    captured = capsys.readouterr()  # as one line: a long path wraps on the rail
    return done.value.exit_code, " ".join((captured.out + captured.err).replace("│", " ").split())


@pytest.mark.parametrize("json_mode", [False, True], ids=["person", "json"])
def test_a_download_that_fills_the_disk_is_e_disk_in_plain_words(capsys, monkeypatch, tmp_path,
                                                                json_mode):
    from huggingface_hub import constants

    from poros.cli import errors
    from poros.core.errors import EXIT_CODES

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    exc = _hub_xet_error()
    assert errors._error_code(exc) == "E_DISK"
    code, text = _said(exc, capsys, monkeypatch, json_mode)
    assert code == EXIT_CODES["E_DISK"] and "bug in Poros" not in text
    assert "Hugging Face cache" in text and str(tmp_path / "hub") in text and "GB free" in text
    if json_mode:
        assert '"E_DISK"' in text


def test_a_write_that_fills_the_disk_names_the_file(capsys, monkeypatch, tmp_path):
    from poros.cli import errors

    where = tmp_path / "runs" / "adapter.safetensors"
    exc = OSError(errno.ENOSPC, "No space left on device", str(where))
    assert errors._error_code(exc) == "E_DISK"
    code, text = _said(exc, capsys, monkeypatch, False)
    assert "Disk full" in text and f"writing {where}" in text and "GB free" in text


def test_another_runtime_error_is_still_a_bug(capsys, monkeypatch):
    from poros.cli import errors

    assert errors._error_code(RuntimeError("shape mismatch")) == "E_INTERNAL"


# ── the store's words ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("store", [None, "ram"])
def test_a_dense_run_says_it_trains_from_ram_whether_or_not_it_says_store_ram(tmp_path, store):
    from poros.cli.sheets import _store_rows

    a = SimpleNamespace(plan=SimpleNamespace(entry="dense"), config={"store": "ram"})
    (row,) = _store_rows(a, {"--store": store}, _ctx(tmp_path, False))
    assert row.source == "dense model: SSD streaming is for MoE" and row.value == store


# ── num_epochs= as the config reads it ─────────────────────────────────────────────────────


@pytest.mark.parametrize("said, steps", [("0.5", None), ("3.0", 3), ("abc", None)])
def test_num_epochs_is_what_the_config_takes_and_its_own_words_otherwise(home, said, steps):  # noqa: F811
    """num_epochs is a whole number of passes, at least 1: 3.0 is 3; 0.5 and abc are refused
    with the config's words, never a traceback."""
    data = _rows(home / "e.jsonl", "chats")
    words = ["train", "Qwen/Qwen2.5-7B-Instruct", "--data", data, "--recipe", "sft",
             f"num_epochs={said}"]
    if steps is None:
        with pytest.raises(Refusal) as found:
            _start(words, _ctx(home, False))
        assert found.value.title == "Bad setting" and "num_epochs" in found.value.reason
        return
    a = _start(words, _ctx(home, False))
    assert a.steps == a.per_epoch * steps


# ── a run's own model, each setting once, one step, one ram figure ─────────────────────────


def _trained(home, capsys, steps: str = "1"):  # noqa: F811
    from tests.cli.test_train_runs import say

    code, obj = say(["train", "Qwen/Qwen2.5-0.5B-Instruct", "--data", "synthetic", "--steps",
                     steps, "lora_alpha=32", "learning_rate=2e-4", "lora_dropout=0",
                     "per_device_batch_size=1", "gradient_accumulation_steps=8"], home, capsys)
    assert code == 0, obj
    from poros import runs

    return next(r for r in runs.all_runs() if r.id == obj["run"])


def test_a_runs_line_names_its_own_model_not_its_base(home, capsys):  # noqa: F811
    from poros.cli.recipes_card import _model_short
    from poros.cli.sheets.listings import _run_detail

    r = _trained(home, capsys)
    said = " ".join(" ".join(t for t, _ in _run_detail(r)).split())
    assert said.startswith("Qwen2.5-0.5B-Instruct ·"), said
    assert _model_short("Qwen/Qwen2.5-7B-Instruct") == "Qwen2.5-7B-Instruct"


def test_a_runs_page_says_each_setting_once_and_one_ram_figure(home, capsys, monkeypatch):  # noqa: F811
    from poros.cli.ui import components as c
    from poros.cli.verbs import runs as runs_verb
    from poros.cli.views import run_screens

    r = _trained(home, capsys)
    out = runs_verb.run(runs_verb.RunsAction(do="show", run=r.id), Context(
        cwd=home, fixed_cached=frozenset(), fixed_interactive=False))
    text = "\n".join("".join(t for t, _ in line) for line in run_screens.runs(out, 200))
    ran = " ".join(text.split("ran", 1)[1].split("files", 1)[0].replace("│", " ").split())
    for word in ("alpha 32", "lr 2e-4", "accum 8", "dropout 0", "batch 1"):
        assert ran.count(word) == 1, (word, ran)
    for raw in ("lora_alpha", "learning_rate", "gradient_accumulation_steps",
                "per_device_batch_size"):
        assert raw not in ran, (raw, ran)
    receipt = json.loads((Path(r.output_dir) / "receipt.json").read_text())
    ram = c.ceil1(c.gib(receipt["peak_host_ram_gb"]))  # the end card's figure and word
    assert f"ram     {ram:.1f} GiB peak" in text and "host peak" not in text


def test_a_count_of_one_step_is_said_in_the_singular():
    from poros.cli.ui import components as c

    assert (c.steps(1), c.steps(16), c.steps(1200)) == ("1 step", "16 steps", "1,200 steps")


def test_the_runs_footer_peak_is_the_end_cards_allocated_peak(home, capsys):  # noqa: F811
    from poros.cli.sheets.listings import _run_detail
    from poros.cli.ui import components as c

    r = _trained(home, capsys)
    receipt = json.loads((Path(r.output_dir) / "receipt.json").read_text())
    said = " ".join(" ".join(t for t, _ in _run_detail(r)).split())
    want = c.ceil1(c.gib(receipt["peak_vram_allocated_gb"]))
    assert f"{want:.1f} GiB peak" in said, said


def test_the_checked_line_names_fast_cache_as_the_recipe_does():
    from poros.cli.verbs import shared

    assert shared.path_choice({"path": "fast", "store": "ssd",
                               "store_options": {"cache": shared.FAST_CACHE}}) == "fast+cache"


def test_the_setup_refusal_names_its_command_first_where_a_sheet_shows_two_lines(tmp_path):
    """A terminal's train sheet shows a refusal's words and its first line of context, not its
    fix: the working command is that line."""
    from poros.cli.verbs import train as tv
    from poros.cli.words.grammar import parse

    with pytest.raises(Refusal) as found:
        tv.resolve(parse(["train", "--recipe", "first-run", "--data", "rows.jsonl"]),
                   _ctx(tmp_path, False))
    assert found.value.context[0] == f"it trains with: {found.value.fix}"
    assert found.value.fix == shlex.join(["poros", *presets.DEFINED_BY_KEY["first-run"].argv(
        "rows.jsonl")])


def test_a_terminals_train_sheet_shows_the_setups_working_command(tmp_path):
    """The terminal's sheet for `poros train --recipe first-run --data rows.jsonl`: the reason it
    shows holds the command that trains the setup, the data filled in."""
    from poros.cli.sheets.common import _resolved

    action, why, asks = _resolved("train", {"--recipe": "first-run", "--data": "rows.jsonl"},
                                  _ctx(tmp_path, False))
    want = shlex.join(["poros", *presets.DEFINED_BY_KEY["first-run"].argv("rows.jsonl")])
    assert action is None and f"it trains with: {want}" in why, why


def test_eval_keeps_the_runs_tables_and_says_nothing_of_them(tmp_path):
    """poros eval scores the run with the tables where the run had them, read from its settings:
    it decides nothing, so it says nothing about them (the 72B's eval printed an estimate as if
    it were deciding them)."""
    from poros import runs
    from poros.cli.verbs import evaluate_pass as ep

    (tmp_path / "config.resolved.json").write_text(json.dumps({
        "model_name_or_path": "Qwen/Qwen2.5-72B-Instruct", "dataset": "synthetic",
        "offload_embeddings_and_head": True, "offload_reason": "an estimate's words"}))
    r = runs.Run(id="r", kind="run", state="succeeded", model="m", name="r",
                 output_dir=str(tmp_path), created_at=None, started_at=None, finished_at=None,
                 step=None, total=None, loss=None, error=None, folder=str(tmp_path))
    cfg, _ = ep.settings(r, tmp_path)
    assert cfg.offload_embeddings_and_head is True and not cfg.offload_reason


def test_the_start_says_the_tables_once_and_only_where_it_decides_them(monkeypatch):
    """poros train says "tables: host RAM" itself where its check decided them; the start says
    nothing more of them then, and says it, in those words, where it decides them itself."""
    from poros.core.config import PorosConfig
    from poros.preflight import fit

    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-72B-Instruct", dataset="synthetic")
    done = SimpleNamespace(path=None, warning=None, vram=None, host=None, p=None)

    def deciding(c, plan, **kw):
        c.decide_offload(True, "with them on the GPU it may not fit")
        return done

    monkeypatch.setattr(fit, "planner_here", lambda *a, **k: None)
    monkeypatch.setattr(fit, "check", deciding)
    said = []
    fit.gate(cfg, "qwen2", None, say=lambda *a: said.append(a[0] % a[1:]))
    assert said == ["tables: host RAM: with them on the GPU it may not fit"]
    said.clear()
    fit.gate(cfg, "qwen2", None, say=lambda *a: said.append(a[0] % a[1:]))  # decided before
    assert said == []
