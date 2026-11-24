"""The training path, chosen and planned: the train card's path row says what an unset path takes
and why, and lists what each path would need on this card; the fit row follows the chosen path;
the planner gives each path's fit (shared.path_plans) and plans path=."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

import pytest

from poros import runs
from poros.cli import sheets
from poros.cli.context import Context
from poros.cli.schemas import Figure, PlanGpu
from poros.cli.ui.sheet import Sheet
from poros.cli.ui.text import width
from poros.cli.verbs import shared
from poros.cli.words.grammar import parse
from tests.conftest import checkout_env

# Qwen2.5-7B on one RTX 4090, each loop at 512 tokens, 1 x 8, block 4, dropout 0.05, PEFT's
# modules (docs/validation/block-major-dropout/): allocated / reserved, GiB. The card below asks
# for that compute (fused_compute=false), so each path's figure is its receipt's.
FOUR_090 = {"qwen2.5-7b-rtx4090-synthetic-efficient": ("efficient", 7.105, 11.16),
            "qwen2.5-7b-rtx4090-synthetic-fast": ("fast", 8.908, 11.004),
            "qwen2.5-7b-rtx4090-synthetic-dropout0.05": ("sequential", 6.245, 11.004)}
CARD = PlanGpu(name="RTX 4090", total_gib=24.0, free_gib=None, source="spec")


@pytest.fixture
def receipts(monkeypatch, tmp_path):
    """Only the three 4090 receipts, whatever cells the release adds; nothing from the Hub."""
    import poros.plan.cells
    import poros.plan.configs

    cells = copy.deepcopy([c for c in poros.plan.cells.CELLS if c["id"] in FOUR_090])
    assert {c["id"]: (c["key"].get("path", "sequential"),
                      c["fields"]["peak_vram_allocated"]["value"],
                      c["fields"]["peak_vram_reserved"]["value"]) for c in cells} == FOUR_090
    monkeypatch.setattr(poros.plan.cells, "CELLS", cells)
    # no config.json: nothing is sized beyond the receipts (the figures here are theirs)
    monkeypatch.setattr(poros.plan.configs, "read_model_config", lambda model, revision=None: None)
    monkeypatch.setattr(poros.plan.configs, "cached_model_config", lambda model: None)
    monkeypatch.setattr(poros.plan.configs, "download_size", lambda cfg: Figure(
        value=None, unit="GiB", source="unknown", why="offline in tests"))
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "home"))
    (tmp_path / "notes.jsonl").write_text('{"text": "hello"}\n' * 40)
    return tmp_path


def _ctx(where, gpu=CARD, host_free_gib=None) -> Context:
    ctx = Context(cwd=where, fixed_gpu=gpu, fixed_cached=frozenset({"Qwen/Qwen2.5-7B"}),
                  fixed_interactive=True)
    ctx.__dict__["host_free_gib"] = host_free_gib  # a card planned by name reads no host RAM
    return ctx


def _host_ram(path: str, gib: float) -> None:
    """The pinned receipt of ``path`` says it peaked at ``gib`` of host RAM."""
    import poros.plan.cells
    import poros.plan.configs

    cell = next(c for c in poros.plan.cells.CELLS if c["key"].get("path", "sequential") == path)
    cell["fields"]["peak_host_ram"]["value"] = gib


CARD_SAYS = ("train Qwen2.5-7B --data notes.jsonl --seq-len 512 lora_dropout=0.05 "
             "fused_compute=false")


def _card(where, gpu=CARD) -> Sheet:
    s = parse(CARD_SAYS)
    return Sheet(sheets.train_form(s, _ctx(where, gpu)))


def _row(sheet: Sheet, key: str):
    return next(f for f in sheet.view.fields if f.key == key)


def _text(lines) -> str:
    return "\n".join("".join(t for t, _ in line) for line in lines)


def test_an_unset_path_reads_as_the_one_it_takes_and_why(receipts):
    sheet = _card(receipts)
    # the fit row says the decision and the path, and no planner figure
    assert sheet.view.result == ("fit", "checked at the start",
                                 "no certified run of these settings here · efficient path")
    more = _row(sheet, "more").options()
    assert next(o for o in more if o.value == "path=").now == "auto → efficient"
    sheet._add("path=", "")
    row = _row(sheet, "path=")
    assert (row.shown, row.source) == ("auto → efficient", "NF4's default")
    line = next(x for x in _text(sheet.render(100)).splitlines() if " path " in x)
    assert line.split()[2:6] == ["auto", "→", "efficient", "NF4's"]
    assert line.rstrip().endswith("NF4's default")
    assert sheets.common.form_argv("train", sheet.form.values)[-1] != "path="  # auto adds nothing


def test_each_path_lists_what_it_needs_on_this_card(receipts):
    sheet = _card(receipts)
    sheet._add("path=", "")
    said = {o.label: o.note for o in _row(sheet, "path=").options()}
    assert list(said) == ["auto → efficient", "efficient", "fast", "sequential"]
    # what each path is, and the run's own choice; no planner figure
    assert said["auto → efficient"] == "what it trains at · NF4's default"
    assert said["efficient"] == ("the default: less VRAM than fast; block inputs in host RAM")
    assert said["fast"].startswith("the fastest")
    assert said["sequential"] == "one micro-batch at a time: slower, and its peak can be lower"
    assert not any("GiB" in s or "s/update" in s for s in said.values())


def test_choosing_a_path_replans_the_fit_row(receipts):
    sheet = _card(receipts)
    sheet._add("path=", "")
    sheet._set("path=", "fast")
    a, _, _ = sheets.common._resolved("train", sheet.form.values, _ctx(receipts))
    assert a.plan.path == "fast" and a.config["path"] == "fast"
    assert sheet.view.result[2].endswith("fast path") and "GiB" not in sheet.view.result[1]
    assert _row(sheet, "path=").shown == "fast"
    assert "path=fast" in sheet.view.command


@pytest.mark.parametrize("w", [60, 80, 100])
def test_the_path_row_and_its_list_fit_every_width(receipts, w):
    sheet = _card(receipts)
    sheet._add("path=", "")
    sheet.key("enter", "")  # its list, open
    assert sheet.mode == "pick"
    lines = sheet.render(w)
    assert all(width(line) <= w for line in lines), _text(lines)
    assert "auto → efficient" in _text(lines)


def test_a_path_that_wont_fit_says_so(receipts):
    small = PlanGpu(name="RTX 4090", total_gib=24.0, free_gib=8.5, source="nvidia-smi")
    sheet = _card(receipts, small)
    sheet._add("path=", "")
    said = {o.value: o.note for o in _row(sheet, "path=").options()}
    assert said["fast"].startswith("won't fit")  # 8.9 GiB allocated > 8.5 free: a decision
    assert not said["efficient"].startswith("won't fit")  # 7.1 fits; 11.2 reserved may not
    sheet._set("path=", "fast")
    assert sheet.view.result[1] == "won't fit" and sheet.view.result_style == "e"
    assert sheet.view.result[2].endswith("· fast path")


AT = {"seq_len": 512, "rank": 16, "dropout": 0.05}


def _paths(ctx):
    """Each path's fit at the receipts' settings, and the plan a run with path unset takes."""
    plans = shared.path_plans("Qwen/Qwen2.5-7B", ctx, config_json=True, **AT)
    first = shared.plan_for("Qwen/Qwen2.5-7B", ctx, **AT)
    took = shared.auto_path(first, lambda k: plans.get(k) or shared.plan_for(
        "Qwen/Qwen2.5-7B", ctx, config_json=True, path_settings={"path": k}, **AT))
    return [_summary(k, v) for k, v in plans.items()], took


def _summary(path, p):
    """A path's plan, short, as these tests compare it: its verdict and allocated peak (or, where
    only a larger run measured it, that run's bound) and where the peak comes from. Internal: no
    command prints a plan's figures."""
    from types import SimpleNamespace

    alloc, bound = p.peak_vram_allocated, p.bound
    if alloc.value is None and bound is not None and bound.fits and bound.peak_gib is not None:
        value, source = bound.peak_gib, "bounded"
    else:
        value = alloc.value
        source = alloc.source if alloc.source in ("measured", "estimated") else "unknown"
    return SimpleNamespace(path=path, verdict=shared.fit_note(p), peak_vram_allocated_gib=value,
                           peak_vram_reserved_gib=p.peak_vram_reserved.value,
                           source="unknown" if value is None else source)


def test_the_planner_gives_each_paths_fit_and_plans_the_one_asked(receipts):
    ctx = Context(cwd=receipts, fixed_gpu=CARD, fixed_cached=frozenset(), fixed_interactive=False)
    paths, took = _paths(ctx)
    # the sequential loop's measured peak is below efficient's: nothing calls efficient the
    # least VRAM
    assert (took.path, shared.path_why(took)) == ("efficient", "NF4's default")
    got = {p.path: (p.verdict, p.peak_vram_allocated_gib, p.peak_vram_reserved_gib, p.source)
           for p in paths}
    assert list(got) == ["efficient", "fast", "sequential"]
    # the fused default: each PEFT-module receipt scaled by the largest fused / PEFT ratio
    # measured, with the margins of a figure from another run
    from poros.plan.estimate import _FUSED_OVER_PEFT, _RESERVED, _SAME_GPU

    want = {path: ("fits", round(alloc * _FUSED_OVER_PEFT[0] * _SAME_GPU, 3),
                   round(reserved * _FUSED_OVER_PEFT[1] * _RESERVED, 3), "estimated")
            for path, alloc, reserved in FOUR_090.values()}
    assert got == want
    fast = shared.plan_for("Qwen/Qwen2.5-7B", ctx, path_settings={"path": "fast"}, **AT)
    assert (fast.path, fast.path_reason) == ("fast", None)


def test_a_path_that_wont_fit_is_offered_one_that_does(receipts):
    _host_ram("fast", 30.0)  # fast's peak is over this host's 20 GiB; efficient's 15.1 is not
    ctx = _ctx(receipts, host_free_gib=20.0)
    fast = shared.plan_for("Qwen/Qwen2.5-7B", ctx, path_settings={"path": "fast"}, **AT)
    assert shared.fit_note(fast) == "won't fit" and fast.host_fits is False
    assert [p.verdict for p in _paths(ctx)[0]] == ["fits", "won't fit", "fits"]
    sheet = _card(receipts)
    sheet.form.build = sheets.train_form(  # the same card, on this host
        parse(CARD_SAYS), ctx).build
    sheet._add("path=", "")
    fast = next(o for o in _row(sheet, "path=").options() if o.value == "fast")
    assert fast.note.startswith("won't fit host RAM")


def test_an_unset_path_short_of_host_ram_takes_the_one_that_fits(receipts):
    """Where efficient's block inputs would not fit host RAM, a run takes fast (as the trainer
    decides it, path_that_fits_host): the plan, the card and their reasons say that path."""
    _host_ram("fast", 15.0)  # efficient's 15.114 GiB peak is over this host's 15.11; fast's not
    ctx = _ctx(receipts, host_free_gib=15.11)
    took = _paths(ctx)[1]
    reason = "the check turned down efficient: its block inputs are past the host RAM available"
    assert took.path == "fast" and took.path_reason == reason
    sheet = Sheet(sheets.train_form(parse(CARD_SAYS), ctx))
    assert sheet.view.result[2].endswith("· fast path")
    assert not any(f.warn for f in sheet.view.fields)  # no "won't fit" to answer: fast fits
    sheet._add("path=", "")
    row = _row(sheet, "path=")
    assert (row.shown, row.source) == ("auto → fast",
                                       reason)
    assert row.options()[0].label == "auto → fast"


def test_a_setting_that_forces_the_sequential_loop_reads_as_it(receipts):
    """A LoRA on the LM head trains outside the blocks, which block-major cannot: a run takes the
    sequential loop (PorosConfig.apply_cuda_defaults, the trainer's own rule, which the plan
    calls), and the card says so, and why."""
    sheet = Sheet(sheets.train_form(parse(CARD_SAYS + " lora_target_modules=lm_head|q_proj"),
                                    _ctx(receipts)))
    a, _, _ = sheets.common._resolved("train", sheet.form.values, _ctx(receipts))
    assert (a.plan.path, a.plan.path_reason) == (
        "sequential", "an embedding table or LM head that trains")
    more = _row(sheet, "more").options()
    assert next(o for o in more if o.value == "path=").now == "auto → sequential"
    sheet._add("path=", "")
    row = _row(sheet, "path=")
    assert (row.shown, row.source) == ("auto → sequential",
                                       "an embedding table or LM head that trains")
    said = {o.label: o.note for o in row.options()}
    assert list(said)[0] == "auto → sequential"
    assert said["fast"] == said["efficient"] == (
        "refused: block-major cannot train an embedding table or LM head that trains")


def test_a_mac_card_has_no_path_row_and_no_paths(tmp_path):
    mac = Context(fixed_gpu=PlanGpu(name="Apple M4", total_gib=16.0, free_gib=None,
                                    source="spec", unified=True), fixed_cached=frozenset())
    assert shared.path_plans("Qwen/Qwen2.5-0.5B", mac, seq_len=512) == {}
    assert "path" not in sheets.config_fields(mac)


# -- through `poros agents` -------------------------------------------------------------------


def _agent(*args, env=None, cwd=None, codes=(0,)) -> dict:
    proc = subprocess.run(
        [sys.executable, "-P", "-c", "from poros.cli.main import cli; cli()", "agents", *args],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=cwd,
        env=checkout_env({**os.environ, "POROS_DEV_SIMULATE": "1", **(env or {})}))
    assert proc.returncode in codes, proc.stdout + proc.stderr[-800:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture
def home(tmp_path, monkeypatch):
    for variable in ("POROS_AGENT", "POROS_OUTPUT", "POROS_JOB"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "me"))
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "poros"))
    (tmp_path / "me").mkdir()
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    return tmp_path


def test_an_agent_starts_a_run_on_the_path_it_set(home):
    (home / "work" / "d.jsonl").write_text(json.dumps(
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}
    ) + "\n")
    done = _agent("train", "qwen2.5-0.5b-instruct", "--data", "d.jsonl", "--steps", "2",
                  "path=fast")
    assert done["schema"] == "poros.train.v1" and "path=fast" in done["command"]
    ran = json.loads((runs.root() / done["run"] / "config.resolved.json").read_text())
    assert ran["path"] == "fast"


def test_machine_json_says_no_fit(home):
    """`poros machine` checks the machine; what trains where is `poros recipes`."""
    out = _agent("machine", env={"POROS_DEV_SIMULATE_GPU": "RTX 4090:24.0"}, codes=(0, 5))
    assert out["schema"] == "poros.machine.v1" and "can_train" not in out
    schema = _agent("schema", "machine")
    assert "can_train" not in schema["properties"]
