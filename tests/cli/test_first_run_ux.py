"""What a first run meets: a plan for any card, true labels,
one way to name a model, one help per command, errors on stderr, and nothing internal in a
user's text."""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys

import pytest

from poros.cli import app, contract, registry
from poros.cli.context import Context
from poros.cli.schemas import DoctorCheck, PlanGpu
from poros.core import card
from poros.core.refusal import Refusal
from tests.conftest import checkout_env, machine_exit_here

CARD = PlanGpu(name="RTX 4090", total_gib=23.99, free_gib=None, source="spec")
CHAT = {"messages": [{"role": "user", "content": "What is two plus two?"},
                     {"role": "assistant", "content": "Four."}]}


def _poros(*args, cwd=None, env=None):
    """One command as a person's shell runs it: no terminal, no JSON asked, the Hub off."""
    base = {k: v for k, v in os.environ.items()
            if k not in ("POROS_OUTPUT", "POROS_AGENT", "POROS_JOB", "POROS_DEV_SIMULATE")}
    return subprocess.run(
        [sys.executable, "-P", "-c", "from poros.cli.main import cli; cli()", *args],
        capture_output=True, text=True, stdin=subprocess.DEVNULL, cwd=cwd,
        env=checkout_env({**base, "HF_HUB_OFFLINE": "1", **(env or {})}))


@pytest.fixture
def home(tmp_path, monkeypatch):
    for variable in ("POROS_AGENT", "POROS_OUTPUT", "POROS_JOB", "POROS_DEV_SIMULATE"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "poros"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    (tmp_path / "work").mkdir()
    monkeypatch.chdir(tmp_path / "work")
    yield tmp_path / "work"
    contract.reset()


def _json(argv, ctx, capsys, monkeypatch):
    monkeypatch.setenv("POROS_OUTPUT", "json")
    code = app.dispatch(argv, ctx)
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1, lines  # the contract: exactly one JSON object
    return code, json.loads(lines[0])


def _ctx(where) -> Context:
    return Context(cwd=where, fixed_cached=frozenset(), fixed_interactive=False)


def _text(lines) -> str:
    return "\n".join("".join(seg[0] for seg in line) for line in lines)


def _flat(text: str) -> str:
    """A screen's words on one line: no rail, no wrapping."""
    return " ".join(text.replace("│", " ").split())


# -- a plan for any card -------------------------------------------------------------------------


@pytest.mark.parametrize("said, again, name, gib, generation", [
    ("8GB", "8GB", "NVIDIA GPU", 8.0, None),
    ("12", "12GB", "NVIDIA GPU", 12.0, None),
    ("24 gib", "24GB", "NVIDIA GPU", 24.0, None),
    ("RTX 4060", "RTX 4060", "RTX 4060", 8.0, "Ada"),
    ("rtx 3070", "RTX 3070", "RTX 3070", 8.0, "Ampere"),
    ("4070 ti super", "RTX 4070 Ti SUPER", "RTX 4070 Ti SUPER", 16.0, "Ada"),
    ("5080", "RTX 5080", "RTX 5080", 16.0, "Blackwell"),
    ("RTX 4060 Ti 16GB", "RTX 4060 Ti 16GB", "RTX 4060 Ti 16GB", 16.0, "Ada"),
    ("a6000", "RTX A6000", "RTX A6000", 48.0, "Ampere"),
    ("rtx 6000 ada", "RTX 6000 Ada", "RTX 6000 Ada", 48.0, "Ada"),
    ("l4", "L4", "L4", 24.0, "Ada"),
    ("A100 40GB", "A100 40GB", "A100 40GB", 40.0, "Ampere"),
    ("4090", "RTX 4090", "RTX 4090", 23.99, "Ada"),  # a measured card keeps nvidia-smi's figure
    ("A100 80GB", "A100 80GB PCIe", "A100 80GB PCIe", 80.0, "Ampere"),
    ("6000", "RTX PRO 6000 Blackwell Workstation Edition",
     "RTX PRO 6000 Blackwell Workstation Edition", 95.59, "Blackwell"),
    ("RTX 9090 48GB", "RTX 9090 48GB", "RTX 9090 48GB", 48.0, None),  # no one knows it: its size
])
def test_gpu_takes_a_card_by_name_by_memory_or_both(said, again, name, gib, generation):
    canonical, gpu = card.named_card(said)
    assert (canonical, gpu.name, gpu.total_gib, gpu.source) == (again, name, gib, "spec")
    assert gpu.free_gib is None  # all of it is taken as free
    cap = card.card_capability(gpu)
    assert (card.class_name(cap) if cap else None) == generation
    assert card.named_card(canonical)[0] == canonical  # a result's command says it again


def test_gpu_refuses_a_name_it_cannot_size_and_a_size_no_card_has():
    for unknown in ("banana", "RTX 4060 Laptop GPU", "H200"):
        with pytest.raises(card.UnknownCard):
            card.named_card(unknown)
    with pytest.raises(card.CardNeedsSize) as two:
        card.named_card("rtx 4060 ti")
    assert two.value.sizes == (8, 16) and two.value.name == "RTX 4060 Ti"
    for size in ("0GB", "4096GB"):
        with pytest.raises(card.CardNeedsSize):
            card.named_card(size)


def _named(home, said: str) -> Context:
    from poros.cli.verbs import shared

    return Context(cwd=home, gpu_name=shared.gpu_name(said), fixed_cached=frozenset(),
                   fixed_interactive=False)


def test_the_planner_plans_an_unmeasured_card_as_estimates(home):
    """A card by memory (8GB) or by name ("RTX 4060"): every figure an estimate, said as one."""
    from poros.cli.verbs import shared

    for said in ("8GB", "RTX 4060"):
        p = shared.plan_for("Qwen/Qwen2.5-0.5B-Instruct", _named(home, said))
        assert p.gpu.total_gib == 8.0 and p.gpu.source == "spec"
        for figure in (p.peak_vram_allocated, p.peak_vram_reserved, p.s_per_step):
            assert figure.source in ("estimated", "unknown")
        assert any(w.startswith("no measured run on this card") for w in p.warnings)


def test_a_measured_card_has_no_unmeasured_note(home):
    from poros.cli.verbs import shared

    p = shared.plan_for("Qwen/Qwen2.5-7B", _named(home, "4090"))
    assert not any("no measured run on this card" in w for w in p.warnings)


@pytest.mark.parametrize("old", ["GTX 1080", "T4 16GB", "RTX 2080 Ti", "V100"])
def test_a_card_poros_does_not_train_on_gets_that_refusal(old):
    from poros.cli.verbs import shared

    with pytest.raises(Refusal) as hit:
        shared.gpu_name(old)
    assert hit.value.code == "E_GPU_UNSUPPORTED" and "no native bf16" in hit.value.reason


def test_a_card_name_with_no_size_says_how_to_give_one():
    from poros.cli.verbs import shared

    with pytest.raises(Refusal) as unknown:
        shared.gpu_name("banana")
    assert "banana" in unknown.value.reason and "--gpu 16GB" in " ".join(unknown.value.context)
    with pytest.raises(Refusal) as two:
        shared.gpu_name("4060 ti")
    assert '--gpu "RTX 4060 Ti 16GB"' in " ".join(two.value.context)


def test_gpu_refusals_are_refusals():
    from poros.cli.verbs import shared

    with pytest.raises(Refusal) as none:
        shared.gpu_name(True)
    assert none.value.code == "E_USAGE" and "8GB" in " ".join(none.value.context)
    assert shared.gpu_name(None) is None and shared.gpu_name(" rtx  4090 ") == "RTX 4090"


# -- the length a config run reaches ---------------------------------------------------------------


def _config(home, rows=20, **more) -> None:
    (home / "chats.jsonl").write_text("".join(json.dumps(CHAT) + "\n" for _ in range(rows)))
    lines = {"model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct", "dataset": "chats.jsonl",
             "output_dir": "./run", **more}
    (home / "run.yaml").write_text("".join(f"{k}: {json.dumps(v)}\n" for k, v in lines.items()))


def test_a_config_run_is_checked_at_the_length_it_reaches(home):
    """`poros train <config>`'s check reads the length the run reaches: the config's where it
    sets one, else its data's longest row, else the default, and says which."""
    from poros.cli.commands.train import _at_data_length
    from poros.core.config import PorosConfig

    _config(home)
    at, said = _at_data_length(PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
                                           dataset="chats.jsonl"))
    assert said.startswith("your data") and at.max_seq_length == 256
    at, said = _at_data_length(PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
                                           dataset="chats.jsonl", max_seq_length=1024))
    assert (at.max_seq_length, said) == (1024, "max_seq_length in the config")
    at, said = _at_data_length(PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct",
                                           dataset="some-org/some-dataset"))
    assert at.max_seq_length == 2048 and said.startswith("the default")


def test_a_long_row_plans_the_length_a_config_run_reaches(home):
    """A config that sets no max_seq_length cuts no row: its plan is for the longest row, and
    says the length `fit` would cut to."""
    from poros.cli.commands.train import _at_data_length
    from poros.core.config import PorosConfig

    long = {"messages": [{"role": "user", "content": "word " * 900},
                         {"role": "assistant", "content": "ok"}]}
    (home / "chats.jsonl").write_text("".join(json.dumps(CHAT) + "\n" for _ in range(40))
                                      + json.dumps(long) + "\n")
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct", dataset="chats.jsonl")
    at, said = _at_data_length(cfg)
    assert at.max_seq_length == 2048 and "max_seq_length" not in at.model_fields_set
    assert "the longest of 41 rows" in said
    assert "95% of the rows are no longer than 256 tokens" in said
    assert cfg.max_seq_length == 2048  # the run's own config is untouched


def test_a_row_of_cjk_text_is_planned_a_token_a_character(home):
    """The length was read at 4 characters a token for any text: 600 Chinese characters read as
    150 tokens (length 256), where the tokenizer makes about 600 of them. The fit check passed
    a run at a quarter of the length it then trained at. A non-ASCII character counts as a
    token, and the note says the length is an estimate."""
    from poros.cli.commands.train import _at_data_length
    from poros.cli.verbs import train_data
    from poros.core.config import PorosConfig

    assert train_data._weight("abcd" * 10) == 40  # ASCII: as before, 4 characters a token
    assert train_data._weight("\u6c49" * 10) == 40 and train_data._weight("a\u6c49") == 5
    cjk = {"messages": [{"role": "user", "content": "\u6c49\u5b57" * 300},
                        {"role": "assistant", "content": "ok"}]}
    (home / "cjk.jsonl").write_text(json.dumps(cjk, ensure_ascii=False) + "\n", encoding="utf-8")
    scan = train_data.scan("cjk.jsonl", home)
    assert 600 <= scan.longest_tokens <= 620 and scan.whole_len == 1024
    cfg = PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct", dataset="cjk.jsonl")
    at, said = _at_data_length(cfg)
    assert at.max_seq_length == 1024  # 256 before: the fit check ran at a quarter of the run
    assert "an estimate from its characters" in said
    # a csv cell and a row of ASCII read as they did
    (home / "rows.csv").write_text("prompt,completion\n" + "word " * 100 + ",ok\n")
    assert train_data.scan("rows.csv", home).longest_tokens == (500 + 2) // 4


# -- fit's labels are true -----------------------------------------------------------------------


def test_a_plan_never_says_no_memory_figure_beside_one(monkeypatch):
    """With no config.json to size the path, the memory figures come from the model's other
    runs: the note names what the sequential loop stands in for, and nothing as missing."""
    import poros.plan.configs
    from poros.cli.verbs import shared

    monkeypatch.setattr(poros.plan.configs, "read_model_config", lambda m, revision=None: None)
    monkeypatch.setattr(poros.plan.configs, "cached_model_config", lambda m: None)
    p = shared.plan_for("Qwen/Qwen2.5-7B", Context(fixed_gpu=CARD, fixed_cached=frozenset()))
    assert p.peak_vram_allocated.value is not None
    assert not any("no memory figure" in w for w in p.warnings)


# -- machine -------------------------------------------------------------------------------------


def _machine(**more):
    from poros.cli.verbs.machine import MachineOutput

    checks = [DoctorCheck(name="Python", value="3.12", status="ok"),
              DoctorCheck(name="torch", value="2.15.0", status="ok"),
              DoctorCheck(name="CUBLAS_WORKSPACE_CONFIG",
                          value="poros train sets it as it starts", status="info")]
    return MachineOutput(**{"verdict": "ready", "version": "0.3.0", "gpu": CARD,
                            "host_ram_free_gib": 40.0, "checks": checks,
                            **more})


def test_a_clean_install_reads_ready_with_no_alarm():
    """Training sets CUBLAS_WORKSPACE_CONFIG itself: unset is a note in both forms of machine."""
    from poros.cli.commands.check import _cublas_check
    from poros.cli.views.machine_screens import _machine_rows

    assert _cublas_check("")[2] == "info"
    text = _text(_machine_rows(_machine(), 100))
    assert text.splitlines()[0].endswith("Ready to train")
    assert "CUBLAS" not in text and "Almost ready" not in text


def test_machine_suggests_the_starter_model_with_its_data():
    from poros.cli.commands.train import STARTER_MODEL
    from poros.cli.verbs.shared import START_WITH
    from poros.cli.views.machine_screens import _machine_next, _machine_rows

    assert START_WITH == STARTER_MODEL == "Qwen/Qwen2.5-0.5B-Instruct"  # one starter, everywhere
    r = _machine()
    assert _machine_next(r) == "train Qwen2.5-0.5B-Instruct --data <file>"
    assert "Qwen2.5-72B" not in _text(_machine_rows(r, 100))  # it names no model


def test_models_shows_no_figure_and_the_recipes_certified_here(home):
    """`poros models` shows no model's memory or speed; a model with a recipe certified on this
    machine says which."""
    from poros.cli.sheets.listings import models_listing

    listing = models_listing(Context(fixed_gpu=CARD, fixed_cached=frozenset()))
    text = " ".join(t for i in listing.items for t, _ in i.cells)
    assert "GiB" not in text and "estimated" not in text and "measured" not in text


def test_machine_draws_no_card_into_a_pipe(home):
    env = {"POROS_DEV_SIMULATE": "1", "POROS_DEV_SIMULATE_GPU": "RTX 4090:23.52", "COLUMNS": "120"}
    proc = _poros("machine", cwd=home, env=env)
    assert proc.returncode == machine_exit_here() and "RTX 4090" in proc.stdout
    assert not re.search("[⠀-⣿]", proc.stdout)  # braille: the drawn card
    # the next step: a first run on the starter model; without the training stack, its install
    first = ("poros train Qwen2.5-0.5B-Instruct --data <file>"
             if importlib.util.find_spec("torch") else "pip install poros-train")
    assert f"Continue with: {first}" in proc.stdout


# -- one way to name a model ---------------------------------------------------------------------


def test_every_command_names_a_model_one_way(home, monkeypatch):
    from poros.preflight import arch as preflight_arch

    assert preflight_arch.named_model("qwen2.5-7b-instruct") == "Qwen/Qwen2.5-7B-Instruct"
    assert preflight_arch.named_model("QWEN/qwen2.5-7b") == "Qwen/Qwen2.5-7B"
    assert preflight_arch.named_model("org/other-model") == "org/other-model"
    (home / "m").mkdir()
    assert preflight_arch.named_model("m") == "m"  # a folder here is that folder


def test_offline_a_catalog_model_is_known_from_the_catalog(home, monkeypatch, tmp_path):
    """Offline with an empty cache, `check` and `arch` know a catalog model's family as `fit`
    does, by its short name too; a model no catalog names says, in one message, that the Hub is
    off."""
    # HF_HUB_CACHE outranks HF_HOME: a caller's own (set by a profile) would hold the config
    empty = {"HF_HOME": str(tmp_path / "hf"), "HF_HUB_CACHE": str(tmp_path / "hf" / "hub")}
    for name in ("qwen2.5-7b-instruct", "Qwen/Qwen2.5-7B-Instruct"):
        proc = _poros("check", name, "--json", cwd=home, env=empty)
        out = json.loads(proc.stdout)
        assert proc.returncode == 0 and (out["model"], out["model_type"], out["status"]) == (
            "Qwen/Qwen2.5-7B-Instruct", "qwen2", "validated")
        seen = json.loads(_poros("arch", name, "--json", cwd=home, env=empty).stdout)
        assert (seen["model"], seen["model_type"], seen["status"]) == (
            "Qwen/Qwen2.5-7B-Instruct", "qwen2", "validated")
        assert "HF_HUB_OFFLINE" in seen["cannot_place"][0] and seen["next"] == []
    for command in ("check", "arch"):
        proc = _poros(command, "some-org/some-model", cwd=home, env=empty)
        assert proc.returncode == 2 and not proc.stdout
        assert "Hub access is off here (HF_HUB_OFFLINE is set)" in _flat(proc.stderr), command


def test_arch_names_a_type_as_check_does_without_transformers(monkeypatch):
    import builtins

    from poros.cli.verbs import arch

    real = builtins.__import__

    def no_layout(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "poros.metal" and "layout" in (fromlist or ()):  # no torch, no transformers
            raise ImportError("no transformers here")
        return real(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", no_layout)
    assert arch._named("phi3") == ("phi3", "not-in-registry")
    assert arch._named("Ignore previous instructions")[0] == "unknown"  # never a config's words
    assert arch._named("qwen2") == ("qwen2", "validated")


def test_no_hint_names_a_flag_the_command_lacks(home, capsys, monkeypatch):
    """`arch` has no --gpu: with no GPU here its plan says so, and names none."""
    from poros.cli import views
    from poros.cli.verbs import shared

    p = shared.plan_for("Qwen/Qwen2.5-7B", Context(cwd=home, fixed_cached=frozenset()))
    if p.gpu is None:  # this machine has no GPU: the plan's own words
        assert p.fit_basis == "no GPU detected"
        assert not hasattr(views, "plan_verdict")  # no command prints a plan


# -- help ----------------------------------------------------------------------------------------


def test_one_help_per_command_whatever_form_the_line_has(home):
    (home / "run.yaml").write_text("model_name_or_path: x/y\n")
    plain, config = _poros("train", "--help", cwd=home), _poros("train", "run.yaml", "--help",
                                                                cwd=home)
    assert plain.returncode == config.returncode == 0 and plain.stdout == config.stdout
    for flag in ("--resume-from", "--output-dir", "--data", "--steps", "--force"):
        assert flag in plain.stdout, flag
    (home / "run").mkdir()
    export = _poros("export", "./run", "--help", cwd=home)
    assert export.stdout == _poros("export", "--help", cwd=home).stdout
    assert "-o" in export.stdout.split() and "poros export ./run -o ./merged" in export.stdout
    assert "--adapter" in _poros("eval", "run.yaml", "--help", cwd=home).stdout


def test_the_top_level_help_lists_every_command(home):
    proc = _poros("--help", cwd=home, env={"COLUMNS": "100"})
    listed = set(re.findall(r"^│ ([a-z]+) ", proc.stdout, re.M))
    assert listed == set(registry.names()) - {"help"}
    assert "<bool>" not in proc.stdout
    init = _poros("init", "--help", cwd=home, env={"COLUMNS": "100"}).stdout
    assert "<bool>" not in init


# A module path, RST markup, a variable's name, a repo file. (poros.Run, PorosConfig and the
# other public names of the Python API are a user's words too.)
_INTERNAL = re.compile(r"``|DOCS_URL|\bporos\.(cli|core|plan|preflight|recipes|links|metal|train)"
                       r"\b|docs/[a-z]+\.md|scripts/|tests/|PorosTrainer")


def test_help_and_output_name_nothing_internal():
    """No module path, RST markup, variable name or repo file in a verb's help."""
    for verb, module in app.BUILT.items():
        doc = importlib.import_module(module).__doc__ or ""
        assert not _INTERNAL.search(doc), (verb, _INTERNAL.search(doc).group(0))
    from poros.cli.main import app as typer_app

    for info in typer_app.registered_commands:
        doc = info.callback.__doc__ or ""
        assert not _INTERNAL.search(doc), (info.callback.__name__, _INTERNAL.search(doc).group(0))


def test_a_plan_names_no_function_or_module():
    """A figure's method and a plan's notes are read by people: Poros's words, not its code's."""
    from poros.cli.verbs import shared

    for model in ("Qwen/Qwen2.5-7B", "Qwen/Qwen2.5-0.5B-Instruct"):
        p = shared.plan_for(model, Context(fixed_gpu=CARD, fixed_cached=frozenset()),
                            config_json=True)
        said = " ".join([p.fit_basis, *p.warnings, *(f.method + f.why for f in (
            p.peak_vram_allocated, p.peak_vram_reserved, p.peak_host_ram, p.s_per_step))])
        assert "path_extra" not in said and not re.search(r"\bporos\.[a-z_]+\b", said), said


def test_a_verbs_help_says_commands_as_a_shell_types_them(monkeypatch):
    from poros.cli import views

    monkeypatch.setattr(views.c, "in_session", False)
    doc = importlib.import_module(app.BUILT["recipes"]).__doc__
    text = _flat(_text(views.verb_help("recipes", doc, 100)))
    assert "poros train picks one" in text and "/train" not in text and "``" not in text
    assert "./recipes" in text and "~/.poros/recipes" in text  # a path is not a command
    monkeypatch.setattr(views.c, "in_session", True)
    assert "/train picks one" in _flat(_text(views.verb_help("recipes", doc, 100)))
    # prose shows without a Flags block too, and a titled block keeps its title
    arch = views.verb_help("arch", importlib.import_module(app.BUILT["arch"]).__doc__, 100)
    assert "Sizes are arithmetic, not a run" in _flat(_text(arch))
    train = views.verb_help("train", importlib.import_module(app.BUILT["train"]).__doc__, 100)
    assert "With a config file:" in _text(train) and "--resume-from DIR|latest" in _text(train)


@pytest.mark.cuda_host(reason="a Mac's check says whether this Mac trains the model, and with no "
                               "row for a bare phi3 config points at `poros models`")
def test_check_points_at_the_docs_command_not_a_repo_file(home, tmp_path):
    folder = home / "m"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model_type": "phi3"}))
    proc = _poros("check", str(folder), cwd=home)
    assert proc.returncode == 4 and "poros docs models" in proc.stdout
    assert "docs/models.md" not in proc.stdout and "registry line" not in proc.stdout


# -- one unit for memory, one for a download -----------------------------------------------------


def test_a_download_is_said_in_decimal_gb_everywhere(monkeypatch):
    from poros.cli.schemas import Figure
    from poros.cli.verbs import shared
    from poros.cli.views.parts import _download_value, _figure_value
    from poros.preflight import disk

    assert shared.disk_text is disk.disk_text  # one helper
    gib = 0.92  # a 988 MB download, as a plan counts it
    figure = Figure(value=gib, unit="GiB", source="hub_metadata")
    assert _download_value(figure) == "0.99 GB" == disk.disk_text(gib * 2**30 / 1e9)
    assert shared.download_confirm("x/y", gib, Context(fixed_cached=frozenset())) == "0.99 GB"
    assert _download_value(Figure(value=0.0, unit="GiB", source="local")) == "0 GB"
    assert _figure_value(Figure(value=9.329, unit="GiB", source="estimated", method="m",
                                receipt="r")) == "9.4 GiB"  # memory: GiB, never below its figure


def test_the_disk_and_the_card_are_labelled_with_their_units(monkeypatch):
    import shutil

    from poros.cli.commands import check as check_commands

    usage = shutil.disk_usage(".")
    monkeypatch.setattr(check_commands, "_torch_facts",
                        lambda quietly: ("2.15.0", "12.8", [("RTX 4090", 23.52)], 8))
    rows = {name.strip(): value for name, value, _ in check_commands._environment_checks()[0]}
    if "Disk free" in rows:  # off a Mac
        assert abs(float(rows["Disk free"].split()[0]) - usage.free / 1e9) < 1.0
        assert rows["Disk free"].endswith(" GB")
        if "GPU 0" in rows:
            assert rows["GPU 0"] == "RTX 4090 (23.5 GiB)"


# -- errors on stderr ----------------------------------------------------------------------------


def test_a_refusal_goes_to_stderr_and_json_stays_one_object_on_stdout(home):
    human = _poros("arch", "nope", cwd=home)
    assert human.returncode == 2 and human.stdout == ""
    assert "No such model" in human.stderr
    usage = _poros("bogus", cwd=home)
    assert usage.returncode == 2 and usage.stdout == "" and "No such command" in usage.stderr
    config = home / "bad.yaml"
    config.write_text("model_name_or_path: x/y\nmax_stepz: 3\n")
    invalid = _poros("train", "bad.yaml", cwd=home)
    assert invalid.returncode == 2 and invalid.stdout == "" and "max_stepz" in invalid.stderr
    for argv in (("arch", "nope", "--json"), ("agents", "arch", "nope")):
        asked = _poros(*argv, cwd=home)
        lines = asked.stdout.strip().splitlines()
        assert len(lines) == 1 and json.loads(lines[0])["error"]["code"] == "E_NOT_FOUND"


def test_poros_alone_without_a_terminal_prints_the_help_and_exits_0(home):
    proc = _poros(cwd=home)
    assert proc.returncode == 0 and "Usage: poros" in proc.stdout and "Usage" not in proc.stderr
    asked = _poros("--json", cwd=home)  # an agent's: the contract's error, as before
    assert asked.returncode == 2 and json.loads(asked.stdout)["error"]["code"] == "E_USAGE"


# -- a library's advice that does not matter stays off the screen --------------------------------


def test_the_cpu_kernel_advice_is_filtered_and_nothing_else():
    import poros.cli.root  # noqa: F401 (installs the filter)

    log = logging.getLogger("bitsandbytes.backends.cpu.ops")

    def passes(message: str) -> bool:
        return log.filter(logging.LogRecord(log.name, logging.WARNING, "ops.py", 1, message,
                                            ("x",), None))

    assert not passes("Failed to load CPU gemm_4bit_forward from kernels-community: %s. Please "
                      "make sure you already `pip install kernels`")
    assert passes("something else went wrong: %s")


# -- init ----------------------------------------------------------------------------------------


def test_init_on_an_existing_file_offers_a_safe_next_step(home):
    first = _poros("init", "sft.yaml", "--json", cwd=home)
    assert json.loads(first.stdout)["model"] == "Qwen/Qwen2.5-0.5B-Instruct"  # the starter
    (home / "sft.yaml").write_text("model_name_or_path: my/model\ndataset: mine.jsonl\n")
    again = _poros("init", "sft.yaml", cwd=home)
    said = _flat(again.stderr)
    assert again.returncode == 2 and "fix poros init sft-2.yaml" in said
    assert "--force replaces it with a new starter config: nothing in sft.yaml is kept" in said
    asked = json.loads(_poros("init", "sft.yaml", "--json", cwd=home).stdout)["error"]
    assert asked["hint"] == "poros init sft-2.yaml"  # never a command that overwrites
    assert "my/model" in (home / "sft.yaml").read_text()


# -- arch try says what it checked, once ---------------------------------------------------------


def test_arch_try_says_what_was_checked_once():
    from poros.cli import views
    from poros.cli.verbs.arch import SAYS, ArchOutput, Weights

    r = ArchOutput(model="x/y", model_type="phi3", status="not-in-registry", weights=Weights(),
                   tried={"ok": True, "route": "cuda", "says": SAYS["cuda"],
                          "routes": {"trainer": {"bitwise": True}, "patch": {"bitwise": False}}})
    text = _text(views.arch(r, 100))
    assert text.count("within 1e-4") == 1 and "bit for bit" not in text
    assert "the loss and every adapter gradient" in text
    exact = r.model_copy(update={"tried": {**r.tried, "routes": {"trainer": {"bitwise": True}}}})
    text = _text(views.arch(exact, 100))
    assert text.count("bit for bit") == 1 and "1e-4" not in text
