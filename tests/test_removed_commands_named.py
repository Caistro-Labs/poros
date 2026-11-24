"""No text in the package names a removed command (``poros fit``, ``poros plan``, ``poros agents
fit``) or its docs anchor (``#poros-fit``, ``#poros-plan``) as if it still ran.

Every string constant in the package's source is read with ``ast`` (f-string parts included,
docstrings too), and every comment with ``tokenize``. A hit fails with its file and line. The
stubs that tell a user where a removed command went are the only text allowed to name it.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

import poros

PACKAGE = Path(poros.__file__).resolve().parent
REMOVED = re.compile(r"\bporos\s+(?:agents\s+)?(?:fit|plan)\b|#poros-(?:fit|plan)\b")
# (file, the text that names a removed command): the stubs that say where it went
ALLOWED: set = set()


def _sources():
    for path in sorted(PACKAGE.rglob("*.py")):
        yield path.relative_to(PACKAGE.parent).as_posix(), path.read_text(encoding="utf-8")


def _strings(text: str):
    """(line, text) of every string constant, f-string parts included."""
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def _comments(text: str):
    """(line, text) of every comment."""
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT:
            yield tok.start[0], tok.string


def _hits(read) -> list:
    hits = []
    for name, text in _sources():
        for line, said in read(text):
            for m in REMOVED.finditer(said):
                if (name, m.group(0)) not in ALLOWED:
                    at = line + said[:m.start()].count("\n")  # the line within a docstring
                    hits.append(f"{name}:{at}: {m.group(0)!r} in "
                                f"{said[max(0, m.start() - 40):m.end() + 40]!r}")
    return hits


def test_no_string_names_a_removed_command():
    hits = _hits(_strings)
    assert not hits, "\n".join(hits)


def test_no_comment_names_a_removed_command():
    hits = _hits(_comments)
    assert not hits, "\n".join(hits)


def test_the_pattern_finds_what_it_is_for():
    """The check itself: each removed name and anchor is found, and a module path is not."""
    for said in ("poros plan <config>", "``poros fit``", "poros agents fit", "poros  plan --gpu",
                 "docs/cli.md#poros-plan", "docs/cli.md#poros-fit"):
        assert REMOVED.search(said), said
    for said in ("poros.plan.make", "poros planned", "poros/plan/cells.json", "poros fitness",
                 "poros train", "poros recipes", "poros.preflight.fit"):
        assert not REMOVED.search(said), said


# -- no planned figure in what a run says ---------------------------------------------------------

# Built only inside the plan (poros/plan, the Mac's metal/plan.py), or a words-only reason: the
# f-strings allowed to read a plan's basis. Anywhere else a basis would carry its figures out.
BASIS_OK = {("poros/core/errors.py", "basis"), ("poros/preflight/fit.py", "basis")}
PLAN_FIELDS = {"fit_basis", "host_fit_basis", "peak_vram_allocated", "peak_vram_reserved",
               "peak_host_ram", "host_floor_gib", "need_gib", "ssd_gib", "basis"}


def test_no_text_formats_a_plans_figure_or_basis():
    hits = []
    for name, text in _sources():
        if name.startswith("poros/plan/") or name == "poros/metal/plan.py":
            continue
        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.FormattedValue):
                continue
            said = ({x.attr for x in ast.walk(node.value) if isinstance(x, ast.Attribute)}
                    | {x.id for x in ast.walk(node.value) if isinstance(x, ast.Name)})
            for field in sorted(said & PLAN_FIELDS):
                if (name, field) not in BASIS_OK:
                    hits.append(f"{name}:{node.lineno}: {field}")
    assert not hits, "\n".join(hits)


_GIB = re.compile(r"(\d+(?:\.\d+)?) GiB")


def _fit_texts(model, gpu_free, host_free, **settings):
    """Every sentence the check before a download says for this run: the path and tables
    reasons, the warning, each refusal's message and fix; and the facts measured here."""
    from poros.core.config import PorosConfig
    from poros.core.schemas import PlanGpu
    from poros.preflight import fit
    from tests.test_plan_upper import CONFIGS

    gpu = PlanGpu(name="RTX 4090", total_gib=23.99, free_gib=gpu_free, source="spec")
    cfg = PorosConfig(model_name_or_path=model, dataset="synthetic", lora_dropout=0.0, **settings)
    mt = CONFIGS[model]["model_type"]
    found = fit.check(cfg, fit.run_planner(mt, gpu=gpu, model_config=CONFIGS[model],
                                           host_free_gib=host_free))
    said = [found.path[1] if found.path else "", found.offload or "", found.warning or "",
            cfg.path_reason or "", cfg.offload_reason or ""]
    for err in (found.vram, found.host):
        if err is not None:
            said += [err.info.message, err.info.remediation]
    return said, {gpu_free, 23.99, host_free}


@pytest.mark.parametrize("model, gpu_free, host_free, settings", [
    ("Qwen/Qwen2.5-72B", 23.5, 30.35, {"max_seq_length": 512}),  # host RAM refuses
    ("Qwen/Qwen2.5-7B", 2.0, 60.0, {"max_seq_length": 2048}),  # the GPU refuses
    ("Qwen/Qwen2.5-32B", 23.5, 60.0, {"max_seq_length": 4096}),  # paths and tables adapt
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", 23.5, 14.0, {"max_seq_length": 1024}),  # SSD advice
])
def test_the_check_says_no_planned_figure(model, gpu_free, host_free, settings):
    """A GiB figure in anything the check says is one measured here (free VRAM, the card, free
    host RAM), never the plan's peak or estimate."""
    said, measured = _fit_texts(model, gpu_free, host_free, **settings)
    for text in said:
        for x in _GIB.findall(text):
            assert float(x) in measured, (x, text)


def test_a_model_picker_says_no_plan_verdict():
    """The model choices (a question's, a sheet's) say what a model is, never "fits" or "won't
    fit" from the plan: what trains where is a certified recipe's."""
    from poros.cli.context import Context
    from poros.cli.verbs import shared
    from poros.core.schemas import PlanGpu

    for gpu in (PlanGpu(name="RTX 4090", total_gib=23.99, free_gib=None, source="spec"),
                PlanGpu(name="RTX 4060", total_gib=8.0, free_gib=None, source="spec")):
        choices = shared.model_choices(Context(fixed_gpu=gpu, fixed_cached=frozenset()))
        assert choices
        for ch in choices:
            assert "fit" not in ch.note and "GiB" not in ch.note, (ch.label, ch.note)


_PROMISES_A_FIT = re.compile(r"\bfit(s|ting)?\b|\bwhat fits\b", re.I)


def test_no_command_description_promises_a_fit():
    """What a command is said to do (poros --help, the registry, the shell's list) never says it
    tells what fits: what trains where is a certified recipe's (poros recipes)."""
    from typer.testing import CliRunner

    from poros.cli import main, registry, shell

    said = [(c.name, c.about) for c in registry.COMMANDS if c.about]
    said += [(name, f"{args} {desc}") for name, (args, desc, _) in shell.COMMANDS.items()]
    said += [("--help", CliRunner().invoke(main.app, ["--help"]).output)]
    hits = [f"{name}: {text.strip()[:120]!r}" for name, text in said
            if _PROMISES_A_FIT.search(text)]
    assert not hits, "\n".join(hits)


# -- the check says what it decided and refused, never that something fits ----------------------

_ASSERTS_A_FIT = re.compile(r"\b(fits|should fit|estimated to fit|will fit)\b", re.I)
_FIT_STEM = re.compile(r"\bfit(s|ting)?\b", re.I)
# The only string constants (docstrings aside) that may carry the word stem fit, fits or
# fitting: each one exact string, and why it is not the planner asserting a fit.
FIT_WORDS_OK = {
    "docs/training.md#what-fits": "the docs anchor of the section headed What fits",
    "fit": "a removed command's name, a verb word, the train card's row label",
    "fits": "the row word of a certified run of these settings on this machine; fit_note's token",
    "Won't fit": "a refusal's title: what the check refused",
    "won't fit": "a refusal: what the check refused",
    "won't fit host RAM": "a refusal: what the check refused",
    " won't fit ": "a refusal's head: what the check refused",
    "may not fit": "a warning's head: an estimate past the card, never a fit",
    "may not fit: %s": "the trainer's warning: an estimate past the card",
    "it may not fit the GPU at these settings, and could stop at its first step":
        "the gate's warning: an estimate past the card",
    "This run may not fit ": "the card's question: an estimate past the card",
    ". The GPU may not fit it either": "the card's question: a second refusal",
    ". Start anyway to try it: it stops with an out-of-memory error if it does not fit":
        "what --force does",
    "big downloads may not fit": "a disk fact: the free disk measured here",
    "free disk may not fit this model plus its cache": "a disk fact: the free disk measured here",
    "Resident reference did not fit in VRAM; falling back to a determinism-only self-check. "
    "Full resident-vs-blockwise parity for this model size is validated by the GPU CI / "
    "benchmark grid.": "a measured fact: the reference ran out of memory",
    " chunks cannot fit a ring of ": "a measured fact: the ring's bytes",
    " does not fit ": "Inkling 2 Mini's codec: a byte count",
    # inside the plan: fit_basis and a model_copy key, which no output formats (checked above)
    " GiB, may not fit ": "the plan's own basis, never printed",
    " GiB fits ": "the plan's own basis, never printed",
    "with the embedding table and LM head in host RAM it would not fit (":
        "the plan's own basis, never printed",
    ", the shortest the plan reads, so it fits at no length":
        "at_no_length's basis: kept on the error for code, never in its message",
}
FIT_FILES_OK = {"poros/bench/baselines/accelerate_offload.py": "a code template a bench runs"}


def _docstrings(tree) -> set:
    out = set()
    for node in ast.walk(tree):
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)):
            out.add(id(node.body[0].value))
    return out


def test_no_string_says_fit_but_the_allowed_ones():
    """Outside FIT_WORDS_OK, each entry one exact string with its reason, no string the package
    can say (its docstrings aside) carries fit, fits or fitting: the check says what it decided
    and refused; a fit is a certified run's own word."""
    hits = []
    for name, text in _sources():
        if name in FIT_FILES_OK:
            continue
        tree = ast.parse(text)
        docs = _docstrings(tree)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docs and _FIT_STEM.search(node.value)
                    and node.value not in FIT_WORDS_OK):
                hits.append(f"{name}:{node.lineno}: {node.value[:100]!r}")
    assert not hits, "\n".join(hits)


# Lines of the shipped docs allowed the stem, each exact, with why.
DOC_FIT_OK = {
    "## What fits": "the section's heading (its anchor is linked from every page)",
    "- [Training](training.md): objectives, data, what fits, checkpoints and resume.":
        "a link to that section, by its name",
    "- [Training](training.md): objectives, data, what fits, checkpoints and resume":
        "a link to that section, by its name",
    "- [Training](training.md): objectives, data, checkpoints and what fits.":
        "a link to that section, by its name",
    "measured. ([What fits](training.md#what-fits).)": "a link to that section, by its name",
    "itself. [Training paths](training.md#training-paths) and [What fits](training.md#what-fits)"
    " say": "a link to that section, by its name",
    "- [What fits](training.md#what-fits): what a run checks before it starts.":
        "a link to that section, by its name",
    "terminal Poros offers the retry. [What fits](training.md#what-fits) lists the other ways.":
        "a link to that section, by its name",
    "- [What fits](training.md#what-fits): what the check refuses, and what to change.":
        "a link to that section, by its name",
    "| 3 | won't fit | `E_OOM_PREDICTED`, `E_HOST_OOM_PREDICTED`, `E_OOM` |":
        "exit code 3's name: a refusal",
    "0 success; 1 a Poros bug; 2 usage, config or an answer needed; 3 won't fit; 4 unsupported, "
    "or a": "exit code 3's name: a refusal",
    "3 won't fit, 4 unsupported (a pre-Ampere GPU: `E_GPU_UNSUPPORTED`), 5 environment,":
        "exit code 3's name: a refusal",
}


def test_no_shipped_doc_says_fit_but_the_allowed_lines():
    """The docs pages and the agent guide the package ships: the stem fit only on DOC_FIT_OK's
    lines (a heading and links by name, an exit code's name) and in code (``check_fit``)."""
    hits = []
    for name, text in _shipped_text():
        if not name.endswith(".md"):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            bare = re.sub(r"`[^`]*`", "", line)  # code: check_fit=, a setting's name
            if _FIT_STEM.search(bare) and line.strip() not in DOC_FIT_OK:
                hits.append(f"{name}:{i}: {line.strip()[:110]!r}")
    assert not hits, "\n".join(hits)


@pytest.mark.parametrize("model, gpu_free, host_free, settings", [
    ("Qwen/Qwen2.5-72B", 23.5, 30.35, {"max_seq_length": 512}),
    ("Qwen/Qwen2.5-7B", 2.0, 60.0, {"max_seq_length": 2048}),
    ("Qwen/Qwen2.5-7B", 7.0, 60.0, {"max_seq_length": 2048}),
    ("Qwen/Qwen2.5-32B", 23.5, 60.0, {"max_seq_length": 4096}),
    ("Qwen/Qwen2.5-32B", 23.5, 60.0, {"max_seq_length": 512}),
    ("Qwen/Qwen3-30B-A3B-Instruct-2507", 23.5, 14.0, {"max_seq_length": 1024}),
])
def test_what_the_check_says_asserts_no_fit(model, gpu_free, host_free, settings):
    said, _ = _fit_texts(model, gpu_free, host_free, **settings)
    for text in said:
        assert not _ASSERTS_A_FIT.search(text), text


# The text files the package ships (pyproject's package-data, links followed: the docs pages and
# the agent guide among them), by suffix.
SHIPPED_TEXT = (".toml", ".json", ".md", ".yaml", ".yml", ".txt")
# The stub that tells a reader where a removed command went: its heading in docs/cli.md, whose
# section says "Not in this release" and names poros train and poros recipes.
SHIPPED_OK = {("poros/docs/cli.md", "### poros plan")}


def _shipped_text():
    import os

    paths = []
    for top, _, files in os.walk(PACKAGE):  # the package's own folders
        paths += [Path(top) / name for name in files]
    # the linked folders package-data ships one level of: docs/*.md and agent_docs/AGENTS.md
    paths += sorted((PACKAGE / "docs").glob("*.md")) + [PACKAGE / "agent_docs" / "AGENTS.md"]
    for path in sorted(set(paths)):
        if path.suffix in SHIPPED_TEXT and path.is_file():
            yield (Path(os.path.relpath(path, PACKAGE.parent)).as_posix(),
                   path.read_text(encoding="utf-8"))


def test_no_shipped_text_file_names_a_removed_command():
    """Every toml, json, md, yaml and txt file the package ships, comments included."""
    hits, seen = [], 0
    for name, text in _shipped_text():
        seen += 1
        for i, line in enumerate(text.splitlines(), 1):
            for m in REMOVED.finditer(line):
                if (name, line.strip()) not in SHIPPED_OK:
                    hits.append(f"{name}:{i}: {line.strip()[:100]!r}")
    assert seen >= 10, seen  # catalog.toml, the plan's json, certified.json, docs, agents.md
    assert not hits, "\n".join(hits)


def test_no_verbs_help_says_fit():
    """A verb's help (poros <verb> --help) is its module's docstring, which the string test
    above leaves out with every docstring: each is read here for the stem fit."""
    hits = []
    for path in sorted((PACKAGE / "cli" / "verbs").glob("*.py")):
        said = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
        for i, line in enumerate(said.splitlines(), 1):
            if _FIT_STEM.search(line):
                hits.append(f"{path.name}:{i}: {line.strip()!r}")
    assert not hits, "\n".join(hits)
