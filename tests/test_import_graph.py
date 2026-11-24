"""The shape of poros's imports: no cycles, every module importable first in a fresh interpreter,
and the library layers never reaching into the CLI.

A package's shared pieces live in a leaf module its submodules import (``poros.plan.cells``,
``poros.cli.views.parts``, ``poros.cli.root``, ...); a submodule never imports back from its
package's ``__init__``.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from tests.conftest import needs_ml

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "poros"


def _modules() -> dict[str, Path]:
    out = {}
    for path in PKG.rglob("*.py"):
        rel = path.relative_to(ROOT).with_suffix("")
        if "docs" in rel.parts:
            continue
        name = ".".join(rel.parts)
        out[name.removesuffix(".__init__")] = path
    return out


MODULES = _modules()


def _imports(tree: ast.AST, top_level: bool):
    """(module, names) of each ``import`` / ``from ... import`` of poros (top level, or all)."""
    nodes = tree.body if top_level else ast.walk(tree)
    for n in nodes:
        if isinstance(n, ast.ImportFrom) and n.level == 0 and (n.module or "").startswith("poros"):
            yield n.module, [a.name for a in n.names]
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith("poros"):
                    yield a.name, []


def _graph() -> dict[str, set[str]]:
    """Top-level imports between poros modules. ``from pkg import sub`` counts as importing both
    the package and the submodule, as Python runs the package's ``__init__`` first."""
    graph: dict[str, set[str]] = {m: set() for m in MODULES}
    for m, path in MODULES.items():
        tree = ast.parse(path.read_text())
        for module, names in _imports(tree, top_level=True):
            if module in MODULES:
                graph[m].add(module)
            for name in names:
                if f"{module}.{name}" in MODULES:
                    graph[m].add(f"{module}.{name}")
        graph[m].discard(m)
    return graph


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on: set[str] = set()
    found: list[list[str]] = []
    counter = [0]
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))

    def visit(v: str) -> None:
        index[v] = low[v] = counter[0]
        counter[0] += 1
        stack.append(v)
        on.add(v)
        for w in graph[v]:
            if w not in index:
                visit(w)
                low[v] = min(low[v], low[w])
            elif w in on:
                low[v] = min(low[v], index[w])
        if low[v] == index[v]:
            comp = []
            while True:
                w = stack.pop()
                on.discard(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) > 1:
                found.append(sorted(comp))

    for v in graph:
        if v not in index:
            visit(v)
    return found


def test_no_import_cycles():
    """A cycle makes import order part of correctness: moving one line breaks the import."""
    assert _cycles(_graph()) == []


# Packages whose code is the library: none of them imports the CLI, at the top or inside a
# function. (poros.metal renders the Mac's prove screen through the CLI today and poros.jobs
# runs the CLI's background jobs: they are not on this list.)
LIBRARY = ("poros.core", "poros.plan", "poros.preflight", "poros.train", "poros.moe",
           "poros.sss", "poros.patch", "poros.adapters",
           "poros.validate", "poros.recipes", "poros.bench", "poros.report", "poros.data",
           "poros.runs", "poros.losses", "poros.stores", "poros.ends", "poros.links")
# The library modules the Mac's own work still ties to the CLI (poros/preflight/mac.py is the Mac
# release's; its CLI imports move with that work).
MAC_OWNED = {"poros.preflight.mac"}


def test_the_library_never_imports_the_cli():
    reached = []
    for m, path in MODULES.items():
        if not m.startswith(LIBRARY) or m in MAC_OWNED:
            continue
        for module, _ in _imports(ast.parse(path.read_text()), top_level=False):
            if module == "poros.cli" or module.startswith("poros.cli."):
                reached.append(f"{m} -> {module}")
    assert reached == []


def _needs_absent(path: Path) -> bool:
    """A module that imports an optional dependency this machine lacks at its top (triton off
    CUDA): it cannot import here, by design."""
    for n in ast.parse(path.read_text()).body:
        names = ([a.name for a in n.names] if isinstance(n, ast.Import)
                 else [n.module or ""] if isinstance(n, ast.ImportFrom) and n.level == 0 else [])
        for name in names:
            top = name.split(".")[0]
            if top in ("triton", "mlx") and importlib.util.find_spec(top) is None:
                return True
    return False


@needs_ml
@pytest.mark.slow
def test_every_module_imports_first_in_a_fresh_interpreter():
    """Whichever module a script imports first, it imports: no partly initialized package."""
    env = {**os.environ, "PYTHONPATH": str(ROOT), "HF_HUB_OFFLINE": "1"}
    names = sorted(m for m, p in MODULES.items() if not _needs_absent(p))

    def first(m: str):
        proc = subprocess.run([sys.executable, "-c", f"import {m}"], cwd=ROOT, env=env,
                              capture_output=True, text=True, timeout=300)
        return m, proc.returncode, proc.stderr.strip().splitlines()[-1:] if proc.returncode else []

    with ThreadPoolExecutor(max_workers=4) as pool:
        failed = [(m, err) for m, rc, err in pool.map(first, names) if rc]
    assert failed == []
