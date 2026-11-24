"""What the wheel carries: every file the package reads at run time, and no receipts as code."""

from __future__ import annotations

import fnmatch
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
needs_checkout = pytest.mark.skipif(not PYPROJECT.is_file(), reason="needs the source checkout")


def _setuptools() -> dict:
    return tomllib.loads(PYPROJECT.read_text())["tool"]["setuptools"]


def _shipped(rel: Path, data: dict) -> bool:
    """Whether package data ships ``rel`` (a path under poros/): a glob of the nearest package
    above it that names it."""
    parts = rel.parts
    for cut in range(len(parts) - 1, 0, -1):
        package = ".".join(parts[:cut])
        inner = "/".join(parts[cut:])
        if any(fnmatch.fnmatchcase(inner, g) and inner.count("/") == g.count("/")
               for g in data.get(package, ())):
            return True
    return False


@needs_checkout
def test_every_data_file_in_the_package_ships():
    """A file beside the code that is not Python reaches a pip install only through package data:
    the grouped-expert pins (PIN_DIR) among them, without which that path refuses from PyPI."""
    data = _setuptools()["package-data"]
    pkg = ROOT / "poros"
    missing = []
    for path in sorted(pkg.rglob("*")):
        rel = path.relative_to(ROOT)
        if rel.parts[1] == "docs" or "__pycache__" in rel.parts or not path.is_file():
            continue
        if path.suffix in (".py", ".pyc", ".so", ".pyd"):
            continue
        if not _shipped(rel, data):
            missing.append(str(rel))
    assert missing == []


@needs_checkout
def test_the_pins_each_runtime_reads_are_package_data():
    pytest.importorskip("torch")
    from poros.moe import grouped

    data = _setuptools()["package-data"]
    for module in (grouped,):
        pins = sorted(module.PIN_DIR.glob("*.json"))
        assert pins, module.__name__
        for pin in pins:
            assert _shipped(pin.resolve().relative_to(ROOT), data), pin


@needs_checkout
def test_the_docs_link_ships_markdown_not_packages():
    """poros/docs links to ../docs; package discovery would follow it and ship every receipt
    script under docs/validation as an importable module."""
    find = _setuptools()["packages"]["find"]
    assert any(fnmatch.fnmatchcase("poros.docs.validation.fused.scripts", p)
               for p in find["exclude"])


@needs_checkout
def test_the_sdist_ships_no_tests_and_the_wheel_none_either():
    """The sdist ships no tests: they need the repository (its docs, configs and receipts), so
    none ship, by MANIFEST.in (not setuptools' default tests/test*.py, a part of them with no
    conftest.py), and no tests package is found."""
    manifest = (ROOT / "MANIFEST.in").read_text().split("\n")
    assert "prune tests" in manifest
    assert _setuptools()["packages"]["find"]["include"] == ["poros*"]


# The versions the RTX 4090 validation runs used (docs/validation/int/4090: torch 2.14.1+cu130,
# transformers 5.22.2, peft 0.20.0, bitsandbytes 0.50.2), with the accelerate, safetensors and
# datasets the CPU suites run beside them.
GATED = {"torch": "2.14.1", "transformers": "5.22.2", "peft": "0.20.0", "bitsandbytes": "0.50.2",
         "accelerate": "1.15.0", "safetensors": "0.8.0", "datasets": "5.1.0"}
# The next release of each, which no validation run has used.
NEXT = {"torch": "2.15.0", "transformers": "5.23.0", "peft": "0.21.0", "bitsandbytes": "0.51.0",
        "accelerate": "1.16.0", "safetensors": "0.9.0", "datasets": "5.2.0"}
MAC = ["torch==2.15.0", "transformers==5.20.2", "peft==0.20.0", "accelerate==1.15.0",
       "safetensors==0.8.0", "datasets==5.1.0"]


def _ml(platform: str) -> dict:
    """The training stack's requirements on ``platform`` (sys_platform), by name: the
    dependencies with a platform marker."""
    from packaging.requirements import Requirement

    deps = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    reqs = [Requirement(line) for line in deps if ";" in line]
    here = [r for r in reqs if r.marker is None or r.marker.evaluate({"sys_platform": platform})]
    assert len({r.name for r in here}) == len(here), "one line a package, on a platform"
    return {r.name: r.specifier for r in here}


@needs_checkout
def test_ml_on_linux_installs_what_the_gates_ran_and_nothing_newer():
    """Every package of the training stack is bounded above on Linux: the tested versions are in,
    and the next minor is out until a validation run has used it."""
    linux = _ml("linux")
    assert set(linux) == set(GATED)
    for name, spec in linux.items():
        assert spec.contains(GATED[name]), (name, str(spec))
        assert not spec.contains(NEXT[name]), (name, str(spec))
    assert linux["torch"].contains("2.10.0") and linux["transformers"].contains("5.20.2")


@needs_checkout
def test_a_mac_keeps_its_one_exact_stack():
    lines = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    assert [line.split(";")[0].strip() for line in lines if "== 'darwin'" in line] == MAC
    mac = _ml("darwin")
    assert {f"{name}{spec}" for name, spec in mac.items()} == set(MAC)


@needs_checkout
def test_the_install_holds_transformers_to_the_range_the_moe_tests_pass_on():
    linux = _ml("linux")["transformers"]
    arch = (ROOT / "poros" / "moe" / "arch.py").read_text()  # read, not imported: torch-free
    assert "TRANSFORMERS_TESTED: tuple[tuple[int, int], tuple[int, int]] = ((5, 13), (5, 22))" in (
        arch)
    assert not linux.contains("5.23.0") and linux.contains("5.22.2")
    assert "moe" not in tomllib.loads(PYPROJECT.read_text())["project"]["optional-dependencies"]


@needs_checkout
def test_typer_is_bounded_to_the_minor_the_cli_tests_ran():
    """Three typer minors broke --help (pyproject's comment); the next is out until a run has it."""
    from packaging.requirements import Requirement

    deps = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    typer = next(Requirement(d) for d in deps if d.startswith("typer"))
    assert "0.29.1" in typer.specifier and "0.30.0" not in typer.specifier
