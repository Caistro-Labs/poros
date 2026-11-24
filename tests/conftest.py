import ast
import functools
import importlib.util
import os
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest

CHECKOUT = str(Path(__file__).resolve().parents[1])

# The training stack (torch, transformers): a test that trains, loads a model or a tokenizer, or
# reads torch's build skips
# cleanly where it is not installed (a torch-free install runs the rest of the suite).
HAS_ML = all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))
needs_ml = pytest.mark.skipif(not HAS_ML,
                              reason="needs torch and transformers (not a light install)")
# The SSD store's reader (poros.sss) drops what it read with os.posix_fadvise, which macOS has not:
# a Mac keeps its own pack (poros.metal), so a test of that reader skips where the call is missing.
needs_fadvise = pytest.mark.skipif(not hasattr(os, "posix_fadvise"),
                                   reason="the SSD store's reader needs os.posix_fadvise, which "
                                          "macOS has not: a Mac keeps its own pack")


def pytest_configure(config):
    """Every run is offline and leaves the user's own Poros home alone, whoever invokes pytest:
    HF_HUB_OFFLINE=1 (the configs tests read come from the Hugging Face cache already here), and
    POROS_HOME a fresh folder unless the caller named one. Subprocesses inherit both.

    The help reads as a pipe gets it, wherever pytest runs: typer forces a terminal, and so
    colour, whenever GITHUB_ACTIONS, FORCE_COLOR or PY_COLORS is set (typer.rich_utils,
    FORCE_TERMINAL, read once at import), and GitHub's runners set GITHUB_ACTIONS. Dropped here,
    before any test imports typer, for this process and every subprocess."""
    for variable in ("GITHUB_ACTIONS", "FORCE_COLOR", "PY_COLORS", "CLICOLOR_FORCE"):
        os.environ.pop(variable, None)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if "POROS_HOME" not in os.environ:
        os.environ["POROS_HOME"] = tempfile.mkdtemp(prefix="poros-test-home-")
    config.addinivalue_line("markers", "cuda_host: tests a CUDA host's route (faking its "
                            "card on the CPU); a Mac takes its own route, so it skips them")


# A test that makes this host a CUDA one (torch.cuda.is_available -> True) tests that route; on
# a Mac the CLI and the trainer take the Mac's instead, whatever torch is told.
_FAKES_CUDA = re.compile(r"is_available[\"']?,?\s*(=\s*)?lambda[^:]*:\s*True")


@functools.lru_cache(maxsize=None)
def _faking_functions(path: str) -> frozenset:
    """The functions of a test module that fake a CUDA host, themselves or through a helper or
    fixture of the same module."""
    tree = ast.parse(Path(path).read_text())
    funcs = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    src = Path(path).read_text().split("\n")
    direct = {name for name, n in funcs.items()
              if _FAKES_CUDA.search("\n".join(src[n.lineno - 1:n.end_lineno]))}
    calls = {name: {x.id for x in ast.walk(n) if isinstance(x, ast.Name) and x.id in funcs}
             | {a.arg for a in n.args.args if a.arg in funcs}
             for name, n in funcs.items()}
    fakes, changed = set(direct), True
    while changed:
        changed = False
        for name, used in calls.items():
            if name not in fakes and used & fakes:
                fakes.add(name)
                changed = True
    return frozenset(fakes)


def pytest_collection_modifyitems(config, items):
    """Capability marks, so ``pytest tests`` collects and skips cleanly on a Mac: a test that
    fakes a CUDA host is marked ``cuda_host`` here, and skipped there. (A test that needs a
    CUDA device skips itself where there is none, and carries ``gpu`` for the GPU lane.)
    ``@pytest.mark.cuda_host(reason="...")`` says what the Mac does instead, as the skip's
    reason."""
    for item in items:
        fn = getattr(item, "function", None)
        if fn is not None and str(item.path).endswith(".py") and "/tests/metal/" not in str(
                item.path) and fn.__name__ in _faking_functions(str(item.path)):
            item.add_marker(pytest.mark.cuda_host)
        mark = item.get_closest_marker("cuda_host")
        if mark is not None and sys.platform == "darwin":
            item.add_marker(pytest.mark.skip(reason=mark.kwargs.get(
                "reason", "a CUDA host's route: a Mac takes its own")))


class _LinuxSys:
    """``sys`` as a module reads it on Linux: its ``platform`` is "linux", the rest is sys's."""

    platform = "linux"

    def __getattr__(self, name):
        return getattr(sys, name)


def as_linux(monkeypatch, *modules) -> None:
    """Route these modules' own platform checks to Linux's, on any host: a test of a CUDA host's
    reader (nvidia-smi, the cgroup, the Linux store rule) runs on a Mac too. Off a Mac it
    changes nothing."""
    for module in modules:
        monkeypatch.setattr(module, "sys", _LinuxSys())


def another_machine(monkeypatch, tools: Optional[Path] = None) -> None:
    """The machine a test fakes is the only one it sees, whatever card the host has. nvidia-smi is
    ``tools``/nvidia-smi (a folder the test fills, now or later) or nowhere: never one on PATH,
    nor the one ``poros.core.env.outside_tool`` falls back to in SYSTEM_BIN (/usr/bin), which a
    GPU server has whatever PATH says. And no driver library answers: CUDA's numbering and each
    device's capability say nothing unless the test says them (``card._cuda_order``,
    ``card._cuda_capabilities``). Every nvidia-smi query Poros makes goes through
    ``poros.core.card.nvidia_smi``, which asks ``card.outside_tool``; a subprocess finds its own."""
    from poros.core import card

    real = card.outside_tool
    nowhere = Path(os.devnull) / "no-nvidia-smi"  # under a file: no folder can ever be there
    monkeypatch.setattr(card, "outside_tool", lambda name: str(
        (tools if tools is not None else nowhere) / name) if name == "nvidia-smi" else real(name))
    monkeypatch.setattr(card, "_cuda_order", lambda: None)
    monkeypatch.setattr(card, "_cuda_capabilities", lambda: {})


def weighing(monkeypatch, folder: Path, gb: float) -> None:
    """``folder``'s shards weigh ``gb`` (decimal) where a Mac's table reads a local checkpoint's
    weights (``poros.metal.table._weights_gb``), and take no disk: a sparse file of that size is
    refused by a filesystem with less room than that (an overlay, as containers have), sparse or
    not. The folder holds a real, empty model.safetensors."""
    from poros.metal import table

    (folder / "model.safetensors").write_bytes(b"")
    real, mine = table._weights_gb, Path(folder).resolve()
    monkeypatch.setattr(table, "_weights_gb",
                        lambda checkpoint: gb if Path(checkpoint).resolve() == mine
                        else real(checkpoint))


@functools.lru_cache(maxsize=None)
def imported_on_this_platform() -> None:
    """What a Mac's path imports the first time it reads a model's layout (``poros.metal.layout``:
    transformers' model classes and what they pull in, psutil, torch's compiler, triton where it is
    installed), imported once on this host's own platform. A test that fakes a Mac sets
    ``sys.platform`` for Poros's own checks, and a library imported for the first time under that
    fake takes macOS's branch at import and keeps it for the rest of the run: psutil picks its
    macOS backend and fails, and which others do depends on what the host has installed."""
    try:
        import psutil  # noqa: F401  its backend is picked at first import
    except ImportError:
        pass
    if not HAS_ML:
        return
    from poros.metal import layout

    tiny = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 1,
            "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": 128}
    for raw in ({"model_type": "llama", **tiny},
                {"model_type": "qwen3_next", **tiny, "num_experts": 2, "num_experts_per_tok": 1}):
        layout.layout_of(raw)


def machine_exit_here() -> int:
    """`poros machine`'s exit on this host, whatever card a test previews: 0, but on a Mac whose
    training stack is not installed at its pins (a light install, say) 5: not ready to train, as the
    Mac's own checks say. (Off a Mac a missing torch is a warning, and machine exits 0.)"""
    if sys.platform != "darwin":
        return 0
    from poros.core.errors import EXIT_CODES
    from poros.metal import _stack_problems

    return EXIT_CODES["E_MACHINE_NOT_READY"] if _stack_problems() else 0


def roomy_disk_on_a_mac(monkeypatch) -> None:
    """On a Mac, the disk its store's rule reads (the model's size and 10% of the drive left free)
    has room, whatever this machine's drive holds: for a test of what a Mac does with a model, not
    of its drive. Off a Mac it changes nothing."""
    if sys.platform != "darwin":
        return
    import shutil

    roomy = shutil._ntuple_diskusage(total=10**12, used=0, free=10**12)
    monkeypatch.setattr(shutil, "disk_usage", lambda path: roomy)


def _shipped_facts() -> dict:
    import json

    return json.loads((Path(CHECKOUT) / "poros" / "plan" / "config_facts.json").read_text())


def hub_holding_configs(monkeypatch, hub: Path, *models: str, weights: bool = False) -> Path:
    """A Hugging Face cache at ``hub`` holding only these models' config.json, as the snapshot
    Poros's shipped facts (poros/plan/config_facts.json) came from, and the Hub read from it (in
    this process and, through HF_HUB_CACHE, in its subprocesses): a test of what Poros says once
    a model's config.json is here (downloaded, or fetched online) answers the same on a machine
    whose own cache never held it, as a CI runner's. ``weights``: a model.safetensors too (an
    empty one), so the model counts as downloaded; nothing reads a tensor of it."""
    import json

    from huggingface_hub import constants

    facts = _shipped_facts()["models"]
    for model in models:
        entry = facts[model]
        repo = hub / f"models--{model.replace('/', '--')}"
        snapshot = repo / "snapshots" / entry["revision"]
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text(json.dumps(entry["fields"]))
        if weights:
            header = b"{}"
            (snapshot / "model.safetensors").write_bytes(len(header).to_bytes(8, "little")
                                                         + header)
        (repo / "refs").mkdir()
        (repo / "refs" / "main").write_text(entry["revision"])
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(hub))
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    return hub


@pytest.fixture
def recipe_configs_cached(tmp_path, monkeypatch) -> Path:
    """Every model a built-in recipe names has its config.json in the Hub cache, and nothing else
    is there: what these tests assumed of the machine that ran them (a developer's cache), made
    the test's own. With HF_HUB_OFFLINE set (pytest_configure) and no config.json, a train is
    refused offline before its gate, and fit and the train card plan no model."""
    return hub_holding_configs(monkeypatch, tmp_path / "recipe-hub", *_shipped_facts()["models"])


def checkout_env(env=None) -> dict:
    """``env`` (default: this process's) with this checkout first on PYTHONPATH: a subprocess run
    as ``python -P`` then imports the poros under test, not an installed one."""
    env = dict(os.environ if env is None else env)
    env["PYTHONPATH"] = os.pathsep.join([CHECKOUT, *filter(None, [env.get("PYTHONPATH")])])
    return env


ROLE = {"system": 10, "user": 11, "assistant": 12}


class Tok:
    """Byte-level fake tokenizer: ``<turn> <role> content <end>``; ids stay under 256."""

    chat_template = "fake"
    eos_token_id = 1
    pad_token_id = 0
    bos_token_id = None

    def enc(self, s):
        return [20 + ord(c) % 200 for c in s]

    def __call__(self, s, add_special_tokens=True):
        return {"input_ids": self.enc(s)}

    def apply_chat_template(self, turns, tokenize=True, add_generation_prompt=False,
                            return_dict=False):
        ids = []
        for t in turns:
            ids += [2, ROLE[t["role"]]] + self.enc(t["content"]) + [1]
        if add_generation_prompt:
            ids += [2, ROLE["assistant"]]
        return ids


@pytest.fixture
def cpu_only(monkeypatch):
    """Run a CPU-path test as on a host without a GPU.

    With CUDA present the engine streams to it, so a test that builds its
    model and inputs on the CPU would mix devices.
    """
    try:
        import torch
    except ImportError:  # no torch, no GPU to hide
        return
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


# The receipts the card tests are written against, by id: Qwen2.5-7B's dense runs on an RTX 5090
# and an RTX 3090 at 512 tokens. The simulated card is an RTX 4090 none of them ran on, so a
# length is "measured" only where one of these matches it.
CARD_RECEIPTS = ("qwen2.5-7b-rtx5090-synthetic", "qwen2.5-7b-rtx5090-synthetic-streamed",
                 "qwen2.5-7b-rtx3090-alpaca")


@pytest.fixture
def path_receipts(monkeypatch):
    """Only ``CARD_RECEIPTS``, as if each had also run block-major accumulation's efficient path
    at the recipes' split (the loop a CUDA run of the recipes' settings takes: block-major holds
    every micro-batch of a step, so a step of fewer would bound nothing) and each with the fused
    compute (a CUDA run's default) as well: for tests of what the CLI does with a measured fit,
    whatever receipts the release adds."""
    import poros.plan.cells

    cells = [c for c in poros.plan.cells.CELLS if c["id"] in CARD_RECEIPTS]
    assert len(cells) == len(CARD_RECEIPTS)
    split = {"batch": 1, "grad_accum": 8}  # poros.recipes.MEASURED on CUDA
    efficient = [{**c, "id": f"{c['id']}-efficient",
                  "key": {**c["key"], "path": "efficient", **split}} for c in cells]
    fused = [{**c, "id": f"{c['id']}-fused", "key": {**c["key"], "fused_compute": True}}
             for c in [*cells, *efficient]]
    monkeypatch.setattr(poros.plan.cells, "CELLS", [*cells, *efficient, *fused])


@pytest.fixture(autouse=True)
def _json_mode_ends_with_its_test():
    """`--json` holds for one invocation, and a test's CLI call is one: the next test starts in the
    mode its own arguments say."""
    yield
    contract = sys.modules.get("poros.cli.contract")
    if contract is not None:
        contract.reset()


@pytest.fixture(autouse=True)
def _ctrl_hints(monkeypatch):
    """Key hints read the same on every test host: ctrl+s, as on Linux.
    (A Mac says ⌥S; tests that mean it say so themselves.)"""
    monkeypatch.setenv("POROS_KEYS", "ctrl")
    from poros.cli.ui import components

    monkeypatch.setattr(components, "mac_glyphs", lambda: False)  # words: ctrl+s, on any host


def cpu_trainer_pick(monkeypatch) -> None:
    """On a real Mac, ``PorosTrainer`` is on mps or refused. A test that builds one on CPU asks
    for the Linux pick; tests/metal covers the gate itself by faking Darwin. Off a Mac, nothing."""
    if sys.platform != "darwin":
        return
    try:
        from poros.train import trainer
    except ImportError:  # no torch: no trainer to build
        return
    monkeypatch.setattr(trainer, "sys", SimpleNamespace(platform="linux"))


@pytest.fixture(autouse=True)
def _darwin_runs_the_cpu_suite_on_cpu(monkeypatch):
    """The CPU pick for every test. A module-scoped fixture is set up before this one, so one
    that builds a trainer asks for it itself: ``with pytest.MonkeyPatch.context() as mp:
    cpu_trainer_pick(mp)``."""
    cpu_trainer_pick(monkeypatch)


def _mac_without_metal() -> bool:
    if sys.platform != "darwin" or not HAS_ML:
        return False
    import torch

    return not torch.backends.mps.is_available()


# A test that trains in a process of its own (scripts/metal's drivers, `poros prove`'s children):
# there a Mac's trainer is on mps or refused, and a Mac with no MPS device (a virtual machine, as
# GitHub's macOS runners are: "Apple M2 Pro Virtual 0-core GPU") has nothing to train on. Off a
# Mac those drivers run on the CPU, and the test runs.
needs_metal_on_a_mac = pytest.mark.skipif(
    _mac_without_metal(), reason="trains in a process of its own, which on a Mac needs an MPS "
                                 "device, and torch reports none here (a virtual machine)")


@pytest.fixture(autouse=True)
def _pin_budget_ends_with_its_test():
    """poros.sss.pinning's budget is process-wide, and a test that fakes its environment (WSL2,
    no CUDA, a ceiling, POROS_PIN_BUDGET) and resets it leaves the state it computed under those
    fakes: a test that fakes WSL2 would otherwise leave every later pin refused.
    After each test (its monkeypatches undone first: this fixture is set up before them), the
    state is forgotten, so the next use computes it afresh in the real environment."""
    yield
    pinning = sys.modules.get("poros.sss.pinning")
    if pinning is not None:
        with pinning._lock:
            pinning._state.clear()


@functools.lru_cache(maxsize=None)
def _cpu_bf16_backward_refused():
    """Why this host's oneDNN refuses a bf16 backward on the CPU, or None where it runs one.

    A depthwise causal conv1d (Gated DeltaNet's) and a matmul, as the CPU MoE tests train them."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.backends.mkldnn.is_available():
        return None
    x = torch.randn(1, 4, 8, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(4, 1, 3, dtype=torch.bfloat16, requires_grad=True)
    m = torch.randn(8, 8, dtype=torch.bfloat16, requires_grad=True)
    try:
        (torch.nn.functional.conv1d(x, w, groups=4, padding=2).sum() + (x @ m).sum()).backward()
    except RuntimeError as exc:
        if "DNNL" in str(exc) or "oneDNN" in str(exc):
            return str(exc).splitlines()[0]
        raise
    return None


@pytest.fixture
def cpu_bf16_backward(record_property):
    """For a test that trains a bf16 model on the CPU. Some CPUs' oneDNN refuses a bf16 backward
    ("DNNL does not support bf16/f16 backward on the platform with avx2_vnni_2": a Core Ultra 9
    285K, torch 2.14.1); there the test runs on PyTorch's native CPU kernels (oneDNN off for this
    test only), so its assertions are unchanged. A Poros run computes on its GPU (CUDA or mps):
    it takes no bf16 backward on the CPU, so this is the test host's, not a run's."""
    why = _cpu_bf16_backward_refused()
    if why is None:
        yield
        return
    import torch

    record_property("onednn_off", why)
    with torch.backends.mkldnn.flags(enabled=False):
        yield
