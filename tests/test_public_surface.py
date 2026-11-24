"""The public import surface: the current ``__all__`` resolves, every error is importable from
``poros``, and the commands that need no torch import none.

A private name (``_x``) is never part of this surface: it is imported from the module that defines
it, wherever that is.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

# The names of poros.__all__ that poros.__getattr__ imports with torch.
NEEDS_TORCH = {"Run", "Datum", "Batch", "Loss", "PorosTrainer", "PorosModel",
               "enforce_determinism"}


def test_the_current_all_resolves():
    import poros

    missing = [n for n in poros.__all__ if n not in NEEDS_TORCH and not hasattr(poros, n)]
    assert not missing, missing
    pytest.importorskip("torch")
    assert all(hasattr(poros, n) for n in poros.__all__)


def test_the_engine_patch_is_not_public():
    """The HF patch is the MoE path's internal detail: not in ``poros`` or its ``__all__``."""
    import poros

    for name in ("patch_hf_model", "unpatch_hf_model", "prepare", "unprepare"):
        assert name not in poros.__all__
        with pytest.raises(AttributeError):
            getattr(poros, name)


# The commands AGENTS.md says work with no torch and no GPU. Each runs in a fresh interpreter,
# where torch is installed: it must stay unimported. (`poros machine` reads the torch build, so
# it is not one of them.)
TORCH_FREE_COMMANDS = [
    ["--help"], ["--version"], ["check", "{folder}", "--json"], ["schema", "config"],
    ["schema", "--list"], ["report", "--help"], ["leaderboard", "--help"], ["init", "--help"],
]
_HEAVY = ("torch", "transformers", "peft", "bitsandbytes", "triton", "datasets", "accelerate")


@pytest.mark.parametrize("argv", TORCH_FREE_COMMANDS, ids=lambda a: " ".join(a))
def test_the_torch_free_commands_import_no_torch(tmp_path, argv):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen2"}))
    argv = [a.format(folder=tmp_path) for a in argv]
    code = (
        "import sys\n"
        f"sys.argv = ['poros', *{argv!r}]\n"
        "from poros.cli.main import cli\n"
        "try:\n"
        "    cli()\n"
        "except SystemExit:\n"
        "    pass\n"
        f"print('HEAVY', [m for m in {_HEAVY!r} if m in sys.modules], file=sys.stderr)\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=120, env={**os.environ, "HF_HUB_OFFLINE": "1",
                                            "POROS_HOME": str(tmp_path / "home")})
    line = [x for x in proc.stderr.splitlines() if x.startswith("HEAVY")]
    assert line == ["HEAVY []"], (argv, proc.stderr[-2000:])


def test_every_poros_error_is_importable_from_poros():
    """Every PorosError subclass is exported from poros, so a caller catches each by name
    (PorosStepCheckError, E_STEP_CHECK, and PorosDiskError, E_DISK, among them)."""
    import inspect

    import poros
    from poros.core import errors

    every = {name for name, cls in inspect.getmembers(errors, inspect.isclass)
             if issubclass(cls, errors.PorosError) and cls.__module__ == errors.__name__}
    assert {"PorosStepCheckError", "PorosDiskError"} <= every
    for name in sorted(every):
        assert name in poros.__all__ and getattr(poros, name) is getattr(errors, name), name


@pytest.mark.cuda_host  # a Mac picks memory or disk for each model itself and refuses --store
def test_a_plan_on_the_ssd_store_runs_with_no_torch_installed(tmp_path):
    """A MoE model's plan on the SSD store with torch not installed failed E_DEPS
    (ModuleNotFoundError): the plan read the disk's cached probe through poros.sss.probe, which
    imported the store's format and so torch. It runs torch-free."""
    code = ("import sys, json\nsys.modules['torch'] = None\n"
            "from poros.cli.context import Context\nfrom poros.cli.verbs import shared\n"
            "from poros.cli.verbs.train import sss_dir\n"
            "ctx = Context(gpu_name='RTX 4090', fixed_cached=frozenset())\n"
            "p = shared.plan_for('Qwen/Qwen3-30B-A3B-Instruct-2507', ctx, store='ssd', "
            "store_path=str(sss_dir()))\nprint(json.dumps({'schema': p.schema_}))\n")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=120, cwd=tmp_path,
                          env={**os.environ, "HF_HUB_OFFLINE": "1", "HOME": str(tmp_path),
                               "POROS_HOME": str(tmp_path / "home")})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert json.loads(proc.stdout)["schema"] == "poros.plan.v1"
