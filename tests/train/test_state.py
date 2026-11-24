"""PorosTrainer's shared state is declared once (poros.train.state._TrainerState), and both
hosts of the blockwise engine provide what it calls (poros.core.blockwise.BlockHost)."""

from __future__ import annotations

import ast
import typing
from pathlib import Path

import pytest

TRAIN = Path(__file__).resolve().parents[2] / "poros" / "train"
MIXINS = {"trainer": "PorosTrainer", "loop": "_Loop", "forward": "_Forward", "head": "_Head",
          "checkpointing": "_Checkpointing", "progress": "_Progress"}


def _declared() -> set[str]:
    tree = ast.parse((TRAIN / "state.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_TrainerState")
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}


def _self_attributes() -> dict[str, set[str]]:
    """Each attribute a mixin reads or writes on self, by module (methods left out)."""
    seen: dict[str, set[str]] = {}
    methods: set[str] = set()
    for module, name in MIXINS.items():
        tree = ast.parse((TRAIN / f"{module}.py").read_text())
        cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == name)
        methods |= {f.name for f in cls.body
                    if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for x in ast.walk(cls):
            if (isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name)
                    and x.value.id == "self"):
                seen.setdefault(x.attr, set()).add(module)
    return {a: m for a, m in seen.items() if a not in methods}


def test_every_attribute_the_mixins_share_is_declared():
    declared = _declared()
    shared = {a for a, mods in _self_attributes().items() if len(mods) > 1}
    assert shared - declared == set()


def test_every_attribute_the_trainer_sets_is_declared():
    tree = ast.parse((TRAIN / "trainer.py").read_text())
    init = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    written = {x.attr for x in ast.walk(init) if isinstance(x, ast.Attribute)
               and isinstance(x.ctx, ast.Store) and isinstance(x.value, ast.Name)
               and x.value.id == "self"}
    assert written - _declared() == set()


def test_both_hosts_provide_what_the_engine_calls():
    pytest.importorskip("torch")
    from poros.core.blockwise import BlockHost
    from poros.patch.hf.forward import _BlockwiseContext
    from poros.train.trainer import PorosTrainer

    def members(proto) -> set[str]:
        return {n for n in vars(proto) if n.startswith("_") and not n.startswith("__")
                and callable(vars(proto)[n])}

    for host in (PorosTrainer, _BlockwiseContext):
        assert all(hasattr(host, m) for m in members(BlockHost)), host
    assert set(typing.get_type_hints(BlockHost, include_extras=False)) >= {
        "config", "block_manager", "block_trainable_params"}
