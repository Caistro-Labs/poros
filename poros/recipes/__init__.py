"""Built-in training loops, written on ``poros.Run``'s public verbs only.

Each recipe is one short file you can read top to bottom, copy and change:

    poros.recipes.sft.main(run, data)      # supervised: chats or text
    poros.recipes.dpo.main(run, pairs)     # preferences; ipo, kto, orpo, simpo take pairs too

``poros.recipes.copy("dpo", "my_loop.py")`` writes one into your folder.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Optional, Union

from poros.recipes.presets import SFT_LR

NAMES = ("sft", "dpo", "ipo", "kto", "orpo", "simpo")
PAIRS = ("dpo", "ipo", "kto", "orpo", "simpo")  # the loops that train on preference pairs
LABEL = {"sft": "SFT", "dpo": "DPO", "ipo": "IPO", "kto": "KTO", "orpo": "ORPO", "simpo": "SimPO",
         "classic": "classic"}

# What every recipe trains at unless you say otherwise: the settings Poros's
# receipts were measured at (NF4, block 4, one row x 8 accumulated, rank 16,
# alpha 32, no dropout), so the fit check reads a measurement, not a guess.
CUDA_MEASURED: Dict[str, Any] = {
    "quantization": "nf4",
    "block_size": 4,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.0,
}
# What a CUDA card's run trains at, whatever machine plans it (`fit --gpu` on a Mac); MEASURED is
# this machine's, and off a Mac the same
MEASURED: Dict[str, Any] = dict(CUDA_MEASURED)
if sys.platform == "darwin":  # one pool, and one example a step: what the Mac table says
    MEASURED.update(block_size=1, gradient_accumulation_steps=1)

# Each recipe's own settings on top. LoRA wants about ten times a full
# fine-tune's learning rate (Tinker's guidance), DPO far less than SFT; SFT
# takes two passes over small data, DPO one. Your loop starts from SFT's.
PRESETS: Dict[str, Dict[str, Any]] = {
    "sft": {**MEASURED, "learning_rate": SFT_LR, "num_epochs": 2, "train_on": "answers"},
    "dpo": {**MEASURED, "learning_rate": 5e-6, "num_epochs": 1, "dpo_beta": 0.1},
    # IPO's beta is its tau; ORPO's weighs the preference against its SFT term and, with no SFT
    # before it, takes an SFT-like rate; SimPO's sits on mean logprobs, so it is larger
    "ipo": {**MEASURED, "learning_rate": 5e-6, "num_epochs": 1, "dpo_beta": 0.1},
    "kto": {**MEASURED, "learning_rate": 5e-6, "num_epochs": 1, "dpo_beta": 0.1},
    "orpo": {**MEASURED, "learning_rate": 8e-6, "num_epochs": 1, "dpo_beta": 0.1},
    "simpo": {**MEASURED, "learning_rate": 5e-6, "num_epochs": 1, "dpo_beta": 2.0},
}
LOOP = {**MEASURED, "learning_rate": SFT_LR, "num_epochs": 2}
LR = {n: p["learning_rate"] for n, p in PRESETS.items()}
LOOP_LR = LOOP["learning_rate"]


def summary(name: str) -> str:
    """The preset in one line: ``2 epochs · lr 2e-4 · rank 16 · NF4 block 4``."""
    p = PRESETS.get(name, LOOP)
    lr = f"{p['learning_rate']:.0e}".replace("e-0", "e-")
    epochs = p["num_epochs"]
    return (f"{epochs} epoch{'s' * (epochs != 1)} · lr {lr} · rank {p['lora_rank']} · "
            f"{p['quantization'].upper()} block {p['block_size']}")


def get(name: str) -> ModuleType:
    if name not in NAMES:
        raise ValueError(f"unknown recipe {name!r}; built in: {', '.join(NAMES)}")
    return importlib.import_module(f"poros.recipes.{name}")


def path(name: str) -> Path:
    """The recipe's source file."""
    get(name)
    return Path(__file__).with_name(f"{name}.py")


def copy(name: str, dest: Union[str, Path, None] = None) -> Path:
    """Copy a recipe's file to ``dest`` (default ``./<name>_loop.py``);
    refuses to overwrite."""
    target = Path(dest) if dest is not None else Path(f"{name}_loop.py")
    if os.path.lexists(target):  # a link too, or the copy lands where it points
        raise FileExistsError(f"{target} already exists")
    shutil.copyfile(path(name), target)
    return target


@dataclass(frozen=True)
class Choice:
    """Which loop a config trains with, and why.

    ``name`` is ``sft``, ``dpo``, ``loop`` (your file) or ``classic`` (the
    engine's own loop: synthetic data, csv, Hub datasets, instruction rows).
    ``shape`` is the data's builder (``chats``, ``pairs``, ``text``) on the
    recipe paths; ``source`` says ``set`` or ``from data``.
    """

    name: str
    shape: Optional[str] = None
    source: str = "from data"

    @property
    def settings(self) -> Dict[str, Any]:
        """The config this loop trains at by default ({} for classic: the
        config's own)."""
        if self.name == "classic":
            return {}
        return dict(PRESETS.get(self.name, LOOP))

    @property
    def lr(self) -> Optional[float]:
        """The learning rate this loop trains at by default (None: the config's)."""
        return self.settings.get("learning_rate")

    @property
    def epochs(self) -> Optional[int]:
        return self.settings.get("num_epochs")

    @property
    def label(self) -> str:
        return LABEL.get(self.name, Path(self.name).name)


def choose(cfg: Any) -> Choice:
    """The loop for ``cfg`` (a PorosConfig), torch-free: reads the first
    row of a local .jsonl/.json. Raises ValueError for a pairing that
    cannot train (DPO on chats, a loop file that is missing)."""
    from poros import data

    loop = getattr(cfg, "loop", None)
    recipe = getattr(cfg, "recipe", "auto")
    spec = str(cfg.dataset or "")
    local = Path(spec).expanduser()
    readable = local.is_file() and local.suffix in (".jsonl", ".json", ".csv")
    shape = None
    if readable:
        try:
            shape = data.detect(local)
        except ValueError:
            if recipe != "auto" or loop:
                raise
    if loop:
        if not Path(loop).expanduser().is_file():
            raise ValueError(f"loop file {loop} does not exist")
        if not readable:
            raise ValueError("a loop file trains on a local .jsonl or .json file")
        return Choice(str(Path(loop).expanduser()), shape, "set")
    if recipe == "auto":
        if shape == "pairs":
            return Choice("dpo", shape)
        if shape == "chats":
            return Choice("sft", shape)
        return Choice("classic")
    if not readable:
        raise ValueError(f"--recipe {recipe} trains on a local .jsonl or .json file")
    if recipe in PAIRS and shape != "pairs":
        raise ValueError(f"{LABEL[recipe]} needs preference pairs (prompt, chosen, rejected); "
                         f"{local.name} holds {shape}")
    if recipe == "sft" and shape == "pairs":
        raise ValueError(f"{local.name} holds preference pairs; train them with --recipe dpo")
    return Choice(recipe, shape, "set")


def apply(cfg: Any, choice: Choice) -> list:
    """Set the recipe's settings on ``cfg`` wherever the config did not say
    one; returns the names set."""
    said = cfg.model_fields_set
    out = []
    for name, value in choice.settings.items():
        if name in said or name not in type(cfg).model_fields:
            continue
        setattr(cfg, name, value)
        out.append(name)
    return out


def load_loop(file: Union[str, Path]) -> Any:
    """A loop file's ``main(run, data, *, steps, stop)``."""
    file = Path(file).expanduser()
    spec = importlib.util.spec_from_file_location(f"poros_loop_{file.stem}", file)
    if spec is None or spec.loader is None:
        raise ValueError(f"{file} is not a Python file")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = getattr(module, "main", None)
    if not callable(main):
        raise ValueError(f"{file} has no main(run, data, ...)")
    return main
