"""The README.md of a finished run's folder, as ``poros train`` leaves it: what the adapter is,
in a few true lines. PEFT writes its model-card template there ("# Model Card for Model ID",
every section "[More Information Needed]"); its metadata header is kept, the template is not.
Only the run's own folder: a checkpoint's files are left as the trainer wrote them."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _header(text: str) -> str:
    """PEFT's metadata block (``---`` to ``---``: base_model, library_name, tags), which the
    Hub reads; "" where the file has none."""
    if not text.startswith("---\n"):
        return ""
    end = text.find("\n---\n", 4)
    return text[: end + 5] if end != -1 else ""


def card(cfg: Any, recipe: str, steps: Optional[int], version: str) -> str:
    """The card's text: the base, what trained it, and the two ways to use it."""
    name = Path(str(cfg.output_dir)).name or "adapter"
    data = Path(str(cfg.dataset)).name if cfg.dataset else ""
    rows = [("base model", f"`{cfg.model_name_or_path}`"),
            ("trained with", f"Poros {version}" + (f", {recipe}" if recipe else "")),
            *([("data", f"`{data}`")] if data else []),
            *([("steps", f"{steps:,}")] if steps else []),
            ("LoRA", f"rank {cfg.lora_rank}, alpha {cfg.lora_alpha}"),
            ("length", f"{cfg.max_seq_length} tokens"),
            ("base weights", f"{str(cfg.quantization).upper()} (the adapter trained against "
                             "them)")]
    table = "\n".join(f"| {k} | {v} |" for k, v in rows)
    return (f"# {name}\n\n"
            f"A LoRA adapter for `{cfg.model_name_or_path}`, trained with Poros.\n\n"
            f"| | |\n|---|---|\n{table}\n\n"
            "## Use it\n\n"
            "- A standalone model for transformers, vLLM or TGI: "
            "`poros export <this folder> --format merged --output ./merged`.\n"
            "- The adapter alone: PEFT's `PeftModel.from_pretrained(base, \"<this folder>\")`, "
            "on the NF4 base.\n\n"
            "`receipt.json` holds the run's settings and measured numbers, and `events.jsonl` "
            "every step.\n")


def write(cfg: Any, recipe: str, steps: Optional[int], version: str) -> None:
    """Replace the template PEFT left in the run's folder with ``card``, under its metadata
    header. Nothing where the folder holds no adapter (a preview writes none), and a folder
    that cannot be written keeps what it has: the card is a courtesy, never a run's failure."""
    from poros.core import checkpoints

    out = Path(str(cfg.output_dir))
    readme = out / "README.md"
    try:
        if not checkpoints.has_weights(out) or readme.is_symlink():
            return
        head = _header(readme.read_text(encoding="utf-8")) if readme.is_file() else ""
        tmp = out / ".README.md.part"
        tmp.write_text(head + ("\n" if head else "") + card(cfg, recipe, steps, version),
                       encoding="utf-8")
        tmp.replace(readme)
    except (OSError, UnicodeError):
        return
