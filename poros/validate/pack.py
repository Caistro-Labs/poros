"""What a passing ``poros prove <model> --unvalidated`` leaves to submit: a folder in Poros's home.

    load_gate.jsonl     the load gate's record, as scripts/streamed_load_gate.py writes it
    parity.jsonl        the training gate's record, as scripts/training_parity.py
                        writes it, with each arm's peak memory and where each surface first differed
    receipt.json        the prove result, without this machine's paths
    README.md           the model, its model_type, the GPU, driver and versions, and the numbers
    registry_line.txt   the exact ArchSpec line the procedure adds to poros/core/arch_registry.py

Submitting is an architecture request issue on the Poros GitHub repo with these files attached;
a maintainer commits them under ``docs/validation/<model_type>/`` with that line. Poros never
marks a family validated itself: the committed artifact is the only way.
Everything written is built from checked fields and numbers: no path, host or user of this machine.
"""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

SUBMIT_STEPS = 200  # the procedure's training gate
REGISTRY_FILE = "poros/core/arch_registry.py"
_HUB_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}")
_GPU = re.compile(r"[A-Za-z0-9 ._()-]{1,80}")
_VERSION = re.compile(r"[A-Za-z0-9.+_-]{1,40}")


def shown_model(model: str) -> str:
    """The model as a pack may name it: a Hub id as it is, a local folder by its own name alone."""
    if _HUB_ID.fullmatch(model) and not Path(model).exists():
        return model
    name = Path(model).name
    return f"a local folder named {name}" if _WORD.fullmatch(name) else "a local folder"


def clean_env(env: dict[str, Any]) -> dict[str, str]:
    """Versions and the card's name, each by pattern: nothing else of this machine."""
    out = {}
    for key, value in env.items():
        text = str(value)
        if key == "gpu":
            ok = _GPU.fullmatch(text)
        else:
            ok = _VERSION.fullmatch(text)
        out[key] = text if ok else "unknown"
    return out


def folder_for(model_type: str) -> str:
    return f"docs/validation/{model_type}/"


def registry_line(model_type: str, model: str, env: dict[str, Any]) -> str:
    """The line a submission adds to the registry (docs/models.md), as it is pasted."""
    note = (f"{shown_model(model)} NF4 0.00e+00 on {env.get('gpu', 'unknown')}, torch "
            f"{env.get('torch', 'unknown')}: load byte-identical and streamed vs resident, "
            f"{folder_for(model_type)}")
    pieces = textwrap.wrap(note, 76, drop_whitespace=False)
    quoted = "\n".join(f"        {json.dumps(p)}" for p in pieces)
    return (f'    "{model_type}": ArchSpec(\n'
            f'        "{model_type}", "target", "0.00e+00",\n'
            f"{quoted},\n"
            "    ),\n")


def _readme(result: dict[str, Any], load: dict[str, Any], training: dict[str, Any],
            line: str) -> str:
    env, kind = result["env"], result["model_type"]
    model = shown_model(result["model"])
    said = model if model == result["model"] else "<model>"  # a Hub id, or a placeholder
    steps, seq = training["steps"], training["seq_len"]
    losses = training.get("resident_losses") or []
    short = steps < SUBMIT_STEPS
    rows = [
        ("model", model), ("model_type", kind), ("registry status before", result["status"]),
        ("GPU", env.get("gpu", "unknown")), ("driver", env.get("driver", "unknown")),
        ("CUDA", env.get("cuda", "unknown")), ("torch", env.get("torch", "unknown")),
        ("transformers", env.get("transformers", "unknown")),
        ("peft", env.get("peft", "unknown")), ("bitsandbytes", env.get("bitsandbytes", "unknown")),
        ("poros", env.get("poros", "unknown")), ("python", env.get("python", "unknown")),
    ]
    table = "\n".join(f"| {k} | {v} |" for k, v in rows)

    def gib(x):
        return "n/a" if x is None else f"{x:g} GiB"

    surfaces = [
        ("loss at every step", training["losses_equal"], steps, training["loss_max_abs_diff"]),
        ("adapter gradients", training["grads_equal"], training["grads_count"],
         training["grads_max_abs_diff"]),
        ("adapter weights", training["weights_equal"], training["weights_count"],
         training["weights_max_abs_diff"]),
        ("optimizer state", training["opt_state_equal"], training["opt_state_count"],
         training["opt_state_max_abs_diff"]),
    ]
    surface_rows = "\n".join(f"| {name} | {'equal' if eq else 'DIFFERS'} | {count} | {diff:.2e} |"
                             for name, eq, count, diff in surfaces)
    first_last = (f"{losses[0]:.6f} at step 1, {losses[-1]:.6f} at step {len(losses)}"
                  if losses else "none")
    submit_note = (
        f"The procedure's training gate is {SUBMIT_STEPS} steps and this pack ran {steps}: a "
        f"pack to submit runs `poros prove {said} --unvalidated --steps {SUBMIT_STEPS}`.\n\n"
        if short else "")
    return f"""# {kind}: load and training gates, NF4

Written by `poros prove <model> --unvalidated` on a pass. Both gates of
docs/models.md, on one GPU, at exactly `0.00e+00`.

{table}

## Load gate (load_gate.jsonl)

The streamed NF4 load against the standard load, a sha256 over the raw bytes of every
parameter, quant-state tensor and buffer.

- byte_identical: {str(load['byte_identical']).lower()}, {load['tensors_compared']} tensors \
compared, {load['n_mismatched']} mismatched, {load['n_missing']} missing, {load['n_extra']} extra
- load peak: standard {gib(load.get('standard_load_peak_vram_gb'))}, streamed \
{gib(load.get('streamed_load_peak_vram_gb'))}

## Training gate (parity.jsonl)

Streamed (block {training['block_size']}) against resident (every block on the GPU, plain
autograd), {steps} steps at {seq} tokens, G={training['g']}, lora_dropout \
{training['lora_dropout']}, seed {training.get('seed', 'unknown')}, synthetic tokens.
`torch.equal`, tolerance (0, 0), at every step.

| surface | result | compared | max abs diff |
|---|---|---|---|
{surface_rows}

- all_equal: {str(training['all_equal']).lower()}
- loss: {first_last}
- peak memory: resident {gib(training.get('resident_peak_vram_gb'))}, streamed \
{gib(training.get('blockwise_peak_vram_gb'))}
- seconds: resident {training['resident_seconds']:.1f}, streamed \
{training['blockwise_seconds']:.1f}

## Reproduce

    poros prove {said} --unvalidated --steps {steps} --seq-len {seq}

or the procedure's scripts, each arm in its own process:

    python scripts/streamed_load_gate.py --model {said} --hash-compare
    POROS_USER_ASSERTED_ARCHS={kind} python scripts/training_parity.py \\
        --model {said} --block-size 4 --g 1 --steps {steps} \\
        --seq-len {seq} --lora-dropout 0.0

## Submit

{submit_note}Open an architecture request issue on the Poros GitHub repo and attach this folder's
files. A maintainer commits them as `{folder_for(kind)}`, with this line in `_ARCHITECTURES` in
`{REGISTRY_FILE}` (also in registry_line.txt):

```python
{line}```

with a row for it in the claim map of `docs/validation/README.md` and the model's config.json in
`tests/metal/real_configs/`, so CI runs the CPU check on it. The maintainer runs the gates again on
their own card first: parity reported from another machine is not accepted alone. The line says
`target`; the maintainer sets it to `validated` once their own run passes. Poros never
marks a family validated itself: this artifact, committed, is the only way. Until then every run
of it says `support: user_asserted`. See docs/models.md.
"""


def write(folder: Path, result: dict[str, Any], load: dict[str, Any],
          training: dict[str, Any]) -> list[str]:
    """The pack, in a new folder. Returns the names written."""
    folder.mkdir(parents=True, exist_ok=False, mode=0o700)
    shown = shown_model(result["model"])
    env = clean_env(result["env"])
    result = {**result, "env": env}
    line = registry_line(result["model_type"], result["model"], env)
    files = {
        "load_gate.jsonl": json.dumps({**load, "model": shown}) + "\n",
        "parity.jsonl": json.dumps({**training, "model": shown}) + "\n",
        "receipt.json": json.dumps({k: v for k, v in {**result, "model": shown}.items()
                                    if k not in ("receipt", "pack", "next", "submit", "files")},
                                   indent=2, sort_keys=True) + "\n",
        "README.md": _readme(result, load, training, line),
        "registry_line.txt": line,
    }
    for name, text in files.items():
        (folder / name).write_text(text)
    return sorted(files)
