#!/usr/bin/env python3
"""The `track` gate: a Mac run's losses, step by step, against a CUDA run's.

    track.py --mac RUN --cuda RUN --out DIR --commit SHA

Each RUN is a finished run's folder, of which this reads events.jsonl and poros_config.json, so copy
the CUDA one over and run this on the Mac: the receipt pins this machine. Both runs train the same
rows at the same settings, at G=1 and lr 2e-5, and ``poros.metal.verify`` rules on the result, with
its tolerance. Same-machine gates cannot see what this saw first: a Mac run going its own way.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from poros.metal.prove import pin
from poros.metal.verify import RECEIPT_SCHEMA, track_relative, write_verdict

SETTINGS = (
    "model_name_or_path", "model_revision", "max_seq_length", "gradient_accumulation_steps",
    "per_device_batch_size", "max_steps", "learning_rate", "warmup_steps", "lr_scheduler",
    "lora_rank", "lora_alpha", "lora_dropout", "weight_decay", "max_grad_norm", "adam_beta1",
    "adam_beta2", "adam_epsilon", "seed", "train_on", "eval_split",
)


def read_run(folder: Path) -> dict:
    config = json.loads((folder / "poros_config.json").read_text())
    steps = [json.loads(line) for line in (folder / "events.jsonl").read_text().splitlines()]
    steps = {e["step"]: e["data"] for e in steps if e.get("event_type") == "step"}
    return {
        "settings": {key: config.get(key) for key in SETTINGS},
        "losses": [steps[s]["loss"] for s in sorted(steps)],
        # what says the rows are the same
        "tokens": [steps[s].get("tokens") for s in sorted(steps)],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mac", required=True, type=Path)
    ap.add_argument("--cuda", required=True, type=Path)
    ap.add_argument("--commit", default="unknown")
    ap.add_argument("--wheel-hash", default="none")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    mac, cuda = read_run(a.mac), read_run(a.cuda)
    config = json.loads((a.mac / "poros_config.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "receipt.json"
    steps = len(mac["losses"])
    path.write_text(json.dumps({
        "schema": RECEIPT_SCHEMA,
        "gate": "track",
        "model": config["model_name_or_path"],
        "seq": config["max_seq_length"],
        "g": config["gradient_accumulation_steps"],
        "steps": steps,
        "first_microstep": 0,
        "expected_records": steps * config["gradient_accumulation_steps"],
        "lora_dropout": config["lora_dropout"],
        "pin": pin(config["store_path"], a.commit, a.wheel_hash),
        "mac": mac,
        "cuda": cuda,
        "max_relative": track_relative(mac["losses"], cuda["losses"])
        if len(cuda["losses"]) == steps else None,
        "pass": None,
    }, indent=1) + "\n")
    failures = write_verdict(path, a.out)
    print(json.dumps({"receipt": str(path), "pass": not failures, "failures": failures}, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
