"""Write one job's config (JSON, which ``poros train`` reads as YAML) from a preset and overrides.

    python make_config.py q30 --data data --out jobs/q30_new_cos_s42.yaml learning_rate=2e-4 seed=42

Presets:
  q30     Qwen3-30B-A3B-Instruct-2507 at the moe-30b recipe's shape (rank 16, alpha 32, dropout
          0, 1 x 8, 512 tokens, experts in RAM, path efficient), on the kit's chats, 600 updates,
          held-out loss every 50, cosine after 100 warmup updates
  gate30  the same shape, 16 updates, rate 2e-4 constant: the exactness gates
  cap8    the laptop row's shape: rank 8, alpha 16, 256 tokens, SSD store, path efficient, rate
          2e-5 constant, 16 updates
  olmoe   allenai/OLMoE-1B-7B-0924 on a Mac (rank 16, one row an update, 128 tokens) on the
          kit's text rows, 300 updates at 2e-5 constant, held-out loss every 25
  macmoe  Qwen3-30B-A3B-Instruct-2507 at the mac-moe recipe's shape, 16 updates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

A3B = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SHAPE = {"lora_rank": 16, "lora_alpha": 32, "lora_dropout": 0.0, "per_device_batch_size": 1,
         "recipe": "sft", "save_steps": 0, "eval_checkpoints": False}
PRESETS = {
    "q30": {**SHAPE, "model_name_or_path": A3B, "gradient_accumulation_steps": 8,
            "max_seq_length": 512, "store": "ram", "path": "efficient", "max_steps": 600,
            "eval_steps": 50, "learning_rate": 2e-4, "lr_scheduler": "cosine",
            "warmup_steps": 100, "seed": 42, "data": "chats"},
    "gate30": {**SHAPE, "model_name_or_path": A3B, "gradient_accumulation_steps": 8,
               "max_seq_length": 512, "store": "ram", "path": "efficient", "max_steps": 16,
               "eval_steps": 0, "learning_rate": 2e-4, "lr_scheduler": "constant",
               "warmup_steps": 0, "seed": 42, "data": "chats"},
    "cap8": {**SHAPE, "model_name_or_path": A3B, "lora_rank": 8, "lora_alpha": 16,
             "gradient_accumulation_steps": 8, "max_seq_length": 256, "store": "ssd",
             "path": "efficient", "max_steps": 16, "eval_steps": 0, "learning_rate": 2e-5,
             "lr_scheduler": "constant", "warmup_steps": 0, "seed": 42, "data": "chats"},
    "olmoe": {**SHAPE, "model_name_or_path": "allenai/OLMoE-1B-7B-0924", "recipe": "auto",
              "gradient_accumulation_steps": 1, "max_seq_length": 128, "max_steps": 300,
              "eval_steps": 25, "learning_rate": 2e-5, "lr_scheduler": "constant",
              "warmup_steps": 0, "seed": 42, "data": "text"},
    "macmoe": {**SHAPE, "model_name_or_path": A3B, "gradient_accumulation_steps": 1,
               "max_seq_length": 128, "max_steps": 16, "eval_steps": 0,
               "learning_rate": 2e-5, "lr_scheduler": "constant", "warmup_steps": 0,
               "seed": 42, "data": "chats"},
}


def _value(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("preset", choices=sorted(PRESETS))
    ap.add_argument("--data", required=True, help="the folder prep_data.py wrote")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sss", help="where an SSD-store run keeps its .sss expert file")
    ap.add_argument("set", nargs="*", help="key=value overrides")
    args = ap.parse_intermixed_args()  # the kit passes the overrides after the options
    cfg = dict(PRESETS[args.preset])
    for item in args.set:
        key, _, value = item.partition("=")
        cfg[key] = _value(value)
    kind = cfg.pop("data")
    data = Path(args.data).resolve()
    cfg["dataset"] = str(data / f"{kind}_train.jsonl")
    if cfg.get("store") == "ssd" and "store_path" not in cfg:
        if not args.sss:
            raise SystemExit("an SSD-store run needs --sss <folder on a local SSD>")
        Path(args.sss).mkdir(parents=True, exist_ok=True)
        cfg["store_path"] = str(Path(args.sss).resolve())
    if cfg.get("eval_steps"):
        cfg["eval_dataset"] = str(data / f"{kind}_eval.jsonl")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cfg, indent=1) + "\n")


if __name__ == "__main__":
    main()
