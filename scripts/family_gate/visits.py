#!/usr/bin/env python3
"""visits.py MODEL_DIR -- whether the model's decoder runs its layers as visits (a looped decoder
declares it: ``poros_visits`` on its class, poros.moe.load.declares_visits), and if so the words
Poros gives for training it on the sequential path only (poros.core.config.sequential_reason of
its config.json). Prints those words, or nothing for any other model. Exit 0 either way."""

import json
import sys
from pathlib import Path

model = sys.argv[1]
from poros.moe.load import declares_visits  # noqa: E402

if declares_visits(model):
    from poros.core.config import PorosConfig, sequential_reason

    config = json.loads((Path(model) / "config.json").read_text())
    why = sequential_reason(PorosConfig(model_name_or_path=model), config)
    if not why:
        sys.exit("a decoder that declares visits but no reason to train it sequentially")
    print(why)
