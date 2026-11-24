"""Drive the real TrainView/Activity with a fake trainer (no GPU needed).

    python scripts/cli/sim_train.py [steps] [step_seconds]

Exercises the exact code `poros train` runs after preflight: load activity,
live surface, checkpoint/eval events, done summary.
"""

import math
import random
import sys
import time

from poros.cli import train_cmd
from poros.cli.ui import components as c
from poros.cli.ui.live import columns
from poros.cli.ui.screens import TrainResult, TrainState, train_done
from poros.cli.ui.text import S
from poros.cli.ui.train_view import Activity, TrainView
from poros.train.reporting import StepReport

steps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
dt = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15


class FakeBM:
    num_blocks = 20
    num_layers = 80
    h2d_bytes_total = 0
    active_block_id = 0


class FakeTrainer:
    block_manager = FakeBM()


w = columns()
train_cmd.out(
    [c.welcome("poros", "Qwen/Qwen2.5-72B-Instruct", "RTX 4090 24.0 GB", max_width=min(w, 100)), []]
)
train_cmd.out(
    [
        c.event("Checked", "train.yaml", "config"),
        c.rail(S("nf4 · block 4 · rank 16 · seq 1024", "dm")),
    ]
)
train_cmd.out(c.rail_path("output", "./runs/qwen2.5-72b-lora", min(w, 100)) + [[]])
with Activity(
    "Loading Qwen2.5-72B-Instruct",
    statusline=["run 7f3a91c2", "ctrl-c to stop"],
    tag=c.tracked("load"),
) as act:
    time.sleep(1.2)
    act.done(
        [
            c.event("Loaded", "Qwen2.5-72B-Instruct", c.duration(act.elapsed)),
            c.rail(S("dense · 80 layers · 20 blocks", "dm")),
        ]
    )

st = TrainState(
    model="Qwen/Qwen2.5-72B-Instruct",
    arch="dense · 80 layers · 20 blocks",
    gpu="RTX 4090",
    card_gb=24.0,
    output_dir="./runs/qwen2.5-72b-lora",
    run="7f3a91c2",
)
view = TrainView(st)
ft = FakeTrainer()
view.start()
view.on_start(ft, steps, 0)
random.seed(3)
t0 = time.monotonic()
for i in range(steps):
    for b in range(20):
        ft.block_manager.active_block_id = b
        ft.block_manager.h2d_bytes_total += int(21.4e9 * dt / 20)
        time.sleep(dt / 20)
    loss = 1.1 + 1.7 * math.exp(-i / (steps / 4)) + random.uniform(-0.05, 0.05)
    view.on_step(
        StepReport(
            step=i + 1,
            total=steps,
            loss=loss,
            step_time=dt,
            tokens=(i + 1) * 4096,
            tokens_per_sec=4096 / dt,
            lr=2e-4,
            vram_gb=17.4 + random.uniform(-0.2, 0.2),
            vram_peak_gb=18.1,
        )
    )
    if (i + 1) == steps // 2:
        view.on_checkpoint(i + 1, f"./runs/qwen2.5-72b-lora/checkpoint-{i + 1}")
view.close()
train_cmd.out(
    [[]]
    + train_done(
        TrainResult(
            steps=steps,
            wall_s=time.monotonic() - t0,
            final_loss=st.losses[-1],
            first_loss=st.losses[0],
            losses=st.losses,
            peak_vram_gb=18.1,
            card_gb=24.0,
            peak_ram_gb=view.peak_ram,
            ram_total_gb=st.ram_total_gb,
            output_dir="./runs/qwen2.5-72b-lora",
            receipt="./runs/qwen2.5-72b-lora/receipt.json",
            tokens_per_sec=4096 / dt,
            next_cmd="poros eval train.yaml",
        ),
        columns(),
    )
)
