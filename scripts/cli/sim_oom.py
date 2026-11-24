"""Show the OOM screen + retry menu with a fake PorosOOMError (no GPU)."""

from poros.cli import train_cmd
from poros.cli.ui import components as c
from poros.cli.ui.screens import TrainState
from poros.core.errors import PorosOOMError


class Cfg:
    block_size = 4
    max_seq_length = 8192


st = TrainState(
    model="Qwen/Qwen2.5-72B-Instruct", gpu="RTX 4090", card_gb=24.0, step=0, vram_peak_gb=23.9
)
train_cmd.out(
    [
        c.welcome("poros", "Qwen/Qwen2.5-72B-Instruct", "RTX 4090 24.0 GB"),
        [],
        c.event("Loaded", "Qwen2.5-72B-Instruct", "41.2s"),
    ]
)
try:
    train_cmd._oom(PorosOOMError(block_size=4, seq_len=8192), Cfg(), "train.yaml", st)
except BaseException:
    pass
