"""Hold all of a CUDA card's memory but ``--free-mib`` in another process, as the head-to-head's
capped rows did (6246 MiB left free on the RTX 4090: what an 8 GB laptop card leaves).
Runs until killed; prints "holding" once the memory is held.

    python ballast.py --free-mib 6246 &   ...   kill %1
"""

from __future__ import annotations

import argparse
import signal
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--free-mib", type=int, default=6246)
    args = ap.parse_args()
    held = []
    while True:
        free, _ = torch.cuda.mem_get_info()
        over = free // 2**20 - args.free_mib
        if over <= 0:
            break
        held.append(torch.empty(min(over, 1024) * 2**20, dtype=torch.uint8, device="cuda"))
    free, total = torch.cuda.mem_get_info()
    print(f"holding: {sum(t.numel() for t in held) // 2**20} MiB, {free // 2**20} MiB free of "
          f"{total // 2**20}", flush=True)
    signal.signal(signal.SIGTERM, lambda *a: exit(0))
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
