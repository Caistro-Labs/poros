"""Byte-identity gate for the EXPERIMENTAL block-streamed NF4 load.

Loads the same model twice — once via the standard GPU load, once via the
block-streamed load — and compares EVERY tensor with ``torch.equal``:
packed 4-bit weight data, every quant-state tensor (absmax, nested absmax,
quant maps, offsets), and every unquantized parameter and persistent buffer.

NF4 quantization is per-tensor and deterministic, so the two loads must be
byte-identical; any difference is a bug in the streamed loader. A pass
certifies the streamed path produces the same base the parity-gated runtime
trains on — no new training gates are required.

Usage (on a CUDA box; pick a model whose standard load fits the card):
    python scripts/streamed_load_gate.py --model Qwen/Qwen2.5-0.5B \
        --out streamed_gate_0p5b.json
    python scripts/streamed_load_gate.py --model Qwen/Qwen2.5-7B \
        --out streamed_gate_7b.json
"""

from __future__ import annotations

import argparse
import json

from poros.validate.load_gate import digest_state as _digest_state  # noqa: F401
from poros.validate.load_gate import run
from poros.validate.load_gate import tensor_state as _tensor_state  # noqa: F401


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument(
        "--hash-compare", action="store_true",
        help="Compare sha256 digests of raw tensor bytes instead of held "
        "tensor copies — one model resident at a time (required at 72B).",
    )
    ap.add_argument("--out", default="streamed_load_gate.json")
    args = ap.parse_args()

    # The gate itself lives in poros.validate.load_gate, where `poros prove
    # <model> --unvalidated` runs it too.
    record = run(args.model, args.dtype, hash_compare=args.hash_compare)
    with open(args.out, "a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))
    if not record["byte_identical"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
