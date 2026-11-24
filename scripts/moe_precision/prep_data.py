"""The kit's data: OpenAssistant/oasst1's English first turns (a prompt and its top-ranked answer),
as chat rows and as plain text rows, split into a training file and a held-out file. The same
command gives the same bytes on every machine: rows are ordered by message id and the split is
by a hash of it.

    python prep_data.py --out data/      # needs `datasets` and one download (about 40 MB)

Writes chats_train.jsonl, chats_eval.jsonl (CUDA runs, Qwen3-30B-A3B-Instruct's chat template)
and text_train.jsonl, text_eval.jsonl (the Mac's OLMoE, a base model with no chat template),
and sha256.txt. The held-out files have 256 rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

EVAL_ROWS = 256


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=3000)
    args = ap.parse_args()
    from datasets import load_dataset

    rows = load_dataset("OpenAssistant/oasst1", split="train")
    by_id = {r["message_id"]: r for r in rows}
    pairs = []
    for r in rows:
        parent = by_id.get(r["parent_id"]) if r["parent_id"] else None
        if (r["role"] == "assistant" and r["rank"] == 0 and r["lang"] == "en" and parent
                and parent["parent_id"] is None and parent["lang"] == "en"
                and len(parent["text"]) + len(r["text"]) <= args.max_chars):
            pairs.append((r["message_id"], parent["text"].strip(), r["text"].strip()))
    pairs.sort()
    held = sorted(pairs, key=lambda p: hashlib.sha256(p[0].encode()).hexdigest())[:EVAL_ROWS]
    held_ids = {p[0] for p in held}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sums = []
    for kind in ("chats", "text"):
        for split, chosen in (("train", [p for p in pairs if p[0] not in held_ids]),
                              ("eval", [p for p in pairs if p[0] in held_ids])):
            path = out / f"{kind}_{split}.jsonl"
            with open(path, "w") as f:
                for _, q, a in chosen:
                    row = ({"messages": [{"role": "user", "content": q},
                                         {"role": "assistant", "content": a}]}
                           if kind == "chats" else {"text": f"Question: {q}\nAnswer: {a}"})
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}"
                        f"  ({len(chosen)} rows)")
    (out / "sha256.txt").write_text("\n".join(sums) + "\n")
    print("\n".join(sums))


if __name__ == "__main__":
    main()
