#!/usr/bin/env python3
"""resume_cmp.py FULL_DIR RESUMED_DIR FROM_STEP OUT.json -- a resumed `poros train` run against the
same run uninterrupted, bit for bit: every step the resumed run logged after FROM_STEP has the
uninterrupted run's loss and grad norm (exact float equality), and the final adapter tensors are
torch.equal (sha256 digest over all of them, as compare_runs.py)."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402

full, res, k, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
argv, sys.argv = sys.argv, [sys.argv[0], full, res, os.devnull]
import compare_runs as cr  # noqa: E402  (its module body compares FULL and RESUMED; we reuse its helpers)

sys.argv = argv


def steps(d):
    ev = [json.loads(line) for line in open(os.path.join(d, "events.jsonl"))]
    return {
        e["step"]: (e["data"].get("loss"), e["data"].get("grad_norm"))
        for e in ev
        if e.get("event_type") == "step"
    }


sf, sr = steps(full), steps(res)
after = sorted(s for s in sr if s >= k)
(af, _), (ar, _) = cr.adapters(full), cr.adapters(res)
uneq = [n for n in sorted(af) if n not in ar or not torch.equal(af[n], ar[n])]
rec = {
    "full": full,
    "resumed": res,
    "from_step": k,
    "resumed_steps": after,
    "full_steps": sorted(sf),
    "steps_equal": bool(after) and all(sr[s] == sf.get(s) for s in after),
    "last_step_equal": bool(after) and max(after) == max(sf),
    "adapter_tensors": len(af),
    "adapter_keys_equal": sorted(af) == sorted(ar),
    "adapter_unequal_n": len(uneq),
    "full_digest": cr.digest(af),
    "resumed_digest": cr.digest(ar),
    "losses": {s: [sf.get(s), sr[s]] for s in after},
}
rec["pass"] = bool(
    rec["steps_equal"] and rec["last_step_equal"] and rec["adapter_keys_equal"] and not uneq
)
json.dump(rec, open(out, "w"), indent=1)
print(
    "RESUME",
    json.dumps(
        {
            x: rec[x]
            for x in (
                "pass",
                "resumed_steps",
                "steps_equal",
                "adapter_tensors",
                "adapter_unequal_n",
                "full_digest",
                "resumed_digest",
            )
        }
    ),
)
