"""refcheck.py PEFT_MB1 FUSED_MB1 PEFT_MB2 OUT.json -- docs/validation/fused's rule, stated
before the numbers: PASS when the mean and the largest |dloss| between the fused and PEFT runs
(update by update, micro-batch 1) are both at or under the mean and the largest |dloss| between
PEFT's micro-batch 1 and micro-batch 2 runs."""

import json
import os
import sys


def losses(d):
    ev = [json.loads(line) for line in open(os.path.join(d, "events.jsonl"))]
    return [e["data"]["loss"] for e in ev if e.get("event_type") == "step"]


p1, f1, p2 = (losses(d) for d in sys.argv[1:4])
n = min(len(p1), len(f1), len(p2))
dev = [abs(a - b) for a, b in zip(f1[:n], p1[:n])]
noise = [abs(a - b) for a, b in zip(p2[:n], p1[:n])]
rec = {
    "updates": n,
    "dev_mean": sum(dev) / n,
    "dev_max": max(dev),
    "noise_mean": sum(noise) / n,
    "noise_max": max(noise),
}
rec["pass"] = bool(
    n > 0 and rec["dev_mean"] <= rec["noise_mean"] and rec["dev_max"] <= rec["noise_max"]
)
json.dump(rec, open(sys.argv[4], "w"), indent=1)
print("REFCHECK", json.dumps(rec))
