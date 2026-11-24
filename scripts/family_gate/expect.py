#!/usr/bin/env python3
"""expect.py RUN_DIR LOG OUT.json SPEC... -- what one `poros train` run says it ran, against what
the gate expects.

Reads RUN_DIR/config.resolved.json (ran_path, path_reason, offload_reason, ran_compute,
compute_reason), the setup event of RUN_DIR/events.jsonl, and the last JSON object of LOG (`--json`:
support, model_type). SPEC items (all must hold):
  path=P          ran_path == P and no path_reason (the path asked is the path that ran, no
                  fallback)
  ran=P           ran_path == P, whatever decided it (a path left unset: the run decides)
  path_reason~TEXT  path_reason contains TEXT (case-insensitive)
  fused=True|False   ran_compute.fused_compute
  moe=fused|reference  ran_compute.moe_expert_compute
  reason=none     compute_reason is null;  reason~TEXT: compute_reason contains TEXT
                  (case-insensitive)
  offload=none    offload_reason is null
  cache=efficient|sequential|sized
                  moe_expert_cache_reason: "one layer: path efficient...", "one layer: the
                  sequential loop...", or sized to the GPU ("fits this GPU" / all experts), not
                  one layer
  support=X       the result's support (null when validated)
Prints `EXPECT {...}` with every field read and the failures; exit 0 iff none failed."""

import json
import os
import sys

run, log, out = sys.argv[1:4]
spec = sys.argv[4:]
rec: dict = {"run": os.path.basename(run.rstrip("/"))}
try:
    c = json.load(open(os.path.join(run, "config.resolved.json")))
except (OSError, ValueError):
    c = {}
rc = c.get("ran_compute") or {}
setup = {}
try:
    for line in open(os.path.join(run, "events.jsonl")):
        e = json.loads(line)
        if e.get("event_type") == "setup":
            setup = e.get("data") or {}
            break
except OSError:
    pass
result = {}
try:
    for line in open(log):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                result = json.loads(line)
            except ValueError:
                pass
except OSError:
    pass
rec.update(
    path=c.get("path"),
    ran_path=c.get("ran_path"),
    path_reason=c.get("path_reason"),
    offload_reason=c.get("offload_reason"),
    offload=c.get("offload_embeddings_and_head"),
    fused_compute=rc.get("fused_compute"),
    moe_expert_compute=rc.get("moe_expert_compute"),
    compute_reason=c.get("compute_reason"),
    moe_expert_cache_gib=rc.get("moe_expert_cache_gib"),
    moe_expert_cache_reason=rc.get("moe_expert_cache_reason"),
    nf4_dequant_cache_mb=rc.get("nf4_dequant_cache_mb"),
    setup={k: v for k, v in setup.items() if k not in ("model",)},
    support=result.get("support"),
    model_type=result.get("model_type"),
    schema=result.get("schema"),
)
fails = []
if not c:
    fails.append("no config.resolved.json")
for item in spec:
    if not item.strip():
        continue
    if "=" not in item and "~" not in item:
        fails.append(f"bad spec item {item!r}")
        continue
    if "~" in item and "=" not in item.split("~")[0]:
        k, v = item.split("~", 1)
        if k == "reason":
            if v.lower() not in str(rec["compute_reason"] or "").lower():
                fails.append(f"compute_reason lacks {v!r}: {rec['compute_reason']!r}")
        elif k == "path_reason":
            if v.lower() not in str(rec["path_reason"] or "").lower():
                fails.append(f"path_reason lacks {v!r}: {rec['path_reason']!r}")
        else:
            fails.append(f"unknown spec item {item}")
        continue
    k, v = item.split("=", 1)
    if k == "path":
        if rec["ran_path"] != v or rec["path_reason"]:
            fails.append(
                f"ran_path {rec['ran_path']!r} (asked {v}), path_reason {rec['path_reason']!r}"
            )
    elif k == "ran":
        if rec["ran_path"] != v:
            fails.append(f"ran_path {rec['ran_path']!r}, expected {v}")
    elif k == "fused":
        if str(rec["fused_compute"]) != v:
            fails.append(f"fused_compute {rec['fused_compute']!r}, expected {v}")
    elif k == "moe":
        if rec["moe_expert_compute"] != v:
            fails.append(f"moe_expert_compute {rec['moe_expert_compute']!r}, expected {v}")
    elif k == "reason" and v == "none":
        if rec["compute_reason"]:
            fails.append(f"compute_reason {rec['compute_reason']!r}, expected none")
    elif k == "offload" and v == "none":
        if rec["offload_reason"]:
            fails.append(f"offload_reason {rec['offload_reason']!r}, expected none")
    elif k == "cache":
        why = str(rec["moe_expert_cache_reason"] or "")
        ok = {
            "efficient": why.startswith("one layer: path efficient"),
            "sequential": why.startswith("one layer: the sequential loop"),
            "sized": bool(why) and not why.startswith("one layer"),
        }[v]
        if not ok:
            fails.append(
                f"moe_expert_cache_reason {why!r} ({rec['moe_expert_cache_gib']} GiB), expected {v}"
            )
    elif k == "support":
        if str(rec["support"]) != v:
            fails.append(f"support {rec['support']!r}, expected {v}")
    else:
        fails.append(f"unknown spec item {item}")
rec["spec"], rec["fails"], rec["pass"] = spec, fails, not fails
json.dump(rec, open(out, "w"), indent=1)
print("EXPECT", json.dumps(rec))
sys.exit(0 if not fails else 1)
