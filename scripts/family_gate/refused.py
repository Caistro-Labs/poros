#!/usr/bin/env python3
"""refused.py LOG RUN_DIR RC WORDS -- a `poros train --json` run that must be refused before it
trains: exit code 2, a poros.error.v1 with code E_CONFIG whose message has WORDS, and no
training step in RUN_DIR (nothing trained, no adapter written). Prints `REFUSED {...}`; exit 0
iff all hold."""

import json
import sys
from pathlib import Path

log, run_dir, rc, words = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
err = {}
for line in open(log, errors="replace"):
    line = line.strip()
    if line.startswith("{") and line.endswith("}"):
        try:
            err = json.loads(line)
        except ValueError:
            pass
steps = 0
events = run_dir / "events.jsonl"
if events.exists():
    steps = sum(1 for line in events.read_text().splitlines()
                if line.strip() and json.loads(line).get("event_type") == "step")
adapters = [p.name for p in run_dir.rglob("adapter_*")] if run_dir.exists() else []
fails = []
if rc != 2:
    fails.append(f"exit {rc}, expected 2")
if err.get("schema") != "poros.error.v1" or err.get("code") != "E_CONFIG":
    fails.append(f"error {err.get('schema')!r} {err.get('code')!r}, expected E_CONFIG")
if words.lower() not in str(err.get("message", "")).lower():
    fails.append(f"message lacks the refusal's words: {str(err.get('message'))[:200]!r}")
if steps or adapters:
    fails.append(f"it trained: {steps} steps, {adapters}")
rec = {"rc": rc, "code": err.get("code"), "message": err.get("message"), "steps": steps,
       "fails": fails, "pass": not fails}
print("REFUSED", json.dumps(rec))
sys.exit(0 if not fails else 1)
