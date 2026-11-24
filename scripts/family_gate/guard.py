#!/usr/bin/env python3
"""guard.py HF_HOME_DIR LIMIT_MB TIMEOUT_S OUT.json -- CMD...: run CMD with HF_HOME=HF_HOME_DIR (a
fresh cache) and stop it the moment that cache holds more than LIMIT_MB (a refusal that should come
before any weight download must not fill the disk while the gate finds out), or after TIMEOUT_S. Its
stdout/stderr pass through. Writes {rc, seconds, killed (null | "download" | "timeout"),
cache_bytes, weight_files} to OUT.json; exits with CMD's code (124 when stopped)."""

import json
import os
import signal
import subprocess
import sys
import time

home, limit, timeout, out = sys.argv[1], float(sys.argv[2]) * 2**20, float(sys.argv[3]), sys.argv[4]
cmd = sys.argv[sys.argv.index("--") + 1 :]
os.makedirs(home, exist_ok=True)


def size():
    total, weights = 0, 0
    for root, _, files in os.walk(home):
        for f in files:
            try:
                total += os.lstat(os.path.join(root, f)).st_size
            except OSError:
                pass
            if f.endswith((".safetensors", ".bin", ".incomplete")):
                weights += 1
    return total, weights


t0 = time.time()
p = subprocess.Popen(cmd, env={**os.environ, "HF_HOME": home}, start_new_session=True)
killed = None
while p.poll() is None:
    time.sleep(0.5)
    b, _ = size()
    if b > limit:
        killed = "download"
    elif time.time() - t0 > timeout:
        killed = "timeout"
    if killed:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        break
b, w = size()
rec = {
    "rc": p.returncode if not killed else 124,
    "seconds": round(time.time() - t0, 1),
    "killed": killed,
    "cache_bytes": b,
    "weight_files": w,
}
json.dump(rec, open(out, "w"))
print("GUARD", json.dumps(rec), file=sys.stderr)
sys.exit(rec["rc"])
