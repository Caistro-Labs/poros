"""What a run ended at, for the gates: one sha256 over its newest checkpoint's adapter tensors,
one over its optimizer state (moments, the remainder, the step), and every update's loss and grad
norm from events.jsonl. ``same`` compares runs and exits 1 where any of those differ.

    python digest.py runs/a                       # one run's digest, as JSON
    python digest.py same runs/a runs/b runs/c    # equal, all of them? exit 0 or 1
    python digest.py stopped runs/a               # stopped and resumed? exit 0 or 1

``same`` cannot run (exit 2, ERROR) on fewer than two runs, or on a run whose newest checkpoint
lacks its adapter or optimizer state: an empty hash compared with an empty hash is no check.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import torch


def newest_checkpoint(run: Path) -> Path:
    found = [(int(m.group(1)), p) for p in run.glob("checkpoint-*")
             if (m := re.fullmatch(r"checkpoint-(\d+)", p.name)) and p.is_dir()]
    if not found:
        raise SystemExit(f"{run}: no checkpoint-N folder")
    return max(found)[1]


def _hash_tensors(h, named: dict) -> None:
    for name in sorted(named):
        t = named[name]
        if torch.is_tensor(t):
            h.update(name.encode())
            h.update(str(t.dtype).encode())
            # reshape first: a 0-dim tensor (the optimizer's step) has no byte view of its own
            h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
                     .tobytes())


def _flatten(prefix: str, obj, out: dict) -> None:
    if torch.is_tensor(obj):
        out[prefix] = obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _flatten(f"{prefix}.{k}", v, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _flatten(f"{prefix}.{i}", v, out)


def _events(run: Path) -> list:
    from poros.core.events import current

    events = run / "events.jsonl"
    lines = events.read_text().splitlines() if events.is_file() else []
    return current(json.loads(line) for line in lines if line.strip())


def stopped(run: Path) -> dict:
    """Whether a run was stopped and resumed: a resume marker at a step below its last update,
    naming a checkpoint-N of this run with N its step less one (where the stop saved)."""
    raw = run / "events.jsonl"
    if not raw.is_file():
        raise SystemExit(f"{run}: no events.jsonl")
    events = [json.loads(line) for line in raw.read_text().splitlines() if line.strip()]
    steps = [e["step"] for e in events if e.get("event_type") == "step"
             and type(e.get("step")) is int]
    marks = [e for e in events if e.get("event_type") == "resume" and type(e.get("step")) is int]
    last = max(steps, default=None)
    found = [{"step": m["step"], "checkpoint": (m.get("data") or {}).get("checkpoint")}
             for m in marks]
    ok = [f for f in found if last is not None and f["step"] < last and f["checkpoint"]
          and Path(f["checkpoint"]).name == f"checkpoint-{f['step'] - 1}"]
    return {"status": "pass" if ok else "fail", "pass": bool(ok), "last_step": last,
            "resumes": found,
            "why": None if ok else "no resume marker below the last update naming the "
            "checkpoint the stop saved: the stop never landed, so nothing was resumed"}


def digest(run: Path) -> dict:
    ck = newest_checkpoint(run)
    adapter = hashlib.sha256()
    if not any((ck / n).is_file() for n in ("adapter_model.safetensors", "adapter_weights.pt")):
        raise SystemExit(f"{ck}: no adapter file (adapter_model.safetensors or "
                         "adapter_weights.pt) to compare")
    if not (ck / "optimizer_state.pt").is_file():
        raise SystemExit(f"{ck}: no optimizer_state.pt to compare")
    for name in ("adapter_model.safetensors", "adapter_weights.pt"):
        path = ck / name
        if path.is_file():
            if name.endswith(".safetensors"):
                from safetensors.torch import load_file
                tensors = load_file(path)
            else:
                tensors = torch.load(path, map_location="cpu", weights_only=True)
            _hash_tensors(adapter, tensors)
            break
    opt = hashlib.sha256()
    state: dict = {}
    if (ck / "optimizer_state.pt").is_file():
        _flatten("state", torch.load(ck / "optimizer_state.pt", map_location="cpu",
                                     weights_only=True)["state"], state)
        _hash_tensors(opt, state)
    losses, norms = [], []
    for e in _events(run):  # as the record stands: steps a resume took again counted once
        if e.get("event_type") == "step":
            losses.append(e["data"].get("loss"))
            norms.append(e["data"].get("grad_norm"))
    rule = ck / "optimizer.json"
    return {"run": str(run), "checkpoint": ck.name, "adapter_sha256": adapter.hexdigest(),
            "optimizer_state_sha256": opt.hexdigest(), "optimizer_tensors": len(state),
            "remainder_tensors": sum(k.endswith(".remainder") for k in state),
            "rule": json.loads(rule.read_text())["rule"] if rule.is_file() else None,
            "losses": losses, "grad_norms": norms}


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "stopped":
        if len(args) != 2:
            raise ValueError("stopped takes one run")
        said = stopped(Path(args[1]))
        print(json.dumps(said, indent=1))
        return 0 if said["pass"] else 1
    if args and args[0] == "same":
        if len(args) < 3:
            raise ValueError(f"same compares two runs or more; given {len(args) - 1}")
        found = [digest(Path(a)) for a in args[1:]]
        keys = ("adapter_sha256", "optimizer_state_sha256", "losses", "grad_norms")
        equal = {k: all(d[k] == found[0][k] for d in found) for k in keys}
        print(json.dumps({"status": "pass" if all(equal.values()) else "fail",
                          "pass": all(equal.values()), "equal": equal,
                          "runs": [{k: d[k] for k in ("run", "checkpoint", "adapter_sha256",
                                                      "optimizer_state_sha256", "rule",
                                                      "remainder_tensors")} for d in found]},
                         indent=1))
        return 0 if all(equal.values()) else 1
    for a in args:
        print(json.dumps(digest(Path(a)), indent=1))
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except (Exception, SystemExit) as exc:  # could not run: said so, never a FAIL
        import traceback

        print(json.dumps({"status": "error", "pass": None,
                          "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()}, indent=1))
        sys.exit(2)
    sys.exit(code)
