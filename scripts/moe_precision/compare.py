"""The verdict of one quality comparison: the new engine against the fp32 reference, with the old
engine beside them. Reads each run's events.jsonl (held-out loss every eval_steps, every update's
loss) and the kit's sample.jsonl.

    python compare.py --new runs/q_new_cos_s42 --fp32 runs/q_fp32_cos_s42 \
        --fp32-other runs/q_fp32_cos_s43 [--new-other runs/q_new_cos_s43] \
        [--control runs/q_ctl_cos_s42] [--old runs/q_old_cos_s42] [--commit 1a8d7e912]

Pass, all of:
  1. at every held-out point after step 0, |new - fp32| <= max(|fp32 - fp32 other seed|, 0.005)
     (the band: what changing only the data order and the init moves the loss by, at least
     0.005 nats); at the last point too
  2. update 1's loss and grad norm are equal in the new and the old run (forward and backward
     are untouched; only the update rule changed) where --old is given
  3. the new run's v tracks its grads: v_over_exact_median within 1 +- 0.05 at every sampled
     update from 20 on
  4. with --commit: every run judged or published (new, fp32, fp32 other, new other, control)
     names that commit in its .kit_commit: the release engine's, and each the same
  5. with --control (torch's AdamW stepping the bf16 adapters in bf16 on the same engine, the
     kit's control arm): its update 1 loss and grad norm equal the new run's (the same engine,
     the optimizer alone differs), and its .kit_arm says it is the control
The control's and the old run's figures are reported, not judged: at 2e-5 they are expected to
fall short. The old run (a checkout from before ExpertAdamW) is not the release engine: what is
published is the control's.
Prints one JSON object; exit 0 on pass, 1 on fail.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BAND = 0.005


def evals(run: Path) -> dict:
    out = {}
    for line in (run / "events.jsonl").read_text().splitlines():
        e = json.loads(line)
        if e.get("event_type") == "eval":
            d = e.get("data") or {}
            loss = d.get("eval_loss", d.get("loss"))
            if loss is not None:
                out[int(e["step"])] = float(loss)
    return out


def steps(run: Path) -> list:
    return [json.loads(x)["data"] for x in (run / "events.jsonl").read_text().splitlines()
            if json.loads(x).get("event_type") == "step"]


def samples(run: Path) -> list:
    path = run / "sample.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.is_file() else []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for name in ("new", "fp32", "fp32-other"):
        ap.add_argument(f"--{name}", required=True, type=Path)
    ap.add_argument("--new-other", type=Path)
    ap.add_argument("--old", type=Path)
    ap.add_argument("--control", type=Path)
    ap.add_argument("--commit", help="the release commit every judged run must name")
    args = ap.parse_args()
    e_new, e_ref, e_ref2 = evals(args.new), evals(args.fp32), evals(args.fp32_other)
    points = sorted(set(e_new) & set(e_ref) & set(e_ref2) - {0})
    rows, ok1 = [], bool(points)
    for s in points:
        band = max(abs(e_ref[s] - e_ref2[s]), BAND)
        diff = e_new[s] - e_ref[s]
        ok1 &= abs(diff) <= band
        rows.append({"step": s, "new": e_new[s], "fp32": e_ref[s], "fp32_other": e_ref2[s],
                     "new_minus_fp32": diff, "band": band})
    report = {"held_out": rows, "check_1_within_band": ok1}
    if args.new_other:
        e2 = evals(args.new_other)
        report["new_other_seed_last"] = e2.get(max(e2)) if e2 else None
    ok2 = True
    if args.old:
        e_old = evals(args.old)
        report["old_held_out"] = {s: e_old.get(s) for s in points}
        a, b = steps(args.new), steps(args.old)
        ok2 = bool(a and b) and a[0].get("loss") == b[0].get("loss") and \
            a[0].get("grad_norm") == b[0].get("grad_norm")
        report["update_1"] = {"new": [a[0].get("loss"), a[0].get("grad_norm")] if a else None,
                              "old": [b[0].get("loss"), b[0].get("grad_norm")] if b else None}
        so = samples(args.old)
        report["old_v_over_exact_last"] = so[-1]["v_over_exact_median"] if so else None
        report["old_weight_unchanged_last"] = so[-1]["weight_unchanged"] if so else None
    report["check_2_update_1_equal"] = ok2
    sn = samples(args.new)
    late = [x["v_over_exact_median"] for x in sn if x["update"] >= 20
            and x["v_over_exact_median"] is not None]
    ok3 = bool(late) and all(abs(r - 1) <= 0.05 for r in late)
    report["new_v_over_exact_range"] = [min(late), max(late)] if late else None
    report["new_weight_unchanged_mean"] = (sum(x["weight_unchanged"] for x in sn) / len(sn)
                                           if sn else None)
    report["check_3_v_tracks"] = ok3
    ok4 = True
    if args.commit:
        named = {}
        for key in ("new", "fp32", "fp32_other", "new_other", "control"):
            run = getattr(args, key)
            if run is not None:
                f = run / ".kit_commit"
                named[key] = f.read_text().strip() if f.is_file() else None
        ok4 = all(c is not None and c.startswith(args.commit) for c in named.values())
        report["commits"] = named
    report["check_4_release_commit"] = ok4
    ok5 = True
    if args.control:
        arm = args.control / ".kit_arm"
        said = arm.read_text().strip() if arm.is_file() else None
        a, c = steps(args.new), steps(args.control)
        ok5 = (bool(a and c) and a[0].get("loss") == c[0].get("loss")
               and a[0].get("grad_norm") == c[0].get("grad_norm")
               and bool(said) and said.startswith("control"))
        e_ctl = evals(args.control)
        sc = samples(args.control)
        report["control"] = {
            "arm": said, "held_out": {s: e_ctl.get(s) for s in points},
            "update_1": [c[0].get("loss"), c[0].get("grad_norm")] if c else None,
            "optimizer": sc[-1].get("optimizer") if sc else None,
            "v_over_exact_last": sc[-1]["v_over_exact_median"] if sc else None,
            "weight_unchanged_mean": (sum(x["weight_unchanged"] for x in sc) / len(sc)
                                      if sc else None)}
    report["check_5_control_same_engine"] = ok5
    report["pass"] = ok1 and ok2 and ok3 and ok4 and ok5
    report["status"] = "pass" if report["pass"] else "fail"
    print(json.dumps(report, indent=1))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:  # the check could not run: said so, never a FAIL of the property
        import traceback

        print(json.dumps({"status": "error", "pass": None,
                          "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()}, indent=1))
        sys.exit(2)
    sys.exit(code)
