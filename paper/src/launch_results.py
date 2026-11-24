#!/usr/bin/env python3
"""Every measured value of the Poros paper, read from benchmark logs; the result tables; Figure 1.

The paper never types a measured number.  It prints \\nv{key}, and this script writes values.tex
with one \\nvset{key}{value} per value it can read from a log.  A key it cannot fill stays unset,
and LaTeX prints a visible marker in its place; `make final` then refuses to build.

One commit only.  Every Poros figure, and every log this script reads but one, ran the release engine,
commit 1a8d7e912b166c373fcfec862ad7297221d33a71 (RELEASE_COMMITS).  The one other log is the other
tool's arm of the November 18 head-to-head (h2h/4090/arms/a30_ga8_uns.json), read for that tool's own
figures only (h2h_previous_unsloth()), so its earlier version can stand beside the release commit's comparison.  A block of logs is used only when the
file that names its commit names that one; inside a block in use, any file that names another commit
stops the script.  Logs of other commits at the same paths are listed in pending_values.txt as not
used and never read for a value.

Where the logs are (each found by the commit it names, not by a fixed folder name), under
$POROS_BENCHMARKS/docs/validation/, a checkout of github.com/Caistro-Labs/poros-benchmarks:
  h2h/*/meta.json                       the head-to-head against Unsloth (two default arms a model)
  int/4090/*/README.md                  the RTX 4090 gates and studies: the family gate, poros prove
                                        on a 7B model, the dropout-on checks, the optimizer self-test
                                        and the two optimizer studies
  recipes/{rtx4090,rtx4060-laptop,m4-16gb}/*/certification.json   the recipe runs (certified where
                                        poros/recipes/certified.json lists the recipe on that machine)
  int/mac/*/                            the Mac's 144-step Qwen2.5-72B-Instruct run
$POROS_MAC_LOGS and $POROS_LAPTOP_LOGS, where set, are further folders read the same way (the
Mac's and the laptop's self-tests and runs outside a checkout of the benchmark logs).

Writes: values.tex, table_{h2h,setup,paths,exact,recipes,opt}.tex, figure1_h2h.pdf,
launch_values.json (every value, its source files and field) and pending_values.txt.

Usage:
  python3 launch_results.py           missing values become visible markers in the PDF
  python3 launch_results.py --final   exits 1 if any key used in the paper is not filled,
                                      or if a log contradicts a sentence of the paper
Needs matplotlib, not torch.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCH = pathlib.Path(os.environ.get("POROS_BENCHMARKS") or ROOT).resolve()
VAL = BENCH / "docs/validation"
if not VAL.is_dir():
    raise SystemExit(f"no benchmark logs at {VAL}: set POROS_BENCHMARKS to the root of a checkout of "
                     "github.com/Caistro-Labs/poros-benchmarks")
_NOWHERE = HERE / ".no-such-folder"  # an unset extra log folder: globs over it find nothing
MACLOGS = pathlib.Path(os.environ["POROS_MAC_LOGS"]).resolve() if os.environ.get("POROS_MAC_LOGS") else _NOWHERE
LAPLOGS = pathlib.Path(os.environ["POROS_LAPTOP_LOGS"]).resolve() if os.environ.get("POROS_LAPTOP_LOGS") else _NOWHERE
FINAL = "--final" in sys.argv

FROZEN = "1a8d7e912b166c373fcfec862ad7297221d33a71"
RELEASE_COMMITS = {FROZEN[:8]}


VALUES: dict[str, str] = {}
SOURCES: dict[str, list[str]] = {}
FIELD: dict[str, str] = {}
RAW: dict[str, object] = {}
PROBLEMS: list[str] = []
NOT_USED: list[str] = []   # logs at the expected places that name another commit

# Where each key that is still pending will come from (pending_values.txt).
EXPECT = {
    "code.": "h2h/<dir>/versions.txt at the frozen commit (RTX 4090 head-to-head)",
    "prev.": "h2h/4090/arms/a30_ga8_uns.json and versions.txt (the other tool's earlier version, November 18)",
    "env.": "h2h/<dir>/meta.json and versions.txt at the frozen commit",
    "card.": "h2h/<dir>/runs/a72_ga8_uns_r1_raw.json (card_gib)",
    "wl.": "h2h/<dir>/runs/a72_ga8_poros_r1_config.resolved.json and gates/rows_check_72b.json",
    "dense.": "h2h/<dir>/arms/a72_ga8_{poros,uns}.json (three runs each)",
    "moe.": "h2h/<dir>/arms/a30_ga8_{ram,uns}.json (three runs each)",
    "loss.": "h2h/<dir>/runs/a72_ga8_poros_r1.json, a30_ga8_ram_r1.json and the Unsloth raw runs",
    "prove.": "int/4090/<round>/prove*/prove.json (poros prove on the 7B model)",
    "fg.": "int/4090/<round>/family_gate/RESULT.txt",
    "do.": "int/4090/<round>/block_major/rec/cmp_*.json (the dropout-on checks: 7B, OLMoE, 32B)",
    "st.4090.": "int/4090/<round>/**/selftest*.txt (device_selftest.py --device cuda --time)",
    "st.4060l.": "the RTX 4060 Laptop GPU's selftest_cuda.txt (device_selftest.py --device cuda --time)",
    "st.mac.": "the Mac's selftest_mps.txt (device_selftest.py --device mps --time)",
    "st.elems": "any device's selftest time line",
    "olmoe.": "the RTX 4090 OLMoE study: int/4090/<round>/**/verdicts/cuda_olmoe.json and runs/o_*_olmoe_*",
    "q30.": "the RTX 4090 Qwen3-30B-A3B study: int/4090/<round>/**/verdicts/q2e5.json and runs/q_*_c2e5_*",
    "lap.": "recipes/rtx4060-laptop/moe-30b-8gb/certification.json and its repeat",
    "rec.": "recipes/*/*/certification.json at the frozen commit (10 RTX 4090, 2 laptop, 8 Mac) and poros/recipes/certified.json",
    "recipe.": "recipes/rtx4090/big-72b/certification.json",
    "r30": "recipes/rtx4090/moe-30b{,-ssd}/certification.json",
    "mac": "recipes/m4-16gb/*/certification.json (Mac); mac72.drive from model-drive.txt beside it",
    "m144.": "the Mac's 144-step run at the frozen commit (receipt.json, events.jsonl, eval.json, model-drive.txt)",
    "paths.": "recipes/rtx4090/big-72b{,-fast}/main/config.resolved.json",
    "fig1.": "Figure 1 is complete when its four points are filled",
}


# ── helpers ────────────────────────────────────────────────────────────────
def rel(p: pathlib.Path) -> str:
    q = pathlib.Path(p).resolve()
    for base, pre in ((BENCH, ""), *((r.parents[1], "") for r in ROUND_ROOTS[1:]), (ROOT, ""), (MACLOGS, "$POROS_MAC_LOGS/"), (LAPLOGS, "$POROS_LAPTOP_LOGS/")):
        try:
            return pre + str(q.relative_to(base))
        except ValueError:
            pass
    return str(q)


def load_json(p):
    return json.loads(pathlib.Path(p).read_text())


def tex_escape(s) -> str:
    return (str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&")
            .replace("%", "\\%").replace("#", "\\#").replace("$", "\\$"))


def as_tex(v) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return str(v)
    return tex_escape(v)


def pr(key: str, value, path, field: str, f=None) -> None:
    """Put a paper key read from a log file, with the field it came from."""
    VALUES[key] = str(f(value) if f else as_tex(value))
    SOURCES[key] = [rel(path)] if not isinstance(path, (list, tuple)) else [rel(p) for p in path]
    FIELD[key] = field
    RAW[key] = value


def same(xs, what):
    xs = list(xs)
    if not xs or any(x != xs[0] for x in xs):
        PROBLEMS.append(f"{what}: the runs disagree ({xs})")
    return xs[0] if xs else None


def sci(v: float) -> str:
    m, e = f"{v:e}".split("e")
    m = m.rstrip("0").rstrip(".")
    return f"${m}\\times10^{{{int(e)}}}$"


def short(commit) -> str:
    return str(commit or "")[:8]


def is_frozen(commit) -> bool:
    return short(commit) in RELEASE_COMMITS


def release(commit, where) -> None:
    """Inside a block in use: a file that names another commit stops the script."""
    if not is_frozen(commit):
        raise SystemExit(f"{rel(where)} names commit {commit}, which is not the frozen release engine "
                         f"{FROZEN[:8]}")


def readme_commit(p: pathlib.Path):
    m = re.search(r"\b([0-9a-f]{40})\b", p.read_text()) if p.exists() else None
    return m.group(1) if m else None


# ── the RTX 4090 round at the frozen commit (found by the commit its README names) ──
# $POROS_4090_LOGS: a further docs/validation/ root searched for the RTX 4090's rounds only (a checkout of
# the branch the 4090's logs are committed to, before they reach the benchmark-logs tree).
ROUND_ROOTS = [VAL] + ([pathlib.Path(os.environ["POROS_4090_LOGS"]).resolve()] if os.environ.get("POROS_4090_LOGS") else [])


def rounds_4090() -> list[pathlib.Path]:
    out = []
    for rd in sorted(r for root in ROUND_ROOTS for r in root.glob("int/4090/*/README.md")):
        t = rd.read_text()
        if FROZEN in t or re.search(rf"\b{FROZEN[:9]}", t):
            out.append(rd.parent)
    return out


ROUNDS: list[pathlib.Path] = []


def in_rounds(pattern: str) -> list[pathlib.Path]:
    return sorted(p for r in ROUNDS for p in r.glob(pattern))


# ── the head-to-head: Poros's default path against Unsloth's automatic offloading ──
H2H: pathlib.Path | None = None
# (paper key suffix, the arm file's median field); host memory's suffix is "host" in the paper's keys
METRICS = (("supd", "s_per_step_s"), ("alloc", "vram_alloc_gb"), ("res", "vram_reserved_gb"),
           ("host", "host_ram_gb"), ("host.train", "host_ram_train_gb"))
# metric -> (median field, "less X" wording for memory, or the speed wording)
VS = {"s_per_step": ("s_per_step_s", "faster"), "vram_reserved": ("vram_reserved_gb", "less VRAM"),
      "host_ram": ("host_ram_gb", "less host RAM"), "host_ram_train": ("host_ram_train_gb", "less host RAM while training")}


def ratio_text(poros: float, other: float, what: str, other_name: str = "Unsloth") -> tuple[str, str]:
    """The comparison's rule (Section 5.1): larger over smaller, to one decimal (a whole percent under
    1.1 times), rounded against Poros: down where Poros is ahead, up where the other tool is ahead.
    Lower is better for every metric here.  Returns (text, ratio)."""
    lo, hi = sorted((poros, other))
    r = hi / lo
    ahead = poros < other
    who = "Poros" if ahead else other_name
    rnd = math.floor if ahead else math.ceil
    if r < 1.1:
        pct = rnd(round((r - 1) * 100, 9))
        if pct == 0:
            return "the same", "1.0$\\times$"
        return f"{who} {pct}\\% {what}", f"{1 + pct / 100:.2f}$\\times$"
    rr = f"{rnd(round(r * 10, 9)) / 10:.1f}$\\times$"
    return f"{who} {rr} {what}", rr

def h2h_dir() -> pathlib.Path | None:
    for mp in sorted(VAL.glob("h2h/*/meta.json")):
        c = load_json(mp).get("commit")
        if is_frozen(c):
            return mp.parent
        if mp.parent.name.startswith("4090"):
            NOT_USED.append(f"{rel(mp)} (commit {short(c)})")
    return None


def best(runs, field):
    return min(r[field] for r in runs)


def h2h() -> None:
    global H2H
    H2H = h2h_dir()
    if H2H is None:
        return
    meta = load_json(H2H / "meta.json")
    release(meta["commit"], H2H / "meta.json")
    pr("code.commit", meta["commit"][:8], H2H / "meta.json", "commit", lambda v: f"\\texttt{{{v}}}")
    m = meta["machine_4090"]
    for k, fld in (("env.gpu", "gpu"), ("env.gpu.mib", "gpu_mib"), ("env.cpu", "cpu"), ("env.host.gib", "ram_gib"),
                   ("env.os", "os"), ("env.driver", "driver")):
        pr(k, m[fld], H2H / "meta.json", f"machine_4090.{fld}")
    import ast
    vt = (H2H / "versions.txt").read_text()
    m_ = re.search(r"^commit (\w+)", vt, re.M)
    release(m_.group(1) if m_ else None, H2H / "versions.txt")
    pst = ast.literal_eval(re.search(r"^poros stack: (\{.*?\})", vt, re.M).group(1))
    ust = ast.literal_eval(re.search(r"^unsloth stack: (\{.*?\})", vt, re.M).group(1))
    for k, fld in (("env.torch", "torch"), ("env.transformers", "transformers"), ("env.peft", "peft"),
                   ("env.bnb", "bitsandbytes")):
        pr(k, pst[fld], H2H / "versions.txt", f"poros stack.{fld}")
    pr("env.cuda", re.search(r"^poros stack: \{.*\} cuda (\S+)", vt, re.M).group(1), H2H / "versions.txt", "poros stack cuda")
    pr("code.version", re.search(r"^poros (\S+)$", vt, re.M).group(1), H2H / "versions.txt", "poros")
    for k, fld in (("env.unsloth", "unsloth"), ("env.unsloth.transformers", "transformers"), ("env.unsloth.peft", "peft")):
        pr(k, ust[fld], H2H / "versions.txt", f"unsloth stack.{fld}")

    arms = {}
    for short_, arm in (("dense.poros", "a72_ga8_poros"), ("dense.unsloth", "a72_ga8_uns"),
                        ("moe.ram", "a30_ga8_ram"), ("moe.unsloth", "a30_ga8_uns")):
        ap = H2H / f"arms/{arm}.json"
        if not ap.exists():
            PROBLEMS.append(f"{rel(ap)} is missing from the head-to-head at the frozen commit")
            continue
        d = load_json(ap)
        release(d["commit"], ap)
        if len(d["runs"]) != 3:
            PROBLEMS.append(f"{rel(ap)}: {len(d['runs'])} runs, not three")
        arms[short_] = d
        for k, f in METRICS:
            pr(f"{short_}.{k}", d["median"][f], ap, f"median.{f}")
        for k, f in (("supd", "s_per_step_s"), ("res", "vram_reserved_gb"), ("host", "host_ram_gb"),
                     ("host.train", "host_ram_train_gb")):
            pr(f"{short_}.{k}.best", best(d["runs"], f), ap, f"min over runs[].{f}")
    pr("env.sampler.s", same((r["host_sampler"]["interval_s"] for r in arms["dense.poros"]["runs"]), "sampler"),
       H2H / "arms/a72_ga8_poros.json", "runs[].host_sampler.interval_s") if "dense.poros" in arms else None
    if "dense.poros" in arms:
        d = arms["dense.poros"]
        pr("dense.poros.load.s", d["median"]["load_s"], H2H / "arms/a72_ga8_poros.json", "median.load_s")
        pr("dense.poros.load.res", d["median"]["load_vram_reserved_gb"], H2H / "arms/a72_ga8_poros.json",
           "median.load_vram_reserved_gb")
    # comparisons: Poros's median against Unsloth's best of three (the paper's rule)
    for p_, u_ in (("dense.poros", "dense.unsloth"), ("moe.ram", "moe.unsloth")):
        if p_ not in arms or u_ not in arms:
            continue
        for m_, (f, what) in VS.items():
            text, ratio = ratio_text(arms[p_]["median"][f], best(arms[u_]["runs"], f), what)
            srcs = [H2H / f"arms/{a}.json" for a in (("a72_ga8_poros", "a72_ga8_uns") if p_.startswith("dense")
                                                      else ("a30_ga8_ram", "a30_ga8_uns"))]
            pr(f"{p_}.vs.{m_}", text, srcs, f"median.{f} against min over Unsloth runs[].{f}", lambda v: v)
            if m_ == "s_per_step":
                pr(f"{p_}.vs.{m_}.ratio", ratio, srcs, "larger / smaller, to one decimal, rounded against Poros", lambda v: v)
    # the workload, from Poros's resolved configuration and Unsloth's record
    cp = H2H / "runs/a72_ga8_poros_r1_config.resolved.json"
    if cp.exists():
        c = load_json(cp)
        for k, fld, f in (("wl.rank", "lora_rank", None), ("wl.seq", "max_seq_length", None),
                          ("wl.ga", "gradient_accumulation_steps", None), ("wl.alpha", "lora_alpha", None),
                          ("wl.dropout", "lora_dropout", None), ("wl.mb", "per_device_batch_size", None),
                          ("wl.updates", "max_steps", None), ("wl.lr", "learning_rate", sci),
                          ("wl.wd", "weight_decay", None), ("wl.clip", "max_grad_norm", None),
                          ("wl.sched", "lr_scheduler", None), ("wl.seed", "seed", None)):
            pr(k, c[fld], cp, fld, f)
        pr("wl.betas", f"{c['adam_beta1']}, {c['adam_beta2']}", cp, "adam_beta1, adam_beta2")
    rcp = H2H / "gates/rows_check_72b.json"
    if rcp.exists():
        rc = load_json(rcp)
        if not (rc["pass"] and rc["poros_full_rows"] and set(rc["unsloth"].values()) == {rc["poros_fed_sha256"]}):
            PROBLEMS.append("rows_check_72b: the two tools were not fed the same rows")
        pr("wl.rows", rc["poros_fed_rows"], rcp, "poros_fed_rows")
    for name in ("rows_check_30b_ga8",):
        p = H2H / f"gates/{name}.json"
        if p.exists():
            r30 = load_json(p)
            if not (r30["pass"] and r30["poros_full_rows"] and set(r30["unsloth"].values()) == {r30["poros_fed_sha256"]}):
                PROBLEMS.append(f"{name}: the two tools were not fed the same rows")
    up = H2H / "runs/a72_ga8_uns_r1_raw.json"
    if up.exists():
        u = load_json(up)
        a = u["args"]
        if not a["deterministic"]:
            PROBLEMS.append("Unsloth's 72B run was not in deterministic mode")
        for k, fld, f in (("wl.unsloth.rank", "rank", None), ("wl.unsloth.lr", "lr", sci), ("wl.unsloth.wd", "wd", None),
                          ("wl.unsloth.clip", "clip", None), ("wl.unsloth.seed", "seed", None)):
            pr(k, a[fld], up, f"args.{fld}", f)
        pr("wl.unsloth.gc", a["gc"], up, "args.gc", lambda v: f"\\code{{{tex_escape(v)}}}")
        pr("wl.unsloth.targets", a["targets"].replace("_proj", "").replace(",", ", "), up, "args.targets")
        pr("wl.trainable72", u["trainable_params"], up, "trainable_params", lambda n: f"{int(n):,}")
        pr("card.gib", u["card_gib"], up, "card_gib")
    for short_, key in (("dense.unsloth", "a72_ga8_uns"), ("moe.unsloth", "a30_ga8_uns")):
        if short_ in arms:
            ap = H2H / f"arms/{key}.json"
            d = arms[short_]
            pr(f"{short_}.offloaded", same((r["unsloth"]["swapped_layers"] for r in d["runs"]), f"{key} offloaded"),
               ap, "runs[].unsloth.swapped_layers")
            pr(f"{short_.split('.')[0]}.layers", same((r["unsloth"]["decoder_layers"] for r in d["runs"]), f"{key} layers"),
               ap, "runs[].unsloth.decoder_layers")
            if short_ == "dense.unsloth":
                pr("dense.unsloth.embed", same((r["unsloth"]["input_embedding_device"] for r in d["runs"]), "embed"),
                   ap, "runs[].unsloth.input_embedding_device")
    if "moe.ram" in arms:
        ap = H2H / "arms/a30_ga8_ram.json"
        d = arms["moe.ram"]
        pr("moe.model", d["model"].split("/", 1)[1], ap, "model")
        arch = same((r["poros"]["arch"] for r in d["runs"]), "30B arch")
        mm = re.fullmatch(r"MoE · (\d+) layers · (\d+) experts", arch or "")
        if mm:
            pr("moe.experts", int(mm.group(2)), ap, "runs[].poros.arch")
        for r in d["runs"]:
            if r["poros"].get("moe_expert_compute") != "fused":  # Section 2.5 says every one did
                PROBLEMS.append(f"{rel(ap)}: a run did not record the fused expert kernels")
            if r["poros"]["ran_path"] != "efficient" or r["poros"]["store"] != "ram":
                PROBLEMS.append(f"{rel(ap)}: a run is not the default path (efficient, experts in host memory)")
    # adapter dtypes and the expert adapters' GPU memory (weight, gradient, m, v, remainder: 5 x 2 bytes)
    dt = lambda xs: {"torch.float32": "fp32", "torch.bfloat16": "bf16"}[same(xs, "dtype")]
    for key, path, fld in (("dense.adapter.dtype", "runs/a72_ga8_poros_r1_digest.json", "dtypes"),
                           ("moe.adapter.dtype", "runs/a30_ga8_ram_r1.json", "trainable.dtypes"),
                           ("dense.unsloth.adapter.dtype", "runs/a72_ga8_uns_r1_raw.json", "trainable_dtypes"),
                           ("moe.unsloth.adapter.dtype", "runs/a30_ga8_uns_r1_raw.json", "trainable_dtypes")):
        p = H2H / path
        if p.exists():
            d = load_json(p)
            v = d
            for part in fld.split("."):
                v = v[part]
            pr(key, dt(v), p, fld)
    p = H2H / "runs/a30_ga8_ram_r1.json"
    if p.exists():
        tr = load_json(p)["trainable"]
        pr("moe.trainable", tr["params"], p, "trainable.params", lambda n: f"{int(n):,}")
        if tr["dtypes"] == ["torch.bfloat16"]:
            pr("moe.adapter.state.gib", tr["params"] * 5 * 2 / 2**30, p,
               "trainable.params x 5 bf16 tensors (weight, gradient, m, v, remainder) x 2 bytes",
               lambda v: f"{v:.2f}")
    p = H2H / "runs/a30_ga8_uns_r1_raw.json"
    if p.exists():
        u = load_json(p)
        if "trainable_params" in u and "moe.trainable" in RAW and u["trainable_params"] != RAW["moe.trainable"]:
            pr("moe.unsloth.oproj", u["trainable_params"] - RAW["moe.trainable"], p,
               "trainable_params minus Poros's trainable.params", lambda n: f"{int(n):,}")
    # both tools' losses on the same rows
    for key, a, b in (("loss.72b", "a72_ga8_poros_r1", "a72_ga8_uns_r1_raw"), ("loss.30b", "a30_ga8_ram_r1", "a30_ga8_uns_r1_raw")):
        pp, upp = H2H / f"runs/{a}.json", H2H / f"runs/{b}.json"
        if pp.exists() and upp.exists():
            p_, u_ = load_json(pp), load_json(upp)
            release(p_["commit"], pp)
            pl = p_["losses"]
            ul = [sum(x) / len(x) for x in u_["loss"]]
            if len(pl) != len(ul):
                PROBLEMS.append(f"{key}: {len(pl)} Poros losses against {len(ul)} Unsloth losses")
                continue
            pr(f"{key}.maxdiff", max(abs(x - y) for x, y in zip(pl, ul)), [pp, upp],
               "max over updates of |Poros losses[i] - mean(Unsloth loss[i])|", lambda v: f"{v:.4f}")


# ── the head-to-head as per-run records (h2h/<dir>/a{72,30}_ga8_{poros|ram,uns}_r<N>.json, h2hf.run.v1) ──
# One record per run; an arm is its three rounds.  Poros's figure is the median of its runs, Unsloth's
# the median too, and each comparison sets Poros's median against Unsloth's best run (the rule of
# Section 5.1).  The arm of a model whose records are not there leaves its keys pending.
R2_ARMS = {"dense.poros": "a72_ga8_poros", "dense.unsloth": "a72_ga8_uns",
           "moe.ram": "a30_ga8_ram", "moe.unsloth": "a30_ga8_uns", "moe.unsmin": "a30_ga8_unsmin"}
R2_FIELDS = (("supd", "s_med_2_N"), ("alloc", "vram_alloc_gib"), ("res", "vram_reserved_gib"),
             ("host", "host_ram_gib"), ("t1", "t1_s"))  # (paper key suffix, the run record's field)


def host_peak_processes(csv: pathlib.Path) -> tuple[int, float]:
    """(processes in the process tree at the run's host peak, the peak in GiB over samples with at
    most three processes)."""
    best, procs, few = -1, 0, 0
    for ln in csv.read_text().splitlines()[1:]:
        t_, n_, anon, shmem, *_ = ln.split(",")
        s_ = int(anon) + int(shmem)
        if s_ > best:
            best, procs = s_, int(n_)
        if int(n_) <= 3:
            few = max(few, s_)
    return procs, few / 2**20


def train_host_peak(rec: dict, csv: pathlib.Path) -> float:
    """Peak RssAnon + RssShmem (GiB) over the run's host samples after its load ended."""
    best = 0
    for ln in csv.read_text().splitlines()[1:]:
        t_, _, anon, shmem, *_ = ln.split(",")
        if float(t_) >= rec["load_end_epoch"]:
            best = max(best, int(anon) + int(shmem))
    return best / 2**20


def h2h_runs() -> None:
    global H2H
    for vt_ in sorted(VAL.glob("h2h/*/versions.txt")):
        if (vt_.parent / "meta.json").exists():
            continue
        m_ = re.search(r"^commit (\w+)", vt_.read_text(), re.M)
        if not (m_ and is_frozen(m_.group(1))):
            continue
        H2H = vt_.parent
        break
    else:
        return
    import ast
    vt = (H2H / "versions.txt").read_text()
    release(re.search(r"^commit (\w+)", vt, re.M).group(1), H2H / "versions.txt")
    pr("code.commit", re.search(r"^commit (\w+)", vt, re.M).group(1)[:8], H2H / "versions.txt", "commit",
       lambda v: f"\\texttt{{{v}}}")
    pr("code.version", re.search(r"^poros (\S+)$", vt, re.M).group(1), H2H / "versions.txt", "poros")
    pst = ast.literal_eval(re.search(r"^poros stack: (\{.*?\})", vt, re.M).group(1))
    ust = ast.literal_eval(re.search(r"^unsloth stack: (\{.*?\})", vt, re.M).group(1))
    for k, fld in (("env.torch", "torch"), ("env.transformers", "transformers"), ("env.peft", "peft"),
                   ("env.bnb", "bitsandbytes")):
        pr(k, pst[fld], H2H / "versions.txt", f"poros stack.{fld}")
    pr("env.cuda", re.search(r"^poros stack: \{.*\} cuda (\S+)", vt, re.M).group(1), H2H / "versions.txt", "poros stack cuda")
    for k, fld in (("env.unsloth", "unsloth"), ("env.unsloth.transformers", "transformers"), ("env.unsloth.peft", "peft")):
        pr(k, ust[fld], H2H / "versions.txt", f"unsloth stack.{fld}")
    # each tool's transformers requirement, from the comparison's note on the other tool's versions
    nt_ = H2H / "NOTE_unsloth_versions.md"
    if nt_.exists():
        nt = nt_.read_text()
        m_u = re.search(r"Requires-Dist: transformers[^\n]*?<=([0-9.]+)", nt)
        m_p = re.search(r"requires `transformers(>=[0-9.]+,<[0-9.]+)`", nt)
        if m_u and m_p:
            pr("env.unsloth.tf.cap", f"<={m_u.group(1)}", nt_, "unsloth's Requires-Dist line for transformers",
               lambda v: f"\\code{{{tex_escape(v)}}}")
            pr("env.tf.req", m_p.group(1), nt_, "Poros's pyproject requirement for transformers",
               lambda v: f"\\code{{{tex_escape(v)}}}")
            if ust["transformers"] != m_u.group(1):
                PROBLEMS.append(f"{rel(nt_)}: Unsloth ran transformers {ust['transformers']}, not its cap {m_u.group(1)}")
        else:
            PROBLEMS.append(f"{rel(nt_)}: the transformers requirement lines were not found")
    dt_ = H2H / "unsloth_versions_date.txt"
    if dt_.exists():
        d_ = re.match(r"(\d{4})-(\d{2})-(\d{2})T", dt_.read_text())
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
                  "November", "December"]
        pr("env.unsloth.date", f"{int(d_.group(3))} {months[int(d_.group(2)) - 1]} {d_.group(1)}", dt_,
           "the date the Unsloth version was the newest on PyPI", lambda v: v)
    mc = load_json(H2H / "machine.json")
    for k, fld in (("env.gpu", "gpu"), ("env.gpu.mib", "gpu_mib"), ("env.cpu", "cpu"), ("env.host.gib", "ram_gib"),
                   ("env.os", "os"), ("env.driver", "driver")):
        pr(k, mc[fld], H2H / "machine.json", fld)
    arms = {}
    for short_, arm in R2_ARMS.items():
        files = sorted(H2H.glob(f"{arm}_r[0-9].json"))
        if not files:
            continue
        recs = [load_json(f) for f in files]
        for f, r in zip(files, recs):
            release(r["commit"], f)
            if r["status"] != "ok" or r["rc"] != 0:
                PROBLEMS.append(f"{rel(f)}: the run did not finish ({r['status']})")
        if len(recs) != 3:
            PROBLEMS.append(f"{arm}: {len(recs)} runs, not three")
        arms[short_] = (files, recs)
        for k, fld in R2_FIELDS:
            pr(f"{short_}.{k}", statistics.median(r[fld] for r in recs), files, f"median over runs of {fld}",
               lambda v: f"{v:.3f}" if k != "t1" else f"{v:.1f}")
        tr = [train_host_peak(r, f.with_name(f.stem + "_host.csv")) for f, r in zip(files, recs)]
        pr(f"{short_}.host.train", statistics.median(tr), [f.with_name(f.stem + "_host.csv") for f in files],
           "median over runs of the peak RssAnon + RssShmem after load_end_epoch", lambda v: f"{v:.3f}")
        for k, fld in (("supd", "s_med_2_N"), ("res", "vram_reserved_gib"), ("host", "host_ram_gib"), ("t1", "t1_s")):
            pr(f"{short_}.{k}.best", min(r[fld] for r in recs), files, f"min over runs of {fld}",
               lambda v: f"{v:.3f}" if k != "t1" else f"{v:.1f}")
        pr(f"{short_}.host.train.best", min(tr), files, "min over runs of the training host peak", lambda v: f"{v:.3f}")
        hp = [host_peak_processes(f.with_name(f.stem + "_host.csv")) for f in files]
        csvs = [f.with_name(f.stem + "_host.csv") for f in files]
        pr(f"{short_}.host.procs", statistics.median(n for n, _ in hp), csvs, "processes in the tree at the host peak (median)",
           lambda v: f"{v:.0f}")
        pr(f"{short_}.host.children", statistics.median(n for n, _ in hp) - 1, csvs,
           "child processes at the host peak (median of processes - 1)", lambda v: f"{v:.0f}")
        pr(f"{short_}.host.few", statistics.median(g for _, g in hp), csvs,
           "peak RssAnon + RssShmem over samples with at most three processes (median)", lambda v: f"{v:.1f}")
    for p_, u_ in (("dense.poros", "dense.unsloth"), ("moe.ram", "moe.unsloth")):
        if p_ not in arms or u_ not in arms:
            continue
        pf, prs = arms[p_]
        uf, urs = arms[u_]
        # a host-memory ratio only where neither tool's peak sums more than three processes: a sum over
        # forked processes can count shared pages more than once
        many = max(host_peak_processes(f.with_name(f.stem + "_host.csv"))[0] for f in pf + uf) > 3
        for m_, (fld, what) in (("s_per_step", ("s_med_2_N", "faster")), ("vram_reserved", ("vram_reserved_gib", "less VRAM")),
                                ("host_ram", ("host_ram_gib", "less host RAM")), ("t1", ("t1_s", "sooner to the first update"))):
            if m_ == "host_ram" and many:
                continue
            text, ratio = ratio_text(statistics.median(r[fld] for r in prs), min(r[fld] for r in urs), what)
            pr(f"{p_}.vs.{m_}", text, pf + uf, f"median {fld} of Poros's runs against Unsloth's best", lambda v: v)
            if m_ == "s_per_step":
                pr(f"{p_}.vs.{m_}.ratio", ratio, pf + uf, "larger / smaller, to one decimal, rounded against Poros", lambda v: v)
        if not many:
            ptr = statistics.median(train_host_peak(r, f.with_name(f.stem + "_host.csv")) for f, r in zip(pf, prs))
            utr = min(train_host_peak(r, f.with_name(f.stem + "_host.csv")) for f, r in zip(uf, urs))
            text, _ = ratio_text(ptr, utr, "less host RAM while training")
            pr(f"{p_}.vs.host_ram_train", text, pf + uf, "training host peaks: Poros's median against Unsloth's best", lambda v: v)
    if "moe.ram" in arms:
        f0, r0 = arms["moe.ram"][0][0], arms["moe.ram"][1][0]
        pr("moe.model", r0["model"].split("/", 1)[1], f0, "model")
        mm = re.fullmatch(r"MoE · (\d+) layers · (\d+) experts", r0["poros"]["arch"])
        pr("moe.experts", int(mm.group(2)), f0, "poros.arch")
        for f, r in zip(*arms["moe.ram"]):
            if r["poros"]["ran_path"] != "efficient" or r["poros"]["store"] != "ram":
                PROBLEMS.append(f"{rel(f)}: not the default path (efficient, experts in host memory)")
            if r["poros"]["moe_expert_compute"] != "fused":
                PROBLEMS.append(f"{rel(f)}: the run did not record the fused expert kernels")
        pr("moe.adapter.dtype", {"torch.bfloat16": "bf16", "torch.float32": "fp32"}[same((d for r in arms["moe.ram"][1] for d in r["trainable"]["dtypes"]), "30B dtypes")],
           f0, "trainable.dtypes")
        params = same((r["trainable"]["params"] for r in arms["moe.ram"][1]), "30B trainable params")
        pr("moe.trainable", params, f0, "trainable.params", lambda n: f"{int(n):,}")
        pr("moe.adapter.gap.gib", params * 6 / 2**30, f0,
           "trainable.params x (16 - 10) bytes: fp32 weight, gradient and moments against Poros's five bf16 tensors",
           lambda v: f"{v:.1f}")
        if arms["moe.ram"][1][0]["trainable"]["dtypes"] == ["torch.bfloat16"]:
            pr("moe.adapter.state.gib", params * 5 * 2 / 2**30, f0,
               "trainable.params x 5 bf16 tensors (weight, gradient, m, v, remainder) x 2 bytes", lambda v: f"{v:.2f}")
        cp = f0.with_name(f0.stem + "_config.resolved.json")
        c = load_json(cp)
        for k, fld, fn in (("wl.rank", "lora_rank", None), ("wl.seq", "max_seq_length", None),
                           ("wl.ga", "gradient_accumulation_steps", None), ("wl.alpha", "lora_alpha", None),
                           ("wl.dropout", "lora_dropout", None), ("wl.mb", "per_device_batch_size", None),
                           ("wl.updates", "max_steps", None), ("wl.lr", "learning_rate", sci),
                           ("wl.wd", "weight_decay", None), ("wl.clip", "max_grad_norm", None),
                           ("wl.sched", "lr_scheduler", None), ("wl.seed", "seed", None)):
            pr(k, c[fld], cp, fld, fn)
        pr("wl.betas", f"{c['adam_beta1']}, {c['adam_beta2']}", cp, "adam_beta1, adam_beta2")
        pr("env.sampler.s", same((r["host_sampler"]["interval_s"] for r in arms["moe.ram"][1]), "sampler"), f0,
           "host_sampler.interval_s")
    if "moe.unsloth" in arms:
        f0, r0 = arms["moe.unsloth"][0][0], arms["moe.unsloth"][1][0]
        pr("moe.unsloth.offloaded", same((r["unsloth"]["swapped_layers"] for r in arms["moe.unsloth"][1]), "offloaded"), f0,
           "unsloth.swapped_layers")
        pr("moe.layers", same((r["unsloth"]["decoder_layers"] for r in arms["moe.unsloth"][1]), "layers"), f0, "unsloth.decoder_layers")
        raw = f0.with_name(f0.stem + "_raw.json")
        u = load_json(raw)
        a = u["args"]
        if not a["deterministic"]:
            PROBLEMS.append("Unsloth's 30B runs were not in deterministic mode")
        for k, fld, fn in (("wl.unsloth.rank", "rank", None), ("wl.unsloth.lr", "lr", sci), ("wl.unsloth.wd", "wd", None),
                           ("wl.unsloth.clip", "clip", None), ("wl.unsloth.seed", "seed", None)):
            pr(k, a[fld], raw, f"args.{fld}", fn)
        pr("wl.unsloth.gc", a["gc"], raw, "args.gc", lambda v: f"\\code{{{tex_escape(v)}}}")
        pr("card.gib", u["card_gib"], raw, "card_gib")
        pr("moe.unsloth.adapter.dtype", {"torch.float32": "fp32", "torch.bfloat16": "bf16"}[same(u["trainable_dtypes"], "dtype")],
           raw, "trainable_dtypes")
        pr("moe.unsloth.oproj", u["frozen_adapters"]["elements"], raw, "frozen_adapters.elements", lambda n: f"{int(n):,}")
    rc_ = H2H / "rows_check_30b_ga8.json"
    if rc_.exists():
        rc = load_json(rc_)
        if not (rc["pass"] and rc["poros_full_rows"] and set(rc["unsloth"].values()) == {rc["poros_fed_sha256"]}):
            PROBLEMS.append("rows_check_30b_ga8: the two tools were not fed the same rows")
        pr("wl.rows", rc["poros_fed_rows"], rc_, "poros_fed_rows")
    if "moe.ram" in arms and "moe.unsloth" in arms:
        for (pf, prs), (uf, urs), key in ((arms["moe.ram"], arms["moe.unsloth"], "loss.30b"),):
            p0, u0 = prs[0], load_json(uf[0].with_name(uf[0].stem + "_raw.json"))
            ul = [sum(x) / len(x) for x in u0["loss"]]
            pr(f"{key}.maxdiff", max(abs(x - y) for x, y in zip(p0["losses"], ul)), [pf[0], uf[0]],
               "max over updates of |Poros losses[i] - mean(Unsloth loss[i])|, round 1", lambda v: f"{v:.4f}")


# ── Qwen2.5-72B: the 18 November comparison session (h2h/4090) ───────────────────
# The 72B comparison was not repeated at the release commit before the release: the paper cites the
# 18 November session's 72B arms, Poros at the commit that session records and Unsloth at the version
# it records, and says so where it gives them.  These two arm files (and their round-1 records) are the
# only logs of another commit read for a Poros figure.
PREV_SESSION_COMMIT = "5d75e31a486cdf19e397e2d0f748def98083ecc8"


def h2h_dense_previous() -> None:
    d0 = VAL / "h2h/4090"
    pa, ua = d0 / "arms/a72_ga8_poros.json", d0 / "arms/a72_ga8_uns.json"
    if not (pa.exists() and ua.exists()):
        return
    P, U = load_json(pa), load_json(ua)
    for f, a in ((pa, P), (ua, U)):
        if a["commit"] != PREV_SESSION_COMMIT:
            raise SystemExit(f"{rel(f)} names {a['commit']}, not the 18 November session's {PREV_SESSION_COMMIT[:8]}")
        if len(a["runs"]) != 3:
            PROBLEMS.append(f"{rel(f)}: {len(a['runs'])} runs, not three")
    pr("dense.commit", PREV_SESSION_COMMIT[:8], pa, "commit", lambda v: f"\\texttt{{{v}}}")
    import ast
    vt_ = (d0 / "versions.txt").read_text()
    pr("dense.poros.version", re.search(r"^poros (\S+)$", vt_, re.M).group(1), d0 / "versions.txt", "poros")
    pst_ = ast.literal_eval(re.search(r"^poros stack: (\{.*?\})", vt_, re.M).group(1))
    if any(pst_[k] != RAW.get(f"env.{e}") for k, e in (("torch", "torch"), ("transformers", "transformers"), ("peft", "peft"),
                                                        ("bitsandbytes", "bnb"))):
        PROBLEMS.append("the 18 November Poros stack differs from the release comparison's (the setup table says it is the same)")
    days = sorted({r["launch_utc"][:10] for r in P["runs"] + U["runs"]})
    if len(days) != 1:
        PROBLEMS.append(f"the 72B arms ran on several days: {days}")
    y, mo, da = days[0].split("-")
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    pr("dense.date", f"{int(da)} {months[int(mo) - 1]} {y}", pa, "runs[].launch_utc", lambda v: v)
    for short_, a, f in (("dense.poros", P, pa), ("dense.unsloth", U, ua)):
        for k, fld in METRICS:
            pr(f"{short_}.{k}", statistics.median(r[fld] for r in a["runs"]), f, f"median over runs[].{fld}",
               lambda v: f"{v:.3f}")
        for k, fld in (("supd", "s_per_step_s"), ("res", "vram_reserved_gb"), ("host", "host_ram_gb"),
                       ("host.train", "host_ram_train_gb")):
            pr(f"{short_}.{k}.best", min(r[fld] for r in a["runs"]), f, f"min over runs[].{fld}", lambda v: f"{v:.3f}")
    pr("dense.poros.load.s", P["median"]["load_s"], pa, "median.load_s")
    pr("dense.poros.load.res", P["median"]["load_vram_reserved_gb"], pa, "median.load_vram_reserved_gb")
    many = max(r["host_sampler"]["processes_max"] for r in P["runs"] + U["runs"]) > 3
    for m_, (fld, what) in (("s_per_step", ("s_per_step_s", "faster")), ("vram_reserved", ("vram_reserved_gb", "less VRAM")),
                            ("host_ram", ("host_ram_gb", "less host RAM")),
                            ("host_ram_train", ("host_ram_train_gb", "less host RAM while training"))):
        if m_.startswith("host") and many:
            continue
        text, ratio = ratio_text(P["median"][fld], min(r[fld] for r in U["runs"]), what)
        pr(f"dense.poros.vs.{m_}", text, [pa, ua], f"median.{fld} against min over Unsloth's runs[].{fld}", lambda v: v)
        if m_ == "s_per_step":
            pr("dense.poros.vs.s_per_step.ratio", ratio, [pa, ua], "larger / smaller, to one decimal, rounded against Poros",
               lambda v: v)
    pr("dense.unsloth.offloaded", same((r["unsloth"]["swapped_layers"] for r in U["runs"]), "72B offloaded"), ua,
       "runs[].unsloth.swapped_layers")
    pr("dense.layers", same((r["unsloth"]["decoder_layers"] for r in U["runs"]), "72B layers"), ua, "runs[].unsloth.decoder_layers")
    pr("dense.unsloth.embed", same((r["unsloth"]["input_embedding_device"] for r in U["runs"]), "72B embedding"), ua,
       "runs[].unsloth.input_embedding_device")
    dt = lambda xs: {"torch.float32": "fp32", "torch.bfloat16": "bf16"}[same(xs, "dtype")]
    dg, ur = d0 / "runs/a72_ga8_poros_r1_digest.json", d0 / "runs/a72_ga8_uns_r1_raw.json"
    pr("dense.adapter.dtype", dt(load_json(dg)["dtypes"]), dg, "dtypes")
    u = load_json(ur)
    pr("dense.unsloth.adapter.dtype", dt(u["trainable_dtypes"]), ur, "trainable_dtypes")
    pr("wl.trainable72", u["trainable_params"], ur, "trainable_params", lambda n: f"{int(n):,}")
    pr("wl.unsloth.targets", u["args"]["targets"].replace("_proj", "").replace(",", ", "), ur, "args.targets")
    pp = d0 / "runs/a72_ga8_poros_r1.json"
    p1 = load_json(pp)
    if p1["commit"] != PREV_SESSION_COMMIT:
        raise SystemExit(f"{rel(pp)} names {p1['commit']}")
    ul = [sum(x) / len(x) for x in u["loss"]]
    if len(ul) == len(p1["losses"]):
        pr("loss.72b.maxdiff", max(abs(x - y) for x, y in zip(p1["losses"], ul)), [pp, ur],
           "max over updates of |Poros losses[i] - mean(Unsloth loss[i])|, round 1", lambda v: f"{v:.4f}")


# ── the previous version of the other tool, as the same harness measured it on November 18 ──────
# Read for the other tool's own figures only (its version, its runs' times and memory), so the paper can
# set the comparison's ratio beside what the earlier version measured.  Nothing of Poros's is read from it: that
# session ran an earlier Poros commit.
def h2h_previous_unsloth() -> None:
    ap = VAL / "h2h/4090/arms/a30_ga8_uns.json"
    vt = VAL / "h2h/4090/versions.txt"
    if not (ap.exists() and vt.exists()):
        return
    import ast
    a = load_json(ap)
    if a.get("tool", "unsloth") != "unsloth" or "unsloth" not in a["runs"][0]:
        PROBLEMS.append(f"{rel(ap)} is not the other tool's arm")
        return
    ust = ast.literal_eval(re.search(r"^unsloth stack: (\{.*?\})", vt.read_text(), re.M).group(1))
    pr("prev.unsloth.version", ust["unsloth"], vt, "unsloth stack.unsloth")
    days = sorted({r["launch_utc"][:10] for r in a["runs"]})
    d_ = days[0].split("-")
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
              "November", "December"]
    pr("prev.unsloth.date", f"{int(d_[2])} {months[int(d_[1]) - 1]} {d_[0]}", ap, "runs[].launch_utc", lambda v: v)
    for k, fld in (("supd", "s_per_step_s"), ("res", "vram_reserved_gb"), ("host", "host_ram_gb")):
        pr(f"prev.moe.unsloth.{k}.best", min(r[fld] for r in a["runs"]), ap, f"min over runs[].{fld}",
           lambda v: f"{v:.3f}")
    pr("prev.moe.unsloth.supd", a["median"]["s_per_step_s"], ap, "median.s_per_step_s", lambda v: f"{v:.3f}")
    pr("prev.moe.unsloth.host.train", a["median"]["host_ram_train_gb"], ap, "median.host_ram_train_gb", lambda v: f"{v:.1f}")
    pr("prev.moe.unsloth.host.procs", max(r["host_sampler"]["processes_max"] for r in a["runs"]), ap,
       "max over runs[].host_sampler.processes_max")
    # how much slower the newer version ran on the same harness and rows (the logs do not say why)
    if "moe.unsloth.supd" in RAW:
        pr("moe.unsloth.slower.median", 100 * (RAW["moe.unsloth.supd"] / a["median"]["s_per_step_s"] - 1), ap,
           "Unsloth's 24 November median over its 18 November median, minus one", lambda v: f"{v:.0f}\\%")
        pr("moe.unsloth.slower.best", 100 * (RAW["moe.unsloth.supd.best"] / min(r["s_per_step_s"] for r in a["runs"]) - 1), ap,
           "Unsloth's 24 November best run over its 18 November best run, minus one", lambda v: f"{v:.0f}\\%")
    # the same version with every decoder layer in host memory (its lowest-memory setting), as figures
    mp = VAL / "h2h/4090/arms/a30_ga8_unsmin.json"
    if mp.exists():
        mn = load_json(mp)
        if "unsloth" not in mn["runs"][0] or mn["runs"][0]["unsloth"]["swapped_layers"] != mn["runs"][0]["unsloth"]["decoder_layers"]:
            PROBLEMS.append(f"{rel(mp)}: not the setting with every decoder layer in host memory")
        pr("prev.moe.unsmin.supd", mn["median"]["s_per_step_s"], mp, "median.s_per_step_s", lambda v: f"{v:.3f}")
        pr("prev.moe.unsmin.supd.best", min(r["s_per_step_s"] for r in mn["runs"]), mp, "min over runs[].s_per_step_s",
           lambda v: f"{v:.3f}")
        pr("prev.moe.unsmin.res.best", min(r["vram_reserved_gb"] for r in mn["runs"]), mp, "min over runs[].vram_reserved_gb",
           lambda v: f"{v:.3f}")
        pr("prev.moe.unsmin.offloaded", mn["runs"][0]["unsloth"]["swapped_layers"], mp, "runs[].unsloth.swapped_layers")
    # One ratio against the earlier version, speed, set across two sessions (Poros's median at the release
    # commit on 24 November against that tool's best run on 18 November); the text flags it as such.  Its
    # memory figures are given as figures, with no ratio.
    if "moe.ram.supd" in RAW:
        for m_, (pk, uk, what) in (("s_per_step", ("supd", "s_per_step_s", "faster")),):
            text, _ = ratio_text(RAW[f"moe.ram.{pk}"], min(r[uk] for r in a["runs"]), what)
            pr(f"moe.ram.vs.prev.{m_}", text, ap,
               f"Poros's median at the release commit against min over the earlier version's runs[].{uk}", lambda v: v)
            SOURCES[f"moe.ram.vs.prev.{m_}"] = SOURCES[f"moe.ram.{pk}"] + [rel(ap)]


def check_claim() -> None:
    """The abstract, the claim box and Section 5 say Poros's default path is faster AND reserves and
    allocates less GPU memory than Unsloth's automatic offloading on both models.  If the logs say
    otherwise, the build must not pass.  Pending values are caught by the pending check."""
    for fam, p, u in (("72B", "dense.poros", "dense.unsloth"), ("30B-A3B", "moe.ram", "moe.unsloth")):
        for m in ("s_per_step", "vram_reserved"):
            t = VALUES.get(f"{p}.vs.{m}")
            if t is not None and not t.startswith("Poros "):
                PROBLEMS.append(f"CLAIM: {fam} {m}: the logs give {t!r}")
        pa, ua = RAW.get(f"{p}.alloc"), RAW.get(f"{u}.alloc")
        if pa is not None and ua is not None and not pa < ua:
            PROBLEMS.append(f"CLAIM: {fam}: Poros allocates no less than Unsloth ({pa} vs {ua} GiB)")


# ── poros prove on the 7B model (RTX 4090) ───────────────────────────────────
def prove() -> None:
    for p in in_rounds("**/prove*.json"):
        d = load_json(p)
        if d.get("schema") != "poros.prove.v1" or d.get("simulated") or "7B" not in d.get("model", ""):
            continue
        if not (d["identical"] and not d["differs"] and d["reference"] == "resident"):
            PROBLEMS.append(f"{rel(p)}: prove did not find the streamed run equal to the resident run")
        pr("prove.model", d["model"].split("/", 1)[1], p, "model")
        pr("prove.gpu", d["env"]["gpu"].replace("NVIDIA GeForce ", ""), p, "env.gpu")
        pr("prove.steps", d["steps"], p, "steps")
        pr("prove.seq", d["seq_len"], p, "seq_len")
        pr("prove.str.bs", d["streamed"]["block_size"], p, "streamed.block_size")
        pr("prove.res.bs", d["resident"]["block_size"], p, "resident.block_size")
        pr("prove.str.s", d["streamed"]["s_per_step"], p, "streamed.s_per_step", lambda v: f"{v:.2f}")
        pr("prove.res.s", d["resident"]["s_per_step"], p, "resident.s_per_step", lambda v: f"{v:.2f}")
        pr("prove.str.vram", d["streamed"]["peak_vram_gib"], p, "streamed.peak_vram_gib", lambda v: f"{v:.2f}")
        pr("prove.res.vram", d["resident"]["peak_vram_gib"], p, "resident.peak_vram_gib", lambda v: f"{v:.2f}")
        comp = d["compared"]
        if "losses" not in comp:
            PROBLEMS.append(f"{rel(p)}: prove did not compare the losses")
        for key, word in (("prove.grads", "adapter gradients"), ("prove.tensors", "adapter tensors"),
                          ("prove.opt", "optimizer tensors")):
            m = [re.fullmatch(rf"(\d+) {word}", x) for x in comp]
            m = [x for x in m if x]
            if m:
                pr(key, int(m[0].group(1)), p, "compared")
            else:
                PROBLEMS.append(f"{rel(p)}: prove did not compare {word}")
        return


# ── the family gate (15 rows) ────────────────────────────────────────────────
def family_gate() -> None:
    for p in in_rounds("**/family_gate/RESULT.txt") + in_rounds("**/family/RESULT.txt"):
        t = p.read_text()
        m = re.search(r"sha (\w+) dirty=(\d)", t)
        release(m.group(1) if m else None, p)
        if m.group(2) != "0":
            PROBLEMS.append(f"{rel(p)}: the tree was dirty")
        rows = re.findall(r"^--- (\S+) \((\S+) @", t, re.M)
        ink = [r[0] for r in rows if "inkling" in r[0].lower() or "inkling" in r[1].lower()]
        if ink:  # Inkling 2 Mini is not validated in 1.0.0: its rows are not counted or named
            NOT_USED.append(f"{rel(p)}: the {', '.join(ink)} rows (Inkling, not in 1.0.0)")
        rows = [r for r in rows if r[0] not in ink]
        fails = re.findall(r"^FAIL\s+(\S+?):", t, re.M)
        other = [f_ for f_ in fails if f_ not in ink]
        m_ = re.search(r"^FAILS=(\d+)", t, re.M)
        if other or not m_ or int(m_.group(1)) != len(fails):
            PROBLEMS.append(f"{rel(p)}: failures outside the Inkling row ({sorted(set(other))}) or a FAILS count that does not match")
        if ink and not re.search(rf"^FAIL\s+{re.escape(ink[0])}:check\s+rc=4 model_type=\S+ status=target", t, re.M):
            PROBLEMS.append(f"{rel(p)}: the Inkling row failed for another reason than being an unvalidated family")
        pr("fg.n", len(rows), p, "the '--- family (model @' lines, Inkling rows left out")
        npass = len([1 for fam_, chk_ in re.findall(r"^PASS\s+(\S+?):(\S+)", t, re.M) if fam_ not in ink and chk_ != "fetch"])
        entry = {}
        try:
            import subprocess
            tsv = subprocess.run(["git", "-C", str(ROOT), "show", f"{FROZEN}:scripts/family_gate/families.tsv"],
                                 capture_output=True, text=True, check=True).stdout
            tsv_src = f"scripts/family_gate/families.tsv at {FROZEN[:8]}"
        except Exception:  # noqa: BLE001  a checkout without the commit's history: the file itself
            tsv = (ROOT / "scripts/family_gate/families.tsv").read_text()
            tsv_src = "scripts/family_gate/families.tsv"
        for ln in tsv.splitlines():
            if ln and not ln.startswith("#"):
                f_ = ln.split("\t")
                entry[f_[0]] = f_[1]
        fam_name = {"qwen3_moe": "Qwen3-MoE", "olmoe": "OLMoE", "qwen2_moe": "Qwen2-MoE", "deepseek_v2": "DeepSeek-V2",
                    "qwen3_5_moe": "Qwen3.5-MoE", "qwen3_5_moe_text": "Qwen3.5-MoE", "gemma4_moe": "Gemma~4-MoE"}
        moe_ok = []
        for fam, _ in rows:
            if entry.get(fam) == "moe" and re.search(rf"^PASS\s+{re.escape(fam)}:paths\s", t, re.M):
                n = fam_name.get(fam)
                if n is None:
                    PROBLEMS.append(f"{rel(p)}: MoE row {fam} has no display name in family_gate()")
                elif n not in moe_ok:
                    moe_ok.append(n)
        if moe_ok:
            said = moe_ok[0] if len(moe_ok) == 1 else ", ".join(moe_ok[:-1]) + " and " + moe_ok[-1]
            pr("fg.moe.families", said, p, f"MoE rows ({tsv_src}: entry moe) with a PASS paths line", lambda v: v)
        pr("fg.checks", npass, p, "PASS <family>:<check> lines, the downloads (:fetch) and the Inkling row left out")
        for fam, _ in rows:
            if not re.search(rf"^PASS\s+{re.escape(fam)}:paths\s+seq==eff==fast==resident digest", t, re.M):
                PROBLEMS.append(f"{rel(p)}: {fam} has no PASS paths line")
        res = re.findall(r"^PASS\s+(\S+):resume\s", t, re.M)
        names = {"qwen2": "Qwen2.5-0.5B", "olmoe": "OLMoE-1B-7B"}
        pr("fg.resume.models", " and ".join(names.get(r, r) for r in res), p, "PASS <family>:resume lines")
        return


# ── the dropout-on checks: block-major against sequential at dropout 0.05 ────
def dropout_checks() -> None:
    cmps = in_rounds("**/cmp_*.json")
    found = {}
    for p in cmps:
        d = load_json(p)
        name = p.name.lower()
        fam = "olmoe" if ("moe" in name or "olmoe" in name) else "32b" if "32" in name else "7b" if ("m7" in name or "7b" in name) else None
        if fam is None or "seq" not in name:
            continue
        if not (d.get("pass") and d.get("losses_equal") and d.get("grad_norms_equal") and not d.get("adapter_unequal")):
            PROBLEMS.append(f"{rel(p)}: the dropout-on check did not pass")
        found.setdefault(fam, []).append((p, d))
    for fam, lst in found.items():
        steps = same((d["steps"] if "steps" in d else len(d["steps_compared"]) for _, d in lst), f"dropout {fam} steps")
        tens = same((d["adapter_tensors"] for _, d in lst), f"dropout {fam} tensors")
        pr(f"do.{fam}.steps", steps, [p for p, _ in lst], "steps (or the length of steps_compared)")
        pr(f"do.{fam}.tensors", tens, [p for p, _ in lst], "adapter_tensors")
        # the model and the dropout, from the compared runs' resolved configs
        for p, _ in lst:
            cfgs = sorted(p.parent.glob("*.config.resolved.json"))
            for cf in cfgs:
                c = load_json(cf)
                key = cf.name.lower()
                if (fam == "7b" and "m7" in key) or (fam == "32b" and "32" in key) or (fam == "olmoe" and "moe" in key):
                    if fam != "olmoe":
                        pr(f"do.{fam}.model", c["model_name_or_path"].split("/", 1)[1], cf, "model_name_or_path")
                    pr("do.dropout", c["lora_dropout"], cf, "lora_dropout")
                    break


# ── the optimizer self-test on each device ───────────────────────────────────
def selftest_file(p: pathlib.Path, key: str) -> None:
    t = p.read_text()
    last = [ln for ln in t.splitlines() if ln.startswith("{")]
    if not last:
        return
    d = json.loads(last[-1])
    bad = [ln for ln in t.splitlines() if ln.startswith("FAIL")]
    if not d.get("pass") or bad or not all(d["checks"].values()):
        PROBLEMS.append(f"{rel(p)}: the self-test did not pass")
    if d.get("implementation", {}).get("path") != "kernel":
        PROBLEMS.append(f"{rel(p)}: the optimizer did not choose the fused kernel")
    if d.get("rule") != "poros.expert-adamw.v4":
        PROBLEMS.append(f"{rel(p)}: rule {d.get('rule')}")
    m = re.search(r"^time: one step over (\S+) bf16 elements", t, re.M)
    if m:
        pr("st.elems", m.group(1), p, "the time line")
    s = d.get("step_seconds") or {}
    if s:
        pr(f"{key}.new", s["expert_adamw"], p, "step_seconds.expert_adamw", lambda v: f"{v:.4f}")
        pr(f"{key}.torch", s["torch_adamw_bf16"], p, "step_seconds.torch_adamw_bf16", lambda v: f"{v:.4f}")


def selftests() -> None:
    for p in in_rounds("**/selftest*.txt"):
        selftest_file(p, "st.4090")
        break
    for lp in sorted(VAL.glob("int/4060l/**/selftest_cuda.txt")):
        ct = lp.parent / "commit.txt"
        if ct.exists() and is_frozen(ct.read_text().split()[0]):
            release(ct.read_text().split()[0], ct)
            selftest_file(lp, "st.4060l")
            SOURCES["st.4060l.new"] = SOURCES.get("st.4060l.new", []) + [rel(ct)]
            break
    lp, pg = LAPLOGS / "verdicts/selftest_cuda.txt", LAPLOGS / "progress.txt"
    if "st.4060l.new" not in VALUES and lp.exists() and pg.exists():
        m = re.search(r"clone: PASS\..*?HEAD ([0-9a-f]{40})", pg.read_text())
        release(m.group(1) if m else None, pg)
        selftest_file(lp, "st.4060l")
        SOURCES["st.4060l.new"] = SOURCES.get("st.4060l.new", []) + [rel(pg)]
    for p in ([] if "st.4060l.new" in VALUES else
              sorted(VAL.glob("recipes/rtx4060-laptop/**/selftest*.txt")) + sorted(VAL.glob("int/4060l/**/selftest*.txt"))):
        # the laptop's folder names its commit in a README or commit.txt beside it
        c = readme_commit(p.parent / "README.md")
        if c is None and (p.parent / "commit.txt").exists():
            c = (p.parent / "commit.txt").read_text().strip()
        if is_frozen(c):
            selftest_file(p, "st.4060l")
            break
    for mp in sorted(VAL.glob("int/mac/**/selftest_mps.txt")):  # in the benchmark logs, its commit in commit.txt
        ct, c_ = mp.parent / "commit.txt", None
        if ct.exists():
            c_ = ct.read_text().split()[0]
        elif (mp.parent / "results.jsonl").exists():  # the Mac queue's record: its setup line names the clone's commit
            ct = mp.parent / "results.jsonl"
            setups = [json.loads(x) for x in ct.read_text().splitlines() if x.strip()]
            setups = [x for x in setups if x.get("check") == "setup" and x.get("status") == "pass"]
            m_ = re.search(r"\b([0-9a-f]{40})\b", setups[-1]["note"]) if setups else None
            c_ = m_.group(1) if m_ else None
        if c_ and is_frozen(c_):
            release(c_, ct)
            selftest_file(mp, "st.mac")
            SOURCES["st.mac.new"] = SOURCES.get("st.mac.new", []) + [rel(ct)]
            break
    mps = [] if "st.mac.new" in VALUES else sorted(MACLOGS.glob("**/selftest_mps.txt"))
    mp = mps[0] if mps else MACLOGS / "selftest_mps.txt"
    rj = MACLOGS / "results.jsonl"
    if mps and mp.exists() and rj.exists():
        setup = [json.loads(x) for x in rj.read_text().splitlines() if x.strip()]
        setup = [x for x in setup if x.get("check") == "setup" and x.get("status") == "pass"]
        m = re.search(r"\b([0-9a-f]{40})\b", setup[0]["note"]) if setup else None
        release(m.group(1) if m else None, rj)
        selftest_file(mp, "st.mac")
        SOURCES["st.mac.new"] = SOURCES.get("st.mac.new", []) + [rel(rj)]
    # the devices whose self-test log is present and passed, and their times, as the text prints them
    devs = [(k, name) for k, name in (("st.4090", "the RTX~4090"), ("st.4060l", "the RTX~4060 Laptop GPU"),
                                      ("st.mac", "the M4 Mac")) if f"{k}.new" in VALUES]
    if devs:
        names = [n for _, n in devs]
        said = names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]
        srcs = [s for k, _ in devs for s in SOURCES[f"{k}.new"]]
        pr("st.devices", said, [], "the devices whose self-test log is present and passed", lambda v: v)
        SOURCES["st.devices"] = srcs
        times = "; ".join(f"{VALUES[k + '.new']}\\,s against {VALUES[k + '.torch']}\\,s on {n}" for k, n in devs)
        pr("st.times", times, [], "step_seconds of each self-test present", lambda v: v)
        SOURCES["st.times"] = srcs
        pr("st.n", {1: "one", 2: "two", 3: "three"}[len(devs)], [], "count of self-tests present", lambda v: v)
        SOURCES["st.n"] = srcs


# ── the certified recipes, all at the frozen commit ──────────────────────────
MACHINES = {"rtx4090": "RTX 4090", "rtx4060-laptop": "RTX 4060 Laptop", "m4-16gb": "M4, 16 GB"}
ORDER = ["first-run", "support", "voice", "prefer", "big-32b", "moe-30b", "moe-30b-ssd", "moe-30b-8gb",
         "big-72b", "big-72b-fast", "mac-first-run", "mac-7b", "mac-support", "mac-voice", "mac-prefer",
         "mac-moe", "mac-32b", "mac-72b"]
RECIPE_ROWS: list[tuple] = []
CERT: dict[tuple, tuple] = {}


REPEATS: dict[tuple, list] = {}  # (machine, recipe) -> certification.json files of its repeat runs


def recipe_files() -> list[tuple[str, pathlib.Path]]:
    out = {}
    REPEATS.clear()
    for mach in MACHINES:
        for f in sorted(VAL.glob(f"recipes/{mach}/*/certification.json")):
            if f.parent.name.endswith("-repeat"):  # a second certification run of a recipe: not a table row
                c = load_json(f)
                if is_frozen(c.get("commit")):
                    REPEATS.setdefault((mach, c["recipe"]), []).append(f)
                else:
                    NOT_USED.append(f"{rel(f)} (commit {short(c.get('commit'))})")
                continue
            c = load_json(f)
            if is_frozen(c.get("commit")):
                out[(mach, c["recipe"])] = f
            else:
                NOT_USED.append(f"{rel(f)} (commit {short(c.get('commit'))})")
    for f in sorted(MACLOGS.glob("**/certification.json")):
        c = load_json(f)
        if is_frozen(c.get("commit")) and ("m4-16gb", c["recipe"]) not in out:
            out[("m4-16gb", c["recipe"])] = f
    return [(m, f) for (m, _), f in out.items()]


def drive(folder: pathlib.Path):
    """Which drive a Mac run read its model from: the model-drive.txt beside the run's log, whose first words are exactly "model read from an external drive: ..." or "model read from
    the Mac's internal SSD: ...".  Nothing else is read for it (a store_path may be redacted); without the
    file, None, and the key stays pending.  Returns (words, file)."""
    f = folder / "model-drive.txt"
    if not f.exists():
        return None, f
    first = f.read_text().strip()
    if first.startswith("model read from an external drive:"):
        return "an external drive", f
    if first.startswith("model read from the Mac's internal SSD:"):
        return "the Mac's internal SSD", f
    PROBLEMS.append(f"{rel(f)}: its first words are neither form ({first[:60]!r})")
    return None, f


GPU_MACHINE = {"RTX 4090": "rtx4090", "RTX 4060 Laptop GPU": "rtx4060-laptop", "Apple M4 10-core GPU 16GB": "m4-16gb"}


def certified_set():
    """{(recipe, machine)} the certification generator certified, from poros/recipes/certified.json at the
    frozen commit (this repository's, or the benchmark checkout's); None while no copy names that commit."""
    for f in (ROOT / "poros/recipes/certified.json", BENCH / "poros/recipes/certified.json"):
        if f.exists():
            d = load_json(f)
            if is_frozen(d.get("commit")):
                out = set()
                for rcp, entries in d["recipes"].items():
                    for e in entries:
                        m = GPU_MACHINE.get(e["machine"].get("gpu") or "")
                        if m is None:
                            PROBLEMS.append(f"{rel(f)}: no machine folder for {e['machine'].get('gpu')!r}")
                        out.add((rcp, m))
                return out, f
    return None, None


def recipes() -> None:
    rows = []
    for mach, f in recipe_files():
        c = load_json(f)
        release(c["commit"], f)
        if not (c["status"] == "ok" and c["tree_clean"] and not c["simulated"]):
            PROBLEMS.append(f"{rel(f)}: status {c['status']}, tree_clean {c['tree_clean']}, simulated {c['simulated']}")
            continue
        CERT[(mach, c["recipe"])] = (f, c)
        if c["main"].get("s_per_step_rule") != "median of updates 2..N":  # Table 3's note says so
            PROBLEMS.append(f"{rel(f)}: s_per_step_rule is {c['main'].get('s_per_step_rule')!r}, not the median of updates 2..N")
        model = c["command"][2].split("/", 1)[1]
        m = c["main"]
        tm, h, e, r = c["torch_memory"], c["host_ram"], c["eval"], c["resume"]
        met = e.get("metric") or {}
        if "eval_loss" in met:
            ev = f"{met['base_loss']:.2f} $\\to$ {met['eval_loss']:.2f}"
        else:
            ev = f"pref.\\ {met['base_loss']:.3f} $\\to$ {met['trained_loss']:.3f}"
        mem = (f"{tm['allocated_peak_gib']:.2f}" if tm.get("reserved_peak_gib") is None
               else f"{tm['allocated_peak_gib']:.2f} / {tm['reserved_peak_gib']:.2f}")
        host = "pool" if h.get("peak_gib") is None else f"{h['peak_gib']:.1f}"
        res = {True: "identical", None: "not run"}.get(r.get("adapter_equal"), "differs")
        if res == "differs":
            PROBLEMS.append(f"{rel(f)}: the continued run's adapter differs from the uninterrupted one")
        evf = f.parent / "main/events.jsonl"
        stp = [json.loads(x)["data"] for x in evf.read_text().splitlines() if x.strip()
               and json.loads(x).get("event_type") == "setup" and "path" in json.loads(x)["data"]][0]
        path = {"efficient": "eff.", "fast": "fast", "sequential": "seq."}[stp["path"]]
        mid = {"rtx4090": "r4090", "rtx4060-laptop": "r4060l", "m4-16gb": "m4"}[mach]
        k = f"rec.{mid}.{c['recipe']}"
        cf = f.parent / "main/config.resolved.json"
        cr = load_json(cf)
        geo = f"{cr['per_device_batch_size']}$\\times${cr['gradient_accumulation_steps']}"
        for fld, v, src, fname in (("tok", c["data"]["seq_len"], f, "data.seq_len"),
                                   ("rank", cr["lora_rank"], cf, "lora_rank"),
                                   ("geo", geo, cf, "per_device_batch_size x gradient_accumulation_steps"),
                                   ("upd", m["updates"], f, "main.updates"),
                                   ("s", f"{m['s_per_step']:.2f}", f, "main.s_per_step"),
                                   ("gpu", mem, f, "torch_memory"),
                                   ("host", host, f, "host_ram.peak_gib"),
                                   ("ev", ev, f, "eval.metric"),
                                   ("res", res, f, "resume.adapter_equal"),
                                   ("path", path, evf, "setup.path")):
            pr(f"{k}.{fld}", v, src, fname, lambda x: x)
        if mach == "m4-16gb" and (stp["path"] != "sequential" or cr.get("block_size") != 1
                                  or cr["per_device_batch_size"] * cr["gradient_accumulation_steps"] != 1):
            PROBLEMS.append(f"{rel(f)}: a Mac recipe not at one row per update, block size 1, sequential "
                            "(Section 2.7 says every Mac run is)")
        # Section 5.6: the held-out loss fell in every supervised recipe and the preference loss rose in
        # both preference recipes; Section 5.5 says the laptop's fell
        if "eval_loss" in met and not met["eval_loss"] < met["base_loss"]:
            PROBLEMS.append(f"{rel(f)}: held-out loss did not fall ({met['base_loss']} -> {met['eval_loss']})")
        if "trained_loss" in met and not met["trained_loss"] > met["base_loss"]:
            PROBLEMS.append(f"{rel(f)}: the preference loss did not rise ({met['base_loss']} -> {met['trained_loss']})")
        rows.append((MACHINES[mach], c["recipe"], model, k, f, mach))
    rows.sort(key=lambda x: (list(MACHINES.values()).index(x[0]), ORDER.index(x[1]) if x[1] in ORDER else 99))
    cert, cf_ = certified_set()
    RECIPE_ROWS.clear()
    RECIPE_ROWS.extend((m_, r_, mo_, k_, f_, None if cert is None else (r_, ma_) in cert) for m_, r_, mo_, k_, f_, ma_ in rows)
    want = {"rtx4090": 10, "rtx4060-laptop": 2, "m4-16gb": 8}
    have = {m: sum(1 for x in rows if x[0] == MACHINES[m]) for m in MACHINES}
    if have == want:  # the counts print only when every expected certification is in
        fs = [x[4] for x in rows]
        pr("rec.n", len(rows), fs, "certification.json files")
        if cert is not None:
            pr("rec.cert.n", sum(1 for x in rows if (x[1], x[5]) in cert), [cf_] + fs,
               "rows whose recipe and machine certified.json lists")
            missing = sorted({(r_, m_) for r_, m_ in cert} - {(x[1], x[5]) for x in rows})
            if missing:
                PROBLEMS.append(f"certified.json lists runs with no certification.json here: {missing}")
        pr("rec.cuda.n", sum(1 for x in rows if not x[0].startswith("M4")), fs, "CUDA certification.json files")
        pr("rec.mac.n", sum(1 for x in rows if x[0].startswith("M4")), fs, "Mac certification.json files")
        pr("rec.resume.n", sum(1 for x in rows if VALUES[f"{x[3]}.res"] == "identical"), fs, "resume.adapter_equal")
        pr("rec.noresume", ", ".join(f"\\code{{{tex_escape(x[1])}}}" for x in rows if VALUES[f"{x[3]}.res"] != "identical")
           or "none", fs, "resume.adapter_equal", lambda v: v)

    def get(mach, rcp):
        return CERT.get((mach, rcp), (None, None))
    f, c = get("rtx4090", "big-72b")
    if c:
        pr("recipe.model", c["command"][2].split("/", 1)[1], f, "command[2]")
        pr("recipe.release", c["poros"], f, "poros")
        pr("recipe.rows", c["data"]["rows"], f, "data.rows")
        pr("recipe.heldout", c["data"]["held_out"], f, "data.held_out")
        pr("recipe.seq", c["data"]["seq_len"], f, "data.seq_len")
        pr("recipe.updates", c["main"]["updates"], f, "main.updates")
        pr("recipe.supd", c["main"]["s_per_step"], f, "main.s_per_step", lambda v: f"{v:.2f}")
        pr("recipe.alloc", c["torch_memory"]["allocated_peak_gib"], f, "torch_memory.allocated_peak_gib", lambda v: f"{v:.2f}")
        pr("recipe.res", c["torch_memory"]["reserved_peak_gib"], f, "torch_memory.reserved_peak_gib", lambda v: f"{v:.2f}")
        pr("recipe.host", c["host_ram"]["peak_gib"], f, "host_ram.peak_gib", lambda v: f"{v:.1f}")
        pr("recipe.eval.before", c["eval"]["metric"]["base_loss"], f, "eval.metric.base_loss", lambda v: f"{v:.2f}")
        pr("recipe.eval.after", c["eval"]["metric"]["eval_loss"], f, "eval.metric.eval_loss", lambda v: f"{v:.2f}")
        pr("recipe.path", VALUES["rec.r4090.big-72b.path"].replace("eff.", "efficient"), f.parent / "main/events.jsonl",
           "setup.path")
    for short_, rcp in (("r30ssd", "moe-30b-ssd"), ("r30", "moe-30b")):
        f, c = get("rtx4090", rcp)
        if c:
            pr(f"{short_}.supd", c["main"]["s_per_step"], f, "main.s_per_step", lambda v: f"{v:.2f}")
            pr(f"{short_}.res", c["torch_memory"]["reserved_peak_gib"], f, "torch_memory.reserved_peak_gib", lambda v: f"{v:.2f}")
            pr(f"{short_}.host", c["host_ram"]["peak_gib"], f, "host_ram.peak_gib", lambda v: f"{v:.1f}")
            pr(f"{short_}.before", c["eval"]["metric"]["base_loss"], f, "eval.metric.base_loss", lambda v: f"{v:.2f}")
            pr(f"{short_}.after", c["eval"]["metric"]["eval_loss"], f, "eval.metric.eval_loss", lambda v: f"{v:.2f}")
            cf = f.parent / "main/config.resolved.json"
            cr = load_json(cf)
            pr(f"{short_}.lr", cr["learning_rate"], cf, "learning_rate", sci)
            pr(f"{short_}.updates", cr["max_steps"], cf, "max_steps")
            if short_ == "r30ssd":
                if cr.get("store") != "ssd":
                    PROBLEMS.append(f"{rel(cf)}: moe-30b-ssd did not keep its experts in the SSD store")
                pr("r30ssd.cache", cr["moe_expert_cache_gib"], cf, "moe_expert_cache_gib")
                hc = (cr.get("store_options") or {}).get("cache")
                if hc:
                    pr("r30ssd.hostcache", str(hc).replace("G", "\\,GiB"), cf, "store_options.cache", lambda v: v)
            if short_ == "r30" and cr.get("store", "ram") != "ram":
                PROBLEMS.append(f"{rel(cf)}: moe-30b did not keep its experts in host memory")
            pr(f"{short_}.path", VALUES[f"rec.r4090.{rcp}.path"].replace("eff.", "efficient"), f.parent / "main/events.jsonl",
               "setup.path", lambda v: v)
    # the laptop: moe-30b-8gb
    f, c = get("rtx4060-laptop", "moe-30b-8gb")
    if c:
        cert, cf_ = certified_set()
        if cert is not None:
            if ("moe-30b-8gb", "rtx4060-laptop") in cert:  # Sections 1, 5.5, 5.6 and 7 call it a measured run
                PROBLEMS.append(f"{rel(cf_)} certifies moe-30b-8gb on the RTX 4060 Laptop GPU; the text says measured")
        mc = c["machine"]
        pr("lap.gpu", mc["gpu"].replace("NVIDIA GeForce ", ""), f, "machine.gpu")
        pr("lap.vram.gib", mc["vram_gib"], f, "machine.vram_gib")
        pr("lap.host.gib", mc["host_ram_gib"], f, "machine.host_ram_gib", lambda v: f"{v:.1f}")
        cf = f.parent / "main/config.resolved.json"
        cr = load_json(cf)
        for k, fld in (("lap.rank", "lora_rank"), ("lap.seq", "max_seq_length"), ("lap.mb", "per_device_batch_size"),
                       ("lap.ga", "gradient_accumulation_steps"), ("lap.updates", "max_steps")):
            pr(k, cr[fld], cf, fld)
        pr("lap.supd", c["main"]["s_per_step"], f, "main.s_per_step", lambda v: f"{v:.2f}")
        pr("lap.alloc", c["torch_memory"]["allocated_peak_gib"], f, "torch_memory.allocated_peak_gib", lambda v: f"{v:.2f}")
        pr("lap.res", c["torch_memory"]["reserved_peak_gib"], f, "torch_memory.reserved_peak_gib", lambda v: f"{v:.2f}")
        pr("lap.host", c["host_ram"]["peak_gib"], f, "host_ram.peak_gib", lambda v: f"{v:.1f}")
        pr("lap.before", c["eval"]["metric"]["base_loss"], f, "eval.metric.base_loss", lambda v: f"{v:.2f}")
        pr("lap.after", c["eval"]["metric"]["eval_loss"], f, "eval.metric.eval_loss", lambda v: f"{v:.2f}")
        evf = f.parent / "main/events.jsonl"
        stp = [json.loads(x)["data"] for x in evf.read_text().splitlines() if x.strip()
               and json.loads(x).get("event_type") == "setup" and "path" in json.loads(x)["data"]][0]
        pr("lap.path", stp["path"], evf, "setup.path")
        # the repeat: a second full certification run of the recipe; its adapter must be the first's
        reps = REPEATS.get(("rtx4060-laptop", "moe-30b-8gb"), [])
        if reps:
            runs_ = [f] + reps
            shas = []
            for rf in runs_:
                rc_ = load_json(rf)
                release(rc_["commit"], rf)
                if not (rc_["status"] == "ok" and rc_["tree_clean"] and not rc_["simulated"]):
                    PROBLEMS.append(f"{rel(rf)}: the repeat run did not pass")
                if rc_["resume"].get("adapter_equal") is not True:
                    PROBLEMS.append(f"{rel(rf)}: the repeat's stop-and-continue did not end at its adapter")
                shas.append(rc_["main"]["adapter_sha256"])
            if len(set(shas)) == 1:
                pr("lap.repeat.runs", len(runs_), runs_, "main.adapter_sha256 of each certification run (one value)")
            else:
                PROBLEMS.append(f"the laptop's moe-30b-8gb runs end at different adapters ({sorted(set(shas))})")
    if c and c["resume"].get("adapter_equal") is not True:
        PROBLEMS.append(f"{rel(f)}: the laptop recipe's stop-and-continue did not end at the uninterrupted adapter "
                        "(Section 5 says it did)")
    # Scope (Section 6): the head-to-head and the recipes used rows of at most wl.seq tokens and at most
    # wl.updates updates
    if "wl.seq" in RAW and "wl.updates" in RAW:
        for (mach, rcp), (f_, c_) in CERT.items():
            if c_["data"]["seq_len"] > RAW["wl.seq"] or c_["main"]["updates"] > RAW["wl.updates"]:
                PROBLEMS.append(f"{rel(f_)}: {c_['data']['seq_len']} tokens, {c_['main']['updates']} updates: "
                                "past the Scope sentence's bound")
    # the Mac recipes the text names
    for short_, rcp in (("mac72", "mac-72b"), ("mac32", "mac-32b"), ("macmoe", "mac-moe")):
        f, c = get("m4-16gb", rcp)
        if c:
            pr(f"{short_}.supd", c["main"]["s_per_step"], f, "main.s_per_step", lambda v: f"{v:.1f}")
            pr(f"{short_}.mem", c["torch_memory"]["allocated_peak_gib"], f, "torch_memory.allocated_peak_gib", lambda v: f"{v:.1f}")
            pr(f"{short_}.updates", c["main"]["updates"], f, "main.updates")
            pr(f"{short_}.seq", c["data"]["seq_len"], f, "data.seq_len")
            if rcp == "mac-72b":
                dv, df = drive(f.parent)
                if dv:
                    pr("mac72.drive", dv, df, "first words of model-drive.txt", lambda v: v)
            if rcp == "mac-72b":
                mc = c["machine"]
                pr("mac.gpu", mc["gpu"].replace(" 16GB", ""), f, "machine.gpu")
                pr("mac.gib", mc["vram_gib"], f, "machine.vram_gib", lambda v: f"{v:.0f}")
                pr("mac.torch", mc["torch"], f, "machine.torch")
                pr("mac.os", mc["os_version"], f, "machine.os_version")


# ── what each path ran with (table_paths): the big-72b and big-72b-fast recipes ──
PATH_SETTINGS = ["ran_path", "block_size", "resident_blocks", "offload_embeddings_and_head", "nf4_dequant_cache_mb",
                 "fused_compute", "cuda_expandable_segments", "host_staging"]


def paths() -> None:
    for k, rcp in (("eff", "big-72b"), ("fast", "big-72b-fast")):
        f, c = CERT.get(("rtx4090", rcp), (None, None))
        if not c:
            continue
        cf = f.parent / "main/config.resolved.json"
        cr = load_json(cf)
        rp = f.parent / "main/receipt.json"
        rc_ = load_json(rp) if rp.exists() else {}
        fmt_ = lambda x: f"\\code{{{tex_escape(x if isinstance(x, str) else json.dumps(x))}}}"
        for s in PATH_SETTINGS:
            ran = (cr.get("ran_compute") or {})
            if s in ran:  # what the run did, where it recorded it, not what was asked
                pr(f"paths.{k}.{s}", ran[s], cf, f"ran_compute.{s}", fmt_)
            elif s == "host_staging" and isinstance(rc_.get("host_staging"), dict):
                pr(f"paths.{k}.{s}", rc_["host_staging"]["mode"], rp, "host_staging.mode", fmt_)
            elif s in cr and cr[s] is not None:
                pr(f"paths.{k}.{s}", cr[s], cf, s, fmt_)
            elif s in cr:
                PROBLEMS.append(f"{rel(cf)}: {s} is null and the run records no value for it")


# ── the Mac's 144-step Qwen2.5-72B-Instruct run ───────────────────────────────
def mac144() -> None:
    cands = [p.parent for p in sorted(VAL.glob("int/mac/*/receipt.json")) + sorted(MACLOGS.glob("**/receipt.json"))]
    for d in cands:
        rc = load_json(d / "receipt.json")
        if rc.get("steps") != 144 or rc.get("simulated"):
            continue
        # the run's commit: a machine-written commit.txt beside it (a direct poros train run records none in
        # its receipt), else the README or .kit_commit of the folder
        c = ((d / "commit.txt").read_text().split()[0] if (d / "commit.txt").exists() else None) \
            or readme_commit(d / "README.md") \
            or ((d / ".kit_commit").read_text().strip() if (d / ".kit_commit").exists() else None)
        if not is_frozen(c):
            NOT_USED.append(f"{rel(d / 'receipt.json')} (commit {short(c)})")
            continue
        release(c, d)
        pr("m144.steps", rc["steps"], d / "receipt.json", "steps")
        pr("m144.seq", rc["config"]["max_seq_length"], d / "receipt.json", "config.max_seq_length")
        pr("m144.rank", rc["config"]["lora_rank"], d / "receipt.json", "config.lora_rank")
        dv, df = drive(d)
        if dv:
            pr("m144.drive", dv, df, "first words of model-drive.txt", lambda v: v)
        pr("m144.peak", rc["peak_vram_allocated_gb"] * 1e9 / 2**30, d / "receipt.json",
           "peak_vram_allocated_gb (decimal GB, as GiB)", lambda v: f"{v:.1f}")
        pr("m144.min", rc["wall_s"] / 60, d / "receipt.json", "wall_s / 60", lambda v: f"{v:.0f}")
        ev = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines() if x.strip()]
        st = [e["data"]["s_per_step"] for e in ev if e["event_type"] == "step"]
        if len(st) != 144:
            PROBLEMS.append(f"{rel(d)}: {len(st)} step events, not 144")
        su = [e["data"] for e in ev if e["event_type"] == "setup" and "rows" in e["data"]]
        if su:
            pr("m144.rows", su[0]["rows"], d / "events.jsonl", "setup.rows (rows trained on, the held-out rows apart)")
            pr("m144.data", su[0]["data"], d / "events.jsonl", "setup.data")
        pr("m144.supd", statistics.median(st[1:]), d / "events.jsonl", "median of step s_per_step, steps 2..144",
           lambda v: f"{v:.1f}")
        ef = d / "eval.json"
        if ef.exists():
            e = load_json(ef)
        else:  # poros eval's own JSON line in eval.log
            ef = d / "eval.log"
            e = [json.loads(x) for x in ef.read_text().splitlines() if x.startswith('{"schema":"poros.eval.v1"')][-1]
        if e["rows_from"] != "held_out":
            PROBLEMS.append(f"{rel(d)}: eval rows are not held out")
        pr("m144.heldout", e["rows"], ef, "rows")
        pr("m144.before", e["base"]["loss"], ef, "base.loss", lambda v: f"{v:.2f}")
        pr("m144.after", e["trained"]["loss"], ef, "trained.loss", lambda v: f"{v:.2f}")
        pr("m144.ends.before", e["base"]["ends"], ef, "base.ends")
        pr("m144.ends.after", e["trained"]["ends"], ef, "trained.ends")
        return


# ── the two optimizer studies (scripts/moe_precision: kit.sh, compare.py), both on the RTX 4090 ──
# A study is a verdict file WORK/verdicts/<name>.json (compare.py's one JSON object) and its runs
# WORK/runs/<run>/ (events.jsonl, config.resolved.json, receipt.json, sample.jsonl, .kit_commit).  Every
# arm must have run the frozen commit.  The control arm is the same engine with PyTorch's AdamW stepping
# the bf16 expert adapters; its keys (olmoe.ctl.*) are filled only where its run is there at the frozen
# commit, and the text prints its sentence only then (\nvif).  CONTROL_RUNS names that run once its
# path is known; until then no control figure is read.  (The verdict file of the OLMoE study keeps the
# OLMoE study's verdict is cuda_olmoe.json: it runs on the RTX 4090.)
STUDIES = {  # prefix: (verdict name, new, fp32, fp32 other seed)
    "olmoe": ("cuda_olmoe", "o_new_olmoe_s42", "o_fp32_olmoe_s42", "o_fp32_olmoe_s43"),
    "q30": ("q2e5", "q_new_c2e5_s42", "q_fp32_c2e5_s42", "q_fp32_c2e5_s43"),
}
# The control run is never named here: its figures are the verdict's own `control` entry (compare.py,
# --control), and its run is found by its .kit_arm.  An old-commit arm a verdict may report is never read.


def kit_commit(run: pathlib.Path):
    f = run / ".kit_commit"
    return f.read_text().strip() if f.exists() else None


def study_samples(run: pathlib.Path):
    for f in (run / "sample.jsonl", run.parent / f"{run.name}.sample.jsonl"):
        if f.exists():
            return [json.loads(x) for x in f.read_text().splitlines() if x.strip()], f
    return [], None


def unchanged_mean(run: pathlib.Path):
    s, sf = study_samples(run)
    w = [x["weight_unchanged"] for x in s if x.get("weight_unchanged") is not None]
    return (100 * sum(w) / len(w), sf) if w else (None, sf)


def studies() -> None:
    fmt4 = lambda x: f"{x:.4f}"
    for pre, (vname, rnew, rfp, rfp2) in STUDIES.items():
        for vp in in_rounds(f"**/verdicts/{vname}.json"):
            work = vp.parent.parent
            runs = {k: work / "runs" / r for k, r in (("new", rnew), ("fp32", rfp), ("fp32b", rfp2))}
            if not all((r / "events.jsonl").exists() for r in runs.values()):
                PROBLEMS.append(f"{rel(vp)}: a run of the study is missing beside its verdict")
                continue
            if not is_frozen(kit_commit(runs["new"])):
                NOT_USED.append(f"{rel(vp)} (new run at {short(kit_commit(runs['new']))})")
                continue
            for k in runs:
                release(kit_commit(runs[k]), runs[k] / ".kit_commit")
            v = load_json(vp)
            if v.get("status") != "pass":
                PROBLEMS.append(f"{rel(vp)}: the study's verdict is {v.get('status')!r}: the text may not stand")
            rows = v["held_out"]
            if not rows:
                PROBLEMS.append(f"{rel(vp)}: no held-out points")
                continue
            # the machine, from each arm's receipt: one GPU for all of them
            gpus = []
            for k, r in runs.items():
                rp = r / "receipt.json"
                gpus.append(load_json(rp)["gpu"]["name"] if rp.exists() else None)
            g = same(gpus, f"{pre} study GPU")
            if g:
                pr(f"{pre}.gpu", g.replace("NVIDIA GeForce ", "").replace("RTX ", "RTX~"), [r / "receipt.json" for r in runs.values()],
                   "receipt.json gpu.name (every arm)", lambda x: x)
            # every arm ran the same number of updates (config.resolved.json max_steps), the control's too
            arms_ = dict(runs)
            arms_.update({r.name: r for r in (work / "runs").glob(rnew.split("_", 1)[0] + "_ctl_*")
                          if (r / ".kit_arm").exists() and (r / ".kit_arm").read_text().startswith("control")})
            same((load_json(r / "config.resolved.json")["max_steps"] for r in arms_.values()), f"{pre} arms' max_steps")
            # Section 5.4: the fp32 arms ran the reference expert compute, the expert optimizer's arm the fused one
            for k, want in (("new", "fused"), ("fp32", "reference"), ("fp32b", "reference")):
                got = (load_json(runs[k] / "config.resolved.json").get("ran_compute") or {}).get("moe_expert_compute")
                if got != want:
                    PROBLEMS.append(f"{rel(runs[k])}: expert compute {got!r}, the text says {want}")
            cn = load_json(runs["new"] / "config.resolved.json")
            pr(f"{pre}.lr", cn["learning_rate"], runs["new"] / "config.resolved.json", "learning_rate", sci)
            pr(f"{pre}.updates", cn["max_steps"], runs["new"] / "config.resolved.json", "max_steps")
            SOURCES[f"{pre}.lr"] = SOURCES[f"{pre}.lr"] + [rel(runs[k] / ".kit_commit") for k in runs]
            last = rows[-1]
            pr(f"{pre}.lastpoint", last["step"], vp, "held_out[-1].step")
            pr(f"{pre}.maxdiff", max(abs(r["new_minus_fp32"]) for r in rows), vp,
               "max over held_out[] of |new_minus_fp32|", fmt4)
            pr(f"{pre}.seedgap", max(abs(r["fp32"] - r["fp32_other"]) for r in rows), vp,
               "max over held_out[] of |fp32 - fp32_other|", fmt4)
            if pre == "olmoe":
                pr("olmoe.heldout.new", last["new"], vp, "held_out[-1].new", fmt4)
                pr("olmoe.heldout.fp32", last["fp32"], vp, "held_out[-1].fp32", fmt4)
                for k in ("new", "fp32"):
                    u, sf = unchanged_mean(runs[k])
                    if u is not None:
                        pr(f"olmoe.unchanged.{k}", u, sf, "mean of weight_unchanged over updates", lambda x: f"{x:.2f}\\%")
                s, sf = study_samples(runs["new"])
                late = [x["v_over_exact_median"] for x in s if x["update"] >= 20 and x.get("v_over_exact_median") is not None]
                if late:
                    pr("olmoe.vratio.new", f"{min(late):.5f} to {max(late):.5f}", sf,
                       "min and max of v_over_exact_median, updates 20 on")
            else:
                at = {r["step"]: r for r in rows}
                if set(at) != {50, 100, 150, 200}:  # Section 5.4: Table opt lists the held-out loss at every scored point
                    PROBLEMS.append(f"{rel(vp)}: scored points {sorted(at)}, not exactly the table's 50, 100, 150 and 200")
                for n in (50, 100, 150, 200):
                    if n not in at:
                        PROBLEMS.append(f"{rel(vp)}: no held-out point at update {n}")
                        continue
                    pr(f"q30.h{n}.new", at[n]["new"], vp, f"held_out[step={n}].new", fmt4)
                    pr(f"q30.h{n}.fp32", at[n]["fp32"], vp, f"held_out[step={n}].fp32", fmt4)
            # every published arm names the frozen commit (compare.py check 4)
            named = v.get("commits") or {}
            if not v.get("check_4_release_commit") or not named:
                PROBLEMS.append(f"{rel(vp)}: the verdict was not run with --commit, or its check 4 failed")
            for arm, c_ in named.items():
                release(c_, vp)
            # the control arm: the same engine, PyTorch's AdamW on the bf16 adapters (compare.py --control);
            # its figures from the verdict's control entry, only where it names the frozen commit
            ctl = v.get("control")
            if pre == "olmoe" and ctl:
                if not (v.get("check_5_control_same_engine") and str(ctl.get("arm") or "").startswith("control")):
                    PROBLEMS.append(f"{rel(vp)}: the control arm is not the same engine (check 5)")
                release(named.get("control"), vp)
                def is_control(r):
                    ro = (load_json(r / "config.resolved.json").get("ran_compute") or {}).get("expert_optimizer") or {}
                    arm = (r / ".kit_arm").read_text() if (r / ".kit_arm").exists() else "control"
                    return ro.get("substituted") == "torch.optim.AdamW" and arm.startswith("control")
                cruns = [r for r in sorted((work / "runs").glob(rnew.split("_", 1)[0] + "_ctl_*"))
                         if (r / "config.resolved.json").exists() and is_control(r)]
                if len(cruns) != 1:
                    PROBLEMS.append(f"{rel(vp)}: {len(cruns)} control runs beside the verdict, not one")
                else:
                    cr_ = cruns[0]
                    release(kit_commit(cr_), cr_ / ".kit_commit")
                    if load_json(cr_ / "config.resolved.json")["max_steps"] != cn["max_steps"]:
                        PROBLEMS.append(f"{rel(cr_)}: the control ran another number of updates")
                    s, sf = study_samples(cr_)
                    late = [x for x in s if x.get("v_over_exact_median") is not None]
                    if late and late[-1]["v_over_exact_median"] != ctl.get("v_over_exact_last"):
                        PROBLEMS.append(f"{rel(vp)}: the control's sample does not match the verdict")
                    hk = ctl["held_out"].get(str(last["step"]), ctl["held_out"].get(last["step"]))
                    if ctl.get("weight_unchanged_mean") is not None and late and hk is not None:
                        pr("olmoe.ctl.unchanged", 100 * ctl["weight_unchanged_mean"], vp, "control.weight_unchanged_mean",
                           lambda x: f"{x:.2f}\\%")
                        pr("olmoe.ctl.vratio", ctl["v_over_exact_last"], vp, "control.v_over_exact_last", lambda x: f"{x:.3f}")
                        pr("olmoe.ctl.vratio.at", late[-1]["update"], sf, "update of the control's last sample")
                        pr("olmoe.ctl.heldout", hk, vp, f"control.held_out[{last['step']}]", fmt4)
                        # time per training update, same engine: the expert optimizer against the control
                        for key, r in (("olmoe.supd.new", runs["new"]), ("olmoe.ctl.supd", cr_)):
                            st_ = [json.loads(x)["data"]["s_per_step"] for x in (r / "events.jsonl").read_text().splitlines()
                                   if x.strip() and json.loads(x).get("event_type") == "step"]
                            pr(key, statistics.median(st_[1:]), r / "events.jsonl", "median of step s_per_step, updates 2..N",
                               lambda x: f"{x:.3f}")
            break


# ── tables (every cell a key; values.tex fills them) ─────────────────────────
def write_tables() -> None:
    def row(label, pre, vs=None, porosrow=False):
        tint = "\\porosrow " if porosrow else ""
        mk = "$^{*}$" if RAW.get(f"{pre}.host.procs", 0) > 3 else ""
        cmp_ = (f"\\nvc{{{vs}.vs.s_per_step}}; \\nvc{{{vs}.vs.vram_reserved}}" if vs else "")
        return (f"{tint}{label} & \\nvc{{{pre}.supd}} & \\nvc{{{pre}.alloc}} / \\nvc{{{pre}.res}} & "
                f"\\nvc{{{pre}.host.train}}{mk} / \\nvc{{{pre}.host}}{mk} & {cmp_} \\\\")

    def bestrow(pre):
        mk = "$^{*}$" if RAW.get(f"{pre}.host.procs", 0) > 3 else ""
        return (f"\\quad its lowest-value run of three & \\nvc{{{pre}.supd.best}} & reserved \\nvc{{{pre}.res.best}} & "
                f"\\nvc{{{pre}.host.train.best}}{mk} / \\nvc{{{pre}.host.best}}{mk} & \\\\")

    h2h_t = rf"""% Generated by launch_results.py -- do not edit.
\begin{{table}}[tbp]
\centering
\caption{{QLoRA on one RTX~4090 (24\,GB) for two models on which Unsloth's automatic offloading
keeps decoder layers in host memory, Unsloth in its deterministic mode.  \nvc{{moe.model}}: Poros at the release
commit \nvc{{code.commit}} against Unsloth \nvc{{env.unsloth}}, the newest version on PyPI on \nvc{{env.unsloth.date}}.
Qwen2.5-72B: the \nvc{{dense.date}} session, Poros at its commit \nvc{{dense.commit}} against Unsloth
\nvc{{prev.unsloth.version}}, with the same harness and rows (the 72B comparison was not repeated at the release commit).  Automatic offloading keeps the
decoder layers that fit on the GPU, so its reserved memory measures that placement, not its lowest setting.
On \nvc{{moe.model}}, Unsloth \nvc{{env.unsloth}} ran \nvc{{moe.unsloth.slower.median}} slower per update by median
(\nvc{{moe.unsloth.slower.best}} by its fastest run) than Unsloth \nvc{{prev.unsloth.version}} did on
\nvc{{prev.unsloth.date}} with the same harness and rows (\nvc{{prev.moe.unsloth.supd.best}}\,s at its fastest run,
\nvc{{prev.moe.unsloth.res.best}}\,GiB reserved); the logs do not show why.  With all
\nvc{{prev.moe.unsmin.offloaded}} decoder layers in host memory, Unsloth \nvc{{prev.unsloth.version}} ran at
\nvc{{prev.moe.unsmin.supd.best}}\,s per update at its fastest run (\nvc{{prev.moe.unsmin.supd}}\,s median) and reserved
\nvc{{prev.moe.unsmin.res.best}}\,GiB on that date.  On \nvc{{moe.model}}, Unsloth stored fp32 adapters and Poros bf16 ones (about
\nvc{{moe.adapter.gap.gib}}\,GiB if Unsloth's moments were fp32; computed).
$^{{*}}$Unsloth's host peak on \nvc{{env.unsloth.date}} sums \nvc{{moe.unsloth.host.procs}} processes (a sum over forked processes can count
shared pages more than once); over samples with three or fewer it was \nvc{{moe.unsloth.host.few}}\,GiB.  s/upd.: seconds per
optimizer update (\nvc{{wl.ga}} micro-batches of \nvc{{wl.mb}} row of \nvc{{wl.seq}} tokens), the median over the
updates after the initial one, then the median of three runs.  GPU: peak allocated / peak reserved by PyTorch.
Host: peak host memory that is not file-backed, summed over the process tree, during training / over the whole run.
Memory in GiB.  The last column compares Poros's median with Unsloth's fastest run of three for time and its
lowest-value run of three for memory, given under its row.  Orange rows are Poros.}}
\label{{tab:h2h}}
\footnotesize
\setlength{{\tabcolsep}}{{3.2pt}}
\begin{{tabular}}{{@{{}}>{{\raggedright\arraybackslash}}p{{4.1cm}}rccp{{3.85cm}}@{{}}}}
\toprule
\tabhead{{Configuration}} & \tabhead{{s/upd.}} & \tabhead{{GPU}} & \tabhead{{Host}} & \tabhead{{Speed; reserved GPU memory, against Unsloth}} \\
\midrule
\multicolumn{{5}}{{@{{}}l}}{{\emph{{Qwen2.5-72B}}, dense, \nvc{{dense.layers}} decoder layers}} \\
{row("Unsloth, automatic offloading (\\nvc{dense.unsloth.offloaded} of \\nvc{dense.layers} layers in host memory)", "dense.unsloth")}
{bestrow("dense.unsloth")}
{row("Poros, efficient path (default)", "dense.poros", "dense.poros", porosrow=True)}
\addlinespace
\multicolumn{{5}}{{@{{}}l}}{{\emph{{\nvc{{moe.model}}}}, mixture of experts, \nvc{{moe.layers}} decoder layers}} \\
{row("Unsloth, automatic offloading (\\nvc{moe.unsloth.offloaded} of \\nvc{moe.layers} layers in host memory)", "moe.unsloth")}
{bestrow("moe.unsloth")}
{row("Poros, experts in host memory, efficient path (default)", "moe.ram", "moe.ram", porosrow=True)}
\bottomrule
\end{{tabular}}
\tabnote{{Settings: \cref{{tab:setup}}; both tools read the same rows in the same order. Comparisons set
Poros's median against the other tool's lowest-value run of three. Ratios are given to one decimal (a whole
percent under 1.1 times), rounded against Poros (down where Poros is ahead, up where the other tool is
ahead); a difference that rounds to 0\% reads ``the same''.}}
\end{{table}}
"""
    (HERE / "table_h2h.tex").write_text(h2h_t)

    setup_rows = [
        ("Base model weights", "NF4 with double quantization and bf16 compute, quantized at load from the bf16 checkpoint; on \\nvc{moe.model} the routed experts only, with attention, norms, router and embeddings in bf16",
         "the same local snapshot, quantized at load with the same NF4, double-quantization and bf16-compute settings (the session's record; its log does not list which layers it quantized)"),
        ("LoRA rank, $\\alpha$, dropout", "\\nvc{wl.rank}, \\nvc{wl.alpha}, \\nvc{wl.dropout}", "rank \\nvc{wl.unsloth.rank}"),
        ("LoRA targets (Qwen2.5-72B)", "every linear layer of every decoder layer", "\\nvc{wl.unsloth.targets}"),
        ("Trainable parameters (Qwen2.5-72B)", "\\nvc{wl.trainable72}", "\\nvc{wl.trainable72}"),
        ("LoRA targets (\\nvc{moe.model})", "every routed expert's gate and up projection and its down projection",
         "the same, and the attention output projection, held at zero"),
        ("Trainable parameters (\\nvc{moe.model})", "\\nvc{moe.trainable}", "\\nvc{moe.trainable}"),
        ("Adapter dtype (Qwen2.5-72B; \\nvc{moe.model})", "\\nvc{dense.adapter.dtype}; \\nvc{moe.adapter.dtype}",
         "\\nvc{dense.unsloth.adapter.dtype}; \\nvc{moe.unsloth.adapter.dtype}"),
        ("Tokens per row", "\\nvc{wl.seq}", "\\nvc{wl.seq}"),
        ("Rows per micro-batch $\\times$ micro-batches", "\\nvc{wl.mb} $\\times$ \\nvc{wl.ga}", "\\nvc{wl.mb} $\\times$ \\nvc{wl.ga}"),
        ("Learning rate, weight decay, clipping", "\\nvc{wl.lr}, \\nvc{wl.wd}, \\nvc{wl.clip}",
         "\\nvc{wl.unsloth.lr}, \\nvc{wl.unsloth.wd}, \\nvc{wl.unsloth.clip}"),
        ("Optimizer, schedule, seed", "AdamW ($\\beta$ \\nvc{wl.betas}): PyTorch's on fp32 adapters, \\cref{sec:expertadamw}'s on bf16 expert adapters; \\nvc{wl.sched}, \\nvc{wl.seed}",
         "seed \\nvc{wl.unsloth.seed}"),
        ("Data", "\\nvc{wl.rows} rows of oasst1", "the same rows (one SHA-256)"),
        ("Updates per run", "\\nvc{wl.updates}", "\\nvc{wl.updates}"),
        ("Determinism", "deterministic algorithms", "deterministic mode"),
        ("Gradient checkpointing", "per block (recompute)", "\\nvc{wl.unsloth.gc}"),
        ("Offloading (Qwen2.5-72B)", "every block streamed",
         "\\nvc{dense.unsloth.offloaded} of \\nvc{dense.layers} layers in host memory; input embedding on \\nvc{dense.unsloth.embed}"),
        ("Software", "PyTorch \\nvc{env.torch}, \\code{transformers} \\nvc{env.transformers}, PEFT \\nvc{env.peft}; on Qwen2.5-72B, Poros at \\nvc{dense.commit} (which records itself as Poros \\nvc{dense.poros.version}) with the same PyTorch, \\code{transformers}, PEFT and \\code{bitsandbytes}",
         "Unsloth \\nvc{env.unsloth}, \\code{transformers} \\nvc{env.unsloth.transformers}, PEFT \\nvc{env.unsloth.peft} (Qwen2.5-72B: Unsloth \\nvc{prev.unsloth.version}, \\nvc{dense.date})"),
    ]
    body = "\n".join(f"{a} & {b} & {c} \\\\" for a, b, c in setup_rows)
    (HERE / "table_setup.tex").write_text(rf"""% Generated by launch_results.py -- do not edit.
\begin{{table}}[tbp]
\centering
\caption{{Settings of the head-to-head runs, as each run recorded them: Poros's resolved configuration
and its run records, and the Unsloth harness's record.  Poros's quantization settings and optimizer
are its code's (\cref{{sec:setup,sec:expertadamw}}).  The Unsloth record does not store LoRA $\alpha$,
dropout, the optimizer's name or its learning-rate schedule.}}
\label{{tab:setup}}
\footnotesize
\begin{{tabular}}{{@{{}}>{{\raggedright\arraybackslash}}p{{3.6cm}}>{{\raggedright\arraybackslash}}p{{4.6cm}}>{{\raggedright\arraybackslash}}p{{4.6cm}}@{{}}}}
\toprule
\tabhead{{Setting}} & \tabhead{{Poros}} & \tabhead{{Unsloth}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
""")

    labels = {"ran_path": "Path the run recorded (no fallback)", "block_size": "Block size (layers)",
              "resident_blocks": "Blocks kept on the GPU", "offload_embeddings_and_head": "Embedding and head in host memory",
              "nf4_dequant_cache_mb": "Dequantization cache (MiB; 0: off)", "fused_compute": "Fused compute",
              "cuda_expandable_segments": "Expandable allocator segments", "host_staging": "Host memory for block inputs and blocks"}
    body = "\n".join(f"{labels[s]} & \\nvc{{paths.eff.{s}}} & \\nvc{{paths.fast.{s}}} \\\\" for s in PATH_SETTINGS)
    (HERE / "table_paths.tex").write_text(rf"""% Generated by launch_results.py -- do not edit.
\begin{{table}}[tbp]
\centering
\caption{{What each path ran with in the \code{{big-72b}} (efficient) and \code{{big-72b-fast}}
(fast) recipes on the RTX~4090, as each run's resolved configuration recorded it.}}
\label{{tab:paths}}
\small
\begin{{tabular}}{{@{{}}lcc@{{}}}}
\toprule
\tabhead{{Setting}} & \tabhead{{efficient}} & \tabhead{{fast}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
""")

    def rows(rs):
        return "\n".join(" & ".join(r) + " \\\\" for r in rs)
    resident = [
        ("\\code{poros prove}: streamed, blocks of \\nvc{prove.str.bs} layers", "resident (\\nvc{prove.res.bs} layers, ordinary autograd)",
         "\\nvc{prove.model}", "4090", "\\nvc{prove.steps}",
         "equal: losses, \\nvc{prove.grads} gradients, \\nvc{prove.tensors} adapter and \\nvc{prove.opt} optimizer tensors"),
    ]
    between = [
        ("Sequential; block-major on the efficient and fast paths", "every block on the GPU (in Poros's trainer)",
         "the family gate's \\nvc{fg.n} rows", "4090", "", "identical (one digest per row)"),
        ("Block-major, efficient and fast paths, dropout \\nvc{do.dropout}", "sequential", "\\nvc{do.7b.model}", "4090",
         "\\nvc{do.7b.steps}", "equal: losses, gradient norms, \\nvc{do.7b.tensors} adapter tensors"),
        ("Block-major, efficient path, dropout \\nvc{do.dropout}", "sequential", "OLMoE-1B-7B", "4090",
         "\\nvc{do.olmoe.steps}", "equal: losses, gradient norms, \\nvc{do.olmoe.tensors} adapter tensors"),
        ("Block-major, dropout \\nvc{do.dropout}", "sequential", "\\nvc{do.32b.model}", "4090",
         "\\nvc{do.32b.steps}", "equal: losses, gradient norms, \\nvc{do.32b.tensors} adapter tensors"),
    ]
    repeats = [
        ("Fast path, run again", "fast path", "the family gate's \\nvc{fg.n} rows", "4090", "", "identical"),
        ("Stopped and continued", "uninterrupted", "\\nvc{fg.resume.models}", "4090", "", "identical"),
        ("Stopped and continued", "uninterrupted", "\\nvc{rec.resume.n} recipe runs", "all", "", "identical adapter"),
        ("\\code{moe-30b-8gb}, run again", "\\nvc{lap.repeat.runs} runs", "Qwen3-30B-A3B", "4060L", "\\nvc{lap.updates}", "identical"),
    ]
    (HERE / "table_exact.tex").write_text(rf"""% Generated by launch_results.py -- do not edit.
\begin{{table}}[tbp]
\centering
\caption{{Gates on the release engine, commit \nvc{{code.commit}} (\cref{{sec:testing}}).  ``Identical'': one
SHA-256 over every adapter tensor's name, dtype, shape and raw bytes, and every loss equal.
``Equal'': the loss at every step and every compared tensor equal under \code{{torch.equal}}, which
counts $+0$ and $-0$ as equal.  4060L: the RTX~4060 Laptop GPU.}}
\label{{tab:exact}}
\footnotesize
\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular}}{{@{{}}>{{\raggedright\arraybackslash}}p{{3.1cm}}>{{\raggedright\arraybackslash}}p{{2.4cm}}>{{\raggedright\arraybackslash}}p{{2.6cm}}lr>{{\raggedright\arraybackslash}}p{{3.2cm}}@{{}}}}
\toprule
\tabhead{{Run under test}} & \tabhead{{Reference}} & \tabhead{{Model}} & \tabhead{{GPU}} & \tabhead{{Steps}} & \tabhead{{Result}} \\
\midrule
\multicolumn{{6}}{{@{{}}l}}{{\emph{{Streamed against the whole model on the GPU under ordinary autograd}}}} \\
{rows(resident)}
\addlinespace
\multicolumn{{6}}{{@{{}}l}}{{\emph{{Between Poros schedules and paths}}}} \\
{rows(between)}
\addlinespace
\multicolumn{{6}}{{@{{}}l}}{{\emph{{Repeats and resumption}}}} \\
{rows(repeats)}
\bottomrule
\end{{tabular}}
\end{{table}}
""")

    lines, last = [], None
    for m, r, mo, k, _, certified in RECIPE_ROWS:
        if m != last:
            lines.append(("\\addlinespace\n" if last else "") + f"\\multicolumn{{12}}{{@{{}}l}}{{\\emph{{{m}}}}} \\\\")
            last = m
        mark = "\\pendcell{rec.certified}" if certified is None else ("" if certified else "$^{\\dagger}$")
        lines.append(f"\\code{{{tex_escape(r)}}}{mark} & {tex_escape(mo.replace('-Instruct-2507', '-I').replace('-Instruct', '-I'))} & "
                     + " & ".join(f"\\nvc{{{k}.{x}}}" for x in ("path", "rank", "geo", "tok", "upd", "s", "gpu", "host", "ev", "res"))
                     + " \\\\")
    if not lines:
        lines = ["\\multicolumn{12}{@{}l}{\\pendcell{rec.rows}} \\\\"]
    body = "\n".join(lines)
    (HERE / "table_recipes.tex").write_text(rf"""% Generated by launch_results.py -- do not edit.
\begin{{table}}[tbp]
\centering
\caption{{The built-in recipes, one run of each built-in recipe at commit \nvc{{code.commit}}:
\nvc{{rec.cert.n}} of the \nvc{{rec.n}} runs are certified on their machine; $^{{\dagger}}$: measured, not certified on that
machine (the recipe is certified on another).  Model: -I is -Instruct (-Instruct-2507 for Qwen3-30B-A3B).  $r$: LoRA rank.
$b\times G$: rows per micro-batch $\times$ micro-batches per update.  Tok.: tokens per row.  s/upd.: the median
time per update over updates 2 to the last (every update after the first), as each certification records it.  GPU: PyTorch's peak allocated / reserved GiB; a Mac has one memory
pool, and its row gives the pool's allocated peak.  Host: peak host memory that is not file-backed, GiB
(``pool'' on the Mac).  Path: eff.\ efficient, fast the fast path, seq.\ the sequential schedule.
Held-out: loss of the base model $\to$ the trained adapter on rows the run did not train on (a
preference recipe gives its preference loss).  Resume: the adapter of a run stopped and continued
against the uninterrupted run's (``identical'': the same bytes; ``not run'': the recipe skips the check).}}
\label{{tab:recipes}}
\scriptsize
\setlength{{\tabcolsep}}{{1.2pt}}
\resizebox{{\linewidth}}{{!}}{{%
\begin{{tabular}}{{@{{}}lllrlrrrllll@{{}}}}
\toprule
\tabhead{{Recipe}} & \tabhead{{Model}} & \tabhead{{Path}} & \tabhead{{$r$}} & \tabhead{{$b\times G$}} & \tabhead{{Tok.}} & \tabhead{{Upd.}} & \tabhead{{s/upd.}} & \tabhead{{GPU}} & \tabhead{{Host}} & \tabhead{{Held-out}} & \tabhead{{Resume}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}}}
\end{{table}}
""")




# ── Figure 1 ─────────────────────────────────────────────────────────────────
def figure() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fontdir = HERE.parent / "fonts"
    for fp in list(font_manager.findSystemFonts()) + [str(p) for p in fontdir.glob("Geist-*.otf")]:
        if "Geist" in fp and "Mono" not in fp and "Pixel" not in fp:
            try:
                font_manager.fontManager.addfont(fp)
            except Exception:  # noqa: BLE001
                pass
    FG, MUTED, ACCENT, RULE = "#18181B", "#71717A", "#FF6B35", "#D4D4D8"
    plt.rcParams.update({
        "font.family": "Geist", "font.size": 7.5, "text.color": FG, "axes.labelcolor": FG,
        "axes.edgecolor": FG, "xtick.color": FG, "ytick.color": FG, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5,
        "ytick.major.size": 2.5, "pdf.fonttype": 42, "axes.titlesize": 8,
    })
    panels = [("Qwen2.5-72B (dense)", [("dense.unsloth", "Unsloth, automatic offloading", False, (6, 0), "left"),
                                       ("dense.poros", "Poros, default", True, (6, 0), "left")]),
              (f"{VALUES.get('moe.model', 'Qwen3-30B-A3B')} (MoE)",
               [("moe.unsloth", "Unsloth, automatic offloading", False, (-6, 0), "right"),
                ("moe.ram", "Poros, default", True, (6, 0), "left")])]
    fig, axes = plt.subplots(1, 2, figsize=(5.85, 2.1), gridspec_kw={"wspace": 0.30})
    complete = True
    cap = RAW.get("card.gib")
    for ax, (title, pts) in zip(axes, panels):
        ymax = 0.0
        for key, label, is_p, off, ha in pts:
            x, y = RAW.get(f"{key}.res"), RAW.get(f"{key}.supd")
            if x is None or y is None:
                complete = False
                continue
            ymax = max(ymax, y)
            ax.scatter([x], [y], s=34, marker="o", zorder=3,
                       facecolor=ACCENT if is_p else "white", edgecolor="#C2410C" if is_p else MUTED,
                       linewidth=0.7 if is_p else 1.0)
            ax.annotate(label, (x, y), xytext=off, textcoords="offset points", fontsize=6.0, ha=ha,
                        va="center", color=FG if is_p else MUTED)
        if not ymax:
            ax.text(0.5, 0.5, "pending: the head-to-head at the frozen commit", transform=ax.transAxes,
                    ha="center", va="center", fontsize=6.5, color=ACCENT)
        ax.set_title(title, loc="left", fontweight="semibold", pad=4)
        ax.set_xlabel("Peak GPU memory reserved (GiB)", color=FG)
        ax.set_ylabel("Seconds per update", color=FG)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_xlim(0, 26)
        ax.set_ylim(0, ymax * 1.2 if ymax else 1)
        if cap:
            ax.axvline(cap, color=MUTED, lw=0.6, ls=(0, (3, 2)))
            ax.text(cap, 0.97, " card", transform=ax.get_xaxis_transform(), fontsize=6.0, color=MUTED,
                    va="top", ha="left")
        ax.grid(axis="y", color=RULE, lw=0.4)
        ax.set_axisbelow(True)
    fig.savefig(HERE / "figure1_h2h.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    if complete:
        VALUES["fig1.status"] = ""
        SOURCES["fig1.status"] = []


# ── keys used by the paper, and which are pending ────────────────────────────
def keys_in_paper() -> set[str]:
    keys = set()
    for f in list(HERE.glob("*.tex")):
        if f.name == "values.tex" or f.name.startswith("table_logs"):
            continue
        text = "\n".join(re.sub(r"(?<!\\)%.*", "", ln) for ln in f.read_text().splitlines())
        keys |= set(re.findall(r"\\nvc?\{([^}]*)\}", text))
        keys |= set(re.findall(r"\\pendcell\{([^}]*)\}", text))
    for v in VALUES.values():
        keys |= set(re.findall(r"\\nvc?\{([^}]*)\}", v))
    return {k for k in keys if not k.startswith("#")}  # \\nv{#1} in a macro definition is no key


def expected_source(key: str) -> str:
    best_ = ""
    for pre in EXPECT:
        if key.startswith(pre) and len(pre) > len(best_):
            best_ = pre
    return EXPECT.get(best_, "unknown")


def main() -> int:
    global ROUNDS
    ROUNDS = rounds_4090()
    h2h()
    if H2H is None:
        h2h_runs()
    h2h_previous_unsloth()
    h2h_dense_previous()
    prove()
    family_gate()
    dropout_checks()
    selftests()
    recipes()
    paths()
    mac144()
    studies()
    check_claim()
    # Sections 5.6 and 6: peak host memory was lower with the experts in the SSD store
    if "r30.host" in RAW and "r30ssd.host" in RAW and not RAW["r30ssd.host"] < RAW["r30.host"]:
        PROBLEMS.append(f"moe-30b-ssd's peak host memory {RAW['r30ssd.host']} is not below moe-30b's {RAW['r30.host']}")
    # Scope (Section 6): the Mac's 144-step run and both optimizer studies ran more updates than wl.updates
    if "wl.updates" in RAW:
        for k in ("m144.steps", "olmoe.updates", "q30.updates"):
            if k in RAW and not RAW[k] > RAW["wl.updates"]:
                PROBLEMS.append(f"{k} = {RAW[k]} is not above wl.updates = {RAW['wl.updates']} (the Scope sentence says it ran longer)")
    write_tables()
    figure()

    used = keys_in_paper()
    conditional = sorted(k for k in used if ".ctl." in k or k == "olmoe.supd.new")  # printed only under \nvif{olmoe.ctl.unchanged}
    pending = sorted(k for k in used if k not in VALUES and k not in conditional)
    lines = ["% Generated by launch_results.py from the benchmark logs -- do not edit."]
    if not pending and not PROBLEMS:
        lines.append("\\def\\poroscomplete{}% every value filled and every check passed: no draft header")
    for k in sorted(VALUES):
        lines.append(f"\\nvset{{{k}}}{{{VALUES[k]}}}")
    (HERE / "values.tex").write_text("\n".join(lines) + "\n")
    unused = sorted(k for k in VALUES if k not in used)
    rep = [f"# {len(pending)} pending of {len(used)} keys used by the paper (frozen commit {FROZEN})", ""]
    rep += [f"{k:28s} <- {expected_source(k)}" for k in pending]
    if unused:
        rep += ["", "# keys filled but not used by the paper"] + unused
    if conditional:
        rep += ["", "# conditional keys (the control arm's): the text prints their sentence only if they are filled"]
        rep += [f"{k:28s} {'filled' if k in VALUES else 'not filled: the mechanism sentence prints instead'}" for k in conditional]
    if NOT_USED:
        rep += ["", "# logs at the expected places that name another commit (never read for a value)"] + sorted(set(NOT_USED))
    if PROBLEMS:
        rep += ["", "# problems"] + PROBLEMS
    (HERE / "pending_values.txt").write_text("\n".join(rep) + "\n")
    out = {"frozen_commit": FROZEN, "values": VALUES, "sources": SOURCES, "field": FIELD, "pending": pending,
           "problems": PROBLEMS}
    (HERE / "launch_values.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    print(f"launch_results: {len(VALUES)} values filled, {len(pending)} of {len(used)} keys pending "
          f"(frozen commit {FROZEN[:8]}; logs at {VAL})")
    for p in PROBLEMS:
        print("PROBLEM:", p)
    if FINAL and (pending or PROBLEMS):
        print("--final: refusing, see pending_values.txt beside this script", file=sys.stderr)
        return 1
    if FINAL:  # nothing pending and no problem: the report would only read as unfinished work
        (HERE / "pending_values.txt").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
