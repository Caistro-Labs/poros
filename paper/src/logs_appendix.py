#!/usr/bin/env python3
"""Write table_logs.tex (Appendix A) and ../log_digests.txt.

table_logs.tex: every group of numbers in the paper, where it appears, its log files (paths under
docs/validation/), how it was produced, and its unit.  The rows are built from launch_values.json,
which launch_results.py writes: a key with sources lists them, a key without is shown as missing.

log_digests.txt: the SHA-256 of every cited log file as it is read under $POROS_BENCHMARKS, a checkout
of the benchmark-logs repository (github.com/Caistro-Labs/poros-benchmarks).

Usage:  python3 paper/src/logs_appendix.py   (after launch_results.py; no torch)
"""
import hashlib
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCH = pathlib.Path(os.environ.get("POROS_BENCHMARKS") or ROOT).resolve()
V = "docs/validation/"
NC = "\\emph{no committed log}"

# (title, key prefixes, where in the paper, unit)
GROUPS = [
    ("Machine, versions and settings (RTX 4090)", ("env.", "code.", "wl.", "card."), "\\cref{sec:setup,tab:setup}", "GiB, MiB, s, counts; versions"),
    ("Qwen2.5-72B head-to-head (RTX 4090)", ("dense.", "loss.72b"), "Abstract, \\cref{fig:hero,tab:h2h,sec:dense}", "GiB, s, counts, ratios"),
    ("Qwen3-30B-A3B head-to-head (RTX 4090)", ("moe.", "loss.30b"), "Abstract, \\cref{fig:hero,tab:h2h,sec:moe-eval}", "GiB, s, counts, ratios"),
    ("Unsloth's earlier version on Qwen3-30B-A3B, the same harness on November 18 (that tool's figures only)", ("prev.",), "\\cref{tab:h2h,sec:moe-eval,sec:conclusion}", "s, GiB; version, date"),
    ("Exactness gates (RTX 4090): poros prove, the family gate, the dropout-on checks", ("prove.", "fg.", "do."), "\\cref{tab:exact,sec:testing}", "counts, steps, s, GiB"),
    ("The expert optimizer: self-tests and the OLMoE study", ("st.", "olmoe."), "\\cref{sec:opt-eval}", "s, shares, ratios, loss"),
    ("The certified recipes (RTX 4090, RTX 4060 Laptop, M4)", ("recipe.", "rec.", "r30ssd.", "r30.", "lap.", "paths.", "mac.", "mac72.", "mac32.", "macmoe."), "\\cref{tab:recipes,tab:paths,sec:recipes,sec:laptop,sec:mac}", "GiB, s per update, loss, counts"),
    ("Qwen2.5-72B-Instruct for 144 steps on the Mac", ("m144.",), "\\cref{sec:mac}", "GiB, s, minutes, loss, counts"),
]


def tt(s):
    return (s.replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("&", "\\&")
             .replace("%", "\\%").replace("#", "\\#").replace("{", "\\{").replace("}", "\\}")
             .replace("~", "\\textasciitilde{}").replace("$", "\\$"))


def ttb(s):
    out = tt(s)
    for ch in ("/", "\\_", "-", ".", ","):
        out = out.replace(ch, ch + "\\allowbreak{}")
    return "\\texttt{" + out + "}"


def sha(rel):
    """SHA-256 of a cited log file as read under $POROS_BENCHMARKS (or this repository)."""
    for p in (BENCH / V / rel, BENCH / rel, ROOT / rel):
        if p.exists():
            return hashlib.sha256(p.read_bytes()).hexdigest()
    raise SystemExit(f"missing log {rel}: set POROS_BENCHMARKS to the root of a checkout of "
                     "github.com/Caistro-Labs/poros-benchmarks")


def logs_cell(recs):
    if not recs:
        return NC
    groups = {}
    for r in recs:
        d, _, f = r.rpartition("/")
        groups.setdefault(d, []).append(f)
    return "; ".join(ttb((d + "/") if d else "") + (", ".join(ttb(f) for f in fs) if len(fs) > 1 else ttb(fs[0]))
                     for d, fs in groups.items())


def main():
    lv = json.loads((HERE / "launch_values.json").read_text())
    vals, srcs, missing = lv["values"], lv["sources"], set(lv["pending"])
    out, files = [], []
    for title, prefixes, where, unit in GROUPS:
        keys = sorted(k for k in set(vals) | missing if k.startswith(prefixes))
        out.append(f"\\midrule\\multicolumn{{4}}{{@{{}}p{{14cm}}}}{{\\textbf{{{title}}}}} \\\\*[1pt]")
        filled = [k for k in keys if k in vals and k not in missing]
        recs = []
        for k in filled:
            for s in srcs.get(k, []):
                s = s[len(V):] if s.startswith(V) else s
                if s not in recs:
                    recs.append(s)
        if filled:
            pres = sorted({k.split(".")[0] + ".*" for k in filled})
            shown = [r for r in recs if not r.startswith(("paper/", "scripts/"))]
            per_row = 10  # log files per table row, so a row fits on a page
            for i in range(0, max(len(shown), 1), per_row):
                if i == 0:
                    out.append(f"{len(filled)} values ({', '.join(tt(x) for x in pres)}; each key, its value and its source fields are in {ttb('paper/src/launch_values.json')}) & {where} & {logs_cell(shown[i:i + per_row])} & "
                               f"launch\\_results.py, from the logs [{unit}] \\\\")
                else:
                    out.append(f"\\quad (continued) & & {logs_cell(shown[i:i + per_row])} & \\\\")
            files += [r for r in shown if r not in files]
        miss = [k for k in keys if k in missing]
        if miss:
            out.append(f"{len(miss)} values missing & {where} & \\emph{{no log read}} & "
                       f"see \\texttt{{paper/src/pending\\_values.txt}} \\\\")

    (HERE / "table_logs.tex").write_text(
        "% Generated by logs_appendix.py -- do not edit.\n"
        "\\begingroup\\footnotesize\\setlength{\\tabcolsep}{3pt}\\renewcommand{\\arraystretch}{1.1}\n"
        "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{3.95cm}>{\\raggedright\\arraybackslash}p{1.85cm}"
        ">{\\raggedright\\arraybackslash}p{4.45cm}>{\\raggedright\\arraybackslash}p{3.6cm}@{}}\n"
        "\\caption{Every number in this paper, with its benchmark logs (paths under \\texttt{docs/validation/}), "
        "how it was produced, and its unit.}\\label{tab:logs}\\\\\n"
        "\\toprule \\tabhead{Numbers} & \\tabhead{Where} & \\tabhead{Logs} & \\tabhead{Produced by [unit]} \\\\\n\\endfirsthead\n"
        "\\toprule \\tabhead{Numbers} & \\tabhead{Where} & \\tabhead{Logs} & \\tabhead{Produced by [unit]} \\\\\\midrule\n\\endhead\n"
        "\\endfoot\n\\bottomrule\n\\endlastfoot\n"
        + "\n".join(out) + "\n\\end{longtable}\\endgroup\n")

    lines = ["# SHA-256 of every log file cited in the paper's Appendix A.  Paths are under docs/validation/ in",
             "# github.com/Caistro-Labs/poros-benchmarks.  Generated by paper/src/logs_appendix.py."]
    # poros/... is a file of the main repository (the certified list), not a benchmark log: Appendix A
    # names it, and this list of log digests leaves it out.
    logs = [r for r in files if not r.startswith("poros/")]
    outside = [r for r in logs if r.startswith("$")]
    for r in logs:
        lines.append(f"{'(outside the benchmark-logs repository)':64s}  {r}" if r.startswith("$") else f"{sha(r)}  {r}")
    (HERE.parent / "log_digests.txt").write_text("\n".join(lines) + "\n")
    print(f"logs_appendix: {len(files)} log files cited")
    if outside:
        print(f"logs_appendix: {len(outside)} cited files are outside the benchmark-logs repository", file=sys.stderr)
        if "--final" in sys.argv:
            sys.exit(1)


if __name__ == "__main__":
    main()
