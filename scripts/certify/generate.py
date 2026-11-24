#!/usr/bin/env python3
"""Write poros/recipes/certified.json from the receipts scripts/certify/sources.toml names.

    python scripts/certify/generate.py             # write the shipped file
    python scripts/certify/generate.py --check     # exit 1: the shipped file is not what it writes
    python scripts/certify/generate.py --strict    # also exit 1 while any certification is pending
    python scripts/certify/generate.py --release [--drop KEY ...]
                                    # the release's file: strict, and each built-in named by
                                    # --drop (one with no certified machine) left out of the set

A receipt certifies a recipe only if it was run at the one certification commit sources.toml pins
(every machine's runs at the same commit), on the recipe's model and
loop, at every setting that decides fit and speed exactly as the recipe says (block size, rank,
alpha, dropout, row length, batch, accumulation, path, store and its cache), on full-length rows,
and is neither a preview (simulated) nor a test fixture. Anything else is refused with the reason,
and nothing is written. A receipt the tree does not hold yet leaves that certification pending.

Two receipt layouts are read: a certification run (scripts/certify/run_recipe.py), a folder with
certification.json; and a Mac table row, a folder with receipt.json and commit.txt
(docs/validation/int/mac/...), which a certification run of the same recipe and Mac replaces.
A head-to-head arm is not one: it trains the engine's own loop from a config, not a recipe.
Memory becomes GiB in one place, poros.recipes.certified.to_gib; a Poros receipt's *_gb is GB.
--release also refuses while the package (poros/ and pyproject.toml) differs from the pinned
commit's in anything but the generated data (GENERATED): a later commit of docs, receipts or that
data keeps the pin, and any other change to the package needs every run again. One case is
narrower: a change to a file only a Mac runs, named under [[mac_only]] with the commit the Mac
was certified at again (mac_commit), needs the Mac's runs again and leaves the CUDA runs standing
(MAC_ONLY).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import sys
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "scripts", ROOT / "scripts" / "certify"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from poros.recipes import certified, presets  # noqa: E402

SOURCES = ROOT / "scripts" / "certify" / "sources.toml"
OUT = certified.DATA
# The planner's measured cells from each certified CUDA run, in plan_sources.yaml's pointer form:
# The planner's cell builder reads this file beside the cells' other sources, which stay with
# the receipts they read in the development repository.
PLAN_OUT = ROOT / "docs" / "validation" / "recipes" / "plan_sources.yaml"
# The settings that decide whether a run fits and how fast it runs. A certified recipe says each
# of the first seven; path and store it may leave to their defaults.
# What a recipe may leave to the run, which the run's preflight then decides on each card: the
# entry records what it decided (ran_path, ran_offload), never as the recipe's setting.
LEFT_TO_RUN = ("path", "offload_embeddings_and_head")
DEFAULT_UPDATES = 16  # a certification run's measured updates, unless its sources entry says
SAID = ("max_seq_length", "lora_rank", "lora_alpha", "lora_dropout", "block_size",
        "per_device_batch_size", "gradient_accumulation_steps")
FAST_CACHE = {"cache": "4G"}  # what path fast+cache sets (poros.cli.verbs.shared.path_config)
MEDIAN = "median of updates 2..N"
MEAN = "mean of all updates"
LOOPS = {"SFT": "sft", "DPO": "dpo", "IPO": "ipo", "KTO": "kto", "ORPO": "orpo",
         "SimPO": "simpo"}


class Refused(ValueError):
    """A receipt that cannot certify its recipe, and why."""


def preset_of(r: presets.Recipe) -> Dict[str, Any]:
    """What a recipe's loop sets where it says nothing, on its platform (presets.loop_preset)."""
    return presets.loop_preset(r)


def moe(r: presets.Recipe) -> bool:
    """A recipe of a MoE model: it streams per expert, with no block."""
    from poros.cli.words.models import by_id

    entry = by_id(r.model)
    return entry is not None and entry.entry == "moe"


def expected(r: presets.Recipe) -> Dict[str, Any]:
    """The fit settings a run of ``r`` resolves to: its own words, else its loop's preset on its
    platform (a recipe that says as little as the earliest recipes did); path and store's defaults."""
    preset = preset_of(r)
    missing = [k for k in SAID if k not in r.settings and k not in preset]
    if missing:
        raise Refused(f"{r.key} does not say {', '.join(missing)}")
    out = {k: r.settings.get(k, preset.get(k)) for k in SAID}
    if moe(r) and "block_size" not in r.settings:
        out["block_size"] = None  # poros train clears a preset's block for a MoE model
    path = r.settings.get("path")
    out["path"] = "fast" if path == "fast+cache" else path
    out["store"] = r.settings.get("store", "ram")
    out["store_options"] = dict(FAST_CACHE) if path == "fast+cache" else {}
    out["offload_embeddings_and_head"] = r.settings.get("offload_embeddings_and_head")
    out["moe_expert_cache_gib"] = r.settings.get("moe_expert_cache_gib")
    return out


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise Refused(f"{path.name} could not be read: {exc}") from None


def _rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _fit_of(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """A resolved config's fit settings (poros train's config.resolved.json)."""
    out = {k: cfg.get(k) for k in SAID}
    out["path"] = cfg.get("path")
    out["store"] = cfg.get("store")
    out["store_options"] = dict(cfg.get("store_options") or {})
    out["offload_embeddings_and_head"] = cfg.get("offload_embeddings_and_head")
    out["moe_expert_cache_gib"] = cfg.get("moe_expert_cache_gib")
    return out


def _median_2n(times: List[float]) -> Optional[float]:
    return round(statistics.median(times[1:]), 4) if len(times) >= 2 else None


def _memory(mac: bool, *, used=None, reserved=None, allocated=None, host=None
            ) -> Dict[str, Any]:
    """Each memory figure by its own name (poros.recipes.certified.MEMORY): CUDA's GPU memory
    used (nvidia-smi), torch's reserved and allocated peaks and the host's RAM; a Mac's one."""
    if mac:
        return {"gpu_memory_used_gib": None, "torch_reserved_peak_gib": None,
                "torch_allocated_peak_gib": None, "memory_peak_gib": allocated,
                "host_ram_peak_gib": None}
    return {"gpu_memory_used_gib": used, "torch_reserved_peak_gib": reserved,
            "torch_allocated_peak_gib": allocated, "memory_peak_gib": None,
            "host_ram_peak_gib": host}


def read_run(folder: Path, root: Path) -> Dict[str, Any]:
    """A certification run's folder (scripts/certify/run_recipe.py)."""
    c = _json(folder / "certification.json")
    if c.get("schema") != "poros.certify.run.v1":
        raise Refused("certification.json is not a certification run's")
    if c.get("status") != "ok":  # said before anything else: what stopped it, and where
        failed = c.get("failed")
        first = (failed[0] if isinstance(failed, list) and failed else {})
        if isinstance(failed, str):  # a receipt from before phases were named
            first = {"phase": "unsaid", "reason": failed}
        where = f"in its {first.get('phase', 'unsaid')} phase"
        why = first.get("reason") or "no reason recorded"
        if c.get("status") == "refused before start":
            raise Refused(f"poros train refused it before it started on this machine ({where}): "
                          f"{why}")
        if not (folder / "main" / "events.jsonl").is_file():
            raise Refused(f"the certification run failed {where}: {why}")
    cfg = _json(folder / "main" / "config.resolved.json")
    receipt = _json(folder / "main" / "receipt.json")
    steps = []
    events = folder / "main" / "events.jsonl"
    if not events.is_file():
        raise Refused("it holds no main/events.jsonl: the main run's steps are not there")
    for line in events.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("event_type") == "step" and isinstance(e.get("data", {}).get("s_per_step"),
                                                         (int, float)):
            steps.append(float(e["data"]["s_per_step"]))
    m = c["machine"]
    machine = certified.Machine(m["platform"], m["gpu"], m.get("vram_gib"), m.get("host_ram_gib"),
                                m.get("mac_class") or "", m.get("os_version") or "")
    mac = machine.platform == "mac"
    alloc = certified.to_gib(receipt.get("peak_vram_allocated_gb"), "GB")
    plan = c.get("plan") or {}
    return {
        "model": cfg.get("model_name_or_path"), "loop": cfg.get("trained_with"),
        "data": cfg.get("data_shape"), "machine": machine, "fit": _fit_of(cfg),
        "ran_path": cfg.get("ran_path") or (None if mac else cfg.get("path")),
        "updates": len(steps), "s_per_step": _median_2n(steps), "rule": MEDIAN,
        **_memory(mac, used=c.get("gpu_memory", {}).get("used_gib"),
                  reserved=certified.to_gib(receipt.get("peak_vram_reserved_gb"), "GB"),
                  allocated=alloc, host=c.get("host_ram", {}).get("peak_gib")),
        "planned": {"updates": plan.get("updates"), "resume": plan.get("resume", True),
                    "eval": plan.get("eval", True)},
        "commit": c.get("commit"),
        "simulated": bool(c.get("simulated") or receipt.get("simulated")),
        "fixture": bool(c.get("fixture")), "rows_full": c.get("data", {}).get("rows_full"),
        "rows_data": c.get("data", {}),
        "resume": c.get("resume", {}).get("adapter_equal"),
        "eval": {"not run": None}.get(c.get("eval", {}).get("status"),
                                      c.get("eval", {}).get("status")),
        # what the eval measured: a metric, on rows the run did not train on
        "eval_metric": c.get("eval", {}).get("has_metric") is True,
        "eval_rows_from": c.get("eval", {}).get("rows_from"),
        "adapter_sha256": (c.get("main") or {}).get("adapter_sha256"),
        "tidy_left": list((c.get("tidy") or {}).get("left") or []),
        "receipt": _rel(folder, root), "driver": True,
        # a Mac trains one micro-batch at a time through every layer, and its table settles
        # where the weights stream from: neither is a recipe's setting there
        "unrecorded": ("path", "store", "store_options", "offload_embeddings_and_head",
                       "moe_expert_cache_gib") if mac else (),
        "status": c.get("status"), "tree_clean": c.get("tree_clean"),
    }


def read_mac_row(folder: Path, root: Path) -> Dict[str, Any]:
    """A Mac table row: receipt.json and commit.txt (docs/validation/int/mac/round4/receipts)."""
    from poros.metal.table import memory_of, name_of

    r = _json(folder / "receipt.json")
    commit = (folder / "commit.txt").read_text().strip()
    klass = str(r.get("device_class") or "")
    if r.get("platform") != "darwin" or not memory_of(klass):
        raise Refused("receipt.json is not a Mac run's")
    cfg = {**(r.get("config") or {}), "lora_dropout": r.get("lora_dropout")}
    fit = {k: cfg.get(k) for k in SAID}
    # A Mac trains one micro-batch at a time through every layer (no path), and its table settles
    # where the weights stream from: neither is a setting its receipt records.
    fit.update(path=None, store="ram", store_options={})
    alloc = certified.to_gib(r.get("peak_vram_allocated_gb"), "GB")
    return {
        "model": r.get("model"), "loop": LOOPS.get(str(r.get("recipe"))), "data": None,
        # a table row records the macOS build, not its version
        "machine": certified.Machine("mac", name_of(klass), memory_of(klass) / 2**30, None, klass,
                                     f"macOS build {r['macos_build']}" if r.get("macos_build")
                                     else ""),
        "fit": fit, "ran_path": None, "updates": r.get("steps"),
        "s_per_step": r.get("s_per_step"), "rule": MEAN, **_memory(True, allocated=alloc),
        "commit": commit, "simulated": bool(r.get("simulated")), "fixture": bool(r.get("fixture")),
        "rows_full": None, "resume": None, "eval": None, "receipt": _rel(folder, root),
        "unrecorded": ("path", "store", "store_options", "offload_embeddings_and_head",
                       "moe_expert_cache_gib"),
        "driver": False,
    }


def read(path: Path, root: Path) -> Dict[str, Any]:
    """Whichever of the two layouts ``path`` is."""
    if path.is_dir() and (path / "certification.json").is_file():
        return read_run(path, root)
    if path.is_dir() and (path / "receipt.json").is_file() and (path / "commit.txt").is_file():
        return read_mac_row(path, root)
    raise Refused(f"{path} is no receipt layout this reads")


# What the package may differ from the pinned commit in: the data this generator and the
# planner's cell builder (which the release repository carries) write from the receipts.
GENERATED = ("poros/recipes/certified.json", "poros/plan/cells.json")
PACKAGE = ("poros", "pyproject.toml")


# What a release may never exempt from the pin, whatever sources.toml says: the code a training
# run executes or decides its settings with, and the recipes it trains.
NEVER_EXEMPT = ("poros/core/", "poros/train/", "poros/moe/", "poros/sss/", "poros/metal/",
                "poros/preflight/", "poros/recipes/presets.py")

# What a Mac alone runs: the files, each by its exact path, that were read through and found to
# be reached by no CUDA training run. A change to one of these is never exempt (above), but it
# need not send the CUDA machines' runs round again: sources.toml names the file under
# [[mac_only]] and the commit every Mac certification was made at again (mac_commit). The CUDA
# certifications stand at ``commit``; the Mac's are accepted at ``mac_commit`` alone, and
# --release holds the package to both. Being under poros/metal/ is not enough: CUDA paths import
# parts of it (the layout and the pack's format, for one), so a file joins this list only after
# its own reading, with a test that holds what was found (test_recipe_certification.py, one of the
# release's own tests).
MAC_ONLY = ("poros/metal/source.py",)


def mac_only_files(src: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    """sources.toml's [[mac_only]] entries ({path, reason}): files of MAC_ONLY that differ from
    the pin and that only a Mac run loads, each by its exact path and why no CUDA run executes
    it; and why any entry is refused (no reason, a file MAC_ONLY does not name, or no
    ``mac_commit`` beside it). ``mac_commit`` without an entry is refused too: the two say one
    thing together."""
    out: Dict[str, str] = {}
    refused: List[str] = []
    entries, mac = src.get("mac_only", []), str(src.get("mac_commit") or "")
    if entries and not mac:
        refused.append("mac_only: it names files only a Mac runs, and sources.toml gives no "
                       "mac_commit, the commit the Mac certifications were made at again")
    if mac and not entries:
        refused.append("mac_commit: no [[mac_only]] entry says which files differ from the "
                       "certification commit and why no CUDA run executes them")
    if mac and mac == src.get("commit"):
        refused.append("mac_commit: it is the certification commit itself; leave it out")
    for e in entries:
        path, why = str(e.get("path", "")), str(e.get("reason", "")).strip()
        if not why:
            refused.append(f"mac_only {path or '(no path)'}: it gives no reason")
        elif path not in MAC_ONLY:
            refused.append(f"mac_only {path}: not a file read through and found Mac-only "
                           f"({', '.join(MAC_ONLY)}); a CUDA run may reach it")
        else:
            out[path] = why
    return out, refused


def pin_for(src: Dict[str, Any], platform: str) -> str:
    """The commit a certification on ``platform`` must have run: ``mac_commit`` on a Mac where
    sources.toml sets it (see MAC_ONLY), else ``commit``."""
    return str(src["mac_commit"]) if platform == "mac" and src.get("mac_commit") else src["commit"]


def exempt_files(src: Dict[str, Any]) -> Tuple[Dict[str, str], List[str]]:
    """sources.toml's [[exempt]] entries ({path, reason}): files of the package that may differ
    from the pin, each by its exact path and why a run made at the pin is the same run; and why
    any entry is refused (no reason, not an exact file, or a file NEVER_EXEMPT names). An entry
    with ``strings_only = true`` may name any Python file, NEVER_EXEMPT's included: it passes
    only while the file differs from the pin's in its text alone (strings_only_files,
    package_moved)."""
    out: Dict[str, str] = {}
    refused: List[str] = []
    for e in src.get("exempt", []):
        path, why = str(e.get("path", "")), str(e.get("reason", "")).strip()
        strings = e.get("strings_only", False)
        if not why:
            refused.append(f"exempt {path or '(no path)'}: it gives no reason")
        elif (not (path.startswith("poros/") or path == "pyproject.toml") or path.endswith("/")
              or any(ch in path for ch in "*?[]")):
            refused.append(f"exempt {path}: name one file of poros/ (or pyproject.toml) by its "
                           "exact path")
        elif strings not in (True, False):
            refused.append(f"exempt {path}: strings_only is true or false")
        elif strings and not path.endswith(".py"):
            refused.append(f"exempt {path}: strings only compares Python code; name a .py file")
        elif not strings and any(path == n or path.startswith(n) for n in NEVER_EXEMPT):
            refused.append(f"exempt {path}: a training run executes it, so it is never exempt")
        else:
            out[path] = why
    return out, refused


def strings_only_files(src: Dict[str, Any]) -> set:
    """The [[exempt]] paths marked ``strings_only = true``."""
    return {str(e.get("path", "")) for e in src.get("exempt", []) if e.get("strings_only") is True}


def code_without_text(source: str) -> str:
    """``source``'s syntax tree with its text blanked: each string constant (docstrings and the
    literal parts of f-strings included) made empty, comments gone (the tree has none). What
    decides behaviour stays: an f-string's expressions and format specs, a comparison's
    operands, a subscript's key, a dict's keys and every parameter default. Two versions of a
    file that give the same result differ in their words alone."""
    tree = ast.parse(source)
    keep = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            keep.update(id(n) for n in (node.left, *node.comparators))
        elif isinstance(node, ast.Subscript):
            keep.update(id(n) for n in ast.walk(node.slice))
        elif isinstance(node, ast.Dict):
            keep.update(id(k) for k in node.keys if k is not None)
        elif isinstance(node, ast.arguments):
            for d in (*node.defaults, *node.kw_defaults):
                if d is not None:
                    keep.update(id(n) for n in ast.walk(d))
        elif isinstance(node, ast.FormattedValue) and node.format_spec is not None:
            keep.update(id(n) for n in ast.walk(node.format_spec))

    def text(n: ast.AST) -> bool:
        return isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in keep

    class Blank(ast.NodeTransformer):
        def visit_JoinedStr(self, node: ast.JoinedStr) -> ast.AST:
            node.values = [self.visit(v) for v in node.values if not text(v)]
            return node

        def visit_Constant(self, node: ast.Constant) -> ast.AST:
            return ast.Constant(value="") if text(node) else node

    return ast.dump(Blank().visit(tree))


def package_moved(commit: str, root: Path = ROOT,
                  exempt: Optional[Dict[str, str]] = None,
                  strings_only: Optional[set] = None) -> List[str]:
    """Why the package here is not the pinned ``commit``'s: one line for each file of PACKAGE
    that differs (committed or not, or untracked), other than GENERATED and the files ``exempt``
    names, or one saying why git cannot compare them. [] where it is the commit's. A file
    ``strings_only`` names (and ``exempt`` too) passes only while its code, its text blanked
    (code_without_text), is the commit's."""
    import subprocess

    exempt = exempt or {}
    cannot = [f"this copy cannot be compared with the certification commit {commit[:12]}: it is "
              "not a git checkout that holds that commit (an archive or an export without its "
              "history); run --release in the repository the runs were made from"]

    def git(*args: str) -> Any:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                              timeout=60)

    try:  # a checkout of this folder that holds the pin, or one plain line: never git's own text
        inside = git("rev-parse", "--show-toplevel")
        if (inside.returncode != 0
                or Path(inside.stdout.strip()).resolve() != Path(root).resolve()
                or git("cat-file", "-e", f"{commit}^{{commit}}").returncode != 0):
            return cannot
        done = git("diff", "--name-only", "--no-renames", commit, "--", *PACKAGE)
    except (OSError, subprocess.SubprocessError):
        return cannot
    if done.returncode != 0:
        return cannot
    untracked = subprocess.run(["git", "-C", str(root), "ls-files", "--others",
                                "--exclude-standard", "--", *PACKAGE],
                               capture_output=True, text=True, timeout=60).stdout
    names = set(done.stdout.split()) | set(untracked.split())
    out = []
    for n in sorted(names):
        if n in GENERATED:
            continue
        if n in exempt and n in (strings_only or ()):
            pin = git("show", f"{commit}:{n}")
            here = Path(root) / n
            try:
                same = (pin.returncode == 0 and here.is_file()
                        and code_without_text(pin.stdout) == code_without_text(here.read_text()))
            except SyntaxError:
                same = False
            if not same:
                out.append(f"{n} changes more than its strings since the certification commit "
                           "(strings only): a run may differ, so move the pin and make them again")
            continue
        if n not in exempt:
            out.append(f"{n} differs from the certification commit's: the runs certify that "
                       "commit, so move the pin and make them again")
    return out


def rows_short(r: presets.Recipe, got: Dict[str, Any]) -> Optional[str]:
    """Why a run's rows were not full length, judged from the receipt's own numbers: its rows are
    the driver's sample rows for this recipe, seeded by its key or (before that) not (data.sha256:
    each runs past the length by construction), and its lowest update counted at least
    (length - ROW_ALLOWANCE) x rows an update (the count leaves out a row's prompt and template).
    A receipt without those numbers (a fixture) keeps its stored rows_full. None: full."""
    import run_recipe as driver

    d = got.get("rows_data") or {}
    tokens, seq = d.get("tokens_per_update"), d.get("seq_len")
    if not tokens or seq is None:
        return "its rows were not full length" if got.get("rows_full") is False else None
    if all(d.get(k) is not None for k in ("kind", "rows", "sha256")):
        def sha(key: str) -> str:
            rows = driver.sample_rows(d["kind"], int(d["rows"]), int(seq), key=key)
            return hashlib.sha256("".join(json.dumps(x) + "\n" for x in rows).encode()).hexdigest()

        # the driver seeds each recipe's rows with its key; before that, every recipe drew the same
        if d["sha256"] not in (sha(r.key), sha("")):
            return "its rows are not the driver's sample rows for this recipe (data.sha256)"
    st = {**presets.loop_preset(r), **r.settings}
    per = int(st.get("per_device_batch_size", 1)) * int(st.get("gradient_accumulation_steps", 1))
    need = (int(seq) - driver.ROW_ALLOWANCE) * per
    if min(tokens) < need:
        return (f"its rows were not full length: an update counted {min(tokens)} tokens, under "
                f"(length {seq} - {driver.ROW_ALLOWANCE}) x {per} rows = {need}")
    return None


def check(r: presets.Recipe, got: Dict[str, Any], commit: str, fixtures: bool,
          cert: Optional[Dict[str, Any]] = None) -> None:
    """Raise Refused unless ``got`` certifies ``r``; ``cert``, its sources.toml entry, says how
    many updates a certification run measures and which of its phases it may skip."""
    why: List[str] = []
    cert = cert or {}
    if got.get("driver"):
        planned, want = got["planned"], cert.get("updates", DEFAULT_UPDATES)
        if planned["updates"] != want or got["updates"] != want:
            why.append(f"it measured {got['updates']} updates, its sources entry {want}")
        for phase in ("resume", "eval"):
            if planned[phase] is False and cert.get(phase, True):
                why.append(f"it skipped its {phase}, which its sources entry asks for")
    if got["commit"] != commit:
        why.append(f"it ran commit {str(got['commit'])[:12]}, not the release's {commit[:12]}")
    if got["simulated"]:
        why.append("it is a preview (simulated): nothing was measured")
    if got["fixture"] and not fixtures:
        why.append("it is a test fixture")
    if got["model"] != r.model:
        why.append(f"it trained {got['model']}, the recipe {r.model}")
    if got["loop"] != r.loop:
        why.append(f"it trained the {got['loop']} loop, the recipe {r.loop}")
    if got["data"] is not None and not r.takes(got["data"]):
        why.append(f"it trained on {got['data']}, which the recipe does not take")
    want = expected(r)
    for k, v in want.items():
        if k in got["unrecorded"]:
            if v not in (None, "ram", {}, False):
                why.append(f"{k} is not recorded there, and the recipe sets it")
            continue
        ran = got["fit"].get(k)
        if k == "path" and v is None:  # left to the run: it decides efficient, or sequential
            if ran not in (None, "efficient", "sequential"):  # where block-major cannot follow
                why.append(f"path ran {ran!r}, which the recipe leaves to the run")
            continue
        if k == "offload_embeddings_and_head" and v is None:  # the run's: recorded as ran_offload
            continue
        if ran != v:
            why.append(f"{k} ran {ran!r}, the recipe says {v!r}")
    if got.get("status") not in (None, "ok"):
        why.append("the certification run did not pass")
    if got.get("tree_clean") is False:
        why.append("it ran a checkout with changed files")
    short = rows_short(r, got)
    if short:
        why.append(short)
    if got.get("driver") and got.get("planned", {}).get("resume", True):
        if got["resume"] is False:
            why.append("its stopped and continued run did not end at the same adapter")
        elif got["resume"] is None:
            why.append("its stopped and continued run's adapter was not compared")
    if got["eval"] not in (None, "ok"):
        why.append("its poros eval did not pass")
    elif got["eval"] == "ok" and got.get("driver"):
        if not got.get("eval_metric"):
            why.append("its poros eval passed with no metric (a loss before and after)")
        if got.get("eval_rows_from") not in ("held_out", "file", "eval_file"):
            why.append(f"its poros eval scored {got.get('eval_rows_from') or 'unsaid'} rows, "
                       "not rows the run did not train on")
    if got.get("driver") and got["simulated"] is False and not got.get("adapter_sha256"):
        why.append("it records no adapter of its main run")
    for k in ("s_per_step", "updates"):
        if not isinstance(got[k], (int, float)) or got[k] <= 0:
            why.append(f"it records no {k}")
    if got["machine"].platform == "cuda":
        if got["host_ram_peak_gib"] is None or got["machine"].host_ram_gib is None:
            why.append("it records no host RAM")
        if got["gpu_memory_used_gib"] is None:
            why.append("it records no GPU memory used (nvidia-smi)")
        if got["torch_reserved_peak_gib"] is None:
            why.append("it records no torch reserved peak")
    elif got["memory_peak_gib"] is None:
        why.append("it records no peak memory")
    if why:
        raise Refused("; ".join(why))


def entry(r: presets.Recipe, got: Dict[str, Any]) -> Dict[str, Any]:
    m = got["machine"]
    return {
        "machine": {"platform": m.platform, "gpu": m.gpu, "vram_gib": m.vram_gib,
                    "host_ram_gib": m.host_ram_gib, "mac_class": m.mac_class or None,
                    "os": m.os or None},
        # as the run resolved them: what its receipt records, which check() held to the recipe
        # (a path the recipe leaves to the run is ran_path's, not a setting)
        "settings": {k: got["fit"][k] for k, v in expected(r).items()
                     if k not in got["unrecorded"] and not (v is None and k in LEFT_TO_RUN)
                     and got["fit"][k] not in (None, {})},
        "loop": got["loop"], "data": got["data"], "ran_path": got["ran_path"],
        "ran_offload": None if got["machine"].platform == "mac"
        else bool(got["fit"].get("offload_embeddings_and_head")),
        "updates": int(got["updates"]), "s_per_step": got["s_per_step"],
        "s_per_step_rule": got["rule"], **{k: got[k] for k in certified.MEMORY},
        "receipt": got["receipt"],
        "commit": got["commit"],
    }


def build(sources: Path = SOURCES, root: Path = ROOT, fixtures: bool = False,
          drop: Tuple[str, ...] = ()) -> Tuple[Dict[str, Any], List[str]]:
    """The file's content, and every refusal (an empty list: it may be written). Each sources
    entry is one recipe on one machine, its receipts in order of preference: the first in the
    tree that certifies is used (a certification run before a Mac table row); none in the tree,
    and that certification is pending. ``drop``: built-ins the release ships without, each of
    which must have no certified machine."""
    src = tomllib.loads(sources.read_text())
    commit, release = src["commit"], src["release"]
    built = {r.key: r for r in presets.DEFINED}
    recipes: Dict[str, List[Dict[str, Any]]] = {}
    pending: Dict[str, List[str]] = {}
    refused: List[str] = list(mac_only_files(src)[1])
    adapters: Dict[str, str] = {}  # each main run's adapter sha256: whose it is
    for cert in src.get("cert", []):
        key = cert["recipe"]
        if key not in built:
            refused.append(f"{key}: no built-in recipe by that name")
            continue
        found = [w for w in cert["receipts"] if (root / w).exists()]
        if not found:
            pending.setdefault(key, []).append(cert["receipts"][0])
            continue
        where = found[0]
        try:
            got = read(root / where, root)
            check(built[key], got, pin_for(src, got["machine"].platform), fixtures, cert)
        except (Refused, KeyError, TypeError, ValueError, OSError) as exc:  # one line, no trace
            refused.append(f"{key} ({where}): {exc}")
            continue
        twin = adapters.get(got.get("adapter_sha256") or "")
        if twin:  # two certifications on the same bytes: one of them measured nothing of its own
            refused.append(f"{key} ({where}): its main run's adapter is {twin}'s, byte for "
                           "byte: two entries never certify on the same training")
            continue
        if got.get("adapter_sha256"):
            adapters[got["adapter_sha256"]] = f"{key} ({where})"
        if got.get("tidy_left"):  # accepted: said again, so the leftovers are removed by hand
            print(f"note: {key} ({where}): its run could not remove {len(got['tidy_left'])} "
                  f"of what it made: {'; '.join(got['tidy_left'][:4])}", file=sys.stderr)
        recipes.setdefault(key, []).append(entry(built[key], got))
    for key in built:  # a built-in no source names is pending too: it ships certified or not at all
        if key not in recipes and key not in pending:
            pending[key] = []
    for key in drop:
        if key not in built:
            refused.append(f"--drop {key}: no built-in recipe by that name")
        elif key in recipes:
            refused.append(f"--drop {key}: it is certified on a machine, so it ships")
        else:
            pending.pop(key, None)
    return {"schema": certified.SCHEMA, "release": release, "commit": commit,
            "certifier": certified.CERTIFIER,
            "generated_by": "scripts/certify/generate.py from scripts/certify/sources.toml",
            "recipes": recipes, "pending": pending, "dropped": sorted(drop)}, refused


def plan_source(r: presets.Recipe, where: str, root: Path = ROOT) -> Dict[str, Any]:
    """One planner cell for a certified CUDA run, in the form the planner cells' sources take:
    every key and figure a pointer into the run's own files, except where none of them
    records the value in the planner's words (each such literal says where it comes from)."""
    from poros.cli.words.models import by_id

    folder = root / where
    cfg = _json(folder / "main" / "config.resolved.json")
    entry = by_id(r.model).entry
    rc, rr, cj = (f"{where}/main/config.resolved.json", f"{where}/main/receipt.json",
                  f"{where}/certification.json")

    def at(ptr: str, receipt: str = rc) -> Dict[str, str]:
        return {"ptr": ptr, "receipt": receipt}

    dense = entry == "dense"
    key: Dict[str, Any] = {
        "model": at("/model_name_or_path"), "quantization": at("/quantization"),
        "seq_len": at("/max_seq_length"),
        # poros.moe streams per expert: block_size applies to dense models only
        "block_size": at("/block_size") if dense else None,
        "batch": at("/per_device_batch_size"), "grad_accum": at("/gradient_accumulation_steps"),
        "lora_rank": at("/lora_rank"), "lora_dropout": at("/lora_dropout"),
        # nf4_load_strategy auto loads block-streamed on CUDA (poros.plan.receipts.Request.of)
        "load_strategy": ("block_streamed" if cfg.get("nf4_load_strategy") == "auto"
                          else at("/nf4_load_strategy")) if dense else None,
        # startup_parity_selfcheck off (config.resolved.json): the self-check did not run
        "selfcheck": cfg.get("startup_parity_selfcheck") == "always",
        "entry": entry, "gpu": at("/gpu/name", rr), "path": at("/path"),
        "offload": at("/offload_embeddings_and_head"),
    }
    if dense:
        key["fused_compute"] = at("/ran_compute/fused_compute")
        if cfg.get("path") != "sequential":
            key["dequant_cache"] = at("/nf4_dequant_cache_mb")
    else:
        key["moe_compute"] = at("/ran_compute/moe_expert_compute")
        key["expert_cache"] = at("/moe_expert_cache_gib")
    fields = {
        "peak_vram_allocated": {"ptr": "/peak_vram_allocated_gb", "unit": "GB"},
        "peak_vram_reserved": {"ptr": "/peak_vram_reserved_gb", "unit": "GB"},
        "peak_host_ram": {"ptr": "/peak_host_ram_gb", "unit": "GB"},
        "s_per_step": {"ptr": "/main/s_per_step", "unit": "s", "receipt": cj,
                       "method": "the median of updates 2..N (the run's events.jsonl)"},
    }
    if not dense and cfg.get("path") == "fast" and cfg.get("moe_expert_cache_gib") is None:
        # the expert cache grew to what this card left (poros.plan.receipts._grows_to_card): the
        # VRAM the process could hold when it was sized, and the size it reached
        fields["vram_budget"] = {"ptr": "/ran_compute/moe_expert_cache_reason", "unit": "GiB",
                                 "receipt": rc, "match": "([0-9.]+) GiB this process may hold"}
        fields["expert_cache_size"] = {"ptr": "/ran_compute/moe_expert_cache_gib",
                                       "unit": "GiB", "receipt": rc}
    return {
        "id": f"recipe-{r.key}-{where.rstrip('/').split('/')[-2]}",
        # a certification run answers its own card, and no other machine's plan, twin or words
        "only_its_machine": True,
        "receipt": rr, "key": key,
        "host": {"harness": "scripts/certify/run_recipe.py (poros train, the recipe as shipped)",
                 "dataset": f"sample {cfg.get('data_shape')} rows longer than the row length"},
        "fields": fields,
    }


def cuda_runs(book: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """(recipe, entry) of every certified CUDA run in ``book``: the runs that have a cell."""
    return [(key, e) for key, found in book["recipes"].items() for e in found
            if e["machine"]["platform"] == "cuda"]


def cuda_groups(book: Dict[str, Any]) -> List[List[Tuple[str, Dict[str, Any]]]]:
    """The certified CUDA runs, grouped by what a cell's key says: the model, the settings, the
    card, the path and the tables. A key names no loop, so support, voice and prefer on one card
    (one model, one length) are one group."""
    built = {r.key: r for r in presets.DEFINED}
    groups: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for key, e in cuda_runs(book):
        same = json.dumps([built[key].model, e["settings"], e["machine"]["gpu"],
                           e["machine"]["vram_gib"], e.get("ran_path"), e.get("ran_offload")],
                          sort_keys=True)
        groups.setdefault(same, []).append((key, e))
    return list(groups.values())


# Each memory figure of a CUDA cell and the entry figure that ranks its runs.
_CUDA_PEAKS = {"peak_vram_allocated": "torch_allocated_peak_gib",
               "peak_vram_reserved": "torch_reserved_peak_gib",
               "peak_host_ram": "host_ram_peak_gib"}


def cuda_sources(book: Dict[str, Any], root: Path = ROOT) -> List[Dict[str, Any]]:
    """One cell per key (cuda_groups). Where runs share it, each memory figure is the run's that
    peaked highest on it, a bound for each, and where their loops differ the cell gives no time:
    an SFT run's step is not a DPO run's. Each run's own figures stay in its certified entry."""
    built = {r.key: r for r in presets.DEFINED}
    out = []
    for runs in cuda_groups(book):
        def top(field: str) -> Tuple[str, Dict[str, Any]]:
            return max(runs, key=lambda x: float(x[1].get(field) or 0))

        key, e = top("torch_reserved_peak_gib")
        src = plan_source(built[key], e["receipt"], root)
        for name, field in _CUDA_PEAKS.items():
            if name in src["fields"]:
                src["fields"][name]["receipt"] = f"{top(field)[1]['receipt']}/main/receipt.json"
        if len({built[k].loop for k, _ in runs}) > 1:
            del src["fields"]["s_per_step"]
        if len(runs) > 1:
            src["host"]["recipes"] = ", ".join(sorted(k for k, _ in runs))
        out.append(src)
    return out


# The Mac class the table measures (poros.metal.DEVICE_CLASSES) and the Macs its rows are planned
# for: what the Mac gate compares a run without a recipe on.
MAC_CLASS = "apple-m4-10gpu-16gb"
MAC_CLASSES = ("apple-m4-10gpu-8gb", MAC_CLASS, "apple-m4-10gpu-24gb", "apple-m4-10gpu-64gb")


def mac_config(r: presets.Recipe, klass: str = MAC_CLASS) -> Any:
    """The PorosConfig a run of ``r`` resolves to on a Mac of ``klass``, before it starts: its
    settings, its loop's where it says none, and the Mac table row's where neither does."""
    from poros.core.config import PorosConfig
    from poros.metal import table

    st = {k: v for k, v in {**presets.loop_preset(r), **r.settings}.items()
          if k not in ("max_steps", "num_epochs")}
    cfg = PorosConfig(model_name_or_path=r.model, dataset="synthetic", **st)
    table.fill(cfg, klass)
    return cfg


def mac_plan(cfg: Any, klass: str = MAC_CLASS) -> Any:
    """What `poros fit` and the start of `poros train` read on a Mac of ``klass``: its plan."""
    from poros.cli.words.models import by_id
    from poros.metal import plan, table

    e = by_id(cfg.model_name_or_path)
    arch = e.arch if e else None
    return plan.unified_plan(cfg, model_type=arch, gpu=table.plan_gpu(klass), download=None,
                             host_free_gib=None)


def _install(cells: List[dict]) -> None:
    """``cells`` as the plan's CUDA receipts, split as poros.plan.cells splits cells.json: the
    certification runs' (only_its_machine) apart, read only on their own card."""
    from poros.plan import cells as plan_cells

    plan_cells.CELLS = [c for c in cells if not c.get("only_its_machine")]
    plan_cells.CERTIFIED = [c for c in cells if c.get("only_its_machine")]


def _hand_kept(cells: List[dict]) -> List[dict]:
    """The cells nobody's recipe run wrote: every one but this generator's (``recipe-``)."""
    return [c for c in cells if not c["id"].startswith("recipe-")]


def mac_runs(book: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """(recipe, entry) of every certified Mac run whose model and settings the Mac's own runs
    (the table's hand-kept cells) do not measure: those get a cell of their own."""
    import tempfile

    from poros.cli.context import Context
    from poros.cli.verbs import shared
    from poros.metal import table
    from poros.plan import cells as plan_cells

    built = {r.key: r for r in presets.DEFINED}
    was = plan_cells.MAC_CELLS
    out = []
    try:
        plan_cells.MAC_CELLS = _hand_kept(was)
        with tempfile.TemporaryDirectory() as tmp:
            ctx = Context(cwd=Path(tmp), fixed_gpu=table.plan_gpu(MAC_CLASS),
                          fixed_interactive=False)
            for key, found in book["recipes"].items():
                for e in found:
                    if e["machine"]["platform"] != "mac" or e["machine"]["mac_class"] != MAC_CLASS:
                        continue
                    st = e["settings"]  # as the run resolved them: what `poros fit` is asked
                    p = shared.plan_for(built[key].model, ctx, seq_len=st["max_seq_length"],
                                        rank=int(st["lora_rank"]), dropout=st["lora_dropout"])
                    if p.peak_vram_allocated.source != "measured":
                        out.append((key, e))
    finally:
        plan_cells.MAC_CELLS = was
    return out


def mac_source(r: presets.Recipe, where: str, updates: int, tier: str) -> Dict[str, Any]:
    """One Mac table cell for a certified Mac run, as the Mac's own cells are written
    (docs/validation/metal/m4_16gb/plan_sources.yaml): every value a pointer into the run's
    receipt, its events or its certification, the load strategy the receipt's tier."""
    from poros.cli.words.models import by_id

    rr, ev, cj = (f"{where}/main/receipt.json", f"{where}/main/events.jsonl",
                  f"{where}/certification.json")
    return {
        "id": f"recipe-{r.key}-{where.rstrip('/').split('/')[-2]}",
        "receipt": rr,
        # a certification run measures its own Mac: the table sizes no other Mac's rows from it
        "only_its_machine": True,
        "key": {
            "model": {"ptr": "/model"}, "quantization": {"ptr": "/config/quantization"},
            "seq_len": {"ptr": "/config/max_seq_length"},
            "block_size": {"ptr": "/config/block_size"},
            "batch": {"ptr": "/config/per_device_batch_size"},
            "grad_accum": {"ptr": "/config/gradient_accumulation_steps"},
            "lora_rank": {"ptr": "/config/lora_rank"}, "lora_dropout": {"ptr": "/lora_dropout"},
            # the receipt's /tier, in the planner's words
            "load_strategy": f"metal_{tier}", "selfcheck": False,
            "entry": by_id(r.model).entry, "gpu": {"ptr": "/gpu/name"},
        },
        "host": {"harness": "scripts/certify/run_recipe.py (poros train, the recipe as shipped)",
                 "dataset": "sample rows longer than the row length"},
        "fields": {
            "peak_vram_allocated": {"ptr": "/peak_vram_allocated_gb", "unit": "GB",
                                    "method": f"{updates} steps"},
            "gpu_visible": {"ptr": "/data/vram_reserved_mb", "unit": "MB", "receipt": ev,
                            "select": {"event_type": "step", "step": updates - 1},
                            "method": f"{updates} steps"},
            "s_per_step": {"ptr": "/main/s_per_step", "unit": "s", "receipt": cj,
                           "method": f"the median of updates 2..N of {updates} steps"},
        },
    }


def mac_sources(book: Dict[str, Any], root: Path = ROOT) -> List[Dict[str, Any]]:
    """One cell per key the plan matches a request on. A cell's key says no loop, so where runs
    of several recipes share one (support, voice and prefer: one model, one length), the cell is
    the run that peaked highest, a bound for each of them, and where their loops differ it gives
    no time: an SFT run's step is not a DPO run's."""
    built = {r.key: r for r in presets.DEFINED}
    groups: Dict[str, List[Tuple[str, Dict[str, Any], Dict[str, Any]]]] = {}
    for key, e in mac_runs(book):
        receipt = _json(root / e["receipt"] / "main" / "receipt.json")
        same = json.dumps([built[key].model, e["settings"], receipt.get("tier")], sort_keys=True)
        groups.setdefault(same, []).append((key, e, receipt))
    out = []
    for runs in groups.values():
        key, e, receipt = max(runs, key=lambda x: float(x[2].get("peak_vram_allocated_gb") or 0))
        src = mac_source(built[key], e["receipt"], e["updates"], receipt.get("tier"))
        if len({built[k].loop for k, _, _ in runs}) > 1:
            del src["fields"]["s_per_step"]
        src["host"]["recipes"] = ", ".join(sorted(k for k, _, _ in runs))
        out.append(src)
    return out


def plan_sources(book: Dict[str, Any], root: Path = ROOT) -> List[Dict[str, Any]]:
    """A cell for every certified CUDA run in ``book``, and one for every certified Mac run whose
    model and settings the Mac table's own runs do not measure."""
    return cuda_sources(book, root) + mac_sources(book, root)


def plan_dumps(cells: List[Dict[str, Any]]) -> str:
    import yaml

    head = ("# The planner's cells from the certified recipe runs: written by\n"
            "# scripts/certify/generate.py, read by the planner's cell builder. Do not edit.\n")
    return head + (yaml.safe_dump(cells, sort_keys=False, width=100) if cells else "[]\n")


# What a new cell may not change, unaccepted, for a run without a recipe at its settings: the
# preflight's decisions (scripts/certify/decisions.py), not the plan's figures.
DECIDED = ("path", "ran_path", "offload", "vram_refusal", "host_refusal", "length_kept_of_2048")


# Other cards a cell also speaks for (it bounds them: poros.plan.receipts.bound): printed with
# every gate's run, never gating, so a change there is seen before the data lands.
OTHER_CARDS = (("NVIDIA GeForce RTX 4080:15.99", 62.5), ("NVIDIA GeForce RTX 4090:23.99", 32.0))


BUILDER_ABSENT = ("the planner-decision check needs the planner's cell builder, which the "
                  "release repository carries and this tree does not")


def has_builder() -> bool:
    """Whether the planner's cell builder is importable here: the release repository has it, the
    public tree does not."""
    import importlib.util

    return importlib.util.find_spec("build_plan_cells") is not None


def decision_changes(book: Dict[str, Any], root: Path = ROOT,
                     accepted: Tuple[str, ...] = (),
                     notes: Optional[List[str]] = None) -> List[str]:
    """Each decision the new cells change for a run without a recipe on their machine, as a
    refusal line, unless its cell is in ``accepted``: path (decide_path_to_fit), tables
    (decide_offload_for_run), refusals (vram_error, host RAM) and the row length kept
    (fitted_length), before the cells and after, on the certified machine's probes. ``notes``
    collects the same decisions on OTHER_CARDS, before and after, for the output (no gate)."""
    sources = cuda_sources(book, root)
    if not sources:
        return []
    try:
        import build_plan_cells as b
    except ImportError:
        return [BUILDER_ABSENT]
    import tempfile

    import decisions as d

    import poros.core.compute as compute
    from poros.plan import cells as plan_cells

    built = {r.key: r for r in presets.DEFINED}
    out: List[str] = []
    was = (b.SOURCES, b.ROOT, plan_cells.CELLS, certified.host_ram_total_gib,
           compute.triton_installed, plan_cells.CERTIFIED)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "plan_sources.yaml"
        path.write_text(plan_dumps(sources))
        data = Path(tmp) / "rows.jsonl"

        def decided(r: presets.Recipe, gpu: str, ram: float, cells: List[dict]) -> Dict[str, Any]:
            _install(cells)
            ctx = d.machine(gpu, ram, ram * 0.9, Path(tmp))
            g = d.run_recipe.gate_of(r, ctx, str(data), d.unset_path(r, str(data)))
            # the decisions, not their words: a reason that cites a measurement where it cited
            # an estimate decides the same
            return {"path": (g["path"] or [None])[0], "ran_path": g["ran_path"],
                    "offload": g["offload"] is not None, "vram_refusal": bool(g["vram_refusal"]),
                    "host_refusal": bool(g["host_refusal"]),
                    "length_kept_of_2048": d.length_kept(r, ctx)}

        try:
            b.SOURCES, b.ROOT = (path,), root
            new = [c for c in b.build() if not c["key"]["gpu"].startswith("Apple")]
            old = _hand_kept(was[2] + was[5])
            for src, (recipe, e) in zip(sources, (max(g, key=lambda x: float(
                    x[1].get("torch_reserved_peak_gib") or 0)) for g in cuda_groups(book))):
                r, m = built[recipe], e["machine"]
                own = f"{m['gpu']}:{m['vram_gib']}"
                runs = {when: decided(r, own, m["host_ram_gib"], cells)
                        for when, cells in (("before", old), ("after", old + new))}
                moved = [k for k in DECIDED if runs["before"].get(k) != runs["after"].get(k)]
                if moved and src["id"] not in accepted:
                    out.append(f"{src['id']}: its cell changes {', '.join(moved)} for a run "
                               f"without a recipe on its {m['gpu']} (before {runs['before']}; "
                               f"after {runs['after']}): name it in sources.toml's "
                               "accept_decision, with why, once a real run has confirmed it")
                for gpu, ram in OTHER_CARDS if notes is not None else ():
                    before, after = (decided(r, gpu, ram, old), decided(r, gpu, ram, old + new))
                    notes.append(f"{src['id']} on {gpu.rpartition(':')[0]}, {ram:g} GiB RAM, a "
                                 f"run without a recipe: {'same' if before == after else 'CHANGED'}"
                                 f"; before {before}; after {after}")
        finally:
            (b.SOURCES, b.ROOT, plan_cells.CELLS, certified.host_ram_total_gib,
             compute.triton_installed, plan_cells.CERTIFIED) = was
    return out


# The words a plan says whose run its figures are (poros.cli.verbs.shared.plan_for): fit words.
SAME_SHAPE = "memory and speed are "
# What the Mac table decides that its fit words may not move: everything but "fit".
MAC_SETTINGS = ("max_seq_length", "store", "gradient_accumulation_steps", "block_size",
                "eval_split", "eval_rows", "save_steps")


def _mac_snapshot(model: str, klass: str, own: List[presets.Recipe]) -> Dict[str, Any]:
    """What the Mac table decides for ``model`` on a Mac of ``klass``, for a run without a recipe
    (left to the table, and at each of its lengths) and for each Mac built-in of the model: its
    rows (length, memory or drive, accumulation, saves, which is the default, its evidence), the
    refusal and evidence `poros models` and `poros check` say, the settings a run takes, and what
    `poros fit` and the start of `poros train` say, by the path they take
    (poros.cli.verbs.shared.plan_for: a same-shape model's own cells, else its base's). ``fit``
    apart: the only part a measured cell may move."""
    import tempfile

    from poros.cli.context import Context
    from poros.cli.verbs import shared
    from poros.core.config import PorosConfig
    from poros.metal import table

    rows = [(r.seq, r.tier, r.grad_accum, r.save_steps, r.default, r.evidence, r.planned_from)
            for r in table.rows(model, klass)]
    out: Dict[str, Any] = {"rows": rows, "refusal": table.refusal(model, klass),
                           "evidence": table.evidence(model, klass), "runs": {}}
    asks: List[Tuple[str, Dict[str, Any]]] = [("no recipe", {})]
    asks += [(f"no recipe at {n}", {"max_seq_length": n}) for n in table.LENGTHS]
    asks += [(r.key, {k: v for k, v in r.settings.items() if k != "max_steps"}) for r in own]
    with tempfile.TemporaryDirectory() as tmp:
        ctx = Context(cwd=Path(tmp), fixed_gpu=table.plan_gpu(klass), fixed_interactive=False)
        for name, said in asks:
            cfg = PorosConfig(model_name_or_path=model, dataset="synthetic",
                              **{k: v for k, v in said.items() if k != "num_epochs"})
            table.fill(cfg, klass)  # the settings the run takes
            p = shared.plan_for(model, ctx, seq_len=said.get("max_seq_length"),
                                rank=int(said.get("lora_rank", 16)),
                                dropout=said.get("lora_dropout"))
            out["runs"][name] = {
                "settings": {k: getattr(cfg, k, None) for k in MAC_SETTINGS},
                "support": p.support, "warnings": list(p.warnings),
                "fit": {"fits": p.fits, "basis": p.fit_basis,
                        "peak": (p.peak_vram_allocated.value, p.peak_vram_allocated.source),
                        "on_gpu": (p.gpu_visible.value, p.gpu_visible.source),
                        "s_per_step": (p.s_per_step.value, p.s_per_step.source),
                        "total": (p.total_time.value, p.total_time.source)}}
    return out


def mac_decision_changes(book: Dict[str, Any], root: Path = ROOT,
                         accepted: Tuple[str, ...] = (),
                         notes: Optional[List[str]] = None) -> List[str]:
    """The Mac table's counterpart of decision_changes: before the new Mac cells and after, on an
    8, 16, 24 and 64 GB M4, for every model the table trains and each Mac built-in, what the
    table decides (_mac_snapshot). A change is a refusal line unless it is only the fit words of
    a newly measured run on the class it was measured on, or its cell is in ``accepted``.
    ``notes`` collects every change, fit words included."""
    sources = mac_sources(book, root)
    if not sources:
        return []
    try:
        import build_plan_cells as b
    except ImportError:
        return [BUILDER_ABSENT]
    import tempfile

    from poros.cli.words.models import catalog
    from poros.metal import table
    from poros.plan import cells as plan_cells

    was = (b.SOURCES, b.ROOT, plan_cells.MAC_CELLS)
    out: List[str] = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan_sources.yaml"
            path.write_text(plan_dumps(sources))
            b.SOURCES, b.ROOT = (path,), root
            new = [c for c in b.build() if c["key"]["gpu"].startswith("Apple")]
        old = _hand_kept(was[2])
        measured = {(c["key"]["model"], c["key"]["seq_len"]): c["id"] for c in new}
        models = list(dict.fromkeys(
            [e.id for e in catalog() if table._own_rows(e.id, MAC_CLASS)]
            + [m for m, _ in measured]))
        mac_recipes = [r for r in presets.DEFINED if r.use == "mac"]
        for model in models:
            own = [r for r in mac_recipes if r.model == model]
            for klass in MAC_CLASSES:
                snap = {}
                for when, cells in (("before", old), ("after", old + new)):
                    plan_cells.MAC_CELLS = cells
                    snap[when] = _mac_snapshot(model, klass, own)
                before, after = snap["before"], snap["after"]
                for part in ("rows", "refusal", "evidence"):
                    if before[part] != after[part]:
                        out.append(f"{model} on {klass}: the Mac cells change its {part} "
                                   f"(before {before[part]}; after {after[part]})")
                for name, run in after["runs"].items():
                    was_run = before["runs"][name]
                    for part in ("settings", "support", "warnings", "fit"):
                        if was_run[part] == run[part]:
                            continue
                        cell = measured.get((model, run["settings"]["max_seq_length"]))
                        # whose figures they are ("memory and speed are <base>'s") is fit words
                        provenance = part == "warnings" and all(
                            w.startswith(SAME_SHAPE) for w in
                            set(was_run[part]) ^ set(run[part]))
                        fit_only = ((part == "fit" or provenance) and klass == MAC_CLASS
                                    and cell is not None)
                        if notes is not None:
                            how = "the fit words of a newly measured run" if fit_only else "CHANGED"
                            notes.append(f"{model} on {klass}, {name}: {part} ({how}); "
                                         f"before {was_run[part]}; after {run[part]}")
                        if not fit_only and not (cell and cell in accepted):
                            out.append(f"{model} on {klass}, {name}: the Mac cells change its "
                                       f"{part} (before {was_run[part]}; after {run[part]}): "
                                       "only the fit of a newly measured run may move")
    finally:
        b.SOURCES, b.ROOT, plan_cells.MAC_CELLS = was
    return out


# The cards a CUDA certification cell must tell nothing new: every one but its own.
REACH = ("NVIDIA GeForce RTX 4090:23.99", "NVIDIA RTX A4000:15.99", "NVIDIA GeForce RTX 3070:8.0",
         "NVIDIA GeForce RTX 4060 Laptop GPU:7.68")


def _cuda_told(cells: List[dict], asks: List[Tuple[str, Optional[int], int]],
               cards: List[str], tmp: Path) -> Dict[str, Any]:
    """What the planner tells every card in ``cards``, with ``cells`` as the receipt set: each
    model's twin, the model list's words, and every ask's plan (poros fit and a train's start)."""
    import poros.core.compute as compute
    from poros.cli.context import Context
    from poros.cli.verbs import models as models_verb
    from poros.cli.verbs import shared
    from poros.cli.words.models import catalog
    from poros.core.schemas import PlanGpu
    from poros.plan import cells as plan_cells

    was, triton = (plan_cells.CELLS, plan_cells.CERTIFIED), compute.triton_installed
    out: Dict[str, Any] = {}
    try:
        _install(cells)
        compute.triton_installed = lambda: True  # as a CUDA box has it
        out["twins"] = {e.id: plan_cells.shape_twin(e.id, None) for e in catalog()}
        for card in cards:
            name, _, gib = card.rpartition(":")
            gpu = PlanGpu(name=name, total_gib=float(gib), free_gib=float(gib) - 0.4,
                          source="nvidia-smi")
            ctx = Context(cwd=tmp, fixed_gpu=gpu, fixed_interactive=False)
            out[f"models {name}"] = models_verb.run(models_verb.ModelsAction(), ctx).model_dump(
                mode="json")
            for model, seq, rank in asks:
                out[f"{name} {model} {seq} {rank}"] = shared.plan_for(
                    model, ctx, seq_len=seq, rank=rank).model_dump(mode="json")
    finally:
        (plan_cells.CELLS, plan_cells.CERTIFIED), compute.triton_installed = was, triton
    return out


def cuda_reach_changes(book: Dict[str, Any], root: Path = ROOT) -> List[str]:
    """Each way a CUDA certification cell would tell another machine something new, as a
    refusal line: a cell without only_its_machine, or one that changes, on any card but its own
    (REACH), a model's twin (shape_twin), the model list's words, or a plan of any catalog model
    or of the cell's own settings. A certification run answers its own card and nothing else."""
    sources = cuda_sources(book, root)
    if not sources:
        return []
    try:
        import build_plan_cells as b
    except ImportError:
        return [BUILDER_ABSENT]
    import tempfile

    from poros.cli.words.models import catalog
    from poros.core.card import _norm
    from poros.plan import cells as plan_cells

    out = [f"{s['id']}: a certification cell without only_its_machine would answer other "
           "machines too" for s in sources if not s.get("only_its_machine")]
    was = (b.SOURCES, b.ROOT)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plan_sources.yaml"
            path.write_text(plan_dumps(sources))
            b.SOURCES, b.ROOT = (path,), root
            new = [c for c in b.build() if not c["key"]["gpu"].startswith("Apple")]
            old = _hand_kept(plan_cells.CELLS + plan_cells.CERTIFIED)
            asks = [(e.id, None, 16) for e in catalog()] + sorted(
                {(c["key"]["model"], c["key"]["seq_len"], c["key"]["lora_rank"]) for c in new},
                key=str)
            own = {_norm(c["key"]["gpu"]) for c in new}
            cards = [c for c in REACH if _norm(c.rpartition(":")[0]) not in own]
            before = _cuda_told(old, asks, cards, Path(tmp))
            after = _cuda_told(old + new, asks, cards, Path(tmp))
    finally:
        b.SOURCES, b.ROOT = was
    for what in before:
        if before[what] == after[what]:
            continue
        which = ([k for k in before[what] if before[what][k] != after[what].get(k)]
                 if what == "twins" else [])
        out.append("the CUDA certification cells change what another machine is told: "
                   f"{what}" + (f" ({', '.join(which)})" if which else ""))
    return out


def dumps(book: Dict[str, Any]) -> str:
    return json.dumps(book, indent=1, sort_keys=False) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, default=SOURCES)
    ap.add_argument("--root", type=Path, default=ROOT, help="where receipt paths start")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--plan-out", type=Path, help="the planner cells' sources (default: "
                    "docs/validation/recipes/plan_sources.yaml; beside --out for another --out)")
    ap.add_argument("--check", action="store_true", help="compare with --out; write nothing")
    ap.add_argument("--strict", action="store_true", help="refuse while anything is pending")
    ap.add_argument("--release", action="store_true",
                    help="the release's file: --strict, with --drop the only way to ship less")
    ap.add_argument("--drop", action="append", default=[], metavar="KEY",
                    help="with --release: leave out a built-in that has no certified machine")
    ap.add_argument("--allow-fixtures", action="store_true",
                    help="accept test fixtures: only for an --out outside the package")
    a = ap.parse_args(argv)
    plan_out = a.plan_out or (PLAN_OUT if a.out == OUT else a.out.with_name("plan_sources.yaml"))
    if a.allow_fixtures and any(p.resolve().is_relative_to(d.resolve()) for p in (a.out, plan_out)
                                for d in (ROOT / "poros", ROOT / "docs")):
        print("generate: fixtures never reach the repository's files; --out and --plan-out "
              "must be elsewhere",
              file=sys.stderr)
        return 2
    if a.drop and not a.release:
        print("generate: --drop is a release step: give --release too", file=sys.stderr)
        return 2
    drop = tuple(a.drop)
    if a.check and not a.release and a.out.is_file():  # the drops the file was written with
        try:
            drop = tuple(json.loads(a.out.read_text()).get("dropped", []))
        except ValueError:
            pass
    book, refused = build(a.sources, a.root, a.allow_fixtures, drop)
    for line in refused:
        print("refused: " + line, file=sys.stderr)
    for key, where in book["pending"].items():
        print(f"pending: {key}" + (f" (waits for {', '.join(where)})" if where
                                   else " (no source names a receipt)"), file=sys.stderr)
    for key in book["dropped"]:
        print(f"dropped from the release: {key} (no certified machine; it stays defined in "
              "poros/recipes/presets.py, and ships once a run certifies it)", file=sys.stderr)
    if a.release:  # a release ships a certified built-in for each platform, or none at all
        have = {e["machine"]["platform"] for found in book["recipes"].values() for e in found}
        for platform in ("cuda", "mac"):
            if platform not in have:
                refused.append(f"--release: no built-in is certified on {platform}: the release "
                               "would ship none there")
                print(f"refused: {refused[-1]}", file=sys.stderr)
    if a.release:  # every run was made at the pin: the package must still be the pin's
        src = tomllib.loads(a.sources.read_text())
        exempt, bad = exempt_files(src)
        for line in bad:
            refused.append(f"--release: {line}")
            print(f"refused: {refused[-1]}", file=sys.stderr)
        strings = strings_only_files(src)
        for path, why in sorted(exempt.items()):  # said every time: never silent
            kind = " (strings only)" if path in strings else ""
            print(f"exempt from the pin{kind}: {path}: {why}", file=sys.stderr)
        mac_only = mac_only_files(src)[0]
        for path, why in sorted(mac_only.items()):  # said every time too
            print(f"only a Mac runs it, so the CUDA certifications stand: {path}: {why}",
                  file=sys.stderr)
        for line in package_moved(src["commit"], a.root, {**exempt, **mac_only}, strings)[:20]:
            refused.append(f"--release: {line}")
            print(f"refused: {refused[-1]}", file=sys.stderr)
        if src.get("mac_commit"):  # the Mac's runs were made there: the package is that commit's
            for line in package_moved(str(src["mac_commit"]), a.root, exempt, strings)[:20]:
                refused.append(f"--release: against the Mac certification commit: {line}")
                print(f"refused: {refused[-1]}", file=sys.stderr)
    builder = has_builder()
    if not builder and not a.check:  # a write is the release repository's: its gate must run
        refused.append("writing certifications or cells needs the planner's cell builder, which "
                       "the release repository carries and this tree does not: the planner-"
                       "decision check runs on every write there, and cannot be skipped")
        print(f"refused: {refused[-1]}", file=sys.stderr)
    elif not builder:  # --check reads: the shipped file against the receipts, without that check
        print("decisions: not checked here: the planner-decision check runs only where the "
              "planner's cell builder is (the release repository)", file=sys.stderr)
    if not refused and builder:  # before any write or check: what the new cells change
        accepted = tuple(x["cell"] for x in
                         tomllib.loads(a.sources.read_text()).get("accept_decision", []))
        notes: List[str] = []
        for line in (decision_changes(book, a.root, accepted, notes)
                     + cuda_reach_changes(book, a.root)
                     + mac_decision_changes(book, a.root, accepted, notes)):
            refused.append(line)
            print(f"refused: {line}", file=sys.stderr)
        for line in notes:
            print(f"decisions: {line}", file=sys.stderr)
    if refused or ((a.strict or a.release) and book["pending"]):
        print("generate: nothing written", file=sys.stderr)
        return 1
    try:  # what the CLI reads back: every text field plain, every number a number
        certified.parse(json.loads(dumps(book)))
    except ValueError as exc:
        print(f"generate: the file would not read back ({exc}); nothing written", file=sys.stderr)
        return 1
    text = dumps(book)
    plan = plan_dumps(plan_sources(book, a.root))
    if a.check:
        same = a.out.is_file() and a.out.read_text() == text
        same_plan = plan_out.is_file() and plan_out.read_text() == plan
        print(f"generate: {a.out.name} is {'current' if same else 'NOT what the receipts say'}; "
              f"{plan_out.name} is {'current' if same_plan else 'NOT what the receipts say'}")
        return 0 if same and same_plan else 1
    a.out.write_text(text)
    plan_out.parent.mkdir(parents=True, exist_ok=True)
    plan_out.write_text(plan)
    n = sum(len(v) for v in book["recipes"].values())
    print(f"generate: wrote {a.out}: {n} certified runs of {len(book['recipes'])} recipes, "
          f"{len(book['pending'])} pending")
    return 0


if __name__ == "__main__":
    sys.exit(main())
