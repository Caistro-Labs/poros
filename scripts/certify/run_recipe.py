#!/usr/bin/env python3
"""Certify one built-in recipe on this machine: run it through the public CLI and write the receipt
scripts/certify/generate.py reads: a certification run of the recipe as shipped, on this machine.

    python scripts/certify/run_recipe.py big-72b --out ./cert/big-72b --download
    python scripts/certify/run_recipe.py first-run --out ./cert/first-run --simulate  # no GPU

How many updates it measures, and whether the resume and the eval run, are the recipe's
scripts/certify/sources.toml entry's (--machine picks one where the recipe has several).

What it runs, each through `poros` exactly as a user would, in a POROS_HOME of its own:
  1. prepare: one update of the recipe, not measured (downloads, builds an SSD store, warms caches);
  2. main: the recipe's own train command on sample data of its kind, --steps N, sampled for the
     GPU memory its processes use (nvidia-smi, load and training) and the host's resident memory
     that is not file-backed (RssAnon + RssShmem of the process tree, 4 times a second); torch's
     reserved and allocated peaks in training come from the run's own receipt.json;
  3. resume (unless its entry says resume = false): the same command again, stopped with
     `poros runs stop` halfway and finished with `poros runs continue`; its adapter must be the
     main run's, byte for byte;
  4. eval (unless eval = false): `poros eval` of the main run.
The receipt (certification.json, and the main run's receipt.json, events.jsonl and
config.resolved.json beside it) holds the resolved config, this machine, every update's time, the
peaks and the commit. Home paths, the user name and the host name are replaced before writing.
"""

from __future__ import annotations

import argparse
import dataclasses
import getpass
import hashlib
import json
import os
import platform
import random
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poros.data import FEWEST, held_count  # noqa: E402
from poros.recipes import certified, presets  # noqa: E402

SCHEMA = "poros.certify.run.v1"
SOURCES = ROOT / "scripts" / "certify" / "sources.toml"
SAMPLE_S = 0.25
# Tokens a row cut to the length may leave out of its update's count: the events count what the
# loss reads, without a row's prompt and its chat template (a Qwen2.5 Instruct template's default
# system line and turn markers come to about 43 tokens on a sample row). Every sample row runs
# past the length (sample_rows), so an update of full rows counts at least
# (length - ROW_ALLOWANCE) x rows.
ROW_ALLOWANCE = 64
# Plain English words for sample rows: license-free, and any tokenizer reads them as text.
WORDS = ("the a an and or but if then when while because so that this these those it they we "
         "you he she time year day way thing man world life hand part child eye woman place work "
         "week case point number group problem fact water room mother area money story month lot "
         "right study book word business issue side kind head house service friend father power "
         "hour game line end member law car city community name team minute idea body back parent "
         "face others level office door health person art war history party result change morning "
         "reason research girl guy moment air teacher force education foot boy age policy music "
         "market sense nation plan college interest death experience effect use class control care "
         "field development role effort rate heart drug show leader light voice wife police mind "
         "price report decision son view relationship town road arm difference value building "
         "action model season society tax director position player record paper space ground form "
         "event official matter center couple site project activity star table need court oil "
         "situation cost industry figure street image phone data picture practice piece land "
         "product doctor wall patient worker news test movie north love support technology step "
         "baby computer type attention film tree source organization hair window evidence "
         "population site truth river garden letter answer question bridge winter summer").split()


def _say(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


# ── sample data ────────────────────────────────────────────────────────


def _words(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n)).capitalize() + "."


def sample_rows(kind: str, rows: int, seq_len: int, seed: int = 0,
                key: str = "") -> List[Dict[str, Any]]:
    """Rows of ``kind`` whose answers run past ``seq_len`` tokens, so every row trains at the full
    length (a builder cuts a longer row at the length; a shorter one would measure less). Each
    kind draws its own words (its own seed): qa rows become chats. A recipe's ``key`` seeds them
    too, so no two recipes train on the same rows (moe-30b and moe-30b-ssd once ended at one
    adapter); the same key gives the same rows every time."""
    rng = random.Random(f"{key}:{kind}:{seed}" if key else f"{kind}:{seed}")
    long = int(seq_len * 1.5) + 16  # words; any tokenizer gives at least one token a word
    out: List[Dict[str, Any]] = []
    for _ in range(rows):
        ask = _words(rng, 12)
        if kind == "chats":
            out.append({"messages": [{"role": "user", "content": ask},
                                     {"role": "assistant", "content": _words(rng, long)}]})
        elif kind == "qa":
            out.append({"question": ask, "answer": _words(rng, long)})
        elif kind == "pairs":
            out.append({"prompt": ask, "chosen": _words(rng, long),
                        "rejected": _words(rng, long)})
        elif kind == "text":
            out.append({"text": _words(rng, long)})
        else:
            raise ValueError(f"no sample data for {kind!r}")
    return out


def sample_kind(r: presets.Recipe) -> str:
    """The kind of rows a run of ``r`` is certified on: its data, a chat where it takes any."""
    return {"any": "chats"}.get(r.data, r.data)


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> str:
    text = "".join(json.dumps(r) + "\n" for r in rows)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


# ── samplers ───────────────────────────────────────────────────────────


def _tree(root: int) -> List[int]:
    """``root`` and every process under it (Linux /proc)."""
    parent: Dict[int, int] = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                parent[int(d.name)] = int((d / "stat").read_text().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
    out, todo = [], [root]
    while todo:
        pid = todo.pop()
        out.append(pid)
        todo += [p for p, pp in parent.items() if pp == pid]
    return out


def _anon_shmem_kb(pid: int) -> int:
    got = 0
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("RssAnon:", "RssShmem:")):
                got += int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return 0
    return got


def _smi(*query: str) -> List[List[str]]:
    try:
        out = subprocess.run(["nvidia-smi", *query, "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [[c.strip() for c in line.split(",")] for line in out.splitlines() if line.strip()]


class Sampler(threading.Thread):
    """The card's memory held by a process tree, and its host memory that is not file-backed."""

    def __init__(self, pid: int, cuda: bool) -> None:
        super().__init__(daemon=True)
        self.pid, self.cuda, self.stop_at = pid, cuda, threading.Event()
        self.vram_mib = 0.0
        self.host_kb = 0
        self.samples = 0
        self.vram_source = "nvidia-smi per process" if cuda else "none"
        base = _smi("--query-gpu=memory.used") if cuda else []
        self.base_mib = float(base[0][0]) if base and base[0][0].replace(".", "").isdigit() else 0

    def run(self) -> None:
        linux = Path("/proc/self/status").is_file()
        while not self.stop_at.is_set():
            pids = _tree(self.pid) if linux else [self.pid]
            if linux:
                self.host_kb = max(self.host_kb, sum(_anon_shmem_kb(p) for p in pids))
            if self.cuda:
                apps = _smi("--query-compute-apps=pid,used_memory")
                mine = [float(m) for p, m in (a[:2] for a in apps if len(a) >= 2)
                        if p.isdigit() and int(p) in pids and m.replace(".", "").isdigit()]
                if mine:
                    self.vram_mib = max(self.vram_mib, sum(mine))
                else:  # no per-process figures here (WSL2, some containers): the card's, less idle
                    used = _smi("--query-gpu=memory.used")
                    if used and used[0][0].replace(".", "").isdigit():
                        self.vram_source = "nvidia-smi card total less its reading before launch"
                        self.vram_mib = max(self.vram_mib, float(used[0][0]) - self.base_mib)
            self.samples += 1
            self.stop_at.wait(SAMPLE_S)


# ── the CLI ────────────────────────────────────────────────────────────


class Cli:
    def __init__(self, poros: List[str], env: Dict[str, str], logs: Path) -> None:
        self.poros, self.env, self.logs = poros, env, logs
        self.started: List[subprocess.Popen] = []  # every command it ran: stopped with it

    def start(self, args: List[str], tag: str) -> subprocess.Popen:
        log = open(self.logs / f"{tag}.log", "w")
        # in the driver's own process group: a guard that stops the driver's group stops the
        # trainer with it (a session of its own would leave it running)
        p = subprocess.Popen([*self.poros, *args], env=self.env, stdout=log,
                             stderr=subprocess.STDOUT)
        self.started.append(p)
        return p

    def stop(self, grace: float = 120) -> None:
        """Stop what it started that still runs, and wait for each (a trainer checkpoints on
        SIGTERM): then nothing it started is left on the GPU."""
        live = [p for p in self.started if p.returncode is None]
        for p in live:
            try:
                p.terminate()
            except ProcessLookupError:
                pass
        # reaped by pid, not Popen.wait: a signal handler can run inside a Popen.wait, whose
        # lock a second wait on the same Popen would block on for ever
        end = time.time() + grace
        while live and time.time() < end:
            live = [p for p in live if not _reaped(p)]
            time.sleep(0.1)
        for p in live:
            p.kill()
            while not _reaped(p):
                time.sleep(0.1)

    def run(self, args: List[str], tag: str, timeout: Optional[float] = None
            ) -> tuple[int, Optional[Dict[str, Any]]]:
        p = self.start(args, tag)
        try:
            rc = p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.terminate()
            rc = p.wait()
        return rc, self.result(tag)

    def result(self, tag: str) -> Optional[Dict[str, Any]]:
        """The JSON object a --json command prints last."""
        for line in reversed((self.logs / f"{tag}.log").read_text(errors="replace").splitlines()):
            if line.startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    continue
        return None


def _reaped(p: subprocess.Popen) -> bool:
    """Whether ``p`` has exited (and is reaped), without Popen's lock."""
    try:
        pid, _ = os.waitpid(p.pid, os.WNOHANG)
    except ChildProcessError:  # reaped already
        return True
    return pid != 0


def forward_term(cli: "Cli") -> None:
    """SIGTERM to the driver (a guard's kill of its pid) stops its trainer too, then the driver."""
    def on_term(signum: int, frame: Any) -> None:
        cli.stop()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, on_term)


LOCK = ".poros-certify-lock"


def hold(folder: Path) -> Path:
    """This run's hold on ``folder``: a lock file with its pid. SystemExit where another driver
    that is still running holds it; a lock left by one that died is taken over."""
    path = folder / LOCK
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            try:
                pid = int(path.read_text().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and _alive(pid):
                raise SystemExit(f"{folder} is held by another certification run (pid {pid})")
            path.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w") as f:
            f.write(f"{os.getpid()}\n")
        return path
    raise SystemExit(f"{folder}: its lock could not be taken")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _runs(home: Path) -> set:
    return set((home / "runs").glob("*"))


def _run_folder(home: Path, name: str, before: set, wait: float = 600) -> Path:
    """The run folder `poros train --name NAME` made under POROS_HOME since ``before``."""
    end = time.time() + wait
    while time.time() < end:
        for d in sorted(_runs(home) - before):
            try:
                if json.loads((d / "action.json").read_text()).get("name") == name:
                    return d
            except (OSError, ValueError):
                continue
        time.sleep(1)
    raise RuntimeError(f"no run folder named {name} appeared")


def step_events(folder: Path) -> List[Dict[str, Any]]:
    out = []
    path = folder / "events.jsonl"
    if not path.is_file():
        return out
    for line in path.read_text().splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get("event_type") == "step":
            out.append(e.get("data") or {})
    return out


def per_update_tokens(steps: List[Dict[str, Any]], recipe_loop: bool) -> List[int]:
    """Each update's tokens. A recipe's loop (poros.Run, every built-in's, on CUDA and a Mac)
    writes each step's own count, the tokens it trained on (poros.train.run: "a step's, never
    the run's count so far"); the engine's own loop and the preview's stand-in write the run's
    running count, which starts again in a new process."""
    if recipe_loop:
        return [int(s.get("tokens") or 0) for s in steps]
    out, prev = [], 0
    for s in steps:
        t = int(s.get("tokens") or 0)
        out.append(t - prev if t >= prev else t)
        prev = t
    return out


# The adapter files a finished run writes in its folder: a dense run PEFT's safetensors, a MoE
# run Poros's native state (poros.train.checkpointing).
ADAPTER_FILES = ("adapter_model.safetensors", "adapter_weights.pt")


def same_adapter(first: Path, second: Path, simulated: bool) -> Dict[str, Any]:
    """Whether two runs ended at the same adapter: the same weight file in each, its bytes equal
    (safetensors), or every key and tensor equal (a .pt, loaded with weights_only=True). Nothing
    to compare passes only for a preview, which writes no weights; any other run fails."""
    for name in ADAPTER_FILES:
        a, b = first / name, second / name
        if not a.is_file() and not b.is_file():
            continue
        out: Dict[str, Any] = {"adapter_file": name, "adapter_sha256": _sha(a),
                               "resumed_adapter_sha256": _sha(b)}
        if not (a.is_file() and b.is_file()):
            return {**out, "adapter_equal": False, "note": "only one of the runs wrote it"}
        if name.endswith(".safetensors"):
            return {**out, "adapter_equal": out["adapter_sha256"] == out["resumed_adapter_sha256"],
                    "note": "bytes compared"}
        import torch

        x, y = (torch.load(f, map_location="cpu", weights_only=True) for f in (a, b))
        equal = _same_state(x, y)
        return {**out, "adapter_equal": equal, "note": "every key and tensor compared"}
    if simulated:
        return {"adapter_equal": None,
                "note": "no adapter files to compare (a preview writes none)"}
    return {"adapter_equal": False,
            "note": f"neither run wrote an adapter file ({', '.join(ADAPTER_FILES)})"}


def _same_state(x: Any, y: Any) -> bool:
    """Two loaded states equal exactly: the same keys, each tensor equal in dtype, shape, values."""
    import torch

    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_same_state(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return len(x) == len(y) and all(_same_state(a, b) for a, b in zip(x, y))
    if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return x.dtype == y.dtype and x.shape == y.shape and torch.equal(x, y)
    return type(x) is type(y) and x == y


def _sha(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def tree_root() -> Path:
    import poros

    return Path(poros.__file__).resolve().parent.parent


def manifest_of(root: Path) -> Dict[str, str]:
    """{path: sha256} of every file under ``root`` (a copy of the tree), as sha256sum lists them."""
    out = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts and ".git" not in path.parts:
            out[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def read_manifest(path: Path) -> Dict[str, str]:
    """A ``sha256sum``-format list (``<sha256>  <path>``) as {path: sha256}."""
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            sha, _, name = line.partition("  ")
            out[name.strip().removeprefix("./")] = sha.strip()
    return out


def tree_matches(root: Path, manifest: Dict[str, str]) -> List[str]:
    """What differs between the copy at ``root`` and ``manifest``: a file changed or missing,
    and any Python file under poros/ the commit does not have. [] where it is the commit's."""
    have = manifest_of(root)
    bad = [f"{n}: {'missing' if n not in have else 'changed'}"
           for n, sha in manifest.items() if have.get(n) != sha]
    bad += [f"{n}: not in the commit" for n in have
            if n.startswith("poros/") and n.endswith(".py") and n not in manifest]
    return bad


def commit_of(stated: Optional[str], manifest: Optional[Path] = None
              ) -> tuple[Optional[str], Optional[bool]]:
    """The commit the imported poros is and whether its files are the commit's: a git
    checkout's HEAD and its tracked files unchanged; a copy (no .git) needs ``stated`` and the
    commit's ``manifest`` (scripts/certify/run_recipe.py --manifest-of on a git archive of it),
    which every file must match."""
    where = tree_root()
    if not (where / ".git").exists():
        if not stated or manifest is None:
            raise SystemExit("this tree is not a git checkout: give --commit and --manifest (the "
                             "release's sha256sum list of that commit's files, which "
                             "--manifest-of prints for a git archive of it)")
        bad = tree_matches(where, read_manifest(manifest))
        if bad:
            raise SystemExit(f"this tree is not {stated[:12]}'s: " + "; ".join(bad[:10]))
        return stated, True
    try:
        head = subprocess.run(["git", "-C", str(where), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=20)
        dirty = subprocess.run(["git", "-C", str(where), "status", "--porcelain",
                                "--untracked-files=no"], capture_output=True, text=True,
                               timeout=20)
    except (OSError, subprocess.SubprocessError):
        return stated, None
    if head.returncode != 0:
        return stated, None
    found = head.stdout.strip()
    if stated and stated != found:
        raise SystemExit(f"--commit {stated} is not this checkout's {found}")
    return found, dirty.stdout.strip() == ""


def _os_release() -> str:
    """``Ubuntu 24.04.1 LTS``, from /etc/os-release; "" where there is none."""
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                return line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return ""


def machine_facts(simulated: bool) -> Dict[str, Any]:
    from poros.cli.context import Context

    m = certified.here(Context())
    if m is None:
        raise SystemExit("no GPU found to certify on")
    facts: Dict[str, Any] = dataclasses.asdict(m)
    facts["mac_class"] = m.mac_class or None
    facts["os"] = platform.platform()
    # the OS version, shown with the certification and never matched (a Mac updates itself)
    if m.platform == "mac":
        ver = subprocess.run(["/usr/bin/sw_vers", "-productVersion"], capture_output=True,
                             text=True, timeout=20).stdout.strip() if not simulated else ""
        facts["os_version"] = f"macOS {ver}" if ver else None
    else:
        from poros.recipes.certified import under_wsl

        facts["os_version"] = (_os_release() or f"Linux {platform.release()}") + (
            " (WSL2)" if under_wsl() else "")
    facts["python"] = platform.python_version()
    for pkg in ("torch", "transformers", "bitsandbytes"):
        try:
            from importlib.metadata import version

            facts[pkg] = version(pkg)
        except Exception:
            facts[pkg] = None
    if m.platform == "cuda" and not simulated:
        drv = _smi("--query-gpu=driver_version")
        facts["driver"] = drv[0][0] if drv else None
    return facts


# Mount points of other disks: their names are someone's (a drive named for its owner)
_MOUNTS = re.compile(r"(/Volumes|/media/[^/\s\"']+|/run/media/[^/\s\"']+|/mnt)/[^/\s\"']+")
_PLACES: List[tuple[str, str]] = []  # (a folder of this run's, its mark), longest first


def name_places(**places: Optional[Path]) -> None:
    """The folders this run wrote to, shown in a receipt as ``<work>``, ``<home>`` ..."""
    _PLACES[:] = sorted(((str(p), f"<{k}>") for k, p in places.items() if p),
                        key=lambda x: -len(x[0]))


def scrub(text: str) -> str:
    """No home path, user name, host name, disk name or folder of this run's leaves this machine
    in a receipt: each becomes a mark (``~``, ``<user>``, ``<host>``, ``<disk>``, ``<work>``)."""
    for place, mark in _PLACES:
        text = text.replace(place, mark)
    text = _MOUNTS.sub("<disk>", text)
    text = text.replace(str(Path.home()), "~")
    for word, mark in ((getpass.getuser(), "<user>"), (socket.gethostname(), "<host>"),
                       (socket.gethostname().split(".")[0], "<host>")):
        if word and len(word) > 2:
            text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(word)}(?![A-Za-z0-9])", mark, text)
    return text


# ── the fit check ──────────────────────────────────────────────────────

# What a fit check flags: a line that tells a user a run may not fit, or that the plan cannot
# say. A certified recipe on its certified machine, once its cell is in, shows none of them.
DOUBT = ("may not fit", "unknown", "won't fit", "estimated")


def recipe_config(r: presets.Recipe, data: str) -> Any:
    """The PorosConfig a run of ``r`` resolves to on CUDA before the preflight: its settings, the
    SSD store where train keeps it, and path fast+cache as the path and the cache it sets."""
    from poros.cli.verbs import shared
    from poros.cli.verbs.train import sss_dir
    from poros.core.config import PorosConfig

    st = {k: v for k, v in {**presets.loop_preset(r), **r.settings}.items()
          if k not in ("max_steps", "num_epochs", "quantization")}
    if "block_size" not in r.settings and _moe(r):
        st.pop("block_size", None)  # poros train clears a preset's block for a MoE model
    if st.get("path"):
        st.update(shared.path_config(st.pop("path")))
    if st.get("store") == "ssd":
        st["store_path"] = str(sss_dir())
    return PorosConfig(model_name_or_path=r.model, dataset=data, quantization="nf4", **st)


def _moe(r: presets.Recipe) -> bool:
    from poros.cli.words.models import by_id

    entry = by_id(r.model)
    return entry is not None and entry.entry == "moe"


def start_refusals(r: presets.Recipe, ctx: Any, data: Path) -> List[str]:
    """What stops ``poros train --recipe <key> --data <rows>`` before the load, needing no GPU
    and no download: the config the train verb builds (resolve, inspect, then
    train_run.config_dict), a Mac's validator, the CUDA defaults and a path you set
    (preflight.start.path_refusal_error), the route's store check for a dense model, the
    compute's refusals (preflight.compute.refuse) and the MoE path's settings
    (preflight.start.moe_settings_refusal). [] where it would start; else each in its words."""
    from types import SimpleNamespace

    from poros import metal
    from poros.cli.verbs import train as tv
    from poros.cli.verbs.train_run import config_dict
    from poros.cli.words.grammar import parse
    from poros.core.config import PorosConfig
    from poros.core.errors import PorosError
    from poros.core.refusal import Refusal
    from poros.plan.configs import cached_model_config
    from poros.preflight import compute as pre_compute
    from poros.preflight.start import moe_settings_refusal, path_refusal_error

    try:
        draft = tv.resolve(parse(r.argv(str(data))), ctx)
        if draft.questions:
            return ["train asks: " + "; ".join(q.prompt for q in draft.questions)]
        a = tv.inspect(draft.action, ctx)
    except (Refusal, PorosError) as exc:  # train refuses this machine before it starts
        return [f"train refuses: {exc}"]
    try:
        cfg = PorosConfig(**config_dict(a, str(Path(data).parent / "out")))
    except Exception as exc:  # what a run's config refuses
        return [f"config: {exc}"]
    moe, out = _moe(r), []
    model_json = cached_model_config(r.model)
    if ctx.unified:
        try:
            metal.validate(cfg)
        except Exception as exc:
            out.append(f"Mac validator: {exc}")
    else:
        cfg.apply_cuda_defaults(model_json)
        if (err := path_refusal_error(cfg, model_json)) is not None:
            out.append(f"path: {err}")
        if not moe and cfg.store != "ram":
            out.append(f"route: store={cfg.store!r} streams a MoE model's experts")
        try:
            pre_compute.refuse(SimpleNamespace(config=cfg, device=SimpleNamespace(type="cuda")),
                               "moe" if moe else "dense", model_json)
        except Exception as exc:
            out.append(f"compute: {exc}")
    if moe and (err := moe_settings_refusal(cfg, ctx.unified)) is not None:
        out.append(f"MoE path: {err}")
    return out


def gate_of(r: presets.Recipe, ctx: Any, data: str, config: Any = None) -> Dict[str, Any]:
    """What the run's preflight decides for ``r`` (or ``config``, another config of its model) on
    the machine ``ctx`` reads: the path it decides (decide_path_to_fit), the tables
    (decide_offload_for_run), the warning it prints ("may not fit"), and its refusals
    (vram_error, host RAM) -- poros.preflight.fit.check, as PorosTrainer runs it."""
    from poros.cli.words.models import by_id
    from poros.plan.configs import cached_model_config
    from poros.preflight import fit as pf

    cfg = config if config is not None else recipe_config(r, data)
    model_config = cached_model_config(r.model)
    if not ctx.unified:
        cfg.apply_cuda_defaults(model_config)
    e = by_id(r.model)
    plan = pf.run_planner(e.arch if e else None, gpu=ctx.gpu, host_free_gib=ctx.host_free_gib,
                          model_config=model_config, entry=e.entry if e else None)
    fit = pf.check(cfg, plan)
    return {"path": list(fit.path) if fit.path else None, "offload": fit.offload,
            "warning": fit.warning, "vram_refusal": str(fit.vram) if fit.vram else None,
            "host_refusal": str(fit.host) if fit.host else None, "ran_path": cfg.path}


def check_fit(r: presets.Recipe, ctx: Any, data: Path, updates: int = 16) -> Dict[str, Any]:
    """``poros train`` of the recipe up to its confirmations (resolve, inspect, the questions it
    would ask), in this process on the machine ``ctx`` reads: nothing trains. With what its
    preflight decides (gate_of). The train card is told every setting the recipe sets, the
    pinned expert cache included."""
    from poros.cli.sheets.fit import _peak
    from poros.cli.verbs import shared
    from poros.cli.verbs import train as tv
    from poros.cli.words.grammar import parse

    out: Dict[str, Any] = {"recipe": r.key}
    from poros.core.errors import PorosError
    from poros.core.refusal import Refusal

    args = train_args(r, data, updates, f"check-{r.key}", False)
    try:
        draft = tv.resolve(parse([a for a in args if a not in ("--yes", "--json")]), ctx)
        a = None if draft.questions else tv.inspect(draft.action, ctx)
        asks = [] if a is None else [q for q in tv.confirmations(a, ctx) if q.id != "start"]
    except (Refusal, PorosError) as exc:  # train refuses before it starts: said as it said it
        draft, a, asks = None, None, []
        out["train"] = {"asks": ["refused"], "asks_text": [str(exc)]}
    if draft is not None and draft.questions:
        out["train"] = {"asks": [q.prompt for q in draft.questions]}
    elif a is not None:
        p = a.plan
        out["train"] = {"verdict": shared.fit_note(p), "peak_gib": _peak(p).value,
                        "source": _peak(p).source, "basis": p.fit_basis,
                        "host_fits": p.host_fits, "asks": [q.id for q in asks],
                        "asks_text": [q.prompt for q in asks], "seq_len": a.seq_len}
    if not ctx.unified:
        out["gate"] = gate_of(r, ctx, str(data))
    said = json.dumps(out).lower()
    out["doubts"] = [w for w in DOUBT if w in said]
    # what the start of train refuses before the load for the recipe's own settings: a
    # simulation that skips these could say fits where train refuses
    out["start_refused"] = start_refusals(r, ctx, data)
    out["refused"] = bool(out["train"].get("asks") or (out.get("gate") or {}).get("vram_refusal")
                          or (out.get("gate") or {}).get("host_refusal") or out["start_refused"])
    return out


def held_here(r: presets.Recipe, ctx: Any, data: Path, updates: int) -> Dict[str, Any]:
    """What ``poros train`` of the recipe on ``data`` will hold out to score here, as train's own
    start summary says it (resolve, inspect, poros.cli.verbs.train_run.plan_said: the recipe's
    setting, else the Mac table row's, else the config's), with nothing trained. {"held": None,
    "why": ...} where train cannot say before it starts."""
    from poros.cli.verbs import train as tv
    from poros.cli.verbs.train_run import plan_said
    from poros.cli.words.grammar import parse
    from poros.core.config import PorosConfig
    from poros.core.errors import PorosError
    from poros.core.refusal import Refusal

    args = train_args(r, data, updates, f"held-{r.key}", False)
    try:
        draft = tv.resolve(parse([w for w in args if w not in ("--yes", "--json")]), ctx)
        if draft.questions:
            return {"held": None, "why": "train asks: " + "; ".join(
                q.prompt for q in draft.questions)}
        a = tv.inspect(draft.action, ctx)
    except (Refusal, PorosError, ValueError) as exc:
        return {"held": None, "why": f"train refuses: {exc}"}
    said = plan_said(a)

    def setting(name: str) -> Any:
        return a.config.get(name, a.preset.get(name, PorosConfig.model_fields[name].default))

    return {"held": said.get("eval_rows"), "split": float(setting("eval_split") or 0),
            "count": int(setting("eval_rows") or 0), "rows": said.get("rows")}


def held_of_run(resolved: Dict[str, Any], rows: int) -> int:
    """What a finished run held out, by its own resolved config and the product's rule."""
    return held_count(rows, float(resolved.get("eval_split") or 0),
                      int(resolved.get("eval_rows") or 0))


def adapter_sha(folder: Path) -> Optional[str]:
    """The sha256 of the weights a finished run wrote (the first of ADAPTER_FILES there)."""
    for name in ADAPTER_FILES:
        if (folder / name).is_file():
            return _sha(folder / name)
    return None


def eval_scored(ev: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What a poros eval scored and measured: the rows (held out, a file, or the trained ones),
    and its metric: a chats run's held-out loss, a pairs run's DPO loss before and after (a pairs
    eval's ``metrics`` is empty by design: its numbers are its base and trained sides)."""
    ev = ev or {}
    base, trained = ev.get("base") or {}, ev.get("trained") or {}
    loss = (ev.get("metrics") or {}).get("eval_loss")
    if ev.get("shape") == "pairs":
        metric = {"base_loss": base.get("loss"), "trained_loss": trained.get("loss"),
                  "base_prefers": base.get("prefers"), "trained_prefers": trained.get("prefers"),
                  "improved": ev.get("improved")}
        has = all(isinstance(x, (int, float)) for x in (base.get("loss"), trained.get("loss")))
    else:
        metric = {"eval_loss": loss, "base_loss": base.get("loss")}
        has = isinstance(loss, (int, float))
    return {"rows_from": ev.get("rows_from"), "rows": ev.get("rows"), "shape": ev.get("shape"),
            "metric": metric, "has_metric": has}


# ── the phases ─────────────────────────────────────────────────────────


def train_args(r: presets.Recipe, data: Path, steps: int, name: str,
               download: bool) -> List[str]:
    """The recipe's own train command (presets.Recipe.argv) for ``steps`` updates."""
    bare = dataclasses.replace(r, settings={k: v for k, v in r.settings.items()
                                            if k not in ("max_steps", "num_epochs")})
    return [*bare.argv(str(data)), "--steps", str(steps), "--name", name, "--yes", "--json",
            *(["--download"] if download else [])]


def main_run(cli: Cli, args: List[str], home: Path, name: str, cuda: bool
             ) -> Dict[str, Any]:
    before = _runs(home)
    p = cli.start(args, "main")
    sampler = Sampler(p.pid, cuda)
    sampler.start()
    rc = p.wait()
    sampler.stop_at.set()
    sampler.join()
    try:
        folder = _run_folder(home, name, before, wait=5)
    except RuntimeError:  # train refused or asked before the run began: its JSON says why
        folder = None
    return {"rc": rc, "folder": folder, "result": cli.result("main"),
            "vram_mib": sampler.vram_mib, "vram_source": sampler.vram_source,
            "host_kb": sampler.host_kb, "samples": sampler.samples}


def resume_run(cli: Cli, args: List[str], home: Path, name: str, steps: int) -> Dict[str, Any]:
    """The same run stopped halfway with `poros runs stop`, then `poros runs continue`."""
    stop_after = max(1, steps // 2)
    before = _runs(home)
    p = cli.start(args, "resume-start")
    folder = _run_folder(home, name, before)
    while p.poll() is None and len(step_events(folder)) < stop_after:
        time.sleep(0.2)
    run_id = folder.name
    stop_rc, _ = cli.run(["runs", "stop", run_id, "--json"], "resume-stop", timeout=600)
    first_rc = p.wait()
    stopped_after = len(step_events(folder))
    cont_rc, cont = cli.run(["runs", "continue", run_id, "--yes", "--json"], "resume-continue")
    return {"folder": folder, "stopped_after": stopped_after, "stop_rc": stop_rc,
            "first_rc": first_rc, "continue_rc": cont_rc,
            "continued": bool(cont and cont.get("runs") and
                              cont["runs"][0].get("state") == "succeeded")}


# A folder the driver made says so, by a file in it: only such a folder is used again, and only
# what this run made is removed after it. A folder of anyone else's is never written into.
WORK_MARK, HOME_MARK = ".poros-certify-work", ".poros-certify-home"


def claim(folder: Path, mark: str) -> List[Path]:
    """Make ``folder`` this driver's: create it, or use it again where it is the driver's own
    (holds ``mark``) or empty. SystemExit for any other: its contents are someone's. Returns
    what this run created (the folder, a new one; else nothing), which ``tidy`` may remove."""
    if folder.exists() and not folder.is_dir():
        raise SystemExit(f"{folder} is not a folder")
    if folder.is_dir() and any(folder.iterdir()) and not (folder / mark).is_file():
        raise SystemExit(f"{folder} holds files the certification driver did not write: name an "
                         "empty folder, or one a run of it made")
    new = not folder.exists()
    folder.mkdir(parents=True, exist_ok=True)
    (folder / mark).write_text("made by scripts/certify/run_recipe.py\n")
    return [folder] if new else []


def tidy(made: List[Path]) -> List[str]:
    """Remove what this run created, and nothing else: the folders it made (with what it wrote
    in them), the files it wrote into a folder that was already there. What it could not
    remove, each with why ([] where all of it went)."""
    left: List[str] = []
    for path in reversed(made):
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        except OSError as exc:
            left.append(f"{path}: {exc.strerror or exc}")
    return left


# A receipt's status: "ok", "failed" (a phase started and broke), or this one: poros train said
# no before the run began (it does not fit this machine, or it asked something), which is a
# finding about the machine, not a fault.
REFUSED = "refused before start"


def refusal_of(result: Optional[Dict[str, Any]]) -> Optional[str]:
    """poros train's own words where its JSON says it refused or asked before starting."""
    said = result or {}
    if said.get("schema") == "poros.error.v1" or said.get("error"):
        return (said.get("error") or {}).get("message") or "train refused it"
    if said.get("questions"):
        return "train asked: " + "; ".join(q.get("prompt", "") for q in said["questions"])
    return None


def keep_logs(out: Path, logs: Path) -> None:
    for f in sorted(logs.glob("*.log")):
        (out / "logs").mkdir(exist_ok=True)
        (out / "logs" / f.name).write_text(scrub(f.read_text(errors="replace")))


def failure_receipt(out: Path, head: Dict[str, Any], logs: Path, phase: str, why: str,
                    log: str, refused: bool = False) -> Dict[str, Any]:
    """The receipt of a run that stopped in ``phase``: the phase, why, and its log's last lines,
    with the logs beside it. ``refused``: poros train said no before the run began."""
    cert = {**head, "status": REFUSED if refused else "failed",
            "failed": [{"phase": phase, "reason": scrub(why),
                        "log_tail": [scrub(x) for x in _tail(logs / log)]}]}
    keep_logs(out, logs)
    (out / "commit.txt").write_text(f"{head['commit']}\n")
    (out / "certification.json").write_text(scrub(json.dumps(cert, indent=1)) + "\n")
    _say(f"{cert['status']} in {phase}: {why}")
    return cert


def sample_count(updates: int, per_update: int, split: float) -> int:
    """Rows enough, by the product's own rule (poros.data: none held out under FEWEST rows, else
    the share), that a run holds rows out to score and still trains ``updates`` of
    ``per_update`` rows with a pass to spare."""
    n = max(FEWEST, updates * per_update + per_update)
    while n - held_count(n, split) < updates * per_update + per_update:
        n += 1
    return n


def memory_record(platform: str, main: Dict[str, Any], receipt: Dict[str, Any],
                  cuda: bool) -> Dict[str, Any]:
    """The memory figures the product measures here, each by its name; a null with the reason
    for one it does not measure here, never a 0.0 that reads as a measurement."""
    gb = receipt.get("peak_vram_allocated_gb")
    if platform == "mac":
        return {"gpu_memory": {"used_gib": None, "why_none": "nvidia-smi: not on a Mac"},
                "torch_memory": {"reserved_peak_gib": None,
                                 "why_none": "a Mac's one pool has no reserved peak",
                                 "allocated_peak_gib": certified.to_gib(gb, "GB"),
                                 "what": "the run's peak of the one pool (receipt.json)"},
                "host_ram": {"peak_gib": None,
                             "why_none": "one pool: the run's peak is torch_memory's"}}
    return {
        "gpu_memory": {"used_gib": certified.to_gib(main["vram_mib"], "MiB") if cuda else None,
                       "source": main["vram_source"],
                       "what": "the GPU memory the run's processes used, load and training, as "
                               "nvidia-smi shows it"},
        "torch_memory": {"reserved_peak_gib": certified.to_gib(
                             receipt.get("peak_vram_reserved_gb"), "GB"),
                         "allocated_peak_gib": certified.to_gib(gb, "GB"),
                         "what": "torch's caching allocator in training (receipt.json)"},
        "host_ram": {"peak_gib": certified.to_gib(main["host_kb"], "KiB")
                     if main["host_kb"] else None,
                     "what": "RssAnon + RssShmem of the process tree, sampled 4 times a second",
                     "samples": main["samples"]},
    }


def _tail(path: Path, n: int = 8) -> List[str]:
    try:
        return path.read_text(errors="replace").splitlines()[-n:]
    except OSError:
        return []


def certify(a: argparse.Namespace) -> int:
    r = next((x for x in presets.BUILT_IN if x.key == a.recipe), None)
    if r is None:
        raise SystemExit(f"{a.recipe} is not a built-in recipe: "
                         f"{', '.join(x.key for x in presets.BUILT_IN)}")
    out = a.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    work = (a.work or out.parent / f".{out.name}-work").resolve()
    home = (a.home or work / "home").resolve()
    # what this run may remove at the end; a --home you name is kept (its SSD store is reused)
    for folder in (work, home):  # the run removes what it made there: a receipt must be outside
        if out == folder or out.is_relative_to(folder):
            raise SystemExit(f"--out is inside {folder}, which the run tidies after itself: put "
                             "the receipt elsewhere")
    made = claim(work, WORK_MARK)
    kept_home = claim(home, HOME_MARK)
    if a.home is None:
        made += kept_home
    locks = [hold(work)] + ([hold(home)] if home != work else [])
    try:
        return _certify(a, r, out, work, home, made)
    finally:
        for lock in locks:
            lock.unlink(missing_ok=True)


def _certify(a: argparse.Namespace, r: presets.Recipe, out: Path, work: Path, home: Path,
             made: List[Path]) -> int:
    out.mkdir(parents=True, exist_ok=True)
    logs = work / "logs"
    if not logs.exists():
        made.append(logs)
    logs.mkdir(exist_ok=True)
    if a.simulate:  # here too: this machine is then the preview's card
        os.environ["POROS_DEV_SIMULATE"] = "1"
    env = {**os.environ, "POROS_HOME": str(home), "POROS_PROGRESS": "0"}
    simulated = env.get("POROS_DEV_SIMULATE", "") not in ("", "0")
    cli = Cli(a.poros or [sys.executable, "-c", "from poros.cli.main import cli; cli()"], env,
              logs)
    forward_term(cli)
    commit, clean = commit_of(a.commit, a.manifest)
    name_places(work=work, home=home, out=out, repo=tree_root())
    free_gb = shutil.disk_usage(home).free / 1e9
    if free_gb < a.disk_floor:
        raise SystemExit(f"{home} has {free_gb:.1f} GB free; this recipe needs {a.disk_floor:g} "
                         "GB there (its sources.toml entry's disk_gb, or --disk-floor)")
    machine = machine_facts(simulated)
    cuda = machine["platform"] == "cuda" and not simulated
    st = {**presets.loop_preset(r), **r.settings}  # what the run takes where it says nothing
    seq = int(st["max_seq_length"])
    per_update = int(st.get("per_device_batch_size", 1)) * int(
        st.get("gradient_accumulation_steps", 1))
    split = float(st.get("eval_split", 0.05))
    rows = sample_count(a.updates, per_update, split)
    kind = sample_kind(r)
    data = work / f"sample-{kind}.jsonl"
    if not data.exists():
        made.append(data)
    data_sha = write_rows(data, sample_rows(kind, rows, seq, key=r.key))
    held: Dict[str, Any] = {"held": None, "why": "not asked (--check-fit)"}
    eval_file: Optional[Path] = None
    if not a.check_fit:
        from poros.cli.context import Context

        held = held_here(r, Context(), data, a.updates)
        if held["held"] is not None and held["split"] != split:  # train's share, not the guess
            split = held["split"]
            rows = sample_count(a.updates, per_update, split)
            data_sha = write_rows(data, sample_rows(kind, rows, seq, key=r.key))
            held = held_here(r, Context(), data, a.updates)
        if a.eval and held["held"] is None:  # before anything trains: nothing to score is no run
            raise SystemExit(f"{r.key}: its sources entry plans an eval, and train cannot say "
                             f"what the run holds out ({held['why']})")
        if a.eval and held["held"] == 0:  # as the docs tell a user: poros eval <run> --data
            eval_file = work / f"eval-{kind}.jsonl"
            if not eval_file.exists():
                made.append(eval_file)
            write_rows(eval_file, sample_rows(kind, max(2, held_count(rows, 0.05)), seq,
                                              seed=1, key=r.key))
    if a.check_fit:  # after the data lands: what train says here, and nothing trains
        from poros.cli.context import Context

        found = check_fit(r, Context(), data, a.updates)
        found.update(schema="poros.certify.check_fit.v1", commit=commit, machine=machine,
                     simulated=simulated)
        (out / "check-fit.json").write_text(scrub(json.dumps(found, indent=1)) + "\n")
        ok = not found["doubts"] and not found["refused"]
        _say(f"check-fit {'ok' if ok else 'FLAGGED: ' + ', '.join(found['doubts']) or 'refused'}"
             f": {out / 'check-fit.json'}")
        if not a.keep_work:
            tidy(made)
        return 0 if ok else 1
    _say(f"certify {r.key} on {machine['gpu']}: {a.updates} updates, {rows} {kind} rows")

    head = {"schema": SCHEMA, "recipe": r.key, "commit": commit, "tree_clean": clean,
            "simulated": simulated, "machine": machine,
            "plan": {"updates": a.updates, "resume": a.resume, "eval": a.eval,
                     "machine": a.machine}}
    at = {"phase": "prepare", "log": "prepare.log"}
    try:
        return _phases(a, r, out, cli, home, logs, data, data_sha, kind, rows, seq, per_update,
                       held, eval_file, commit, clean, simulated, machine, cuda, made, head, at)
    except Exception as exc:  # a phase that broke the driver itself: said, as every failure is
        failure_receipt(out, head, logs, at["phase"], f"{type(exc).__name__}: {exc}", at["log"])
        if not a.keep_work:
            tidy(made)
        return 1


def _phases(a: argparse.Namespace, r: presets.Recipe, out: Path, cli: Cli, home: Path,
            logs: Path, data: Path, data_sha: str, kind: str, rows: int, seq: int,
            per_update: int, held: Dict[str, Any], eval_file: Optional[Path], commit: str,
            clean: Any, simulated: bool, machine: Dict[str, Any], cuda: bool,
            made: List[Path], head: Dict[str, Any], at: Dict[str, str]) -> int:
    """Prepare, the main run, the stop and continue, the eval, the receipt: each phase that
    fails ends the run there, with a receipt that names it (failure_receipt)."""
    def stop(phase: str, why: str, log: str, refused: bool = False) -> int:
        failure_receipt(out, head, logs, phase, why, log, refused)
        if not a.keep_work:
            tidy(made)
        return 1

    if a.prepare:
        _say("prepare: one update, not measured")
        rc, said = cli.run(train_args(r, data, 1, f"cert-{r.key}-prepare", a.download), "prepare")
        if rc != 0:
            refused = refusal_of(said)
            return stop("prepare", refused or f"its one update ended with exit code {rc}",
                        "prepare.log", refused=refused is not None)
    at.update(phase="main", log="main.log")
    _say("main run")
    main = main_run(cli, train_args(r, data, a.updates, f"cert-{r.key}", a.download), home,
                    f"cert-{r.key}", cuda)
    if main["folder"] is None:
        refused = refusal_of(main["result"])
        return stop("main", refused or f"train started no run (exit code {main['rc']})",
                    "main.log", refused=refused is not None)
    steps = step_events(main["folder"])
    times = [float(s["s_per_step"]) for s in steps if isinstance(s.get("s_per_step"),
                                                                   (int, float))]
    resolved = json.loads((main["folder"] / "config.resolved.json").read_text()) if (
        main["folder"] / "config.resolved.json").is_file() else {}
    recipe_loop = not simulated and resolved.get("trained_with") not in (None, "classic")
    tokens = per_update_tokens(steps, recipe_loop)
    resume: Dict[str, Any] = {"adapter_equal": None,
                              "note": "not run: its sources entry says resume = false"}
    if a.resume:
        at.update(phase="resume", log="resume-continue.log")
        _say("resume run: stopped halfway, then continued")
        res = resume_run(cli, train_args(r, data, a.updates, f"cert-{r.key}-resume", False),
                         home, f"cert-{r.key}-resume", a.updates)
        same = same_adapter(main["folder"], res["folder"], simulated)
        stopped = res["first_rc"] != 0 and res["stopped_after"] < a.updates
        resume = {"stopped": stopped, "stopped_after": res["stopped_after"],
                  "stop_rc": res["stop_rc"],
                  "first_rc": res["first_rc"], "continue_rc": res["continue_rc"],
                  "continued": res["continued"], **same,
                  # a run that finished before the stop took was not resumed: no pass
                  "adapter_equal": same["adapter_equal"] if res["continued"] and stopped
                  else False}
    ev_rc, ev, ok_eval = None, None, True
    if a.eval:
        at.update(phase="eval", log="eval.log")
        _say("eval")
        ev_rc, ev = cli.run(["eval", main["folder"].name, "--json",
                             *(["--data", str(eval_file)] if eval_file else [])], "eval")
        scored = eval_scored(ev)
        ok_eval = (ev_rc == 0 and bool(ev) and ev.get("schema") == "poros.eval.v1"
                   and scored["has_metric"] and scored["rows_from"] != "trained")
    receipt = json.loads((main["folder"] / "receipt.json").read_text()) if (
        main["folder"] / "receipt.json").is_file() else {}
    ran_held = held_of_run(resolved, rows) if resolved else None
    cert = {
        "schema": SCHEMA, "recipe": r.key, "commit": commit, "tree_clean": clean,
        "poros": receipt.get("poros"), "simulated": simulated, "machine": machine,
        "command": ["poros", *train_args(r, Path(f"<{kind} sample>"), a.updates,
                                         f"cert-{r.key}", a.download)],
        "data": {"kind": kind, "rows": rows, "held_out": ran_held,
                 "held_out_planned": held["held"],
                 "held_out_rule": "the run's own resolved eval_split and eval_rows "
                                  "(config.resolved.json), by poros.data's rule",
                 "rows_rule": f"at least {FEWEST} rows, by poros.data's rule, and enough to "
                              f"train {a.updates} updates",
                 "sha256": data_sha, "seq_len": seq, "tokens_per_update": tokens,
                 "tokens_are": "each update's own" if recipe_loop else "a running count",
                 # every row runs past the length (sample_rows), so each is cut to it; the
                 # events' count can leave out a row's prompt and template tokens
                 "rows_full": bool(tokens) and min(tokens) >= (seq - ROW_ALLOWANCE) * per_update,
                 "rows_full_rule": f"each update's tokens >= (length - {ROW_ALLOWANCE}) x rows"},
        "main": {"rc": main["rc"], "updates": len(times), "s_per_update": times,
                 "adapter_sha256": adapter_sha(main["folder"]),
                 "s_per_step": round(statistics.median(times[1:]), 4) if len(times) >= 2
                 else None, "s_per_step_rule": "median of updates 2..N"},
        # two figures, by name, never one for the other: what "will it fit" reads (the GPU
        # memory the process tree used, as nvidia-smi shows it, load and training, the CUDA
        # context included), and torch's caching allocator's peaks in training
        **memory_record(machine["platform"], main, receipt, cuda),
        # what the preflight printed of the fit, as it said it: recorded, never worked around
        "preflight_said": [line.strip() for line in
                           (logs / "main.log").read_text(errors="replace").splitlines()
                           if any(w in line.lower() for w in ("may not fit", "not to fit",
                                                               "won't fit", "memory is short"))
                           ][:20],
        "plan": {"updates": a.updates, "resume": a.resume, "eval": a.eval,
                 "machine": a.machine},
        "resume": resume,
        "eval": {"status": ("ok" if ok_eval else "failed") if a.eval else "not run", "rc": ev_rc,
                 "metrics": (ev or {}).get("metrics"), "verdict": (ev or {}).get("verdict"),
                 **(eval_scored(ev) if a.eval else {}),
                 "data": "a file of rows it did not train on (the run held none out)"
                 if eval_file else "the rows the run held out" if a.eval else None},
    }
    held_ok = held["held"] is None or ran_held == held["held"]
    ok = (main["rc"] == 0 and len(times) == a.updates and ok_eval and held_ok
          and resume.get("adapter_equal") is not False)
    cert["status"] = "ok" if ok else "failed"
    if not ok:  # which phase, and what its log said last
        failed = [("main", "main.log", main["rc"] != 0 or len(times) != a.updates,
                   f"exit code {main['rc']}, {len(times)} of {a.updates} updates"),
                  ("data", "main.log", not held_ok,
                   f"it held out {ran_held} rows; train said {held['held']} before it started"),
                  ("resume", "resume-continue.log", resume.get("adapter_equal") is False,
                   "the stopped and continued run did not end at the main run's adapter"),
                  ("eval", "eval.log", not ok_eval,
                   f"exit code {ev_rc}; a metric: {scored['has_metric'] if a.eval else None}; "
                   f"rows: {scored['rows_from'] if a.eval else None}")]
        cert["failed"] = [{"phase": phase, "reason": why,
                           "log_tail": [scrub(x) for x in _tail(logs / log)]}
                          for phase, log, bad, why in failed if bad]
    (out / "main").mkdir(exist_ok=True)
    for f in ("receipt.json", "events.jsonl", "config.resolved.json", "action.json"):
        src = main["folder"] / f
        if src.is_file():
            (out / "main" / f).write_text(scrub(src.read_text()))
    keep_logs(out, logs)
    (out / "commit.txt").write_text(f"{commit}\n")
    at.update(phase="receipt", log="main.log")
    left = [] if a.keep_work else tidy(made)
    if left:  # what it made and could not remove: the measurement stands; the operator removes it
        cert["tidy"] = {"status": "failed", "reason": "could not remove what the run made",
                        "left": [scrub(x) for x in left]}
    (out / "certification.json").write_text(scrub(json.dumps(cert, indent=1)) + "\n")
    _say(f"{cert['status']}: {out / 'certification.json'}")
    if left:
        _say("TIDY FAILED: the run could not remove these; remove them by hand:")
        for line in left:
            _say(f"  {line}")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recipe", nargs="?", help="a built-in recipe's name (poros recipes lists "
                    "them; any word with --manifest-of)")
    ap.add_argument("--out", type=Path, help="an empty folder for the receipt (needed for a run)")
    ap.add_argument("--machine", help="which of the recipe's sources.toml entries this is (the "
                    "only one, when it has one): it says the updates and the phases")
    ap.add_argument("--work", type=Path, help="sample data and logs (default: beside --out)")
    ap.add_argument("--home", type=Path, help="POROS_HOME for the runs (default: in --work); an "
                    "SSD store is built here")
    ap.add_argument("--commit", help="the commit installed, where poros is a copy of the tree, "
                    "not a git checkout (with --manifest)")
    ap.add_argument("--manifest", type=Path, help="that commit's file list (sha256sum format; "
                    "--manifest-of a git archive of it): every file of the copy must match")
    ap.add_argument("--manifest-of", type=Path, metavar="TREE",
                    help="print TREE's file list in that format, and do nothing else")
    ap.add_argument("--disk-floor", type=float, help="GB that must be free where the runs go "
                    "(default: the recipe's sources.toml entry's disk_gb)")
    ap.add_argument("--poros", nargs="+", help="the poros command (default: this Python's)")
    ap.add_argument("--download", action="store_true", help="let train download the model")
    ap.add_argument("--no-prepare", dest="prepare", action="store_false")
    ap.add_argument("--sources", type=Path, default=SOURCES)
    ap.add_argument("--simulate", action="store_true",
                    help="a preview with the CLI's stand-in trainer: the receipt says simulated, "
                         "and the generator refuses it")
    ap.add_argument("--check-fit", action="store_true",
                    help="no training: what the start of poros train says for the recipe "
                         "here, saved as check-fit.json (after the data lands)")
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args(argv)
    if a.manifest_of:
        for name, sha in manifest_of(a.manifest_of).items():
            print(f"{sha}  {name}")
        return 0
    missing = [n for n, v in (("recipe", a.recipe), ("--out", a.out)) if v is None]
    if missing:  # a run needs both; --manifest-of neither
        ap.error(f"the following arguments are required: {', '.join(missing)}")
    try:
        plan = planned(a.recipe, a.machine, a.sources)
    except ValueError as exc:
        ap.error(str(exc))
    a.machine, a.updates, a.resume, a.eval = (plan["machine"], plan["updates"], plan["resume"],
                                              plan["eval"])
    if a.disk_floor is None:
        a.disk_floor = plan["disk_gb"]
    return certify(a)


def planned(recipe: str, machine: Optional[str], sources: Path = None) -> Dict[str, Any]:
    """The recipe's sources.toml entry for ``machine`` (or its only one): its machine, measured
    updates (default 16, at least 4: the time is the median of updates 2..N, and the resume stops
    halfway) and whether the resume and the eval run."""
    import tomllib

    found = [c for c in tomllib.loads((sources or SOURCES).read_text()).get("cert", [])
             if c["recipe"] == recipe and machine in (None, c.get("machine"))]
    if len(found) != 1:
        raise ValueError(f"{recipe}{' on ' + machine if machine else ''}: "
                         f"{'no' if not found else 'more than one'} sources.toml entry "
                         "(--machine names one)")
    c = found[0]
    updates = int(c.get("updates", 16))
    if updates < 4:
        raise ValueError(f"{recipe}: updates {updates} is under 4")
    return {"machine": c["machine"], "updates": updates, "resume": bool(c.get("resume", True)),
            "eval": bool(c.get("eval", True)), "disk_gb": float(c.get("disk_gb", 0))}


if __name__ == "__main__":
    sys.exit(main())
