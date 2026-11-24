"""Suggestions for the interactive shell: tab completion + inline ghost text.

Two sources, fish-style:

1. **History autosuggest**: if an earlier command line starts with what you've
   typed, the rest of the most recent one is offered as ghost text.
2. **Completion** for the token under the caret, derived from the real Typer
   command tree (never a hand-kept list), so commands, subcommands and flags
   can't drift from the CLI. Values come from the machine:
   models in the Hugging Face cache, local model dirs, YAML configs, datasets,
   checkpoints, output dirs.

Pure logic over strings and the filesystem: no tty, torch-free, fast. Every
filesystem scan is bounded and cached for a few seconds.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Candidate:
    text: str
    desc: str = ""
    kind: str = ""  # command | flag | value | path


@dataclass
class Completion:
    start: int  # buffer index where the token being completed begins
    token: str
    candidates: List[Candidate] = field(default_factory=list)
    ghost: str = ""  # text to insert at the caret if accepted (may be "")
    source: str = ""  # "history" | "complete" | ""

    @property
    def best(self) -> Optional[Candidate]:
        return self.candidates[0] if self.candidates else None


# --------------------------------------------------------------------------
# the command tree (introspected from Typer, cached)
# --------------------------------------------------------------------------

SHELL_WORDS: Dict[str, str] = {
    "help": "Show commands (or one command's help)",
    "settings": "Theme, motion and what Poros remembers",
    "theme": "Switch color theme",
    "version": "Show the Poros version",
    "clear": "Clear the screen",
    "quit": "Exit the shell",
}
THEME_VALUES = ("auto", "dark", "light")
SCHEMA_KINDS = ("config", "events", "artifact")


@dataclass
class Param:
    opts: Tuple[str, ...]
    help: str
    is_flag: bool
    metavar: str = ""


@dataclass
class Cmd:
    name: str
    help: str
    params: List[Param] = field(default_factory=list)
    args: List[str] = field(default_factory=list)  # positional names, in order
    sub: Dict[str, "Cmd"] = field(default_factory=dict)


_TREE: Optional[Dict[str, Cmd]] = None


def _first_line(s: Optional[str]) -> str:
    return (s or "").strip().splitlines()[0].rstrip(".") if (s or "").strip() else ""


def _walk(click_cmd, name: str) -> Cmd:
    # Duck-typed on purpose: some typer versions vendor their own click, so
    # isinstance against the installed click module would see nothing.
    node = Cmd(
        name=name,
        help=_first_line(getattr(click_cmd, "help", "") or getattr(click_cmd, "short_help", "")),
    )
    for p in getattr(click_cmd, "params", []):
        kind = getattr(p, "param_type_name", "")
        if kind == "option":
            if getattr(p, "hidden", False):
                continue
            node.params.append(
                Param(
                    tuple(p.opts) + tuple(getattr(p, "secondary_opts", ())),
                    _first_line(getattr(p, "help", "")),
                    bool(getattr(p, "is_flag", False)),
                    getattr(p, "metavar", "") or "",
                )
            )
        elif kind == "argument":
            node.args.append(p.name or "")
    for sub_name, sub in (getattr(click_cmd, "commands", None) or {}).items():
        node.sub[sub_name] = _walk(sub, sub_name)
    return node


def _verbs() -> Dict[str, Cmd]:
    """The verbs, first in every list, with their own one-liners."""
    import importlib

    from poros.cli.app import BUILT

    out = {}
    for verb, module in BUILT.items():
        try:
            doc = importlib.import_module(module).__doc__ or ""
        except Exception:  # pragma: no cover
            doc = ""
        out[verb] = Cmd(name=verb, help=_first_line(doc).split(":", 1)[-1].strip())
    return out


def command_tree() -> Dict[str, Cmd]:
    """``{name: Cmd}`` for every CLI command, plus the shell's own words."""
    global _TREE
    if _TREE is None:
        tree: Dict[str, Cmd] = {}
        try:
            import typer

            from poros.cli.main import app

            root = _walk(typer.main.get_command(app), "poros")
            tree.update(root.sub)
        except Exception:  # pragma: no cover - a broken import must not kill the shell
            pass
        for w, h in SHELL_WORDS.items():
            tree.setdefault(w, Cmd(name=w, help=h))
        for name, verb in _verbs().items():
            if name in tree:  # train, export: keep the config form's flags and arguments
                tree[name].help = verb.help
            else:
                tree[name] = verb
        _TREE = tree
    return _TREE


# --------------------------------------------------------------------------
# value providers (bounded, cached)
# --------------------------------------------------------------------------

_CACHE: Dict[str, Tuple[float, List[Candidate]]] = {}
_TTL = 5.0


def _cached(key: str, fn: Callable[[], List[Candidate]]) -> List[Candidate]:
    now = time.monotonic()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    try:
        val = fn()
    except Exception:
        val = []
    _CACHE[key] = (now, val)
    return val


def _hf_cache_dir() -> Path:
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


# Ids the code itself names (init/config defaults, bench configs) -- the
# fallback when nothing is cached yet.
KNOWN_MODELS = (
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen3-32B",
    "google/gemma-3-27b-pt",
    "google/gemma-4-31B-it",
)


def _iterdir(p: Path, limit: int = 400) -> Iterable[Path]:
    try:
        for i, child in enumerate(p.iterdir()):
            if i >= limit:
                return
            yield child
    except OSError:
        return


def models() -> List[Candidate]:
    def scan() -> List[Candidate]:
        out: Dict[str, Candidate] = {}
        for d in _iterdir(_hf_cache_dir()):
            if d.name.startswith("models--"):
                mid = d.name[len("models--") :].replace("--", "/", 1)
                out[mid] = Candidate(mid, "cached", "value")
        for d in _iterdir(Path.cwd(), 200):
            if d.is_dir() and (d / "config.json").is_file():
                out.setdefault("./" + d.name, Candidate("./" + d.name, "local model", "value"))
        for mid in KNOWN_MODELS:
            out.setdefault(mid, Candidate(mid, "hub", "value"))
        return sorted(
            out.values(),
            key=lambda c: ({"cached": 0, "local model": 1}.get(c.desc, 2), c.text.lower()),
        )

    return _cached("models", scan)


_DATA_EXT = (".jsonl", ".json", ".csv", ".parquet", ".txt")


def datasets() -> List[Candidate]:
    def scan() -> List[Candidate]:
        out: List[Candidate] = []
        for root in (Path.cwd(), Path.cwd() / "data"):
            for f in _iterdir(root, 300):
                if f.is_file() and f.suffix in _DATA_EXT and f.name != "config.json":
                    out.append(Candidate(_rel(f), "local file", "value"))
        for d in _iterdir(_hf_cache_dir()):
            if d.name.startswith("datasets--"):
                out.append(
                    Candidate(d.name[len("datasets--") :].replace("--", "/", 1), "cached", "value")
                )
        out.append(Candidate("OpenAssistant/oasst1", "hub", "value"))
        return out

    return _cached("datasets", scan)


def _rel(p: Path) -> str:
    try:
        r = p.relative_to(Path.cwd())
        return str(r) if str(r).startswith(".") or "/" in str(r) else str(r)
    except ValueError:
        return str(p)


def _yaml_model(p: Path) -> str:
    try:
        with p.open() as fh:
            for i, line in enumerate(fh):
                if i > 40:
                    break
                s = line.strip()
                if s.startswith(("model_name_or_path:", "model:", "model_id:", "model_name:")):
                    return s.split(":", 1)[1].strip().strip("'\"")
    except OSError:
        pass
    return ""


def configs() -> List[Candidate]:
    def scan() -> List[Candidate]:
        out = []
        roots = [Path.cwd(), Path.cwd() / "configs"]
        for root in roots:
            for f in _iterdir(root, 300):
                if f.is_file() and f.suffix in (".yaml", ".yml"):
                    out.append(Candidate(_rel(f), _yaml_model(f), "path"))
        return sorted(out, key=lambda c: (c.text.count("/"), c.text))

    return _cached("configs", scan)


def output_dirs() -> List[Candidate]:
    """Run directories: anything holding events.jsonl, an adapter, or checkpoints."""

    def scan() -> List[Candidate]:
        out = []
        for root in (Path.cwd(), Path.cwd() / "runs"):
            for d in _iterdir(root, 300):
                if not d.is_dir() or d.name.startswith("."):
                    continue
                marks = [
                    m
                    for m in ("events.jsonl", "adapter_config.json", "receipt.json")
                    if (d / m).exists()
                ]
                ckpts = [c for c in _iterdir(d, 200) if c.name.startswith("checkpoint-")]
                if marks or ckpts:
                    desc = (
                        (f"{len(ckpts)} checkpoint" + ("s" if len(ckpts) != 1 else ""))
                        if ckpts
                        else marks[0]
                    )
                    out.append(Candidate(_rel(d), desc, "path"))
        return out

    return _cached("outputs", scan)


def checkpoints() -> List[Candidate]:
    out = [Candidate("latest", "newest checkpoint in --output-dir", "value")]
    for run in output_dirs():
        for c in _iterdir(Path(run.text), 200):
            if c.name.startswith("checkpoint-"):
                out.append(Candidate(_rel(c), "", "path"))
    return out


def paths(token: str, dirs_only: bool = False) -> List[Candidate]:
    base = Path(token).expanduser()
    parent = base if token.endswith("/") else base.parent
    stem = "" if token.endswith("/") else base.name
    out = []
    for c in _iterdir(parent if str(parent) else Path("."), 400):
        if c.name.startswith(".") and not stem.startswith("."):
            continue
        if not c.name.startswith(stem):
            continue
        if dirs_only and not c.is_dir():
            continue
        shown = os.path.join(str(parent) if token and str(parent) != "." else "", c.name)
        if token.startswith("./") and not shown.startswith("./"):
            shown = "./" + shown
        out.append(Candidate(shown + ("/" if c.is_dir() else ""), "", "path"))
    return sorted(out, key=lambda c: c.text)[:64]


# option name -> provider; positional (command, index) -> provider
OPTION_VALUES: Dict[str, Callable[[], List[Candidate]]] = {
    "--model": models,
    "-m": models,
    "--dataset": datasets,
    "-d": datasets,
    "--resume-from": checkpoints,
    "-r": checkpoints,
    "--theme": lambda: [Candidate(v, "", "value") for v in THEME_VALUES],
}
DIR_OPTIONS = {"--output-dir", "-o", "--output", "--out"}

POSITIONAL: Dict[Tuple[str, int], Callable[[], List[Candidate]]] = {
    ("train", 0): configs,
    ("eval", 0): configs,
    ("init", 0): lambda: [Candidate("train.yaml", "new config", "path")],
    ("check", 0): models,
    ("export", 0): output_dirs,
    ("report", 0): output_dirs,
    ("leaderboard", 0): output_dirs,
    ("schema", 0): lambda: [Candidate(k, "", "value") for k in SCHEMA_KINDS],
    ("theme", 0): lambda: [Candidate(v, "", "value") for v in THEME_VALUES],
    ("help", 0): lambda: [Candidate(n, c.help, "command") for n, c in command_tree().items()],
}


# --------------------------------------------------------------------------
# ranking
# --------------------------------------------------------------------------


def _subseq(needle: str, hay: str) -> bool:
    it = iter(hay)
    return all(ch in it for ch in needle)


def rank(token: str, cands: Sequence[Candidate]) -> List[Candidate]:
    """Prefix (case-sensitive, then insensitive) > word-prefix > substring > subsequence."""
    if not token:
        return list(cands)
    t = token.lower()
    buckets: List[List[Candidate]] = [[], [], [], [], []]
    for c in cands:
        x = c.text.lower()
        tail = x.rsplit("/", 1)[-1]
        if c.text.startswith(token):
            buckets[0].append(c)
        elif x.startswith(t):
            buckets[1].append(c)
        elif tail.startswith(t):
            buckets[2].append(c)
        elif t in x:
            buckets[3].append(c)
        elif _subseq(t, x):
            buckets[4].append(c)
    return [c for b in buckets for c in b]


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _split(s: str) -> Tuple[List[str], int]:
    """Whitespace tokens of ``s`` and the index where the last token starts.
    A trailing space opens a new empty token."""
    toks: List[str] = []
    start = 0
    i = 0
    cur_start = 0
    quote = ""
    buf = ""
    while i < len(s):
        ch = s[i]
        if quote:
            if ch == quote:
                quote = ""
            else:
                buf += ch
        elif ch in "'\"":
            quote = ch
            if not buf:
                cur_start = i
        elif ch.isspace():
            if buf:
                toks.append(buf)
                buf = ""
            cur_start = i + 1
        else:
            if not buf:
                cur_start = i
            buf += ch
        i += 1
    toks.append(buf)
    start = cur_start if buf or not s or s[-1:].isspace() else len(s)
    return toks, start


def complete(buf: str, history: Sequence[str] = (), caret: Optional[int] = None) -> Completion:
    """Suggestions for ``buf`` (cut at ``caret``). Leading ``/`` and ``poros`` are allowed."""
    caret = len(buf) if caret is None else caret
    s = buf[:caret]

    # 1) fish-style history autosuggest (only with the caret at the end)
    if caret == len(buf) and s.strip():
        for h in reversed(history):
            if h.startswith(s) and h != s:
                comp = complete_tokens(s)
                comp.ghost = h[len(s) :]
                comp.source = "history"
                return comp
    comp = complete_tokens(s)
    if comp.best and comp.best.text.startswith(comp.token) and comp.best.text != comp.token:
        comp.ghost = comp.best.text[len(comp.token) :]
        if comp.best.kind in ("command", "flag") or (
            comp.best.kind == "value" and not comp.best.text.endswith("/")
        ):
            comp.ghost += (
                " " if comp.best.kind != "flag" or not _flag_takes_value(s, comp.best.text) else " "
            )
        comp.source = "complete"
    return comp


def _flag_takes_value(s: str, flag: str) -> bool:
    toks, _ = _split(s)
    cmd = _resolve_cmd([t for t in toks[:-1]])
    if cmd is None:
        return False
    for p in cmd.params:
        if flag in p.opts:
            return not p.is_flag
    return False


def _resolve_cmd(words: List[str]) -> Optional[Cmd]:
    tree = command_tree()
    words = [w for w in words if w]
    if words and words[0].lstrip("/") == "poros":
        words = words[1:]
    if not words:
        return None
    node = tree.get(words[0].lstrip("/"))
    if node and node.sub and len(words) > 1 and words[1] in node.sub:
        node = node.sub[words[1]]
    return node


def complete_tokens(s: str) -> Completion:
    toks, start = _split(s)
    token = toks[-1]
    before = toks[:-1]
    if before and before[0].lstrip("/") == "poros":
        before = before[1:]
    if not before:  # completing the command word itself
        slash = token.startswith("/")
        word = token.lstrip("/")
        if slash:  # the session's menu: the nine commands only
            from poros.cli.session.parts import MENU

            cands = [Candidate("/" + n, d, "command") for n, d in MENU]
        else:
            cands = [Candidate(n, c.help, "command") for n, c in command_tree().items()]
        return Completion(start, token, rank(("/" if slash else "") + word, cands))

    from poros.cli.app import BUILT
    from poros.cli.words.grammar import VERBS

    head = before[0].lstrip("/").lower()
    verb = head if head in VERBS else None
    legacy = verb in ("train", "eval") and any(t.endswith((".yaml", ".yml")) for t in before[1:])
    if verb in BUILT and not legacy:
        return Completion(start, token, sentence_tokens(verb, before, token) or [])
    root = command_tree().get(head)
    if root is None:
        return Completion(start, token, [])
    node = root
    rest = before[1:]
    if root.sub:
        if not rest:  # completing the subcommand
            return Completion(
                start,
                token,
                rank(token, [Candidate(n, c.help, "command") for n, c in root.sub.items()]),
            )
        if rest[0] in root.sub:
            node = root.sub[rest[0]]
            rest = rest[1:]

    # value for the previous option?
    if rest and rest[-1].startswith("-"):
        opt = rest[-1]
        p = next((p for p in node.params if opt in p.opts), None)
        if p is not None and not p.is_flag:
            if opt in OPTION_VALUES:
                return Completion(start, token, rank(token, OPTION_VALUES[opt]()))
            if opt in DIR_OPTIONS:
                return Completion(start, token, paths(token, dirs_only=True))
            return Completion(start, token, paths(token) if token else [])

    used = {o for t in rest if t.startswith("-") for o in [t]}
    flags = [
        Candidate(p.opts[0] if p.opts[0].startswith("--") else max(p.opts, key=len), p.help, "flag")
        for p in node.params
        if not (set(p.opts) & used) and "--help" not in p.opts
    ]
    flags.sort(key=lambda c: c.text)
    if token.startswith("-"):
        return Completion(start, token, rank(token, flags))

    # positional
    n_pos = _count_positionals(node, rest)
    provider = POSITIONAL.get(
        (root.name if node is root else f"{root.name} {node.name}", n_pos)
    ) or POSITIONAL.get((node.name, n_pos))
    if provider is not None:
        vals = rank(token, provider())
    elif token and ("/" in token or token.startswith(".")):
        vals = paths(token)
    else:
        vals = []
    return Completion(start, token, vals + ([] if token else flags))


def _count_positionals(node: Cmd, rest: List[str]) -> int:
    n = 0
    skip = False
    for t in rest:
        if skip:
            skip = False
            continue
        if t.startswith("-"):
            p = next((p for p in node.params if t in p.opts), None)
            skip = bool(p and not p.is_flag)
            continue
        n += 1
    return n


# --------------------------------------------------------------------------
# the verbs: a subject, then flags (poros.cli.words.grammar)
# --------------------------------------------------------------------------

VERB_FLAGS = {
    "train": ("--data", "--steps", "--epochs", "--seq-len", "--rank", "--block-size", "--output",
              "--name", "--recipe", "--from", "--eval", "--store", "--detach", "--yes", "--force",
              "--overwrite", "--allow-unvalidated", "--unvalidated", "--json"),
    "models": ("--yes", "--json"),
    "recipes": ("--output", "--name", "--json"),
    "arch": ("--json",),
    "runs": ("--follow", "--live", "--frame", "--wait", "--more", "--limit", "--copy", "--delete",
             "--yes", "--json"),
    "export": ("--format", "--output", "--overwrite", "--json"),
    "prove": ("--steps", "--seq-len", "--download", "--force", "--block-sizes", "--unvalidated",
              "--json"),
    "machine": ("--json",),
    "data": ("--convert", "--fix", "--output", "--json"),
    "eval": ("--data", "--checkpoint", "--rows", "--show", "--json"),
}
# The words after a verb that are its own action: `runs stop`, `models get`.
VERB_ACTIONS = {
    "runs": (("stop", "stop a run, with a checkpoint"), ("continue", "continue a stopped run"),
             ("report", "save a run's page as Markdown"),
             ("checkpoints", "what a run saved: export, copy or delete one")),
    "models": (("get", "download a model"),),
    "arch": (("try", "a tiny model of its config, against a dense reference"),),
    "recipes": (("copy", "fork a recipe's file here"), ("show", "one, and its train command"),
                ("save", "keep a finished run as your recipe"),
                ("share", "write a finished run's recipe file here"),
                ("sync", "fetch the shared recipes from GitHub")),
}


def catalog_names() -> List[Candidate]:
    from poros.cli.words.models import catalog

    return [Candidate(e.short, f"{e.size.upper()} {e.entry} · {e.evidence}", "value")
            for e in catalog()]


def run_ids() -> List[Candidate]:
    from poros import runs

    out = [Candidate("last", "the newest run", "value")]
    return out + [Candidate(r.id, f"{r.state} · {r.name}", "value") for r in runs.all_runs()[:20]]


def data_files() -> List[Candidate]:
    from poros.cli.verbs.train_data import local_files

    return [Candidate(f, "data here", "value") for f in local_files(Path.cwd())[:20]]


def _card_words() -> List[str]:
    """Every card `--gpu` reads, and the usual memory sizes (poros.core.card)."""
    from poros.core.card import card_words

    return card_words()


def _words(n: Sequence[str]) -> Callable[[], List[Candidate]]:
    return lambda: [Candidate(x, "", "value") for x in n]


FLAG_VALUES = {
    "--data": data_files,
    "--seq-len": _words(("256", "512", "1024", "2048", "4096")),
    "--rank": _words(("8", "16", "32", "64")),
    "--lora-dropout": _words(("0", "0.05", "0.1")),
    "--gpu": lambda: [Candidate(x, "", "value") for x in _card_words()],
    "--format": _words(("merged", "adapter")),
}


def sentence_tokens(verb: str, before: List[str], token: str) -> Optional[List[Candidate]]:
    """Candidates after a verb: a flag's value, the verb's action, its
    subject (a model or a run), else its flags."""
    last = before[-1] if before else ""
    if last in FLAG_VALUES:
        return rank(token, _cached(f"flag:{last}", FLAG_VALUES[last]))
    if last.startswith("-") and last not in BOOL_FLAGS_SET:
        return paths(token) if token else []
    flags = [Candidate(f, "", "flag") for f in VERB_FLAGS.get(verb, ()) if f not in before]
    if token.startswith("-"):
        return rank(token, flags)
    rest = [t for t in before[1:] if not t.startswith("-")]
    actions = VERB_ACTIONS.get(verb, ())
    if actions and not rest:
        cands = [Candidate(a, d, "command") for a, d in actions]
        if verb == "runs":
            cands += _cached("words:runs", run_ids)
        return rank(token, cands)
    has_subject = len(rest) > (1 if rest and rest[0] in dict(actions) else 0)
    if not has_subject and verb in ("runs", "export", "eval"):
        cands = _cached("words:runs", run_ids)
        return rank(token, configs() + cands if verb == "eval" else cands)  # `eval run.yaml` too
    if not has_subject and verb in ("train", "models", "prove"):
        cands = _cached("words:catalog", catalog_names)
        if verb == "train" and not rest:
            cands = configs() + cands  # `train run.yaml` still works
        return rank(token, cands)
    return rank(token, flags)


def _bool_flags():
    from poros.cli.words.grammar import BOOL_FLAGS

    return BOOL_FLAGS


BOOL_FLAGS_SET = _bool_flags()
