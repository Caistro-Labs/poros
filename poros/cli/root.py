"""The Typer app every ``poros`` command registers on (``app``, ``bench_app``, the top-level
callback), and what the commands share: the ``--json`` option, a config file's loading and the
grants it cannot give, the benchmark results loader, and the torch and Mac guards.

The commands themselves are ``poros.cli.commands`` (importing that package registers every one,
in the order ``--help`` lists them); the console-script entry is ``poros.cli.main.cli``.
Torch-free.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Optional

# transformers' import-time advice ("None of PyTorch ... found") would print
# over a screen; Poros reports a missing torch itself, as a check with a fix
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# the hub's own tqdm bars draw full-width over the screen; Poros shows its
# downloads as an activity line instead
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

import typer  # noqa: E402

from poros.cli import contract, theme  # noqa: E402
from poros.cli.contract import refuse  # noqa: E402
from poros.core.env import quiet_libraries  # noqa: E402
from poros.links import DOCS_URL  # noqa: E402

if TYPE_CHECKING:  # torch-free at runtime; PorosConfig is imported per-command
    from poros.core.config import PorosConfig

# bitsandbytes' CPU-kernel advice and the Hub's unauthenticated notice: off the screen
# (poros.core.env.quiet_libraries, which `import poros` runs too, for the Python API)
quiet_libraries()

# Brand the --help rendering before any command can print it: typer's
# defaults are cyan, which is off-palette everywhere Poros appears.
theme.apply_typer_help_theme()

# NOTE: no_args_is_help is handled manually in the main callback below so
# that a bare `poros` on a real terminal opens the interactive shell, while
# every non-TTY invocation (pipes, CI, agents) prints the help and exits 0:
# never an animation, never a prompt.
app = typer.Typer(
    name="poros",
    help=(
        "Fine-tune models bigger than your GPU.\n\n"
        "A command, then a name and flags: poros train Qwen2.5-0.5B-Instruct --data chats.jsonl. "
        "In a terminal each command opens its settings first, and poros alone opens the "
        "session. poros <command> --help says what one takes.\n\n"
        "The docs: " + DOCS_URL + ", and offline from this install: poros docs (every command: "
        "poros docs cli). For coding agents: poros agents."
    ),
    rich_markup_mode="rich",
)


bench_app = typer.Typer(
    name="bench",
    help="Benchmark harness: run, matrix, compare, report.",
    no_args_is_help=True,
)
app.add_typer(bench_app, name="bench")


@bench_app.callback()
def _bench() -> None:
    _not_on_a_mac("poros bench")
    # Benchmark commands write their results as RunArtifact files; they have
    # no single result object yet, so JSON mode refuses rather than print none.
    if contract.json_mode():
        refuse("E_USAGE", "Can't run that",
               "poros bench does not support JSON mode yet; it writes "
               "RunArtifact files (poros schema artifact) under its output directory.")


def _json_flag(value: bool) -> bool:
    if value:
        contract.force_json()
    return value


# One --json for every command: one JSON object on stdout, logs on stderr.
_JSON_OPTION = typer.Option(
    False, "--json", callback=_json_flag, is_eager=True,
    help="Print one JSON object (poros schema <command>) instead of the formatted view.",
)


def load_run_artifacts(path: Path) -> list:
    """Load RunArtifacts from a file OR a directory, JSON or JSONL.

    Accepts, in order of preference:
      - a single ``.json`` RunArtifact file, or a ``.jsonl`` file (one row per line);
      - a directory containing ``*.json`` artifacts (recursing if none at top level);
      - a directory containing ``*.jsonl`` files (e.g. ``poros_grid_pro.jsonl``).

    This is the single loader shared by ``report``, ``leaderboard``, ``bench
    compare`` and ``bench report`` so every command accepts the same inputs --
    an agent can point any of them at a file or a directory and it just works.
    """
    import json as _json

    from poros.bench.artifacts import RunArtifact, load_grid_pro_row, read_artifact

    artifacts: list[RunArtifact] = []
    skipped: list[str] = []

    def _warn_skipped() -> None:
        # A dropped row is a missing benchmark cell; a report that quietly
        # renders without it reads as "complete". Say what was left out.
        if skipped:
            from poros.cli import views

            _ui()[0].print(views.note(
                f"Skipped {len(skipped)} non-artifact row(s)/file(s)",
                f"first: {skipped[0]}; the report may be missing cells"))

    def _ingest_jsonl(p: Path) -> None:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            skipped.append(str(p))
            return
        for i, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            # Canonical RunArtifact first (real bench output), then the legacy
            # grid-pro row shape. A line matching neither is skipped rather than
            # turned into a zero-filled artifact.
            try:
                artifacts.append(RunArtifact.model_validate_json(line))
                continue
            except Exception:
                pass
            try:
                artifacts.append(load_grid_pro_row(_json.loads(line)))
            except Exception:
                skipped.append(f"{p}:{i}")

    if path.is_file():
        if path.suffix == ".jsonl":
            _ingest_jsonl(path)
        else:
            try:
                artifacts.append(read_artifact(path))
            except Exception:
                skipped.append(str(path))
        _warn_skipped()
        return artifacts

    # Directory: JSON artifacts first (top level, then recursive), else JSONL.
    json_files = sorted(path.glob("*.json")) or sorted(path.glob("**/*.json"))
    for af in json_files:
        try:
            artifacts.append(read_artifact(af))
        except Exception:
            pass  # skip non-artifact JSON files
    if not artifacts:
        # Top level first, else recursive ("**" re-matches top-level files, so
        # never concatenate the two — that double-counts every artifact).
        for jf in sorted(path.glob("*.jsonl")) or sorted(path.glob("**/*.jsonl")):
            _ingest_jsonl(jf)
    _warn_skipped()
    return artifacts


def _ui():
    """The output sink, the grammar and the width (see poros/cli/ui/term.py)."""
    from poros.cli.ui import components as uc
    from poros.cli.ui import term as _term

    t = _term.get()
    return t, uc, t.width()


def _no_artifacts(path: Path) -> NoReturn:
    refuse("E_NO_RESULTS", "No benchmark artifacts", f"nothing to read in {path}",
           context=["expected a directory of RunArtifact JSON files, or a .json/.jsonl results "
                    "file", "training runs write events.jsonl, which is not report input"])


def _version_callback(value: bool) -> None:
    """``--version``: the document ``poros 1.0.0`` (POROS_OUTPUT=json too: a document prints as
    it is); with ``--json`` on the line, the result ``poros.version.v1``, as ``poros agents
    version`` prints it."""
    if value:
        from poros import __version__

        if "--json" in sys.argv[1:]:
            from poros.cli.schemas import VersionOutput

            contract.force_json()
            contract.emit(VersionOutput(version=__version__))
        else:
            print(f"poros {__version__}", file=contract.stdout())
        raise typer.Exit()


def _load_yaml_config(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict.

    Falls back gracefully if the file is not valid YAML.
    """
    import yaml

    if not path.exists():
        refuse("E_NOT_FOUND", "Not found", f"no config file at {path}",
               fix="poros init", note="writes a starter train.yaml")
    if path.is_dir():
        refuse("E_USAGE", "Not a config file", f"{path} is a folder, not a YAML config file",
               fix="poros init", note="writes a starter train.yaml")
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except UnicodeDecodeError:
        refuse("E_USAGE", "Not a config file", f"{path} is not UTF-8 text: a config is a YAML "
               "file", fix="poros init", note="writes a starter train.yaml")
    except OSError as exc:  # a file this user may not read
        import os

        why = os.strerror(exc.errno) if isinstance(exc.errno, int) else type(exc).__name__
        refuse("E_USAGE", "Can't read the config", f"{path}: {why.lower()}")
    except Exception as exc:
        refuse("E_USAGE", "Invalid YAML", f"{path} does not parse",
               context=[str(exc).splitlines()[0]])
    if data is None:
        refuse("E_USAGE", "Invalid config", f"{path} is empty: a config names a model, a "
               "dataset and an output folder", fix="poros init",
               note="writes a starter train.yaml")
    if not isinstance(data, dict):
        refuse("E_USAGE", "Invalid config", f"{path} holds a YAML {type(data).__name__}, where "
               "a config is a mapping of fields (model_name_or_path: ...)",
               fix="poros schema config", note="lists every field")
    return data


def _no_grants(raw: dict, config: Path, *, flagged: bool = False) -> None:
    """A config file comes with a project, so it cannot say to run anyone's code: a repo's takes
    --trust-remote-code (``flagged``), a loop file --recipe. A job's own config is Poros's, under
    POROS_HOME, and holds what its submitter's flags said."""
    from poros import runs

    jobs = Path(os.path.abspath(runs.home() / "jobs"))
    said = Path(os.path.abspath(config))
    if (os.environ.get("POROS_JOB") == "1"  # the job's own runner (poros.jobs.launch sets it)
            and said.name == "config.yaml" and said.parent.parent == jobs
            and not said.is_symlink() and not said.parent.is_symlink()
            and said.resolve().parent.parent == jobs.resolve()):
        return
    if raw.get("trust_remote_code") and not flagged:
        refuse("E_CONFIG", "A flag, not a setting",
               "trust_remote_code in a config file grants nothing",
               context=["for a base you trust: train or eval --trust-remote-code"])
    if raw.get("loop"):
        refuse("E_CONFIG", "A flag, not a setting", "loop in a config file grants nothing",
               context=["poros train <model> --data <file> --recipe your_loop.py"])


def _adapter_revision(cfg: "PorosConfig") -> None:
    """A revision read out of an adapter's own files is the adapter's word: the same rule."""
    from poros.core.config import no_pull_request

    try:
        no_pull_request(cfg.model_revision)
    except ValueError as exc:
        refuse("E_CONFIG", "Not that revision", str(exc))


def _require_torch(feature: str = "This command", *, import_it: bool = True) -> None:
    """Assert torch is importable, or exit with a friendly message. ``import_it=False``: only
    that it is installed (no import: what refuses before it stays torch-free)."""
    try:
        if import_it:
            import torch  # noqa: F401
        else:
            import importlib.util

            if importlib.util.find_spec("torch") is None:
                raise ImportError("torch")
    except ImportError:
        refuse("E_NO_TORCH", "Needs PyTorch",
               f"{feature} needs PyTorch and the training libraries, which are not installed here",
               fix="pip install poros-train",
               note="torch, transformers, peft" if sys.platform == "darwin"
               else "torch, transformers, peft, bitsandbytes")


def _not_on_a_mac(what: str, hint: str = "poros machine says what this Mac can do.") -> None:
    if sys.platform == "darwin":
        refuse("E_GUARANTEE_UNAVAILABLE", "Error", f"{what} does not run on a Mac in this release.",
               fix=hint)


def _click_module():
    """The click module typer actually uses (vendored on some versions).

    Only its exception classes are read from here. ``Exit``/``Abort`` are
    taken from typer itself: typer >= 0.27.2 defines them outside its vendored
    click, and ``typer.Exit``/``typer.Abort`` name the right classes on
    every typer.
    """
    try:
        from typer.core import _click as click_mod
    except ImportError:
        import click as click_mod
    return click_mod


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        help="Print version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
    plain: bool = typer.Option(
        False,
        "--plain",
        "--no-anim",
        help="Skip the banner animation when opening the interactive shell.",
    ),
    theme: Optional[str] = typer.Option(
        None,
        "--theme",
        help="Color theme: auto (default), dark, or light.",
    ),
    json_output: bool = _JSON_OPTION,
) -> None:
    """Fine-tune large models on small GPUs."""
    contract.reset()  # JSON mode is per invocation (the shell runs many)
    if theme is not None:
        from poros.cli import theme as _theme

        _theme.set_mode(theme)

    if ctx.invoked_subcommand is not None:
        return  # a real subcommand: never greet, never animate
    if contract.json_mode():
        refuse("E_USAGE", "Can't run that", "no command given. poros --help lists them.")

    # Interactive shell ONLY when BOTH ends are real TTYs. Piped/redirected/
    # CI/agent invocations keep the exact historical no-args behavior below.
    if sys.stdin.isatty() and sys.stdout.isatty():
        from poros.cli.shell import run_shell

        run_shell(animate=not plain)
        raise typer.Exit()

    # No terminal (a pipe, CI, a script): the help, on stdout, and exit 0, as `poros --help`.
    # Never the session, an animation or a prompt; with --json it is the E_USAGE above.
    ctx.get_help()  # typer prints its help itself
    raise typer.Exit()


