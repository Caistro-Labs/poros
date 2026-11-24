"""What a command's exception becomes: its contract code (``_error_code``) and the refusal a
person or an agent sees (``_friendly_error``). A refusal a command raises itself is
``poros.cli.contract.refuse``.
"""

from __future__ import annotations

import sys
from typing import NoReturn, Optional

import typer

from poros.cli import contract
from poros.core.errors import EXIT_CODES


def _error_code(exc: Exception) -> str:
    """The contract code for an exception (``poros.core.errors.EXIT_CODES``)."""
    import errno

    from poros.core.errors import PorosError

    if isinstance(exc, PorosError):
        return exc.code
    name, module = type(exc).__name__, type(exc).__module__
    if name == "OutOfMemoryError":  # torch.cuda.OutOfMemoryError, torch-free check
        return "E_OOM"
    if (hub := _hub_code(exc)) is not None:
        return hub
    if isinstance(exc, ImportError):
        return "E_DEPS"
    if isinstance(exc, OSError) and exc.errno == errno.ENOENT:
        return "E_NOT_FOUND"  # a file or folder the command was pointed at is not there
    if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EROFS, errno.EACCES,
                                                  errno.EPERM):
        return "E_DISK"  # no room, or none it may write
    if isinstance(exc, OSError) and exc.errno in (errno.ENAMETOOLONG, errno.EEXIST, errno.ENOTDIR,
                                                  errno.EISDIR):
        return "E_USAGE"  # a path the command was given: too long, or a file where a folder goes
    if sys.platform == "darwin" and isinstance(exc, OSError) and exc.errno == errno.EIO:
        return "E_DISK"  # a drive pulled during a run
    if disk_full(exc) is not None:  # no room, said in a library's own words (hf_xet's)
        return "E_DISK"
    if module.split(".")[0] in ("requests", "urllib3", "httpx", "huggingface_hub"):
        return "E_NETWORK"
    return "E_INTERNAL"


_NO_ROOM = ("No space left on device", "os error 28")


def disk_full(exc: BaseException) -> Optional[tuple[str, str]]:
    """(what filled the disk, the fix) where ``exc`` or anything in its cause chain is a disk
    with no room (ENOSPC, a quota, or a library's words for it); None otherwise."""
    import errno
    import os
    import shutil
    import traceback
    from pathlib import Path

    seen, chain, e = set(), [], exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        chain.append(e)
        e = e.__cause__ or e.__context__
    full = [e for e in chain if (isinstance(e, OSError) and e.errno in (errno.ENOSPC,
            errno.EDQUOT)) or any(w in str(e) for w in _NO_ROOM)]
    if not full:
        return None
    frames = [f.filename for e in chain for f in traceback.extract_tb(e.__traceback__)]
    named = next((e.filename for e in full if isinstance(e, OSError) and e.filename), None)
    if any(m in f for f in frames for m in ("huggingface_hub", "hf_xet")) and not named:
        from huggingface_hub import constants

        where, what = constants.HF_HUB_CACHE, "downloading the model into the Hugging Face cache"
        fix = "free space there, or set HF_HOME to a folder on a larger disk, then run it again"
    else:
        where = str(named) if named else os.getcwd()
        what = f"writing {named}" if named else "writing a file"
        fix = "free space there, then run it again"
    here = next((q for q in (Path(where), *Path(where).parents) if q.exists()), Path("/"))
    free = shutil.disk_usage(here).free / 1e9
    return f"The disk is full: it filled while {what} ({where}); {free:.1f} GB free there now.", fix


# What a load says when the Hub does: transformers wraps huggingface_hub's errors in an OSError
# of its own words, so each is known by its type's name, or by those words, down the causes.
_HUB_SAID = {
    "E_HUB_AUTH": "the model is gated on the Hugging Face Hub: accept its license on its page "
                  "at huggingface.co, then log in (hf auth login)",
    "E_NOT_FOUND": "the model, or a file of it, is not on the Hugging Face Hub: check its name; "
                   "a private repo answers the same until you log in (hf auth login)",
    "E_NETWORK": "the Hugging Face Hub could not be reached, and the files are not in the local "
                 "cache: check the network, or unset HF_HUB_OFFLINE if it is set",
}


def _hub_code(exc: BaseException) -> Optional[str]:
    """The code of a Hub failure a load raised (a gated repo, one that is not there, no way to
    reach the Hub with the files not cached), whatever wrapped it; None for anything else."""
    from poros.core.errors import NOT_CACHED

    seen: Optional[BaseException] = exc
    for _ in range(6):  # transformers' OSError, then huggingface_hub's own
        if seen is None:
            return None
        name, text = type(seen).__name__, str(seen)
        if name == "GatedRepoError" or (isinstance(seen, OSError) and "gated repo" in text):
            return "E_HUB_AUTH"
        if name in ("RepositoryNotFoundError", "RevisionNotFoundError") or (
                isinstance(seen, OSError) and "is not a valid model identifier" in text):
            return "E_NOT_FOUND"
        if name in ("LocalEntryNotFoundError", "OfflineModeIsEnabled") or (
                isinstance(seen, OSError) and seen.errno is None and NOT_CACHED.search(text)):
            return "E_NETWORK"
        seen = seen.__cause__  # what it was raised from: never an error it only met
    return None


def show_traceback() -> bool:
    """Whether an error's traceback goes to stderr: in JSON mode and in a job (stderr is the
    log a person reads later), or where POROS_TRACEBACK asks for it. Never onto a person's
    screen unasked."""
    import os

    return (contract.json_mode() or bool(os.environ.get("POROS_JOB"))
            or os.environ.get("POROS_TRACEBACK", "") not in ("", "0"))


def _crash_log(exc: BaseException) -> Optional[str]:
    """The whole traceback of an error that is Poros's own fault, written where its user can
    find and send it: POROS_HOME/crash.log (the newest one, its owner's alone, keys taken
    out). Its path, or None where it could not be written."""
    import os
    import traceback

    from poros.cli.context import home
    from poros.core.errors import no_keys

    path = home() / "crash.log"
    try:
        from poros.cli.ui import components as uc

        text = (f"poros {uc.version()} · python {sys.version.split()[0]} · {sys.platform}\n"
                f"argv: {' '.join(sys.argv[1:])}\n\n"
                + "".join(traceback.format_exception(exc)))
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
                     0o600)
        with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as f:
            f.write(no_keys(text))
    except Exception:  # a home that cannot be written: the screen says the error's own line
        return None
    return str(path)


def internal_error(exc: BaseException) -> NoReturn:
    """An error that is a bug in Poros, for a person: what it was in one line, where the whole
    of it is, and where to report it; exit 1. The traceback itself goes to the crash log, and
    to stderr only where ``show_traceback`` says."""
    from poros.cli.ui import components as uc
    from poros.cli.ui.live import columns
    from poros.cli.ui.term import err as out  # an error: stderr in a one-shot command
    from poros.cli.ui.text import S
    from poros.core.env import outside_words_out
    from poros.core.errors import library_error, no_keys
    from poros.links import ISSUES_URL

    if show_traceback():
        import traceback

        traceback.print_exception(exc, file=sys.stderr)
    log = _crash_log(exc)
    first = " ".join(str(exc).split())[:200]
    said = library_error(exc) if outside_words_out() or not first else (
        f"{type(exc).__name__}: {no_keys(first)}")
    lines = uc.error_event("Internal error", said, min(columns(), uc.CAP))
    lines.append(uc.rail(S("a bug in Poros, not something you did", "dm")))
    if log:
        lines.append(uc.rail(S("the whole error  ", "dm"), S(uc.short_path(log), "tx")))
    lines.append(uc.rail(S("report it  ", "dm"), S(ISSUES_URL, "tx"),
                         S("  with that file" if log else "", "dm")))
    if not show_traceback():
        lines.append(uc.rail(S("POROS_TRACEBACK=1 prints the traceback here", "dm")))
    out([[]] + lines)
    raise typer.Exit(code=EXIT_CODES["E_INTERNAL"])


def plain_fit_error(exc: Exception) -> Exception:
    """A fit refusal before the download (PorosVramError, PorosHostRamError, raised by the run's
    own check) as the CLI says it: its head built from the error's fields (the model, the card,
    the host RAM available), and the error's own remediation and docs, word for word. The same
    code and exit; any other exception as it is."""
    from poros.core.errors import PorosError, PorosHostRamError, PorosVramError

    if not isinstance(exc, (PorosHostRamError, PorosVramError)):
        return exc
    head = exc.head + (f": {exc.cause}" if exc.cause else "")  # the error's own words
    out = PorosError(message=f"{head}. Refusing before anything is downloaded or loaded.",
                     remediation=exc.info.remediation, doc_url=exc.info.doc_url)
    out.code = exc.code
    return out


def _friendly_error(exc: Exception) -> NoReturn:
    """Report an exception (✕ title · rails · docs, or the JSON error) and exit with its code."""
    from rich.markup import escape

    from poros.cli.ui import components as uc
    from poros.cli.ui.live import columns
    from poros.cli.ui.term import err as out  # an error: stderr in a one-shot command
    from poros.cli.ui.text import S
    from poros.core.env import outside_words_out
    from poros.core.errors import PorosError, library_error

    if isinstance(exc, typer.Exit):  # a refusal raised inside a command's try block
        raise exc
    exc = plain_fit_error(exc)  # a fit refusal: the decision in words, no planner figure
    try:
        from pydantic import ValidationError
    except ImportError:  # pragma: no cover
        ValidationError = ()
    w = min(columns(), uc.CAP)
    if isinstance(exc, ValidationError):
        import difflib

        from poros.core.config import PorosConfig

        rails, plain = [], []
        for err in exc.errors():
            loc = ".".join(str(x) for x in err["loc"])
            if (err["type"] == "extra_forbidden"
                    and getattr(exc, "title", "") in ("PorosConfig",
                                                      "BenchmarkSpec")):
                near = difflib.get_close_matches(loc, PorosConfig.model_fields.keys(), n=1)
                rails.append(uc.rail(S("unknown field ", "tx"), S(loc, "hi"),
                                     S(" — did you mean ", "dm") if near else None,
                                     S(near[0], "a") if near else None,
                                     S("?", "dm") if near else None))
                plain.append(f"unknown field {loc}"
                             + (f" — did you mean {near[0]}?" if near else ""))
            else:
                rails.append(uc.rail(S(loc, "hi"), S(": " + err["msg"], "tx")))
                plain.append(f"{loc}: {err['msg']}")
        if contract.json_mode():
            contract.emit_error("E_CONFIG", escape("invalid config: " + "; ".join(plain)),
                                hint="poros schema config")  # every field, with its type
        else:
            n = len(plain)
            out([[]] + uc.error_event("Invalid config", f"{n} problem" + ("s" if n != 1 else ""), w)
                + rails
                + [uc.rail(S("every field, with types: ", "dm"), S("poros schema config", "tx"))])
        raise typer.Exit(code=EXIT_CODES["E_CONFIG"])
    code = _error_code(exc)
    library = not isinstance(exc, PorosError) or exc.__cause__ is not None
    hub = _HUB_SAID.get(code) if not isinstance(exc, PorosError) and _hub_code(exc) else None
    if code == "E_INTERNAL" and not contract.json_mode():
        internal_error(exc)
    full = disk_full(exc) if not isinstance(exc, PorosError) else None
    if full is not None:  # a disk with no room: the disk's refusal, never a bug
        if contract.json_mode():
            contract.emit_error(code, escape(full[0]), hint=escape(full[1]))
        else:
            out([[]] + uc.error_event("Disk full", full[0], w)
                + [uc.rail(S("fix  ", "dm"), S(full[1], "tx"))])
        raise typer.Exit(code=EXIT_CODES[code])
    if library and show_traceback():
        import traceback  # a library's words: in a job's log for a person, and in no result

        traceback.print_exception(exc, file=sys.stderr)
    if contract.json_mode():
        if isinstance(exc, PorosError):
            contract.emit_error(code, escape(exc.info.message), hint=escape(exc.info.remediation),
                                fix=exc.info.fix, docs=exc.info.doc_url)
        elif hub:  # the Hub's answer, in our words: nothing of the library's text
            contract.emit_error(code, hub)
        else:
            if code == "E_INTERNAL":
                _crash_log(exc)
            contract.emit_error(code, escape(library_error(exc)),
                                hint="The whole error is on stderr, and in a job's log.")
        raise typer.Exit(code=EXIT_CODES[code])
    if isinstance(exc, PorosError):
        lines = uc.error_event("Error", exc.info.message, w)
        if exc.__cause__ is not None and not outside_words_out():  # on a person's own screen
            lines.append(uc.rail(S("why  ", "dm"), S(" ".join(str(exc.__cause__).split())[:300])))
        if exc.info.remediation:
            lines.append(uc.rail(S("fix  ", "dm"), S(exc.info.remediation, "tx")))
        if exc.info.doc_url:
            lines.append(uc.rail(S("docs ", "dm"), S(exc.info.doc_url, "dm")))
    elif hub:
        lines = uc.error_event("Error", hub, w)
    else:
        lines = uc.error_event("Error", library_error(exc) if outside_words_out() else str(exc), w)
    out([[]] + lines)
    raise typer.Exit(code=EXIT_CODES[code])
