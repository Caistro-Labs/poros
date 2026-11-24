"""Shared recipes: the ones people publish in a GitHub repo (``recipes/**/*.toml``), fetched only
when you ask -- ``poros recipes sync``, or ⌃R in /recipes -- and read after that from this
machine's copy, so listing them never touches the network.

The repo is anyone's words. A fetch reads the tarball in memory and never unpacks it: a member is
read only if it is a plain file under ``recipes/``, every part of its path a recipe's word, under
the cap; each goes through ``rules.parse`` like any file here, and is written by ``rules.write_new``
into a fresh folder that replaces the last copy whole. ``POROS_RECIPES_REPO`` names another repo
(``owner/name``), or a folder holding one.
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from poros.recipes import presets, rules

# The one place the shared recipes' source is named. Not published yet: until it is, a sync of
# it says so and how to name another (POROS_RECIPES_REPO), and the copy synced last stays.
REPO = "Caistro-Labs/poros-recipes"
_REPO = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}")
MOST = 500  # recipes kept from one repo
MAX_BYTES = 8 << 20  # the tarball, as it comes
MAX_READ = 64 << 20  # the tarball unpacked: decompressed only this far, then refused (_gunzip)
TIMEOUT = 20


def repo() -> str:
    """The repo shared recipes come from: ``owner/name``, or a folder holding one."""
    named = os.environ.get("POROS_RECIPES_REPO", "").strip()
    if not named:
        return REPO
    if local_path(named):
        if Path(named).expanduser().is_dir():
            return named
        raise ValueError(f"POROS_RECIPES_REPO names no folder: {named}")
    if _REPO.fullmatch(named):
        return named
    raise ValueError("POROS_RECIPES_REPO is neither owner/name nor a folder (/..., ./..., ../... "
                     "or ~/...)")


def local_path(name: str) -> bool:
    """A folder is named as one: it starts with /, ./, ../ or ~. ``owner/name`` is always the
    GitHub repo, whatever folder of that name the current one holds."""
    return name.startswith(("/", "./", "../", "~"))


def shown(name: str) -> str:
    """The repo as the card says it: ``github.com/owner/name``, or the folder itself."""
    return os.path.expanduser(name) if local_path(name) else f"github.com/{name}"


def demo() -> bool:
    """A preview (POROS_DEV_SIMULATE): the shared set is Poros's demo one; nothing is fetched."""
    return os.environ.get("POROS_DEV_SIMULATE", "") not in ("", "0")


def folder() -> Path:
    return presets._home() / "shared-recipes"


def _meta_path() -> Path:
    return presets._home() / "shared-recipes.json"


def meta() -> Dict[str, Any]:
    """What the last sync said: its repo, when, how many, and the files it refused."""
    if demo():
        return {"repo": repo(), "at": None, "count": len(_demo()), "refused": [], "demo": True}
    path = _meta_path()
    try:
        if path.is_symlink() or not path.is_file():
            return {}
        with open(path, "rb") as f:
            got = json.loads(f.read(1 << 16))
    except (ValueError, OSError):
        return {}
    return got if isinstance(got, dict) else {}


def cached() -> Tuple[List[presets.Recipe], Dict[str, Any]]:
    """The shared recipes this machine holds, newest first; and what the last sync said."""
    if demo():
        return _demo(), meta()
    where = folder()
    out: List[presets.Recipe] = []
    if where.is_dir() and not where.is_symlink():
        for path in sorted(where.glob("*.toml"))[:MOST]:
            try:
                if rules.WORD.fullmatch(path.stem) and not presets.is_built_in(path.stem):
                    fields = rules.parse(rules.read_file(path), shared=True)
                    out.append(presets.from_fields(path.stem, fields, "shared", ""))
            except ValueError:
                continue  # it passed when synced: a file changed here since is simply not shown
    out.sort(key=lambda r: r.updated.toordinal() if r.updated else 0, reverse=True)
    return out, meta()


def _demo() -> List[presets.Recipe]:
    from poros.recipes.demo import FILES

    out = [presets.from_fields(key, rules.parse(text, shared=True), "shared", "")
           for key, text in FILES]
    out.sort(key=lambda r: r.updated.toordinal() if r.updated else 0, reverse=True)
    return out


# -- fetching ---------------------------------------------------------------------------------


def _wanted(parts: List[str]) -> bool:
    """``recipes/<a>/<b>.toml``, up to two folders deep, every part a recipe's word."""
    return (2 <= len(parts) <= 4 and parts[0] == "recipes" and parts[-1].endswith(".toml")
            and all(rules.WORD.fullmatch(p) for p in parts))


def _from_tarball(data: bytes) -> Iterator[Tuple[List[str], Optional[str]]]:
    """Each wanted member's path (after the tarball's top folder) and text; None when it is not
    text under the cap. Nothing is written: the members are only read."""
    with tarfile.open(fileobj=io.BytesIO(_gunzip(data)), mode="r:") as tar:
        for n, m in enumerate(tar):
            if n > 20 * MOST:
                return
            parts = m.name.split("/")[1:]
            if not m.isreg() or not _wanted(parts):
                continue
            if m.size > rules.CAP_BYTES:
                yield parts, None
                continue
            raw = tar.extractfile(m)
            try:
                yield parts, raw.read(rules.CAP_BYTES + 1).decode() if raw else None
            except UnicodeDecodeError:
                yield parts, None


def _gunzip(data: bytes) -> bytes:
    """The tarball unpacked, never past MAX_READ bytes: decompressed here, before the tar parser
    reads a header, so no member or extended header can expand past the cap. Over it, or not one
    whole gzip stream, is a ValueError (the copy synced last stays)."""
    import zlib

    d = zlib.decompressobj(wbits=31)  # gzip
    out = d.decompress(data, MAX_READ + 1)
    if len(out) > MAX_READ or d.unconsumed_tail:
        raise ValueError(f"the recipes tarball unpacks to over {MAX_READ >> 20} MB; the copy "
                         "synced last stays")
    if not d.eof:
        raise EOFError("cut short")
    return out


def _from_folder(root: Path) -> Iterator[Tuple[List[str], Optional[str]]]:
    base = root / "recipes"
    if not base.is_dir() or base.is_symlink():
        return
    for top, dirs, files in os.walk(base):  # links are not followed
        dirs[:] = sorted(d for d in dirs if not (Path(top) / d).is_symlink())
        for f in sorted(files):
            parts = list((Path(top) / f).relative_to(root).parts)
            if _wanted(parts):
                try:
                    yield parts, rules.read_file(Path(top) / f)
                except ValueError:
                    yield parts, None


def _read_tarball(name: str, data: bytes) -> List[Tuple[List[str], Optional[str]]]:
    """Every wanted member of the downloaded tarball, read whole before anything is written: a
    body cut short or not a gzip tarball at all is a ValueError in one line, never a traceback
    halfway through the sync."""
    import zlib

    try:
        return list(_from_tarball(data))
    except (tarfile.TarError, EOFError, zlib.error, OSError):
        raise ValueError(f"github.com/{name} sent no whole recipes tarball (cut short, or not "
                         "a gzip tarball); the copy synced last stays") from None


def _download(name: str) -> bytes:
    import http.client
    import urllib.error
    import urllib.request

    req = urllib.request.Request(f"https://api.github.com/repos/{name}/tarball",
                                 headers={"User-Agent": "poros",
                                          "Accept": "application/vnd.github+json"})
    other = "POROS_RECIPES_REPO=owner/name (or a folder) syncs another"
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = resp.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ValueError(
                f"github.com/{name} is not published yet; {other}" if name == REPO else
                f"there is no public repo github.com/{name}; {other}") from None
        raise ValueError("GitHub said no for now: try again in an hour" if exc.code in (403, 429)
                         else f"GitHub answered {exc.code} for github.com/{name}") from None
    except (urllib.error.URLError, OSError, TimeoutError):
        raise ValueError(f"GitHub could not be reached for github.com/{name}: check the "
                         f"network; {other}") from None
    except http.client.HTTPException:  # IncompleteRead: the connection dropped mid-body
        raise ValueError(f"GitHub's answer for github.com/{name} was cut off; try again") from None
    if len(data) > MAX_BYTES:
        raise ValueError(f"the repo is over {MAX_BYTES >> 20} MB")
    return data


def sync() -> Dict[str, Any]:
    """Fetch the repo's recipes and keep them, in place of the last copy. What it said (as
    ``meta``), or ValueError in one line."""
    name = repo()
    if demo():
        return meta()
    if local_path(name):
        found = _from_folder(Path(name).expanduser())
    else:
        found = _read_tarball(name, _download(name))
    where = folder()
    if where.is_symlink():
        raise ValueError("the shared recipes folder is a link")
    fresh = where.with_name(where.name + ".new")
    if fresh.is_symlink():
        fresh.unlink()
    elif fresh.exists():
        shutil.rmtree(fresh)
    fresh.mkdir(parents=True)
    try:
        kept, refused = _fill(fresh, found)
    except BaseException:  # whatever stopped it: no half-written .new is left beside the copy
        shutil.rmtree(fresh, ignore_errors=True)
        raise
    old = where.with_name(where.name + ".old")
    if old.exists() and not old.is_symlink():
        shutil.rmtree(old)
    if where.exists():
        where.rename(old)
    fresh.rename(where)
    if old.exists():
        shutil.rmtree(old)
    said = {"repo": name, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "count": kept, "refused": refused[:50]}
    _keep_meta(said)
    return said


def _fill(fresh: Path, found) -> Tuple[int, List[List[str]]]:
    """The recipes ``found`` that pass the rules, written into ``fresh``; how many, and what was
    refused (path, why)."""
    kept, refused = 0, []
    for parts, text in found:
        path = "/".join(parts)
        stem = parts[-1][: -len(".toml")]
        if kept >= MOST:
            refused.append([path, f"only the first {MOST} are kept"])
            break
        try:
            if text is None:
                raise ValueError(f"it is not text under {rules.CAP_BYTES} bytes")
            if presets.is_built_in(stem):
                raise ValueError("it takes the name of one of Poros's")
            rules.parse(text, shared=True)
            rules.write_new(fresh, stem, text)
            kept += 1
        except ValueError as exc:
            refused.append([path, str(exc)])
    return kept, refused


def _keep_meta(said: Dict[str, Any]) -> None:
    """The sync's word, written whole: a new file through no link, then moved over the last."""
    path = _meta_path()
    tmp = path.with_name(path.name + ".new")
    if tmp.is_symlink() or tmp.exists():
        tmp.unlink()
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o644)
    with os.fdopen(fd, "w") as f:
        json.dump(said, f)
    os.replace(tmp, path)


def ago(meta_: Dict[str, Any], now: Optional[datetime] = None) -> str:
    """When the copy was synced, as people say it: ``2 hours ago``; "" for never."""
    at = meta_.get("at")
    if not isinstance(at, str):
        return ""
    try:
        then = datetime.fromisoformat(at)
    except ValueError:
        return ""
    s = ((now or datetime.now(timezone.utc)) - then).total_seconds()
    for size, word in ((86400 * 7, "week"), (86400, "day"), (3600, "hour"), (60, "minute")):
        if s >= size:
            n = int(s // size)
            return f"{n} {word}{'s' * (n != 1)} ago"
    return "just now"
