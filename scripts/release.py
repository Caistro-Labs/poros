#!/usr/bin/env python3
"""Release helper for Poros. Versions are semver, MAJOR.MINOR.PATCH, and a release's tag is
v<version>.

  python scripts/release.py version                         print the version, or fail if the
                                                            files that hold it disagree
  python scripts/release.py bump patch|minor|major          write the next version to each file
  python scripts/release.py check-signers [FILE]            fail unless FILE holds real signing keys
  python scripts/release.py check-unpublished NAME VERSION  fail if PyPI already has it
  python scripts/release.py verify-dist DIR --expected VERSION [--name NAME]

pyproject.toml is the version the build uses. poros/__init__.py and CITATION.cff carry the same
number, and `version` refuses while any of the three differs. Standard library only, so it runs
anywhere build.sh does.
"""

from __future__ import annotations

import argparse
import email.parser
import re
import sys
import tarfile
import tomllib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "poros" / "__init__.py"
CITATION = ROOT / "CITATION.cff"
SIGNERS = ROOT / ".github" / "allowed_signers"

SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
# The line in each file that holds the version, written back by `bump`.
LINES = {
    PYPROJECT: re.compile(r'^version = "(?P<v>[^"]+)"$', re.MULTILINE),
    INIT: re.compile(r'^__version__ = "(?P<v>[^"]+)"$', re.MULTILINE),
    CITATION: re.compile(r"^version: (?P<v>\S+)$", re.MULTILINE),
}
# One allowed_signers line: principal, namespaces="git", key type, base64 key, optional comment.
SIGNER = re.compile(
    r'^\S+@\S+ namespaces="git" '
    r"(ssh-ed25519|sk-ssh-ed25519@openssh\.com|ecdsa-sha2-nistp256|sk-ecdsa-sha2-nistp256@openssh\.com)"
    r" AAAA[0-9A-Za-z+/]+={0,3}( \S.*)?$"
)

# Nothing that looks like a credential ships in a public package.
FORBIDDEN = re.compile(
    r"(^|/)(\.env(\.[^/]*)?|[^/]*\.pem|[^/]*\.key|id_(rsa|ecdsa|ed25519)(_sk)?|\.netrc|\.pypirc|\.git)(/|$)"
)


def fail(msg: str) -> None:
    sys.exit(f"error: {msg}")


def _found(path: Path) -> str:
    rel = path.relative_to(ROOT)
    if not path.is_file():
        fail(f"{rel} not found")
    m = LINES[path].search(path.read_text(encoding="utf-8"))
    if not m:
        fail(f"no version line in {rel}")
    return m["v"]


def current_version() -> str:
    """The version, once pyproject.toml, poros/__init__.py and CITATION.cff agree on it."""
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8")).get("project", {})
    version = project.get("version")
    if not isinstance(version, str) or version != _found(PYPROJECT):
        fail("pyproject.toml's [project] version is not on its own `version = \"...\"` line")
    found = {p.relative_to(ROOT).as_posix(): _found(p) for p in LINES}
    if len(set(found.values())) != 1:
        fail("the version differs between files: "
             + ", ".join(f"{k} {v}" for k, v in found.items()))
    if not SEMVER.match(version):
        fail(f"{version} is not MAJOR.MINOR.PATCH")
    return version


def next_version(current: str, part: str) -> str:
    major, minor, patch = (int(x) for x in current.split("."))
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def cmd_bump(args: argparse.Namespace) -> None:
    old = current_version()
    new = next_version(old, args.part)
    for path, line in LINES.items():
        text = path.read_text(encoding="utf-8")
        m = line.search(text)
        path.write_text(text[: m.start("v")] + new + text[m.end("v"):], encoding="utf-8")
    print(f"{old} -> {new}")
    print(f"Add a `## [{new}]` section to CHANGELOG.md in the same PR.")


def cmd_version(_: argparse.Namespace) -> None:
    print(current_version())


def signer_problems(text: str) -> list[str]:
    """Why an allowed_signers file can't verify a release tag; empty when it can."""
    lines = [s.strip() for s in text.splitlines() if s.strip() and not s.lstrip().startswith("#")]
    if not lines:
        return ["it holds no key"]
    problems = []
    for n, line in enumerate(lines, 1):
        if "EDIT" in line:
            problems.append(f"key line {n} is still the placeholder")
        elif not SIGNER.match(line):
            problems.append(f'key line {n} is not `<email> namespaces="git" <key type> <key>`')
    return problems


def cmd_check_signers(args: argparse.Namespace) -> None:
    path = Path(args.file)
    if not path.is_file():
        fail(f"{path} not found")
    problems = signer_problems(path.read_text(encoding="utf-8"))
    if problems:
        fail(f"{path} can't verify a release tag: " + "; ".join(problems)
             + ". Paste the release key's public line in first.")
    print(f"ok: {path} holds a signing key")


def cmd_check_unpublished(args: argparse.Namespace) -> None:
    url = f"https://pypi.org/pypi/{args.name}/{args.version}/json"
    try:
        with urllib.request.urlopen(url, timeout=15):
            fail(f"{args.name} {args.version} is already on PyPI. Bump first: "
                 "python scripts/release.py bump patch")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"warning: PyPI answered {e.code}; could not confirm {args.version} is new",
                  file=sys.stderr)
    except OSError as e:
        print(f"warning: could not reach PyPI ({e}); PyPI will still refuse a duplicate",
              file=sys.stderr)


def _metadata(raw: bytes) -> email.message.Message:
    return email.parser.BytesParser().parsebytes(raw)


def cmd_verify_dist(args: argparse.Namespace) -> None:
    dist = Path(args.dir)
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        fail(f"expected one wheel and one sdist in {dist}, found {len(wheels)} and {len(sdists)}")
    wheel, sdist = wheels[0], sdists[0]

    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        meta = [n for n in names if n.endswith(".dist-info/METADATA")]
        if len(meta) != 1:
            fail(f"{wheel.name} has {len(meta)} METADATA files")
        wheel_meta = _metadata(z.read(meta[0]))
    with tarfile.open(sdist) as t:
        snames = t.getnames()
        pkg_info = [n for n in snames if n.count("/") == 1 and n.endswith("/PKG-INFO")]
        if len(pkg_info) != 1:
            fail(f"{sdist.name} has no top-level PKG-INFO")
        sdist_meta = _metadata(t.extractfile(pkg_info[0]).read())

    for label, meta in (("wheel", wheel_meta), ("sdist", sdist_meta)):
        if meta["Version"] != args.expected:
            fail(f"{label} version is {meta['Version']}, expected {args.expected}")
        if args.name and meta["Name"] != args.name:
            fail(f"{label} name is {meta['Name']}, expected {args.name}")

    leaked = [n for n in names + snames if FORBIDDEN.search(n)]
    if leaked:
        fail("files that look like credentials are in the package:\n  " + "\n  ".join(leaked))

    size = sum(p.stat().st_size for p in (wheel, sdist)) / 1e6
    print(f"ok: {wheel.name} + {sdist.name} at {args.expected}, {len(names)} files in the "
          f"wheel, {size:.1f} MB")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("version").set_defaults(fn=cmd_version)
    b = sub.add_parser("bump")
    b.add_argument("part", choices=["patch", "minor", "major"])
    b.set_defaults(fn=cmd_bump)
    s = sub.add_parser("check-signers")
    s.add_argument("file", nargs="?", default=str(SIGNERS))
    s.set_defaults(fn=cmd_check_signers)
    c = sub.add_parser("check-unpublished")
    c.add_argument("name")
    c.add_argument("version")
    c.set_defaults(fn=cmd_check_unpublished)
    v = sub.add_parser("verify-dist")
    v.add_argument("dir")
    v.add_argument("--expected", required=True)
    v.add_argument("--name", default="")
    v.set_defaults(fn=cmd_verify_dist)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
