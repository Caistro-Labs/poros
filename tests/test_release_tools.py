"""scripts/release.py and build.sh: the version they read, and each release they refuse."""

from __future__ import annotations

import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.error
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "scripts" / "release.py"
BUILD = ROOT / "build.sh"
KEY = ('dev@example.com namespaces="git" ssh-ed25519 '
       "AAAAC3NzaC1lZDI1NTE5AAAAIK1x0f2nDq2c6m8bJ3m8Wq9s6l2a9f6xg5zC2q3xY0aB")

pytestmark = pytest.mark.skipif(not RELEASE.is_file() or not BUILD.is_file(),
                                reason="needs the repository's release scripts")


def _project(tmp: Path, *, pyproject="1.0.0", init="1.0.0", citation="1.0.0") -> Path:
    """A tree with the three version files and the release scripts, as release.py reads it."""
    (tmp / "scripts").mkdir(parents=True)
    (tmp / "poros").mkdir()
    (tmp / ".github").mkdir()
    shutil.copy(RELEASE, tmp / "scripts" / "release.py")
    shutil.copy(BUILD, tmp / "build.sh")
    (tmp / "pyproject.toml").write_text(
        f'[project]\nname = "poros-train"\nversion = "{pyproject}"\n')
    (tmp / "poros" / "__init__.py").write_text(f'"""Poros."""\n\n__version__ = "{init}"\n')
    (tmp / "CITATION.cff").write_text(f"cff-version: 1.2.0\ntitle: Poros\nversion: {citation}\n")
    (tmp / ".github" / "allowed_signers").write_text(f"# keys\n{KEY}\n")
    return tmp


def _release(tree: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(tree / "scripts" / "release.py"), *args],
                          capture_output=True, text=True, cwd=tree)


def test_the_version_is_the_one_all_three_files_hold():
    out = _release(ROOT, "version")
    assert out.returncode == 0, out.stderr
    import tomllib

    assert out.stdout.strip() == tomllib.loads(
        (ROOT / "pyproject.toml").read_text())["project"]["version"]


@pytest.mark.parametrize("files", [{"init": "1.0.1"}, {"citation": "0.9.0"}])
def test_version_refuses_files_that_disagree(tmp_path, files):
    out = _release(_project(tmp_path, **files), "version")
    assert out.returncode == 1 and "differs between files" in out.stderr


def test_version_refuses_one_that_is_not_semver(tmp_path):
    tree = _project(tmp_path, pyproject="1.0.0rc1", init="1.0.0rc1", citation="1.0.0rc1")
    out = _release(tree, "version")
    assert out.returncode == 1 and "MAJOR.MINOR.PATCH" in out.stderr


@pytest.mark.parametrize(("part", "new"), [("patch", "1.0.1"), ("minor", "1.1.0"),
                                           ("major", "2.0.0")])
def test_bump_writes_the_next_version_to_each_file(tmp_path, part, new):
    tree = _project(tmp_path)
    out = _release(tree, "bump", part)
    assert out.returncode == 0 and f"1.0.0 -> {new}" in out.stdout
    assert _release(tree, "version").stdout.strip() == new
    assert f'__version__ = "{new}"' in (tree / "poros" / "__init__.py").read_text()


@pytest.mark.parametrize(("text", "why"), [
    ('EDIT-release-email namespaces="git" EDIT_PASTE_THE_PUBLIC_KEY\n', "placeholder"),
    ("# only comments\n", "holds no key"),
    ("dev@example.com not-a-key-line\n", "is not"),
])
def test_check_signers_refuses_a_file_that_cannot_verify_a_tag(tmp_path, text, why):
    signers = tmp_path / "allowed_signers"
    signers.write_text(text)
    out = _release(ROOT, "check-signers", str(signers))
    assert out.returncode == 1 and why in out.stderr


def test_check_signers_passes_a_real_key_line(tmp_path):
    signers = tmp_path / "allowed_signers"
    signers.write_text(f"# keys\n{KEY}\n")
    assert _release(ROOT, "check-signers", str(signers)).returncode == 0


def _dist(tmp: Path, *, version="1.0.0", sdist_version=None, extra: str = "") -> Path:
    dist = tmp / "dist"
    dist.mkdir()
    meta = f"Metadata-Version: 2.4\nName: poros-train\nVersion: {version}\n\n"
    with zipfile.ZipFile(dist / f"poros_train-{version}-py3-none-any.whl", "w") as z:
        z.writestr(f"poros_train-{version}.dist-info/METADATA", meta)
        z.writestr("poros/__init__.py", "")
        if extra:
            z.writestr(extra, "SECRET=1")
    smeta = meta.replace(version, sdist_version or version).encode()
    with tarfile.open(dist / f"poros_train-{version}.tar.gz", "w:gz") as t:
        info = tarfile.TarInfo(f"poros_train-{version}/PKG-INFO")
        info.size = len(smeta)
        t.addfile(info, io.BytesIO(smeta))
    return dist


def test_verify_dist_passes_a_matching_pair(tmp_path):
    out = _release(ROOT, "verify-dist", str(_dist(tmp_path)), "--expected", "1.0.0",
                   "--name", "poros-train")
    assert out.returncode == 0, out.stderr


@pytest.mark.parametrize(("dist_args", "args", "why"), [
    ({"version": "1.0.1"}, ["--expected", "1.0.0"], "wheel version is 1.0.1"),
    ({"sdist_version": "0.9.9"}, ["--expected", "1.0.0"], "sdist version is 0.9.9"),
    ({}, ["--expected", "1.0.0", "--name", "poros"], "name is poros-train"),
    ({"extra": "poros/.env"}, ["--expected", "1.0.0"], "look like credentials"),
])
def test_verify_dist_refuses(tmp_path, dist_args, args, why):
    out = _release(ROOT, "verify-dist", str(_dist(tmp_path, **dist_args)), *args)
    assert out.returncode == 1 and why in out.stderr


def _release_module():
    spec = importlib.util.spec_from_file_location("release_under_test", RELEASE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_check_unpublished_refuses_a_version_pypi_has(monkeypatch):
    mod = _release_module()

    class Found:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *a, **k: Found())
    with pytest.raises(SystemExit, match="already on PyPI"):
        mod.cmd_check_unpublished(type("A", (), {"name": "poros-train", "version": "1.0.0"}))


def test_check_unpublished_passes_a_new_version(monkeypatch):
    mod = _release_module()

    def missing(*a, **k):
        raise urllib.error.HTTPError("u", 404, "Not Found", {}, None)

    monkeypatch.setattr(mod.urllib.request, "urlopen", missing)
    mod.cmd_check_unpublished(type("A", (), {"name": "poros-train", "version": "9.9.9"}))


# build.sh publish: each refusal before anything is built, signed or pushed.


def _git(cwd: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    return subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def checkout(tmp_path):
    """A clone of a one-commit origin, on main, with stand-ins for the build tools."""
    if shutil.which("git") is None or shutil.which("bash") is None:
        pytest.skip("needs git and bash")
    src = _project(tmp_path / "src")
    _git(src, "init", "-q", "-b", "main")
    _git(src, "add", "-A")
    _git(src, "commit", "-q", "-m", "one")
    origin = tmp_path / "origin.git"
    _git(tmp_path, "clone", "-q", "--bare", str(src), str(origin))
    work = tmp_path / "work"
    _git(tmp_path, "clone", "-q", str(origin), str(work))
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("build", "twine"):
        (tools / f"{name}.py").write_text("")
    (tools / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    (tools / "python").chmod(0o755)
    env = {**os.environ, "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
           "PYTHONPATH": str(tools)}
    return work, env


def _publish(work: Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(work / "build.sh"), *args], cwd=work, env=env,
                          capture_output=True, text=True)


def test_build_refuses_an_unknown_argument(checkout):
    work, env = checkout
    out = _publish(work, env, "upload")
    assert out.returncode == 1 and "usage: ./build.sh [publish]" in out.stderr


def test_publish_refuses_off_main(checkout):
    work, env = checkout
    _git(work, "checkout", "-q", "-b", "side")
    assert "check out main first" in _publish(work, env, "publish").stderr


def test_publish_refuses_a_dirty_tree(checkout):
    work, env = checkout
    (work / "pyproject.toml").write_text((work / "pyproject.toml").read_text() + "# x\n")
    assert "working tree has changes" in _publish(work, env, "publish").stderr


def test_publish_refuses_main_ahead_of_origin(checkout):
    work, env = checkout
    _git(work, "commit", "-q", "--allow-empty", "-m", "local")
    assert "differs from origin/main" in _publish(work, env, "publish").stderr


def test_publish_refuses_a_tag_that_exists(checkout):
    work, env = checkout
    _git(work, "tag", "v1.0.0")
    assert "v1.0.0 already exists" in _publish(work, env, "publish").stderr


def test_publish_refuses_the_placeholder_signing_key(checkout, tmp_path):
    work, env = checkout
    signers = work / ".github" / "allowed_signers"
    signers.write_text('EDIT-release-email namespaces="git" EDIT_PASTE_THE_PUBLIC_KEY\n')
    _git(work, "commit", "-q", "-am", "placeholder")
    _git(work, "push", "-q", "origin", "main")
    out = _publish(work, env, "publish")
    assert out.returncode == 1 and "is still the placeholder" in out.stderr


def _signed_publish(work: Path, env: dict, tmp_path: Path, listed: bool = True,
                    config: tuple = ()) -> tuple:
    """build.sh publish with a fresh ssh key "mine" as the signing key; allowed_signers lists
    it when ``listed``, else another key. ``config`` is set in the checkout's git config last."""
    keys = tmp_path / "keys"
    keys.mkdir()
    for name in ("mine", "other"):
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "", "-f",
                        str(keys / name)], check=True)
    kind, key = (keys / ("mine.pub" if listed else "other.pub")).read_text().split()[:2]
    (work / ".github" / "allowed_signers").write_text(
        f'release@example.com namespaces="git" {kind} {key}\n')
    _git(work, "commit", "-q", "-am", "the release key")
    _git(work, "push", "-q", "origin", "main")
    for name, value in (("gpg.format", "ssh"), ("user.signingkey", str(keys / "mine.pub")),
                        ("user.name", "t"), ("user.email", "t@example.com"), *config):
        _git(work, "config", name, value)
    # Stand-ins for what needs the network or a real wheel: the PyPI lookup and the wheel checks.
    tools = Path(env["PATH"].split(os.pathsep)[0])
    (tools / "python").write_text('#!/bin/sh\n'
                                  'case "$2" in check-unpublished|verify-dist) exit 0;; esac\n'
                                  f'exec "{sys.executable}" "$@"\n')
    (tools / "build.py").write_text("import pathlib\npathlib.Path('dist').mkdir(exist_ok=True)\n"
                                    "pathlib.Path('dist/poros.whl').write_text('x')\n")
    out = _publish(work, env, "publish")
    return out, _git(work, "ls-remote", "--tags", "origin")


@pytest.mark.parametrize("listed", [True, False])
def test_publish_pushes_only_a_tag_that_a_listed_key_signed(checkout, tmp_path, listed):
    """build.sh runs CI's signature check before the push, so a tag signed by a key that
    allowed_signers does not list is deleted here and never reaches the remote."""
    if shutil.which("ssh-keygen") is None:
        pytest.skip("needs ssh-keygen")
    work, env = checkout
    out, pushed = _signed_publish(work, env, tmp_path, listed=listed)
    if listed:
        assert out.returncode == 0, out.stderr
        assert "refs/tags/v1.0.0" in pushed
    else:
        assert out.returncode == 1 and "nothing was pushed" in out.stderr
        assert pushed == "" and _git(work, "tag", "--list") == ""


def test_publish_signs_with_ssh_whatever_git_config_says(checkout, tmp_path):
    """A checkout whose git config asks for OpenPGP still signs the tag with the listed ssh key,
    and the tag is pushed: build.sh forces the ssh format."""
    if shutil.which("ssh-keygen") is None:
        pytest.skip("needs ssh-keygen")
    work, env = checkout
    out, pushed = _signed_publish(work, env, tmp_path, config=(("gpg.format", "openpgp"),))
    assert out.returncode == 0, out.stderr
    assert "refs/tags/v1.0.0" in pushed
    assert "-----BEGIN SSH SIGNATURE-----" in _git(work, "cat-file", "tag", "v1.0.0")


def test_publish_refuses_a_tag_whose_signature_is_not_ssh(checkout, tmp_path):
    """verify-tag checks an OpenPGP signature against your own keyring, not allowed_signers, so
    a tag that carries one is deleted and never pushed, even when that keyring vouches for it.
    git's own settings make the tag: gpg.ssh.program is the signer git runs, here one that
    writes an OpenPGP block, and gpg.program is the checker verify-tag runs for that block,
    here one that reports a good signature as a keyring that trusts the key would."""
    if shutil.which("ssh-keygen") is None:
        pytest.skip("needs ssh-keygen")
    work, env = checkout
    fake = tmp_path / "fake"
    fake.mkdir()
    signer, checker = fake / "signer", fake / "checker"
    # git runs: <signer> -Y sign -n git -f <key> <file>, and reads the signature from <file>.sig
    signer.write_text('#!/bin/sh\nfor last; do :; done\n'
                      'printf -- "-----BEGIN PGP SIGNATURE-----\\n\\nAAAA\\n'
                      '-----END PGP SIGNATURE-----\\n" > "$last.sig"\n')
    checker.write_text('#!/bin/sh\necho "[GNUPG:] NEWSIG"\n'
                       'echo "[GNUPG:] GOODSIG 0123456789ABCDEF t <t@example.com>"\n'
                       'echo "[GNUPG:] VALIDSIG 0123456789ABCDEF0123456789ABCDEF01234567 x"\n'
                       'echo "[GNUPG:] TRUST_ULTIMATE"\nexit 0\n')
    for f in (signer, checker):
        f.chmod(0o755)
    out, pushed = _signed_publish(work, env, tmp_path, config=(
        ("gpg.ssh.program", str(signer)), ("gpg.program", str(checker))))
    assert out.returncode == 1 and "does not carry an ssh signature" in out.stderr, out.stderr
    assert "nothing was pushed" in out.stderr
    assert pushed == "" and _git(work, "tag", "--list") == ""


def test_the_shipped_signers_file_refuses_until_its_key_is_pasted_in():
    """While .github/allowed_signers holds the placeholder, no release can publish."""
    signers = ROOT / ".github" / "allowed_signers"
    if not signers.is_file():
        pytest.skip("no .github in this tree")
    text = signers.read_text()
    if "EDIT" in text:
        assert _release(ROOT, "check-signers", str(signers)).returncode == 1


# scripts/repo_rules.sh: a dry run prints each call and its JSON, and changes nothing.


def test_repo_rules_dry_run_prints_valid_calls_and_the_real_check_names(tmp_path):
    import json
    import re

    script = ROOT / "scripts" / "repo_rules.sh"
    if not script.is_file() or shutil.which("jq") is None or shutil.which("bash") is None:
        pytest.skip("needs scripts/repo_rules.sh, bash and jq")
    stub = tmp_path / "bin"
    stub.mkdir()
    calls = tmp_path / "gh_calls"
    (stub / "gh").write_text(f'#!/bin/sh\necho "$@" >> "{calls}"\nexit 1\n')  # no network
    (stub / "gh").chmod(0o755)
    env = {**os.environ, "PATH": f"{stub}{os.pathsep}{os.environ['PATH']}"}
    out = subprocess.run(["bash", str(script), "example-owner/example-repo"], cwd=ROOT, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "Nothing was changed" in out.stdout
    # the only gh calls a dry run makes are reads of the rulesets
    assert all(line.startswith("api repos/example-owner/example-repo/rulesets?")
               for line in calls.read_text().splitlines())
    blocks = re.findall(r"^(PATCH|PUT|POST) /(\S+)\n(\{.*?^\})?", out.stdout, re.M | re.S)
    assert [m for m, _, _ in blocks] == ["PATCH", "POST", "POST", "POST", "PUT", "PUT", "PATCH",
                                         "PUT"]
    bodies = [json.loads(b) for _, _, b in blocks if b]
    assert bodies[0]["pull_request_creation_policy"] == "collaborators_only"  # no outside PRs
    kept = next(b for b in bodies if b.get("name") == "main: never rewritten")
    assert kept["bypass_actors"] == [] and {r["type"] for r in kept["rules"]} == {
        "deletion", "non_fast_forward"}  # no one, an admin included, rewrites main
    main = next(b for b in bodies if b.get("name") == "main: checks")
    assert [a["bypass_mode"] for a in main["bypass_actors"]] == ["always"]  # the export push
    checks = next(r for r in main["rules"] if r["type"] == "required_status_checks")
    names = [c["context"] for c in checks["parameters"]["required_status_checks"]]
    workflows = "".join(p.read_text() for p in (ROOT / ".github" / "workflows").glob("*.yml"))
    for name in names:
        assert re.search(rf"^    name: {re.escape(name)}$", workflows, re.M), name
    assert {"Check", "Lint", "Workflow lint"} == set(names)
    tags = next(b for b in bodies if b.get("name") == "release tags")
    assert tags["target"] == "tag" and {r["type"] for r in tags["rules"]} == {
        "creation", "update", "deletion"}
