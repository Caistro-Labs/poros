#!/usr/bin/env bash
# Build and release Poros.
#
#   ./build.sh            build the sdist and wheel, then verify them
#   ./build.sh publish    the same, from a clean origin/main, then sign and push the
#                         release tag v<version> with your hardware key. CI publishes it
#                         to PyPI after you approve. No PyPI token is involved.
#
# To cut a release:
#   python scripts/release.py bump patch   ->  PR  ->  merge  ->  ./build.sh publish
set -euo pipefail
cd "$(dirname "$0")"

PYPI_NAME="poros-train"

die() { echo "error: $*" >&2; exit 1; }

publish=false
case "${1:-}" in
  "") ;;
  publish) publish=true ;;
  *) die "usage: ./build.sh [publish]" ;;
esac

python -c "import build, twine" 2>/dev/null || die "install the build tools: python -m pip install build twine"

version="$(python scripts/release.py version)"
tag="v$version"

if $publish; then
  git fetch --quiet origin main --tags
  [[ "$(git branch --show-current)" == "main" ]] || die "check out main first"
  [[ -z "$(git status --porcelain --untracked-files=no)" ]] || die "working tree has changes; commit them first"
  [[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] || die "local main differs from origin/main; run: git pull --ff-only"
  if git rev-parse -q --verify "refs/tags/$tag" >/dev/null; then
    die "$tag already exists; bump first: python scripts/release.py bump patch"
  fi
  # CI checks the tag against main's allowed_signers; a placeholder there fails it after the push.
  python scripts/release.py check-signers .github/allowed_signers
  python scripts/release.py check-unpublished "$PYPI_NAME" "$version"
fi

rm -rf dist
# The commit's time stamps the files, so the same commit builds the same wheel here and in CI.
SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-$(git log -1 --format=%ct)}" python -m build
python scripts/release.py verify-dist dist --expected "$version" --name "$PYPI_NAME"
python -m twine check --strict dist/*
# The release job prints the same sums for the files it builds from the tag: compare them.
if command -v sha256sum >/dev/null; then sha256sum dist/*; else shasum -a 256 dist/*; fi

if $publish; then
  echo "Signing $tag. Approve with Touch ID or your key."
  # ssh, whatever git config says: allowed_signers holds ssh keys, and CI checks the tag with it.
  git -c gpg.format=ssh tag -s "$tag" -m "Poros $version"
  # verify-tag picks its checker from the signature it finds, so an OpenPGP or X.509 signature
  # would be checked against your own keyring, not allowed_signers: only an ssh one goes on.
  if ! grep -qx -- '-----BEGIN SSH SIGNATURE-----' <<<"$(git cat-file tag "$tag")"; then
    git tag -d "$tag" >/dev/null
    die "$tag does not carry an ssh signature, so nothing was pushed; set git config gpg.format ssh and user.signingkey to your release key"
  fi
  # The check CI runs after the push, run here before it: a tag that this key signed and
  # allowed_signers does not list never leaves this machine.
  if ! git -c gpg.format=ssh -c gpg.ssh.allowedSignersFile=.github/allowed_signers verify-tag "$tag"; then
    git tag -d "$tag" >/dev/null
    die "$tag is not signed by a key in .github/allowed_signers, so nothing was pushed; set git config gpg.format ssh and user.signingkey to the listed key"
  fi
  git push origin "$tag"
  echo "Pushed $tag. Approve the pypi deployment in GitHub Actions to publish it."
fi
