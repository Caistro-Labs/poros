#!/usr/bin/env bash
# Install the built wheel into a fresh venv and run it from outside the repo, the way a
# user would. Catches files missing from the package.
#
#   scripts/smoke_wheel.sh [dist-dir] [expected-version]
set -euo pipefail

dist="$(cd "${1:-dist}" && pwd)"
expected="${2:-}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

wheels=("$dist"/*.whl)
[[ ${#wheels[@]} -eq 1 && -f "${wheels[0]}" ]] || { echo "error: want one wheel in $dist" >&2; exit 1; }
wheel="${wheels[0]}"

python -m venv "$work/venv"
py="$work/venv/bin/python"
"$py" -m pip install --quiet --upgrade pip
if [[ "$(uname -s)" == "Linux" ]]; then
  # CPU torch inside the wheel's own torch range: CI runners have no GPU, and this skips
  # gigabytes of CUDA wheels.
  torch_req="$("$py" - "$wheel" <<'PY'
import email.parser, sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    meta = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
    msg = email.parser.BytesParser().parsebytes(z.read(meta))
for req in msg.get_all("Requires-Dist") or []:
    name, _, marker = req.partition(";")
    if name.strip().startswith("torch") and "!=" in marker and "darwin" in marker:
        print(name.strip())
        break
else:
    sys.exit("no Linux torch requirement in the wheel")
PY
)"
  "$py" -m pip install --quiet "$torch_req" --index-url https://download.pytorch.org/whl/cpu
fi
"$py" -m pip install --quiet "$wheel"

cd "$work"
"$py" - "$expected" <<'PY'
import sys

import poros

v = getattr(poros, "__version__", None)
print(f"poros {v} imported from {poros.__file__}")
if sys.argv[1] and v != sys.argv[1]:
    sys.exit(f"installed version {v} != expected {sys.argv[1]}")
if "site-packages" not in poros.__file__:
    sys.exit("poros was imported from outside the venv")
PY
"$work/venv/bin/poros" --help >/dev/null
"$py" -c "import torch, transformers, peft; from poros import PorosTrainer"
echo "wheel smoke ok"
