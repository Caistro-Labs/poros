"""Where the tests find the benchmark logs (docs/validation/).

The logs live in the companion repository github.com/Caistro-Labs/poros-benchmarks, at the same
paths the package and the docs cite (docs/validation/...). A test that reads one asks for it
here:

- POROS_BENCHMARKS set: the root of a checkout of poros-benchmarks. It must hold docs/validation/:
  a variable that points nowhere is an error, not a reason to skip.
- unset: the repository's own root, which holds docs/validation/ in the development tree.

Without the folder at all (a checkout of Caistro-Labs/poros alone), the test skips and says where
the logs are. A single log missing from a checkout that has the folder is not skipped: the test
reads it and fails, as it should.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FOLDER = "docs/validation"
COMPANION = "github.com/Caistro-Labs/poros-benchmarks"


def root() -> Path | None:
    """The checkout that holds docs/validation/, or None when there is none."""
    said = os.environ.get("POROS_BENCHMARKS")
    base = Path(said).expanduser().resolve() if said else REPO
    if (base / FOLDER).is_dir():
        return base
    if said:
        raise RuntimeError(f"POROS_BENCHMARKS={said} has no {FOLDER}/: point it at the root of a "
                           f"checkout of {COMPANION}")
    return None


def available() -> bool:
    return root() is not None


REASON = (f"needs the benchmark logs ({FOLDER}/), in {COMPANION}: set POROS_BENCHMARKS to the "
          "root of a checkout of it")
needs_logs = pytest.mark.skipif(not available(), reason=REASON)


def benchmark_log(path: str) -> Path:
    """The file at ``path`` (as the package and the docs cite it, ``docs/validation/...``); a path
    outside docs/validation is the repository's own. Skips without the logs."""
    if not (path == FOLDER or path.startswith(FOLDER + "/")):
        return REPO / path
    base = root()
    if base is None:
        pytest.skip(REASON)
    return base / path
