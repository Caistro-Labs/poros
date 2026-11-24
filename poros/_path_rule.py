"""Run first, by ``import poros``: a Python path names its folders.

An empty or relative entry reads the folder a command runs in, whoever's project that is. The common
``export PYTHONPATH=$PYTHONPATH:/x`` leaves one, and ``-P`` does not cover it. So this drops them
from ``sys.path``, where ``site`` has by now made them absolute, and from ``PYTHONPATH``, which
every child inherits. Python's own first entry stays, without ``-P``: whoever typed ``python -c`` or
sits at a prompt chose the folder.

This runs once ``poros`` itself is found. On such a path a planted ``poros/``, or a
``sitecustomize.py``, has run by then, and nothing in here can stop that: ``poros machine`` flags
the entry, from a clean folder, so a person can remove it. Imports nothing looked for on the path.
"""

import os
import sys

_said = os.environ.get("PYTHONPATH") or ""
_entries = _said.split(os.pathsep) if _said else []
dropped = [entry for entry in _entries if not os.path.isabs(entry)]
_kept = [entry for entry in _entries if os.path.isabs(entry)]
_gone = {os.path.abspath(entry) for entry in dropped} - set(map(os.path.abspath, _kept))

_own = 0 if sys.flags.safe_path else 1
sys.path[_own:] = [p for p in sys.path[_own:]
                   if not isinstance(p, str) or (os.path.isabs(p) and p not in _gone)]

if _kept and dropped:
    os.environ["PYTHONPATH"] = os.pathsep.join(_kept)
elif dropped:
    del os.environ["PYTHONPATH"]
