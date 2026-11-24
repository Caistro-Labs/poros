"""Files the SSD store makes, torch-free: what the plan and the disk probe need
(``poros.sss.probe``) without importing the store's format, which needs torch."""

from __future__ import annotations

import os
from pathlib import Path


def new_file(path: Path, flags: int) -> int:
    """A file of ours, made new: a link left at its name does not send the write elsewhere."""
    path.unlink(missing_ok=True)
    return os.open(path, flags | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
