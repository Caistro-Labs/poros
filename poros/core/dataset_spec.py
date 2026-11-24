"""Classify a ``dataset:`` value -- torch-free.

Shared by the CLI preflight and the trainer so that a bad dataset spec fails
BEFORE any model download, and a local file trains without ceremony:
``dataset: ./my_data.jsonl`` just works.
"""

from __future__ import annotations

import glob
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from poros.core.errors import PorosDataError, PorosDatasetNotFoundError

# Local-file suffix -> the datasets-library builder that loads it.
_LOCAL_BUILDERS = {
    ".jsonl": "json",
    ".json": "json",
    ".csv": "csv",
    ".parquet": "parquet",
    ".txt": "text",
}

_REMEDIATION = (
    "Point 'dataset:' at a local .jsonl/.json/.csv/.parquet/.txt file, a "
    "Hugging Face dataset id (e.g. OpenAssistant/oasst1), or 'synthetic'."
)


# A file's name for a split, as datasets reads one: train.jsonl beside test.jsonl trains the first.
_NAMED_TRAIN = re.compile(r"(?:^|[-._ 0-9])(?:train|training)[-._ 0-9]")
_NAMED_HELD = re.compile(
    r"(?:^|[-._ 0-9])(?:validation|valid|dev|val|test|testing|eval|evaluation)[-._ 0-9]")


# A dataset's own papers, which datasets never read as rows either.
_NEVER_DATA = ("config.json", "dataset_info.json", "dataset_infos.json", "dataset_dict.json")


def own_data_files(folder: "int | Path") -> list[str]:
    """A folder's own regular files with a data suffix, by name.

    Nothing in the folder says otherwise. Handed the folder, datasets obeys its README's
    ``data_files``, which can name any file on this machine, and follows links, so a project could
    put a token in the rows and in the error. Hidden files and names of secrets are left out as
    well, and a subfolder is named on its own.
    """
    from poros.data import secret_name

    with os.scandir(folder) as entries:
        return sorted(e.name for e in entries if e.is_file(follow_symlinks=False)
                      and not secret_name(e.name) and e.name not in _NEVER_DATA
                      and Path(e.name).suffix.lower() in _LOCAL_BUILDERS)


def folder_files(folder: Path) -> tuple[str, list[str]]:
    """(builder, files) of a folder of data, from its own files."""
    own = own_data_files(folder)
    picked = ([n for n in own if _NAMED_TRAIN.search(n)]
              or [n for n in own if not _NAMED_HELD.search(n)])
    if not picked:
        raise PorosDataError(
            f"{folder} has no data file of its own to train on.",
            remediation="Name a file, or the subfolder that holds them. Links are not followed. "
                        + _REMEDIATION,
            doc_url="docs/training.md#data",
        )
    kinds = Counter(_LOCAL_BUILDERS[Path(n).suffix.lower()] for n in picked).most_common()
    if len(kinds) > 1 and kinds[0][1] == kinds[1][1]:
        raise PorosDataError(
            f"{folder} holds as many {kinds[0][0]} files as {kinds[1][0]} files.",
            remediation="Name one file, or a folder of one kind.",
            doc_url="docs/training.md#data",
        )
    builder = kinds[0][0]
    return builder, [str(Path(folder, n)) for n in picked
                     if _LOCAL_BUILDERS[Path(n).suffix.lower()] == builder]


def data_files(path: str) -> list[str]:
    """What ``load_dataset`` is handed for a local spec: exact paths. It reads a name as a glob, so
    a file called *.csv would read every csv beside it."""
    files = folder_files(Path(path))[1] if os.path.isdir(path) else [path]
    return [glob.escape(f) for f in files]


def load_hub_dataset(name: str, split: str = "train", downloading: Any = None) -> Any:
    """A Hub dataset, the way ``load_dataset`` loads it, with what its README named looked at first.
    A README can name an absolute path or a URL. datasets 5.1 resolves those and then refuses them
    itself; pyproject allows it back to 2.16, and this holds whichever is here: only the repo's own
    files are its dataset. ``downloading()`` is called first where the dataset is not on this
    machine yet and is about to be fetched, for a caller that shows the wait."""
    from datasets import load_dataset_builder
    from huggingface_hub import constants

    own = (f"hf://datasets/{name}@", f"hf://datasets/{name}/",
           f"{constants.ENDPOINT}/datasets/{name}/resolve/")
    builder = load_dataset_builder(name)
    for files in (builder.config.data_files or {}).values():
        if not all(str(f).startswith(own) and "::" not in str(f) for f in files):
            raise PorosDataError(
                f"The Hub dataset {name} names files outside its own repo, and they are not read.",
                remediation=_REMEDIATION,
                doc_url="docs/training.md#data",
            )
    if downloading is not None and not os.path.exists(
            os.path.join(str(getattr(builder, "cache_dir", "")), "dataset_info.json")):
        downloading()  # no prepared copy in the datasets cache: the files are fetched now
    builder.download_and_prepare()
    return builder.as_dataset(split=split)


def is_synthetic(dataset: str) -> bool:
    """The built-in generator, by the trainer's routing: any case, with an optional ':<n>' rows."""
    name = dataset.strip().lower()
    return name == "synthetic" or name.startswith("synthetic:")


def resolve_dataset_spec(dataset: str) -> tuple[str, str, str]:
    """Return ``(kind, builder, path)`` for a dataset spec.

    ``kind`` is one of:

    - ``"synthetic"`` -- the built-in generator.
    - ``"local"`` -- a data file, or a folder of them, on disk; ``builder`` is
      the datasets-library builder and ``path`` the resolved path. ``data_files``
      says what it reads.
    - ``"raw"`` -- everything else: a Hub id. ``path`` is "".

    Classification order matters and is deliberate: an existing file always
    wins; an explicit path prefix (``./``, ``../``, ``/``, ``~``) must point
    at something that exists; a bare filename with a known suffix and no
    ``/`` is a missing local file. Anything shaped ``org/name`` stays raw
    even when it ends in a data suffix -- hub repo names may contain dots,
    so ``org/data.csv`` must still reach the Hub.
    """
    dataset = dataset.strip()
    if not dataset:
        raise PorosDataError(
            "Dataset spec is empty.",
            remediation=_REMEDIATION,
            doc_url="docs/training.md#data",
        )
    if is_synthetic(dataset):
        return ("synthetic", "", "")
    path = Path(dataset).expanduser()
    suffix = path.suffix.lower()
    if path.is_file():
        builder = _LOCAL_BUILDERS.get(suffix)
        if builder is None:
            raise PorosDataError(
                f"Unsupported local dataset format: {dataset}",
                remediation="Local datasets must be .jsonl, .json, .csv, "
                ".parquet, or .txt. " + _REMEDIATION,
                doc_url="docs/training.md#data",
            )
        return ("local", builder, str(path))
    if path.is_dir():
        return ("local", folder_files(path)[0], str(path))
    path_like = dataset.startswith(("./", "../", "/", "~"))
    bare_local_file = "/" not in dataset and suffix in _LOCAL_BUILDERS
    if path_like or bare_local_file:
        raise PorosDatasetNotFoundError(
            f"Dataset file not found: {dataset}",
            remediation=_REMEDIATION,
            doc_url="docs/training.md#data",
        )
    return ("raw", "", "")
