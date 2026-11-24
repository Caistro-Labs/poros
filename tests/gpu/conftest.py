"""Every test in tests/gpu needs a CUDA device: all of them carry the ``gpu`` mark, so
``pytest -m gpu`` (the release check's GPU lane) selects them and ``-m "not gpu"`` (CPU CI)
leaves them out, whether or not a file remembered to mark itself."""

from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.hookimpl(tryfirst=True)  # before -m deselects by mark
def pytest_collection_modifyitems(config, items):
    # a conftest's collection hook sees the whole session's items: mark only this folder's
    for item in items:
        if HERE in Path(str(item.path)).resolve().parents:
            item.add_marker(pytest.mark.gpu)
