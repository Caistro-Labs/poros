"""A merged export stages beside its folder in ``<out>.partial`` and deletes only a staging folder
an export made (marked when made): a folder of that name of anyone else's is refused, untouched."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from poros.cli.commands import export as export_command  # noqa: E402
from poros.core.errors import PorosExportError  # noqa: E402
from poros.train import export as train_export  # noqa: E402


def test_a_folder_named_like_the_stage_is_refused_and_left_whole(tmp_path):
    theirs = tmp_path / "merged.partial"
    theirs.mkdir()
    (theirs / "notes.txt").write_text("mine")
    with pytest.raises(PorosExportError, match="merged.partial is in the way"):
        with train_export.staged_output(tmp_path / "merged"):
            raise AssertionError("never staged")
    assert (theirs / "notes.txt").read_text() == "mine"
    assert not (tmp_path / "merged").exists()
    assert "is in the way" in export_command._foreign_stage(tmp_path / "merged")
    (tmp_path / "f.partial").write_text("a file")  # a file or a link of that name: the same
    assert export_command._foreign_stage(tmp_path / "f")
    (tmp_path / "l.partial").symlink_to(theirs)
    assert export_command._foreign_stage(tmp_path / "l")
    with pytest.raises(PorosExportError):
        with train_export.staged_output(tmp_path / "l"):
            pass
    assert (theirs / "notes.txt").exists()


def test_an_exports_own_stage_is_marked_cleared_and_never_moved_in(tmp_path):
    out = tmp_path / "merged"
    with pytest.raises(RuntimeError):  # an export stopped mid-write: its stage is cleaned up
        with train_export.staged_output(out) as stage:
            assert (stage / train_export.STAGING_MARKER).is_file()
            raise RuntimeError("stopped")
    left = tmp_path / "merged.partial"  # one a kill -9 left: marked, so it is cleared
    left.mkdir()
    (left / train_export.STAGING_MARKER).write_text("")
    (left / "half.safetensors").write_bytes(b"x")
    assert export_command._foreign_stage(out) is None
    with train_export.staged_output(out) as stage:
        (stage / "model.safetensors").write_bytes(b"y")
    assert sorted(p.name for p in out.iterdir()) == ["model.safetensors"]
    assert not left.exists()


def test_the_command_and_the_helper_name_one_marker():
    assert export_command.STAGING_MARKER == train_export.STAGING_MARKER
