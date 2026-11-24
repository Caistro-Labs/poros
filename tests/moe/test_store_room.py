"""The SSD store's room is checked before anything downloads: its
.sss, every expert in NF4 sized from config.json, against store_path's disk; refused E_DISK, not a
plain OSError after the download. A .sss already there is reused and needs no room."""

from __future__ import annotations

import json
from collections import namedtuple

import pytest

pytest.importorskip("torch")

from poros.core.config import PorosConfig  # noqa: E402
from poros.core.errors import EXIT_CODES, PorosDiskError  # noqa: E402
from poros.preflight.start import store_room  # noqa: E402

MOE = {"model_type": "qwen3_moe", "hidden_size": 2048, "num_hidden_layers": 48,
       "num_attention_heads": 32, "num_key_value_heads": 4, "head_dim": 128,
       "vocab_size": 151936, "intermediate_size": 6144, "moe_intermediate_size": 768,
       "num_experts": 128, "num_experts_per_tok": 8}


def _cfg(tmp_path):
    folder = tmp_path / "model"
    folder.mkdir(exist_ok=True)
    (folder / "config.json").write_text(json.dumps(MOE))
    return PorosConfig(model_name_or_path=str(folder), store="ssd",
                       store_path=str(tmp_path / "store"))


def test_a_store_disk_without_room_is_refused_before_the_download(tmp_path, monkeypatch):
    import shutil

    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10**12, 10**12 - 2**30, 2**30))
    cfg = _cfg(tmp_path)
    with pytest.raises(PorosDiskError, match=r"needs 1[0-9]\.\d GiB .* 1\.0 GiB free") as e:
        store_room(cfg, MOE)
    assert EXIT_CODES[e.value.code] == EXIT_CODES["E_DISK"]


def test_room_enough_or_a_store_already_built_passes(tmp_path, monkeypatch):
    import shutil

    from poros.moe.load import sss_file_path

    cfg = _cfg(tmp_path)
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10**13, 0, 10**13))
    store_room(cfg, MOE)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10**12, 10**12 - 1, 1))
    built = sss_file_path(cfg.store_path, cfg.model_name_or_path)
    built.parent.mkdir(parents=True)
    built.write_bytes(b"")
    store_room(cfg, MOE)
    store_room(cfg.model_copy(update={"store": "ram"}), MOE)


def test_the_room_counts_the_files_header_and_every_slabs_padding(tmp_path, monkeypatch):
    """The room asked for is the .sss file's, not the experts' bare NF4 bytes: its 4 KiB header
    and every (layer, expert) slab padded to the next 4 KiB, so a disk with room for the bytes
    alone is refused before the download."""
    import shutil

    from poros.plan.estimate import _sizes
    from poros.sss.format import ALIGNMENT, HEADER_NBYTES

    odd = {**MOE, "moe_intermediate_size": 776}  # an expert's slab that ends mid-page
    sizes = _sizes(odd)
    assert sizes.expert_slabs == 48 * 128
    slab = -(-sizes.experts_nf4 // sizes.expert_slabs)
    need = HEADER_NBYTES + sizes.expert_slabs * (-(-slab // ALIGNMENT) * ALIGNMENT)
    assert need - HEADER_NBYTES - sizes.experts_nf4 > sizes.expert_slabs * ALIGNMENT // 2
    usage = namedtuple("usage", "total used free")
    cfg = _cfg(tmp_path)
    for free, refused in ((sizes.experts_nf4, True), (need - 1, True), (need, False)):
        monkeypatch.setattr(shutil, "disk_usage", lambda p, free=free: usage(10**13, 0, free))
        if refused:
            with pytest.raises(PorosDiskError):
                store_room(cfg, odd)
        else:
            store_room(cfg, odd)
