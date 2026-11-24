"""The Mac table and the store rule, on this Mac's real numbers (no Mac needed)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from poros import metal
from poros.cli.words.models import by_id, catalog
from poros.core.config import PorosConfig
from poros.core.errors import EXIT_CODES, PorosDiskError, PorosGuaranteeUnavailableError
from poros.metal import store, table
from tests.conftest import needs_ml, weighing

M4 = "Apple M4 10-core GPU 16GB"
CLASS = "apple-m4-10gpu-16gb"
BASE7, INSTRUCT7, BASE32, BASE72 = (
    "Qwen/Qwen2.5-7B",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-32B",
    "Qwen/Qwen2.5-72B",
)
# `mount` on a Mac: the sealed system, its hidden volumes, an SSD, Recovery.
MOUNT = """\
/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)
devfs on /dev (devfs, local, nobrowse)
/dev/disk3s6 on /System/Volumes/VM (apfs, local, noexec, journaled, noatime, nobrowse)
/dev/disk3s5 on /System/Volumes/Data (apfs, local, journaled, nobrowse, protect, root data)
map auto_home on /System/Volumes/Data/home (autofs, automounted, nobrowse)
/dev/disk7s1 on /Volumes/Ext (apfs, local, nodev, nosuid, journaled, noowners)
/dev/disk3s3 on /Volumes/Recovery (apfs, local, journaled, nobrowse)
/dev/disk9s1 on /Volumes/Shared (smbfs, nodev, nosuid)
/dev/disk8s1 on /Volumes/Stick (msdos, local, nodev, nosuid)
"""


@pytest.fixture
def mini(tmp_path, monkeypatch):
    """A Mac with an internal disk and an external SSD, each with an empty Hub cache."""
    ssd = tmp_path / "Ext"
    ssd.mkdir()
    disks = [
        store.Volume(Path.home(), 245_075_738_624, 59_357_413_376),
        store.Volume(ssd, 499_898_105_856, 283_055_382_528),
    ]
    monkeypatch.setattr(store, "volumes", lambda: disks)
    real = store.shutil.disk_usage

    def usage(path):  # a store's room is its fake disk's, never this machine's own fill
        p = Path(path)
        hit = next((d for d in disks[1:] if p == d.path or d.path in p.parents), None)
        return real(path) if hit is None else SimpleNamespace(
            total=hit.total, used=hit.total - hit.free, free=hit.free)

    monkeypatch.setattr(store.shutil, "disk_usage", usage)
    monkeypatch.setattr(store, "_first_hub", lambda: (str(tmp_path / "default-hub"), False))
    monkeypatch.setattr(store.runs, "home", lambda: tmp_path / "poros-home")
    monkeypatch.setenv("POROS_HOME", str(tmp_path / "poros-home"))
    return ssd


# ---------------------------------------------------------------------------
# The table.
# ---------------------------------------------------------------------------


def test_each_model_has_one_default_and_its_rows_run_by_length():
    tables = [e.metal for e in catalog() if e.metal]
    assert tables
    for rows in tables:
        seqs = [r.seq for r in rows]
        assert seqs == sorted(seqs) and sum(r.default for r in rows) == 1


def test_names_and_classes_are_one_rule():
    assert table.class_of(M4) == CLASS and table.name_of(CLASS) == M4
    assert table.name_of("apple-m4-pro-14gpu-24gb") == "Apple M4 Pro 14-core GPU 24GB"
    assert table.class_of("Apple M4 Pro 14-core GPU 24GB") == "apple-m4-pro-14gpu-24gb"


def test_the_working_set_is_the_one_measured_here():
    gpu = table.plan_gpu(CLASS)
    assert gpu.unified and gpu.total_gib == 16.0 and gpu.free_gib is None and gpu.source == "spec"
    assert round(gpu.working_set_gib * 2**30 / 1e9, 2) == 12.71
    # Another Mac, planned for from elsewhere: two thirds of its memory, as reported
    assert table.plan_gpu("apple-m4-10gpu-24gb").working_set_gib == 16.0


def test_instruct_uses_its_bases_rows_and_other_models_have_none():
    assert table.pick(INSTRUCT7, CLASS) == table.pick(BASE7, CLASS)
    assert table.evidence(INSTRUCT7, CLASS) == "same-shape"
    assert table.evidence(BASE7, CLASS) == table.pick(BASE7, CLASS).evidence
    for model in ("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "org/unknown"):
        assert table.pick(model, CLASS) is None and table.evidence(model, CLASS) == "absent"


@pytest.mark.parametrize(
    "tokens, seq", [(1, 128), (128, 128), (129, 256), (256, 256), (300, 512), (9000, 512)]
)
def test_a_scan_takes_the_smallest_row_that_holds_it_else_the_largest(tokens, seq):
    assert table.pick(BASE7, CLASS, tokens=tokens).seq == seq


def test_a_set_length_takes_its_row_or_the_default_and_no_length_the_default():
    assert table.pick(BASE7, CLASS, seq=512).seq == 512
    assert table.pick(BASE7, CLASS, seq=1024).seq == 128  # off the table: the default's settings
    assert table.pick(BASE7, CLASS).seq == 128
    assert table.pick("Qwen/Qwen2.5-0.5B", CLASS).seq == 256


def test_another_mac_reads_rows_derived_for_it_with_no_flag():
    row = table.pick(BASE7, "apple-m4-10gpu-24gb")
    assert row.device_class == "apple-m4-10gpu-24gb" and row.planned_from == CLASS
    assert row.evidence == "derived" and not row.receipt and not row.macos_build


def test_a_length_the_config_left_unset_is_not_a_length_asked_for(monkeypatch):
    asked = []
    real = table.pick
    monkeypatch.setattr(table, "pick", lambda *a, **kw: asked.append(kw["seq"]) or real(*a, **kw))
    table.fill(PorosConfig(model_name_or_path=BASE7), CLASS)
    table.fill(PorosConfig(model_name_or_path=BASE7, max_seq_length=512), CLASS)
    assert asked == [None, 512]


def test_fill_takes_only_what_the_config_left_unset():
    cfg = PorosConfig(model_name_or_path=BASE72)
    row = table.fill(cfg, CLASS)
    assert row.seq == 128
    assert (cfg.max_seq_length, cfg.gradient_accumulation_steps, cfg.save_steps) == (128, 1, 25)
    assert (cfg.store, cfg.block_size) == ("ssd", 1)
    assert table.fill(cfg, CLASS) == row  # twice is once: every entry path reaches it

    set_ = PorosConfig(
        model_name_or_path=BASE7,
        max_seq_length=2048,
        gradient_accumulation_steps=2,
        save_steps=7,
    )
    row = table.fill(set_, CLASS)
    assert row.seq == 128 and row.seq != set_.max_seq_length  # left the table, and says so
    assert (set_.max_seq_length, set_.gradient_accumulation_steps, set_.save_steps) == (2048, 2, 7)


def test_a_run_from_disk_holds_out_no_rows_unless_asked():
    disk = PorosConfig(model_name_or_path="Qwen/Qwen2.5-72B")
    table.fill(disk, "apple-m4-10gpu-16gb")
    assert disk.eval_split == 0.0
    asked = PorosConfig(model_name_or_path="Qwen/Qwen2.5-72B", eval_split=0.1)
    table.fill(asked, "apple-m4-10gpu-16gb")
    assert asked.eval_split == 0.1
    ram = PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B")
    table.fill(ram, "apple-m4-10gpu-16gb")
    assert ram.eval_split == PorosConfig().eval_split


def test_fill_leaves_a_model_with_no_row_alone():
    cfg = PorosConfig(model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    before = cfg.model_dump()
    assert table.fill(cfg, CLASS) is None and cfg.model_dump() == before


def test_a_model_with_no_row_is_refused_and_another_mac_trains_with_no_flag():
    table.require_row(BASE7, CLASS)
    with pytest.raises(PorosGuaranteeUnavailableError, match="poros models"):
        table.require_row("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", CLASS)
    for other in ("apple-m4-10gpu-24gb", "apple-m2-10gpu-16gb", "apple-m1-7gpu-8gb"):
        table.require_row(BASE7, other)
        assert table.pick(BASE7, other).device_class == other
    with pytest.raises(PorosGuaranteeUnavailableError, match="is too large for this Mac's 8 GB"):
        table.require_row(BASE72, "apple-m1-7gpu-8gb")
    with pytest.raises(PorosGuaranteeUnavailableError, match="could not read"):
        table.require_row(BASE7, "unknown-0gpu-0gb")


def test_require_settles_a_run_for_every_entry_path(monkeypatch):
    calls = []
    monkeypatch.setattr(store, "settle", calls.append)
    monkeypatch.setattr(metal, "device_class", lambda: CLASS)
    cfg = PorosConfig(model_name_or_path=BASE7)
    metal.settle(cfg)
    assert cfg.gradient_accumulation_steps == 1 and calls == [cfg]
    with pytest.raises(PorosGuaranteeUnavailableError):
        metal.settle(PorosConfig(model_name_or_path="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"))


def test_a_local_directory_is_a_developers_own_model(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "settle", lambda _: pytest.fail("no store for a local model"))
    monkeypatch.setattr(metal, "device_class", lambda: CLASS)
    metal.settle(PorosConfig(model_name_or_path=str(tmp_path)))


def _checkpoint(monkeypatch, path, arch, gb):
    """A config and a shard that weighs ``gb`` as the table reads it, taking no disk: only their
    sizes are read (tests.conftest.weighing)."""
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": arch}))
    weighing(monkeypatch, path, gb)
    return str(path)


@pytest.mark.parametrize(
    "arch, gb, trains_as",
    [
        ("qwen2", 0.9, "Qwen/Qwen2.5-0.5B"),
        ("qwen2", 3.0, BASE7),  # the smallest that holds it
        ("qwen2", 150.0, BASE72),  # past them all, the largest
        ("qwen3", 7.0, "Qwen/Qwen3-32B"),
        ("qwen3_5", 2.0, "Qwen/Qwen3.5-4B"),
        ("qwen3_5_text", 2.0, "Qwen/Qwen3.5-4B"),
        ("gemma3_text", 1.0, "google/gemma-3-1b-pt"),
        ("gemma3", 1.0, "google/gemma-3-1b-pt"),  # a wrapper of the same family
        ("gemma3", 7.0, "google/gemma-3-27b-pt"),
        ("gemma4", 9.0, "google/gemma-4-31B-it"),
        ("gemma4_text", 9.0, "google/gemma-4-31B-it"),
        ("olmoe", 1.0, "allenai/OLMoE-1B-7B-0924"),
        ("qwen2_moe", 1.0, "Qwen/Qwen1.5-MoE-A2.7B"),
        ("deepseek_v2", 1.0, "deepseek-ai/DeepSeek-V2-Lite"),
        ("qwen3_moe", 1.0, "Qwen/Qwen3-30B-A3B-Instruct-2507"),
        ("qwen3_5_moe", 1.0, "Qwen/Qwen3.6-35B-A3B"),
        ("qwen3_5", 20.0, "Qwen/Qwen3.8-27B"),
    ],
)
def test_a_local_checkpoint_trains_as_its_family_at_its_size(tmp_path, monkeypatch, arch, gb,
                                                              trains_as):
    here = _checkpoint(monkeypatch, tmp_path / "mine", arch, gb)
    table.require_row(here, CLASS)
    assert table.rows(here, CLASS) == table.rows(trains_as, CLASS)
    assert table.pick(here, CLASS) == table.pick(trains_as, CLASS)
    assert table.evidence(here, CLASS) == table.evidence(trains_as, CLASS)
    assert table.floor_gb(here, 128) == table.floor_gb(trains_as, 128)
    assert table.disk_gb(here) == table.pack_gb(here) == min(table.pack_gb(trains_as), gb)


def test_a_local_checkpoint_needs_the_room_its_own_weights_do(tmp_path, monkeypatch):
    """Six layers of a 61 GB family are a 9 GB folder: its pack cannot take the family's 18 GB."""
    trim = _checkpoint(monkeypatch, tmp_path / "trim", "qwen3_moe", 8.7)
    whole = _checkpoint(monkeypatch, tmp_path / "whole", "qwen3_moe", 61.0)
    assert table.pack_gb(trim) == table.disk_gb(trim) == pytest.approx(8.7)
    assert table.pack_gb(whole) == table.disk_gb(whole) == table.pack_gb(
        "Qwen/Qwen3-30B-A3B-Instruct-2507")

    def free(gb):
        monkeypatch.setattr(store.shutil, "disk_usage", lambda _: SimpleNamespace(
            total=465 * 10**9, free=gb * 10**9))

    free(56)  # 8.7 and 10% of the disk
    assert store._checked(tmp_path / "store", trim) == tmp_path / "store"
    with pytest.raises(PorosDiskError, match="8.7 GB of disk"):
        free(50)
        store._checked(tmp_path / "store", trim)
    with pytest.raises(PorosDiskError, match="18.0 GB of disk"):
        free(56)
        store._checked(tmp_path / "store", whole)


@pytest.mark.parametrize("arch", ["llama", ""])
def test_a_local_checkpoint_of_another_family_has_no_row(tmp_path, monkeypatch, arch):
    here = _checkpoint(monkeypatch, tmp_path / "mine", arch, 1.0)
    assert table.pick(here, CLASS) is None
    with pytest.raises(PorosGuaranteeUnavailableError, match="does not train on a Mac"):
        table.require_row(here, CLASS)


def test_a_folder_that_is_no_checkpoint_has_no_row(tmp_path):
    assert table.pick(str(tmp_path), CLASS) is None
    (tmp_path / "config.json").write_text("not json")
    assert table.pick(str(tmp_path), CLASS) is None


def test_a_local_checkpoint_settles_like_the_model_it_trains_as(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(store, "settle", calls.append)
    monkeypatch.setattr(metal, "device_class", lambda: CLASS)
    here = _checkpoint(monkeypatch, tmp_path / "mine", "qwen3", 7.0)
    cfg = PorosConfig(model_name_or_path=here)
    metal.settle(cfg)
    assert calls == [cfg]
    row = table.pick("Qwen/Qwen3-32B", CLASS)
    assert (cfg.max_seq_length, cfg.store) == (row.seq, row.tier)


# ---------------------------------------------------------------------------
# The floor: only what cannot fit is refused.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model, seq, fits",
    [
        (BASE7, 512, True),
        (BASE32, 128, True),
        (BASE72, 128, True),
        (BASE72, 512, True),
        (BASE72, 2048, False),
        (BASE72, 4096, False),
    ],
)
def test_the_floor_refuses_only_past_the_pool(model, seq, fits):
    pool = 16 * 2**30 / 1e9
    assert (table.floor_gb(model, seq) + table.OS_RESERVE_GB <= pool) is fits


def test_instruct_needs_what_its_base_needs():
    assert table.disk_gb(INSTRUCT7) == table.disk_gb(BASE7) == pytest.approx(20.78)
    assert table.floor_gb(INSTRUCT7, 128) == table.floor_gb(BASE7, 128)


# ---------------------------------------------------------------------------
# Which volumes count.
# ---------------------------------------------------------------------------


def test_only_local_writable_apfs_that_macos_shows(monkeypatch):
    monkeypatch.setattr(store, "run_tool", lambda *_: MOUNT)
    assert store._mounted() == [Path("/Volumes/Ext")]  # not the system, Recovery, smb or FAT


def test_home_stands_for_its_disk_and_a_second_mount_of_it_is_not_counted(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setattr(store, "_mounted", lambda: [tmp_path, other])
    devices = {Path.home(): 1, tmp_path: 1, other: 2}
    monkeypatch.setattr(store.os, "stat", lambda p: type("S", (), {"st_dev": devices[p]})())
    assert [v.path for v in store.volumes()] == [Path.home(), other]


# ---------------------------------------------------------------------------
# The rule, on a two-disk Mac's numbers.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model", [BASE32, BASE72])
def test_the_external_ssd_is_picked_without_a_word(mini, model):
    assert store.pick(model) == mini / "poros"


@pytest.mark.parametrize("model", ["Qwen/Qwen2.5-0.5B", BASE7])
def test_two_that_qualify_are_asked_about_most_room_first(mini, model):
    assert store.pick(model) is None
    assert [v.path for v in store.qualifying(model)] == [mini, Path.home()]


def test_a_local_checkpoint_is_packed_on_its_own_disk_with_no_question(mini, monkeypatch):
    # both disks qualify: a Hub model would ask
    own = _checkpoint(monkeypatch, mini / "mine", "qwen2", 0.9)
    monkeypatch.setattr(store, "_same_disk", lambda a, b: mini in a.parents and mini in b.parents)
    assert store.pick(own) == store.qualifying(own)[0].store == mini / "poros"
    assert store.missing_gb(mini / "poros", own) == table.pack_gb(own)
    assert store.missing_gb(store.internal_store(), own) == table.pack_gb(own)  # never its shards
    assert own in store.survey(own)[1]
    packed = mini / "poros" / "mine-abcd1234"
    packed.mkdir(parents=True)
    (packed / "model.sss.idx").touch()
    assert store.missing_gb(mini / "poros", own) == 0
    assert "prepare" not in store.survey(own)[1]


def test_a_pack_with_its_shards_gone_needs_nothing_and_stays_put(mini):
    packed = store.internal_store() / "Qwen--Qwen2.5-7B-abcd1234"
    packed.mkdir(parents=True)
    (packed / "model.sss.idx").touch()
    assert store.missing_gb(store.internal_store(), BASE7) == 0
    assert store.missing_gb(mini / "poros", BASE7) == table.disk_gb(BASE7)
    assert store.pick(BASE7) == store.internal_store()  # two disks qualify: this one holds it
    assert store.built(store.pick(BASE7), BASE7)


def test_an_answer_is_remembered_while_it_still_qualifies(mini, monkeypatch):
    import poros.cli.questions as questions

    monkeypatch.setattr(questions, "setting", lambda name: str(mini / "poros"))
    monkeypatch.setattr(store.shutil, "disk_usage",
                        lambda _: type("U", (), {"total": 500e9, "free": 283e9})())
    assert store.pick(BASE7) == mini / "poros"
    monkeypatch.setattr(questions, "setting", lambda name: "/gone/disk/poros")
    assert store.pick(BASE7) is None


def test_a_remembered_store_is_read_without_a_terminal(mini, monkeypatch, capsys):
    """HF_HOME-shaped, on no volume's default path: what a machine is told once."""
    import poros.cli.questions as questions

    monkeypatch.setattr(questions, "interactive", lambda: False)
    monkeypatch.setattr(store.shutil, "disk_usage",
                        lambda _: type("U", (), {"total": 500e9, "free": 283e9})())
    home = mini / "own" / "hf"
    assert store.pick(BASE7) is None
    questions.remember("store", str(home))
    assert store.pick(BASE7) == home and store.hub_of(home) == str(home / "hub")
    assert str(home / "hub") in store.hubs()


def _whole(hub, model):
    """Every weight of a model, one byte each: what is on disk, not how large."""
    index = json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}})
    _cached(
        hub,
        model,
        ("model.safetensors.index.json", index),
        ("model-1.safetensors", "w"),
        ("model-2.safetensors", "w"),
    )


def test_what_is_already_there_needs_no_room(mini, monkeypatch):
    """On a disk under 10% free only what is missing is charged: all of it there, no E_DISK."""
    import poros.cli.questions as questions

    home = mini / "own" / "hf"
    monkeypatch.setattr(questions, "setting", lambda name: str(home))
    monkeypatch.setattr(store.shutil, "disk_usage",
                        lambda _: type("U", (), {"total": 500e9, "free": 37e9})())
    with pytest.raises(PorosDiskError, match="20.8 GB"):
        store.pick(BASE7)
    _whole(home / "hub", BASE7)
    assert store.missing_gb(home, BASE7) == table.pack_gb(BASE7)
    with pytest.raises(PorosDiskError, match="5.6 GB"):  # the pack is new bytes, and is charged
        store.pick(BASE7)
    (home / "Qwen--Qwen2.5-7B-abc").mkdir()
    assert store.missing_gb(home, BASE7) == table.pack_gb(BASE7)  # a killed pack is not a pack
    (home / "Qwen--Qwen2.5-7B-abc" / "model.sss.idx").touch()
    assert store.missing_gb(home, BASE7) == 0 and store.pick(BASE7) == home


def test_part_of_the_shards_is_charged_what_is_left(mini):
    index = json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}})
    hub = mini / "own" / "hub"
    _cached(hub, BASE7, ("model.safetensors.index.json", index), ("model-1.safetensors", ""))
    (next(hub.glob("*/snapshots/abc")) / "model-1.safetensors").write_bytes(b"\0" * 10**6)
    assert store.missing_gb(mini / "own", BASE7) == pytest.approx(table.disk_gb(BASE7) - 0.001)


def test_a_run_keeps_its_files_beside_a_model_off_the_internal_disk(mini, monkeypatch):
    def device(path, **_):
        return type("S", (), {"st_dev": 2 if str(path).startswith(str(mini)) else 1})()

    own = mini / "own" / "hf"
    with monkeypatch.context() as patch:
        patch.setattr(store.os, "stat", device)
        assert store.run_files(own) is None  # POROS_HOME, set: the fixture's
        patch.delenv("POROS_HOME")
        assert store.run_files(own) == own / "runs"
        assert store.run_files(Path.home() / "anywhere") is None


def test_a_model_already_on_a_volume_stays_there(mini, monkeypatch):
    monkeypatch.setattr(store, "_holds", lambda s, m: s == store.internal_store() and m == BASE7)
    monkeypatch.setattr(store, "_checked", lambda s, m: s)
    assert store.pick(BASE7) == store.internal_store()


def _cached(hub, model, *files):
    """A real Hub cache layout, for the real lookup."""
    repo = hub / f"models--{model.replace('/', '--')}"
    snapshot = repo / "snapshots" / "abc"
    snapshot.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("abc")
    for name, text in files:
        (snapshot / name).write_text(text)


def test_a_config_json_alone_does_not_hold_a_model(mini, tmp_path):
    """Asking about a model fetches its config. The internal disk has one for each."""
    _cached(tmp_path / "default-hub", BASE72, ("config.json", "{}"))
    assert store.pick(BASE72) == mini / "poros"


def test_any_of_a_models_weights_hold_it(mini, tmp_path, monkeypatch):
    monkeypatch.setattr(store, "_checked", lambda s, m: s)
    index = json.dumps({"weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}})
    _cached(
        tmp_path / "default-hub",
        BASE72,
        ("model.safetensors.index.json", index),
        ("model-2.safetensors", "w"),
    )
    assert store.pick(BASE72) == store.internal_store()
    _cached(tmp_path / "default-hub", BASE32, ("model.safetensors.index.json", index))
    assert store.pick(BASE32) == mini / "poros"  # an index with no shard has nothing to resume
    _cached(tmp_path / "default-hub", "Qwen/Qwen2.5-0.5B", ("model.safetensors", "w"))
    assert store.pick("Qwen/Qwen2.5-0.5B") == store.internal_store()


def test_none_that_qualifies_says_what_is_needed_and_what_is_free(mini, monkeypatch):
    full = [store.Volume(Path.home(), 245_000_000_000, 40_000_000_000)]
    monkeypatch.setattr(store, "volumes", lambda: full)
    with pytest.raises(PorosDiskError) as refused:
        store.pick(BASE72)
    assert refused.value.code == "E_DISK" and EXIT_CODES["E_DISK"] == 5
    assert "186.6 GB" in str(refused.value) and "internal 40 GB free" in str(refused.value)


def test_the_ten_percent_rule_is_the_rule_at_its_edge(mini):
    need = table.disk_gb(BASE72) * 1e9
    terabyte = 1_000_000_000_000
    assert store.Volume(Path("/v"), terabyte, int(need + 100e9)).qualifies(table.disk_gb(BASE72))
    assert not store.Volume(Path("/v"), terabyte, int(need + 99e9)).qualifies(table.disk_gb(BASE72))


def test_a_named_store_must_be_mounted_and_have_the_room(mini, monkeypatch):
    with pytest.raises(PorosDiskError, match="/gone/disk is not mounted"):
        store.pick(BASE7, "/gone/disk/poros")
    monkeypatch.setattr(
        store.shutil, "disk_usage", lambda _: type("U", (), {"total": 100e9, "free": 15e9})()
    )
    with pytest.raises(PorosDiskError, match="15 GB free"):
        store.pick(BASE7, str(mini / "here"))
    monkeypatch.setattr(store, "missing_gb", lambda s, m: 0.0)
    assert store.pick(BASE7, str(mini / "here")) == mini / "here"  # already there: nothing to fit


def test_a_hub_cache_the_user_set_up_is_theirs(mini, tmp_path, monkeypatch):
    own = tmp_path / "theirs" / "hub"
    own.mkdir(parents=True)
    monkeypatch.setattr(store, "_first_hub", lambda: (str(own), True))
    monkeypatch.setattr(store, "missing_gb", lambda s, m: 0.0)
    assert store.pick(BASE7) == own.parent
    assert store.hub_of(mini / "poros") == str(own)


def test_use_points_the_process_and_its_children_at_the_stores_cache(mini, monkeypatch):
    from huggingface_hub import constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", constants.HF_HUB_CACHE)
    monkeypatch.setenv("HF_HUB_CACHE", "before")
    store.use(mini / "poros")
    hub = str(mini / "poros" / "hub")
    assert constants.HF_HUB_CACHE == hub == store.os.environ["HF_HUB_CACHE"]
    store.use(store.internal_store())
    assert constants.HF_HUB_CACHE == str(mini.parent / "default-hub")


def test_settle_decides_for_a_run_with_no_one_to_ask(mini, monkeypatch):
    used = []
    monkeypatch.setattr(store, "use", used.append)
    cfg = PorosConfig(model_name_or_path=BASE72)
    store.settle(cfg)
    assert cfg.store_path == str(mini / "poros") and used == [mini / "poros"]
    with pytest.raises(store.PorosConfigError, match="store_path"):
        store.settle(PorosConfig(model_name_or_path=BASE7))


def test_survey_never_asks_and_never_raises(mini, monkeypatch):
    store_, line = store.survey(BASE72)
    assert store_ == mini / "poros" and "186.6 GB" in line
    store_, line = store.survey(BASE7)
    assert store_ is None and "train asks which" in line and "Ext 283 GB free" in line
    monkeypatch.setattr(store, "_holds", lambda s, m: True)
    monkeypatch.setattr(store, "_checked", lambda s, m: s)
    store_, line = store.survey(BASE7)
    assert store_ is not None and line.startswith("the model is in")
    assert line.endswith("and 5.6 GB to prepare it")
    (store_ / "Qwen--Qwen2.5-7B-abcd1234").mkdir(parents=True)
    (store_ / "Qwen--Qwen2.5-7B-abcd1234" / "model.sss.idx").touch()
    assert "prepare" not in store.survey(BASE7)[1]
    monkeypatch.setattr(store, "_holds", lambda s, m: False)
    monkeypatch.setattr(store, "volumes", lambda: [store.Volume(Path.home(), 245e9, 5e9)])
    store_, line = store.survey(BASE72)
    assert store_ is None and "no disk has room" in line


def test_a_catalog_row_with_no_size_cannot_exist():
    for e in catalog():
        if e.metal:
            assert table.disk_gb(e.id) > table.pack_gb(e.id) > 0
            assert by_id(e.id) is e


@needs_ml
@pytest.mark.parametrize("arch, kind, entry", [("llama", {}, "dense"),
                                               ("qwen3_next", {"num_experts": 8}, "moe")])
def test_a_folder_of_an_asserted_family_trains_as_the_nearest_of_its_kind(
    tmp_path, monkeypatch, arch, kind, entry
):
    here = _checkpoint(monkeypatch, tmp_path / "mine", arch, 1.0)
    config = tmp_path / "mine" / "config.json"
    config.write_text(json.dumps({**json.loads(config.read_text()), **kind}))
    assert table.pick(here, CLASS) is None
    monkeypatch.setenv("POROS_USER_ASSERTED_ARCHS", arch)
    row = table.pick(here, CLASS)
    assert row.derived and row.evidence == table.evidence(here, CLASS) == "none"
    assert table._entry(here).entry == entry


def test_a_local_checkpoints_weights_are_its_safetensors_sizes_summed(tmp_path):
    """poros.metal.table._weights_gb, on real files (tests.conftest.weighing stands in for it
    elsewhere): every *.safetensors in the folder counts, in decimal GB, and nothing else does
    (a config, an index, another suffix, a subfolder's shard)."""
    from poros.metal import table

    (tmp_path / "model-00001-of-00002.safetensors").write_bytes(b"\0" * 3_000)
    (tmp_path / "model-00002-of-00002.safetensors").write_bytes(b"\0" * 2_000)
    (tmp_path / "model.safetensors.index.json").write_bytes(b"{}" * 500)
    (tmp_path / "config.json").write_bytes(b"{}")
    (tmp_path / "pytorch_model.bin").write_bytes(b"\0" * 7_000)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "extra.safetensors").write_bytes(b"\0" * 9_000)
    assert table._weights_gb(str(tmp_path)) == 5_000 / 1e9
    assert table._weights_gb(str(tmp_path / "empty-or-missing")) == 0.0
