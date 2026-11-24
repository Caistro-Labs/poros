"""Users are offered their own data, never test-run data. A preview offers its own: made-up chats
for SFT and made-up pairs for DPO, each with rows held out so every checkpoint is scored."""

import json
from types import SimpleNamespace

from poros import data
from poros.cli import sheets
from poros.cli.ui import preview_data


def _offered(monkeypatch, tmp_path, simulated):
    from poros.cli.context import Context

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("POROS_DEV_SIMULATE", "1" if simulated else "0")
    (tmp_path / "mine.jsonl").write_text("{}\n")
    return [o.value for o in sheets._data_options(Context(cwd=tmp_path))()]


def test_the_train_sheet_offers_no_test_run_data(monkeypatch, tmp_path):
    assert _offered(monkeypatch, tmp_path, simulated=False) == ["mine.jsonl"]


def test_a_preview_offers_its_chats_and_pairs_first(monkeypatch, tmp_path):
    got = _offered(monkeypatch, tmp_path, simulated=True)
    assert [g.rsplit("/", 1)[-1] for g in got[:2]] == [preview_data.CHATS, preview_data.PAIRS]
    assert got[2:] == ["mine.jsonl"]


def test_the_preview_files_train_sft_and_dpo_and_hold_rows_out(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    for p, shape, n in ((preview_data.chats(), "chats", preview_data.CHAT_ROWS),
                        (preview_data.pairs(), "pairs", preview_data.PAIR_ROWS)):
        rows = [json.loads(x) for x in p.read_text().splitlines()]
        assert len(rows) == n and data.detect(p) == shape
        cfg = SimpleNamespace(dataset=str(p), eval_dataset=None, eval_split=0.05, eval_rows=0)
        assert data.held_rows(cfg) > 0
    first = preview_data.chats().read_text()
    assert preview_data.chats().read_text() == first  # written once, the same every time
