"""A data file saved with a UTF-8 byte-order mark (Notepad, Excel's CSV UTF-8) reads as the same
rows as one without: .jsonl, .json and the preflight's scan."""

from __future__ import annotations

import json

import pytest

from poros import data as pdata

ROWS = [{"messages": [{"role": "user", "content": f"q{i}"},
                      {"role": "assistant", "content": f"a{i}"}]} for i in range(3)]


@pytest.fixture(params=["jsonl", "json"])
def files(tmp_path, request):
    text = ("".join(json.dumps(r) + "\n" for r in ROWS) if request.param == "jsonl"
            else json.dumps(ROWS))
    plain, bom = tmp_path / f"plain.{request.param}", tmp_path / f"bom.{request.param}"
    plain.write_text(text, encoding="utf-8")
    bom.write_text(text, encoding="utf-8-sig")
    assert bom.read_bytes()[:3] == b"\xef\xbb\xbf"
    return plain, bom


def test_a_bom_file_reads_as_its_rows(files):
    plain, bom = files
    assert pdata.detect(bom) == pdata.detect(plain) == "chats"
    assert [r for _, r in pdata._rows(bom)] == [r for _, r in pdata._rows(plain)] == ROWS


def test_the_preflight_scan_reads_a_bom_file_as_its_rows(files):
    from poros.preflight import data as pre

    plain, bom = files
    assert list(pre._read(bom)) == list(pre._read(plain))
    assert all(isinstance(r, dict) for r in pre._read(bom))
