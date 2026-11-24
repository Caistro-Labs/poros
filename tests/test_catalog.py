"""poros/catalog.toml: every name Poros offers is backed by a committed receipt."""

from __future__ import annotations

import pytest

from poros.cli.words.models import by_id, catalog
from poros.core.arch_registry import validated_arch_types
from tests.benchmark_logs import benchmark_log

ENTRIES = catalog()


def test_ids_are_unique():
    ids = [e.id.lower() for e in ENTRIES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("e", ENTRIES, ids=[e.id for e in ENTRIES])
def test_entry_is_backed(e):
    assert e.entry in ("dense", "moe") and e.variant in ("base", "instruct", "coder")
    assert e.arch in validated_arch_types(e.entry if e.entry == "moe" else "dense"), e.arch
    if e.evidence in ("validated", "measured"):  # a parity receipt, or a measured run, of it
        assert not e.via
    elif e.evidence == "none":
        assert not e.via
    else:
        assert e.evidence == "same-shape"
        base = by_id(e.via)
        assert base is not None and base.evidence in ("validated", "measured"), e.via
        assert (base.arch, base.size, base.entry) == (e.arch, e.size, e.entry)
    # the benchmark logs last: without them (poros-benchmarks) the checks above still ran
    if e.evidence in ("validated", "measured"):
        receipt = benchmark_log(e.receipt)
        assert receipt.is_file(), e.receipt
        assert e.id in receipt.read_text(), f"{e.receipt} never names {e.id}"
    if e.receipt and e.evidence != "validated":
        assert benchmark_log(e.receipt).is_file()


@pytest.mark.parametrize(
    "e, row",
    [(e, r) for e in ENTRIES for r in e.metal if r.receipt],
    ids=lambda x: getattr(x, "receipt", getattr(x, "id", "")),
)
def test_a_mac_row_with_a_receipt_is_that_receipt(e, row):
    import json

    from poros.metal.table import class_of
    from poros.metal.verify import RECEIPT_SCHEMA

    receipt = json.loads(benchmark_log(row.receipt).read_text())
    pin = receipt["pin"]
    assert receipt["pass"] is True and receipt["schema"] == RECEIPT_SCHEMA
    assert row.device_class.startswith(class_of(f"{pin['chip']} {pin['gpu_cores']}-core GPU"))
    assert receipt["model"] == e.id and (receipt["seq"], receipt["g"]) == (row.seq, row.grad_accum)
