"""poros eval and the host-RAM estimate: where the run's own receipt measured less, when it
trained, than is available now (with room to spare), that measurement stands and eval says so
once; anywhere else the refusal is today's, with eval's remedy."""

from __future__ import annotations

import json

import pytest

from poros.core.config import PorosConfig
from poros.core.errors import PorosHostRamError

MOE = "Qwen/Qwen3-30B-A3B-Instruct-2507"
REMEDY = "close other programs, then run poros eval again."


def _cfg(model: str = MOE, store: str = "ssd") -> PorosConfig:
    where = {"store_path": "/nonexistent/sss"} if store == "ssd" else {}
    return PorosConfig(model_name_or_path=model, dataset="synthetic", store=store,
                       path="sequential", **where)


def _run(tmp_path, *, peak_gb=7.305, model=MOE, store="ssd", quantization="nf4", **receipt):
    """A run folder as poros train leaves it: its resolved settings and its receipt."""
    (tmp_path / "config.resolved.json").write_text(json.dumps(
        {"model_name_or_path": model, "store": store, "quantization": quantization}))
    if peak_gb is not ...:
        (tmp_path / "receipt.json").write_text(json.dumps({"peak_host_ram_gb": peak_gb,
                                                           **receipt}))
    return str(tmp_path)


@pytest.fixture
def builds(monkeypatch):
    """PorosTrainer.from_pretrained as eval calls it: the host check refuses (an estimate of
    9.7 GiB against what is ``free``) unless it is off. Each call's keyword arguments are kept."""
    pytest.importorskip("torch")  # the trainer eval builds is torch's
    from poros.train.trainer import PorosTrainer

    calls, free = [], {"gib": 9.5}

    class Built:
        _event_writer = "events"

    def build(model, **kw):
        calls.append(kw)
        if kw.get("check_host_ram", True):
            raise PorosHostRamError(model, 9.7, free["gib"], "estimated from config.json",
                                    path="sequential")
        return Built()

    monkeypatch.setattr(PorosTrainer, "from_pretrained", build)
    return calls, free


def test_a_run_that_measured_less_than_is_available_loads_and_says_so(tmp_path, builds):
    """A run whose receipt measured 7.305 GB (6.8 GiB), the estimate 9.7 GiB,
    9.5 GiB available: the engine is built with the host check off and one line says why."""
    from poros.cli.verbs import evaluate_pass as ep

    calls, _ = builds
    said = []
    trainer = ep.engine(_cfg(), _run(tmp_path), said.append)
    assert calls[-1].get("check_host_ram") is False and trainer._event_writer is None
    assert said == ["9.5 GiB of host RAM is available, and this run used 6.8 GiB when it "
                    "trained, so eval goes ahead."]


def test_a_run_whose_measurement_does_not_fit_now_is_refused_with_evals_remedy(tmp_path,
                                                                               builds):
    """(2) The same run with 6.0 GiB available: 6.8 GiB and the margin do not fit; refused."""
    from poros.cli.verbs import evaluate_pass as ep

    calls, free = builds
    free["gib"] = 6.0
    said = []
    with pytest.raises(PorosHostRamError) as found:
        ep.engine(_cfg(), _run(tmp_path), said.append)
    assert found.value.info.remediation.endswith(REMEDY) and not said
    assert all(kw.get("check_host_ram", True) for kw in calls)


@pytest.mark.parametrize("folder", [
    dict(peak_gb=...),  # no receipt
    dict(peak_gb=None),  # a receipt without the figure (a Mac's)
    dict(peak_gb="7.3"),  # not a number
    dict(model="Qwen/Qwen3-30B-A3B"),  # another model
    dict(store="ram"),  # another store
    dict(quantization="int8"),  # another quantization
    dict(simulated=True),  # a preview: it measured nothing real
])
def test_without_the_runs_own_figure_the_refusal_is_todays(tmp_path, builds, folder):
    """(3) No usable figure, from this model, quantization and store: refused as today."""
    from poros.cli.verbs import evaluate_pass as ep

    calls, _ = builds
    said = []
    with pytest.raises(PorosHostRamError) as found:
        ep.engine(_cfg(), _run(tmp_path, **folder), said.append)
    assert found.value.info.remediation.endswith(REMEDY) and not said and len(calls) == 1


def test_an_unreadable_receipt_changes_nothing(tmp_path, builds):
    from poros.cli.verbs import evaluate_pass as ep

    _run(tmp_path)
    (tmp_path / "receipt.json").write_text("{not json")
    with pytest.raises(PorosHostRamError):
        ep.engine(_cfg(), str(tmp_path), [].append)


def test_a_dense_run_that_fits_is_built_as_before(tmp_path, monkeypatch):
    """(4) A dense run the host check passes: one build, the check left on, nothing said, and
    its receipt not read."""
    pytest.importorskip("torch")
    from poros.cli.verbs import evaluate_pass as ep
    from poros.train.trainer import PorosTrainer

    calls, said = [], []

    class Built:
        _event_writer = "events"

    monkeypatch.setattr(PorosTrainer, "from_pretrained",
                        lambda model, **kw: calls.append(kw) or Built())
    monkeypatch.setattr(ep, "measured_host", lambda *a: pytest.fail("receipt read"))
    ep.engine(_cfg("Qwen/Qwen2.5-0.5B", "ram"), _run(tmp_path, model="Qwen/Qwen2.5-0.5B",
                                                     store="ram"), said.append)
    assert len(calls) == 1 and "check_host_ram" not in calls[0] and not said


def test_the_receipts_gb_is_read_as_decimal_and_given_in_gib(tmp_path):
    """peak_host_ram_gb is decimal GB (poros.train.reporting): 7.305 GB is 6.8 GiB."""
    from poros.cli.verbs import evaluate_pass as ep

    assert ep.measured_host(_run(tmp_path), _cfg()) == 6.8
