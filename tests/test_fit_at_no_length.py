"""PorosTrainer.from_pretrained with no max_seq_length leaves the GPU fit at a length to train();
before the download it still refuses a run that fits the GPU at no length, asked at the shortest
length the plan reads (fit.SHORTEST_SEQ), and never one that could fit at some length. A
simulated card; nothing is downloaded or loaded."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from poros.core import card, compute, env  # noqa: E402
from poros.core.config import PorosConfig  # noqa: E402
from poros.core.errors import PorosVramError  # noqa: E402
from poros.plan import configs as plan_configs  # noqa: E402
from poros.preflight import fit as F  # noqa: E402
from poros.preflight import start as S  # noqa: E402
from tests.test_plan_upper import CONFIGS, RTX4090  # noqa: E402

MODEL = "Qwen/Qwen2.5-7B"


class Loaded(Exception):
    """from_pretrained got past every check before the download."""


@pytest.fixture
def card_of(monkeypatch):
    """A simulated CUDA card with ``free`` GiB free (an RTX 4090's compute), the 7B's config.json
    sizes, and the download and load stubbed to say they were reached."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.get_device_capability", lambda device=None: (8, 9))
    monkeypatch.setattr("torch.cuda.get_device_name", lambda device=None: "NVIDIA GeForce RTX 4090")
    monkeypatch.setattr("torch.cuda.get_arch_list", lambda: ["sm_80", "sm_86", "sm_89"])
    monkeypatch.setattr(compute, "triton_installed", lambda: True)
    monkeypatch.setattr(env, "host_ram_free", lambda: 200.0)
    monkeypatch.setattr(plan_configs, "read_model_config", lambda *a, **k: CONFIGS[MODEL])

    def route(*a, **k):
        raise Loaded

    monkeypatch.setattr(S, "route", route)  # the first step past the checks: the load

    def on(free):
        monkeypatch.setattr(card, "detect_gpu", lambda name=None, **kw: RTX4090.model_copy(
            update={"free_gib": free, "total_gib": free + 0.5, "source": "torch"}))
    return on


def _load(**kw):
    from poros.train.trainer import PorosTrainer

    return PorosTrainer.from_pretrained(MODEL, dataset="synthetic", lora_dropout=0.0, **kw)


def test_a_run_that_fits_at_no_length_is_refused_before_the_download(card_of):
    """Seven resident blocks hold 28 layers in NF4 (a floor no length lowers) past a 2 GiB card:
    refused with no max_seq_length set, before the load, with what would fit."""
    card_of(2.0)
    with pytest.raises(PorosVramError) as err:
        _load(resident_blocks=7)
    said = err.value.info.message
    assert f"max_seq_length {F.SHORTEST_SEQ}, the shortest the plan reads" in said, said
    assert "GiB" not in said, said  # the plan's figures stay in its reason, never said
    basis = err.value.basis
    assert "resident layers" in basis and "resident_blocks=" in basis
    # the advice, in words: what the plan's figures said, without them
    assert err.value.info.remediation.startswith(
        "What to change: fewer resident_blocks (0 keeps none on the GPU)")
    with pytest.raises(Loaded):  # check_fit=False goes on, as the refusal's remedy says
        _load(resident_blocks=7, check_fit=False)


def test_a_run_that_fits_at_a_shorter_length_is_not_refused(card_of):
    """On 6 GiB free the 7B is refused at 2048 (a measured run no larger peaked past it) but may
    fit at 128: with no length set, from_pretrained goes on to the load (train() asks at its own
    length); with max_seq_length 2048 set it refuses as before."""
    card_of(6.0)
    with pytest.raises(PorosVramError):
        _load(max_seq_length=2048)
    with pytest.raises(Loaded):
        _load()


def test_the_check_asks_a_copy_at_the_shortest_length(card_of):
    """The load still plans at its own length: the shortest-length question leaves the config
    as it was."""
    card_of(6.0)
    cfg = PorosConfig(model_name_or_path=MODEL, dataset="synthetic", lora_dropout=0.0)
    before = cfg.model_dump()
    assert F.at_no_length(cfg, "qwen2", CONFIGS[MODEL]) is None
    assert cfg.model_dump() == before
