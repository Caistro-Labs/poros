"""``PorosTrainer``'s Mac branches on a real tiny model, with ``mps`` faked on Linux.

The trainer believes it is on ``mps``; the tensors stay on CPU. So this checks the branches
(device pick, what a checkpoint holds, what the loop does and in what order), not Metal.
"""

from __future__ import annotations

import tempfile
import threading
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import poros.metal as metal  # noqa: E402
import poros.train.trainer as trainer_mod  # noqa: E402
from poros.core import checkpoints  # noqa: E402
from poros.core.config import PorosConfig  # noqa: E402
from poros.core.errors import (  # noqa: E402
    PorosConfigError,
    PorosGuaranteeUnavailableError,
    PorosOOMError,
)
from poros.metal import machine, watchdog  # noqa: E402
from tests.test_train_loop_cpu import _loader, _make_trainer, _tiny_peft_qwen2  # noqa: E402

M4 = "apple-m4-10gpu-16gb"
MPS = torch.device("mps")


@pytest.fixture
def darwin(monkeypatch):
    monkeypatch.setattr(trainer_mod, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def fake_mps(monkeypatch):
    gen = torch.Generator().manual_seed(3)
    monkeypatch.setattr(torch.mps, "get_rng_state", lambda device="mps": gen.get_state())
    monkeypatch.setattr(
        torch.mps, "set_rng_state", lambda state, device="mps": gen.set_state(state)
    )
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    monkeypatch.setattr(metal, "device_class", lambda: M4)
    return gen


def _trainer(d, *, dropout=0.0, grad_accum=1, max_steps=3):
    model, cfg = _tiny_peft_qwen2(torch, dropout=dropout)
    tr = _make_trainer(torch, model, d, grad_accum=grad_accum, max_steps=max_steps)
    tr.config.lora_dropout = dropout
    tr._save_tokenizer = lambda out_path: None  # "dummy" would ask the Hub
    return tr, cfg


def _on_mps(tr, monkeypatch):
    """Report mps; the batch stays where it is, since the tensors stay on CPU."""
    tr.device = MPS
    monkeypatch.setattr(tr, "_move_batch_to_device", lambda batch: batch)
    sample = machine.MemSample(allocated=1, driver=2, footprint=3, swap=4, pressure=1)
    monkeypatch.setattr(machine, "memory_sample", lambda device: sample)
    idle = SimpleNamespace(fired=threading.Event(), stop=lambda: None, swap_growth=0)
    monkeypatch.setattr(watchdog, "Watchdog", lambda guard: idle)


# ---------------------------------------------------------------------------
# Device pick.
# ---------------------------------------------------------------------------


def test_a_mac_goes_through_the_gate_to_mps(darwin, monkeypatch):
    seen = []
    monkeypatch.setattr(metal, "require", seen.append)
    cfg = PorosConfig()
    assert trainer_mod.PorosTrainer(cfg).device == MPS
    assert seen == [cfg]


def test_a_mac_never_falls_back_to_cpu(darwin, monkeypatch):
    def refuse(config):
        raise PorosGuaranteeUnavailableError("no")

    monkeypatch.setattr(metal, "require", refuse)
    with pytest.raises(PorosGuaranteeUnavailableError):
        trainer_mod.PorosTrainer(PorosConfig())


def test_off_darwin_the_gate_is_not_called(monkeypatch):
    def boom(config):
        raise AssertionError("require called off Darwin")

    monkeypatch.setattr(metal, "require", boom)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert trainer_mod.PorosTrainer(PorosConfig()).device.type == "cpu"


def test_cuda_stays_first_even_on_darwin(darwin, monkeypatch):
    def boom(config):
        raise AssertionError("require called with CUDA present")

    monkeypatch.setattr(metal, "require", boom)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert trainer_mod.PorosTrainer(PorosConfig()).device == torch.device("cuda:0")


# ---------------------------------------------------------------------------
# block_size on mps, from from_pretrained.
# ---------------------------------------------------------------------------


class _Stop(Exception):
    """Past the block_size rule: the rest of setup is not under test."""


def _from_pretrained(monkeypatch, **kwargs):
    monkeypatch.setattr(metal, "require", lambda config: None)
    monkeypatch.setattr("poros.preflight.arch.cheap_model_type", lambda *a: None)
    monkeypatch.setattr(trainer_mod, "enforce_determinism", lambda **k: None)
    monkeypatch.setattr(trainer_mod.torch, "manual_seed", lambda s: None)
    monkeypatch.setattr(
        trainer_mod.PorosModel,
        "from_pretrained",
        lambda *a, **k: SimpleNamespace(
            model=torch.nn.Linear(1, 1),
            peft_model=None,
            adapter_backend="peft",
            adapter_summary=None,
        ),
    )

    def stop(self):
        raise _Stop

    monkeypatch.setattr(trainer_mod.PorosTrainer, "_setup_blockwise_training", stop)
    cfg = PorosConfig(blockwise_arch_check="off", **kwargs)
    with pytest.raises(_Stop):
        trainer_mod.PorosTrainer.from_pretrained("x", config=cfg)
    return cfg


def test_block_size_defaults_to_1_on_mps(darwin, monkeypatch):
    assert _from_pretrained(monkeypatch).block_size == 1
    assert _from_pretrained(monkeypatch, block_size=1).block_size == 1


def test_off_mps_the_default_is_still_the_validated_one(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert _from_pretrained(monkeypatch).block_size == trainer_mod._DEFAULT_BLOCK_SIZE
    assert _from_pretrained(monkeypatch, block_size=2).block_size == 2


@pytest.mark.parametrize("model_device,config_device", [("mps", "cpu"), ("cpu", "mps")])
def test_prepare_moe_refuses_mps_before_touching_the_model(model_device, config_device):
    from poros.moe.prepare import MoeConfig, prepare_moe

    class Model:
        def parameters(self):
            return iter([SimpleNamespace(device=torch.device(model_device))])

    with pytest.raises(PorosGuaranteeUnavailableError, match="MoE"):
        prepare_moe(Model(), MoeConfig(device=config_device))


# ---------------------------------------------------------------------------
# Checkpoints.
# ---------------------------------------------------------------------------


def _saved_on_mps(d, fake_mps, monkeypatch, name="ckpt", **kw):
    tr, _ = _trainer(d, **kw)
    tr.device = MPS
    ckpt = Path(d) / name
    tr.save(str(ckpt))
    tr.device = torch.device("cpu")
    return ckpt


def test_a_mac_checkpoint_holds_its_rng_and_class(fake_mps, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        ckpt = _saved_on_mps(d, fake_mps, monkeypatch)
        rng = torch.load(ckpt / "rng_state.pt", weights_only=True)
        assert torch.equal(rng["mps_rng"], fake_mps.get_state())
        assert rng["device_class"] == M4


def _saves_at_8(d, monkeypatch, *, metal, limit, before):
    """What is on disk as a save at step 8 starts, and once it is done."""
    def on_disk():
        return sorted(int(c.name.split("-")[1]) for c in Path(d).glob("checkpoint-*"))

    tr, _ = _trainer(d)
    tr._metal, tr.config.save_total_limit, tr.global_step = metal, limit, 8
    for step in before:
        ckpt = Path(d) / f"checkpoint-{step}"
        ckpt.mkdir()
        (ckpt / "adapter_config.json").write_text("{}")
        (ckpt / "adapter_model.safetensors").write_bytes(b"0")
        (ckpt / ".poros_run").write_text(tr._run_id)  # this run's own (pruning reads no other's)
    seen, real = [], tr.save
    monkeypatch.setattr(tr, "save",
                        lambda path, **kw: (seen.append(on_disk()), real(path, **kw)))
    tr._save_checkpoint()
    return seen[0], on_disk()


@pytest.mark.parametrize(
    "limit, before, during, after",
    [(3, [2, 4, 6], [4, 6], [4, 6, 8]), (1, [6], [6], [8]), (3, [8], [8], [8]),
     (0, [2, 4], [2, 4], [2, 4, 8])],
)
def test_a_mac_makes_room_before_it_saves_and_keeps_the_newest(
    limit, before, during, after, monkeypatch
):
    with tempfile.TemporaryDirectory() as d:
        got = _saves_at_8(d, monkeypatch, metal=True, limit=limit, before=before)
        assert got == (during, after)


def test_cuda_saves_first_and_prunes_after_as_it_always_has(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        got = _saves_at_8(d, monkeypatch, metal=False, limit=3, before=[2, 4, 6])
        assert got == ([2, 4, 6], [4, 6, 8])


def test_a_linux_checkpoint_is_as_it_was(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("torch.mps touched off mps")

    monkeypatch.setattr(torch.mps, "get_rng_state", boom)
    with tempfile.TemporaryDirectory() as d:
        tr, _ = _trainer(d)
        ckpt = Path(d) / "ckpt"
        tr.save(str(ckpt))
        assert (ckpt / checkpoints.FORMAT_MARKER).read_text() == "1"
        rng = torch.load(ckpt / "rng_state.pt", weights_only=True)
        # the CPU generator, and the CUDA one on a host that has CUDA, as always: no mps state
        assert set(rng) == {"torch_rng", "global_step",
                            *(["cuda_rng"] if torch.cuda.is_available() else [])}


def test_resume_on_the_same_class_restores_the_mps_rng(fake_mps, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        ckpt = _saved_on_mps(d, fake_mps, monkeypatch)
        saved = fake_mps.get_state()
        torch.rand(3, generator=fake_mps)  # the stream moves on
        tr, _ = _trainer(d)
        tr.device = MPS
        tr.optimizer, tr._lr_scheduler = None, None  # map_location=mps cannot load on Linux
        tr.load_checkpoint(str(ckpt))
        assert torch.equal(fake_mps.get_state(), saved)


def _rewrite_rng(ckpt, *drop):
    rng = torch.load(ckpt / "rng_state.pt", weights_only=True)
    for key in drop:
        del rng[key]
    torch.save(rng, ckpt / "rng_state.pt")


def _resume_on_mps(d, ckpt, **flags):
    tr, _ = _trainer(d)
    tr.device = MPS
    tr.optimizer, tr._lr_scheduler = None, None
    tr.load_checkpoint(str(ckpt), **flags)


def test_another_device_class_is_refused_at_resume_only(fake_mps, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        ckpt = _saved_on_mps(d, fake_mps, monkeypatch)
        monkeypatch.setattr(metal, "device_class", lambda: "apple-m4-10gpu-24gb")
        with pytest.raises(PorosConfigError, match="same device class"):
            _resume_on_mps(d, ckpt)
        _resume_on_mps(d, ckpt, load_optimizer=False, load_rng=False)


def test_a_checkpoint_with_no_mps_generator_state_is_refused(fake_mps, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        ckpt = _saved_on_mps(d, fake_mps, monkeypatch)
        _rewrite_rng(ckpt, "mps_rng")
        with pytest.raises(PorosConfigError, match="no MPS generator state"):
            _resume_on_mps(d, ckpt)


def test_a_mac_checkpoint_is_refused_off_a_mac(fake_mps, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        ckpt = _saved_on_mps(d, fake_mps, monkeypatch)
        tr, _ = _trainer(d)
        with pytest.raises(PorosConfigError, match="same device class") as exc:
            tr.load_checkpoint(str(ckpt))
        assert "the adapter exports anywhere" in str(exc.value)
        tr.load_checkpoint(str(ckpt), load_optimizer=False, load_rng=False)


def test_a_mac_refuses_a_checkpoint_with_no_device_class(fake_mps, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        ckpt = _saved_on_mps(d, fake_mps, monkeypatch)
        _rewrite_rng(ckpt, "device_class")
        with pytest.raises(PorosConfigError, match="same device class"):
            _resume_on_mps(d, ckpt)


def test_a_mac_refuses_a_cuda_checkpoint(fake_mps, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        tr, _ = _trainer(d)
        ckpt = Path(d) / "from_linux"
        tr.save(str(ckpt))
        with pytest.raises(PorosConfigError, match="same device class"):
            _resume_on_mps(d, ckpt)


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------


def _alive_at_each_forward(tr, monkeypatch):
    """Saved boundaries still alive as each microstep's forward starts. No gc.collect:
    this one pool cannot wait for the cyclic collector."""
    alive, refs = [], []
    real_forward = tr._forward_blockwise_adapter_model
    real_layers = tr._forward_block_layers

    def forward(batch):
        alive.append(sum(r() is not None for r in refs))
        return real_forward(batch)

    def layers(block_id, hidden, *a, **k):
        refs.append(weakref.ref(hidden))
        return real_layers(block_id, hidden, *a, **k)

    monkeypatch.setattr(tr, "_forward_blockwise_adapter_model", forward)
    monkeypatch.setattr(tr, "_forward_block_layers", layers)
    return alive


def test_boundaries_are_freed_after_backward_on_mps(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        tr, cfg = _trainer(d, max_steps=3)
        _on_mps(tr, monkeypatch)
        alive = _alive_at_each_forward(tr, monkeypatch)
        tr.train(_loader(torch, cfg.vocab_size), steps=3)
        assert alive == [0, 0, 0]


def test_the_count_sees_a_boundary_something_holds(monkeypatch):
    """The control: that the count above can see a boundary kept alive."""
    with tempfile.TemporaryDirectory() as d:
        tr, cfg = _trainer(d, max_steps=3)
        _on_mps(tr, monkeypatch)
        alive = _alive_at_each_forward(tr, monkeypatch)
        held, counted = [], tr._forward_block_layers

        def holding(block_id, hidden, *a, **k):
            held.append(hidden)
            return counted(block_id, hidden, *a, **k)

        monkeypatch.setattr(tr, "_forward_block_layers", holding)
        tr.train(_loader(torch, cfg.vocab_size), steps=3)
        assert alive[0] == 0 and alive[1] > 0


def _oom_at(tr, monkeypatch, attr, exc):
    def raises(*a, **k):
        raise exc

    monkeypatch.setattr(tr, attr, raises)


@pytest.mark.parametrize("exc", [
    RuntimeError("MPS backend out of memory (MPS allocated: 11 GiB, other allocations: 1 GiB)."),
    torch.OutOfMemoryError("Insufficient Memory"),
], ids=["allocator", "command-buffer"])
@pytest.mark.parametrize("attr", ["_forward_blockwise_adapter_model", "_adapter_optimizer_step"])
def test_both_oom_forms_map_wherever_they_arrive(fake_mps, monkeypatch, exc, attr):
    with tempfile.TemporaryDirectory() as d:
        tr, cfg = _trainer(d)
        _on_mps(tr, monkeypatch)
        _oom_at(tr, monkeypatch, attr, exc)
        with pytest.raises(PorosOOMError) as raised:
            tr.train(_loader(torch, cfg.vocab_size), steps=3)
        assert "This Mac ran out of memory at step 0" in str(raised.value)
        assert "poros runs continue" in raised.value.info.remediation
        assert "block_size" not in str(raised.value)
        assert raised.value.info.fix == {}
        assert raised.value.__cause__ is exc


def test_any_other_runtime_error_passes_through_on_mps(fake_mps, monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        tr, cfg = _trainer(d)
        _on_mps(tr, monkeypatch)
        exc = RuntimeError("a shape mismatch")
        _oom_at(tr, monkeypatch, "_forward_blockwise_adapter_model", exc)
        with pytest.raises(RuntimeError) as raised:
            tr.train(_loader(torch, cfg.vocab_size), steps=3)
        assert raised.value is exc


def test_off_mps_nothing_maps_differently(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        tr, cfg = _trainer(d)
        exc = RuntimeError("MPS backend out of memory")
        _oom_at(tr, monkeypatch, "_forward_blockwise_adapter_model", exc)
        with pytest.raises(RuntimeError) as raised:
            tr.train(_loader(torch, cfg.vocab_size), steps=3)
        assert raised.value is exc
        tr, cfg = _trainer(d)
        _oom_at(tr, monkeypatch, "_forward_blockwise_adapter_model", torch.OutOfMemoryError("x"))
        with pytest.raises(PorosOOMError) as raised:
            tr.train(_loader(torch, cfg.vocab_size), steps=3)
        assert "block_size" in raised.value.info.remediation


# ---------------------------------------------------------------------------
# The live memory bar.
# ---------------------------------------------------------------------------


def test_the_live_report_reads_the_one_pool(tmp_path, monkeypatch):
    tr, _ = _trainer(str(tmp_path))
    assert tr._step_report(1.0, 1.0).vram_gb == 0.0
    _on_mps(tr, monkeypatch)
    sample = machine.MemSample(allocated=1, driver=7_000_000_000, footprint=9_000_000_000)
    monkeypatch.setattr(machine, "memory_sample", lambda device: sample)
    report = tr._step_report(1.0, 1.0)
    assert (report.vram_gb, report.vram_peak_gb) == (7.0, 9.0)
