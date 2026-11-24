"""The whole Mac path through the real ``PorosTrainer``, with the device left on the CPU.

Source, residency, load, witness and watchdog as the trainer wires them. Nothing here says
anything about Metal: ``scripts/metal/arms.py`` runs this same path there.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("accelerate")

import peft  # noqa: E402
from transformers import Qwen2ForCausalLM  # noqa: E402

from poros import metal as metal_gate  # noqa: E402
from poros import recipes  # noqa: E402
from poros.core import blockwise, checkpoints  # noqa: E402
from poros.core.config import PorosConfig  # noqa: E402
from poros.core.errors import (  # noqa: E402
    EXIT_CODES,
    PorosCancelledError,
    PorosDiskError,
    PorosOOMError,
    PorosStepCheckError,
)
from poros.metal import machine, watchdog  # noqa: E402
from poros.metal import source as source_mod  # noqa: E402
from poros.metal.capture import capture  # noqa: E402
from poros.metal.head import ChunkedHead, EmbedRows, FileRows  # noqa: E402
from poros.metal.load import build  # noqa: E402
from poros.metal.nf4 import Nf4Linear  # noqa: E402
from poros.metal.pack import pack  # noqa: E402
from poros.metal.residency import UnifiedBlockManager  # noqa: E402
from poros.metal.source import FileLayers, RamLayers  # noqa: E402
from poros.metal.verify import Transcript, first_difference  # noqa: E402
from poros.sss.format import SssCorruptError, SssFile  # noqa: E402
from poros.train import model as model_mod  # noqa: E402
from poros.train import trainer as trainer_mod  # noqa: E402
from poros.train.batch import Batch, Datum  # noqa: E402
from poros.train.run import Run  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402
from tests.conftest import cpu_trainer_pick, needs_metal_on_a_mac  # noqa: E402
from tests.metal import nf4_tiny as tiny  # noqa: E402

arms = tiny.arms
CPU = torch.device("cpu")
G = 2
STEPS = 4
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("mac") / "tiny-qwen2"
    tiny.write_checkpoint(path, tiny.tiny_config())
    return path


@pytest.fixture(scope="module")
def store_file(checkpoint, tmp_path_factory):
    store = tmp_path_factory.mktemp("store")
    packed = pack(checkpoint, store, model_id="tiny", revision="abcdef01")
    return packed / "model.sss"


def _config(checkpoint, out, *, store="ram", dropout=0.05, **kwargs) -> PorosConfig:
    fields = dict(
        model_name_or_path=str(checkpoint),
        store=store,
        store_path=str(Path(out) / "store"),
        gradient_accumulation_steps=G,
        max_seq_length=16,
        lora_dropout=dropout,
        learning_rate=1e-3,
        warmup_steps=0,
        save_steps=0,
        eval_steps=0,
        output_dir=str(out),
        progress=False,
    )
    return PorosConfig(**{**fields, **kwargs})


def _trainer(checkpoint, out, **kwargs):
    config = _config(checkpoint, out, **kwargs)
    trainer = arms.CpuDouble.from_pretrained(config.model_name_or_path, config=config)
    torch.set_num_threads(1)
    return trainer


def _gate(trainer) -> None:
    manager = trainer.block_manager
    manager.source = arms.GateLayers(manager.source.path, CPU)
    arms.poison_released(manager, manager.source)


def _data(trainer):
    return arms.token_stream(trainer.model.config.vocab_size, 16, G * STEPS)


def _finals(trainer) -> dict:
    params = {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}
    return {
        "adapter": capture(params)["b2"],
        "adam": capture(trainer.optimizer.state_dict())["b2"],
        "scheduler": capture(trainer._lr_scheduler.state_dict())["b2"],
    }


@pytest.fixture(scope="module")
def reference(checkpoint, tmp_path_factory):
    """One run on the RAM tier: what every other run here must match, byte for byte."""
    with pytest.MonkeyPatch.context() as mp:  # module-scoped: before the autouse CPU pick
        cpu_trainer_pick(mp)
        trainer = _trainer(checkpoint, tmp_path_factory.mktemp("reference"))
        trainer.train(_data(trainer), steps=STEPS)
        return _finals(trainer), trainer.history.losses


# -- sources ----------------------------------------------------------------------------------


def _bytes_of(path, layer) -> torch.Tensor:
    with SssFile.open(path) as file:
        out = torch.empty(file.entry((layer, 0)).nbytes, dtype=torch.uint8)
        file.read_into((layer, 0), out)
    return out


@pytest.fixture
def syncs(monkeypatch):
    calls = []
    monkeypatch.setattr(source_mod, "sync", calls.append)
    return calls


def test_ram_holds_every_layer_and_syncs_at_each_visit(store_file, syncs):
    layers = RamLayers(store_file, CPU)
    assert sorted(layers.slabs) == [0, 1, 2, 3]
    for layer in range(4):
        assert torch.equal(layers.acquire(layer), _bytes_of(store_file, layer))
    assert syncs == [CPU] * 4


def test_the_reader_thread_touches_cpu_memory_only(store_file, monkeypatch):
    layers = FileLayers(store_file, CPU)
    real = layers.source.read_into
    seen = []

    def spy(key, dst):
        seen.append((threading.current_thread(), dst.device.type))
        real(key, dst)

    monkeypatch.setattr(layers.source, "read_into", spy)
    layers.prefetch(0)
    assert torch.equal(layers.acquire(0), _bytes_of(store_file, 0))
    (thread, device), = seen
    assert thread is not threading.main_thread()
    assert thread.name.startswith("poros-read")
    assert device == "cpu"


def _count_reads(layers, monkeypatch) -> list:
    reads, real = [], layers.source.read_into

    def counted(key, dst):
        reads.append(key[0])
        real(key, dst)

    monkeypatch.setattr(layers.source, "read_into", counted)
    return reads


def _microstep(manager, n=4):
    for layer in [*range(n), *reversed(range(n))]:
        manager.load_block_for_inference(layer)
        manager.release_block(layer)


def _manager(store_file, source):
    """The real manager over the real skeleton, with ``source`` as its bytes."""
    model = build(store_file.parent, tiny.tiny_config())
    model.poros_source = source
    return UnifiedBlockManager(model, block_size=1)


def test_a_microstep_reads_each_layer_twice_but_for_the_turns(store_file, monkeypatch, syncs):
    layers = FileLayers(store_file, CPU)
    reads = _count_reads(layers, monkeypatch)
    manager = _manager(store_file, layers)
    _microstep(manager)
    _microstep(manager)
    layers.reading.result()
    # 3 is warm at the turn, and 0 is warm from one microstep to the next.
    assert reads == [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0]
    assert syncs == [CPU] * 16  # every visit: the 3 warm hits, and each of the 13 reads


def test_a_read_commits_the_queued_work_before_it_waits_on_the_reader(store_file, monkeypatch):
    """What is already queued runs on the device while the next layer is read: ``acquire`` starts
    the read, commits the queue, and only then waits on the reader."""
    layers = FileLayers(store_file, CPU)
    order, synced, real = [], threading.Event(), layers.source.read_into

    def held_read(key, dst):  # the read cannot end before the queue is committed (or 2 s pass)
        synced.wait(2)
        real(key, dst)
        order.append("read lands")

    def sync(device):
        order.append(("sync", layers.landed, layers.in_slot))
        synced.set()

    monkeypatch.setattr(layers.source, "read_into", held_read)
    monkeypatch.setattr(source_mod, "sync", sync)
    assert torch.equal(layers.acquire(2), _bytes_of(store_file, 2))
    # committed while the read was under way and the layer not yet in the slot: a sync placed
    # after the wait on the reader would come second here
    assert order == [("sync", 2, None), "read lands"]


def test_the_file_layers_count_their_reads_as_the_ssd_store_does(store_file, syncs):
    """A microstep up and down, twice: every read but the first was asked for ahead (a prefetch
    hit); the warm turns are no read at all; the bytes are the layers'; a pass is a mark."""
    layers = FileLayers(store_file, CPU)
    manager = _manager(store_file, layers)
    for _ in range(2):
        _microstep(manager)
        layers.mark_step()
    layers.reading.result()
    st = layers.stats
    sizes = {k[0]: layers.source.nbytes(k) for k in [(i, 0) for i in range(4)]}
    order = [0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0]  # what was read (3 and 0 warm at the turns)
    assert st["reads"] == len(order) and st["sync_reads"] == 1 and st["prefetch_hits"] == 12
    assert st["bytes_read"] == sum(sizes[i] for i in order) and st["steps"] == 2
    assert st["prefetch_dropped"] == 0 and st["cache_hits"] == 0
    assert st["blocked_s"] >= 0 and st["depth"] == 1 and not st["pinned"] and not st["cache"]
    layers.prefetch(2)  # a wrong guess, as an evaluation makes: dropped, and 1 read on demand
    layers.acquire(1)
    assert layers.stats["prefetch_dropped"] == 1 and layers.stats["sync_reads"] == 2


def test_a_gate_run_reads_again_at_the_turns_and_poisons(store_file, monkeypatch):
    layers = arms.GateLayers(store_file, CPU)
    reads = _count_reads(layers, monkeypatch)
    manager = _manager(store_file, layers)
    arms.poison_released(manager, layers)
    _microstep(manager)
    layers.reading.result()
    assert reads == [0, 1, 2, 3, 3, 2, 1, 0, 0]  # the last one reads ahead for the next microstep
    assert bool((layers.slot == arms.POISON).all())
    manager.load_block_for_inference(0)
    assert torch.equal(layers.slot[: _bytes_of(store_file, 0).numel()], _bytes_of(store_file, 0))


def test_any_order_serves_the_right_bytes(store_file):
    """A wrong guess, as an evaluation makes, costs a read and never a wrong byte."""
    layers = FileLayers(store_file, CPU)
    for layer in (2, 0, 0, 3, 1, 2):
        layers.prefetch((layer + 1) % 4)
        assert torch.equal(layers.acquire(layer), _bytes_of(store_file, layer))


def test_a_bad_slab_is_refused_on_its_first_read(store_file, tmp_path):
    bad = tmp_path / "model.sss"
    bad.write_bytes(store_file.read_bytes())
    Path(f"{bad}.idx").write_bytes(Path(f"{store_file}.idx").read_bytes())
    with SssFile.open(bad) as file:
        offset = file.entry((1, 0)).offset
    with open(bad, "r+b") as f:
        f.seek(offset + 100)
        f.write(bytes([f.read(1)[0] ^ 1]))
        f.flush()
    with pytest.raises(SssCorruptError):
        FileLayers(bad, CPU).acquire(1)
    with pytest.raises(SssCorruptError):
        RamLayers(bad, CPU)


# -- residency ---------------------------------------------------------------------------------


def test_the_schedule_is_up_and_down_and_finds_itself_after_an_evaluation(store_file):
    source = SimpleNamespace(acquire=lambda layer: None, prefetch=None, release=lambda layer: None)
    asked = []
    source.prefetch = asked.append
    manager = _manager(store_file, source)
    _microstep(manager)
    assert asked == [1, 2, 3, 3, 2, 1, 0, 0]
    asked.clear()
    for layer in range(4):  # an evaluation only goes up
        manager.load_block_for_inference(layer)
    manager.load_block_for_inference(0)
    assert asked == [1, 2, 3, 3, 1]


def test_loading_points_the_linears_and_releasing_lets_go(checkpoint, tmp_path):
    trainer = _trainer(checkpoint, tmp_path)
    manager = trainer.block_manager
    assert isinstance(manager, UnifiedBlockManager)
    linears = manager._linears[2]
    assert len(linears) == 7 and all(isinstance(m, Nf4Linear) for m in linears)
    manager.load_block_for_inference(2)
    assert all(m.slab is manager.source.slabs[2] for m in linears)
    manager.release_block(2)
    assert all(m.slab is None for m in linears)


# -- load --------------------------------------------------------------------------------------


@pytest.mark.parametrize("store", ["ram", "ssd"])
def test_from_pretrained_takes_the_mac_route(checkpoint, tmp_path, monkeypatch, store):
    def no_bnb(*args, **kwargs):
        raise AssertionError("the Mac path loads no bitsandbytes model")

    monkeypatch.setattr(model_mod, "_load_model", no_bnb)
    monkeypatch.setattr(model_mod, "_prepare_model_for_kbit_training", no_bnb)
    trainer = _trainer(checkpoint, tmp_path, store=store)
    model = trainer.model
    assert model.config._attn_implementation == "eager"
    assert not [n for n, p in model.named_parameters() if p.is_meta]
    assert isinstance(model.lm_head, ChunkedHead)
    assert isinstance(model.model.embed_tokens, EmbedRows)
    assert isinstance(model.model.embed_tokens.rows, FileRows) == (store == "ssd")
    assert isinstance(model.poros_source, FileLayers if store == "ssd" else RamLayers)
    # the SSD store's reads are recorded as a CUDA run's are; the RAM tier has none
    assert trainer._sss_source is (model.poros_source if store == "ssd" else None)
    assert trainer.config.block_size == 1
    assert trainer.block_manager.num_blocks == 4
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert len(trainable) == 2 * 7 * 4 and all("lora_" in n for n in trainable)


def test_the_skeleton_holds_the_checkpoints_small_tensors(checkpoint, tmp_path):
    """Every arm is built by the same ``build``, so no comparison between arms can see this."""
    vanilla = Qwen2ForCausalLM.from_pretrained(checkpoint, dtype=torch.bfloat16).state_dict()
    model = _trainer(checkpoint, tmp_path).model
    small = [(n, p) for n, p in model.named_parameters() if not p.requires_grad and p.numel()]
    assert len(small) == 4 * 2 + 1 + 4 * 3  # two norms a layer, the final norm, q, k and v biases
    for name, param in small:
        same = vanilla[name.replace(".base_layer", "")]
        assert torch.equal(param.view(torch.int16), same.view(torch.int16)), name


def test_a_tied_model_keeps_one_copy_of_its_rows(tmp_path):
    source = tmp_path / "tied"
    tiny.write_checkpoint(source, tiny.tiny_config(tied=True))
    model = _trainer(source, tmp_path).model
    assert model.lm_head.rows is model.model.embed_tokens.rows


def test_a_rewritten_local_checkpoint_is_packed_again(checkpoint, tmp_path):
    source = tmp_path / "local"
    tiny.write_checkpoint(source, tiny.tiny_config())
    first = _trainer(source, tmp_path / "a", store_path=str(tmp_path / "store"))
    tiny.write_checkpoint(source, tiny.tiny_config(), seed=2)
    time.sleep(0.01)
    second = _trainer(source, tmp_path / "b", store_path=str(tmp_path / "store"))
    assert not torch.equal(first.model.poros_source.slabs[0], second.model.poros_source.slabs[0])


def test_cuda_still_refuses_a_dense_store(tmp_path, monkeypatch):
    from poros.core.errors import PorosConfigError

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = _config("Qwen/x", tmp_path, store="ssd")
    monkeypatch.setattr("poros.preflight.arch.cheap_model_type", lambda *a: "qwen2")
    with pytest.raises(PorosConfigError, match="streams MoE"):
        PorosTrainer.from_pretrained("Qwen/x", config=config)


# -- the run -----------------------------------------------------------------------------------


@pytest.mark.parametrize("gate", [False, True], ids=["user", "gate"])
def test_how_the_bytes_arrive_changes_nothing(checkpoint, tmp_path, reference, gate):
    trainer = _trainer(checkpoint, tmp_path, store="ssd")
    if gate:
        _gate(trainer)
    trainer.train(_data(trainer), steps=STEPS)
    assert _finals(trainer) == reference[0]
    assert trainer.history.losses == reference[1]


def test_the_block_loop_matches_one_graph_with_no_function(checkpoint, tmp_path, reference):
    """The anchor: no block loop, no remat, every matrix held by autograd."""
    trainer = _trainer(checkpoint, tmp_path)
    trainer._witness.remove()
    trainer._witness = None
    model = trainer.model
    for block in range(4):
        trainer.block_manager.load_block_for_inference(block)
    for module in model.modules():
        if isinstance(module, Nf4Linear):
            module.__class__ = arms.PlainLinear
    trainer._forward_blockwise_adapter_model = lambda batch: arms._one_graph(trainer, batch)
    trainer.train(_data(trainer), steps=STEPS)
    assert _finals(trainer) == reference[0]


def test_a_stopped_run_resumes_to_the_same_bytes(checkpoint, tmp_path, reference):
    first = _trainer(checkpoint, tmp_path, store="ssd")
    with pytest.raises(PorosCancelledError) as stopped:
        first.train(_data(first), steps=STEPS, should_stop=lambda: first.global_step == 1)
    second = _trainer(checkpoint, tmp_path, store="ssd")
    second.train(_data(second), steps=STEPS, resume_from=stopped.value.checkpoint)
    assert _finals(second) == reference[0]


def test_the_adapter_loads_on_vanilla_weights(checkpoint, tmp_path):
    trainer = _trainer(checkpoint, tmp_path)
    trainer.train(_data(trainer), steps=1)
    trainer.save(str(tmp_path / "out"))
    base = Qwen2ForCausalLM.from_pretrained(checkpoint)
    loaded = peft.PeftModel.from_pretrained(base, tmp_path / "out")
    assert sum("lora_" in n for n, _ in loaded.named_parameters()) == 2 * 7 * 4


def test_the_step_events_carry_the_unified_fields(checkpoint, tmp_path, monkeypatch):
    trainer = _trainer(checkpoint, tmp_path)
    sample = machine.MemSample(allocated=10**6, driver=2 * 10**6, footprint=3, swap=4, pressure=1)
    monkeypatch.setattr(machine, "memory_sample", lambda device: sample)
    trainer.device = torch.device("mps")
    trainer._emit_step_event(1.0, 1.0)
    trainer.device = CPU
    event = json.loads(Path(tmp_path / "events.jsonl").read_text().splitlines()[-1])
    assert (event["data"]["vram_allocated_mb"], event["data"]["vram_reserved_mb"]) == (1.0, 2.0)
    data = event["data"]
    assert (data["footprint"], data["swap"], data["pressure"]) == (3, 4, 1)


# -- the witness -------------------------------------------------------------------------------


class _NoRestore(arms._NoRestore):
    pass


def _events(out) -> list:
    return [json.loads(line) for line in Path(out, "events.jsonl").read_text().splitlines()]


def test_a_clean_run_checks_every_layer_of_every_microstep(checkpoint, tmp_path):
    trainer = _trainer(checkpoint, tmp_path)
    trainer.train(_data(trainer), steps=STEPS)
    assert (trainer._witness.checked, trainer._witness.mismatches) == (4 * G * STEPS, 0)
    assert trainer._witness._forward == {} and trainer._witness._flags == []


def test_the_witness_keeps_nothing_on_the_autograd_node(checkpoint, tmp_path, monkeypatch):
    """A tensor on ctx would need the cyclic GC to free; the forward's output must not be there."""
    seen = []
    real = blockwise.JITAdapterBlockFunction.backward

    def backward(ctx, g):
        seen.append(sorted(k for k, v in vars(ctx).items() if isinstance(v, torch.Tensor)))
        return real(ctx, g)

    monkeypatch.setattr(blockwise.JITAdapterBlockFunction, "backward", staticmethod(backward))
    trainer = _trainer(checkpoint, tmp_path, dropout=0.0)
    trainer.train(_data(trainer), steps=1)
    assert seen and all("output" not in " ".join(names) for names in seen)


def test_an_evaluation_holds_nothing_for_a_remat_that_never_comes(checkpoint, tmp_path):
    trainer = _trainer(checkpoint, tmp_path)
    trainer._evaluate_batches(_data(trainer))
    assert trainer._witness._forward == {}
    assert trainer.model.training


def test_a_remat_that_differs_stops_the_run_where_a_bad_step_would_be_kept(
    checkpoint, tmp_path, monkeypatch
):
    trainer = _trainer(checkpoint, tmp_path, save_steps=1)
    monkeypatch.setattr(blockwise, "torch", _NoRestore(torch))
    with pytest.raises(PorosStepCheckError) as stopped:
        trainer.train(_data(trainer), steps=STEPS)
    # lora_B is zero until the first update, so the first step cannot show it.
    assert trainer.global_step == 1
    assert trainer._lr_scheduler.last_epoch == 1  # stopped before the bad step's update
    assert stopped.value.code == "E_STEP_CHECK" and EXIT_CODES["E_STEP_CHECK"] == 4
    assert stopped.value.witness == {
        "mismatches": 2 * 4, "first_layer": 0, "step": 2, "nonfinite_loss": False,
    }
    # No save has happened yet: the first periodic one is at the end of the step that failed.
    assert checkpoints.list_checkpoints(Path(tmp_path)) == []
    assert stopped.value.info.message == "A check on step 2 failed, so this run stopped. " \
        "Nothing was kept."
    assert stopped.value.info.remediation == "Start the run again."
    error, = [e for e in _events(tmp_path) if e["event_type"] == "error"]
    assert error["data"]["code"] == "E_STEP_CHECK" and error["data"]["witness"]["step"] == 2
    assert error["step"] == 1  # the update's index from 0, as a step event's; its record from 1


def test_a_nonfinite_loss_after_a_save_says_the_value_and_what_was_kept(
    checkpoint, tmp_path, monkeypatch
):
    """A non-finite loss on the Mac path is the trainer's one stop (as on CUDA): it names the
    value, the remedy and the checkpoint that holds the run, and nothing of that update is kept."""
    trainer = _trainer(checkpoint, tmp_path, save_steps=1)
    real = trainer._sequential_pass

    def nan_late(batch, g, G):
        out = real(batch, g, G)
        return out * float("nan") if trainer.global_step == 2 else out

    monkeypatch.setattr(trainer, "_sequential_pass", nan_late)
    with pytest.raises(PorosStepCheckError) as stopped:
        trainer.train(_data(trainer), steps=STEPS)
    message = stopped.value.info.message
    assert message.startswith("At update 3 the loss is nan, so this run stopped before applying "
                              "it: nothing of update 3 was kept, and ")
    assert "holds the run through update 2" in message
    assert stopped.value.info.remediation.startswith("Lower learning_rate (now 0.001)")
    assert stopped.value.witness == {"step": 3, "loss": "nan", "torch_dtype": "bfloat16"}
    assert [step for step, _ in checkpoints.list_checkpoints(Path(tmp_path))] == [1]
    error, = [e for e in _events(tmp_path) if e["event_type"] == "error"]
    assert error["step"] == 2 and error["data"]["nonfinite"]["loss"] == "nan"  # index, from 0
    assert trainer._witness.mismatches == 0


def test_with_clipping_off_a_nonfinite_gradient_norm_stops_before_its_update(
    checkpoint, tmp_path, monkeypatch
):
    """max_grad_norm 0 on the Mac path: the norm is still read, and a non-finite one stops the
    run before the optimizer steps and before any checkpoint holds it."""
    trainer = _trainer(checkpoint, tmp_path, save_steps=1, max_grad_norm=0.0)
    real = torch.nn.utils.get_total_norm
    calls = {"n": 0}

    def norm(grads, *a, **k):
        calls["n"] += 1
        return torch.tensor(float("inf")) if calls["n"] == 3 else real(grads, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "get_total_norm", norm)
    with pytest.raises(PorosStepCheckError) as stopped:
        trainer.train(_data(trainer), steps=STEPS)
    assert trainer.global_step == 2 and trainer._lr_scheduler.last_epoch == 2
    assert stopped.value.witness["grad_norm"] == "inf" and stopped.value.witness["step"] == 3
    assert "At update 3 the gradient norm is inf" in stopped.value.info.message
    assert [step for step, _ in checkpoints.list_checkpoints(Path(tmp_path))] == [1]


def _flip_on_read(layers, layer, monkeypatch, *, read=1):
    """One bit flipped in the bytes of one read of ``layer``."""
    real, count = layers.source.read_into, {"reads": 0}

    def flipped(key, dst):
        real(key, dst)
        if key[0] == layer:
            count["reads"] += 1
            if count["reads"] == read:
                dst[0] ^= 0x10

    monkeypatch.setattr(layers.source, "read_into", flipped)
    return count


def test_a_bad_read_in_a_middle_layer_is_seen(checkpoint, tmp_path, monkeypatch):
    trainer = _trainer(checkpoint, tmp_path, store="ssd")
    # The first read of a slab is checksummed, so the flip lands on the second.
    _flip_on_read(trainer.block_manager.source, 1, monkeypatch, read=2)
    with pytest.raises(PorosStepCheckError) as stopped:
        trainer.train(_data(trainer), steps=STEPS)
    assert stopped.value.witness["first_layer"] == 1


def test_a_bad_read_at_the_turn_shows_only_in_a_gate_run(checkpoint, tmp_path, monkeypatch):
    """A warm hit reads once, so both passes of the last layer see the same bytes."""
    user = _trainer(checkpoint, tmp_path / "user", store="ssd")
    source = user.block_manager.source
    source.source._verified.update(k for k in source.source.keys())  # the second-read flip, at once
    _flip_on_read(source, 3, monkeypatch)
    user.train(_data(user), steps=1)
    assert user._witness.mismatches == 0

    gate = _trainer(checkpoint, tmp_path / "gate", store="ssd")
    _gate(gate)
    source = gate.block_manager.source
    source.source._verified.update(k for k in source.source.keys())
    _flip_on_read(source, 3, monkeypatch)
    with pytest.raises(PorosStepCheckError) as stopped:
        gate.train(_data(gate), steps=1)
    assert stopped.value.witness["first_layer"] == 3


def test_a_use_after_release_is_poison_in_a_gate_run(checkpoint, tmp_path, monkeypatch):
    trainer = _trainer(checkpoint, tmp_path, store="ssd")
    _gate(trainer)
    manager = trainer.block_manager

    linear = manager._linears[0][0]
    manager.load_block_for_inference(0)
    slab = linear.slab  # what a use after release would still be reading
    manager.release_block(0)
    assert linear.slab is None
    assert bool((slab == arms.POISON).all())


# -- the watchdog ------------------------------------------------------------------------------


@pytest.fixture
def fast_watchdog(monkeypatch):
    # The stops are shares of this Mac's memory (swap_stop_bytes): these tests state the 16 GB
    # reference's figures, so they read that memory whatever Mac runs them. A 7 GB runner's swap
    # stop is 0.875 GB, and a growth of 2 GB fired there and exited the whole session.
    monkeypatch.setattr(watchdog, "this_memory", lambda: watchdog.REFERENCE_MEMORY)
    monkeypatch.setattr(watchdog, "INTERVAL_S", 0.002)
    monkeypatch.setattr(watchdog, "STEP_WAIT_S", 0.05)
    monkeypatch.setattr(machine, "swap_used", lambda: 0)
    monkeypatch.setattr(watchdog, "_boot_free", lambda: 10**12)


def _wait_for(flag) -> None:
    for _ in range(2000):
        if flag():
            return
        time.sleep(0.001)
    raise AssertionError("never happened")


def test_critical_pressure_fires_and_lesser_pressure_does_not(fast_watchdog, monkeypatch):
    level = [2]
    monkeypatch.setattr(machine, "pressure", lambda: level[0])
    dog = watchdog.Watchdog()
    time.sleep(0.02)
    assert not dog.fired.is_set()
    level[0] = watchdog.CRITICAL
    _wait_for(dog.fired.is_set)
    dog.stop()


def test_no_step_boundary_in_time_exits_e_oom(fast_watchdog, monkeypatch, capsys):
    monkeypatch.setattr(machine, "pressure", lambda: watchdog.CRITICAL)
    exits = []
    monkeypatch.setattr(os, "_exit", exits.append)
    watchdog.Watchdog()
    _wait_for(lambda: exits)
    assert exits == [EXIT_CODES["E_OOM"]]
    assert "poros runs continue" in capsys.readouterr().err


def test_a_boundary_in_time_keeps_the_process(fast_watchdog, monkeypatch):
    monkeypatch.setattr(machine, "pressure", lambda: watchdog.CRITICAL)
    exits = []
    monkeypatch.setattr(os, "_exit", exits.append)
    dog = watchdog.Watchdog()
    _wait_for(dog.fired.is_set)
    dog.stop()
    time.sleep(0.1)
    assert exits == []


def _swap_grows_to(monkeypatch, grown: int) -> None:
    start = iter([0])
    monkeypatch.setattr(machine, "swap_used", lambda: next(start, grown))


@pytest.mark.parametrize(
    "grown, free, level, said",
    [(2_000_000_001, 10**12, 1, "Swap grew by 2.0 GB"),
     (0, 2_900_000_000, 1, "The boot disk, where swap lives, fell to 2.9 GB free"),
     (0, 10**12, watchdog.CRITICAL, "This Mac ran out of memory")],
)
def test_each_limit_stops_a_run_and_says_what_to_try(
    fast_watchdog, monkeypatch, grown, free, level, said
):
    monkeypatch.setattr(machine, "pressure", lambda: level)
    monkeypatch.setattr(watchdog, "_boot_free", lambda: free)
    _swap_grows_to(monkeypatch, grown)
    dog = watchdog.Watchdog()
    _wait_for(dog.fired.is_set)
    dog.stop()
    assert dog.why == said
    trainer = SimpleNamespace(
        optimizer=SimpleNamespace(zero_grad=lambda set_to_none: None),
        _release_retained_forward_blocks=lambda: None, global_step=7,
    )
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    err = watchdog.oom_error(trainer, dog.why)
    assert err.code == "E_OOM" and err.info.message == f"{said} at step 7."
    for word in ("poros runs continue", "--seq-len", "--rank", "memory_guard=false"):
        assert word in err.info.remediation
    assert ("Free some space" in err.info.remediation) == said.startswith("The boot disk")


def test_at_every_limit_and_no_further_nothing_fires(fast_watchdog, monkeypatch):
    monkeypatch.setattr(machine, "pressure", lambda: 2)
    monkeypatch.setattr(watchdog, "_boot_free", lambda: watchdog.BOOT_STOP_BYTES)
    _swap_grows_to(monkeypatch, watchdog.SWAP_STOP_BYTES)
    exits = []  # a stop that fires here fails this test, not the session
    monkeypatch.setattr(os, "_exit", exits.append)
    dog = watchdog.Watchdog()
    _wait_for(lambda: dog.swap_growth == watchdog.SWAP_STOP_BYTES)
    time.sleep(0.02)
    dog.stop()
    assert not dog.fired.is_set() and dog.why == "" and exits == []


def test_the_setting_turns_off_the_stop_and_keeps_the_figure(fast_watchdog, monkeypatch):
    monkeypatch.setattr(machine, "pressure", lambda: watchdog.CRITICAL)
    monkeypatch.setattr(watchdog, "_boot_free", lambda: 0)
    _swap_grows_to(monkeypatch, 5_000_000_000)
    dog = watchdog.Watchdog(guard=False)
    _wait_for(lambda: dog.swap_growth == 5_000_000_000)
    time.sleep(0.02)
    assert not dog.fired.is_set() and dog.why == ""
    dog.stop()


@pytest.mark.parametrize(
    "on_boot, free, saves, error",
    [(True, 2_000_000_000, 0, None), (True, 9_000_000_000, 1, None),
     (False, 2_000_000_000, 1, None), (False, 2_000_000_000, 1, PorosDiskError("full"))],
)
def test_it_saves_the_run_if_that_is_safe_and_a_full_disk_does_not_change_the_stop(
    monkeypatch, tmp_path, on_boot, free, saves, error
):
    from poros.metal import store

    monkeypatch.setattr(store, "_same_disk", lambda a, b: on_boot)
    monkeypatch.setattr(watchdog, "_boot_free", lambda: free)
    saved = []

    def save():
        saved.append(1)
        if error:
            raise error

    folder = SimpleNamespace(output_dir=str(tmp_path))
    watchdog.checkpoint_first(SimpleNamespace(config=folder, _save_checkpoint=save))
    assert len(saved) == saves


def test_swap_growth_is_the_peak_over_the_start(fast_watchdog, monkeypatch):
    monkeypatch.setattr(machine, "pressure", lambda: 1)
    swap = [5]
    monkeypatch.setattr(machine, "swap_used", lambda: swap[0])
    dog = watchdog.Watchdog()
    assert dog.swap_growth == 0
    swap[0] = 8
    _wait_for(lambda: dog.swap_growth == 3)
    swap[0] = 1  # another process's swap coming back does not undo this run's spill
    time.sleep(0.02)
    dog.stop()
    assert dog.swap_growth == 3


def test_at_the_boundary_a_fired_watchdog_checkpoints_and_stops(
    checkpoint, tmp_path, monkeypatch, reference
):
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    trainer = _trainer(checkpoint, tmp_path, store="ssd")
    fired = threading.Event()
    stopped_at = []
    trainer._watchdog = SimpleNamespace(
        fired=fired, stop=lambda: stopped_at.append(trainer.global_step),
        why="This Mac ran out of memory",
    )
    real_step = trainer._adapter_optimizer_step

    def fire_during_step_1():
        real_step()
        if trainer.global_step == 1:
            fired.set()

    trainer._adapter_optimizer_step = fire_during_step_1
    with pytest.raises(PorosOOMError, match="ran out of memory at step 1"):
        trainer.train(_data(trainer), steps=STEPS)
    assert stopped_at == [1]
    (saved, path), = checkpoints.list_checkpoints(Path(tmp_path))
    assert saved == 1
    resumed = _trainer(checkpoint, tmp_path, store="ssd")
    resumed.train(_data(resumed), steps=STEPS, resume_from=str(path))
    assert _finals(resumed) == reference[0]


def test_a_run_whose_last_save_failed_continues_from_its_last_checkpoint(checkpoint, tmp_path):
    first = _trainer(checkpoint, tmp_path)
    first.train(_data(first), steps=2)
    path = first._save_checkpoint()
    again = _trainer(checkpoint, tmp_path)
    metrics = again.train(_data(again), steps=2, resume_from=path)
    assert again.global_step == first.global_step and metrics is not None
    again.save(str(tmp_path / "final"))
    assert (tmp_path / "final" / "adapter_model.safetensors").exists()


def test_the_last_step_finishes_whatever_the_watchdog_says(checkpoint, tmp_path):
    trainer = _trainer(checkpoint, tmp_path)
    fired = threading.Event()
    fired.set()
    trainer._watchdog = SimpleNamespace(fired=fired, stop=lambda: None)
    trainer.train(_data(trainer), steps=1)
    assert checkpoints.list_checkpoints(Path(tmp_path)) == []


# -- machine ---------------------------------------------------------------------------------


def test_the_readers_import_anywhere_and_read_only_on_a_mac():
    code = "import sys; import poros.metal.machine; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False, cwd=ROOT).returncode == 0


@pytest.mark.skipif(sys.platform != "darwin", reason="reads this Mac's kernel")
def test_the_readers_read_this_mac():
    assert machine.footprint() > 10**6
    assert machine.swap_used() >= 0
    assert machine.pressure() in (1, 2, 4)
    assert machine.disk_read_bytes() >= 0


# -- on Metal ----------------------------------------------------------------------------------

metal = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs Metal")


def _on_metal(checkpoint, out, monkeypatch, **kwargs) -> PorosTrainer:
    """The real gate, the real device: conftest asks the CPU suite for the Linux pick."""
    monkeypatch.setattr(trainer_mod, "sys", sys)
    config = _config(checkpoint, out, **kwargs)
    trainer = PorosTrainer.from_pretrained(config.model_name_or_path, config=config)
    assert trainer.device.type == "mps"
    return trainer


@metal
def test_the_real_trainer_saves_and_resumes_on_metal(checkpoint, tmp_path, monkeypatch):
    whole = _on_metal(checkpoint, tmp_path / "whole", monkeypatch)
    whole.train(_data(whole), steps=STEPS)
    assert (whole._witness.checked, whole._witness.mismatches) == (4 * G * STEPS, 0)
    whole.save(str(tmp_path / "out"))
    base = Qwen2ForCausalLM.from_pretrained(checkpoint)
    peft.PeftModel.from_pretrained(base, tmp_path / "out")

    first = _on_metal(checkpoint, tmp_path / "parts", monkeypatch, store="ssd")
    with pytest.raises(PorosCancelledError) as stopped:
        first.train(_data(first), steps=STEPS, should_stop=lambda: first.global_step == 1)
    second = _on_metal(checkpoint, tmp_path / "parts", monkeypatch, store="ssd")
    second.train(_data(second), steps=STEPS, resume_from=stopped.value.checkpoint)
    assert _finals(second) == _finals(whole)
    state = second.optimizer.state_dict()["state"][0]
    assert state["step"].device.type == "cpu" and state["exp_avg"].device.type == "mps"


@metal
def test_a_gate_source_past_the_page_cache_serves_the_same_bytes(store_file, monkeypatch):
    device = torch.device("mps")
    layers = arms.GateLayers(store_file, device)
    for layer in (0, 3, 3, 1):
        layers.prefetch(layer)
        assert torch.equal(layers.acquire(layer).cpu(), _bytes_of(store_file, layer))


def test_an_evicted_file_keeps_its_bytes_and_on_a_mac_leaves_the_page_cache(store_file):
    before = store_file.read_bytes()
    left = arms._evict(store_file)
    assert left < 0.05 if arms.MAC else 0 <= left <= 1  # Linux keeps the pages
    assert store_file.read_bytes() == before


# -- poros.Run, which recipes train through -----------------------------------------------------


def _batch(vocab: int, step: int) -> Batch:
    """A row whose length depends on the step: a run's batches are as long as their longest row."""
    ids = torch.randint(0, vocab, (8 + step % 5,), generator=torch.Generator().manual_seed(step))
    return Batch([Datum.from_sequence(ids.tolist())])


def _drive(run: Run, steps: int, **kwargs) -> None:
    vocab = run._engine.model.config.vocab_size
    for step in run.steps(steps, **kwargs):
        run.forward_backward(_batch(vocab, step))
        run.optim_step()


def _run_of(checkpoint, out, **kwargs) -> Run:
    return Run.from_trainer(_trainer(checkpoint, out, gradient_accumulation_steps=1, **kwargs))


def test_a_run_checks_every_layer_and_an_evaluation_is_not_a_step(checkpoint, tmp_path):
    run = _run_of(checkpoint, tmp_path)
    run.forward(_batch(run._engine.model.config.vocab_size, 0))
    _drive(run, 3)
    witness = run._engine._witness
    assert (witness.checked, witness.mismatches) == (4 * 3, 0)
    assert witness._forward == {}


def test_a_run_stops_on_a_step_it_cannot_keep(checkpoint, tmp_path, monkeypatch):
    run = _run_of(checkpoint, tmp_path)
    monkeypatch.setattr(blockwise, "torch", arms._NoRestore(torch))
    with pytest.raises(PorosStepCheckError):
        _drive(run, 4)
    assert run.step < 4


def test_a_stopped_run_continues_to_the_same_bytes(checkpoint, tmp_path):
    whole = _run_of(checkpoint, tmp_path / "whole", save_steps=0)
    _drive(whole, 6)
    first = _run_of(checkpoint, tmp_path / "parts", save_steps=0)
    with pytest.raises(PorosCancelledError) as stopped:
        _drive(first, 6, stop=lambda: first.step == 3)
    second = _run_of(checkpoint, tmp_path / "parts", save_steps=0)
    second.load(stopped.value.checkpoint)
    _drive(second, 6)
    assert _finals(second._engine) == _finals(whole._engine)


def test_a_run_under_critical_pressure_keeps_itself_and_stops(checkpoint, tmp_path, monkeypatch):
    run = _run_of(checkpoint, tmp_path)
    fired = threading.Event()
    fired.set()
    engine = run._engine
    monkeypatch.setattr(engine, "_start_watchdog", lambda: None)
    engine._watchdog = SimpleNamespace(fired=fired, stop=lambda: None, why="out")
    monkeypatch.setattr(watchdog, "oom_error", lambda trainer, why: PorosOOMError(why))
    with pytest.raises(PorosOOMError):
        _drive(run, 4)
    assert [step for step, _ in checkpoints.list_checkpoints(tmp_path)] == [0]


def test_a_run_watches_memory_on_a_mac_for_its_steps_alone(checkpoint, tmp_path, monkeypatch):
    run = _run_of(checkpoint, tmp_path)
    engine = run._engine
    seen = []
    monkeypatch.setattr(engine, "_start_watchdog", lambda: seen.append("start"))
    monkeypatch.setattr(engine, "_stop_watchdog", lambda: seen.append("stop"))
    _drive(run, 2)
    assert seen == ["start", "stop"]


def test_on_a_mac_every_batch_is_the_tables_length(checkpoint, tmp_path, monkeypatch):
    run = _run_of(checkpoint, tmp_path)
    engine = run._engine
    monkeypatch.setattr(engine, "device", torch.device("mps"))
    batch = _batch(engine.model.config.vocab_size, 0)
    for _, rows in run._slices(batch):
        assert rows == [0]
    assert batch.length == engine.config.max_seq_length == 16
    assert batch.tokens < 16  # what it reports is what is real
    masked = batch._build("cpu")["attention_mask"]
    assert masked.shape == (1, 16) and int(masked.sum()) == batch.tokens


def test_off_a_mac_a_batch_is_as_long_as_its_longest_row(checkpoint, tmp_path):
    run = _run_of(checkpoint, tmp_path)
    batch = _batch(run._engine.model.config.vocab_size, 0)
    run._slices(batch)
    assert batch.length == batch.tokens


@pytest.mark.parametrize(
    "error",
    [RuntimeError("MPS backend out of memory (MPS allocated: 1 GB)"), torch.OutOfMemoryError("cb")],
    ids=["allocator", "command-buffer"],
)
def test_a_run_on_mps_names_either_oom_for_a_mac(checkpoint, tmp_path, monkeypatch, error):
    run = _run_of(checkpoint, tmp_path)
    engine = run._engine
    batch = _batch(engine.model.config.vocab_size, 0)

    def boom(feed):
        raise error

    monkeypatch.setattr(engine, "_forward_train_logits", boom)
    monkeypatch.setattr(engine, "device", torch.device("mps"))
    monkeypatch.setattr(batch, "tensors", lambda device: batch._build("cpu"))
    monkeypatch.setattr(engine.config, "max_seq_length", 8)
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: None)
    with pytest.raises(PorosOOMError, match="This Mac ran out of memory"):
        run.forward_backward(batch)


def test_a_recipe_on_a_mac_takes_the_mac_tables_shape(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    try:
        reloaded = importlib.reload(recipes)
        assert reloaded.MEASURED["block_size"] == 1
        assert reloaded.MEASURED["gradient_accumulation_steps"] == 1
        config = PorosConfig(model_name_or_path="x")
        reloaded.apply(config, reloaded.Choice("sft", "chats"))
        metal_gate.validate(config)
    finally:
        monkeypatch.undo()
        importlib.reload(recipes)


def test_a_checkpoint_the_disk_could_not_take_is_a_disk_error_with_nothing_left_torn(
    checkpoint, tmp_path, monkeypatch
):
    trainer = _trainer(checkpoint, tmp_path / "run")
    trainer.train(_data(trainer), steps=1)

    def full(*args, **kwargs):
        raise RuntimeError("[enforce fail at inline_container.cc:671] . unexpected pos 704 vs 598")

    monkeypatch.setattr(torch, "save", full)
    with pytest.raises(PorosDiskError, match="full, or it is gone"):
        trainer._save_checkpoint()
    assert not list((tmp_path / "run").glob("checkpoint-*"))


def _script(name: str, *args: str, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "metal" / name), *args],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT), "POROS_COMMIT": "abc", **(env or {})},
    )


def _direct(checkpoint, out, arm_names: str) -> subprocess.CompletedProcess:
    return _script(
        "arms.py", "--model", str(checkpoint), "--arm", arm_names, "--gate", "direct",
        "--seq", "16", "--g", str(G), "--steps", "3", "--learning-rate", "1e-3",
        "--warmup-steps", "0", "--out", str(out), env={"POROS_HOME": str(out / "home")},
    )


@needs_metal_on_a_mac
def test_a_direct_gate_at_three_steps_is_a_clean_probe(checkpoint, tmp_path):
    done = _direct(checkpoint, tmp_path, "plain,mono,ram-loop")
    assert done.returncode == 1, done.stdout + done.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    assert receipt["schema"] == "poros.metal.receipt.v1" and receipt["failures"] == []
    assert receipt["pass"] is False
    assert receipt["probe"] == "a probe, not a pass: 3 steps, and direct at this size needs 20"
    assert {a["name"] for a in receipt["arms"]} == {"plain", "mono", "ram-loop"}
    loop = json.loads((tmp_path / "ram-loop.json").read_text())
    assert loop["witness"] == {"checked": 4 * G * 3, "mismatches": 0}
    assert loop["decodes"] == 3 * 7 * 4 * G * 3  # forward, remat, VJP: 7 matrices in 4 layers


@needs_metal_on_a_mac
def test_the_chats_gate_compares_run_on_padded_rows(checkpoint, tmp_path):
    done = _script(
        "arms.py", "--model", str(checkpoint), "--arm", "mono,ram-loop,ram-loop.rng-off",
        "--gate", arms.CHATS_GATE, "--seq", "16", "--g", str(G), "--steps", "3",
        "--learning-rate", "1e-3", "--warmup-steps", "0", "--out", str(tmp_path),
        env={"POROS_HOME": str(tmp_path / "home")},
    )
    assert done.returncode == 1, done.stdout + done.stderr
    receipt = json.loads((tmp_path / "receipt.json").read_text())
    differ = [f for f in receipt["failures"] if "differ" in f]
    assert len(differ) == 2 and all("ram-loop.rng-off" in f for f in differ), receipt["failures"]
    assert first_difference(Transcript(tmp_path / "mono.jsonl"),
                            Transcript(tmp_path / "ram-loop.jsonl")) is None
    report = json.loads((tmp_path / "ram-loop.json").read_text())
    assert report["witness"] == {"checked": 4 * G * 3, "mismatches": 0}

    rows = arms.chat_rows(100, 16, G * 3, G)
    assert len({len(item) for item in rows.items}) == G * 3
    assert all(0 < sum(item.weights) < len(item) <= 16 for item in rows.items)


@needs_metal_on_a_mac
def test_without_the_rng_restore_digests_differ_and_the_witness_stops_the_run(checkpoint, tmp_path):
    _direct(checkpoint, tmp_path, "ram-loop,ram-loop.rng-off,ram-loop.rng-off-witnessed")
    clean = Transcript(tmp_path / "ram-loop.jsonl")
    fault = Transcript(tmp_path / "ram-loop.rng-off.jsonl")
    assert first_difference(clean, fault) is not None
    fired = json.loads((tmp_path / "ram-loop.rng-off-witnessed.json").read_text())
    assert fired["error"] == "E_STEP_CHECK" and fired["witness"]["mismatches"] > 0


@pytest.mark.skipif(arms.MAC, reason="on a Mac this needs a real store: a tiny slab is cached")
def test_streaming_compares_and_only_a_mac_can_pass_it(checkpoint, tmp_path):
    """Off a Mac the OS disk counters are absent, so the receipt fails for that reason alone."""
    done = _script(
        "arms.py", "--model", str(checkpoint), "--arm", "ram-loop,ssd", "--gate", "streaming",
        "--seq", "16", "--g", "2", "--steps", "2", "--out", str(tmp_path),
        env={"POROS_HOME": str(tmp_path / "home")},
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert json.loads((tmp_path / "receipt.json").read_text())["failures"] == [
        "arm ssd: disk read_bytes and predicted_read_bytes malformed",
        "arm ssd: preflight resident fraction is None, required under 5%",
    ]
