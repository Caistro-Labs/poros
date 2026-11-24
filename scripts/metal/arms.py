#!/usr/bin/env python3
"""Arms through the real trainer, one process each, and a receipt over them.

    arms.py --out DIR --model Qwen/Qwen2.5-7B --arm ram-loop --seq 128 --steps 20
    arms.py --out DIR --model Qwen/Qwen2.5-0.5B --arm plain,mono,ram-loop --gate direct --seq 256
    arms.py --out DIR --model Qwen/Qwen2.5-0.5B --arm mono,ram-loop --gate direct-chats --seq 256

``ram-loop`` and ``ssd`` are the product's block loop. ``mono`` is one graph over the same frozen
Function, and ``plain`` has no Poros autograd Function. A tag names a repeat (``ram-loop.2``), a
fault (``ram-loop.rng-off``), or an arm with no capture hooks at all (``ram-loop.quiet``). Every
gate trains through the classic forward on full-length rows, but ``direct-chats``: that drives
``poros.Run`` as ``poros train`` does on chats, with padded rows, a mask, and cross entropy per row.
Writes <out>/<arm>.jsonl, <out>/<arm>.json and <out>/receipt.json. Off a Mac it runs on the CPU
through the same Mac path, which proves the wiring and nothing about Metal.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import fcntl
import json
import mmap
import os
import subprocess
import sys
import threading
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import Qwen2Config, Qwen2ForCausalLM

from poros.core import blockwise
from poros.core.config import PorosConfig
from poros.core.errors import EXIT_CODES, PorosStepCheckError
from poros.data import Data
from poros.losses import cross_entropy
from poros.metal import load, machine, nf4, source, store
from poros.metal.capture import TranscriptWriter, capture
from poros.metal.head import ChunkedHead
from poros.metal.prove import pin
from poros.metal.source import FileLayers
from poros.metal.tiny import CpuDouble
from poros.metal.verify import RECEIPT_SCHEMA, arm_from_transcript, write_verdict
from poros.train.batch import Datum
from poros.train.run import Run
from poros.train.trainer import PorosTrainer

CHATS_GATE = "direct-chats"
LOOP_KINDS = ("ram-loop", "ssd", "ssd-sync")
MAC = sys.platform == "darwin"
POISON = 0xAB  # a nibble stream of this decodes to a wrong matrix, not to zeros
MS_INVALIDATE, MINCORE_INCORE = 2, 1  # the same on Darwin and Linux


class _NoRestore:
    """``torch`` as blockwise sees it, except that a restore does nothing. The fork stays."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        found = getattr(self._module, name)
        return _NoRestore(found) if name == "mps" else found

    @staticmethod
    def set_rng_state(state):
        pass


class PlainLinear(nf4.Nf4Linear):
    """No Poros Function, so autograd holds a copy of every matrix."""

    def matmul(self, x2d: torch.Tensor) -> torch.Tensor:
        return F.linear(x2d, self.decoded().clone())


class PlainHead(ChunkedHead):
    """Pure autograd through ``torch.cat``, no Poros Function."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        h2d = hidden.reshape(-1, hidden.shape[-1])
        logits = [F.linear(h2d, self.chunk(a, b, h2d.device)) for a, b in self.spans]
        return torch.cat(logits, dim=1).reshape(*hidden.shape[:-1], -1)


def make_plain(model):
    """The ``plain`` arm. These classes add no state, so a built model can become them."""
    for module in model.modules():
        if isinstance(module, nf4.Nf4Linear):
            module.__class__ = PlainLinear
    model.lm_head.__class__ = PlainHead
    return model


class GateLayers(FileLayers):
    """Past the page cache, and every layer read again at each turn, so a bad read there shows."""

    def __init__(self, path, device):
        super().__init__(path, device)
        if MAC:
            fcntl.fcntl(self.source.file.fd, fcntl.F_NOCACHE, 1)

    def _warm(self, layer):
        return False


def poison_released(manager, layers) -> None:
    """A released slot is poison, so a use after release shows."""
    release = manager.release_block

    def poisoned(block_id):
        release(block_id)
        layers.slot.fill_(POISON)

    manager.release_block = poisoned


class _NoReadAhead:
    """A source whose every read happens when it is asked for."""

    def __init__(self, layers):
        self.acquire = layers.acquire

    def prefetch(self, layer):
        pass


def tiny_config(*, tied=False, vocab_size=40000, layers=4) -> Qwen2Config:
    """vocab_size 40000 gives the head three chunks, so its sum order is a real question."""
    return Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=8,
        num_key_value_heads=2,
        vocab_size=vocab_size,
        max_position_embeddings=64,
        use_cache=False,
        tie_word_embeddings=tied,
        attn_implementation="eager",
    )


def write_checkpoint(path, cfg, *, seed=1, **save_kwargs) -> None:
    """A bf16 checkpoint with vanilla weights.

    Fresh init has zero biases and unit norms, so a gate over it cannot see either one.
    """
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(cfg)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.normal_(0.0, 0.02)
            elif "norm" in name:
                param.add_(0.1 * torch.randn_like(param))
    model.to(torch.bfloat16).save_pretrained(path, **save_kwargs)


def token_stream(vocab: int, seq: int, rows: int) -> DataLoader:
    """The gate scripts' stream: uniform tokens from a CPU generator at seed 99."""
    ids = torch.randint(0, vocab, (rows, seq), generator=torch.Generator().manual_seed(99))
    return DataLoader([{"input_ids": r, "labels": r.clone()} for r in ids], batch_size=1)


def chat_rows(vocab: int, seq: int, rows: int, g: int) -> Data:
    """What ``poros.data.chats`` makes of a file: rows of unequal length, the shorter ones padded,
    a prompt that is only read and an answer that is scored. At seed 99, as ``token_stream``."""
    ids = torch.randint(0, vocab, (rows, seq + 1), generator=torch.Generator().manual_seed(99))
    items = []
    for i, row in enumerate(ids.tolist()):
        length = seq // 2 + 2 + i % (seq // 2)
        prompt = length // 3
        items.append(Datum.from_sequence(row[:length], [0.0] * prompt + [1.0] * (length - prompt)))
    return Data(items, batch_size=g)


def _one_graph_logits(trainer, batch):
    """The trainer's own pieces with no block Function and no remat."""
    hidden, mask, positions, labels = trainer._build_inputs(batch)
    hidden = hidden.detach().requires_grad_(True)
    for block in range(trainer.block_manager.num_blocks):
        hidden = trainer._forward_block_layers(block, hidden, mask, positions, checkpoint=False)
    return trainer._logits_from_hidden(hidden), labels


def _one_graph(trainer, batch):
    return trainer._compute_loss(*_one_graph_logits(trainer, batch))


class Recorder:
    """The capture hooks. Digests are taken as things happen and written in the transcript's order,
    so nothing but digests outlives its moment. Each ``.cpu()`` syncs, which is why gate arms have
    these and the ``observer`` gate compares an arm that has not."""

    FORWARD = "_forward_blockwise_adapter_model"

    def __init__(self, trainer, writer, forward, layers: int):
        self.trainer, self.writer, self.forward, self.layers = trainer, writer, forward, layers
        self.first_turn = None
        self.params = {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}
        self.microstep, self.open_step, self.at_layer = 0, None, {}
        trainer._get_lm_head().register_forward_hook(self._on_logits)
        self._real_step = trainer._adapter_optimizer_step
        self._real_layers = trainer._forward_block_layers
        setattr(trainer, self.FORWARD, self._on_forward)
        trainer._adapter_optimizer_step = self._on_step
        if layers:
            trainer._forward_block_layers = self._on_layers
            trainer._record_remat_event = self._on_event

    def _on_logits(self, module, args, output):
        self.open_step = {"logits": capture(output)}

    def _close_step(self):
        if self.open_step is None:
            return
        for layer in range(self.layers):
            self.writer.layer(self.microstep, layer, **self.at_layer[layer])
        self.writer.microstep(self.microstep, loss=self.open_step["loss"],
                              logits=self.open_step["logits"],
                              grads={n: p.grad for n, p in self.params.items()})
        self.microstep, self.open_step = self.microstep + 1, None

    def _on_forward(self, batch):
        self._close_step()
        loss = self.forward(batch)
        self.open_step["loss"] = capture(loss)
        return loss

    def _on_step(self):
        self._close_step()
        self._real_step()

    def _on_layers(self, block, hidden, mask, positions, checkpoint):
        if torch.is_grad_enabled():
            output = self._real_layers(block, hidden, mask, positions, checkpoint)
            self.first_turn = self.first_turn or machine.memory_sample(hidden.device)
            surfaces = self.at_layer[block]

            def on_cotangent(g):
                surfaces["cotangent"] = capture(g)

            output.register_hook(on_cotangent)
            return output
        rng = {"cpu": torch.get_rng_state()}
        if hidden.device.type == "mps":
            rng["mps"] = torch.mps.get_rng_state()
        surfaces = self.at_layer[block] = {"rng": capture(rng), "input": capture(hidden)}
        output = self._real_layers(block, hidden, mask, positions, checkpoint)
        surfaces["output"] = capture(output)
        return output

    def _on_event(self, event, block):
        if event == "adapter_backward_release":
            params = self.trainer.block_trainable_params[block]
            self.at_layer[block]["lora_grad"] = capture({n: p.grad for n, p in params})


class RunRecorder(Recorder):
    """The same hooks under ``poros.Run``: its forward stops at the logits, and its loss is named
    by the caller, so each row's cross entropy is digested as well as the mean over them."""

    FORWARD = "_forward_train_logits"

    def _on_forward(self, batch):
        self._close_step()
        return self.forward(batch)

    def loss(self, logprobs, batch):
        out = cross_entropy(logprobs, batch)
        self.open_step["loss"] = capture({"logprobs": logprobs, "mean": out.value})
        return out


def _train_on_chats(trainer, recorder, a) -> None:
    """What the SFT recipe does. A Mac pads to the run's length itself; a CPU double is told to."""
    data = chat_rows(trainer.model.config.get_text_config().vocab_size, a.seq, a.g * a.steps, a.g)
    run = Run.from_trainer(trainer)
    for step in run.steps(a.steps):
        batch = data.batch(step)
        batch.pad_to = a.seq
        run.forward_backward(batch, loss=recorder.loss if recorder else "cross_entropy")
        run.optim_step()


def _evict(path: Path) -> float:
    """Drop a file from the page cache, and say how much of it is still there. ``msync`` with
    ``MS_INVALIDATE`` on a read-only map needs no copy and no sudo, and ``mincore`` touches no page.
    """
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                          ctypes.c_int, ctypes.c_longlong)
    libc.msync.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    libc.mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p)
    libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    length = path.stat().st_size
    with path.open("rb") as file:
        address = libc.mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED, file.fileno(), 0)
    if address == ctypes.c_void_p(-1).value:
        raise OSError(ctypes.get_errno(), "mmap")
    pages = ctypes.create_string_buffer(-(-length // mmap.PAGESIZE))
    failed = libc.msync(address, length, MS_INVALIDATE) or libc.mincore(address, length, pages)
    error = ctypes.get_errno()
    libc.munmap(address, length)
    if failed:
        raise OSError(error, f"msync or mincore on {path}")
    return sum(page & MINCORE_INCORE for page in pages.raw) / len(pages.raw)


def _read_from_disk(trainer, kind: str) -> dict:
    """Gate the arm's source: past the page cache, from a store evicted from it. An arm that served
    from memory would read no disk, and only the OS's own counters can show that. Embed and head are
    ``FileRows``, on a descriptor of their own: they go through the cache, and off the prediction.
    """
    manager = trainer.block_manager
    path = Path(manager.source.path)
    manager.source = trainer.model.poros_source = None  # its slot and landing go first
    layers = GateLayers(path, trainer.device)
    manager.source = _NoReadAhead(layers) if kind == "ssd-sync" else layers
    trainer._sss_source = layers  # the trainer records the source the loop now reads
    poison_released(manager, layers)
    return {
        "preflight_resident_fraction": _evict(path) if MAC else None,
        "layer_bytes": sum(e.nbytes for e in layers.source.file.entries if e.kind == "layer"),
    }


def _instrument() -> dict:
    """Decodes counted, and the time spent waiting on the device."""
    stats = {"decodes": 0, "decode_off_main_thread": False, "waited_s": 0.0}
    decoded, sync = nf4.Nf4Linear.decoded, source.sync

    def counted_decode(self):
        stats["decodes"] += 1
        stats["decode_off_main_thread"] |= threading.current_thread() != threading.main_thread()
        return decoded(self)

    def timed_sync(device):
        begun = time.perf_counter()
        sync(device)
        stats["waited_s"] += time.perf_counter() - begun

    nf4.Nf4Linear.decoded, source.sync = counted_decode, timed_sync
    return stats


def run_arm(a) -> None:
    """One arm, in this process."""
    kind, _, tag = a.arm.partition(".")
    loop = kind in LOOP_KINDS
    out = Path(a.out)
    if kind == "plain":
        built = load.build
        load.build = lambda *args, **kwargs: make_plain(built(*args, **kwargs))
    config = PorosConfig(
        model_name_or_path=a.model,
        store="ssd" if kind.startswith("ssd") else "ram",
        max_seq_length=a.seq,
        gradient_accumulation_steps=a.g,
        max_steps=a.steps,
        lora_dropout=a.dropout,
        learning_rate=a.learning_rate,
        warmup_steps=a.warmup_steps,
        save_steps=0,
        eval_steps=0,
        output_dir=str(out / f"{a.arm}.run"),
        store_path=str(out / "store") if Path(a.model).is_dir() else str(store.pick(a.model)),
        progress=False,
    )
    trainer_class = PorosTrainer if torch.backends.mps.is_available() else CpuDouble
    trainer = trainer_class.from_pretrained(a.model, config=config)
    if trainer.device.type == "cpu":
        torch.set_num_threads(1)

    manager, witness = trainer.block_manager, trainer._witness
    if not loop or tag in ("rng-off", "quiet"):
        witness.remove()
        trainer._witness = None
    if tag.startswith("rng-off"):
        blockwise.torch = _NoRestore(torch)
    disk = _read_from_disk(trainer, kind) if config.store == "ssd" else {}
    if not loop:
        for block in range(manager.num_blocks):
            manager.load_block_for_inference(block)

    params = {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}
    stats = _instrument()
    memory = {"loaded": machine.memory_sample(trainer.device)}
    records = 0 if tag == "quiet" else a.g * a.steps
    layers = manager.num_layers if loop and records else 0
    writer = TranscriptWriter(
        out / f"{a.arm}.jsonl", arm=a.arm, gate=a.gate, model=_label(a.model), seq=a.seq, g=a.g,
        steps=a.steps, lora_dropout=a.dropout, layers=layers, expected_records=records,
        pin=pin(config.store_path, a.commit, a.wheel_hash),
    )
    report = {"adapter_init_b2": capture(dict(params))["b2"], "layers": layers}
    chats = a.gate == CHATS_GATE
    recording = RunRecorder if chats else Recorder
    one_graph = _one_graph_logits if chats else _one_graph
    forward = getattr(trainer, recording.FORWARD) if loop else partial(one_graph, trainer)
    setattr(trainer, recording.FORWARD, forward)
    recorder = recording(trainer, writer, forward, layers) if records else None

    read_before = machine.disk_read_bytes() if MAC else 0
    begun = time.perf_counter()
    try:
        if chats:
            _train_on_chats(trainer, recorder, a)
        else:
            vocab = trainer.model.config.get_text_config().vocab_size
            trainer.train(token_stream(vocab, a.seq, a.g * a.steps), steps=a.steps)
    except PorosStepCheckError as exc:
        report.update(error=exc.code, witness=exc.witness)
        _write_json(out / f"{a.arm}.json", report)
        sys.exit(EXIT_CODES[exc.code])
    report["wall_s"] = time.perf_counter() - begun
    writer.final(
        a.g * a.steps,
        adapter=dict(params),
        adam=trainer.optimizer.state_dict(),
        scheduler=trainer._lr_scheduler.state_dict(),
    )
    writer.close()

    if disk:
        report["disk"] = {
            "preflight_resident_fraction": disk["preflight_resident_fraction"],
            "predicted_read_bytes": 2 * records * disk["layer_bytes"],  # each turn reads again
            "read_bytes": machine.disk_read_bytes() - read_before if MAC else None,
        }
    memory["first_turn"] = recorder.first_turn if recorder else None
    memory["end"] = machine.memory_sample(trainer.device)
    report.update(stats, witness={"checked": witness.checked, "mismatches": witness.mismatches})
    report["memory"] = {k: dataclasses.asdict(v) for k, v in memory.items() if v}
    report["swap_delta_bytes"] = trainer._watchdog.swap_growth if MAC else 0
    _write_json(out / f"{a.arm}.json", report)


def _label(model: str) -> str:
    return Path(model).name if Path(model).is_dir() else model


def _write_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=1) + "\n")


def write_receipt(a, names: list[str]) -> list[str]:
    """The receipt, from the arms' transcripts, and the verifier's ruling on it: its failures."""
    out, arms, swap, witness = Path(a.out), [], 0, {"checked": 0, "expected": 0, "mismatches": 0}
    for name in names:
        try:
            arm = arm_from_transcript(out / f"{name}.jsonl")
            report = json.loads((out / f"{name}.json").read_text())
        except (OSError, ValueError):  # the verifier says what is wrong with this arm
            arms.append({"name": name})
            continue
        arms.append({**arm, **({"disk": report["disk"]} if "disk" in report else {})})
        swap += report["swap_delta_bytes"]
        witness["expected"] += a.g * a.steps * report["layers"]
        if report["layers"]:
            witness["checked"] += report["witness"]["checked"]
            witness["mismatches"] += report["witness"]["mismatches"]
    path = out / "receipt.json"
    _write_json(path, {
        "schema": RECEIPT_SCHEMA,
        "gate": a.gate,
        "model": _label(a.model),
        "seq": a.seq,
        "g": a.g,
        "steps": a.steps,
        "first_microstep": 0,
        "expected_records": a.g * a.steps,
        "lora_dropout": a.dropout,
        "pin": next((arm["pin"] for arm in arms if "pin" in arm), {}),
        "arms": arms,
        "witness": witness,
        "swap_delta_gb": swap / 1e9,
        "pass": None,
    })
    return write_verdict(path, out)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True, help="comma-separated")
    ap.add_argument("--gate", default="witness")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--g", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--warmup-steps", type=int, default=100)
    # A stock Mac has no git, so whoever syncs the tree passes this. The verifier refuses "unknown".
    ap.add_argument("--commit", default=os.environ.get("POROS_COMMIT", "unknown"))
    ap.add_argument("--wheel-hash", default="none")
    ap.add_argument("--out", required=True)
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return ap


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    a = parser().parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.child:
        run_arm(a)
        return 0
    names = a.arm.split(",")
    for name in names:  # a first call, a heap state and an RNG stream are each arm's own
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), *argv, "--arm", name, "--child"],
            check=False,
        )
    failures = write_receipt(a, names)
    print(json.dumps({"receipt": str(Path(a.out) / "receipt.json"), "pass": not failures,
                      "failures": failures}, indent=1))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
