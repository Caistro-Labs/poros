"""A Mac no run measured checks itself, once, before its first training run. Seconds, not minutes.

``ensure`` runs the checks in a process of their own, so the run that asked has touched no MPS
memory yet, and keeps what they found in Poros's home (``machine/self-check.json``). A pass stands
for this Mac until its class, its macOS build, torch, the kernel or Poros changes; a failure is
checked again the next time a run asks, and refuses it until it passes. Nothing in the record names
the Mac, its user or a path. The checks:

- NF4: the kernel against the CPU's decode (``poros.moe.quant.dequantize``) on random weights at the
  sizes real ones have, at two offsets in a slab, and on a block built so a contracted multiply-add
  would show: bit for bit, twice. The kernel and the CPU do the same operations in the same order,
  each rounded to nearest even, so nothing is left to differ. MSL lets a GPU flush a subnormal and
  choose a zero's sign (MSL 4.1, sections 8.1 and 8.5); weights like these give neither, so any
  difference at all fails the check.
- Matmul: the frozen GEMM's route in bf16 (``F.linear`` forward, ``torch.mm`` for its input's
  gradient) and the adapter's in fp32, at training shapes, against float64 on the CPU from the same
  values. Within 2^-7 of it in relative norm for bf16, seven times bf16's own rounding, and 2^-12
  for fp32, where a wrong product is near 1; and the same bytes twice.
- Repeat: a tiny random Qwen2, written here with no download, trained a few steps from RAM and then
  from the SSD through the real trainer on MPS: the same losses, adapter and optimizer, byte for
  byte.

It also reads the GPU's working set as torch does, beside what planning read from Metal.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

from poros import metal, runs
from poros.core.errors import PorosStackUnsupportedError

SCHEMA = "poros.metal.selfcheck.v1"
TIMEOUT_S = 600
STEPS, SEQ, VOCAB = 4, 32, 2048
BF16_TOLERANCE, FP32_TOLERANCE = 2.0**-7, 2.0**-12
CHILD = [sys.executable, "-P", "-m", "poros.metal.selfcheck"]
_inside = False  # True in the process that runs the checks: its own trainer must not ask again
_store = ""  # the store its tiny model trains from, in its own temporary folder


def inside() -> bool:
    return _inside


def scratch_store() -> str:
    """The check's own store, in its process: a few MB in a folder it deletes, so no disk rule."""
    return _store if _inside else ""


def path() -> Path:
    return runs.home() / "machine" / "self-check.json"


def shown() -> str:
    """``path()`` as a person reads it, ``~`` for the home folder, so no message names the user."""
    home, at = str(Path.home()), str(path())
    return "~" + at[len(home):] if at.startswith(home + "/") else at


def key(here: str) -> dict[str, str]:
    """What a record stands for. Any of it changed, the check runs again."""
    from poros import __version__
    from poros.metal import shaders

    return {"device_class": here, "macos_build": metal.macos_build(),
            "torch": metal.installed("torch") or "", "shader_sha256": shaders.NF4_DECODE_SHA256,
            "poros": __version__}


def _read() -> Optional[dict]:
    try:
        record = json.loads(path().read_text())
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) and record.get("schema") == SCHEMA else None


def recorded(here: str) -> Optional[dict]:
    """This Mac's record as it is now, or None: never checked, or something it stands for moved."""
    record = _read()
    return record if record is not None and record.get("key") == key(here) else None


def recorded_working_set(here: str) -> Optional[int]:
    """The working set torch read when this Mac checked itself: what a plan falls back to."""
    found = (((recorded(here) or {}).get("checks") or {}).get("working_set") or {}).get("torch")
    return found if isinstance(found, int) and found > 0 else None


def check_row(here: str) -> tuple[str, str, str]:
    """``poros machine``'s row for it."""
    record = recorded(here)
    if record is None:
        return ("Self-check", "not run yet: the first `train` here runs it, in seconds", "info")
    if record.get("pass"):
        when = str(record.get("when", ""))[:10]
        return ("Self-check", f"passed {when}: the NF4 kernel bit for bit, bf16 products, and RAM "
                "and SSD the same", "ok")
    return ("Self-check", f"failed: {record.get('failed')}. Send {shown()} to the Poros team",
            "error")


def ensure(here: str) -> None:
    """Pass if this Mac passed its check as it is now; else run it. Raise if it fails."""
    if _inside:
        return
    record = recorded(here)
    if record is None or not record.get("pass"):
        record = run(here)
    if record.get("key") != key(here):  # it checked another Mac than this one: no pass here
        record = {**record, "pass": False,
                  "failed": record.get("failed") or "the check ran as another Mac than this one"}
    if not record.get("pass"):
        raise PorosStackUnsupportedError(
            f"This Mac's own check failed, so it cannot train yet: {record.get('failed')}.",
            remediation=f"Send {shown()} and what `poros agents machine` prints to the Poros "
                        "team; neither names you or this Mac. `train` checks again each time.",
        )


def run(here: str) -> dict:
    """The checks in a process of their own, kept in Poros's home whatever they found."""
    print("Poros is checking this Mac once, before its first run here. It takes seconds.",
          file=sys.stderr, flush=True)
    with tempfile.TemporaryDirectory(prefix="poros-check-") as work:
        out = Path(work) / "result.json"
        try:
            done = subprocess.run([*CHILD, "--out", str(out), "--work", work],
                                  capture_output=True, text=True, timeout=TIMEOUT_S, check=False)
            ended = f"its process ended with exit {done.returncode} before it said why"
        except subprocess.TimeoutExpired:
            ended = f"it did not finish in {TIMEOUT_S // 60} minutes"
        try:
            record = json.loads(out.read_text())
        except (OSError, ValueError):
            record = None
    if not isinstance(record, dict) or record.get("schema") != SCHEMA:
        record = {"schema": SCHEMA, "key": key(here), "when": runs.now(), "pass": False,
                  "failed": ended, "checks": {}}
    path().parent.mkdir(parents=True, exist_ok=True)
    runs.write_json(path(), record)
    return record


# -- In the check's own process ---------------------------------------------------------------


class Failed(Exception):
    """A check that ran and found this Mac wrong, in Poros's words."""


def _bytes_differ(got: Any, want: Any) -> tuple[int, Optional[dict]]:
    """(how many elements differ in their bits, the first as index and both bit patterns)."""
    import torch

    a, b = got.reshape(-1).view(torch.int16), want.reshape(-1).view(torch.int16)
    differ = (a != b).nonzero()
    if not len(differ):
        return 0, None
    i = int(differ[0])
    return len(differ), {"index": i, "gpu": f"{int(a[i]) & 0xFFFF:04x}",
                         "cpu": f"{int(b[i]) & 0xFFFF:04x}"}


def _contracted_block() -> Any:
    """One block whose bf16 a contracted ``a * b + c`` changes: tests/metal/test_nf4.py's."""
    import torch

    from poros.moe import quant

    n = 64
    blob = torch.zeros(quant.quantized_nbytes((n,)), dtype=torch.uint8)
    packed_b, nblocks, nabsmax2 = quant._section_sizes(n)
    pa = quant._align4(packed_b)
    a2 = pa + quant._align4(nblocks)
    off = a2 + 4 * nabsmax2
    blob[:packed_b] = 0xFF  # code 15 everywhere: 1.0
    blob[pa : pa + nblocks] = 4
    blob[a2:off] = torch.tensor([float.fromhex("0x1.5bd214p-9")]).view(torch.uint8)
    blob[off : off + 4] = torch.tensor([float.fromhex("0x1.3deb62p-9")]).view(torch.uint8)
    return blob


def check_nf4(device: str = "mps") -> dict:
    """The kernel against the CPU's decode, bit for bit, twice, at two offsets in a slab."""
    import torch

    from poros.metal import nf4
    from poros.moe import quant

    decode = nf4._kernel_decode

    gen = torch.Generator().manual_seed(0)
    # A lone element, a block's tail, one block, the edge of an absmax group, and a 0.5B MLP matrix,
    # each at a spread of scales as real layers have.
    blobs = []
    for n in (1, 63, 64, 65, 64 * 256 + 1, 896 * 4864):
        scale = torch.exp(torch.randn(-(-n // 64), generator=gen)).repeat_interleave(64)[:n]
        blobs.append((n, quant.quantize_tensor(torch.randn(n, generator=gen) * 0.02 * scale)))
    blobs.append((64, _contracted_block()))
    elements, differ, first = 0, 0, None
    for n, blob in blobs:
        want = quant.dequantize(blob, (n,), torch.bfloat16)
        for start in (0, 64):
            slab = torch.zeros(start + blob.numel(), dtype=torch.uint8)
            slab[start:] = blob
            slab = slab.to(device)
            twice = []
            for _ in range(2):
                out = torch.empty(n, dtype=torch.bfloat16, device=device)
                decode(out, slab, start, n)
                twice.append(out.cpu())
            for got in twice:
                count, where = _bytes_differ(got, want)
                elements, differ = elements + n, differ + count
                first = first or where
    return {"pass": differ == 0, "elements": elements, "bit_differences": differ,
            "first": first, "says": f"the NF4 kernel differs from the CPU in {differ} of "
                                    f"{elements} elements, first {first}" if differ else ""}


def _relative(got: Any, want: Any) -> float:
    return float((got.double() - want).norm() / want.norm().clamp_min(1e-300))


def check_matmul(device: str = "mps") -> dict:
    """bf16 products on the frozen route and fp32 on the adapter's, against float64 on the CPU."""
    import torch
    import torch.nn.functional as F

    gen = torch.Generator().manual_seed(1)
    worst, cases, repeat = {"bf16": 0.0, "fp32": 0.0}, 0, True
    # a 0.5B MLP, a square, an odd small one, and a 7B MLP at the rows of a short example
    for m, k, n in ((256, 896, 4864), (256, 2048, 2048), (7, 64, 130), (64, 3584, 18944)):
        x = torch.randn(m, k, generator=gen).to(torch.bfloat16)
        w = (torch.randn(n, k, generator=gen) * 0.05).to(torch.bfloat16)
        g = torch.randn(m, n, generator=gen).to(torch.bfloat16)
        xm, wm, gm = x.to(device), w.to(device), g.to(device)
        for got, want in (((lambda: F.linear(xm, wm)), x.double() @ w.double().T),
                          ((lambda: torch.mm(gm, wm)), g.double() @ w.double())):
            first, second = got().cpu(), got().cpu()
            repeat = repeat and torch.equal(first.view(torch.int16), second.view(torch.int16))
            worst["bf16"] = max(worst["bf16"], _relative(first, want))
            cases += 1
        a = torch.randn(16, k, generator=gen) * 0.02
        x3 = torch.randn(2, m, k, generator=gen)
        first = F.linear(x3.to(device), a.to(device)).cpu()
        second = F.linear(x3.to(device), a.to(device)).cpu()
        repeat = repeat and torch.equal(first, second)
        worst["fp32"] = max(worst["fp32"], _relative(first, x3.double() @ a.double().T))
        cases += 1
    ok = worst["bf16"] <= BF16_TOLERANCE and worst["fp32"] <= FP32_TOLERANCE and repeat
    says = "" if ok else (
        f"matrix products on MPS are off: relative error {worst['bf16']:.2e} in bf16 (at most "
        f"{BF16_TOLERANCE:.2e}), {worst['fp32']:.2e} in fp32 (at most {FP32_TOLERANCE:.2e}), "
        f"{'the same' if repeat else 'not the same'} twice")
    return {"pass": ok, "cases": cases, "relative_error": worst, "repeat_equal": repeat,
            "says": says}


def _tiny(source: Path) -> None:
    """A tiny Qwen2 in bf16, its biases and norms off their fresh values so a check sees them."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=VOCAB,
                      max_position_embeddings=64, use_cache=False, tie_word_embeddings=False,
                      attn_implementation="eager")
    torch.manual_seed(1)
    model = Qwen2ForCausalLM(cfg)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.normal_(0.0, 0.02)
            elif "norm" in name:
                param.add_(0.1 * torch.randn_like(param))
    model.to(torch.bfloat16).save_pretrained(source)


def check_repeat(work: Path, trainer_class: Any = None) -> dict:
    """The tiny model trained from RAM and from the SSD: the same bytes. ``trainer_class`` is the
    real trainer, which is on MPS on a Mac; a test off one gives the CPU double of the Mac path."""
    import torch
    from torch.utils.data import DataLoader

    from poros.core.config import PorosConfig
    from poros.metal.capture import capture
    from poros.train.trainer import PorosTrainer

    trainer_class = trainer_class or PorosTrainer
    source = work / "model"
    _tiny(source)
    ids = torch.randint(0, VOCAB, (STEPS, SEQ), generator=torch.Generator().manual_seed(99))
    seen = {}
    for tier in ("ram", "ssd"):
        config = PorosConfig(
            model_name_or_path=str(source), store=tier, store_path=str(work / "store"),
            max_seq_length=SEQ, gradient_accumulation_steps=1, max_steps=STEPS, save_steps=0,
            eval_steps=0, eval_split=0.0, lora_dropout=0.05, learning_rate=1e-3, warmup_steps=0,
            output_dir=str(work / tier), progress=False,
        )
        torch.manual_seed(0)
        trainer = trainer_class.from_pretrained(str(source), config=config)
        data = DataLoader([{"input_ids": r, "labels": r.clone()} for r in ids], batch_size=1)
        metrics = trainer.train(data, steps=STEPS)
        params = {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}
        seen[tier] = {"losses": [float(v) for v in metrics["loss_history"]],
                      "adapter": capture(params)["b2"],
                      "optimizer": capture(trainer.optimizer.state_dict())["b2"]}
        on_mps = trainer.device.type == "mps"
        del trainer, params
        if on_mps:
            torch.mps.empty_cache()
    same = seen["ram"] == seen["ssd"]
    differs = [k for k in ("losses", "adapter", "optimizer") if seen["ram"][k] != seen["ssd"][k]]
    return {"pass": same and len(seen["ram"]["losses"]) == STEPS, "steps": STEPS, "seq": SEQ,
            "ram": seen["ram"], "ssd": seen["ssd"],
            "says": "" if same else f"training from RAM and from the SSD differ in "
                                     f"{', '.join(differs)}"}


def check_mps() -> dict:
    """MPS with bf16 at all: what every other check stands on."""
    import torch

    if not torch.backends.mps.is_available():
        raise Failed(metal._no_mps())
    try:
        got = (torch.tensor([1.5, -2.25], dtype=torch.bfloat16, device="mps") * 2).cpu()
    except (RuntimeError, TypeError) as exc:
        raise Failed("bf16 on MPS is not available here, and Poros needs it: macOS 14 or "
                     f"later has it ({type(exc).__name__})") from exc
    if got.tolist() != [3.0, -4.5]:
        raise Failed(f"bf16 on MPS gave {got.tolist()} for [3.0, -4.5]")
    return {"pass": True, "macos": ".".join(map(str, metal.macos_version()))}


def check_working_set() -> dict:
    """What torch reads as the GPU's working set, beside what planning read from Metal."""
    import torch

    from poros.metal import machine

    found, planned = int(torch.mps.recommended_max_memory()), machine.gpu_working_set()
    return {"pass": planned in (None, found), "torch": found, "metal": planned,
            "says": "" if planned in (None, found) else
            f"the GPU's working set reads {found} bytes in torch and {planned} from Metal"}


def _clean(text: str, work: str) -> str:
    """One line of an exception's message, with no path of this Mac's in it."""
    line = (text.strip().splitlines() or [""])[0].replace(work, "<check>")
    return re.sub(r"(?:~|/)[^\s'\"():,]*/[^\s'\"():,]*", "<path>", line)[:300]


def main(argv: Optional[list[str]] = None) -> int:
    global _inside, _store
    _inside = True
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    a = ap.parse_args(argv)
    _store = str(Path(a.work) / "store")
    here = metal.device_class()
    record: dict[str, Any] = {"schema": SCHEMA, "key": key(here), "when": runs.now(),
                              "checks": {}}
    failed, start = [], time.monotonic()
    try:
        record["checks"]["mps"] = check_mps()
        for name, check in (("working_set", check_working_set), ("nf4", check_nf4),
                            ("matmul", check_matmul),
                            ("repeat", lambda: check_repeat(Path(a.work)))):
            found = record["checks"][name] = check()
            if not found["pass"]:
                failed.append(found["says"])
    except Failed as exc:
        failed.append(str(exc))
    except Exception as exc:  # anything else is a check that could not finish: say what stopped it
        failed.append(f"a check stopped: {type(exc).__name__}: {_clean(str(exc), a.work)}")
    record.update({"pass": not failed, "failed": "; ".join(failed),
                   "seconds": round(time.monotonic() - start, 1)})
    Path(a.out).write_text(json.dumps(record) + "\n")
    return 0 if record["pass"] else 1


if __name__ == "__main__":
    from poros.metal import selfcheck  # the module ``ensure`` reads, not this ``__main__`` copy

    sys.exit(selfcheck.main())
