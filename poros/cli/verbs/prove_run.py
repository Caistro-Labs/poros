"""The two training arms `prove` compares, the comparison, and its receipt.

Two references, named in the receipt's ``reference`` (never one for the other):

- ``resident`` (the default on CUDA): ``poros.validate.training_gate``'s two arms, the gate a new
  family passes. One holds every layer on the GPU for the whole run and trains under ordinary
  autograd (no streaming, no rematerialization); the other streams blocks of 4. One seed, one data
  stream; torch.equal on the loss at every step, the last step's adapter gradients, the adapter
  weights and the Adam state. A model whose resident arm the card cannot hold is said before
  anything runs.
- ``block_sizes`` (``--block-sizes``): the same streamed trainer twice, blocks of 4 and one block
  of every layer, both streamed; torch.equal on the loss history and every adapter tensor saved.
  It shows the block size changes nothing; it is no resident reference.

Nothing is rounded or tolerance-checked.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from poros import runs
from poros.cli.ui import components as c
from poros.cli.ui import simulate, term

if TYPE_CHECKING:
    from poros.cli.context import Context
    from poros.cli.verbs.prove import Arm, ProveAction, ProveOutput

STREAMED_BLOCK = 4


def _env() -> dict[str, Any]:
    """What a bitwise result is pinned to."""
    env: dict[str, Any] = {"poros": c.version(), "python": platform.python_version()}
    try:
        import torch

        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
            env["cuda"] = torch.version.cuda
            from poros.core.card import nvidia_smi

            driver = nvidia_smi("gpu=driver_version", units=True, raw=True)
            if driver and driver[0].strip():
                env["driver"] = driver[0].strip()
    except ImportError:
        pass
    try:
        import transformers

        env["transformers"] = transformers.__version__
    except ImportError:
        pass
    if simulate.enabled():
        env["gpu"] = simulate.GPU[0]
    return env


def _layers(model: str) -> int:
    from poros.plan.configs import read_model_config

    cfg = read_model_config(model) or {}
    return (
        int(
            cfg.get("num_hidden_layers") or cfg.get("text_config", {}).get("num_hidden_layers") or 0
        )
        or 80
    )


def _arm(a: "ProveAction", block: int, out: Path) -> tuple[dict, list[float]]:
    from poros.core.config import PorosConfig

    if simulate.enabled():
        trainer_cls = simulate.SimTrainer
    else:
        from poros.train.trainer import PorosTrainer as trainer_cls
    cfg = PorosConfig(
        model_name_or_path=a.model,
        dataset="synthetic",
        max_steps=a.steps,
        max_seq_length=a.seq_len,
        block_size=block,
        output_dir=str(out),
        save_steps=0,
        progress=False,
    )
    label = "streamed" if block == STREAMED_BLOCK else "streamed, one block"
    with term.get().activity(f"Training {label}, block {block}") as act:
        trainer = trainer_cls.from_pretrained(a.model, block_size=block, config=cfg)
        metrics = trainer.train("synthetic", steps=a.steps, seq_len=a.seq_len)
        trainer.save(str(out))
        act.done([c.event("Trained", label, c.duration(act.elapsed))])
    del trainer
    _free_gpu()
    return metrics, list(metrics.get("loss_history") or [])


def _free_gpu() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _tensors(out: Path) -> dict[str, Any]:
    """Every adapter tensor an arm saved (safetensors, else the .pt state)."""
    f = out / "adapter_model.safetensors"
    if f.is_file():
        from safetensors.torch import load_file

        return load_file(str(f))
    pt = out / "adapter_weights.pt"
    if pt.is_file():
        import torch

        return torch.load(str(pt), map_location="cpu", weights_only=True)
    return {}


def _digest(tensors: dict[str, Any], losses: list[float]) -> str:
    """One hash of the exact loss history and every adapter tensor's bytes."""
    h = hashlib.sha256(json.dumps([repr(x) for x in losses]).encode())
    if tensors:
        import torch

        for name in sorted(tensors):
            h.update(name.encode())
            h.update(tensors[name].contiguous().cpu().view(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def _gib(gb: Any) -> Any:
    """A trainer's peak (``peak_vram_allocated_gb``, decimal GB) in GiB."""
    return round(gb * 1e9 / 2**30, 3) if isinstance(gb, (int, float)) else None


def resident_floor_gib(model: str) -> Any:
    """What the resident arm certainly holds on the GPU for its whole run (GiB): every layer's NF4
    weights, the embedding table and LM head, the adapters' training state (the planner's
    ``_resident_hold`` for every block resident); its activations come on top. None: no sizes."""
    from poros.core.config import PorosConfig
    from poros.plan.configs import read_model_config
    from poros.plan.estimate import _sizes
    from poros.plan.make import _resident_hold

    sizes = _sizes(read_model_config(model))
    if sizes is None:
        return None
    cfg = PorosConfig(model_name_or_path=model, dataset="synthetic", block_size=1,
                      resident_blocks=sizes.layers)
    hold = _resident_hold(cfg, "dense", sizes)
    return round(hold.floor, 2) if hold is not None else None


def one_block_floor_gib(model: str) -> Any:
    """What the one-block arm of --block-sizes certainly holds on the GPU while its block is in:
    every layer's NF4 weights at once (as the frozen layers take in host RAM,
    poros.plan.estimate._nf4_host_floor). None: no sizes."""
    from poros.core.config import PorosConfig
    from poros.plan.configs import read_model_config
    from poros.plan.estimate import _nf4_host_floor, _sizes

    cfg = PorosConfig(model_name_or_path=model, dataset="synthetic")
    return _nf4_host_floor(cfg, "dense", _sizes(read_model_config(model)))


def _refuse_one_block(a: "ProveAction", free: float) -> None:
    """--block-sizes's one-block arm holds every layer on the GPU at once: a model whose layers
    alone are past the free memory is refused before anything loads."""
    from poros.core.refusal import Refusal

    one = one_block_floor_gib(a.model)
    if one is not None and one > free:
        raise Refusal(
            "E_OOM_PREDICTED", "Can't hold one block of every layer",
            f"prove --block-sizes trains one block of every layer of {a.model}: its NF4 layers "
            f"alone take {one:.2f} GiB, and {free:.2f} GiB is free",
            context=["this card has no room for a comparison of block sizes for this model; a "
                     "smaller model proves the same claim"])


def _refuse_unheld(a: "ProveAction") -> None:
    """The resident arm needs every layer on this card at once: a model it cannot hold is said
    before anything loads, with the comparison that runs here under its own name."""
    import shlex

    import torch

    from poros.core.refusal import Refusal

    floor = resident_floor_gib(a.model)
    free = torch.cuda.mem_get_info()[0] / 2**30
    if floor is not None and floor > free:
        one = one_block_floor_gib(a.model)
        if one is not None and one > free:  # the other comparison cannot hold it either
            raise Refusal(
                "E_OOM_PREDICTED", "Can't hold it resident",
                f"the resident reference holds every layer of {a.model} on the GPU: at least "
                f"{floor:.2f} GiB, and {free:.2f} GiB is free",
                context=[f"nor can prove --block-sizes: its one-block arm holds every layer too "
                         f"({one:.2f} GiB); a smaller model proves the same claim"])
        raise Refusal(
            "E_OOM_PREDICTED", "Can't hold it resident",
            f"the resident reference holds every layer of {a.model} on the GPU: at least "
            f"{floor:.2f} GiB, and {free:.2f} GiB is free",
            context=["prove --block-sizes compares blocks of 4 with one block of every layer, "
                     "both streamed: it shows the block size changes nothing, and is no "
                     "resident reference"],
            fix=shlex.join(["poros", "prove", a.model, "--block-sizes"]))


def _resident_record(a: "ProveAction") -> tuple[dict, dict]:
    """training_gate's two arms, as its ``run`` builds them (resident first), with the
    streamed arm's final adapter weights kept for the digest; then its comparison."""
    import torch

    from poros.core.refusal import Refusal
    from poros.validate import training_gate as gate

    dumps: dict[str, Any] = {}
    peaks: dict[str, Any] = {}
    for arm, label in (("resident", "every layer resident"), ("blockwise", "streamed, block 4")):
        gate.free_gpu()
        with term.get().activity(f"Training {label}") as act:
            try:
                trainer = gate.build_trainer(a.model, g=1, steps=a.steps, lora_dropout=0.0,
                                             block_size=STREAMED_BLOCK, load_strategy="gpu")
                dumps[arm] = gate.run_arm(trainer, arm, g=1, steps=a.steps, seq_len=a.seq_len)
            except torch.OutOfMemoryError:
                if arm != "resident":
                    raise
                raise Refusal(
                    "E_OOM", "Can't hold it resident",
                    f"the resident reference ran out of GPU memory on {a.model} at "
                    f"{a.seq_len} tokens", context=["prove --block-sizes compares two streamed "
                                                    "block sizes; it is no resident reference"],
                    fix=f"poros prove {a.model} --block-sizes") from None
            peaks[arm] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
            act.done([c.event("Trained", label, c.duration(act.elapsed))])
        del trainer
        gate.free_gpu()
    record = gate.compare(dumps["resident"], dumps["blockwise"])
    record.update(resident_peak_vram_gib=peaks["resident"],
                  blockwise_peak_vram_gib=peaks["blockwise"])
    return record, dumps["blockwise"]


def run(a: "ProveAction", ctx: "Context") -> "ProveOutput":
    from poros.cli.sheets.listings import largest_certified
    from poros.cli.verbs.prove import Arm, ProveOutput

    if not simulate.enabled():  # train's own first check: torch, then a CUDA GPU
        from poros.preflight.command import device_refusal

        if (no_device := device_refusal()) is not None:
            raise no_device
    stamp = datetime.now().strftime("%m%d-%H%M%S")
    work = runs.home() / "prove" / stamp
    whole = _layers(a.model)
    resident = a.reference == "resident" and not simulate.enabled()
    if a.reference == "block_sizes" and not simulate.enabled():  # before any load
        import torch

        _refuse_one_block(a, torch.cuda.mem_get_info()[0] / 2**30)
    if resident:
        _refuse_unheld(a)
        record, kept = _resident_record(a)
        loss_s = [float(x) for x in kept["losses"]]
        t_s = kept["weights"]
        surfaces = (("losses", "losses_equal"), ("adapter gradients", "grads_equal"),
                    ("adapter weights", "weights_equal"), ("optimizer state", "opt_state_equal"))
        differs = [word for word, key in surfaces if not record[key]]
        compared = ["losses", f"{record['grads_count']} adapter gradients",
                    f"{record['weights_count']} adapter tensors",
                    f"{record['opt_state_count']} optimizer tensors"]
        identical = not differs and bool(loss_s) and bool(t_s)
        if not t_s:
            differs.append("no adapter tensors were trained to compare")

        def gate_arm(which: str, block: int, losses: list, seconds: Any) -> "Arm":
            return Arm(block_size=block, peak_vram_gib=record.get(f"{which}_peak_vram_gib"),
                       s_per_step=round(seconds / a.steps, 4) if seconds else None,
                       final_loss=losses[-1] if losses else None)

        streamed_arm = gate_arm("blockwise", STREAMED_BLOCK, record["blockwise_losses"],
                                record.get("blockwise_seconds"))
        reference_arm = gate_arm("resident", whole, record["resident_losses"],
                                 record.get("resident_seconds"))
    else:
        m_s, loss_s = _arm(a, STREAMED_BLOCK, work / "streamed")
        m_r, loss_r = _arm(a, whole, work / "one_block")
        t_s, t_r = _tensors(work / "streamed"), _tensors(work / "one_block")
        differs = [] if loss_s == loss_r else ["losses"]
        if t_s or t_r:
            import torch

            for name in sorted(set(t_s) | set(t_r)):
                if name not in t_s or name not in t_r or not torch.equal(t_s[name], t_r[name]):
                    differs.append(name)
        compared = ["losses"] + [f"{len(t_s)} adapter tensors"]
        if not (t_s or t_r) and not simulate.enabled():
            differs.append("no adapter tensors were saved to compare")
        # Outside simulation "bit for bit" needs weights compared, not losses alone.
        identical = not differs and bool(loss_s) and (simulate.enabled() or bool(t_s))

        def arm(m: dict, block: int) -> "Arm":
            sps = m.get("avg_step_time_ms")
            return Arm(block_size=block, peak_vram_gib=_gib(m.get("peak_vram_allocated_gb")),
                       s_per_step=round(sps / 1000, 4) if sps else None,
                       final_loss=m.get("final_loss"))

        streamed_arm, reference_arm = arm(m_s, STREAMED_BLOCK), arm(m_r, whole)
    env = _env()
    digest = _digest(t_s, loss_s) if not differs else None

    out = ProveOutput(
        model=a.model,
        steps=a.steps,
        seq_len=a.seq_len,
        simulated=simulate.enabled(),
        reference=a.reference,
        streamed=streamed_arm,
        resident=reference_arm if a.reference == "resident" else None,
        one_block=reference_arm if a.reference == "block_sizes" else None,
        identical=identical,
        compared=compared,
        differs=differs,
        receipt="",
        env=env,
        largest_certified=largest_certified(ctx),
    )
    if a.receipt:
        before = json.loads(Path(a.receipt).read_text())
        out.against = a.receipt
        out.reproduced = digest is not None and digest == before.get("digest")
        out.pinned_differs = {
            k: [before["env"].get(k), env.get(k)]
            for k in ("gpu", "driver", "cuda", "torch", "transformers")
            if before.get("env", {}).get(k) != env.get(k)
        }
    path = runs.home() / "receipts" / f"prove-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.receipt = str(path)
    payload = {**out.model_dump(mode="json", by_alias=True), "digest": digest}
    runs.write_json(path, payload)
    out.exit_code = 0 if out.identical and out.reproduced is not False else 1
    return out
