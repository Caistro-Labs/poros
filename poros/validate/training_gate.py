"""The TRAINING gate: streamed against resident, bit for bit, on every surface a step touches.

One arm holds every block on the GPU at once and trains under plain autograd (no streaming, no
rematerialization); the other streams the same blocks. Both start from one seed and one data
stream, and are compared with ``torch.equal``, tolerance (0, 0):

* the loss at every step
* the adapter gradients immediately before each optimizer step
* the adapter weights after each step
* the Adam state (``exp_avg``, ``exp_avg_sq``, ``step``) after each step

The script (``scripts/training_parity.py``) runs each arm in its own process and
compares the final step's tensors. :func:`run` runs both in this process, one after the other, as
``poros prove`` does, and with ``trace`` it also keeps a sha256 of every tensor at every step, so
a difference is named by surface, step and tensor. Tracing reads tensors and changes none.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Optional

SEED = 1234
DATA_SEED = 99
SURFACES = ("grads", "weights", "opt_state")


def make_batches(vocab: int, g: int, steps: int, seq_len: int, device):
    """Deterministic data stream, independent of the global RNG."""
    import torch

    gen = torch.Generator(device="cpu").manual_seed(DATA_SEED)
    out = []
    for _ in range(steps):
        micro = []
        for _ in range(g):
            ids = torch.randint(0, vocab, (1, seq_len), generator=gen, device="cpu").to(device)
            micro.append({"input_ids": ids, "labels": ids.clone()})
        out.append(micro)
    return out


def seed_all() -> None:
    import torch

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def ran_strategy(load_strategy: str) -> str:
    """The load a record names: "gpu" here passes no strategy, so the trainer's default ran."""
    return "auto" if load_strategy == "gpu" else load_strategy


def build_trainer(model_id: str, *, g: int, steps: int, lora_dropout: float = 0.0,
                  block_size: int = 4, load_strategy: str = "gpu", block_major: bool = False):
    """The trainer one arm runs, seeded first, as the script builds it."""
    from poros.train.trainer import PorosTrainer

    seed_all()
    kwargs: dict[str, Any] = {}
    if load_strategy != "gpu":
        kwargs["nf4_load_strategy"] = load_strategy
    return PorosTrainer.from_pretrained(
        model_id,
        block_size=block_size,
        lora_dropout=lora_dropout,
        blockwise_arch_check="error",
        deterministic=True,  # a bitwise gate
        path="efficient" if block_major else "sequential",
        gradient_accumulation_steps=g,
        max_steps=steps,
        **kwargs,
    )


def _cpu(t):
    return t.detach().to("cpu", copy=True)


def _digests(named: dict[str, Any]) -> dict[str, str]:
    from poros.validate.load_gate import digest

    return {name: digest(t) for name, t in named.items()}


def run_arm(trainer, arm: str, *, g: int, steps: int, seq_len: int, block_major: bool = False,
            trace: bool = False) -> dict[str, Any]:
    """One arm's whole run on a built trainer: ``resident`` holds every block, ``blockwise``
    streams them. Returns the losses, the last step's gradients, the final weights and optimizer
    state, and with ``trace`` a digest of each at every step."""
    import torch

    assert trainer.block_manager is not None
    bm = trainer.block_manager
    model = trainer.model
    vocab = getattr(getattr(model, "config", None), "vocab_size", 32000)
    data = make_batches(vocab, g, steps, seq_len, trainer.device)

    def adapter_named_params():
        return [(n, p) for n, p in model.named_parameters()
                if p.requires_grad and "lora" in n.lower()]

    def opt_state() -> dict[str, Any]:
        out: dict[str, Any] = {}
        for gi, group in enumerate(trainer.optimizer.param_groups):
            for pi, p in enumerate(group["params"]):
                for k, v in trainer.optimizer.state.get(p, {}).items():
                    out[f"g{gi}.p{pi}.{k}"] = _cpu(v) if torch.is_tensor(v) else torch.tensor(
                        float(v))
        return out

    def resident_loss(batch):
        hidden, attn, pos, labels = trainer._build_inputs(batch)
        with torch.amp.autocast("cuda", enabled=False), torch.enable_grad():
            for block_id in range(bm.num_blocks):
                hidden = trainer._forward_block_layers(block_id, hidden, attn, pos,
                                                       checkpoint=False)
            # the streamed arm's own head and loss (DeviceHead.loss): one compute in both arms
            return trainer.head.loss(hidden, labels, normed=False)

    if arm == "resident":
        for block_id in range(bm.num_blocks):
            bm.load_block_for_inference(block_id, trainer.device)

    dump: dict[str, Any] = {"losses": [], "h2d_bytes_total": None}
    if trace:
        dump["trace"] = {s: [] for s in SURFACES}
    trainer.optimizer.zero_grad(set_to_none=True)
    t0 = time.time()
    for step, micro in enumerate(data):
        if arm == "blockwise" and block_major:
            from poros.train import block_major as driver

            batches = [trainer._move_batch_to_device(b) for b in micro]
            trainer.begin_passes(step)
            rng = driver.rng_for(trainer, len(batches))
            assert rng is not None, "block-major does not run this arm"
            for b in batches:
                rng.add(driver.positions(b))
            losses = driver.update(trainer, batches, rng)
            assert losses is not None, "block-major stopped this update"
        else:
            losses = []
            for b in micro:
                b = trainer._move_batch_to_device(b)
                if arm == "resident":
                    loss = resident_loss(b)
                else:
                    loss = trainer._forward_blockwise_adapter_model(b)
                (loss / g).backward()
                losses.append(loss.detach())
                if arm == "blockwise":
                    trainer._release_retained_forward_blocks()
        # every arm's microstep losses summed the same way, in fp32, so the reported per-step
        # average compares bitwise
        acc = losses[0].float()
        for loss in losses[1:]:
            acc = acc + loss.float()
        dump["losses"].append((acc / g).to("cpu", copy=True))
        grads = {n: p.grad for n, p in adapter_named_params() if p.grad is not None}
        if trace:
            dump["trace"]["grads"].append(_digests(grads))
        if step == len(data) - 1:
            dump["grads"] = {n: _cpu(t) for n, t in grads.items()}
        trainer._adapter_optimizer_step()
        if trace:
            dump["trace"]["weights"].append(_digests(dict(adapter_named_params())))
            dump["trace"]["opt_state"].append(_digests(opt_state()))

    dump["seconds"] = time.time() - t0
    dump["weights"] = {n: _cpu(p) for n, p in adapter_named_params()}
    dump["opt_state"] = opt_state()
    h2d = getattr(bm, "h2d_bytes_total", None)
    dump["h2d_bytes_total"] = int(h2d) if h2d is not None else None
    return dump


def _first(a: list[dict[str, str]], b: list[dict[str, str]]) -> Optional[dict[str, Any]]:
    """The first step (1-based) and tensor at which two traces differ, or None."""
    for step, (x, y) in enumerate(zip(a, b), start=1):
        if x != y:
            names = sorted(set(x) ^ set(y)) or sorted(k for k in x if x[k] != y[k])
            return {"step": step, "tensor": names[0]}
    if len(a) != len(b):
        return {"step": min(len(a), len(b)) + 1, "tensor": None}
    return None


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """The resident dump ``a`` against the blockwise dump ``b``: the script's record, and with
    traces ``first_difference`` per surface (``losses``, ``grads``, ``weights``, ``opt_state``)."""
    import torch

    result: dict[str, Any] = {}
    pairs = list(zip(a["losses"], b["losses"]))
    same_count = len(a["losses"]) == len(b["losses"])
    result["losses_equal"] = bool(same_count and all(torch.equal(x, y) for x, y in pairs))
    result["loss_max_abs_diff"] = max((float((x - y).abs().max()) for x, y in pairs),
                                      default=0.0)
    result["resident_losses"] = [float(x) for x in a["losses"]]
    result["blockwise_losses"] = [float(x) for x in b["losses"]]

    for section in SURFACES:
        keys_a, keys_b = set(a[section]), set(b[section])
        eq = keys_a == keys_b and all(torch.equal(a[section][k], b[section][k]) for k in keys_a)
        diffs = [float((a[section][k].float() - b[section][k].float()).abs().max())
                 for k in (keys_a & keys_b)]
        result[f"{section}_equal"] = bool(eq)
        result[f"{section}_count"] = len(keys_a)
        result[f"{section}_max_abs_diff"] = max(diffs) if diffs else 0.0

    result["resident_seconds"] = a["seconds"]
    result["blockwise_seconds"] = b["seconds"]
    result["blockwise_h2d_bytes_total"] = b["h2d_bytes_total"]
    result["all_equal"] = all(result[k] for k in (
        "losses_equal", "grads_equal", "weights_equal", "opt_state_equal"))
    if "trace" in a and "trace" in b:
        first: dict[str, Any] = {"losses": next(
            ({"step": i} for i, (x, y) in enumerate(pairs, start=1) if not torch.equal(x, y)),
            None if same_count else {"step": min(len(a["losses"]), len(b["losses"])) + 1})}
        for section in SURFACES:
            first[section] = _first(a["trace"][section], b["trace"][section])
        result["first_difference"] = first
    return result


def free_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run(model_id: str, *, steps: int, seq_len: int, block_size: int = 4, g: int = 1,
        lora_dropout: float = 0.0, trace: bool = True) -> dict[str, Any]:
    """Both NF4 arms in this process, resident first, each from the same seed: the record the
    script writes, plus each arm's peak memory and, with ``trace``, ``first_difference``."""
    import torch

    dumps: dict[str, Any] = {}
    peaks: dict[str, Optional[float]] = {}
    for arm in ("resident", "blockwise"):
        free_gpu()
        trainer = build_trainer(model_id, g=g, steps=steps, lora_dropout=lora_dropout,
                                block_size=block_size, load_strategy="gpu")
        dumps[arm] = run_arm(trainer, arm, g=g, steps=steps, seq_len=seq_len, trace=trace)
        peaks[arm] = (round(torch.cuda.max_memory_allocated() / 1024**3, 3)
                      if torch.cuda.is_available() else None)
        del trainer
        free_gpu()
    return {
        "model": model_id,
        "quantization": "nf4",
        # nothing is passed for "gpu", so the trainer's default ran: auto, streamed where it can
        "load_strategy": ran_strategy("gpu"),
        "block_size": block_size,
        "g": g,
        "steps": steps,
        "seq_len": seq_len,
        "block_major": False,
        "lora_dropout": lora_dropout,
        "seed": SEED,
        "resident_peak_vram_gb": peaks["resident"],
        "blockwise_peak_vram_gb": peaks["blockwise"],
        **compare(dumps["resident"], dumps["blockwise"]),
    }
