"""A tiny Qwen2 for the Mac path's tests, built the way the real one is: a checkpoint on disk,
packed, then a skeleton with its linears swapped. Decoding takes the CPU route."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "metal"))

import arms  # noqa: E402
from arms import tiny_config, write_checkpoint  # noqa: E402, F401

from poros.core.arch_registry import discover_layers  # noqa: E402
from poros.core.config import PorosConfig  # noqa: E402
from poros.metal.capture import capture, digest_tensor  # noqa: E402
from poros.metal.load import build  # noqa: E402
from poros.metal.nf4 import Nf4Linear  # noqa: E402
from poros.metal.pack import pack  # noqa: E402
from poros.metal.source import RamLayers  # noqa: E402
from poros.train.model import _attach_peft_lora  # noqa: E402
from poros.train.trainer import PorosTrainer  # noqa: E402

MODEL_ID = "tiny/qwen2"
REVISION = "0123456789abcdef"
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}


def packed(tmp_path, cfg):
    """Checkpoint and pack. Returns the source directory and the pack directory."""
    source = tmp_path / "source"
    write_checkpoint(source, cfg)
    return source, pack(source, tmp_path / "store", model_id=MODEL_ID, revision=REVISION)


def build_model(pack_path, cfg, *, plain=False, file_rows=False, device="cpu"):
    """The real skeleton, every layer resident."""
    device = torch.device(device)
    model = build(pack_path, cfg, file_rows=file_rows, device=device)
    if plain:
        arms.make_plain(model)
    model.poros_source = RamLayers(pack_path / "model.sss", device)
    for index, layer in enumerate(discover_layers(model)[0]):
        for module in layer.modules():
            if isinstance(module, Nf4Linear):
                module.slab = model.poros_source.slabs[index]
    return model


def attach_lora(model, *, dropout, seed=7, rank=4):
    torch.manual_seed(seed)
    _, peft_model = _attach_peft_lora(
        model,
        lora_rank=rank,
        lora_alpha=8,
        lora_dropout=dropout,
        lora_bias="none",
        target_modules="all-linear",
    )
    peft_model.train()
    return peft_model


def make_trainer(model, peft_model, out, *, dropout, grad_accum=2, steps=5, device="cpu"):
    config = PorosConfig(
        model_name_or_path="dummy",
        quantization="nf4",
        block_size=1,
        deterministic=True,
        output_dir=str(out),
        blockwise_arch_check="off",
        enable_prefetch=False,
        per_device_batch_size=1,
        gradient_accumulation_steps=grad_accum,
        max_steps=steps,
        warmup_steps=0,
        lr_scheduler="cosine",
        save_steps=0,
        eval_steps=0,
        logging_steps=1,
        lora_dropout=dropout,
        learning_rate=1e-3,
        progress=False,
        seed=3,
    )
    trainer = PorosTrainer(config)
    trainer.model = model
    trainer._peft_model = peft_model
    trainer.device = torch.device(device)
    trainer._adapter_backend = "peft"
    trainer._save_tokenizer = lambda out_path: None  # "dummy" would ask the Hub
    trainer._metal = True
    trainer._setup_blockwise_training()
    trainer._witness.remove()  # the arms compare grads, and these faults must reach them
    trainer._witness = None
    return trainer


def loader(vocab_size, *, n=64, seq=16, seed=99):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(100, min(vocab_size, 30000), (n, seq), generator=gen)
    rows = [{"input_ids": row, "labels": row.clone()} for row in ids]
    return torch.utils.data.DataLoader(rows, batch_size=1, shuffle=False, drop_last=True)


def one_graph(trainer, batch):
    """The trainer's own pieces with no block Function and no remat."""
    hidden, mask, positions, labels = trainer._build_inputs(batch)
    hidden = hidden.detach().requires_grad_(True)
    for block in range(trainer.block_manager.num_blocks):
        hidden = trainer._forward_block_layers(block, hidden, mask, positions, checkpoint=False)
    return trainer._compute_loss(trainer._logits_from_hidden(hidden), labels)


def hf_graph(trainer, batch):
    """Transformers' own forward and loss. No Poros forward code at all."""
    return trainer.model(input_ids=batch["input_ids"], labels=batch["labels"]).loss


def lora_params(model) -> dict[str, nn.Parameter]:
    return {n: p for n, p in model.named_parameters() if p.requires_grad}


@dataclass
class Record:
    """Byte digests of one arm. Equal means equal in dtype, shape and every bit."""

    losses: list[str] = field(default_factory=list)
    logits: list[str] = field(default_factory=list)
    microstep_grads: list[str] = field(default_factory=list)  # accumulated, before any clip
    finals: dict[str, str] = field(default_factory=dict)  # adapter, adam, scheduler
    decodes: int = 0
    nonfinite: int = 0

    def surfaces(self) -> dict:
        return {"losses": self.losses, "logits": self.logits,
                "microstep_grads": self.microstep_grads, **self.finals}


def _count_decodes(model, rec) -> None:
    def counting(decoded):
        def counted():
            rec.decodes += 1
            return decoded()

        return counted

    for module in model.modules():
        if isinstance(module, Nf4Linear):
            module.decoded = counting(module.decoded)


def run_arm(pack_path, cfg, arm, out, *, dropout, grad_accum=2, steps=5, file_rows=False,
            after_setup=None, device="cpu"):
    """One arm through the real ``PorosTrainer.train``, at one thread. Only the forward differs.

    ``loop`` is the real path: ``JITAdapterBlockFunction``, remat, ``Nf4Matmul``. ``mono`` keeps
    ``Nf4Matmul`` and the head Function in one graph. ``plain`` has no Poros autograd Function,
    and ``hf`` also has no Poros forward: both are plain ``loss.backward()``.
    """
    model = build_model(pack_path, cfg, plain=arm in ("plain", "hf"), file_rows=file_rows,
                        device=device)
    peft_model = attach_lora(model, dropout=dropout)
    trainer = make_trainer(model, peft_model, out, dropout=dropout, grad_accum=grad_accum,
                           steps=steps, device=device)
    torch.set_num_threads(1)
    if after_setup is not None:
        after_setup(trainer)
    rec = Record()
    _count_decodes(model, rec)
    model.lm_head.register_forward_hook(lambda m, i, o: rec.logits.append(digest_tensor(o)))

    if arm == "loop":
        forward = trainer._forward_blockwise_adapter_model
    else:
        for block in range(trainer.block_manager.num_blocks):
            trainer.block_manager.load_block_for_inference(block, device=trainer.device)
        graph = hf_graph if arm == "hf" else one_graph
        forward = lambda batch: graph(trainer, batch)  # noqa: E731

    def grads() -> dict:
        return capture({n: p.grad for n, p in lora_params(model).items()})

    def traced_forward(batch):
        if any(p.grad is not None for p in lora_params(model).values()):
            rec.microstep_grads.append(grads()["b2"])
        loss = forward(batch)
        rec.losses.append(digest_tensor(loss))
        rec.nonfinite += capture(loss)["nonfinite_count"]
        return loss

    real_step = trainer._adapter_optimizer_step

    def traced_step():
        rec.microstep_grads.append(grads()["b2"])
        rec.nonfinite += grads()["nonfinite_count"]
        real_step()

    trainer._forward_blockwise_adapter_model = traced_forward
    trainer._adapter_optimizer_step = traced_step
    assert torch.get_num_threads() == 1
    trainer.train(loader(cfg.get_text_config().vocab_size), steps=steps)
    rec.finals = {
        "adapter": capture(dict(lora_params(model)))["b2"],
        "adam": capture(trainer.optimizer.state_dict())["b2"],
        "scheduler": capture(trainer._lr_scheduler.state_dict())["b2"],
    }
    return rec, trainer
