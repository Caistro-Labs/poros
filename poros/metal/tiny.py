"""The first rung of proving a model: a tiny random one from its real config, through the real Mac
path on the CPU, bit for bit against the model itself.

The reference is the real transformers model from ``from_pretrained``, under plain autograd, with
nothing of the Mac path in it but the NF4 round trip every frozen weight takes. Equal means
``torch.equal`` on the loss and on every adapter gradient. It says nothing about Metal or about real
shapes. A config may be anyone's: its sizes are overridden before anything is built, ``auto_map`` is
ignored, nothing here runs a repo's own code, and the only weights are the random ones written here,
as safetensors.
"""

from __future__ import annotations

import contextlib
import copy
import re
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM

from poros.core.arch_registry import asserted, discover_layers, registry_type, validated_arch_types
from poros.core.config import PorosConfig
from poros.metal.layout import config_of, empty
from poros.moe import quant
from poros.train.trainer import PorosTrainer

FACTORS = ("lora_A_gate_up", "lora_B_gate_up", "lora_A_down", "lora_B_down")
SEQ = 16
LARGEST_TINY_BYTES = 1 << 30  # a config with a width these names miss stays large: never build it
_SHOWN_MODULE = re.compile(r"[A-Za-z0-9_.]{1,120}")


class CpuDouble(PorosTrainer):
    """The Mac path with the device left on the CPU, for a machine with no Metal."""

    def __init__(self, config):
        super().__init__(config)
        self._metal, self.device = True, torch.device("cpu")


def shrink(raw: dict, *, hidden: int = 64, vocab: int = 2000, **text: Any) -> Any:
    """A real model's config.json, every setting as it ships, at sizes a test can hold: only these
    numbers change. Layer types keep their real order up to the first full attention layer, so each
    kind of layer is there."""
    raw = copy.deepcopy(raw)
    real, vision = raw.get("text_config", raw), raw.get("vision_config", {})
    if real.get("loop_config"):
        # Its layers are a structure, not a size: a loop over whole residual blocks, run several
        # times a forward (Inkling 2 Mini). Cutting the layers cuts the loop out, and moving the
        # bounds changes more than the sizes this promises to change.
        raise ValueError(
            "a looped decoder (loop_config) does not shrink: its loop spans whole blocks of its "
            "layers, so fewer layers is another model. Its own tiny models are in "
            "tests/moe/inkling2_tiny.py.")
    types = real.get("layer_types")
    layers = (types.index("full_attention") + 1 if "full_attention" in (types or ()) else 2)
    real.update(hidden_size=hidden, vocab_size=vocab, num_hidden_layers=layers,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8, use_cache=False,
                max_position_embeddings=64, attn_implementation="eager", **text)
    if types:
        real["layer_types"] = types[:layers]
    if "intermediate_size" in real:
        real["intermediate_size"] = 2 * hidden
    if "linear_num_value_heads" in real:  # Gated DeltaNet
        real.update(linear_num_value_heads=8, linear_num_key_heads=2, linear_key_head_dim=4,
                    linear_value_head_dim=4)
        if vision:
            vision.update(depth=1, hidden_size=16, intermediate_size=16, num_heads=2,
                          out_hidden_size=hidden)
    elif vision:  # Gemma 4
        real.update(global_head_dim=8, sliding_window=4)
        vision.update(hidden_size=16, intermediate_size=16, num_hidden_layers=1,
                      num_attention_heads=2, head_dim=8, global_head_dim=8,
                      num_key_value_heads=2)
    if "num_experts" in real:
        real.update(num_experts=12, moe_intermediate_size=hidden // 2)
        real.update({"top_k_experts" if "top_k_experts" in real else "num_experts_per_tok": 3})
        if "shared_expert_intermediate_size" in real:
            real["shared_expert_intermediate_size"] = hidden // 2
    elif "num_local_experts" in real:  # an expert's width is its intermediate size
        real.update(num_local_experts=12, num_experts_per_tok=3, intermediate_size=hidden // 2)
    raw.update(attn_implementation="eager", tie_word_embeddings=real.get("tie_word_embeddings",
                                                                         False),
               **({"hidden_size": hidden} if "hidden_size" in raw else {}))
    for where in {id(real): real, id(raw): raw}.values():
        _keep_special_ids_inside(where, vocab)
    return config_of(raw)


SPECIAL_IDS = ("pad_token_id", "bos_token_id", "eos_token_id", "decoder_start_token_id",
               "sep_token_id", "unk_token_id")


def _keep_special_ids_inside(cfg: dict, vocab: int) -> None:
    """The special token ids a shrunk vocabulary no longer holds, moved to its last ids: a real
    model's ids (Phi-3's pad 32000) past a tiny vocab_size fail transformers' padding_idx check
    before anything is built. Which id a special token takes changes nothing a tiny check reads."""
    for key in SPECIAL_IDS:
        v = cfg.get(key)
        if isinstance(v, int) and not isinstance(v, bool) and v >= vocab:
            cfg[key] = vocab - 1
        elif isinstance(v, list):
            cfg[key] = [vocab - 1 if isinstance(x, int) and x >= vocab else x for x in v]


def write_random(path: Path, cfg: Any, *, seed: int = 1) -> None:
    """A bf16 checkpoint as transformers saves this family. Fresh init has unit norms, zero biases
    and zero routers, so nothing could see them."""
    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        for param in model.parameters():
            param.normal_(0.0, 0.05)
    model.to(torch.bfloat16).save_pretrained(path, max_shard_size="60KB")


# -- The dense reference ---------------------------------------------------------------------


class DenseLora(nn.Module):
    """A frozen linear as training sees it, with a LoRA in plain autograd: ``x`` is fp32, the base
    GEMM is bf16 over the NF4 round trip, and the sum is fp32."""

    def __init__(self, weight, bias, a, b, scaling):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.bias = None if bias is None else nn.Parameter(bias, requires_grad=False)
        self.a, self.b, self.scaling = nn.Parameter(a.clone()), nn.Parameter(b.clone()), scaling

    def forward(self, x):
        base = F.linear(x.to(torch.bfloat16).reshape(-1, x.shape[-1]), self.weight)
        if self.bias is not None:
            base = base + self.bias
        base = base.reshape(*x.shape[:-1], -1).to(x.dtype)
        return base + F.linear(F.linear(x, self.a), self.b) * self.scaling


class DenseHead(nn.Module):
    """fp32 rows, one GEMM per chunk, summed by autograd through ``cat``, in the same chunks."""

    def __init__(self, weight, chunk=16384):
        super().__init__()
        self.weight = nn.Parameter(weight.float(), requires_grad=False)
        self.chunk = chunk

    def forward(self, hidden):
        h2d = hidden.reshape(-1, hidden.shape[-1])
        logits = [F.linear(h2d, self.weight[a : a + self.chunk])
                  for a in range(0, self.weight.shape[0], self.chunk)]
        return torch.cat(logits, dim=1).reshape(*hidden.shape[:-1], -1)


def dense_reference(source, lora: dict[str, nn.Parameter], scaling: float):
    """From the checkpoint and ``from_pretrained`` alone, as a bf16 load and kbit preparation leave
    it: every other float param in fp32. Only the LoRA's start comes from the run it checks.
    Returns the model and, per LoRA parameter's name, the parameter holding its reference grad."""
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16,
                                                 attn_implementation="eager")
    # training has none: Qwen3.5's changes its grads
    model.config.get_text_config().use_cache = False
    pairs = {}
    state = model.state_dict()
    for name, module in list(model.named_modules()):
        a = lora.get(f"{name}.lora_A.default.weight")
        if a is None:
            continue
        b = lora[f"{name}.lora_B.default.weight"]
        weight = state[f"{name}.weight"]
        decoded = quant.dequantize(quant.quantize_tensor(weight), weight.shape, torch.bfloat16)
        bias = state.get(f"{name}.bias")
        dense = DenseLora(decoded, bias, a.detach().to(a.device), b.detach(), scaling)
        parent, _, leaf = name.rpartition(".")
        setattr(model.get_submodule(parent), leaf, dense)
        pairs[f"{name}.lora_A.default.weight"] = dense.a
        pairs[f"{name}.lora_B.default.weight"] = dense.b
    model.set_output_embeddings(DenseHead(model.get_output_embeddings().weight.detach()))
    for module in model.modules():
        if not isinstance(module, (DenseLora, DenseHead)):
            for param in module.parameters(recurse=False):
                param.requires_grad_(False)
                param.data = param.data.float()
    return model.train(), pairs


def _decoded(fused: torch.Tensor) -> torch.Tensor:
    """Each expert's NF4 round trip."""
    return torch.stack([quant.dequantize(quant.quantize_tensor(w), w.shape, torch.bfloat16)
                        for w in fused])


class DenseExperts(nn.Module):
    """``Qwen3MoeExperts.forward`` as transformers writes it, plus a LoRA at each projection."""

    def __init__(self, fused: nn.Module, a_b: dict[str, torch.Tensor] | None, scaling: float):
        super().__init__()
        self.num_experts, self.act_fn, self.scaling = fused.num_experts, fused.act_fn, scaling
        self.gate_up_proj = nn.Parameter(_decoded(fused.gate_up_proj), requires_grad=False)
        self.down_proj = nn.Parameter(_decoded(fused.down_proj), requires_grad=False)
        self.factors = nn.ParameterDict(
            {f: nn.Parameter(a_b[f].detach().clone()) for f in FACTORS} if a_b else {}
        )
        self.routed: set[int] = set()

    def _lora(self, x, factor_a, factor_b, e):
        if not self.factors:
            return 0.0
        a, b = self.factors[factor_a][e], self.factors[factor_b][e]
        return (x @ a.mT @ b.mT) * self.scaling

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            self.routed.add(int(expert_idx))
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            x = hidden_states[token_idx]
            gate_up = F.linear(x, self.gate_up_proj[expert_idx])
            gate_up = gate_up + self._lora(x, "lora_A_gate_up", "lora_B_gate_up", expert_idx)
            gate, up = gate_up.chunk(2, dim=-1)
            y = self.act_fn(gate) * up
            down = F.linear(y, self.down_proj[expert_idx])
            down = down + self._lora(y, "lora_A_down", "lora_B_down", expert_idx)
            down = down * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, down.to(final_hidden_states.dtype))
        return final_hidden_states


class BiasApart(nn.Linear):
    """A bias added after the product, as ``Nf4Linear`` adds it. ``F.linear`` folds it into the GEMM
    when it can, which rounds differently in bf16, on CUDA or Metal as here, and matters to nothing
    but bits. Qwen1.5's attention is the one that has a bias."""

    def forward(self, x):
        y = F.linear(x.reshape(-1, x.shape[-1]), self.weight) + self.bias
        return y.reshape(*x.shape[:-1], -1)


def vanilla_moe(source) -> nn.Module:
    model = AutoModelForCausalLM.from_pretrained(source, dtype=torch.bfloat16,
                                                 attn_implementation="eager",
                                                 experts_implementation="eager")
    model.config.get_text_config().use_cache = False  # training has none
    return model


def sparse_blocks(model) -> list[tuple[str, nn.Module]]:
    return [(n, m) for n, m in model.named_modules() if hasattr(m, "experts")]


def moe_reference(source, lora, scaling):
    """Returns the model, and each adapter parameter's name -> the parameter of its reference."""
    model = vanilla_moe(source)
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if type(module) is nn.Linear and module.bias is not None:
            module.__class__ = BiasApart
    pairs = {}
    for name, block in sparse_blocks(model):
        block.experts = DenseExperts(
            block.experts, {f: lora[f"{name}.adapter.{f}"] for f in FACTORS}, scaling)
        pairs.update({f"{name}.adapter.{f}": block.experts.factors[f] for f in FACTORS})
    return model.train(), pairs


# -- The comparison --------------------------------------------------------------------------


def _record_outputs(model: nn.Module) -> tuple[dict[str, torch.Tensor], list]:
    """Each module's first output by name, as the model runs."""
    seen: dict[str, torch.Tensor] = {}

    def hook(name: str):
        def record(_module, _inputs, out):
            out = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(out, torch.Tensor):
                seen.setdefault(name, out.detach().float().clone())
        return record

    return seen, [m.register_forward_hook(hook(n)) for n, m in model.named_modules() if n]


def _difference(module: str, tensor: str) -> dict[str, str]:
    """A module's name comes from this install's class: shown only if it reads like one."""
    return {"module": module if _SHOWN_MODULE.fullmatch(module) else "unknown", "tensor": tensor}


def _first_difference(got, want, inside: str, lora, pairs, same_loss) -> Optional[dict[str, str]]:
    for name, tensor in want.items():
        if name.startswith(inside) and name in got and not torch.equal(got[name], tensor):
            return _difference(name, "output")
    for name, param in lora.items():
        if not torch.equal(param.grad, pairs[name].grad):
            module, _, tensor = name.rpartition(".adapter.")
            if not module:
                module, _, tensor = name.partition(".lora_")
                tensor = f"lora_{tensor}"
            return _difference(module, f"{tensor} gradient")
    return None if same_loss else {"module": "the loss", "tensor": "its value"}


def _text_only_batch(model: nn.Module, ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """What a wrapper's forward asks of a batch with no images."""
    import inspect

    takes = inspect.signature(model.forward).parameters
    return {k: torch.zeros_like(ids) for k in ("token_type_ids", "mm_token_type_ids") if k in takes}


def prove(raw: dict, root: Path) -> dict[str, Any]:
    """One microstep of a tiny model of this config on the real path and on the reference, from the
    same random adapter. A family the registry does not name runs as ``user_asserted``, for this
    call alone. ValueError for a config that does not shrink to a tiny model."""
    kind = registry_type(raw)
    validated = kind in validated_arch_types() | validated_arch_types("moe")
    status = "validated" if validated else "user_asserted"
    cfg = shrink(raw)
    model = empty(cfg)
    held = sum(t.numel() for t in (*model.parameters(), *model.buffers()))
    if 2 * held > LARGEST_TINY_BYTES:
        raise ValueError("its config did not shrink to a tiny model")
    del model
    source = root / "source"
    write_random(source, cfg)
    config = PorosConfig(
        model_name_or_path=str(source), store="ram", max_seq_length=SEQ,
        store_path=str(root / "store"), gradient_accumulation_steps=1, max_steps=2,
        lora_dropout=0.0, learning_rate=1e-2, warmup_steps=0, save_steps=0, eval_steps=0,
        output_dir=str(root / "run"), progress=False, lora_rank=4, lora_alpha=8,
    )
    threads = torch.get_num_threads()
    try:
        with asserted(kind) if not validated else contextlib.nullcontext():
            trainer = CpuDouble.from_pretrained(str(source), config=config)
        torch.set_num_threads(1)
        lora = {n: p for n, p in trainer.model.named_parameters() if p.requires_grad}
        gen = torch.Generator().manual_seed(5)
        with torch.no_grad():
            for param in lora.values():  # lora_B starts at zero, and then no gradient of A shows
                param.copy_(0.03 * torch.randn(param.shape, generator=gen, dtype=param.dtype))
        moe = bool(sparse_blocks(trainer.model))
        reference, pairs = (moe_reference if moe else dense_reference)(
            source, lora, config.lora_alpha / config.lora_rank)
        vocab = cfg.get_text_config().vocab_size
        ids = torch.randint(0, vocab, (1, SEQ), generator=torch.Generator().manual_seed(3))
        batch = {"input_ids": ids, "labels": ids.clone()}
        got, got_hooks = _record_outputs(trainer.model)
        loss = trainer._forward_blockwise_adapter_model(batch)
        for handle in got_hooks:
            handle.remove()
        loss.backward()
        want, _ = _record_outputs(reference)
        logits = reference(input_ids=ids, **_text_only_batch(reference, ids)).logits.float()
        target = trainer._compute_loss(logits, batch["labels"])
        target.backward()
        same_loss = torch.equal(loss, target)
        inside = discover_layers(trainer.model)[1]
        first = _first_difference(got, want, inside, lora, pairs, same_loss)
        nonzero = sum(bool(p.grad.abs().sum() > 0) for p in lora.values())
        return {"ok": first is None and nonzero > 0, "status": status, "moe": moe,
                "first_difference": first, "compared": len(lora), "nonzero": nonzero}
    finally:
        torch.set_num_threads(threads)
