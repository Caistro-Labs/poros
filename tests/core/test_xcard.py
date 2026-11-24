"""Every NVIDIA card class through the choosers, on the CPU: the device queries are faked.

For each class the outcome is one of: the fast path, a fallback with its reason recorded, or a
refusal before the download with the exact remedy (poros.core.card). Real-GPU evidence per class:
scripts/xcard_check.sh.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from poros.core import card  # noqa: E402
from poros.core.compute import choose  # noqa: E402
from poros.core.config import PorosConfig  # noqa: E402
from poros.core.errors import EXIT_CODES, PorosConfigError  # noqa: E402

# (capability, a card of the class, the arch list of a torch build that has kernels for it)
CU126 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
CU130 = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]
CARDS = [
    ((6, 1), "NVIDIA GeForce GTX 1080 Ti", CU126),
    ((7, 0), "Tesla V100-SXM2-32GB", CU126),
    ((7, 5), "Tesla T4", CU130),
    ((8, 0), "NVIDIA A100-SXM4-80GB", CU130),
    ((8, 6), "NVIDIA GeForce RTX 3090", CU130),
    ((8, 6), "NVIDIA A10", CU130),
    ((8, 9), "NVIDIA GeForce RTX 4090", CU130),
    ((8, 9), "NVIDIA L4", CU130),
    ((9, 0), "NVIDIA H100 80GB HBM3", CU130),
    ((10, 0), "NVIDIA B200", CU130),
    ((12, 0), "NVIDIA GeForce RTX 5090", CU130),
]


def _card(monkeypatch, cap, name, arch_list):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: cap)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device=None: name)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: list(arch_list))


def _cfg(**kw):
    return PorosConfig(model_name_or_path="Qwen/Qwen2.5-0.5B", dataset="synthetic", **kw)


# -- the torch build ----------------------------------------------------------------------------

@pytest.mark.parametrize("cap, arch_list, runs", [
    ((8, 9), ["sm_80", "sm_86"], True),            # sm_86 SASS runs on sm_89
    ((8, 6), ["sm_80", "sm_89"], True),            # sm_80 SASS runs on sm_86
    ((8, 0), ["sm_86", "sm_90"], False),           # a newer minor never runs on an older one
    ((12, 0), ["sm_80", "sm_86", "sm_90"], False),  # an old wheel on Blackwell: no kernel image
    ((12, 0), ["sm_80", "compute_90"], True),      # PTX JIT-compiles forward
    ((9, 0), ["sm_80", "sm_90a"], False),          # another major's SASS never runs; nor 90a's
    ((9, 0), ["sm_80", "sm_90"], True),
    ((10, 0), ["sm_90a"], False),                  # arch-specific code never runs forward
    ((7, 0), CU130, False),                        # a CUDA 13 wheel dropped Volta
    ((7, 5), CU130, True),
    ((8, 6), [], True),                            # a build that does not say: taken as able
])
def test_build_reason(cap, arch_list, runs):
    assert (card.build_reason(cap, arch_list) is None) is runs


def test_a_build_without_kernels_for_the_card_is_refused_before_the_download(monkeypatch):
    _card(monkeypatch, (12, 0), "NVIDIA GeForce RTX 5090", ["sm_80", "sm_86", "sm_90"])
    with pytest.raises(PorosConfigError) as e:
        card.check(_cfg())
    assert "sm_120" in str(e.value) and "pytorch.org" in e.value.info.remediation
    # an environment problem (cli-contract.md: exit 5), not a config one
    assert e.value.code == "E_STACK_UNSUPPORTED" and EXIT_CODES[e.value.code] == 5


# -- bf16 -----------------------------------------------------------------------------------------

@pytest.mark.parametrize("cap, name, arch_list", CARDS)
def test_bf16_runs_from_ampere_and_is_refused_before_with_the_remedy(monkeypatch, cap, name,
                                                                     arch_list):
    _card(monkeypatch, cap, name, arch_list)
    if cap >= (8, 0):
        card.check(_cfg())  # the default: NF4 over bf16 compute
        return
    with pytest.raises(PorosConfigError) as e:
        card.check(_cfg())
    msg = str(e.value)
    assert msg.startswith("Poros needs an NVIDIA GPU with bf16 (Ampere / RTX 30xx or newer)")
    assert name in msg and card.sm(cap) in msg and "no native bf16" in msg
    # this GPU is not supported (cli-contract.md: exit 4), not a config to edit
    assert e.value.code == "E_GPU_UNSUPPORTED" and EXIT_CODES[e.value.code] == 4
    remedy = e.value.info.remediation
    assert remedy.startswith("Train on a supported GPU")
    assert "Not a supported target, and not bitwise" in remedy and "torch_dtype: float16" in remedy
    card.check(_cfg(torch_dtype="float16"))  # the remedy runs


def test_from_pretrained_refuses_bf16_on_turing_before_any_download(monkeypatch):
    import poros.plan.configs
    from poros.train.trainer import PorosTrainer

    _card(monkeypatch, (7, 5), "Tesla T4", CU130)

    def no_network(*a, **k):
        raise AssertionError("asked the Hub before refusing")

    monkeypatch.setattr(poros.plan.configs, "read_model_config", no_network)
    with pytest.raises(PorosConfigError, match="Poros needs an NVIDIA GPU with bf16"):
        PorosTrainer.from_pretrained("Qwen/Qwen2.5-0.5B", dataset="synthetic")


@pytest.mark.parametrize("cap, limit", [((7, 0), 98304), ((7, 5), 65536), ((8, 0), 166912),
                                        ((8, 6), 101376), ((8, 9), 101376), ((9, 0), 232448),
                                        ((10, 0), 232448), ((12, 0), 101376),
                                        ((12, 1), 101376), ((8, 8), 101376)])
def test_shared_memory_limits(cap, limit):
    assert card.shared_memory_limit(cap) == limit


@pytest.mark.parametrize("cap, cls", [((6, 1), "pre-Volta"), ((7, 0), "Volta"),
                                      ((7, 5), "Turing"), ((8, 6), "Ampere"), ((8, 9), "Ada"),
                                      ((9, 0), "Hopper"), ((10, 0), "Blackwell"),
                                      ((12, 0), "Blackwell")])
def test_class_names(cap, cls):
    assert card.class_name(cap) == cls


# -- Triton: the fused compute's kernels ---------------------------------------------------------

@pytest.mark.parametrize("cap, name, arch_list", CARDS)
def test_fused_compute_by_card(monkeypatch, cap, name, arch_list):
    """fused.available: Volta and newer run the Triton kernels; pre-Volta keeps PEFT's modules,
    the reason in words (RunCompute records it as compute_reason)."""
    import sys
    import types

    from poros.core import fused

    _card(monkeypatch, cap, name, arch_list)
    for mod in ("triton", "bitsandbytes", "peft"):  # installed, as on a CUDA host
        monkeypatch.setitem(sys.modules, mod, sys.modules.get(mod) or types.ModuleType(mod))
    why = fused.available()
    if cap >= (7, 0):
        assert why is None
    else:
        assert why is not None and "Triton" in why and "sm_61" in why


def test_moe_expert_compute_takes_the_reference_where_triton_cannot_compile():
    cfg = _cfg()
    why = card.triton_reason((6, 1))
    c = choose(cfg, entry="moe", cuda=True, triton=True, gpu_reason=why)
    assert c.moe_expert == "reference" and not c.fused and "sm_61" in c.reason
    c = choose(cfg, entry="moe", cuda=True, triton=True, gpu_reason=None)
    assert c.moe_expert == "auto" and c.fused
    # fp16 (the remedy on Turing): the fused experts are bf16's, the reference computes fp16
    c = choose(_cfg(torch_dtype="float16"), entry="moe", cuda=True, triton=True)
    assert c.moe_expert == "reference" and "float16" in c.reason


def test_dense_choice_records_the_card_reason():
    c = choose(_cfg(), entry="dense", cuda=True, triton=True, gpu_reason=card.triton_reason((6, 1)))
    assert not c.fused and "sm_61" in c.reason
    assert choose(_cfg(), entry="dense", cuda=True, triton=True).fused


# -- the fused kernels' shared memory, by LoRA rank ----------------------------------------------

def _peft_4bit(rank):
    """A tiny Qwen2 with PEFT LoRA over bitsandbytes Linear4bit layers (CPU: only the routing is
    checked, nothing is quantized or run)."""
    bnb = pytest.importorskip("bitsandbytes")
    peft = pytest.importorskip("peft")
    transformers = pytest.importorskip("transformers")
    cfg = transformers.Qwen2Config(vocab_size=64, hidden_size=128, intermediate_size=256,
                                   num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2)
    model = transformers.Qwen2ForCausalLM(cfg)
    names = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    for parent in list(model.modules()):
        for n in names:
            lin = getattr(parent, n, None)
            if isinstance(lin, torch.nn.Linear):
                setattr(parent, n, bnb.nn.Linear4bit(lin.in_features, lin.out_features,
                                                     bias=lin.bias is not None,
                                                     compute_dtype=torch.bfloat16))
    model.is_loaded_in_4bit = True
    model = peft.get_peft_model(model, peft.LoraConfig(r=rank, lora_alpha=2 * rank,
                                                       target_modules=list(names)))
    for m in model.modules():
        if hasattr(m, "lora_A"):
            assert type(m).__module__.endswith("bnb"), type(m)
    return model


@pytest.mark.parametrize("rank, fits", [(16, True), (32, True), (64, False), (128, False)])
def test_a_rank_whose_kernels_do_not_fit_keeps_peft_and_says_why(monkeypatch, rank, fits):
    """fused.install asks fit.reason per padded rank (on a GPU: the kernels launched once);
    a rank that does not fit keeps PEFT's modules everywhere, the reason in refusals()."""
    from poros.core import fused
    from poros.core.fused import fit

    model = _peft_4bit(rank)
    asked = []

    def reason(rp, dtype, head_dim=None):
        asked.append((rp, dtype, head_dim))
        return None if rp <= 32 else f"LoRA rank {rp} (padded): too much shared memory"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fit, "reason", reason)
    try:
        counts = fused.install(model)
        assert asked and {a[0] for a in asked} == {max(16, rank)}
        assert {a[1] for a in asked} == {torch.bfloat16}
        if fits:
            assert counts["swiglu_mlps"] == 2 and counts["linears"] == 14
            assert fused.refusals(model) == []
        else:
            assert counts == {"layers": 0, "linears": 0, "qkv_groups": 0, "swiglu_mlps": 0}
            assert fused.refusals(model) == [f"LoRA rank {rank} (padded): too much shared memory"]
            assert all("_poros_peft_forward" not in vars(m) for m in model.modules())
    finally:
        fused.uninstall(model)


@pytest.mark.parametrize("rank, rp", [(16, 16), (64, 64), (100, 128)])
def test_a_moe_rank_whose_expert_kernels_do_not_fit_takes_the_reference(monkeypatch, rank, rp):
    """block_reason asks fit_reason for the adapter's padded rank on a GPU; a rank that does not
    fit is a reason, so prepare_moe's "auto" takes the reference compute and says why."""
    from types import SimpleNamespace

    from poros.moe import fused as mf

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(mf, "fit_reason",
                        lambda p: None if p <= 64 else f"LoRA rank {p} (padded): too large")
    block = SimpleNamespace(
        act_fn=torch.nn.SiLU(), store=SimpleNamespace(compute_dtype=torch.bfloat16),
        expert_group=None,
        adapter=SimpleNamespace(rank=rank, lora_A_gate_up=torch.zeros(1, dtype=torch.bfloat16)))
    why = mf.block_reason(block)
    assert (why is None) is (rp <= 64)
    if why:
        assert "rank 128" in why


def test_an_unpinned_grouped_moe_compute_is_refused_before_the_load(monkeypatch):
    """moe_expert_compute: grouped is pinned per host: an unpinned host is refused when the
    compute is requested, before load_moe_streamed downloads anything."""
    from types import SimpleNamespace

    from poros.moe import grouped
    from poros.train.compute import RunCompute

    _card(monkeypatch, (8, 9), "NVIDIA L4", CU130)

    def unpinned():
        raise grouped.GroupedPinError("no grouped-expert pin for this host")

    monkeypatch.setattr(grouped, "activate", unpinned)
    t = SimpleNamespace(config=_cfg(moe_expert_compute="grouped"),
                        device=torch.device("cuda", 0))
    with pytest.raises(PorosConfigError, match="grouped cannot run on this host"):
        RunCompute(t).moe_request()
