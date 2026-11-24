"""Every dense family the registry calls validated, through the HF patch (``poros.patch.hf``, the
MoE path's dense streaming), on CPU: each family's real model class, wrapper and text backbone,
against its own resident run.
"""

import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM  # noqa: E402

from poros.core.arch_registry import registry_type, validated_arch_types  # noqa: E402
from poros.patch.hf import patch_hf_model  # noqa: E402
from tests.metal import families as fam  # noqa: E402

VOCAB = 64
FAMILIES = ["qwen2", "qwen3", "qwen3_5", "qwen3_5_text", "gemma3", "gemma3_text", "gemma4",
            "gemma4_text"]
TEXT_LORA = r".*(language_model|model\.layers).*\.(q_proj|v_proj)"


def _model(family):
    cfg = fam.config(family, vocab=VOCAB)
    torch.manual_seed(5)
    cls = fam._WRAPPERS.get(family)
    base = cls(cfg) if cls else AutoModelForCausalLM.from_config(cfg)
    return get_peft_model(base, LoraConfig(r=4, target_modules=TEXT_LORA))


def _run(model, family):
    ids = torch.randint(0, VOCAB, (2, 10), generator=torch.Generator().manual_seed(9))
    batch = {"input_ids": ids, "labels": ids}
    if family in ("gemma3", "gemma4"):  # the wrappers ask for these in training
        batch["token_type_ids" if family == "gemma3" else "mm_token_type_ids"] = ids * 0
    model.zero_grad(set_to_none=True)
    out = model(**batch)
    out.loss.backward()
    return out.loss.detach(), {
        n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None
    }


def test_this_covers_every_dense_family():
    assert {registry_type(fam.config(f, vocab=VOCAB)) for f in FAMILIES} == validated_arch_types()


@pytest.mark.usefixtures("cpu_only")
@pytest.mark.parametrize("block_size", [1, 3])
@pytest.mark.parametrize("family", FAMILIES)
def test_streamed_runs_and_matches_resident(family, block_size):
    model = _model(family)
    resident = _run(model, family)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        patch_hf_model(model, block_size=block_size, deterministic=False)
    streamed = _run(model, family)
    assert torch.equal(resident[0], streamed[0]), (resident[0].item(), streamed[0].item())
    assert resident[1].keys() == streamed[1].keys() and resident[1]
    for name in resident[1]:
        assert torch.equal(resident[1][name], streamed[1][name]), name
