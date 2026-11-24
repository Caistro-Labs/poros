"""Poros's Inkling 2 Mini decoder against the official implementation (transformers >= 5.24).

Poros pins a transformers without Inkling 2 Mini and carries its own decoder
(poros.moe.inkling2.modeling), so it is held to the official one here, in a child process
(tests/moe/inkling2_oracle.py) that imports transformers from ``POROS_INKLING_ORACLE``:

    pip install --target "$DIR" --no-deps transformers==5.24.0
    POROS_INKLING_ORACLE="$DIR" pytest tests/moe/test_inkling2_oracle.py

- the decoder alone, every weight random: the loss, the logits and the gradient of every one of
  its parameters, fp32, with a loop of one residual block and two loops of two (as the release's
  is); right padding the same; in bf16 against the official class's own bf16 rounding;
- a run's merged export, loaded by the official class's ``from_pretrained``: the trained logits;
- Poros's whole training path (NF4 experts, managed expert compute, expert and attention
  adapters, the visits) against the official model holding the same NF4-rounded experts with
  the same adapters written out by definition: the loss and every adapter gradient;
- with ``POROS_INKLING_TINY_CHECKPOINT`` (a download of hf-internal-testing/tiny-inkling2-mini
  at ee242a48af34ff0729518f4ac28cfcb847e470ab, sliced from the released checkpoint with its
  NVFP4 and MXFP8 experts): Poros's conversion of the released format gives the very tensors
  transformers loads, and the same logits and gradients.

Tolerances are relative, per tensor (norm of the difference over the norm of the official one).
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from tests.moe import require_tested_transformers  # noqa: E402

require_tested_transformers()

from tests.moe import inkling2_tiny as tiny  # noqa: E402

ORACLE = os.environ.get("POROS_INKLING_ORACLE", "")
TINY_CHECKPOINT = os.environ.get("POROS_INKLING_TINY_CHECKPOINT", "")
pytestmark = pytest.mark.skipif(
    not (ORACLE and (Path(ORACLE) / "transformers" / "models" / "inkling2_loop").is_dir()),
    reason="POROS_INKLING_ORACLE names no transformers with Inkling 2 Mini (>= 5.24)",
)

#: Logits and gradients, official against Poros, fp32 (measured: logits 4e-7 to 5e-7, the worst
#: gradient 1.5e-6 to 2.9e-6, on a CPU).
RTOL = 1e-5


def _oracle(job: dict, tmp_path: Path) -> dict:
    job_path, out_path = tmp_path / "job.pt", tmp_path / "out.pt"
    torch.save(job, job_path)
    env = dict(os.environ, PYTHONPATH=ORACLE)
    child = Path(__file__).with_name("inkling2_oracle.py")
    run = subprocess.run(
        [sys.executable, str(child), str(job_path), str(out_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    out = torch.load(out_path, weights_only=False)
    out_path.unlink()  # its pages leave the cache with it (the tests run under a memory cap)
    job_path.unlink()
    assert tuple(int(x) for x in out["transformers"].split(".")[:2]) >= (5, 24)
    return out


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-30)).item()


@pytest.mark.parametrize("shape", list(tiny.SHAPES))
def test_the_decoder_matches_the_official_one(tmp_path, shape):
    over = tiny.SHAPES[shape]
    m = tiny.model(torch.float32, **over)
    ids = tiny.ids()
    ref = _oracle(
        dict(config={**tiny.CONFIG, **over}, state=m.state_dict(), input_ids=ids, labels=ids),
        tmp_path,
    )
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    assert _rel(out.loss, ref["loss"]) < RTOL
    assert _rel(out.logits, ref["logits"]) < RTOL
    params = dict(m.named_parameters())
    assert ref["grads"].keys() == params.keys()
    worst = max((_rel(p.grad, ref["grads"][n]), n) for n, p in params.items())
    assert worst[0] < RTOL, worst


#: bf16. Both decoders in bf16 are far from the fp32 truth on these random weights (logits 17%
#: to 34%, gradients up to 150%: a deep random stack amplifies every rounding), so a fixed
#: tolerance against fp32 says nothing. The honest bar is relative to the official class's own
#: rounding, per tensor (the logits and every gradient), measured on a CPU over the three shapes:
#: - Poros in bf16 is no further from the fp32 official result than the official class in bf16
#:   is: at worst 1.020 times as far (a relative position bias norm's scale); bound 1.05;
#: - Poros in bf16 is far nearer the official bf16 result than that is to fp32: at worst 0.082
#:   of the official class's own distance (a sink projection), 2.7e-2 in absolute terms; bound
#:   0.12, about 1.5 times the worst measured.
#: A tensor whose gradient is zero in both (a message query where the loop is one block) passes
#: with both distances zero. The two round in a few different places (Poros keeps the official
#: fp32 modules in fp32, and rounds its fused head and its expert sum its own way).
BF16_NO_WORSE = 1.05
BF16_NEARER = 0.12


@pytest.mark.parametrize("shape", list(tiny.SHAPES))
def test_the_decoder_in_bf16_rounds_as_the_official_one_does(tmp_path, shape, cpu_bf16_backward):
    from poros.moe.inkling2.modeling import cast

    over = tiny.SHAPES[shape]
    config = {**tiny.CONFIG, **over}
    state = tiny.model(torch.float32, **over).state_dict()
    ids = tiny.ids()
    truth = _oracle(dict(config=config, state=state, input_ids=ids, labels=ids), tmp_path)
    official = _oracle(
        dict(config=config, state=state, input_ids=ids, labels=ids, dtype="bfloat16"), tmp_path
    )
    m = tiny.model(torch.float32, **over)
    m.load_state_dict(state)
    cast(m, torch.bfloat16)
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    mine = {"logits": out.logits, **{n: p.grad for n, p in m.named_parameters()}}
    theirs = {"logits": official["logits"], **official["grads"]}
    fp32 = {"logits": truth["logits"], **truth["grads"]}
    assert mine.keys() == theirs.keys() == fp32.keys()
    for name in mine:
        own, ref = _rel(mine[name], fp32[name]), _rel(theirs[name], fp32[name])
        assert own <= BF16_NO_WORSE * ref, (name, own, ref)
        assert _rel(mine[name], theirs[name]) <= BF16_NEARER * ref, (name, ref)


@pytest.mark.parametrize("shape", list(tiny.SHAPES))
def test_right_padding_matches_the_official_one(tmp_path, shape):
    """A batch whose second row ends in four pad tokens (attention mask 0, labels -100): the loss,
    the logits of every real token and every gradient, fp32."""
    over = tiny.SHAPES[shape]
    m = tiny.model(torch.float32, **over)
    ids = tiny.ids()
    mask = torch.ones_like(ids)
    mask[1, -4:] = 0
    labels = ids.masked_fill(mask == 0, -100)
    ref = _oracle(
        dict(
            config={**tiny.CONFIG, **over},
            state=m.state_dict(),
            input_ids=ids,
            labels=labels,
            attention_mask=mask,
        ),
        tmp_path,
    )
    out = m(input_ids=ids, labels=labels, attention_mask=mask)
    out.loss.backward()
    real = mask.bool()
    assert _rel(out.loss, ref["loss"]) < RTOL
    assert _rel(out.logits[real], ref["logits"][real]) < RTOL
    worst = max((_rel(p.grad, ref["grads"][n]), n) for n, p in m.named_parameters())
    assert worst[0] < RTOL, worst


@pytest.mark.parametrize("held", ["released", "nf4"])
@pytest.mark.parametrize("shape", ["one_block", "two_blocks_8"])
def test_the_official_class_loads_a_merged_export(tmp_path, shape, held):
    """A run's merged export (poros.moe.inkling2.export, from a released-format base with NVFP4
    and MXFP8 experts, its attention and expert adapters drawn) loads in the official class
    through ``from_pretrained``, and computes the trained model's logits: with the experts held
    as released, and as nf4 (the merge then rounds them through NF4 as the run held them)."""
    from poros.moe.inkling2.export import export_merged
    from poros.moe.load import load_moe_streamed
    from poros.moe.prepare import MoeConfig, prepare_moe

    base = tmp_path / "base"
    base.mkdir()
    tiny.write_released(base, torch.float32, **tiny.SHAPES[shape])
    model = load_moe_streamed(str(base), torch.float32, expert_format=held)
    assert model._poros_expert_format == held
    prepare_moe(model, MoeConfig(streamed=False, look_ahead=0))
    tiny.randomize_lora_b(model)
    adapter = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    a = next(m for m in model.modules() if hasattr(m, "lora_A_gate_up"))
    out = tmp_path / "merged"
    export_merged(
        str(base),
        adapter,
        str(out),
        scaling=a.alpha / a.rank,
        dtype=torch.float32,
        expert_codec=held,
    )
    ids = tiny.ids()
    ref = _oracle(dict(checkpoint=str(out), input_ids=ids, labels=ids), tmp_path)
    model.eval()
    with torch.no_grad():
        want = model(input_ids=ids, labels=ids)
    assert _rel(ref["logits"], want.logits) < 1e-4, _rel(ref["logits"], want.logits)
    assert _rel(ref["loss"], want.loss) < 1e-4


def test_poros_training_matches_the_official_one_with_its_adapters(tmp_path):
    from poros.moe.arch import MoeArch
    from poros.moe.inkling2.visits import install_visit_streaming
    from poros.moe.prepare import dense_managed_filter, prepare_moe, streaming_config

    m = tiny.model(torch.float32)
    base = copy.deepcopy(m)
    prepare_moe(m, streaming_config(m.config, lora_dropout=0.0))
    tiny.randomize_lora_b(m)
    m.get_decoder().poros_visit_runner = None  # on the CPU: stream the dense weights here
    install_visit_streaming(m, 2, dense_managed_filter(MoeArch.from_config(m.config)))

    state = base.state_dict()
    lora = {"rank": 16, "alpha": 32, "targets": [], "A": {}, "B": {}, "experts": {}}
    for i, layer in enumerate(m.model.layers):
        block = layer.mlp
        block.residency.ensure([(i, e) for e in range(block.num_experts)])
        for tensor in ("gate_up_proj", "down_proj"):  # the NF4-rounded experts Poros trains on
            state[f"model.layers.{i}.mlp.experts.{tensor}"] = torch.stack(
                [
                    block.store.dequantize_tensor(
                        (i, e), tensor, block.residency.resident_qbytes((i, e))
                    )
                    for e in range(block.num_experts)
                ]
            )
        a = block.adapter
        lora["rank"], lora["alpha"] = a.rank, a.alpha
        lora["experts"][i] = {
            "A_gate_up": a.lora_A_gate_up.detach(),
            "B_gate_up": a.lora_B_gate_up.detach(),
            "A_down": a.lora_A_down.detach(),
            "B_down": a.lora_B_down.detach(),
        }
    for name, module in m.named_modules():
        if hasattr(module, "lora_A") and hasattr(module, "base_layer"):
            lora["targets"].append(name)
            lora["A"][name], lora["B"][name] = module.lora_A.detach(), module.lora_B.detach()
    assert len(lora["targets"]) == 4 * len(m.model.layers)

    ids = tiny.ids()
    ref = _oracle(
        dict(config=tiny.CONFIG, state=state, input_ids=ids, labels=ids, lora=lora), tmp_path
    )
    m.train()
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    assert _rel(out.loss, ref["loss"]) < RTOL
    mine = {}
    for name, param in m.named_parameters():
        if not param.requires_grad:
            continue
        key = name
        if (
            ".mlp.adapter." in name
        ):  # the oracle's expert adapter: mlp.experts.poros_lora.<A|B>_<proj>
            prefix, _, leaf = name.partition(".mlp.adapter.")
            key = f"{prefix}.mlp.experts.poros_lora.{leaf.removeprefix('lora_')}"
        mine[key] = param.grad
    assert mine.keys() == ref["grads"].keys()
    worst = max((_rel(g, ref["grads"][n]), n) for n, g in mine.items())
    assert worst[0] < RTOL, worst


@pytest.mark.skipif(not TINY_CHECKPOINT, reason="POROS_INKLING_TINY_CHECKPOINT not set")
def test_the_released_format_converts_as_transformers_loads_it(tmp_path):
    from safetensors import safe_open

    from poros.moe.inkling2.codec import convert_state_dict
    from poros.moe.inkling2.config import text_config_of
    from poros.moe.inkling2.modeling import Inkling2LoopForCausalLM

    config = text_config_of(json.loads((Path(TINY_CHECKPOINT) / "config.json").read_text()))
    ids = torch.randint(0, 2000, (1, 40), generator=torch.Generator().manual_seed(1))
    ref = _oracle(
        dict(checkpoint=TINY_CHECKPOINT, input_ids=ids, labels=ids, return_state=True), tmp_path
    )
    with safe_open(str(Path(TINY_CHECKPOINT) / "model.safetensors"), "pt") as f:
        state = convert_state_dict(
            f.get_tensor,
            list(f.keys()),
            unpadded_vocab_size=config.unpadded_vocab_size,
            dtype=torch.float32,
            expert_dtype=torch.float32,
        )
    state = {k: v.float() for k, v in state.items()}
    assert state.keys() == ref["state"].keys()
    assert all(torch.equal(state[k], ref["state"][k].float()) for k in state)  # decoded exactly
    # Held at most three copies of the weights at once (this test also runs on a
    # real-weight slice): the official state goes once compared, the converted one once loaded.
    del ref["state"]
    m = Inkling2LoopForCausalLM(config).float()
    m.load_state_dict(state)
    del state
    out = m(input_ids=ids, labels=ids)
    out.loss.backward()
    assert _rel(out.logits, ref["logits"]) < RTOL
    worst = max((_rel(p.grad, ref["grads"][n]), n) for n, p in m.named_parameters())
    assert worst[0] < RTOL, worst
