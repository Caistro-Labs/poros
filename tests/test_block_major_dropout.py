"""Block-major accumulation draws the sequential loop's dropout masks (poros.core.sequential_rng).

On the CPU, where the generator is the CPU's and every draw is replayed: rows of uneven length
padded within each micro-batch, LoRA dropout on every projection, the classic loop and
``poros.Run`` (rows and preference pairs), resumed or not. The same update, bit for bit, as the
sequential loop; and a draw the walk does not know about stops the run instead of training with
other masks. tests/test_run_block_major_cuda.py and tests/gpu/ take the CUDA generator.
"""

from __future__ import annotations

import pytest

from tests.conftest import Tok

ALL = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@pytest.fixture
def torch():
    t = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    return t


def _padded(torch, vocab, *, n=24, rows=2, seed=0):
    """Rows of 5 to 16 tokens, ``rows`` to a micro-batch, each padded to its longest."""
    g = torch.Generator().manual_seed(seed)
    seqs = [torch.randint(3, vocab, (int(k),), generator=g)
            for k in torch.randint(5, 17, (n,), generator=g)]

    def collate(items):
        width = max(len(x) for x in items)
        ids = torch.zeros(len(items), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, x in enumerate(items):
            ids[i, :len(x)], mask[i, :len(x)] = x, 1
        return {"input_ids": ids, "attention_mask": mask,
                "labels": ids.masked_fill(mask == 0, -100)}

    return torch.utils.data.DataLoader(seqs, batch_size=rows, shuffle=False, collate_fn=collate)


def _tiny(torch, layers, dropout):
    """_tiny_peft_qwen2 with ``layers`` decoder layers and LoRA on every projection."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=64)
    return get_peft_model(Qwen2ForCausalLM(cfg), LoraConfig(
        r=8, lora_alpha=16, lora_dropout=dropout, target_modules=list(ALL), bias="none",
        task_type="CAUSAL_LM"))


def _trainer(torch, outdir, *, block_major, dropout=0.1, G=4, model=None, block_size=2,
             layers=None):
    from poros.core.config import PorosConfig
    from poros.train.trainer import PorosTrainer
    from tests.test_train_loop_cpu import _tiny_peft_qwen2

    torch.manual_seed(0)
    if model is None and layers is not None:
        model = _tiny(torch, layers, dropout)
    if model is None:
        model, _ = _tiny_peft_qwen2(torch, targets=ALL, dropout=dropout)
    cfg = PorosConfig(
        model_name_or_path="dummy", block_size=block_size,
        deterministic=True,
        output_dir=str(outdir), blockwise_arch_check="off", enable_prefetch=False,
        per_device_batch_size=2, gradient_accumulation_steps=G, max_steps=6, warmup_steps=0,
        save_steps=0, eval_steps=0, logging_steps=1, progress=False,
        lora_dropout=dropout, path="fast" if block_major else "sequential",
    )
    tr = PorosTrainer(cfg)
    tr.model = model
    tr.device = torch.device("cpu")
    tr._adapter_backend = "peft"
    tr._setup_blockwise_training()
    return tr


def _state(torch, tr):
    return ({n: p.detach().clone() for n, p in tr.model.named_parameters() if p.requires_grad},
            list(tr.history.losses), list(tr.history.grad_norms), torch.get_rng_state())


def _same(torch, a, b):
    assert a[1] == b[1]  # every step's loss
    assert a[2] == b[2]  # every step's grad norm
    assert a[0].keys() == b[0].keys()
    for name in a[0]:
        assert torch.equal(a[0][name], b[0][name]), name
    assert torch.equal(a[3], b[3])  # the generator, where the next update starts


def _trained(torch, outdir, *, block_major, dropout=0.1, steps=6, G=4):
    """Six updates of four micro-batches over 20 rows (10 micro-batches an epoch): a new epoch,
    and the loader's draw for it, starts inside the third and the fifth update."""
    tr = _trainer(torch, outdir, block_major=block_major, dropout=dropout, G=G)
    tr.train(_padded(torch, 256, n=20), steps=steps, seq_len=16)
    return tr


@pytest.mark.parametrize("dropout", [0.0, 0.1])
def test_block_major_is_the_sequential_update_with_dropout(torch, tmp_path, dropout):
    seq = _trained(torch, tmp_path / "s", block_major=False, dropout=dropout)
    bm = _trained(torch, tmp_path / "b", block_major=True, dropout=dropout)
    _same(torch, _state(torch, seq), _state(torch, bm))


def test_one_micro_batch_a_step_runs_block_major_as_the_sequential_update(torch, tmp_path):
    """At one micro-batch a step block-major runs too (no slower there), the same update."""
    seq = _trained(torch, tmp_path / "s", block_major=False, G=1, steps=12)
    bm = _trained(torch, tmp_path / "b", block_major=True, G=1, steps=12)
    assert (seq.ran_path, bm.ran_path) == ("sequential", "fast")
    _same(torch, _state(torch, seq), _state(torch, bm))


@pytest.mark.parametrize("block_size", [1, 2, 4, 8])
def test_every_block_size_is_the_sequential_update(torch, tmp_path, block_size):
    """Eight layers at block size 1, 2, 4 and 8 (one block): block-major and the sequential loop
    at that size train the sequential loop's update at block size 4, bit for bit."""
    def trained(name, block_major, size):
        tr = _trainer(torch, tmp_path / name, block_major=block_major, block_size=size, layers=8)
        tr.train(_padded(torch, 256, n=20), steps=4, seq_len=16)
        return _state(torch, tr)

    reference = trained("ref", False, 4)
    _same(torch, reference, trained("seq", False, block_size))
    _same(torch, reference, trained("bm", True, block_size))


def _mask_digests(torch, model):
    """Every active dropout's mask, a digest per call, by module (``(digests, hooks)``)."""
    import hashlib

    got, hooks = {}, []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Dropout) and m.p > 0:
            def hook(mod, inputs, out, name=name):
                kept = ((out != 0) | (inputs[0] == 0)).cpu().numpy().tobytes()
                got.setdefault(name, []).append(hashlib.sha256(kept).hexdigest())
            hooks.append(m.register_forward_hook(hook))
    return got, hooks


@pytest.mark.parametrize("block_size", [1, 2, 4])
def test_every_recompute_draws_its_forwards_masks(torch, tmp_path, block_size):
    """At dropout 0.1, every mask a block's backward recompute draws is one its forward drew, on
    either loop, and both loops draw the same masks: one per micro-batch and LoRA dropout."""
    steps, G = 3, 4
    masks = {}
    for name, bm in (("seq", False), ("bm", True)):
        tr = _trainer(torch, tmp_path / name, block_major=bm, block_size=block_size)
        got, hooks = _mask_digests(torch, tr.model)
        tr.train(_padded(torch, 256, n=24), steps=steps, seq_len=16)
        for h in hooks:
            h.remove()
        assert len(got) == 4 * len(ALL)  # every LoRA-wrapped linear's dropout drew
        for module, calls in got.items():
            seen = {d: calls.count(d) for d in calls}
            assert len(seen) == steps * G, module  # a mask a micro-batch
            assert min(seen.values()) >= 2, module  # each drawn again by its recompute
        masks[name] = {module: set(calls) for module, calls in got.items()}
    assert masks["seq"] == masks["bm"]


def test_without_the_walk_the_masks_differ(torch, tmp_path, monkeypatch):
    """The control: block-major order drawing from the live generator trains other masks."""
    from poros.core.sequential_rng import SequentialRng

    seq = _state(torch, _trained(torch, tmp_path / "s", block_major=False))
    monkeypatch.setattr(SequentialRng, "set", lambda self, g, b: None)
    monkeypatch.setattr(SequentialRng, "check", lambda self, g, b, what: None)
    bm = _state(torch, _trained(torch, tmp_path / "b", block_major=True))
    assert seq[1] != bm[1]


@pytest.mark.parametrize("rows", [24, 20], ids=["at-an-epoch-start", "mid-epoch"])
@pytest.mark.parametrize("first, then", [(True, True), (False, True), (True, False)],
                         ids=["block-major", "sequential-then-block-major",
                              "block-major-then-sequential"])
def test_resuming_is_the_uninterrupted_sequential_run(torch, tmp_path, first, then, rows):
    """Stopped after its third update and resumed, on either loop: the uninterrupted sequential
    run's steps, adapters and generator."""
    from poros.core.errors import PorosCancelledError

    def data():
        return _padded(torch, 256, n=rows)

    whole = _trainer(torch, tmp_path / "whole", block_major=False)
    whole.train(data(), steps=6, seq_len=16)
    whole = _state(torch, whole)
    part = _trainer(torch, tmp_path / "part", block_major=first)
    with pytest.raises(PorosCancelledError) as stopped:
        part.train(data(), steps=6, seq_len=16, should_stop=lambda: part.global_step == 2)
    resumed = _trainer(torch, tmp_path / "part", block_major=then)
    resumed.train(data(), steps=6, seq_len=16, resume_from=str(stopped.value.checkpoint))
    got = _state(torch, resumed)
    assert got[1] == whole[1][3:] and got[2] == whole[2][3:]
    for name in whole[0]:
        assert torch.equal(got[0][name], whole[0][name]), name
    assert torch.equal(got[3], whole[3])


def _run(torch, outdir, *, block_major, dropout=0.1):
    from poros.train.run import Run

    tr = _trainer(torch, outdir, block_major=block_major, dropout=dropout)
    tr.config.per_device_batch_size = 1
    tr.config.learning_rate = 1e-2
    return Run.from_trainer(tr, Tok())


@pytest.mark.parametrize("shape", ["chats", "pairs"])
def test_poros_run_block_major_is_its_sequential_loop_with_dropout(torch, tmp_path, shape):
    """Each slice of a batch is a micro-batch; preference pairs too (a slice holds a pair)."""
    from poros import data as pdata
    from poros.recipes import dpo, sft

    if shape == "chats":
        rows = [{"messages": [{"role": "user", "content": f"say {i}" * (1 + i % 3)},
                              {"role": "assistant", "content": f"{i} " * (2 + i % 4)}]}
                for i in range(8)]
        data = pdata.chats(rows, Tok(), batch_size=4)

        def train(run):
            sft.main(run, data, steps=4)
    else:
        rows = [{"prompt": f"pick {i}" * (1 + i % 2), "chosen": f"yes {i}" * (1 + i % 3),
                 "rejected": f"no no {i}"} for i in range(8)]
        data = pdata.pairs(rows, Tok(), batch_size=4)

        def train(run):
            dpo.main(run, data, steps=4, beta=0.5)

    out = []
    for name, bm in (("seq", False), ("bm", True)):
        run = _run(torch, tmp_path / name, block_major=bm)
        train(run)
        # the loop said is the loop that ran, and the record says so
        assert run._engine.ran_path == ("fast" if bm else "sequential")
        out.append(_state(torch, run._engine))
    _same(torch, *out)


def _two(torch, tmp_path, made):
    """The sequential run and the block-major one of the model ``made()`` builds."""
    out = []
    for name, bm in (("seq", False), ("bm", True)):
        tr = _trainer(torch, tmp_path / name, block_major=bm, model=made())
        tr.train(_padded(torch, 256), steps=3, seq_len=16)
        out.append((tr, _state(torch, tr)))
    return out


def test_a_draw_outside_the_lora_dropouts_trains_sequentially(torch, tmp_path, caplog):
    """The model's own dropout (here attention dropout) draws in the forward: the check after the
    first block stops the update before anything is applied, and it and the rest of the run
    train sequentially, the sequential run's update, with a warning."""
    from tests.test_train_loop_cpu import _tiny_peft_qwen2

    def made():
        torch.manual_seed(0)
        model, _ = _tiny_peft_qwen2(torch, targets=ALL, dropout=0.1)
        model.base_model.model.config.attention_dropout = 0.2
        for layer in model.base_model.model.model.layers:
            layer.self_attn.attention_dropout = 0.2
        return model

    with caplog.at_level("WARNING", logger="poros.train.trainer"):
        (seq, a), (bm, b) = _two(torch, tmp_path, made)
    _same(torch, a, b)
    assert "block 0 of micro-batch 0 drew" in caplog.text and "train sequentially" in caplog.text
    assert bm._block_major_off and bm.ran_path == seq.ran_path == "sequential"


def test_a_dropout_module_the_walk_cannot_size_trains_sequentially(torch, tmp_path, caplog):
    from tests.test_train_loop_cpu import _tiny_peft_qwen2

    def made():
        torch.manual_seed(0)
        model, _ = _tiny_peft_qwen2(torch, targets=ALL, dropout=0.1)
        model.base_model.model.model.layers[1].mlp.extra = torch.nn.Dropout(0.1)
        return model

    with caplog.at_level("WARNING", logger="poros.train.trainer"):
        (_, a), (bm, b) = _two(torch, tmp_path, made)
    _same(torch, a, b)
    assert "dropout outside the LoRA adapters" in caplog.text
    assert bm.ran_path == "sequential"


def test_the_walk_records_where_each_block_starts(torch):
    """Each (micro-batch, block) state is the generator after every earlier draw, and the walk
    leaves the generator where the sequential forwards would."""
    import torch.nn.functional as F

    from poros.core.sequential_rng import SequentialRng

    draws = [[(0.1, 8, torch.float32)], [], [(0.2, 4, torch.float32), (0.1, 8, torch.float32)]]
    torch.manual_seed(3)
    walk = SequentialRng(torch.device("cpu"), draws, {})
    for tokens in (5, 2):
        walk.add(tokens)
    end = torch.get_rng_state()

    torch.manual_seed(3)
    for g, tokens in enumerate((5, 2)):
        for b, block in enumerate(draws):
            assert torch.equal(torch.get_rng_state(), walk.states[g][b][0])
            for p, width, dtype in block:
                F.dropout(torch.ones(tokens * width, dtype=dtype), p, training=True)
        assert torch.equal(torch.get_rng_state(), walk.states[g][len(draws)][0])
    assert torch.equal(torch.get_rng_state(), end)


def _vlm(torch, family, dropout=0.1):
    """A tiny Gemma VLM (``*ForConditionalGeneration``, its vision tower included) under the
    NF4 default target "all-linear": LoRA on the tower's linears too."""
    from transformers import AutoModelForCausalLM

    from poros.train.model import _attach_peft_lora
    from tests.metal.families import config

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config(family, vocab=256))
    _, peft = _attach_peft_lora(model, lora_rank=8, lora_alpha=16, lora_dropout=dropout,
                                lora_bias="none", target_modules="all-linear")
    return peft


@pytest.mark.parametrize("family", ["gemma3", "gemma4"])
def test_a_vlm_s_tower_adapters_leave_block_major_the_sequential_update(torch, tmp_path, family):
    """The vision tower's adapters sit outside the decoder blocks, and a text-only forward never
    runs them: counted out (ArchSpec.text_unreached), block-major trains the sequential update."""
    out = []
    for name, bm in (("seq", False), ("bm", True)):
        tr = _trainer(torch, tmp_path / name, block_major=bm, model=_vlm(torch, family))
        assert tr._unreached_params and all("vision" in n for n, _ in tr._unreached_params)
        tr.train(_padded(torch, 256, n=20), steps=3, seq_len=16)
        assert not tr._unreached_params  # checked after the first update: no gradient
        out.append(_state(torch, tr))
    _same(torch, *out)


@pytest.mark.parametrize("block_major", [False, True])
def test_a_tower_adapter_that_gets_a_gradient_stops_the_run(torch, tmp_path, block_major):
    """Fail closed: if a submodule counted out of the blocks did get a gradient, the first update
    stops before it is applied, on either loop."""
    from poros.core.errors import PorosConfigError

    tr = _trainer(torch, tmp_path, block_major=block_major, model=_vlm(torch, "gemma3"))
    name, tower = tr._unreached_params[0]
    before = {n: p.detach().clone() for n, p in tr.model.named_parameters() if p.requires_grad}
    real = tr._adapter_optimizer_step

    def reached():
        tower.grad = torch.zeros_like(tower)  # as if the forward had run the tower
        real()

    tr._adapter_optimizer_step = reached
    with pytest.raises(PorosConfigError, match="received a gradient"):
        tr.train(_padded(torch, 256, n=20), steps=2, seq_len=16)
    for n, p in tr.model.named_parameters():
        if p.requires_grad:
            assert torch.equal(p, before[n]), n
