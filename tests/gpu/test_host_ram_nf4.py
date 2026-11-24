"""NF4 host RAM on CUDA: where the bytes live changes, no value does.

A tiny random Qwen2 checkpoint (bf16 safetensors). The streamed load carves pinned weights from
shared slabs and keeps resident blocks on the GPU, byte-identical to the plain streamed load; the
frozen tables either become fp32 on the GPU (PEFT's values) or stay bf16 in host RAM; and every
placement trains to the same losses and adapters, bit for bit, on both block-major paths and the
sequential loop.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.gpu

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("NF4 needs CUDA", allow_module_level=True)
pytest.importorskip("bitsandbytes")
transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    torch.manual_seed(0)
    # vocab past the synthetic dataset's token ids (< 30000): an id out of range is a
    # device-side assert, not a test failure
    cfg = transformers.Qwen2Config(vocab_size=32000, hidden_size=128, intermediate_size=320,
                                   num_hidden_layers=6, num_attention_heads=4,
                                   num_key_value_heads=2, tie_word_embeddings=False)
    model = transformers.AutoModelForCausalLM.from_config(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0, 0.05)
    path = tmp_path_factory.mktemp("qwen2_tiny")
    model.to(torch.bfloat16).save_pretrained(path)
    return str(path)


def _quant_bytes(param):
    qs = param.quant_state.as_dict(packed=True)
    return {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in qs.items()}


def test_resident_layers_load_onto_the_gpu_byte_identical(checkpoint):
    import bitsandbytes as bnb

    from poros.train.streamed_load import load_nf4_block_streamed

    plain = load_nf4_block_streamed(checkpoint, torch.bfloat16)
    kept = load_nf4_block_streamed(checkpoint, torch.bfloat16, resident=(2, 1))  # layers 4, 5
    ref, got = dict(plain.named_parameters()), dict(kept.named_parameters())
    assert sorted(ref) == sorted(got)
    on_gpu = set()
    storages, pinned = set(), 0
    for name, a in ref.items():
        b = got[name]
        assert torch.equal(a.detach().cpu(), b.detach().cpu()), name
        if isinstance(a, bnb.nn.Params4bit):
            if a.data.is_pinned():  # a host that caps or refuses page-locked memory: pageable
                storages.add(a.untyped_storage().data_ptr())
                pinned += 1
            qa, qb = _quant_bytes(a), _quant_bytes(b)
            assert sorted(qa) == sorted(qb)
            for k in qa:
                assert (torch.equal(qa[k], qb[k]) if isinstance(qa[k], torch.Tensor)
                        else qa[k] == qb[k]), f"{name}::{k}"
            if b.device.type == "cuda":
                on_gpu.add(name.split(".")[2])
    assert on_gpu == {"4", "5"}
    if pinned:
        assert len(storages) < pinned  # carved from shared slabs, not one pinned block each


def _train(checkpoint, out, **knobs):
    from poros.train.trainer import PorosTrainer

    base = dict(output_dir=str(out), quantization="nf4", block_size=1, max_steps=3, save_steps=0,
                progress=False, lora_dropout=0.0, learning_rate=1e-3, warmup_steps=0,
                gradient_accumulation_steps=4, per_device_batch_size=1, max_seq_length=48,
                blockwise_arch_check="off", eval_split=0)
    base.update(knobs)
    tr = PorosTrainer.from_pretrained(checkpoint, **base)
    tables = {n: (p.device.type, p.dtype) for n, p in tr.model.named_parameters()
              if "embed_tokens" in n or "lm_head" in n}  # placed, before training
    metrics = tr.train("synthetic", steps=3, seq_len=48)
    adapters = {n: p.detach().cpu().clone()
                for n, p in tr.model.named_parameters() if p.requires_grad}
    resident = sorted(getattr(tr.block_manager, "resident_block_ids", ()))
    host = sorted(tr.block_manager._cpu_weight_store)
    del tr
    torch.cuda.empty_cache()
    return metrics["loss_history"], adapters, tables, resident, host


def test_every_placement_trains_the_same_bits(checkpoint, tmp_path):
    ref_loss, ref_ad, tables, _, _ = _train(checkpoint, tmp_path / "ref", path="sequential")
    assert set(tables.values()) == {("cuda", torch.float32)}  # PEFT's cast, made on the GPU
    off = dict(offload_embeddings_and_head=True)
    arms = {
        "fast": dict(path="fast"),
        "efficient": dict(path="efficient"),
        "offload_fast": dict(path="fast", **off),
        "offload_eff_resident": dict(path="efficient", resident_blocks=2, **off),
        "resident_fast": dict(path="fast", resident_blocks=3),
        "offload_sequential": dict(path="sequential", **off),
    }
    for name, knobs in arms.items():
        loss, ad, tables, resident, host = _train(checkpoint, tmp_path / name, **knobs)
        assert loss == ref_loss, name
        assert sorted(ad) == sorted(ref_ad), name
        for k in ref_ad:
            assert torch.equal(ad[k], ref_ad[k]), f"{name}: {k}"
        if knobs.get("offload_embeddings_and_head"):
            assert set(tables.values()) == {("cpu", torch.bfloat16)}, name  # compact
        n_res = knobs.get("resident_blocks", 0)
        assert len(resident) == n_res, name
        assert not set(resident) & set(host), name  # a resident block has no host copy


def test_an_offloaded_head_crosses_from_page_locked_chunks(checkpoint, tmp_path):
    """On the efficient path, from the second update the offloaded head's uploads read
    page-locked row chunks (at most 256 MiB each, carved from shared slabs) that cover it."""
    from poros.train.trainer import PorosTrainer

    tr = PorosTrainer.from_pretrained(
        checkpoint, output_dir=str(tmp_path), quantization="nf4", block_size=1, max_steps=3,
        save_steps=0, progress=False, lora_dropout=0.0, warmup_steps=0,
        gradient_accumulation_steps=2, per_device_batch_size=1, max_seq_length=48,
        blockwise_arch_check="off", eval_split=0, path="efficient",
        offload_embeddings_and_head=True)
    tr.train("synthetic", steps=3, seq_len=48)
    head = tr._get_lm_head().weight
    chunks = tr.head._pinned
    assert chunks and chunks[0][0] == 0 and chunks[-1][1] == head.shape[0]
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))  # contiguous rows
    assert all(c.is_pinned() and c.numel() * c.element_size() <= tr.head.CHUNK_BYTES
               for _, _, c in chunks)
    assert all(torch.equal(c, head.data[r0:r1]) for r0, r1, c in chunks)
    del tr
    torch.cuda.empty_cache()
