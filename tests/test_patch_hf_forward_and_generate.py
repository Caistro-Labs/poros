"""The HF patch's (``poros.patch.hf``) forward and backward kwargs, generate on a patched model,
state_dict purity, and a double patch refused.

  1. a second forward between forward and backward must not make the backward rematerialize with
     the last forward's masks (silently wrong adapter gradients, or a shape crash on different
     lengths)
  2. generate() on a patched model must never return garbage after the first new token (an empty
     KV cache while positions advance)
  3. the bookkeeping refs must not register as submodules (state_dict() and every PEFT adapter
     save would carry _poros_* keys)
  4. a second patch_hf_model() is refused, and the model still works
"""

import warnings

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

warnings.filterwarnings("ignore")


def _tiny_patched(block_size=2, seed=7):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    from poros.patch.hf import patch_hf_model

    torch.manual_seed(seed)
    model = get_peft_model(
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=256, hidden_size=64, intermediate_size=128,
                num_hidden_layers=4, num_attention_heads=4,
                num_key_value_heads=2, max_position_embeddings=64,
            )
        ),
        LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]),
    )
    patch_hf_model(model, block_size=block_size, arch_check="off")
    return model


def _grads(model):
    return {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    }


@pytest.mark.usefixtures("cpu_only")
class TestBackwardUsesItsOwnForwardsKwargs:
    def test_interleaved_no_grad_forward_does_not_corrupt_grads(self):
        """Reference run: forward -> backward. Corruption run: forward ->
        DIFFERENT no_grad forward (other mask/length) -> backward. Gradients
        must be identical -- the backward must replay ITS forward's kwargs,
        not the last forward's."""
        torch.manual_seed(3)
        ids = torch.randint(0, 256, (2, 8))
        mask = torch.ones(2, 8, dtype=torch.long)
        other_ids = torch.randint(0, 256, (2, 16))

        model = _tiny_patched()
        out = model(input_ids=ids, attention_mask=mask, labels=ids)
        out.loss.backward()
        ref = _grads(model)

        model2 = _tiny_patched()
        out2 = model2(input_ids=ids, attention_mask=mask, labels=ids)
        with torch.no_grad():
            model2(input_ids=other_ids)  # eval / ref-model style interleave
        out2.loss.backward()
        got = _grads(model2)

        assert ref.keys() == got.keys() and ref
        for name in ref:
            assert torch.equal(ref[name], got[name]), name

    def test_interleaved_forward_with_different_mask_same_shape(self):
        """The nastiest variant: same shape, different padding mask -- the
        original defect produced silently wrong grads here (no crash)."""
        torch.manual_seed(4)
        ids = torch.randint(0, 256, (2, 8))
        mask_a = torch.ones(2, 8, dtype=torch.long)
        mask_b = torch.ones(2, 8, dtype=torch.long)
        mask_b[:, -3:] = 0

        model = _tiny_patched()
        out = model(input_ids=ids, attention_mask=mask_a, labels=ids)
        out.loss.backward()
        ref = _grads(model)

        model2 = _tiny_patched()
        out2 = model2(input_ids=ids, attention_mask=mask_a, labels=ids)
        with torch.no_grad():
            model2(input_ids=ids, attention_mask=mask_b)
        out2.loss.backward()
        got = _grads(model2)

        for name in ref:
            assert torch.equal(ref[name], got[name]), name


class TestGenerateOnPatchedModel:
    def test_default_generate_matches_resident_bitwise(self):
        torch.manual_seed(7)
        ids = torch.randint(0, 256, (1, 8))
        # resident reference: same seed, unpatched
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(7)
        ref_model = get_peft_model(
            Qwen2ForCausalLM(
                Qwen2Config(
                    vocab_size=256, hidden_size=64, intermediate_size=128,
                    num_hidden_layers=4, num_attention_heads=4,
                    num_key_value_heads=2, max_position_embeddings=64,
                )
            ),
            LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]),
        ).eval()
        with torch.no_grad():
            ref = ref_model.generate(
                input_ids=ids, max_new_tokens=6, do_sample=False
            )

        model = _tiny_patched().eval()
        with torch.no_grad():
            got = model.generate(input_ids=ids, max_new_tokens=6, do_sample=False)
        assert torch.equal(ref, got)

    def test_patch_turns_use_cache_off_everywhere(self):
        model = _tiny_patched()
        base = model.get_base_model()
        assert model.config.use_cache is False
        assert base.config.use_cache is False
        gen_cfg = getattr(model, "generation_config", None)
        if gen_cfg is not None and hasattr(gen_cfg, "use_cache"):
            assert gen_cfg.use_cache is False

    def test_forced_use_cache_raises_instead_of_garbage(self):
        """A caller forcing use_cache=True must get a clear error, never
        silently-wrong tokens (the cache stays empty while positions
        advance)."""
        model = _tiny_patched().eval()
        ids = torch.randint(0, 256, (1, 8))
        with pytest.raises(RuntimeError, match="KV cache"):
            with torch.no_grad():
                model.generate(
                    input_ids=ids, max_new_tokens=4, do_sample=False,
                    use_cache=True,
                )


class TestStateDictPurity:
    def test_no_poros_keys_in_state_dict_or_adapter_save(self, tmp_path):
        from peft import get_peft_model_state_dict

        model = _tiny_patched()
        keys = list(model.state_dict().keys())
        polluted = [k for k in keys if "_poros" in k]
        assert polluted == [], polluted[:5]

        adapter = get_peft_model_state_dict(model)
        junk = [k for k in adapter if "_poros" in k]
        assert junk == [], junk[:5]

    def test_state_dict_size_unchanged_by_patch(self):
        from peft import LoraConfig, get_peft_model
        from transformers import Qwen2Config, Qwen2ForCausalLM

        from poros.patch.hf import patch_hf_model

        torch.manual_seed(7)
        model = get_peft_model(
            Qwen2ForCausalLM(
                Qwen2Config(
                    vocab_size=256, hidden_size=64, intermediate_size=128,
                    num_hidden_layers=4, num_attention_heads=4,
                    num_key_value_heads=2, max_position_embeddings=64,
                )
            ),
            LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]),
        )
        before = len(model.state_dict())
        patch_hf_model(model, block_size=2, arch_check="off")
        assert len(model.state_dict()) == before


@pytest.mark.usefixtures("cpu_only")
class TestDoublePatchRefused:
    def test_second_patch_raises_and_the_model_still_works(self):
        from poros.core.errors import PorosError
        from poros.patch.hf import patch_hf_model

        model = _tiny_patched()
        with pytest.raises(PorosError, match="already"):
            patch_hf_model(model, block_size=2, arch_check="off")
        ids = torch.randint(0, 256, (1, 8))
        out = model(input_ids=ids, labels=ids)
        assert torch.isfinite(out.loss)
