import pytest


def require_tested_transformers() -> None:
    """Skip the calling module below ``poros.moe``'s tested transformers.

    The only skip on transformers' account: above the floor every module
    imports its classes directly, so a release that moves one fails loudly.
    """
    pytest.importorskip("transformers")
    from poros.core.errors import PorosStackUnsupportedError
    from poros.moe.arch import check_transformers

    try:
        check_transformers()
    except PorosStackUnsupportedError as e:
        pytest.skip(str(e), allow_module_level=True)


def fused_mask_digests(monkeypatch) -> dict:
    """The fused expert compute's dropout masks, a digest per micro-batch a call draws for, by
    layer (``poros.moe.fused._masks``: the gate/up adapter's input mask and the down adapter's,
    in sorted-row order). The fused kernels apply them themselves, so no ``nn.Dropout`` runs for
    a hook to see; a call that batches several micro-batches (block-major's backward) draws each
    one's from its own state, into its own rows, and is digested a micro-batch at a time."""
    import hashlib

    from poros.moe import fused

    got: dict = {}
    real = fused._masks

    def probe(block, plan, h, i, device, states):
        masks = real(block, plan, h, i, device, states)
        if masks is not None:
            for g in range(1 if states is None else len(states)):
                rows = (slice(None) if plan.row_seg is None
                        else (plan.row_seg == g).nonzero().squeeze(1))
                kept = masks.x[rows].cpu().numpy().tobytes() + masks.h[rows].cpu().numpy().tobytes()
                got.setdefault(f"layer {block.layer_idx}", []).append(
                    hashlib.sha256(kept).hexdigest())
        return masks

    monkeypatch.setattr(fused, "_masks", probe)
    return got
