"""Block-major accumulation: an update's micro-batches through the frozen blocks block by block.

The sequential loop runs one micro-batch through every block, then the next. Block-major runs
block 0 for every micro-batch, then block 1, and so on, in the forward and again (blocks
reversed) in the backward, so each frozen block streams to the GPU once a pass for all of them
instead of once per micro-batch: 2·P bytes an update instead of 2·G·P. The ops are the
sequential loop's, micro-batch by micro-batch: the same no-grad block forward, the same
rematerializing backward from each block's saved input (``autograd.grad``), the same head loss
(``/ G``, or the caller's own), and adapter gradients summed in micro-batch order 0..G-1, so
the update is the sequential loop's bit for bit. Dropout too: every forward and recompute
starts from the random state the sequential loop gives it (``UpdateRng``: a dense model's walk
of that loop's stream, a MoE model's keyed states; poros.core.sequential_rng).

``rng_for`` says whether an update runs here (and gives its random states); ``update`` runs one.
The trainer's classic loop and ``poros.Run`` both take this one driver. Both loops give the same
update at any number of micro-batches, so which runs is a matter of speed: block-major, where it
can. At one micro-batch it streams what the sequential loop streams but skips the per-layer
recompute inside a block, and is no slower (Qwen2.5-7B NF4 on an RTX 4090, 512 tokens, one
micro-batch a step: 3.05 s a step against the sequential loop's 3.81 s at 8 rows, 0.64 against
0.65 s at 1; docs/validation/block-major-dropout), for a higher peak at these sizes.
"""

from __future__ import annotations

import collections
import contextlib
import functools
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional

import torch

from poros.core.blockwise import _collect_block_trainable_params
from poros.core.config import train_path
from poros.core.errors import PorosConfigError
from poros.core.sequential_rng import SequentialRng, UnknownDraw, UpdateRng, block_draws
from poros.moe.batching import BatchState

if TYPE_CHECKING:  # pragma: no cover
    from poros.train.trainer import PorosTrainer

logger = logging.getLogger("poros.train.trainer")

Batch = Dict[str, torch.Tensor]
HeadFn = Callable[[torch.Tensor], torch.Tensor]

# Blocks of work the host may have queued on the GPU past the one it is running. Unpaced, the
# host ran as far ahead as CUDA's launch queue let it (about 17 of a 72B update's 80 blocks),
# and every queued block held its weights (prefetch stream) and its inputs (record_stream) in
# memory the allocator could not hand out again until the GPU got there: 9 of the 24 GB it
# reserved for an 11 GB peak (path: efficient, RTX 4090). Two keep the GPU fed: a block is far
# more GPU time than host time to queue. Host-side waits only: the GPU runs the same work in
# the same order on the same streams.
_BLOCKS_AHEAD = 2
# Blocks before the end of the forward at which an offloaded LM head starts crossing: its
# page-locked copy (72B: 2.3 GiB, ~0.1 s on PCIe beside the blocks' own weights) is then done by
# the head phase. Started with the forward it would hold the forward's peak up too.
_HEAD_LEAD = 4



def moe_batch(trainer) -> BatchState:
    """The trainer's MoE batching state (poros.moe.batching), made on first use."""
    state = trainer._moe_batch
    if state is None:
        state = trainer._moe_batch = BatchState()
    return state

def _shape(batch: Batch) -> tuple[int, int]:
    """A micro-batch's (rows, padded length)."""
    ids = batch.get("input_ids")
    if ids is None:
        ids = batch["inputs_embeds"]
    return int(ids.shape[0]), int(ids.shape[1])


def positions(batch: Batch) -> int:
    """The positions a micro-batch's forward runs (rows x length, padding included): each of its
    LoRA dropouts draws over this many times the layer's input width."""
    ids = batch.get("input_ids")
    if ids is not None:
        return int(ids.numel())
    embeds = batch["inputs_embeds"]
    return int(embeds.numel() // embeds.shape[-1])


def _blocks(trainer: "PorosTrainer") -> Optional[Any]:
    """The BlockManager block-major walks: the dense engine's, or the one a ``poros.moe`` model
    streams its dense blocks with; None where no block streams here."""
    if trainer.block_manager is not None:
        if trainer._sss_source is not None:  # an SSD-store dense model: its own loop
            return None
        return trainer.block_manager
    return trainer._moe_block_manager()


def rng_for(trainer: "PorosTrainer", micro_batches: int) -> Optional[UpdateRng]:
    """The random states of an update of ``micro_batches`` run block-major here, or None where
    it runs sequentially: a sequential path, the Mac (its own loop), no streamed blocks, or a run
    block-major gave up on (``update``). The one place an
    update's loop is decided; both loops are exact at any ``micro_batches``, so a choice by it
    is one of speed alone."""
    cfg = trainer.config
    if (train_path(cfg) == "sequential" or trainer.device.type == "mps"
            or trainer._block_major_off):
        return None
    if getattr(trainer.model, "poros_visits", False):  # its own loop: poros.moe.inkling2.visits
        return None
    bm = _blocks(trainer)
    if bm is None:
        return None
    if trainer.keyed_rng is not None:
        return trainer.keyed_rng
    try:
        draws = [block_draws(bm.get_block_modules(b)) for b in range(bm.num_blocks)]
    except UnknownDraw as exc:
        _give_up(trainer, exc)
        return None
    return SequentialRng(trainer.device, draws, trainer._rng_memo)


def _give_up(trainer: "PorosTrainer", exc: UnknownDraw) -> None:
    trainer._block_major_off = True
    logger.warning("%s This update and the rest of the run train sequentially: the same update, "
                   "each block streamed once per micro-batch.", exc.info.message)


def update(trainer: "PorosTrainer", batches: List[Batch], rng: UpdateRng,
           head_fns: Optional[List[HeadFn]] = None) -> Optional[List[torch.Tensor]]:
    """One update's micro-batches, block-major. Returns their losses (detached; the caller sums
    them), or None where a draw the walk cannot follow stopped it before anything was applied:
    the generators are back where it started, and the caller runs it sequentially (as every
    later update, ``rng_for``). The generators then hold what the sequential loop leaves."""
    try:
        with _moe_scope(trainer, batches) as is_moe, _hf_checkpointing_off(trainer.model):
            losses = _drive(trainer, batches, rng, head_fns, is_moe)
    except UnknownDraw as exc:
        rng.rewind()
        _land_head_upload(trainer)
        trainer.head.weight_device = None
        _give_up(trainer, exc)
        return None
    except BaseException:
        rng.rewind()  # nothing to check on the way out: the update failed
        _land_head_upload(trainer)
        raise
    rng.finish()
    trainer._ran_loop(block_major=True)
    return losses


def _land_head_upload(trainer: "PorosTrainer") -> None:
    """An update given up while the offloaded head's copy was in flight: let the copy land (the
    compute stream waits on it) before its device buffer can be freed and reused."""
    upload, trainer.head.upload_pending = trainer.head.upload_pending, None
    if upload is not None:
        try:
            trainer.head.end_upload(upload)
        except BaseException:  # noqa: BLE001 -- the update's own error is the one raised
            pass


@contextlib.contextmanager
def _hf_checkpointing_off(model: Any) -> Iterator[None]:
    """HF's per-layer gradient checkpointing off inside the update. PEFT's
    prepare_model_for_kbit_training turns it on, so every decoder layer called inside the
    block's own rematerialization would run forward a second time inside ``autograd.grad``
    (4 forwards per layer per micro-batch instead of 3; on 72B 17,920 4-bit matmuls an update).
    The block's recompute already holds what it needs: with it off a block's backward holds the
    activations of its layers for one micro-batch (block_size 1: one layer's). Kept on, the
    efficient path ran 26-30% slower at 7B and 32B
    (docs/validation/block-major-dropout/). Restored afterwards."""
    flipped = [m for m in model.modules() if getattr(m, "gradient_checkpointing", False) is True]
    for m in flipped:
        m.gradient_checkpointing = False
    try:
        yield
    finally:
        for m in flipped:
            m.gradient_checkpointing = True


@contextlib.contextmanager
def _moe_scope(trainer: "PorosTrainer", batches: List[Batch]) -> Iterator[bool]:
    """For a ``poros.moe`` model (yields True; a dense one: False, nothing set up): the driver
    walks the BlockManager its dense blocks stream with, so a block's dense weights move once a
    pass, and a MoE layer's experts are ensured once a pass for the union of the micro-batches'
    routes, each layer keeping one rematerializable output per micro-batch (forward order =
    backward order). Expert adapter grads accumulate in ``_ManagedExpertFunction.backward``, in
    micro-batch order. The head is the model's own HF loss, what its sequential loop runs."""
    if trainer.block_manager is not None:
        yield False
        return
    from poros.moe.modules import ManagedMoeBlock
    from poros.moe.store import EXPERT_DQ_CACHE

    bm = trainer._moe_block_manager()
    model = trainer.model
    assert bm is not None and model is not None
    text = getattr(model.config, "get_text_config", lambda: model.config)()
    if getattr(text, "output_router_logits", False):
        raise PorosConfigError(
            "block-major accumulation does not train a MoE model's router aux loss "
            "(output_router_logits).",
            remediation="Set path: sequential for this model.",
        )
    if (trainer._prefetch_stream is None and trainer.config.enable_prefetch
            and trainer.device.type == "cuda"):
        trainer._prefetch_stream = torch.cuda.Stream(device=trainer.device)
    saved = (trainer.block_manager, trainer.block_trainable_params)
    trainer.block_manager = bm
    block_params, static = trainer._partition_trainable_parameters_by_block()
    if static:
        trainer.block_manager, trainer.block_trainable_params = saved
        raise RuntimeError("block-major MoE: trainable params outside the streamed blocks: "
                           + ", ".join(n for n, _ in static[:5]))
    trainer.block_trainable_params = block_params
    managed = [m for m in model.modules() if isinstance(m, ManagedMoeBlock)]
    shapes = [_shape(b) for b in batches]
    # sized on the update before, through its optimizer step; one beyond the full shape drops it
    trainer.compute.cache_before_update(model, shapes)
    for m in managed:
        m.remat_slots = len(batches)
    EXPERT_DQ_CACHE.enabled = True
    try:
        yield True
    finally:
        EXPERT_DQ_CACHE.enabled = False
        EXPERT_DQ_CACHE.clear()
        moe_batch(trainer).reset()
        if not trainer._moe_dq_logged:
            trainer._moe_dq_logged = True
            # a diagnostic, not a warning: in the debug log, never on a run's screen
            logger.info("block-major MoE expert dequant cache after update 1: hits %d "
                        "misses %d", EXPERT_DQ_CACHE.hits, EXPERT_DQ_CACHE.misses)
        for m in managed:
            m.remat_slots = 1
            m._remat_cache = None
            m._fwd_counts = None
        trainer.block_manager, trainer.block_trainable_params = saved
        if trainer._sss_source is not None:
            trainer._sss_after_pass()  # one replayable schedule per update
        # Either store: size the device expert cache to what fits once update 1 and its
        # optimizer step are measured (poros.train.compute) -- otherwise every MoE layer's
        # experts are copied host->device again on every pass (py-spy: ~25% of a Qwen3-30B-A3B
        # update). Decides when bytes move, never which.
        trainer.compute.cache_after_update(model, shapes)


def _drive(trainer: "PorosTrainer", batches: List[Batch], rng: UpdateRng,
           head_fns: Optional[List[HeadFn]], is_moe: bool) -> List[torch.Tensor]:
    """The update itself: every block's forward for all G micro-batches, the head per
    micro-batch, then every block's rematerializing backward for all of them, blocks reversed.

    It prefetches the next block on the prefetch stream while the current one computes (INV-2
    is load_block_for_inference's record_stream), keeps HF's mask and rotary per micro-batch for
    every block, may keep each block's inputs in host RAM (``path: efficient``) and one block's
    dequantized NF4 weights for all its micro-batches (``nf4_dequant_cache_mb``).
    """
    bm = trainer.block_manager
    assert bm is not None
    num_blocks = bm.num_blocks
    G = len(batches)
    dev = trainer.device
    prefetch = trainer._get_adapter_prefetch_stream()
    offload_head = bool(getattr(trainer.config, "offload_embeddings_and_head", False))
    cache = trainer._get_nf4_dequant_cache()
    cache_scope = cache.scope if cache is not None else contextlib.nullcontext

    boundary: List[List[torch.Tensor]] = [[] for _ in range(G)]
    # path: efficient: block inputs wait in pinned host RAM for the backward (one pool of
    # host slabs for the whole run, made by the first update that needs them and reused)
    offl = trainer._boundary_offload
    if offl is None and train_path(trainer.config) == "efficient":
        from poros.core.boundary_offload import make_boundary_offload

        offl = trainer._boundary_offload = make_boundary_offload(True, dev)
    if offl is not None:
        offl.begin_update()
    host_b: List[Dict[int, Any]] = [dict() for _ in range(G)]
    attn: List[Any] = [None] * G
    pos: List[Any] = [None] * G
    labels: List[Any] = [None] * G
    for g, batch in enumerate(batches):
        rng.set(g, 0)
        h, attn[g], pos[g], labels[g] = trainer._build_inputs(batch)
        rng.check(g, 0, "the embedding")
        boundary[g].append(h.detach())

    # HF's mask/rotary for micro-batch g, built once and reused by every block in both passes:
    # create_causal_mask's padding check (padding_mask.all()) is a host sync, which per block x
    # micro-batch x pass was 1,280 syncs per 72B update. Keyed by everything it reads from the
    # hidden state.
    dec_cache: Dict[Any, Any] = {}

    def _dec(g: int, h: torch.Tensor) -> Any:
        key = (g, h.dtype, h.device, tuple(h.shape))
        d = dec_cache.get(key)
        if d is None:
            d = dec_cache[key] = trainer._prepare_decoder_attention(h, attn[g], pos[g])
        return d

    def _load(block_id: int, ahead: Optional[int]) -> None:
        """Make ``block_id`` usable on the default stream; start ``ahead``."""
        if prefetch is not None:
            torch.cuda.current_stream().wait_stream(prefetch)
        bm.load_block_for_inference(block_id, device=dev)  # no-op if prefetched
        if prefetch is not None and ahead is not None:
            bm.load_block_for_inference(ahead, device=dev, stream=prefetch)

    if prefetch is not None and num_blocks > 0:
        bm.load_block_for_inference(0, device=dev, stream=prefetch)

    queued: "collections.deque[torch.cuda.Event]" = collections.deque()

    def _pace() -> None:
        """After a block's work is queued: wait while more than _BLOCKS_AHEAD are."""
        if dev.type != "cuda":
            return
        done = torch.cuda.Event()
        done.record()
        queued.append(done)
        while len(queued) > _BLOCKS_AHEAD:
            queued.popleft().synchronize()

    head_upload = None
    # ---- Forward: each block once, all G micro-batches ----
    for block_id in range(num_blocks):
        _load(block_id, block_id + 1 if block_id + 1 < num_blocks else None)
        if offload_head and block_id == max(0, num_blocks - _HEAD_LEAD) and head_upload is None:
            # the head's weight crosses while the last blocks compute, and is freed after the
            # head phase
            head_upload = trainer.head.upload_pending = trainer.head.begin_upload()
        try:
            with cache_scope():
                # a MoE layer's experts once for all G (poros.moe.batching), added in by finish()
                visit = moe_batch(trainer).forward(bm.get_block_modules(block_id), G) \
                    if is_moe else None
                for g in range(G):
                    trainer._record_remat_event("bm_forward", block_id)
                    rng.set(g, block_id)
                    with torch.no_grad():
                        run = functools.partial(
                            trainer._forward_block_layers,
                            block_id, boundary[g][block_id], attn[g], pos[g],
                            checkpoint=False, dec=_dec(g, boundary[g][block_id]),
                        )
                        out = run() if visit is None else visit.run(g, run)
                    rng.check(g, block_id + 1, f"block {block_id}")
                    boundary[g].append(out.detach())
                    if offl is not None and block_id < num_blocks - 1:
                        # this block's input is next needed by its backward
                        host_b[g][block_id] = offl.offload(boundary[g][block_id])
                        boundary[g][block_id] = None  # type: ignore[call-overload]
                if visit is not None:
                    with torch.no_grad():
                        visit.finish()
        finally:
            # keep the last block: the backward starts with it (unless the head is offloaded:
            # then it makes room for the head's weight)
            if block_id != num_blocks - 1 or offload_head:
                bm.release_block(block_id)
        _pace()

    # ---- Head per micro-batch: grad of its loss w.r.t. the final hidden ----
    if offload_head:
        if head_upload is None:  # no blocks: nothing to overlap
            head_upload = trainer.head.begin_upload()
        trainer.head.upload_pending = None
        trainer.head.weight_device = trainer.head.end_upload(head_upload)
        head_upload = None  # the copy is the trainer's alone, freed with it below
    losses: List[torch.Tensor] = []
    grad_hidden: List[torch.Tensor] = []
    for g in range(G):
        rng.set(g, num_blocks)
        final_hidden = boundary[g][num_blocks].detach().requires_grad_(True)
        with torch.enable_grad():
            logits = None
            if head_fns is not None:
                # the caller's own loss for micro-batch g, already weighted by its share
                # (poros.Run: out.value * rows / batch rows); summed, not divided by G
                logits = trainer._logits_from_hidden(final_hidden)
                loss = head_fns[g](logits)
                scaled = loss
            else:  # the sequential loop's loss, chosen in one place (poros.train.head)
                loss = trainer.head.loss(final_hidden, labels[g], normed=False)
                scaled = loss / G
            grad = torch.autograd.grad(scaled, final_hidden, retain_graph=False,
                                       create_graph=False)[0]
        del logits
        rng.check(g, num_blocks, "the head")
        grad_hidden.append(grad)
        losses.append(loss.detach())
        boundary[g].pop()  # the final hidden is no longer needed
    # off the GPU between uses: the head's copy is freed before the backward (whose blocks reuse
    # its memory); the next update's forward uploads it again
    trainer.head.weight_device = None

    # ---- Backward: blocks reversed, each once, micro-batches 0..G-1 ----
    pending: List[Dict[int, torch.Tensor]] = [dict() for _ in range(G)]
    for block_id in reversed(range(num_blocks)):
        trainable_params = _collect_block_trainable_params(trainer, block_id)
        _load(block_id, block_id - 1 if block_id > 0 else None)
        if offl is not None:
            for g in range(G):
                if block_id in pending[g]:
                    boundary[g][block_id] = offl.ready(pending[g].pop(block_id))
                if block_id - 1 in host_b[g]:  # one block ahead, on its own stream
                    pending[g][block_id - 1] = offl.fetch(host_b[g].pop(block_id - 1))
        try:
            with cache_scope():
                if is_moe:  # every micro-batch's expert backward in one call
                    moe_batch(trainer).backward(bm.get_block_modules(block_id), grad_hidden)
                for g in range(G):
                    trainer._record_remat_event("bm_backward", block_id)
                    rng.set(g, block_id)  # the masks its forward drew
                    with torch.enable_grad(), trainer._remat_scope(block_id):
                        remat_input = boundary[g][block_id].detach().requires_grad_(True)
                        remat_output = trainer._forward_block_layers(
                            block_id, remat_input, attn[g], pos[g], checkpoint=False,
                            dec=_dec(g, remat_input),
                        )
                    grads = torch.autograd.grad(
                        remat_output,
                        [remat_input, *trainable_params],
                        grad_outputs=grad_hidden[g],
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    grad_hidden[g] = grads[0]
                    for param, grad in zip(trainable_params, grads[1:]):
                        if grad is None:
                            continue
                        if param.grad is None:
                            param.grad = grad.detach()
                        else:
                            param.grad.add_(grad.detach())
                    del remat_output, remat_input, grads
                    boundary[g].pop()  # this block's input is consumed
        finally:
            if is_moe:
                moe_batch(trainer).end_backward(bm.get_block_modules(block_id))
            bm.release_block(block_id)
        _pace()

    if is_moe:  # the batched expert visits' checks, read once (fail closed before the step)
        moe_batch(trainer).verify()
    if offl is not None and not trainer._boundary_offload_logged:
        from poros.sss import pinning

        trainer._boundary_offload_logged = True
        # where the block inputs went, kept for the run's receipt (block_inputs, beside
        # host_staging) and the debug log: a diagnostic, never a line on a run's screen
        trainer._boundary_offload_stats = dict(offl.stats)
        logger.info("path efficient after update 1: block inputs %s; pin budget %s",
                    offl.stats, pinning.staging())
    return losses
