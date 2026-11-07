"""First-block cache: skip FLUX blocks 1..56 when block 0's output barely changed since the last step (Task 8,
docs/engine/08-block-cache.md).

A denoising job runs the same window batch through the transformer once per step. If block 0's residual r0 (its
image-stream output minus its input) moved little since the batch's previous step, the other 56 blocks would
likely add about what they added last time, so the pass reuses that: hidden = block 0's output + the residual blocks
1..56 added at the batch's last computed step (TeaCache / First-Block-Cache).

    computed pass:  x_emb -> [blk 0] -probe-> [blk 1] ... [blk 56] -> norm_out, proj_out     store h56 - h0
    skipped pass:   x_emb -> [blk 0] -probe-> h0 + stored residual -> norm_out, proj_out     (no blocks 1..56,
                                                                                               no weight copies)

The probe sums |r0(t) - r0(t-1)|_1 / |r0(t-1)|_1 over consecutive steps of a batch; while that sum stays under the
threshold the pass skips, and a computed pass starts the sum again. It reads one scalar on the host (a sync after
block 0), so with the cache off (threshold 0) the probe does nothing at all.

Staleness: an entry belongs to key = (job id, window ids of the batch). begin_job() drops everything (and only keys
of the current job are accepted); the painter clears on Stopped and at the end of each job; a batch whose window set
differs from every earlier one (a different composition at another step) finds no entry and computes. Step 0 never
skips.
"""
from collections import Counter

import torch


class _Entry:
    __slots__ = ("prev", "acc", "residual")

    def __init__(self):
        self.prev: torch.Tensor | None = None       # r0 at the batch's previous step
        self.acc = 0.0                              # change summed over the steps skipped since the last computed
        self.residual: torch.Tensor | None = None   # what blocks 1..56 added at the last computed step


class BlockCache:
    """Per-batch first-block cache. threshold 0 (config.FLUX_CACHE_THRESHOLD's default) is off: probe never skips and
    never syncs. stats counts "probes", "skips" and the relative changes seen ("rel": their list, for the sweep)."""

    def __init__(self, threshold: float):
        if threshold < 0:
            raise ValueError(f"cache threshold {threshold} < 0")
        self.threshold = float(threshold)
        self.job = None
        self._e: dict[tuple, _Entry] = {}
        self.stats: Counter = Counter()
        self.rel: list[float] = []

    @property
    def active(self) -> bool:
        return self.threshold > 0

    def __len__(self) -> int:
        return len(self._e)

    def begin_job(self, job_id) -> None:
        """A new job: drop every entry; only keys of this job are accepted until the next begin_job."""
        self.clear()
        self.job = job_id

    def clear(self) -> None:
        """Drop every entry (on Stopped, and when a job ends: frees the residuals' memory)."""
        self._e.clear()
        self.job = None

    def _entry(self, key: tuple) -> _Entry:
        if key[0] != self.job:
            raise ValueError(f"cache key {key[:1]} is not of the current job {self.job!r} (begin_job first)")
        e = self._e.get(key)
        if e is None:
            e = self._e[key] = _Entry()
        return e

    def probe(self, key: tuple, r0: torch.Tensor, step: int) -> bool:
        """True: skip blocks 1..56 for this batch (apply() gives the pass's hidden state). r0 is block 0's output
        residual; it becomes the batch's previous r0 either way."""
        if not self.active:
            return False
        e = self._entry(key)
        self.stats["probes"] += 1
        skip = False
        if step > 0 and e.prev is not None and e.residual is not None and e.prev.shape == r0.shape:
            num = torch.linalg.vector_norm(r0 - e.prev, 1, dtype=torch.float32)
            den = torch.linalg.vector_norm(e.prev, 1, dtype=torch.float32)
            rel = float(num / den.clamp(min=1e-12))         # the one host sync of a probe
            self.rel.append(rel)
            if e.acc + rel < self.threshold:
                e.acc += rel
                skip = True
            else:
                e.acc = 0.0
        else:
            e.acc = 0.0
        e.prev = r0
        self.stats["skips"] += skip
        return skip

    def store(self, key: tuple, residual: torch.Tensor) -> None:
        """What blocks 1..56 added to block 0's output at this computed pass."""
        self._entry(key).residual = residual

    def apply(self, key: tuple, x: torch.Tensor) -> torch.Tensor:
        """x (block 0's output) + the residual stored for key."""
        e = self._entry(key)
        if e.residual is None:
            raise KeyError(f"no residual stored for {key}")
        return x + e.residual


def cached_forward(tr, cache: BlockCache, key: tuple, step: int, hidden_states, encoder_hidden_states,
                   pooled_projections, timestep, img_ids, txt_ids, guidance=None, slots=None, return_dict=False):
    """FluxTransformer2DModel.forward (diffusers 0.41, without controlnet, IP-adapter or gradient checkpointing) with
    the cache's decision between block 0 and the rest: the one pass loop for eager blocks on the weight slots and for
    their CUDA graphs (a block's forward is whatever the runner installed). With the cache computing, it equals the
    diffusers forward bit for bit. slots (engine.slots.BlockSlots): while the cache is active, block 0's slot holds
    its prefetch of the slot's next user (block 3) until the probe has decided: a computed pass releases it, a skipped
    pass copies nothing and block 0 is still in place for the next pass."""
    held = slots is not None and cache.active
    if held:
        slots.hold = {0}
    try:
        return _cached_forward(tr, cache, key, step, hidden_states, encoder_hidden_states, pooled_projections, timestep,
                               img_ids, txt_ids, guidance, slots if held else None, return_dict)
    finally:
        if held:
            slots.hold = set()


def _cached_forward(tr, cache, key, step, hidden_states, encoder_hidden_states, pooled_projections, timestep, img_ids,
                    txt_ids, guidance, slots, return_dict):
    hs = tr.x_embedder(hidden_states)
    timestep = timestep.to(hs.dtype) * 1000
    if guidance is None:
        temb = tr.time_text_embed(timestep, pooled_projections)
    else:
        temb = tr.time_text_embed(timestep, guidance.to(hs.dtype) * 1000, pooled_projections)
    es = tr.context_embedder(encoder_hidden_states)
    rot = tr.pos_embed(torch.cat((txt_ids, img_ids), dim=0))
    blocks = [*tr.transformer_blocks, *tr.single_transformer_blocks]
    kw = dict(temb=temb, image_rotary_emb=rot, joint_attention_kwargs=None)
    es, h0 = blocks[0](hidden_states=hs, encoder_hidden_states=es, **kw)
    if cache.active and cache.probe(key, h0 - hs, step):
        h = cache.apply(key, h0)
    else:
        if slots is not None:
            slots.release(0)
        # h0 may be a graph's reused output buffer, overwritten by block 2: keep it for the residual
        keep = h0.clone() if cache.active else None
        h = h0
        for b in blocks[1:]:
            es, h = b(hidden_states=h, encoder_hidden_states=es, **kw)
        if cache.active:
            cache.store(key, h - keep)
    out = tr.proj_out(tr.norm_out(h, temb))
    return (out,) if not return_dict else {"sample": out}
