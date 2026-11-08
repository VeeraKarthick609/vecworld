"""Fixed-address device slots the FLUX blocks' weights stream through (Task 7, docs/engine/07-cuda-graphs.md).

diffusers' group offload re-allocates a block's weights on every onload, so their device addresses change from pass
to pass, and a CUDA graph (which bakes in every address it reads) can't replay over them. BlockSlots instead keeps
`ring` device buffers ("slots") for the whole run, each the size of the largest block. Every tensor of block i (its
packed Q4_K weights, biases, norm weights) is a fixed view into slot i % ring, at an offset the layout aligns to
ALIGN bytes. Before block i runs, its tensors are copied from page-locked host memory into that slot on a side
stream; CUDA events order the copy after the slot's previous user (block i - ring) and the block after the copy.

    main stream   [block i-1 ][block i   ][block i+1 ][block i+2 ] ...
    side stream        [copy i+1  ]  [copy i+2  ]  [copy i+3  ]       (slot (i+1) % 2 is free once block i-1 ends)

So block i + 1's copy overlaps block i's compute. A block that runs out of order (or first) has its copy issued on
demand. A block in `hold` keeps its slot's next copy back until release(i): the first-block cache (engine.cache)
holds block 0's, so a pass that skips blocks 1..56 copies nothing and block 0 stays in place for the next pass.

The copies and events are issued by the extension's SlotRing (csrc/stream.cpp), one call per hook that keeps the GIL
(Task 10a: a PyTorch op releases and re-takes it, milliseconds each with another Python thread busy).
engine.stream.BlockStreamer feeds the slots from one contiguous run of bytes per block (a host arena now, the SSD in
Task 10b) instead of per-tensor host copies.
"""
import threading
from collections.abc import Callable

import torch

from vecworld.engine.q4k import check_aligned

ALIGN = 256     # each tensor's offset in a slot (cudaMalloc bases are 512-aligned): >= the kernels' 16-byte loads


def _tensors(block: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {**dict(block.named_parameters()), **dict(block.named_buffers())}


def block_layout(block: torch.nn.Module) -> tuple[dict[str, int], int]:
    """Where each of a block's tensors (named_parameters + named_buffers, in that order) sits in its slot: byte offsets,
    each rounded up to ALIGN, and the block's size (its end, rounded up to ALIGN). engine.stream.HostArena packs a
    block's host bytes in this same layout, so a block is copied in with one copy."""
    lay, off = {}, 0
    for name, t in _tensors(block).items():
        lay[name] = off
        off += -(-t.numel() * t.element_size() // ALIGN) * ALIGN
    return lay, off


class BlockSlots:
    """`ring` device slots of fixed address the blocks' weights are copied into, block i into slot i % ring.

    host(i) gives block i's tensors in host memory by their names in the block (named_parameters + named_buffers;
    every one of them, page-locked for the copies to be asynchronous). From construction on, each block's parameters
    and buffers are views into its slot (their .data replaced); close() puts the host tensors back.

    Forward hooks on the blocks drive the copies: before block i runs, the current stream waits until block i's
    weights are in its slot (issuing the copy if it isn't under way); after it, the next block that uses the same
    slot (cyclically: blocks wrap to the next pass) is copied in on the side stream. Hooks run outside a block's
    forward, so a CUDA graph replayed inside it (engine.graphs.GraphRunner) sees the weights in place.

    Subclasses (engine.stream.BlockStreamer) give the copies another source by overriding _copies(i) and _restore(i),
    and may defer install()."""

    def __init__(self, blocks: list[torch.nn.Module], host: Callable[[int], dict[str, torch.Tensor]], ring: int = 2,
                 device="cuda"):
        self._init(blocks, ring, device)
        self._host = []
        for i, b in enumerate(self.blocks):
            mine, h = _tensors(b), host(i)
            if set(mine) != set(h):
                raise ValueError(f"block {i}: host tensors {sorted(set(h) ^ set(mine))} don't match the block's")
            for name, t in mine.items():
                ht = h[name]
                if ht.shape != t.shape or ht.dtype != t.dtype:
                    raise ValueError(f"block {i} {name}: host {ht.dtype} {tuple(ht.shape)} for {t.dtype} "
                                     f"{tuple(t.shape)}")
            bad = [k for k in mine if not h[k].is_contiguous()]
            if bad:
                # Loud, like check_aligned: a contiguous copy would be pageable (no asynchronous copy) and close()
                # would hand it back in place of the page-locked original.
                raise ValueError(f"block {i}: host tensors {bad} are not contiguous")
            self._host.append({k: h[k] for k in mine})
        self.install()

    def _init(self, blocks, ring, device) -> None:
        """The layout and the state that outlives install() / close()."""
        if ring < 1 or len(blocks) == 0:
            raise ValueError(f"BlockSlots needs a ring >= 1 and blocks (ring {ring}, {len(blocks)} blocks)")
        self.blocks, self.ring, self.n = list(blocks), ring, len(blocks)
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device(self.device.type, torch.cuda.current_device())
        self._layout, self.nbytes = [], []
        for b in self.blocks:
            lay, size = block_layout(b)
            self._layout.append(lay)
            self.nbytes.append(size)
        self.slot_bytes = max(self.nbytes)
        self.slots: list[torch.Tensor] = []
        self.loaded: list[int | None] = [None] * ring       # the block each slot holds (or is receiving)
        self._used = [False] * ring
        self._hooks: list = []
        self._ring = None                                   # the extension's SlotRing while installed
        self._views: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        self.copies = 0
        # Blocks whose after() holds back the prefetch of their slot's next user until release(i) (engine.cache: block
        # 0's, until the cache decides whether blocks 1..56 run at all), and the prefetches held back.
        self.hold: set[int] = set()
        self._held: dict[int, int] = {}
        self._owner: int | None = None                      # the thread that ran the first block (see before())

    def install(self) -> None:
        """Allocate the slots, make every block's tensors views into its slot and hook the blocks (a no-op while
        installed). Everything that can fail (the extension, the side stream, the SlotRing, the slots' memory, the
        alignment checks, the copy lists) happens before any block tensor is repointed; if repointing or hooking
        fails, the tensors and hooks are put back. A failed install() leaves the blocks as they were."""
        from vecworld.engine.ext import load_ext

        if self._hooks:
            return
        ring = self.ring
        try:
            self._ext = load_ext()
            self.stream = torch.cuda.Stream(self.device)
            self._ring = self._ext.SlotRing(ring, self.stream.cuda_stream, self.device.index)
            self.slots = [torch.empty(self.slot_bytes, dtype=torch.uint8, device=self.device) for _ in range(ring)]
            # (block tensor, slot view) per block: what the block's tensors are while installed
            self._views = []
            for i, b in enumerate(self.blocks):
                slot, views = self.slots[i % ring], []
                for name, t in _tensors(b).items():
                    off, nb = self._layout[i][name], t.numel() * t.element_size()
                    view = slot[off:off + nb].view(t.dtype).view(t.shape)
                    if nb:
                        check_aligned(view, f"block {i} {name} in slot {i % ring} (offset {off})")
                    views.append((t, view))
                self._views.append(views)
            self._prepare_copies()
        except BaseException:
            self._teardown()
            raise
        swapped, hooks = [], []
        try:
            for views in self._views:
                for t, view in views:
                    swapped.append((t, t.data))
                    t.data = view
            # always_call: the slot is released after a block whose forward raised too, ordered after whatever
            # kernels it queued, so a later copy into the slot can't overwrite weights they still read
            for i, b in enumerate(self.blocks):
                hooks.append(b.register_forward_pre_hook(lambda m, a, i=i: self.before(i)))
                hooks.append(b.register_forward_hook(lambda m, a, o, i=i: self.after(i), always_call=True))
        except BaseException:
            for h in hooks:
                h.remove()
            for t, old in reversed(swapped):
                t.data = old
            self._teardown()
            raise
        self.loaded, self._used = [None] * ring, [False] * ring
        self._held.clear()
        self._hooks = hooks

    def _teardown(self) -> None:
        """Drop what install() made (the blocks' tensors must not view the slots any more)."""
        self.slots, self._views, self.loaded = [], [], [None] * self.ring
        self._ring = None

    def _prepare_copies(self) -> None:
        """Called by install() before the blocks' tensors become slot views: the per-tensor copy lists (device view,
        host tensor, bytes), fixed for the run."""
        self._flat = []
        for i, views in enumerate(self._views):
            flat = []
            for (t, view), name in zip(views, _tensors(self.blocks[i])):
                ht = self._host[i][name]
                if ht.numel():
                    flat += [view.data_ptr(), ht.data_ptr(), ht.numel() * ht.element_size()]
            self._flat.append(flat)

    def _copies(self, i: int) -> list[int]:
        """Block i's copies into its slot: flat (device dst, host src, bytes) triples."""
        return self._flat[i]

    def _restore(self, i: int) -> dict[str, torch.Tensor]:
        """What block i's tensors become at close()."""
        return self._host[i]

    def next_user(self, i: int) -> int:
        """The block that uses block i's slot after it, cyclically (the blocks repeat every pass)."""
        j = i + self.ring
        return j if j < self.n else i % self.ring

    def load(self, i: int) -> int:
        """Copy block i's tensors into its slot on the side stream, after the slot's last user is done. Returns the
        copy's ticket (SlotRing: 1, 2, 3, ... in issue order)."""
        s = i % self.ring
        ticket = self._ring.load(s, self._used[s], self._copies(i))
        self.loaded[s] = i
        self.copies += 1
        return ticket

    def before(self, i: int) -> None:
        """Block i is about to run: make the current stream wait for its weights. The slots' bookkeeping and event
        order assume one thread (the game's GPU queue worker): the first block run claims the slots for its thread
        and a block run from any other thread raises."""
        me = threading.get_ident()
        if self._owner is None:
            self._owner = me
        elif self._owner != me:
            raise RuntimeError(f"BlockSlots belongs to thread {self._owner}; block {i} was run from thread {me}")
        s = i % self.ring
        if self.loaded[s] != i:
            self.load(i)
        self._ring.before(s)

    def after(self, i: int) -> None:
        """Block i's work is queued: its slot is free once it ends; start copying the slot's next user in."""
        s = i % self.ring
        self._ring.after(s)                                 # (a wait already queued keeps its state)
        self._used[s] = True
        j = self.next_user(i)
        if j == i:
            return
        if i in self.hold:
            self._held[i] = j
        else:
            self.load(j)

    def release(self, i: int) -> None:
        """Issue the prefetch block i's after() held back (see hold), unless its block is already in the slot."""
        j = self._held.pop(i, None)
        if j is not None and self.loaded[j % self.ring] != j:
            self.load(j)

    def unhold(self) -> None:
        """End a hold: hold nothing, and forget the prefetches held back and never released (a pass that skipped
        blocks 1..56, or raised, leaves block 0 in its slot, so its next user's copy is simply issued again by the next
        pass's after())."""
        self.hold = set()
        self._held.clear()

    def close(self) -> None:
        """Remove the hooks and put each block's host tensors back as its parameters and buffers (safe twice). The
        slots are freed once nothing views into them."""
        if not self._hooks:
            return
        torch.cuda.synchronize(self.device)
        for h in self._hooks:
            h.remove()
        self._hooks = []
        for i, b in enumerate(self.blocks):
            back = self._restore(i)
            for name, t in _tensors(b).items():
                t.data = back[name]
        self._teardown()
