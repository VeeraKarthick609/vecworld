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
demand. Task 10 grows this into BlockStreamer (reads from the SSD instead of host memory).
"""
from collections.abc import Callable

import torch

from vecworld.engine.q4k import check_aligned

ALIGN = 256     # each tensor's offset in a slot (cudaMalloc bases are 512-aligned): >= the kernels' 16-byte loads


def _tensors(block: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {**dict(block.named_parameters()), **dict(block.named_buffers())}


class BlockSlots:
    """`ring` device slots of fixed address the blocks' weights are copied into, block i into slot i % ring.

    host(i) gives block i's tensors in host memory by their names in the block (named_parameters + named_buffers;
    every one of them, page-locked for the copies to be asynchronous). From construction on, each block's parameters
    and buffers are views into its slot (their .data replaced); close() puts the host tensors back.

    Forward hooks on the blocks drive the copies: before block i runs, the current stream waits until block i's
    weights are in its slot (issuing the copy if it isn't under way); after it, the next block that uses the same
    slot (cyclically: blocks wrap to the next pass) is copied in on the side stream. Hooks run outside a block's
    forward, so a CUDA graph replayed inside it (engine.graphs.GraphRunner) sees the weights in place."""

    def __init__(self, blocks: list[torch.nn.Module], host: Callable[[int], dict[str, torch.Tensor]], ring: int = 2,
                 device="cuda"):
        if ring < 1 or len(blocks) == 0:
            raise ValueError(f"BlockSlots needs a ring >= 1 and blocks (ring {ring}, {len(blocks)} blocks)")
        self.blocks, self.ring, self.n = list(blocks), ring, len(blocks)
        self.device = torch.device(device)
        self._host, self._layout, size = [], [], 0
        for i, b in enumerate(self.blocks):
            mine, h = _tensors(b), host(i)
            if set(mine) != set(h):
                raise ValueError(f"block {i}: host tensors {sorted(set(h) ^ set(mine))} don't match the block's")
            lay, off = {}, 0
            for name, t in mine.items():
                ht = h[name]
                if ht.shape != t.shape or ht.dtype != t.dtype:
                    raise ValueError(f"block {i} {name}: host {ht.dtype} {tuple(ht.shape)} for {t.dtype} "
                                     f"{tuple(t.shape)}")
                lay[name] = off
                off += -(-ht.numel() * ht.element_size() // ALIGN) * ALIGN
            self._host.append({k: h[k].contiguous() for k in mine})
            self._layout.append(lay)
            size = max(size, off)
        self.slot_bytes = size
        self.slots = [torch.empty(size, dtype=torch.uint8, device=self.device) for _ in range(ring)]
        # (dst view, host tensor) per block: the copies, and the views the block's tensors become
        self._pairs: list[list[tuple[torch.Tensor, torch.Tensor]]] = []
        for i, b in enumerate(self.blocks):
            slot, pairs = self.slots[i % ring], []
            for name, t in _tensors(b).items():
                ht, off = self._host[i][name], self._layout[i][name]
                view = slot[off:off + ht.numel() * ht.element_size()].view(ht.dtype).view(ht.shape)
                if ht.numel():
                    check_aligned(view, f"block {i} {name} in slot {i % ring} (offset {off})")
                t.data = view
                pairs.append((view, ht))
            self._pairs.append(pairs)
        self.stream = torch.cuda.Stream(self.device)
        self.loaded: list[int | None] = [None] * ring       # the block each slot holds (or is receiving)
        self._copied = [torch.cuda.Event() for _ in range(ring)]   # slot s holds loaded[s] (recorded on the side)
        self._freed = [torch.cuda.Event() for _ in range(ring)]    # slot s's last user ended (recorded on main)
        self._used = [False] * ring
        self.copies = 0
        self._hooks = [h for i, b in enumerate(self.blocks) for h in (
            b.register_forward_pre_hook(lambda m, a, i=i: self.before(i)),
            b.register_forward_hook(lambda m, a, o, i=i: self.after(i)))]

    def next_user(self, i: int) -> int:
        """The block that uses block i's slot after it, cyclically (the blocks repeat every pass)."""
        j = i + self.ring
        return j if j < self.n else i % self.ring

    def load(self, i: int) -> None:
        """Copy block i's tensors into its slot on the side stream, after the slot's last user is done."""
        s = i % self.ring
        with torch.cuda.stream(self.stream):
            if self._used[s]:
                self.stream.wait_event(self._freed[s])
            for dst, src in self._pairs[i]:
                dst.copy_(src, non_blocking=True)
            self._copied[s].record(self.stream)
        self.loaded[s] = i
        self.copies += 1

    def before(self, i: int) -> None:
        """Block i is about to run: make the current stream wait for its weights."""
        s = i % self.ring
        if self.loaded[s] != i:
            self.load(i)
        torch.cuda.current_stream(self.device).wait_event(self._copied[s])

    def after(self, i: int) -> None:
        """Block i's work is queued: its slot is free once it ends; start copying the slot's next user in."""
        s = i % self.ring
        self._freed[s].record(torch.cuda.current_stream(self.device))   # (a wait already queued keeps its state)
        self._used[s] = True
        j = self.next_user(i)
        if j != i:
            self.load(j)

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
            for name, t in _tensors(b).items():
                t.data = self._host[i][name]
        self.slots, self._pairs, self.loaded = [], [], [None] * self.ring
