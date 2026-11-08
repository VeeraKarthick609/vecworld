r"""The layer streamer: every FLUX block's weights in one page-locked host arena, streamed block by block through a
ring of fixed-address device slots (Task 10a, docs/engine/10-streamer.md).

Built on Task 7's BlockSlots (engine.slots), which already keeps the ring, the side stream and the events. What the
streamer changes is where a block's bytes come from:

- HostArena packs block i's tensors into one contiguous run of bytes laid out exactly as in its slot (block_layout),
  all blocks in one host buffer registered with the driver once (cudaHostRegister), so the DMA engine reads it
  directly. The blocks' host tensors become views of the arena (their old copies are freed as they are moved).
- BlockStreamer(blocks, source, ring) copies source(i), block i's bytes, into slot i % ring with ONE copy (BlockSlots
  made one per tensor, ~20 a block). source is the seam Task 10b plugs the SSD reader into; here it is the arena.

Copies and events go through the extension's SlotRing, which keeps the GIL (csrc/stream.cpp): the per-block
sequencing costs a few microseconds of Python whatever other threads do.

    host arena  [ block 0 | block 1 | block 2 | ... | block 56 ]      one cudaHostRegister'ed buffer (6.7 GB)
                     \  one cudaMemcpyAsync per block, side stream, after the slot's last user (event)
    device      [ slot 0 ][ slot 1 ][ slot 2 ]                        block i in slot i % 3, 191 MB each
"""
from collections.abc import Callable

import numpy as np
import torch

from vecworld.engine.slots import ALIGN, BlockSlots, _tensors, block_layout


def _register(ptr: int, n: int) -> None:
    err = torch.cuda.cudart().cudaHostRegister(ptr, n, 0)
    if int(err) != 0:
        raise RuntimeError(f"cudaHostRegister of the {n / 1e9:.2f} GB block arena failed ({err})")


class HostArena:
    """Every block's tensors moved into one page-locked host buffer, block i a contiguous run in its slot layout.

    Packing copies a block's tensors in and points them (.data) at their place in the arena, block by block, so the
    old host copies are freed as it goes (resident memory stays about one model's size; the arena's pages are only
    committed as they are written). The blocks' tensors must be on the CPU and must not be page-locked in place
    (cudaHostRegister'ed memory freed while registered breaks later copies: painter._pin_in_place). close()
    unregisters the arena (safe twice); the blocks keep viewing it, as ordinary memory."""

    def __init__(self, blocks: list[torch.nn.Module]):
        lays = [block_layout(b) for b in blocks]
        self.offsets, off = [], 0
        for _, size in lays:
            self.offsets.append(off)
            off += size
        self.sizes = [size for _, size in lays]
        self.nbytes = off
        self._raw = torch.empty(off + ALIGN, dtype=torch.uint8)    # the CPU allocator aligns to 64 bytes only
        skip = -self._raw.data_ptr() % ALIGN
        self.buf = self._raw[skip:skip + off]
        for i, b in enumerate(blocks):
            lay, base = lays[i][0], self.offsets[i]
            for name, t in _tensors(b).items():
                if t.device.type != "cpu":
                    raise ValueError(f"block {i} {name} is on {t.device}, not the CPU")
                nb = t.numel() * t.element_size()
                v = self.buf[base + lay[name]:base + lay[name] + nb].view(t.dtype).view(t.shape)
                v.copy_(t.data)
                t.data = v
        _register(self._raw.data_ptr(), self._raw.numel())
        self._registered = True
        self._blocks = [self.buf[o:o + n] for o, n in zip(self.offsets, self.sizes)]

    def block(self, i: int) -> torch.Tensor:
        """Block i's bytes (uint8, a view of the arena; the same tensor object every call)."""
        return self._blocks[i]

    def close(self) -> None:
        """Unregister the arena (safe twice): call it after the streamer's close(), whose sync ends the copies."""
        if self._registered:
            self._registered = False
            torch.cuda.cudart().cudaHostUnregister(self._raw.data_ptr())


class BlockStreamer(BlockSlots):
    """The blocks streamed from source(i) through `ring` device slots of fixed address (BlockSlots).

    source(i) returns block i's bytes, laid out as block_layout(block i) (HostArena.block, or a reader's buffer: a
    tensor or anything with the buffer protocol), exactly that block's size and page-locked; it is called when block
    i's copy is issued (ring - 1 blocks ahead of its use, or on demand), and its bytes must stay valid until that
    copy has run (copied(slot)). A wrong size or pageable memory raises ValueError there.

    Nothing changes in the blocks until install(): then their tensors become views into the slots and hooks sequence
    the copies, as BlockSlots does. close() syncs, unhooks, frees the slots and gives every tensor back what it was
    before install(); install() again starts afresh. For the arena the streamer owns (packed()), self.arena."""

    def __init__(self, blocks: list[torch.nn.Module], source: Callable[[int], object], ring: int = 2,
                 device="cuda"):
        self._init(blocks, ring, device)
        self.source = source
        self.arena: HostArena | None = None
        self._orig: list[dict[str, torch.Tensor]] = []
        self._src: list = [None] * ring                   # each slot's last source buffer, kept alive
        self._info: dict[int, tuple] = {}                   # id(source object) -> (object, address, bytes)

    @classmethod
    def packed(cls, blocks: list[torch.nn.Module], ring: int = 2, device="cuda") -> "BlockStreamer":
        """Pack the blocks into a HostArena and stream them from it (installed); the arena is self.arena."""
        arena = HostArena(blocks)
        st = cls(blocks, arena.block, ring=ring, device=device)
        st.arena = arena
        st.install()
        return st

    @property
    def pinned_bytes(self) -> int:
        """The page-locked host memory the streamer holds (its arena's)."""
        return self.arena.nbytes if self.arena is not None and self.arena._registered else 0

    def _installed(self) -> None:
        self._orig = [{name: t.data for name, t in _tensors(b).items()} for b in self.blocks]
        self._slot_ptr = [t.data_ptr() for t in self.slots]
        self._src = [None] * self.ring

    def _restore(self, i: int) -> dict[str, torch.Tensor]:
        return self._orig[i]

    def _copies(self, i: int) -> list[int]:
        # Called per block in the hot loop: what a source object is (address, size, page-locked) is worked out once
        # per object; a PyTorch call here (a slice, numel()) would release the GIL (see the module docstring).
        src = self.source(i)
        e = self._info.get(id(src))
        if e is None or e[0] is not src:
            e = self._inspect(i, src)
        _, ptr, n = e
        if n != self.nbytes[i]:
            raise ValueError(f"block {i}: source gave {n} bytes; the block's layout is {self.nbytes[i]} bytes")
        s = i % self.ring
        self._src[s] = src
        return [self._slot_ptr[s], ptr, n]

    def _inspect(self, i: int, src) -> tuple:
        if isinstance(src, torch.Tensor):
            if src.device.type != "cpu" or not src.is_contiguous():
                raise ValueError(f"block {i}: source gave a {src.device} tensor, contiguous {src.is_contiguous()}; "
                                 "it must be contiguous host memory")
            ptr, n = src.data_ptr(), src.numel() * src.element_size()
        else:
            a = np.frombuffer(src, dtype=np.uint8)
            ptr, n = a.ctypes.data, a.nbytes
        if n and not self._ring_ext.host_pinned(ptr):
            raise ValueError(f"block {i}: source's bytes are not page-locked (a copy from pageable memory is "
                             "synchronous); register them (cudaHostRegister) or use a HostArena")
        if len(self._info) > 4 * self.n:                    # sources that come and go (a reader's buffers)
            self._info.clear()
        e = self._info[id(src)] = (src, ptr, n)
        return e

    def copied(self, slot: int) -> bool:
        """Whether slot's last copy has finished (its source buffer may be reused)."""
        return self._ring.copied(slot)

    def install(self) -> None:
        from vecworld.engine.ext import load_ext

        self._ring_ext = load_ext()
        super().install()

    def close(self) -> None:
        super().close()
        self._src = [None] * self.ring
        self._info.clear()
