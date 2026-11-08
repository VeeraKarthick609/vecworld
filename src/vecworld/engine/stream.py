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
import warnings
import weakref
from collections import deque
from collections.abc import Callable

import numpy as np
import torch

from vecworld.engine.slots import ALIGN, BlockSlots, _tensors, block_layout


_libc = None


def _trim() -> None:
    """malloc_trim(0): glibc keeps freed heap memory (the GGUF tensors the painter read, each under its 32 MB mmap
    threshold) mapped and counted in RSS, so without it packing would hold the old copies and the arena both (a 13.8 GB
    process, OOM-killed on this 14 GB laptop). Outside glibc it warns once and does nothing."""
    global _libc
    if _libc is None:
        import ctypes

        lib = ctypes.CDLL(None)
        _libc = getattr(lib, "malloc_trim", False)
        if not _libc:
            warnings.warn("no malloc_trim (not glibc): packing the block arena may hold the old host copies too")
    if _libc:
        _libc(0)


def _register(ptr: int, n: int) -> None:
    err = torch.cuda.cudart().cudaHostRegister(ptr, n, 0)
    if int(err) != 0:
        raise RuntimeError(f"cudaHostRegister of the {n / 1e9:.2f} GB block arena failed ({err})")


def _unregister(ptr: int) -> None:
    err = torch.cuda.cudart().cudaHostUnregister(ptr)
    if int(err) != 0:
        raise RuntimeError(f"cudaHostUnregister of the block arena failed ({err})")


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
            _trim()             # hand the block's freed host copies back to the OS before the next block's pages
        _register(self._raw.data_ptr(), self._raw.numel())
        self._registered = True
        self._blocks = [self.buf[o:o + n] for o, n in zip(self.offsets, self.sizes)]

    def block(self, i: int) -> torch.Tensor:
        """Block i's bytes (uint8, a view of the arena; the same tensor object every call)."""
        return self._blocks[i]

    @property
    def registered(self) -> bool:
        """Whether the arena is still page-locked (close() not called)."""
        return self._registered

    def close(self) -> None:
        """Unregister the arena (safe twice): call it after the streamer's close(), whose sync ends the copies."""
        if self._registered:
            self._registered = False
            _unregister(self._raw.data_ptr())


_ISSUING = -1      # BlockStreamer._ticket[i] while block i's copy is being issued (source(i) called, no ticket yet)


class BlockStreamer(BlockSlots):
    """The blocks streamed from source(i) through `ring` device slots of fixed address (BlockSlots).

    source(i) returns block i's bytes, laid out as block_layout(block i) (HostArena.block, or a reader's buffer: a
    tensor or anything with the buffer protocol), exactly that block's size and page-locked; it is called when block
    i's copy is issued (ring - 1 blocks ahead of its use, or on demand). A wrong size or pageable memory raises
    ValueError there. A source object's address must not change while the streamer may see it again (what an object
    is, is worked out once per object).

    The reader's contract (Task 10b): the bytes source(i) returned must not be changed until copy_done(i) is True.
    copy_done(i) is False from the moment source(i) is called until that copy has finished on the GPU (never True in
    between, whatever thread asks: the methods run under the GIL, and the copy's ticket is issued with its record);
    the streamer also keeps every source object alive until its copy has finished. load(i) returns the copy's ticket;
    done(ticket) answers the same for a ticket.

    Nothing changes in the blocks until install(): then their tensors become views into the slots and hooks sequence
    the copies, as BlockSlots does. close() syncs, unhooks, frees the slots and gives every tensor back what it was
    before install(); install() again starts afresh. For the arena the streamer owns (packed()), self.arena."""

    def __init__(self, blocks: list[torch.nn.Module], source: Callable[[int], object], ring: int = 2,
                 device="cuda"):
        self._init(blocks, ring, device)
        self.source = source
        self.arena: HostArena | None = None
        self._orig: list[dict[str, torch.Tensor]] = []
        self._ticket: list[int | None] = [None] * self.n    # block i's latest copy (_ISSUING while source(i) runs)
        self._pending: deque = deque()                      # (ticket, source object) of copies maybe still running
        self._src = None                                    # the source object _copies just took
        self._info: dict[int, tuple] = {}                   # id(source) -> (weakref to it, address, bytes)

    @classmethod
    def packed(cls, blocks: list[torch.nn.Module], ring: int = 2, device="cuda") -> "BlockStreamer":
        """Pack the blocks into a HostArena and stream them from it (installed); the arena is self.arena."""
        arena = HostArena(blocks)
        try:
            st = cls(blocks, arena.block, ring=ring, device=device)
            st.arena = arena
            st.install()
        except BaseException:
            arena.close()                                   # nobody else owns it yet: never leave it registered
            raise
        return st

    @property
    def pinned_bytes(self) -> int:
        """The page-locked host memory the streamer holds (its arena's)."""
        return self.arena.nbytes if self.arena is not None and self.arena.registered else 0

    def _prepare_copies(self) -> None:
        self._orig = [{name: t.data for name, t in _tensors(b).items()} for b in self.blocks]
        self._slot_ptr = [t.data_ptr() for t in self.slots]
        self._ticket = [None] * self.n
        self._pending.clear()

    def _restore(self, i: int) -> dict[str, torch.Tensor]:
        return self._orig[i]

    def _copies(self, i: int) -> list[int]:
        # Called per block in the hot loop: what a source object is (address, size, page-locked) is worked out once
        # per object; a PyTorch call here (a slice, numel()) would release the GIL (see the module docstring).
        src = self.source(i)
        e = self._info.get(id(src))
        if e is None or e[0]() is not src:
            e = self._inspect(i, src)
        _, ptr, n = e
        if n != self.nbytes[i]:
            raise ValueError(f"block {i}: source gave {n} bytes; the block's layout is {self.nbytes[i]} bytes")
        self._src = src
        return [self._slot_ptr[i % self.ring], ptr, n]

    def _inspect(self, i: int, src) -> tuple:
        if isinstance(src, torch.Tensor):
            if src.device.type != "cpu" or not src.is_contiguous():
                raise ValueError(f"block {i}: source gave a {src.device} tensor, contiguous {src.is_contiguous()}; "
                                 "it must be contiguous host memory")
            ptr, n = src.data_ptr(), src.numel() * src.element_size()
        else:
            a = np.frombuffer(src, dtype=np.uint8)
            ptr, n = a.ctypes.data, a.nbytes
        if n and not self._ext.host_pinned(ptr, n):
            raise ValueError(f"block {i}: source's bytes are not page-locked (a copy from pageable memory is "
                             "synchronous); register them (cudaHostRegister) or use a HostArena")
        try:
            ref = weakref.ref(src)                          # never keeps a source alive (a reader's buffers)
        except TypeError:                                   # (a memoryview: worked out again each time, numpy only)
            return (lambda: src), ptr, n
        if len(self._info) > 4 * self.n:                    # sources that come and go
            self._info = {k: v for k, v in self._info.items() if v[0]() is not None}
        e = self._info[id(src)] = (ref, ptr, n)
        return e

    def load(self, i: int) -> int:
        prev, self._ticket[i] = self._ticket[i], _ISSUING
        try:
            ticket = super().load(i)
        except BaseException:
            self._ticket[i] = prev
            self._src = None
            raise
        self._ticket[i] = ticket
        self._pending.append((ticket, self._src))
        self._src = None
        while self._pending and self._ring.done(self._pending[0][0]):
            self._pending.popleft()                        # finished copies: their sources may go
        return ticket

    def _check_open(self) -> None:
        if self._ring is None:
            raise RuntimeError("BlockStreamer is closed (install() it first)")

    def done(self, ticket: int) -> bool:
        """Whether the copy load() returned `ticket` for has finished reading its source."""
        self._check_open()
        return self._ring.done(ticket)

    def copy_done(self, i: int) -> bool:
        """Whether block i's latest copy has finished reading the bytes source(i) gave (True if block i was never
        copied): False from the call of source(i) on, until the GPU has done that copy."""
        self._check_open()
        t = self._ticket[i]
        if t is None:
            return True
        return t != _ISSUING and self._ring.done(t)

    def close(self) -> None:
        super().close()
        self._pending.clear()
        self._info.clear()
