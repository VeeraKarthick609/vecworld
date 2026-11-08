r"""The SSD reader: the FLUX blocks' weights read straight from the GGUF file into page-locked staging buffers with
O_DIRECT reads through io_uring, so they never stay resident in RAM (Task 10b, docs/engine/11-io-uring.md).

    GGUF on the SSD   [ ... | block i: its tensors back to back (191 MB / 80 MB) | block i+1 | ... ]
                          |  one O_DIRECT read per block (its 4096-aligned superset), in 1 MB chunks, io_uring
                          v
    staging ring      [ buffer 0 ][ buffer 1 ][ buffer 2 ]          page-locked, block i in whichever is free
                          |  per tensor: one DMA copy (a Gather), file order -> slot order, + the resident biases
                          v
    device slots      [ slot 0 ][ slot 1 ][ slot 2 ]                engine.stream.BlockStreamer, as in Task 10a

- DirectReader (csrc/uring.cpp): io_uring on the raw syscalls (no liburing headers here), O_DIRECT, one thread.
- flux_file_layout(blocks, gguf_path): where each block's packed Q4_K weights lie in the file. diffusers renames the
  file's tensors and splits a double block's qkv and a single block's linear1 by rows (FLUX_DOUBLE / FLUX_SINGLE); a
  Q4_K row is a contiguous run of bytes, so every weight is one range of the file. The rest of a block (biases and
  norm weights, converted F32 -> bf16 on load: ~6 MB in all) stays resident.
- SsdSource(path, layout): the BlockStreamer source. source(i) makes sure block i is read (reading block i+1 .. i+ahead
  behind it), and hands out a Gather; a staging buffer whose block was handed out is read into again only once the
  streamer says that block's copy has run (copy_done, the reader's contract in docs/engine/10-streamer.md).
"""
import time
import weakref
from dataclasses import dataclass, field

import torch

from vecworld.engine.slots import ALIGN as SLOT_ALIGN   # resident tensors keep the slots' alignment in the residue

ALIGN = 4096            # O_DIRECT: offsets, lengths and addresses in multiples of this (the NVMe's block is 512)


def _ext():
    from vecworld.engine.ext import load_uring

    return load_uring()


class DirectReader:
    """O_DIRECT reads of one file through io_uring (raw syscalls). read_into(offset, nbytes, dst) reads exactly
    file[offset:offset + nbytes] into dst (blocking; any range: an unaligned one goes through an aligned superset);
    submit(offset, nbytes, need, dst_ptr) / done(id) / wait(id) are the asynchronous form, for 4096-aligned ranges
    and addresses. A short read (the file ends first) raises IOError naming the offset. queue_depth: chunks of
    `chunk` bytes in the kernel at once. One thread at a time; close() waits for reads in flight (safe twice)."""

    def __init__(self, path: str, queue_depth: int = 8, chunk: int = 1 << 20, direct: bool | None = None):
        # direct=False opens without O_DIRECT (buffered, through the page cache): a test switch only, to show the
        # page-cache checks can fail. None: FLUX_SSD_BUFFERED_TEST=1 in the environment turns it off, else on.
        if direct is None:
            import os

            direct = os.environ.get("FLUX_SSD_BUFFERED_TEST") != "1"
        self._r = _ext().DirectReader(str(path), queue_depth, chunk, direct)
        self.path = str(path)

    def read_into(self, offset: int, nbytes: int, dst: torch.Tensor) -> None:
        self._r.read_into(offset, nbytes, dst)

    def submit(self, offset: int, nbytes: int, need: int, dst: int) -> int:
        return self._r.submit(offset, nbytes, need, dst)

    def done(self, req: int) -> bool:
        return self._r.done(req)

    def wait(self, req: int) -> None:
        self._r.wait(req)

    @property
    def size(self) -> int:
        return self._r.size

    @property
    def inflight(self) -> int:
        """Chunks in the kernel now."""
        return self._r.inflight

    @property
    def queued(self) -> int:
        """Chunks waiting here for room in the kernel's queue."""
        return self._r.queued

    @property
    def completions(self) -> int:
        """Chunk completions reaped so far."""
        return self._r.completions

    def _inject_fault(self, nth: int, res: int) -> None:
        """Test hook: the nth chunk completion from now is reported as `res` (-errno, or a short byte count)."""
        self._r._inject_fault(nth, res)

    def _inject_enter_fault(self, err: int) -> None:
        """Test hook: the next io_uring_enter that reaps completions fails with errno err."""
        self._r._inject_enter_fault(err)

    def close(self) -> None:
        self._r.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --- where the FLUX blocks' weights lie in the GGUF ---------------------------------------------------------------------

H = 3072
# diffusers module path of a block's Q4_K linear -> (the file tensor's name under the block, first row, rows). A double
# block's img_attn.qkv / txt_attn.qkv split into q, k, v; a single block's linear1 into q, k, v and the MLP's input.
FLUX_DOUBLE = {
    "norm1.linear": ("img_mod.lin", 0, 6 * H),
    "norm1_context.linear": ("txt_mod.lin", 0, 6 * H),
    "attn.to_q": ("img_attn.qkv", 0, H),
    "attn.to_k": ("img_attn.qkv", H, H),
    "attn.to_v": ("img_attn.qkv", 2 * H, H),
    "attn.add_q_proj": ("txt_attn.qkv", 0, H),
    "attn.add_k_proj": ("txt_attn.qkv", H, H),
    "attn.add_v_proj": ("txt_attn.qkv", 2 * H, H),
    "attn.to_out.0": ("img_attn.proj", 0, H),
    "attn.to_add_out": ("txt_attn.proj", 0, H),
    "ff.net.0.proj": ("img_mlp.0", 0, 4 * H),
    "ff.net.2": ("img_mlp.2", 0, H),
    "ff_context.net.0.proj": ("txt_mlp.0", 0, 4 * H),
    "ff_context.net.2": ("txt_mlp.2", 0, H),
}
FLUX_SINGLE = {
    "norm.linear": ("modulation.lin", 0, 3 * H),
    "attn.to_q": ("linear1", 0, H),
    "attn.to_k": ("linear1", H, H),
    "attn.to_v": ("linear1", 2 * H, H),
    "proj_mlp": ("linear1", 3 * H, 4 * H),
    "proj_out": ("linear2", 0, H),
}
N_DOUBLE, N_SINGLE = 19, 38


def _block_name(k: int) -> tuple[str, int, dict]:
    return ("double_blocks", k, FLUX_DOUBLE) if k < N_DOUBLE else ("single_blocks", k - N_DOUBLE, FLUX_SINGLE)


def gguf_meta(path) -> dict[str, tuple[int, int, str, list[int]]]:
    """name -> (file offset, bytes, quant type name, shape in GGUF order (inner first)); metadata only."""
    import gguf

    r = gguf.GGUFReader(str(path))
    return {t.name: (int(t.data_offset), int(t.n_bytes), t.tensor_type.name, [int(x) for x in t.shape])
            for t in r.tensors}


def flux_block_extents(meta) -> list[tuple[int, int]]:
    """(first byte, end) of each of the 57 blocks' tensors in the file, in the transformer's block order."""
    out = []
    for k in range(N_DOUBLE + N_SINGLE):
        kind, b, _ = _block_name(k)
        mine = [v for name, v in meta.items() if name.startswith(f"{kind}.{b}.")]
        out.append((min(v[0] for v in mine), max(v[0] + v[1] for v in mine)))
    return out


@dataclass
class BlockFile:
    """How block i comes off the disk: read file bytes [lo, lo + nbytes) (4096-aligned; the first `need` of them must
    be there), then copy (slot offset, offset in what was read, bytes) for each weight, and the resident tensors
    (slot offset, tensor) from RAM. A block whose tensors lie in several runs of the file (T5's GGUF orders its
    tensors by name: flux.text.t5_file_layout) gives `extents` instead: (file offset, bytes, need, offset in the
    buffer) per run, each read like the single range (lo and nbytes then span the buffer)."""
    lo: int
    nbytes: int
    need: int
    copies: list[tuple[int, int, int]]
    resident: list[tuple[int, torch.Tensor]] = field(default_factory=list)
    extents: list[tuple[int, int, int, int]] | None = None

    def runs(self) -> list[tuple[int, int, int, int]]:
        """(file offset, bytes, need, offset in the buffer) of each read."""
        return list(self.extents) if self.extents is not None else [(self.lo, self.nbytes, self.need, 0)]


def flux_file_layout(blocks: list[torch.nn.Module], gguf_path) -> list[BlockFile]:
    """Each FLUX block's BlockFile: its Q4KLinears' packed weights (qweight) found in the GGUF by FLUX_DOUBLE /
    FLUX_SINGLE (checked against the file's types, shapes and sizes: a mismatch raises ValueError), every other tensor
    of the block resident (its current values). Reads the file's metadata only."""
    from vecworld.engine.slots import _tensors, block_layout

    meta = gguf_meta(gguf_path)
    if len(blocks) != N_DOUBLE + N_SINGLE:
        raise ValueError(f"flux_file_layout: {len(blocks)} blocks, FLUX has {N_DOUBLE} + {N_SINGLE}")
    out = []
    for k, blk in enumerate(blocks):
        kind, b, table = _block_name(k)
        lay, _ = block_layout(blk)
        file_copies, resident = [], []
        for name, t in _tensors(blk).items():
            nb = t.numel() * t.element_size()
            if not name.endswith(".qweight"):
                resident.append((lay[name], t.data))
                continue
            mod = name[:-len(".qweight")]
            if mod not in table:
                raise ValueError(f"block {k} ({kind}.{b}): no file tensor known for {mod}")
            suffix, row0, rows = table[mod]
            gname = f"{kind}.{b}.{suffix}.weight"
            if gname not in meta:
                raise ValueError(f"block {k} {mod}: {gname} is not in {gguf_path}")
            off, n, typ, shape = meta[gname]
            bpr = shape[0] // 256 * 144                     # Q4_K: 144 bytes per 256 weights of a row
            if (typ != "Q4_K" or t.dtype != torch.uint8 or t.dim() != 2 or tuple(t.shape) != (rows, bpr)
                    or (row0 + rows) * bpr > n or rows * bpr != nb):
                raise ValueError(f"block {k} {mod}: {t.dtype} {tuple(t.shape)} doesn't match rows {row0}..{row0 + rows}"
                                 f" of {gname} ({typ}, {shape}, {n} bytes)")
            file_copies.append((lay[name], off + row0 * bpr, nb))
        if not file_copies:
            raise ValueError(f"block {k}: no Q4_K weights (VECWORLD_KERNEL=off streams through group offload)")
        lo, hi = min(f for _, f, _ in file_copies), max(f + n for _, f, n in file_copies)
        alo, ahi = lo // ALIGN * ALIGN, -(-hi // ALIGN) * ALIGN
        out.append(BlockFile(lo=alo, nbytes=ahi - alo, need=hi - alo,
                             copies=[(so, f - alo, n) for so, f, n in file_copies], resident=resident))
    return out


# --- the streamer's source ----------------------------------------------------------------------------------------------

def _aligned_host(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(the allocation, a 4096-aligned view of n bytes in it)."""
    raw = torch.empty(n + ALIGN, dtype=torch.uint8)
    skip = -raw.data_ptr() % ALIGN
    return raw, raw[skip:skip + n]


class SsdSource:
    """BlockStreamer's source for blocks read from the SSD (see the module docstring).

    `buffers` page-locked staging buffers, each the size of the largest block's read; source(i) returns an
    engine.stream.Gather (the copies that put block i in its slot), after making sure block i's read has finished
    (waiting for it with the GIL released if it hasn't) and starting the reads of blocks i+1 .. i+ahead (cyclically:
    passes repeat) into buffers that are free. A buffer is free when its read has finished and the block last handed
    out from it has been copied (bind(streamer): its copy_done). A block still in a buffer is handed out again without
    a read. pin=False skips page-locking (tests without a GPU). wrap=False: one pass, no read-ahead past the last
    block. stats: reads, bytes read, seconds waited.

    One thread (the streamer's). close() waits for the reads in flight, then unregisters the buffers (safe twice);
    call it after the streamer's close()."""

    def __init__(self, path: str, layout: list[BlockFile], buffers: int = 3, ahead: int = 2, queue_depth: int = 64,
                 chunk: int = 1 << 20, pin: bool = True, wrap: bool = True):
        from vecworld.engine.stream import Gather

        if buffers < 2:
            raise ValueError(f"SsdSource needs 2 or more staging buffers, got {buffers}")
        self._Gather = Gather
        self.path, self.n = str(path), len(layout)
        self.buffers, self.ahead, self.wrap = buffers, max(0, min(ahead, buffers - 1)), wrap
        self._plans = [(b.runs(), list(b.copies)) for b in layout]
        self.buf_bytes = -(-max(bo + n for runs, _ in self._plans for _, n, _, bo in runs) // ALIGN) * ALIGN
        # the resident tensors, packed into one buffer (copied: the blocks' own tensors can then be freed)
        res, off = [], 0
        for b in layout:
            mine = []
            for so, t in b.resident:
                nb = t.numel() * t.element_size()
                mine.append((so, off, nb, t))
                off += -(-nb // SLOT_ALIGN) * SLOT_ALIGN
            res.append(mine)
        self._res_raw, self._res = _aligned_host(max(off, ALIGN))
        for mine in res:
            for _, o, nb, t in mine:
                if nb:
                    self._res[o:o + nb].copy_(t.detach().contiguous().view(-1).view(torch.uint8))
        self._res_copies = [[(so, o, nb) for so, o, nb, _ in mine if nb] for mine in res]
        del res
        self._stage_raw, self._stage = _aligned_host(buffers * self.buf_bytes)
        self._pinned: list[int] = []
        self._reader = None
        try:
            if pin:
                from vecworld.engine.stream import _register

                for t in (self._stage, self._res):
                    _register(t.data_ptr(), t.numel())
                    self._pinned.append(t.data_ptr())
            self._reader = DirectReader(self.path, queue_depth=queue_depth, chunk=chunk)
        except BaseException:
            self._unpin()
            raise
        # Collected without close(): the reads in flight are waited for before the buffers are unpinned and freed (the
        # finalizer holds the buffers, so they outlive the reader's close()).
        self._finalizer = weakref.finalize(self, SsdSource._release, self._reader, self._pinned,
                                           (self._stage_raw, self._res_raw))
        base = self._stage.data_ptr()
        self._buf = [base + k * self.buf_bytes for k in range(buffers)]
        self._res_ptr = self._res.data_ptr()
        self._block: list[int | None] = [None] * buffers      # the block buffer k holds (or is reading)
        self._req: list[list[int] | None] = [None] * buffers  # buffer k's reads in flight (one per run)
        self._handed: list[int | None] = [None] * buffers     # the block handed out from k, its copy maybe pending
        self._where: dict[int, int] = {}                      # block -> the buffer that holds it
        self._gathers: dict[tuple[int, int], object] = {}
        self._st = None
        self.stats = dict(calls=0, reads=0, bytes=0, hits=0, read_wait_s=0.0, buffer_wait_s=0.0)

    @property
    def pinned_bytes(self) -> int:
        return (self._stage.numel() + self._res.numel()) if self._pinned else 0

    def bind(self, streamer) -> None:
        """The streamer whose copy_done(i) says when a buffer handed out for block i may be read into again."""
        self._st = streamer

    def buffer_of(self, i: int) -> int | None:
        return self._where.get(i)

    def buffer_block(self, k: int) -> int | None:
        return self._block[k]

    def __call__(self, i: int):
        self.stats["calls"] += 1
        k = self._where.get(i)
        if k is None:
            k = self._claim(i, urgent=True)
            self._start(k, i)
        else:
            self.stats["hits"] += 1
        for m in range(1, self.ahead + 1):                  # read ahead, behind block i's own read
            if not self.wrap and i + m >= self.n:
                break                                       # one pass: nothing after the last block
            j = (i + m) % self.n
            if j in self._where:
                continue
            kk = self._claim(i, urgent=False)
            if kk is None:
                break
            self._start(kk, j)
        if self._req[k] is not None:
            try:
                self._wait_all(k)                           # GIL released while the SSD catches up
            except OSError:
                self._forget(k)                             # the next source(i) reads the block again
                raise
            self._req[k] = None
        self._handed[k] = i
        g = self._gathers.get((i, k))
        if g is None:
            buf = self._buf[k]
            flat = []
            for so, bo, n in self._plans[i][1]:
                flat += [so, buf + bo, n]
            for so, o, n in self._res_copies[i]:
                flat += [so, self._res_ptr + o, n]
            g = self._gathers[(i, k)] = self._Gather(flat, self)
        return g

    def _start(self, k: int, i: int) -> None:
        old = self._block[k]
        if old is not None and self._where.get(old) == k:
            del self._where[old]
        runs, _ = self._plans[i]
        self._block[k], self._handed[k] = i, None
        reqs = self._req[k] = []                            # kept as they go: a later submit failing leaves them known
        for lo, nbytes, need, bo in runs:
            reqs.append(self._reader.submit(lo, nbytes, need, self._buf[k] + bo))
            self.stats["bytes"] += nbytes
        self._where[i] = k
        self.stats["reads"] += 1

    def _wait_all(self, k: int) -> None:
        """Wait for every read into buffer k; if one failed, the others are still waited for (the buffer is then
        free of reads in flight) before its error is raised."""
        err = None
        for req in self._req[k]:
            try:
                if not self._reader.done(req):
                    t = time.perf_counter()
                    self._reader.wait(req)
                    self.stats["read_wait_s"] += time.perf_counter() - t
            except OSError as e:
                err = err or e
        if err is not None:
            raise err

    def _forget(self, k: int) -> None:
        """Buffer k holds nothing (its read failed: every chunk of it has completed or been dropped)."""
        blk = self._block[k]
        if blk is not None and self._where.get(blk) == k:
            del self._where[blk]
        self._block[k] = self._req[k] = self._handed[k] = None

    def _usable(self, k: int) -> bool:
        reqs = self._req[k]
        if reqs is not None:
            try:
                if not all(self._reader.done(req) for req in reqs):
                    return False
            except OSError:                                 # a read-ahead that failed: nobody wants it yet
                try:
                    self._wait_all(k)                       # its other runs' reads end first
                except OSError:
                    pass
                self._forget(k)
                return True
            self._req[k] = None
        h = self._handed[k]
        if h is not None:
            if self._st is None or not self._st.copy_done(h):
                return False
            self._handed[k] = None
        return True

    def _claim(self, i: int, urgent: bool) -> int | None:
        """A buffer to read into, for block i (urgent) or for a block after it (read-ahead). Never one whose read is in
        flight or whose handed-out block's copy may be pending; of the rest, the one holding the block needed furthest
        ahead (empty first). Read-ahead never takes one holding blocks i .. i+ahead (None instead); block i waits for
        a buffer if it must."""
        t0 = None
        while True:
            best, bestd = None, -1
            for k in range(self.buffers):
                if not self._usable(k):
                    continue
                blk = self._block[k]
                d = self.n + 1 if blk is None else (blk - i) % self.n
                if d > bestd:
                    best, bestd = k, d
            if best is not None and (urgent or bestd > self.ahead):
                if t0 is not None:
                    self.stats["buffer_wait_s"] += time.perf_counter() - t0
                return best
            if not urgent:
                return None
            if self._st is None:
                raise RuntimeError("SsdSource: bind(streamer) first: every buffer is handed out, and only the "
                                   "streamer's copy_done can say when one may be read into again")
            if t0 is None:
                t0 = time.perf_counter()
            time.sleep(0.0002)                              # a copy (GPU) or a read (SSD) to finish; GIL released

    @staticmethod
    def _unpin_all(pinned: list[int]) -> None:
        if pinned:
            from vecworld.engine.stream import _unregister

            while pinned:
                _unregister(pinned.pop())

    def _unpin(self) -> None:
        SsdSource._unpin_all(self._pinned)

    @staticmethod
    def _release(reader, pinned: list[int], buffers) -> None:
        """Close the reader (it waits for the reads in flight), then unpin; `buffers` are held until then. If the
        reader can't be closed, the buffers stay pinned rather than be freed under a read."""
        if reader is not None:
            reader.close()
        SsdSource._unpin_all(pinned)

    def close(self) -> None:
        """Wait for the reads in flight and unregister the buffers (safe twice)."""
        try:
            self._finalizer()
        finally:
            self._gathers.clear()
            self._where.clear()
            self._st = None


def ssd_source(gguf_path, layout: list[BlockFile], **kw) -> SsdSource:
    """The BlockStreamer source reading `layout`'s blocks from gguf_path (SsdSource; kw: buffers, ahead, queue_depth,
    chunk). Bind it to the streamer (source.bind(streamer)) before the first block runs."""
    return SsdSource(str(gguf_path), layout, **kw)


def drop_file_cache(root) -> int:
    """Ask the kernel to drop the clean page cache of every regular file under root (posix_fadvise DONTNEED; symlinks
    skipped; no sudo, nothing deleted). Returns the size of the files advised (not what was dropped: pages still mapped
    or dirty stay). The game calls it on the model cache once its models
    are loaded: the loads leave ~4.5 GB of model files cached, and with that much clean cache beside the painter and
    the dreamer the kernel was seen swapping 0.48 GB of the server out at the first walk step (MemAvailable 5.3 GB);
    started without it, a walk swapped nothing (docs/engine/11-io-uring.md)."""
    import os
    from pathlib import Path

    total = 0
    root = Path(root)
    if not root.exists():
        return 0
    for p in root.rglob("*"):
        try:
            if p.is_symlink() or not p.is_file():
                continue
            fd = os.open(p, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
            total += p.stat().st_size
        except OSError:
            continue
    return total


def resident_bytes(path) -> int:
    """How many bytes of the file are in the page cache now (its pages, by mmap + mincore; the last page counted whole).
    Reads nothing: mapping a file and asking mincore doesn't fault pages in."""
    import ctypes
    import mmap
    import os

    libc = ctypes.CDLL(None, use_errno=True)
    libc.mmap.restype = ctypes.c_void_p
    libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
    libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
    size = os.path.getsize(path)
    if size == 0:
        return 0
    fd = os.open(path, os.O_RDONLY)
    try:
        addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
        if addr in (None, ctypes.c_void_p(-1).value):
            raise OSError(ctypes.get_errno(), f"mmap of {path} failed")
        try:
            page = mmap.PAGESIZE
            vec = ctypes.create_string_buffer((size + page - 1) // page)
            if libc.mincore(addr, size, vec) != 0:
                raise OSError(ctypes.get_errno(), f"mincore of {path} failed")
            return sum(b & 1 for b in vec.raw) * page
        finally:
            libc.munmap(addr, size)
    finally:
        os.close(fd)
