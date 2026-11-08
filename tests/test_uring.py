"""The SSD reader: O_DIRECT reads through io_uring (engine.uring, csrc/uring.cpp; Task 10b, docs/engine/11-io-uring.md).

The reader's own tests run on the CPU against small files written to the SSD (O_DIRECT needs a real block device:
/tmp is tmpfs here), removed afterwards. The FLUX file layout is checked against the GGUF's metadata (no weights read).
GPU tests (the streamer fed from the SSD, the painter's picture) are in test_uring_gpu.py.
"""
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

ALIGN = 4096


@pytest.fixture
def ssd_dir():
    d = Path.home() / ".cache/vecworld/test-uring"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _file(d: Path, n: int, seed: int = 0) -> tuple[Path, bytes]:
    data = np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()
    p = d / f"f{n}-{seed}.bin"
    p.write_bytes(data)
    return p, data


def _aligned(n: int) -> torch.Tensor:
    raw = torch.empty(n + ALIGN, dtype=torch.uint8)
    skip = -raw.data_ptr() % ALIGN
    return raw[skip:skip + n]


def _pread(path, off, n) -> bytes:
    fd = os.open(path, os.O_RDONLY)
    try:
        return os.pread(fd, n, off)
    finally:
        os.close(fd)


def test_uring_reads_exact_bytes(ssd_dir):
    """read_into(offset, nbytes, dst) puts exactly os.pread's bytes in dst[:nbytes]; submit / wait, the asynchronous
    form, reads aligned ranges into an aligned buffer, split in chunks with more chunks than the queue holds."""
    from vecworld.engine.uring import DirectReader

    path, data = _file(ssd_dir, 5 * 2**20)
    r = DirectReader(str(path), queue_depth=4, chunk=64 << 10)
    try:
        for off, n in [(0, ALIGN), (0, len(data)), (3 * ALIGN, 40 * ALIGN), (2**20, 2**20 + 8 * ALIGN)]:
            dst = _aligned(n)
            r.read_into(off, n, dst)
            assert bytes(dst.numpy()) == _pread(path, off, n), (off, n)
        dst = _aligned(len(data))
        ids = [r.submit(o, 2**20, 2**20, dst.data_ptr() + o) for o in range(0, len(data), 2**20)]
        for i in ids:
            r.wait(i)
        assert all(r.done(i) for i in ids)
        assert bytes(dst.numpy()) == data
        real = os.environ.get("VECWORLD_TEST_GGUF")
        if real is None:
            from vecworld.config import FLUX_GGUF

            real = str(FLUX_GGUF)
        if os.path.exists(real):                            # the real file, read-only: an unaligned range in its middle
            g = DirectReader(real, queue_depth=8)
            off, n = 54048 + 12345, 3 * 2**20 + 77
            dst = _aligned(n)
            g.read_into(off, n, dst)
            assert bytes(dst.numpy()) == _pread(real, off, n)
            g.close()
    finally:
        r.close()
        r.close()                                           # safe twice


def test_uring_unaligned_tail(ssd_dir):
    """A range whose start or length isn't 4096-aligned (a block's first tensor in the GGUF starts at byte 54048), or
    that ends at a file end that isn't aligned, reads correctly via an aligned superset; a misaligned submit() (the
    raw form) is refused."""
    from vecworld.engine.uring import DirectReader

    path, data = _file(ssd_dir, 3 * 2**20 + 1234)          # the file's own end is unaligned
    r = DirectReader(str(path), queue_depth=8, chunk=256 << 10)
    try:
        for off, n in [(1, 10), (54048, 100_000), (ALIGN - 1, 2), (ALIGN, 5000), (777, len(data) - 777),
                       (len(data) - 3000, 3000), (0, len(data))]:
            dst = torch.empty(n, dtype=torch.uint8)         # any CPU buffer: unaligned ones go through a bounce buffer
            r.read_into(off, n, dst)
            assert bytes(dst.numpy()) == data[off:off + n], (off, n)
        dst = _aligned(2 * ALIGN)
        for off, n in [(1, ALIGN), (0, 100)]:
            with pytest.raises(ValueError, match="4096"):
                r.submit(off, n, n, dst.data_ptr())
        with pytest.raises(ValueError, match="4096"):
            r.submit(0, ALIGN, ALIGN, dst.data_ptr() + 1)
    finally:
        r.close()


def test_uring_short_read_raises(ssd_dir):
    """A file that ends before the bytes asked for (truncated) raises IOError naming the offset, from read_into and
    from wait / done (every time it is asked); the reader keeps working after it."""
    from vecworld.engine.uring import DirectReader

    path, data = _file(ssd_dir, 2**20)
    os.truncate(path, 300_000)
    r = DirectReader(str(path), queue_depth=8, chunk=64 << 10)
    try:
        with pytest.raises(IOError, match=r"offset \d+"):
            r.read_into(200_000, 200_000, torch.empty(200_000, dtype=torch.uint8))
        dst = _aligned(2**20)
        i = r.submit(0, 2**20, 2**20, dst.data_ptr())
        with pytest.raises(IOError, match=r"short read at offset 3\d\d\d\d\d"):
            r.wait(i)
        with pytest.raises(IOError, match="short read"):
            r.done(i)
        out = torch.empty(1000, dtype=torch.uint8)
        r.read_into(299_000, 1000, out)                      # the end of what is there still reads
        assert bytes(out.numpy()) == data[299_000:300_000]
    finally:
        r.close()
    with pytest.raises(IOError, match="closed"):
        r.read_into(0, 10, torch.empty(10, dtype=torch.uint8))


@pytest.mark.parametrize("qd, nth, res, what", [(1, 1, -5, "Input/output error"), (1, 3, -5, "Input/output error"),
                                                (4, 3, -5, "Input/output error"), (1, 2, 1000, "short read"),
                                                (4, 16, 0, "short read")])
def test_uring_failed_chunk_fails_the_whole_request(ssd_dir, qd, nth, res, what):
    """A request whose chunk fails (an injected EIO, or a torn / empty read in the middle of the file) fails as a whole:
    wait() raises naming the offset, and only once every chunk of it has completed or been dropped, so nothing of it is
    in the kernel or queued afterwards and no completion can still land in its buffer; done() raises the same, every
    time. 16 chunks through 1 or 4 queue slots: most of the request is still queued when the chunk fails (with one
    slot the failing completion is the only one in flight, the case that returned as if the read had succeeded)."""
    from vecworld.engine.uring import DirectReader

    path, data = _file(ssd_dir, 2**20)
    r = DirectReader(str(path), queue_depth=qd, chunk=64 << 10)
    try:
        dst = _aligned(2**20)
        i = r.submit(0, 2**20, 2**20, dst.data_ptr())
        r._inject_fault(nth, res)
        with pytest.raises(IOError, match=rf"{what}") as ei:
            r.wait(i)
        assert "offset" in str(ei.value)
        assert r.inflight == 0 and r.queued == 0
        seen = r.completions
        with pytest.raises(IOError, match=what):
            r.done(i)
        assert r.completions == seen and r.inflight == 0 and r.queued == 0   # no completion arrives afterwards
        out = torch.empty(5000, dtype=torch.uint8)
        r.read_into(100, 5000, out)                     # the reader still works
        assert bytes(out.numpy()) == data[100:5100]
    finally:
        r.close()


def test_uring_partial_read_is_read_again(ssd_dir):
    """O_DIRECT may return fewer bytes than asked mid-file (a multiple of the block size): the rest is read again and
    the request succeeds with every byte in place."""
    from vecworld.engine.uring import DirectReader

    path, data = _file(ssd_dir, 2**20)
    r = DirectReader(str(path), queue_depth=1, chunk=64 << 10)
    try:
        dst = _aligned(2**20)
        i = r.submit(0, 2**20, 2**20, dst.data_ptr())
        r._inject_fault(2, 8192)
        r.wait(i)
        assert bytes(dst.numpy()) == data and r.completions == 17
    finally:
        r.close()


def test_uring_missing_file_raises(ssd_dir):
    from vecworld.engine.uring import DirectReader

    with pytest.raises(IOError, match="O_DIRECT"):
        DirectReader(str(ssd_dir / "nope.bin"))


# --- the FLUX file layout ----------------------------------------------------------------------------------------------

def _gguf_meta():
    import gguf

    from vecworld.config import FLUX_GGUF

    if not FLUX_GGUF.exists():
        pytest.skip("no FLUX GGUF here")
    r = gguf.GGUFReader(str(FLUX_GGUF))
    return {t.name: (int(t.data_offset), int(t.n_bytes), t.tensor_type.name, [int(x) for x in t.shape]) for t in r.tensors}


def test_flux_map_covers_every_block_weight_once():
    """The map from the model's block linears to the file's tensors (engine.uring.FLUX_DOUBLE / FLUX_SINGLE: diffusers
    renames them and splits a double block's qkv and a single block's linear1 by rows) covers every Q4_K weight of every
    block in the GGUF exactly once, row ranges tiling each tensor, and nothing else."""
    from vecworld.engine.uring import FLUX_DOUBLE, FLUX_SINGLE

    meta = _gguf_meta()
    for kind, n, table in (("double_blocks", 19, FLUX_DOUBLE), ("single_blocks", 38, FLUX_SINGLE)):
        for b in range(n):
            q4k = {k for k, v in meta.items() if k.startswith(f"{kind}.{b}.") and v[2] == "Q4_K"}
            rows: dict[str, list] = {}
            for mod, (suffix, row0, nrows) in table.items():
                rows.setdefault(f"{kind}.{b}.{suffix}.weight", []).append((row0, nrows))
            assert set(rows) == q4k, (kind, b, sorted(set(rows) ^ q4k))
            for name, rr in rows.items():
                rr.sort()
                assert rr[0][0] == 0 and all(a[0] + a[1] == c[0] for a, c in zip(rr, rr[1:])), (name, rr)
                assert rr[-1][0] + rr[-1][1] == meta[name][3][1], (name, rr, meta[name][3])   # gguf shape: (in, out)


def test_block_extents_are_contiguous_in_the_file():
    """Every block's tensors are one contiguous run of the file, the blocks in order (what lets a block be one read)."""
    from vecworld.engine.uring import flux_block_extents

    meta = _gguf_meta()
    ext = flux_block_extents(meta)
    assert len(ext) == 57
    for k, (lo, hi) in enumerate(ext):
        kind, b = ("double_blocks", k) if k < 19 else ("single_blocks", k - 19)
        mine = [v for name, v in meta.items() if name.startswith(f"{kind}.{b}.")]
        assert lo == min(v[0] for v in mine) and hi == max(v[0] + v[1] for v in mine)
        assert sum(v[1] for v in mine) == hi - lo, k                  # no gaps, nothing else inside


# --- the reader's staging ring, without a GPU --------------------------------------------------------------------------

class _FakeStreamer:
    """copy_done(i) as the streamer answers it, driven by the test: block i's copy is done once released."""

    def __init__(self):
        self.pending: set[int] = set()

    def copy_done(self, i):
        return i not in self.pending


def _toy_layout(path: Path, sizes: list[int], start: int = 54048):
    """BlockFiles for a file of blocks back to back from `start`, each block one tensor copied to slot offset 0, plus
    one resident tensor at the slot's end."""
    from vecworld.engine.uring import BlockFile

    blocks, off = [], start
    for k, n in enumerate(sizes):
        lo, hi = off // ALIGN * ALIGN, -(-(off + n) // ALIGN) * ALIGN
        res = torch.full((64,), k, dtype=torch.uint8)
        blocks.append(BlockFile(lo=lo, nbytes=hi - lo, need=off + n - lo, copies=[(0, off - lo, n)],
                                resident=[(-(-n // 256) * 256, res)]))
        off += n
    return blocks


def _gathered(g) -> dict[int, bytes]:
    """slot offset -> the bytes a Gather's copies would send there."""
    import ctypes

    out = {}
    for k in range(0, len(g.copies), 3):
        so, ptr, n = g.copies[k:k + 3]
        out[so] = ctypes.string_at(ptr, n)
    return out


def test_ssd_source_hands_out_each_block_and_reuses_buffers_only_after_their_copy(ssd_dir):
    """SsdSource(path, layout)(i) gives a Gather whose copies send block i's file bytes (and its resident tensors) to their
    slot offsets. It reads ahead into a ring of staging buffers, and never reads into a buffer whose block was handed out
    until the streamer says that block's copy is done (copy_done): with every copy held back, it waits; released, it
    goes on. Passes repeat (block 0 after the last)."""
    import threading
    import time

    from vecworld.engine.uring import SsdSource

    sizes = [300_000, 123_457, 500_001, 77_777, 250_000]
    total = 54048 + sum(sizes) + 999
    path, data = _file(ssd_dir, total, seed=3)
    lay = _toy_layout(path, sizes)
    src = SsdSource(str(path), lay, buffers=3, ahead=1, pin=False, queue_depth=4, chunk=64 << 10)
    fake = _FakeStreamer()
    src.bind(fake)
    try:
        off = 54048
        starts = []
        for n in sizes:
            starts.append(off)
            off += n
        for rnd in range(2):
            for i, n in enumerate(sizes):
                g = src(i)
                got = _gathered(g)
                assert got[0] == data[starts[i]:starts[i] + n], (rnd, i)
                assert got[-(-n // 256) * 256] == bytes([i]) * 64
                fake.pending.add(i)                    # its copy is now "issued", not done
                assert src.buffer_of(i) is not None
                # nothing handed out and not yet copied has been read over
                for j in fake.pending:
                    assert src.buffer_block(src.buffer_of(j)) == j
                fake.pending.discard(i - 1 if i else len(sizes) - 1)   # the one before has been copied
    finally:
        src.close()
        src.close()
    # two buffers, both handed out and their copies not done: the next block waits until one is released
    src = SsdSource(str(path), lay, buffers=2, ahead=0, pin=False, queue_depth=4, chunk=64 << 10)
    fake = _FakeStreamer()
    src.bind(fake)
    try:
        src(0)
        fake.pending.add(0)
        src(1)
        fake.pending.add(1)
        released = []

        def release():
            time.sleep(0.3)
            released.append(time.time())
            fake.pending.discard(0)

        t = threading.Thread(target=release)
        t.start()
        t0 = time.time()
        g = src(2)
        t.join()
        assert released and time.time() - t0 >= 0.25
        assert _gathered(g)[0] == data[starts[2]:starts[2] + sizes[2]]
        assert src.buffer_block(src.buffer_of(1)) == 1          # block 1's buffer (copy still pending) untouched
    finally:
        src.close()
        src.close()


# --- the switch --------------------------------------------------------------------------------------------------------

def test_weights_source_rule(monkeypatch):
    """FLUX_WEIGHTS_SOURCE (environment, else config) picks ssd or ram under an engine kernel; under VECWORLD_KERNEL=off
    (diffusers\x27 group offload from RAM) it is always ram; anything else is refused."""
    from vecworld.config import FLUX_WEIGHTS_SOURCE
    from vecworld.engine import engine_weights_source

    monkeypatch.delenv("FLUX_WEIGHTS_SOURCE", raising=False)
    assert engine_weights_source("w4a8") == FLUX_WEIGHTS_SOURCE == "ssd"
    monkeypatch.setenv("FLUX_WEIGHTS_SOURCE", " RAM ")
    assert engine_weights_source("w4a8") == "ram"
    monkeypatch.setenv("FLUX_WEIGHTS_SOURCE", "ssd")
    assert engine_weights_source("mma") == "ssd" and engine_weights_source(None) == "ram"
    monkeypatch.setenv("FLUX_WEIGHTS_SOURCE", "nvme")
    with pytest.raises(ValueError, match="ssd or ram"):
        engine_weights_source("w4a8")


def test_dreamer_auto_is_on_unless_flux_holds_its_weights_in_ram(monkeypatch):
    """VECWORLD_DREAMER=auto (the default): the Qwen dreamer runs beside FLUX reading its blocks from the SSD (Task 10b)
    and beside Turbo; beside FLUX with the 6.66 GB arena in RAM (FLUX_WEIGHTS_SOURCE=ram, or VECWORLD_KERNEL=off) it
    stays off. on / off are obeyed."""
    import vecworld.__main__ as main
    import vecworld.dream.ideas as ideas

    monkeypatch.setattr(ideas, "QwenDreamer", lambda: "qwen")
    for k in ("FLUX_WEIGHTS_SOURCE", "VECWORLD_KERNEL"):
        monkeypatch.delenv(k, raising=False)
    assert main.make_dreamer("auto", "flux") == "qwen"
    assert main.make_dreamer("auto", "turbo") == "qwen"
    monkeypatch.setenv("FLUX_WEIGHTS_SOURCE", "ram")
    assert isinstance(main.make_dreamer("auto", "flux"), main.NoDreamer)
    assert main.make_dreamer("on", "flux") == "qwen"
    monkeypatch.setenv("FLUX_WEIGHTS_SOURCE", "ssd")
    monkeypatch.setenv("VECWORLD_KERNEL", "off")
    assert isinstance(main.make_dreamer("auto", "flux"), main.NoDreamer)
    monkeypatch.delenv("VECWORLD_KERNEL")
    assert isinstance(main.make_dreamer("off", "flux"), main.NoDreamer)


def test_drop_file_cache_advises_every_regular_file_once(ssd_dir):
    """drop_file_cache(root) asks the kernel to drop the clean page cache of every regular file under root
    (posix_fadvise DONTNEED; no sudo, nothing is deleted): the server calls it on the model cache once its models are
    loaded, so the pages its loads left cached don\x27t crowd the dreamer into swap. Symlinks (the HF hub\x27s snapshots
    point at its blobs) are skipped, so nothing is counted twice; files keep their bytes."""
    from vecworld.engine.uring import drop_file_cache

    a, da = _file(ssd_dir, 10_000, seed=1)
    (ssd_dir / "sub").mkdir()
    b, db = _file(ssd_dir / "sub", 30_000, seed=2)
    (ssd_dir / "link").symlink_to(a)
    assert drop_file_cache(ssd_dir) == 40_000
    assert a.read_bytes() == da and b.read_bytes() == db
    assert drop_file_cache(ssd_dir / "missing") == 0


def test_ssd_source_recovers_from_a_failed_read_and_refuses_to_run_unbound(ssd_dir):
    """A block read that fails raises OSError from source(i), and the buffer is forgotten, so the next source(i)
    reads the block again (the painter isn't dead until a restart). An SsdSource that was never bind()-ed to its
    streamer can't tell when a handed-out buffer is free: once it would have to wait for one, it raises instead of
    waiting forever."""
    from vecworld.engine.uring import SsdSource

    sizes = [300_000, 123_457, 500_001]
    path, data = _file(ssd_dir, 54048 + sum(sizes) + 999, seed=4)
    lay = _toy_layout(path, sizes)
    src = SsdSource(str(path), lay, buffers=2, ahead=0, pin=False, queue_depth=1, chunk=64 << 10)
    try:
        src._reader._inject_fault(1, -5)
        with pytest.raises(IOError, match="Input/output error"):
            src(0)
        assert src.buffer_of(0) is None
        assert _gathered(src(0))[0] == data[54048:54048 + sizes[0]]
        src(1)                                          # both buffers handed out, nobody to say their copies are done
        with pytest.raises(RuntimeError, match="bind"):
            src(2)
    finally:
        src.close()
