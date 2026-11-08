# 11: Reading the blocks from the SSD: O_DIRECT through io_uring, nothing resident (Task 10b)

**What this is:** `engine/csrc/uring.cpp` (`DirectReader`: io_uring on the raw syscalls), `engine/uring.py`
(`DirectReader`, `flux_file_layout`, `SsdSource`, `ssd_source`), `engine/stream.py` (`Gather`,
`BlockStreamer.from_ssd`), `FLUX_WEIGHTS_SOURCE=ssd|ram` (config, `engine.engine_weights_source`), and the dreamer
back on by default beside FLUX (`__main__.make_dreamer`). Under an engine kernel the painter no longer keeps the
blocks' 6.66 GB of weights in RAM: every pass reads them from the GGUF file on the SSD, block by block, into three
page-locked staging buffers, and streams them from there into Task 7's slots.

**The result:** the same pictures (bit for bit: `torch.equal` against the RAM arena at batch 1 and 8, and the walk gate),
the game's speed kept at batch 8 (walk step 38.0 s vs 38.9 s from RAM, panorama 30.6 vs 29.9 s; the GPU idles 1.7 ms a
pass), the painter's RSS down from 7.52 to 1.58 GB after setup (8.54 -> 2.60 GB in a walk step), and the whole game
with the Qwen dreamer **on** at 7.9 GB idle, 9.5 GB at most in play, no swap. The cost: a lone single-window pass
(tile512) is SSD-bound, 3.64 s instead of 0.88 s, at the drive's 1.84 GB/s.

## The problem

After Task 10a the painter held 7.52 GB after setup (8.54 GB during a walk step), 6.66 GB of it the page-locked
block arena. Qwen, the dreamer, is ~6 GB more in fp32: together they don't fit in 14 GB, so the dreamer has been off
beside FLUX since Task 1. The weights are read once a pass anyway (57 copies, one per block, host to GPU); the
question was whether they can come from the SSD instead of RAM without costing the game speed.

## Page cache vs O_DIRECT

An ordinary `read()` goes through the **page cache**: the kernel reads the file into its own memory and copies from
there into the buffer. Read 6.66 GB a pass that way and the kernel keeps (up to) 6.66 GB of the file cached: it is
"free" memory on paper (reclaimable, not in the process' RSS) but it pushes everything else out first, Qwen's pages
included, and on this laptop that is swap. `O_DIRECT` asks the kernel to skip the cache: the NVMe drive DMAs the
file's bytes straight into the process' buffer, and nothing is left behind. The price is alignment: the file offset,
the length and the buffer's address must all be multiples of the device's logical block size (512 B here; we use
4096). A block's tensors start at arbitrary 32-byte-aligned offsets in the GGUF (block 0 at byte 54,048), so a block
is read as its aligned superset (at most 4 KB more on each end) and the copies start at the right place inside it.

```
 page cache read:   SSD --DMA--> page cache --memcpy--> buffer        (and the cache keeps a copy)
 O_DIRECT read:     SSD --DMA------------------------> buffer        (offset, length, address 4096-aligned)
```

## io_uring's two rings

io_uring is the kernel's asynchronous I/O interface: two ring buffers shared between the process and the kernel
(`io_uring_setup` makes them, `mmap` maps them in).

```
            process writes                          kernel writes
 SQ ring  [ sqe | sqe | sqe | ... ]  tail ->     CQ ring  [ cqe | cqe | ... ]  tail ->
   each:  READ fd, offset, len, dst, tag            each:  tag, result (bytes read or -errno)
   io_uring_enter(to_submit)  -> the kernel takes them, starts the reads, returns at once
   the drive DMAs into dst ... the kernel posts a cqe ... we read it (no syscall) and move the head
```

`liburing` (the usual C wrapper) has no headers installed here, so `csrc/uring.cpp` uses the three syscalls and
`<linux/io_uring.h>` directly (~380 lines with the bookkeeping). A block's read is split into 1 MB chunks; up to
`queue_depth` (64) of them are in the kernel at once, the rest wait in a queue and go in as completions free room.
`done(id)` reaps the completion ring without a syscall (one `io_uring_enter(GETEVENTS)` if it looks empty) and never
blocks; `wait(id)` blocks in `io_uring_enter` with the GIL released. A short read (the file ends first) raises
`IOError` naming the offset; a partial read in the middle (allowed by O_DIRECT) is resubmitted for the rest.

## The file layout, and why a block is a Gather, not one copy

Each block's tensors are one contiguous run of the GGUF (191.5 MB a double block, 79.8 MB a single; no gaps, in
block order). But the run is in the **file's** order and form, not the slot's (`block_layout`): diffusers renames the
tensors and splits a double block's `img_attn.qkv` into `to_q / to_k / to_v` (and a single block's `linear1` into
four), and the biases and norm weights are F32 in the file and bf16 in the model. Two consequences:

- The packed Q4_K weights (the bulk: 99.7% of the bytes) are the file's bytes verbatim: a Q4_K row is 144 bytes per
  256 weights, rows are contiguous, so `to_k` is rows 3072..6143 of `qkv`, one byte range. `flux_file_layout` maps
  each block's `*.qweight` to its range by a table (`FLUX_DOUBLE`, `FLUX_SINGLE`), checked against the file's types,
  shapes and sizes, and tested both ways: the table tiles every Q4_K block tensor in the file exactly once
  (`test_flux_map_covers_every_block_weight_once`), and every block's slot from the SSD equals the RAM path's bytes
  (`test_ssd_blocks_match_the_ram_bytes`; with `to_k` / `to_v` swapped in the table it fails on block 0's `to_k`).
- The converted tensors (~6 MB over 57 blocks) stay **resident**, copied once into a small page-locked buffer.

So `source(i)` returns a `Gather`: one DMA copy per tensor (the staging buffer's range to its slot offset, or the
resident copy), ~34 a double block, issued by the same `SlotRing.load` call that did one copy in 10a. A Gather is
worked out once per (block, buffer) and handed out again as the same object, so the hot path is still a few
microseconds of Python that never releases the GIL. (Repacking each block into slot order on the CPU instead would
cost a 191 MB memcpy a block and a second set of buffers: see "Tried and not chosen".)

The load changes too: with the SSD source the painter's `_read_gguf` leaves the blocks' Q4_K tensors **unread**
(allocated, never written, so their pages are never touched), diffusers builds the model on them, and
`BlockStreamer.from_ssd` drops them when the blocks become slot views. The painter never reads 6.6 GB into RAM.

## The three-stage pipeline

```
 SSD (1.84 GB/s)      |== read i+1 ==|== read i+2 ==|== read i+3 ==|        io_uring, 1 MB chunks, 64 in flight
 staging (3 x 191 MB)   buf a          buf b          buf c                  page-locked; reused once copy_done
 PCIe (13.4 GB/s)          |c i+1|        |c i+2|        |c i+3|          SlotRing, side stream, per-tensor DMA
 GPU compute          [ block i      ][ block i+1    ][ block i+2    ]     slot i % 3
```

`SsdSource(i)`: if block i isn't already in a buffer, claim one and submit its read; then submit block i+1 .. i+ahead
into buffers that are free; then wait for block i's read (GIL released only if it hasn't finished) and hand out the
Gather. A buffer is free once its read is done and the block last handed out from it has been copied:
`streamer.copy_done(j)`, the reader's contract from Task 10a (False from the moment `source(j)` is called until that
copy has run on the GPU). A block still in a buffer is handed out again without a read. Passes repeat, so "ahead"
wraps from block 56 to block 0.

## The numbers

All on AC, game server stopped for the bench rows (`bench/t10b-uring-*.json`, run at e226147); the game
rows with the full server (painter + Qwen dreamer + depth) driven through a session start and three walk steps over
HTTP (`~/t10b/play.py`, sampling `/proc` every 0.5 s).

**The drive.** Intel 670p 512 GB (QLC), PCIe 3 x4. `DirectReader` over the 57 block extents (6.67 GB), two blocks in
flight: **1.80-1.84 GB/s** at every queue depth from 4 to 128 and chunk from 128 KB to 16 MB. A plain synchronous
`pread` with O_DIRECT gets 1.50 GB/s with 1 MB reads, 1.82 GB/s with 16 MB ones. The drive is the limit: one pass of
weights takes 3.6 s to read whatever we do.

**Speed, SSD vs RAM** (one process per row):

| | RAM arena (10a) | **SSD (10b)** |
|---|---|---|
| tile512, batch 1: s / pass | 0.88 | **3.64** (SSD-bound: 1.85 GB/s achieved) |
| panorama, batch 8: s (passes) / s per window-step | 29.9 (4) / 0.887 | **30.6 (4) / 0.906** (+2.3%) |
| walk step, batch 8: s (passes) / s per window-step | 38.9 (5) / 0.936 | **38.0 (5) / 0.915** |
| one batch-8 pass (`--what peak`, own process) | 6.95 / 7.42 (10a-g / 10a rows, earlier sessions) | **7.18** |
| GPU idle per pass, nsys, 3 passes: batch 1 / batch 8 | 1.9 / 1.2 ms | **2,769 / 1.7 ms** |
| SSD read rate over the run (wall): tile512 / walk step | - | 1.85 / 0.88 GB/s (a batch-8 pass needs 0.92) |
| peak allocated, batch 1 / 8 | 1.22 / 2.05 GB | 1.22 / 2.05 GB (identical) |
| painter load | 13.1 s | **6.8-7.1 s** (6.6 GB fewer bytes read) |

A batch-8 pass computes for ~7.2 s and needs its 6.66 GB in that time: 0.92 GB/s of the drive's 1.84, so the reads
hide behind compute with room to spare. A batch-1 pass computes for 0.85 s and waits ~2.8 s for the SSD.

**FLUX_BATCH 4 vs 8 on the SSD** (the 10a note: "halving the passes should matter more"):

| | batch 4 | **batch 8** |
|---|---|---|
| panorama s (passes) / s per window-step | 34.4 (8) / 1.024 | **30.6 (4) / 0.906** |
| walk step s (passes) / s per window-step | 42.9 (10) / 1.038 | **38.0 (5) / 0.915** |
| s / pass | 4.06-4.09 (SSD-bound: compute ~3.7 s, read 3.6 s) | 7.21-7.23 |
| peak allocated | 1.76-1.77 GB | 2.05 GB |

Batch 8 is 11-13% faster from the SSD (it was 1.4-2.9% from RAM): at batch 4 every pass waits for the drive.

**Memory, the painter alone** (bench, `test_painter_from_ssd_holds_no_weights`):

| | RAM arena | **SSD** |
|---|---|---|
| peak RSS while loading (VmHWM) | 7.86 GB | **3.56 GB** (diffusers' qkv split copies of the unread weights, freed at install) |
| RSS after setup / during a walk step | 7.52 / 8.54 GB | **1.58 / 2.60 GB** |
| page-locked host memory | 6.66 GB | **0.58 GB** (3 x 191.5 MB staging + 6 MB resident) |

**Memory, the whole game with the dreamer on** (R4; server RSS, the server's own swap):

| | RAM arena + Qwen | **SSD + Qwen** |
|---|---|---|
| load peak (VmHWM) | 12.70 GB | **11.0 GB** |
| idle | 10.19 GB, **3.53 GB swapped out**, system swap 4.0 of 4.1 GB | **7.92 GB, no swap** |
| during play (start + 3 steps) | 13.67 GB, MemAvailable 0.40 GB: stopped by the watchdog after one step | **9.45 GB max, no swap**, MemAvailable >= 5.0 GB |
| start / walk steps | 37.9 s / 61.9 s (swapping) | 37.7 s / 50.4, 48.8, 43.4 s |

The first SSD run swapped 0.48 GB of the server out once, at the first walk step, with MemAvailable still 5.3 GB:
the server started beside a warm page cache (6.2 GB, most of it the model files its own loads had just read), free
memory hit the watermark and the kernel (swappiness 60) reclaimed some cold anonymous pages along with the cache. Started
with those files' cache dropped, the same walk swapped nothing; so the game now drops them itself once loaded
(`uring.drop_file_cache`, `posix_fadvise(DONTNEED)` on the HF model cache, no sudo), and from an 11.8 GB warm cache it
swapped nothing (the row above).

**The page cache (R6).** `/proc/meminfo` Cached before and after: tile512 (4 passes, 26.9 GB read) 7.91 -> 7.91 GB;
walk step (33.3 GB read) 6.57 -> 6.57 GB; panorama 6.09 -> 6.10 GB; two passes in a fresh painter 8.03 -> 8.03 GB.
O_DIRECT reads leave nothing behind (a buffered read would have cached up to the whole 6.66 GB a pass).

**The reader's knobs** (tile512 batch 1 is the SSD-bound case, the walk step at batch 8 the compute-bound one):

| setting | tile512 s / pass | walk step s | |
|---|---|---|---|
| **3 buffers, ahead 2, 64 x 1 MB in flight (chosen)** | **3.64-3.66** | **38.0** | 575 MB staging |
| 2 buffers, ahead 1 | 3.69 | 38.1 | 191 MB less |
| 4 buffers, ahead 3 | 3.66 | 38.7 | 191 MB more |
| 3 buffers, ahead 0 (read on demand) | 3.68 | 38.9 | |
| 3 buffers, ahead 1 | 3.65 | | |
| queue depth 8 / 4 | 3.66 / 3.65 | | |
| chunk 256 KB / 4 MB | 3.66 / 3.69 | | |

Batch 1 can't go faster than the drive, and every setting reaches it within 1%; at batch 8, read-ahead is worth 2.4%
(38.9 -> 38.0 s). Two buffers are within noise at batch 8 and save 191 MB; three keep the batch-1 edge (speed first;
RSS has 2.5 GB of room).


## What to read next

- `engine/csrc/uring.cpp` (the rings), `engine/uring.py` (`flux_file_layout`, `SsdSource`), `engine/stream.py`
  (`Gather`, `from_ssd`), `tests/test_uring.py`, `tests/test_uring_gpu.py`.
- `docs/engine/10-streamer.md` (the slots, SlotRing, and the reader's contract this builds on).
- Task 10c moves T5 onto the engine.
