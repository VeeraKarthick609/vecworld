# 10: The layer streamer: one page-locked arena, one copy a block, sequencing that keeps the GIL (Task 10a)

**What this is:** `engine/stream.py` (`HostArena`, `BlockStreamer`), `engine/csrc/stream.cpp` (`SlotRing`,
`host_pinned`), `engine/slots.py` (BlockSlots split into layout / `install()` / `close()`, its copies and events moved
to `SlotRing`). The painter, under an engine kernel, packs the 57 blocks' weights into one page-locked host arena and
streams them from it through Task 7's 3 fixed-address slots with `BlockStreamer`.

**The result:** the same pictures (bit for bit: 18 of 18 gate-walk spots byte-identical to Task 9), the same speed
when the process is quiet (tile512 0.86 s, walk step 38.6 s), one copy per block instead of 20, and the per-block
sequencing no longer hands the GIL away. With two Python threads spinning in the process, a **graphs** pass now runs
at its quiet speed once going (median 0.86 s, was 7.2 s); an **eager** pass still collapses (median 22.9 s, was 23.7 s),
because the eager block forward's own PyTorch calls (tens per block) each hand the GIL away. And `source(i)`, the seam the
SSD reader (Task 10b) plugs into.

## What Task 7 already did, and what was left

Task 7's `BlockSlots` was already most of a layer streamer: page-locked host weights, a ring of device slots, copies
on a side stream ordered by CUDA events, block i + 1 copying while block i computes. Its own report left two things:

1. **The host side was ~20 tensors per block**, each pinned in place (`cudaHostRegister` on every numpy array the
   GGUF was read into), each copied with its own `copy_`. An SSD reader wants the opposite: one contiguous run of
   bytes per block that a read can land in and a copy can send.
2. **Concern 3: GIL sensitivity.** With two Python threads spinning in the process, a tile512 pass took 20-30 s
   eager and 7-10 s with graphs instead of 0.85 s.

## Why a busy Python thread slowed the GPU 25x

Every PyTorch call made from Python releases the GIL around its work (`pybind11::gil_scoped_release` in the generated
bindings) and takes it back afterwards. Alone that costs nothing. With another thread wanting the GIL, taking it back
means waiting until that thread gives it up, which a busy thread does only when the interpreter's switch interval
(5 ms) runs out. So under contention **every torch call costs up to a switch interval**:

| call (microbenchmark, `~/t10a/gilprobe.py`) | alone | 2 threads spinning |
|---|---|---|
| a Python no-op | 0.0 us | 0.1 us |
| `x.view(-1)` | 0.6 us | 3,457 us |
| `x.add_(1)` | 3.9 us | 4,977 us |
| `dst.copy_(pinned_src, non_blocking=True)` | 3.2 us | 44,295 us |
| `event.record()` / `stream.wait_event(e)` | 3.8 / 3.7 us | 431 / 529 us |
| `torch.cuda.current_stream()` | 2.9 us | 7.2 us |
| the extension's `glue_gated_residual` (pybind, keeps the GIL) | 3.8 us | 11.1 us |

BlockSlots' hooks made ~25 such calls per block (a `copy_` per tensor, a stream wait, two records): 57 x 25 x
milliseconds is the 20 s. The GPU sat idle waiting for the CPU to enqueue the next block.

## How it works now

```
 host arena (one buffer, cudaHostRegister'ed once, 6.66 GB)
 [ block 0 | block 1 | block 2 | ............................................ | block 56 ]
     |  block i is laid out exactly as its slot: each tensor at its slot offset (256-aligned)
     v  ONE cudaMemcpyAsync per block, on the side stream
 device ring   [ slot 0 ]  [ slot 1 ]  [ slot 2 ]      block i in slot i % 3, 191 MB each, fixed addresses
```

- **`HostArena(blocks)`** computes each block's slot layout (`slots.block_layout`), allocates one host buffer of the
  total size, copies each block's tensors to their place in it and points the tensors there (`.data`), block by block,
  then registers the whole buffer with the driver once. The old copies are freed as it goes, and `malloc_trim` after
  each block hands them back to the OS: without it glibc keeps the freed heap (each GGUF tensor is under its 32 MB
  mmap threshold) and the process holds both copies (9.1 GB peak in a fresh process, and OOM-killed at 13.8 GB inside
  the GPU suite). With it the painter's peak RSS is 7.87 GB, Task 9 7.52 after setup (Task 9 bench) GB.
- **`BlockStreamer(blocks, source, ring)`** is a `BlockSlots` whose copy list for block i is one triple: (slot i % ring,
  `source(i)`, block i's size). `source(i)` returns block i's bytes (a tensor or any buffer), page-locked and exactly
  the block's size, or the copy raises `ValueError`. Nothing touches the blocks until `install()`; `close()` syncs,
  unhooks, frees the slots and puts back what the tensors were (the arena views). The painter's source is
  `arena.block`; Task 10b's will be the SSD reader.
- **`graph_launch(exec)`** (C++, in the extension): the GraphRunner replays a captured block graph with
  `cudaGraphLaunch` on its `raw_cuda_graph_exec()` instead of `CUDAGraph.replay()`, which releases the GIL.
- **`SlotRing`** (C++, in the extension) holds the ring's CUDA events and issues the copies, waits and records. Each
  hook is one call into it that keeps the GIL: `load(slot, wait_freed, copies)` (side stream: wait the slot's
  "freed" event, the memcpys, record "copied"), `before(slot)` (compute stream waits "copied"), `after(slot)` (record
  "freed" on the compute stream). The bookkeeping (which block is in which slot, holds for the block cache, the owner
  thread) stays in Python, where it costs microseconds and never releases the GIL.

### Pinned vs pageable memory, DMA, streams and events (the background)

- The GPU's copy engines read host memory by **DMA**: they need physical pages that the OS won't move or swap.
  Ordinary (pageable) memory gives no such promise, so `cudaMemcpyAsync` from it first copies through a pinned
  staging buffer on the CPU, and returns only when that is done: not asynchronous at all. **Page-locked** (pinned)
  memory, from `cudaHostAlloc` or `cudaHostRegister`, is the promise: the copy engine reads it directly while the CPU
  and the SMs do other work, at PCIe speed (13.4 GB/s measured here).
- A **stream** is a queue of GPU work that runs in order; work on two streams may overlap. The weights copy on a
  side stream so the copy engine and the SMs run at once.
- An **event** is a marker in a stream. `cudaEventRecord(e, s)` puts it in; `cudaStreamWaitEvent(s2, e)` makes
  stream s2 wait (on the GPU, not the CPU) until s reaches it. Two per slot order everything: a copy into slot s waits
  for "freed" (its last block ended), a block waits for "copied".

```
 compute  | blk 0 | blk 1 | blk 2 | blk 3 | blk 4 | blk 5 | ...
 side     |c3|      |c4|    |c5|    |c6|    |c7|    ...          c(i+3) waits "freed" of blk i, then copies
            ^ freed(slot 0)            each block waits "copied" of its own slot first
```

## The numbers

All on AC, game server stopped. Bench rows `bench/t10a-stream-*.json` (git a1d78dc); GIL stress `~/t10a/gil.py`
(Task 7's `alt.py` probe, extended: one process per variant, two Python threads spinning in the painter's process,
batch-1 tile512 passes timed by CUDA events); Task 9 / BASE numbers from `bench/t9-*.json` and the same probe run at
fcbfe73 this session.

**Speed (quiet process): unchanged, as expected** (the copies were already hidden behind compute):

| | Task 9 (BASE) | **Task 10a** |
|---|---|---|
| tile512 s / pass | 0.87 (same session at BASE: 0.854) | **0.86** (same session: 0.853) |
| panorama s (passes) | 29.8 (4) | **29.3** (4) |
| walk step s (passes) | 39.3 (5) | **38.6** (5) |
| s / window-step (pano / step) | 0.884 / 0.946 | **0.861 / 0.929** |
| peak allocated, one pass at batch 1 / 4 / 8 | 1.225 / 1.576 / 2.045 GB | **1.225 / 1.576 / 2.045 GB** (identical) |
| painter host RSS after setup / during a walk step | 7.52 / 8.54 GB | **7.52 / 8.54 GB** |
| page-locked host memory | ~6.66 GB, ~1,100 registered ranges | **6.66 GB, one range** |
| copies a pass / bytes / overlapped with a kernel | 1,140 / 6.66 GB / 98.8% | **57** / 6.66 GB / 98.8% |

**Copy / compute overlap** (nsys, 3 back-to-back tile512 passes, `scripts/profile_gaps.py` + `~/t10a/overlap.py`):
the copy engine is busy 495 ms of each 854 ms pass (58%, at 13.4 GB/s), 98.8% of it while a kernel runs; the SMs
are idle 1.6-1.7 ms a pass, before and after. In the test, 12 single blocks at one window: resident 162.1 ms,
streamed 166.6 ms (1.028x; bar 1.1x).

```
 before (Task 7/9)  side  |c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c| ... 20 copies per block, ~20 Python calls each
 now                side  |=========== one copy per block ==========|
 both               main  [ block i compute ~15 ms                   ]  copy ~8.7 ms hidden underneath
```

**Under GIL contention** (2 spinning threads, batch-1 pass, min / median / max s; quiet: 0.85 s):

| sequencing | eager (default) | graphs (opt-in) |
|---|---|---|
| BASE (fcbfe73: per-tensor `copy_` + torch events in Python) | 20.2 / 23.7 / 30.2 | 7.07 / 7.23 / 9.86 |
| per-tensor `copy_` from Python, on the arena (`pytensors`) | 22.9 / 64.2 / 166.8 | 7.15 / 8.25 / 12.0 |
| one `copy_` per block from Python (`pypacked`) | 14.3 / 26.7 / 88.1 | 2.16 / 11.9 / 34.7 |
| **SlotRing, one copy per block, keeps the GIL** | 13.1 / 22.9 / 40.8 | **0.92 / 1.35 / 10.1** |
| **+ graph replay through the extension (`graph_launch`, keeps the GIL)** | (n/a) | **0.85 / 0.86 / 10.5** (5 runs: medians 0.885 / 3.76 / 0.858 / 0.862 / 0.860; `replay()` at a1d78dc, 3 alternating runs: 1.24 / 0.95 / 0.98) |

What it says:

- **With graphs the pass no longer collapses once going**: median 7.2 -> 0.86 s, the quiet speed. A replayed block's
  Python is now the hooks and the graph launch, neither of which releases the GIL. `CUDAGraph.replay()` released it
  once a block (median ~1.0-1.35 s with only that left); `graph_launch` calls `cudaGraphLaunch` on the graph's
  `raw_cuda_graph_exec()` itself (skipping replay()'s RNG prologue, which these graphs never use). The slow passes
  left (up to 10 s) come in the first few passes of a run: the per-pass Python outside the blocks (embedders, the
  rowwise outer layers, the Euler step) still makes PyTorch calls.
- **Eager barely moves** (median 23.7 -> 22.9 s): the eager block forward itself makes tens of PyTorch calls per
  block (reshapes, views, `.to`, `new_empty`, chunk, the attention dispatch, a Q4KLinear's own reshapes), each a GIL
  hand-off (~20 s over 57 blocks at 5-10 ms each is ~40-70 hand-offs a block). Removing the hooks' ~25 leaves the
  forward's. Under contention that cost belongs to eager execution itself, and graphs are the
  remedy that exists (opt-in since Task 7, "revisit after Task 10").
- The spread is wide (a pass of 0.9 s next to one of 10 s in the same run): which thread gets the GIL after a
  switch interval is up to the OS scheduler. Read medians; the probe is a stress case (the game's other threads mostly
  wait on I/O).

**Batch 4 vs 8 under the streamer** (Task 9 ruling b):

| | batch 4 | **batch 8** |
|---|---|---|
| panorama s (passes) / s per window-step | 29.9 (8) / 0.887 | **29.3 (4) / 0.861** |
| walk step s (passes) / s per window-step | 39.1 (10) / 0.942 | **38.6 (5) / 0.929** |
| stop latency (a stop waits for the pass under way: longest pass, pano / step) | **3.63 / 4.37 s** | 7.07 / 7.91 s |
| peak allocated (pano / step) | 1.76 / 1.77 GB | 2.05 / 2.05 GB |

Batch 8 is 1.4-2.9% faster per window-step, so it stays (4 is not equal). The final call waits for 10b, where each
pass also reads 6.66 GB from the SSD and halving the passes should matter more.

**For Task 10b:** at batch 1 the copies already use 58% of a pass (13.4 GB/s over PCIe). From an SSD at 2-3 GB/s a
pass's 6.66 GB take 2.2-3.3 s, longer than a batch-1 pass (0.85 s) and about half a batch-8 one (7.4 s). Batch 8 and
read-ahead are what keep the SSD hidden. The server baseline before 10b: painter RSS 7.52 GB after setup, 8.54 GB in
a walk step, of which 6.66 GB is the arena that 10b removes.

## What to read next

- `engine/stream.py`, `engine/csrc/stream.cpp`, `engine/slots.py`, `engine/graphs.py` (`graph_launch`).
- `docs/engine/07-cuda-graphs.md` (the slots and the GIL probe this answers).
- Task 10b replaces `source` with an io_uring reader of the SSD; Task 10c moves T5 onto the engine.
