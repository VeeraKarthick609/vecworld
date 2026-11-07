# 07: CUDA graphs over the blocks, from fixed weight slots (Task 7)

**What this is:** the transformer's 57 blocks no longer stream their weights through diffusers' group offload. They
stream through three device buffers of fixed address ("slots", `engine/slots.py`), and each block is replayed as a
CUDA graph, one per block and input shape (`engine/graphs.py`). The CPU still runs between blocks, to sequence the
weight copies. `VECWORLD_GRAPHS=0` runs the same blocks eagerly on the same slots; `VECWORLD_KERNEL=off` keeps the
diffusers reference with group offload.

**The result is bit-identical** (graph replay = eager on slots = group offload, `torch.equal` on whole transformer
passes; the gate walk is byte-identical to Task 6's). The step drops **0.92 → 0.85 s** (same session; 19.0 →
**20.8** bf16-equivalent TFLOP/s). The surprise: almost all of that comes from the **slots**, not the graphs.

## Measure first: where the GPU waited (Task 6 build)

Nsight Systems (`nsys`, no sudo needed) over three tile512 passes run back to back, as a job runs them
(`scripts/profile_gaps.py`). "Idle" is time inside the passes when no kernel ran; weight copies on the side stream
don't count as busy, since they leave the SMs idle.

| per pass (Task 6, group offload) | ms | share |
|---|---|---|
| GPU span of a pass | 940 | |
| kernels running | 838 | |
| **idle** | **102** | **10.9%** |
| of which in 3 big gaps (pass end ~39 ms, pass start ~15 + ~13 ms) | ~67 | 7.1% |
| of which 57 gaps of ~0.87 ms, one after every block | ~50 | 5.3% (overlaps the line above in the total) |
| of which in gaps under 100 µs (what launch overhead looks like) | **3.2** | **0.34%** |

So the brief's premise needs a correction. **Launch overhead is not where the time goes.** 1,220 kernels a pass
leave a median gap of 0.64 µs, and every gap under 100 µs together is 3.2 ms, a third of a percent. The GPU waited
on **group offload**:

- ~0.87 ms after every block, while its hooks synchronise the onload stream before the next block can start;
- ~39 ms at the end of each pass, when the group holding the non-block layers (embedders, output norm and
  projection) is offloaded with a blocking `.to()`;
- ~28 ms at the start, onloading that group and block 0, with nothing to overlap them.

That is under the brief's "about 2%" line for the part graphs address (0.34%), so graphs on their own buy
steadiness, not speed. Removing the waits needs the fixed slots, which graphs need anyway.

## What a launch costs, and what a graph does

Every kernel is started by the CPU: Python → PyTorch dispatch → CUDA driver → the GPU's queue, roughly 5–20 µs of
CPU per op. The GPU runs the queue in order. While the CPU stays ahead (the queue never empties), launch cost is
hidden; the GPU only idles when the CPU falls behind or something makes it wait (a sync).

A **CUDA graph** records a sequence of launches once (*capture*) and submits all of them with one call (*replay*).
No Python, no dispatch, no per-kernel driver work; the GPU gets the whole chain at once.

```
 eager:   CPU |k1|k2|k3|k4|k5|...           GPU  [k1][k2][k3][k4]...   gaps if the CPU falls behind
 graph:   CPU |replay|                      GPU  [k1][k2][k3][k4]...   CPU free after one call
```

The price: a graph is frozen. Each kernel's grid size and **every device address it reads or writes** are baked in
at capture. Replaying it reads the same addresses again, whatever is there now. Hence the constraints:

1. **Static shapes.** One graph per input shape: key = (batch, image tokens, text tokens).
2. **Static inputs and outputs.** New inputs must be *copied into* the buffers the graph reads; outputs appear in the
   buffers it wrote.
3. **Static weights.** diffusers' group offload allocates a block's weights anew on every onload, at a different
   address each time, so a graph captured over them would read freed memory on replay. Hence the slots.
4. **No CPU in the middle.** Nothing inside a capture may sync or branch on GPU values.

## Why a graph per block, with the CPU between blocks

The whole transformer is 6.7 GB of Q4_K weights; the card has 4 GB. The weights must stream in during the pass,
and that is CPU work: Task 10 will read them from the SSD with io_uring into the slots. So one graph per block,
and between blocks the CPU (outside any graph) queues the next weight copies:

```
                      block i uses slot i % 3, its weights copied in on a side stream
 main stream   [x_emb][blk 0 ][blk 1 ][blk 2 ][blk 3 ]...[blk 56][out]   each [blk] = one graph replay
 side stream     [copy 2][copy 3   ][copy 4   ][copy 5 ]...[copy 0][copy 1]   (next pass's first blocks)
 events        copy i done ──> blk i may start;  blk i done ──> slot i % 3 may take block i + 3
```

A whole-pass graph would need all 57 blocks' weights resident (6.7 GB); a graph per two or three blocks would need
the weights of all of them in slots at once and buys nothing (the GPU is already saturated, below).

## The slots (`engine/slots.py`)

- **Layout.** Each block's tensors (packed Q4_K, biases, norm weights) get an offset in a slot, every offset a
  multiple of `ALIGN` = 256 bytes. Slots are sized to the largest block (a double block, 191 MB; a single block is
  80 MB). Every tensor of block i is a fixed view into slot i % ring (`param.data = view`), for the whole run.
- **Alignment is loud.** The Q4_K kernels read with 16-byte vector loads. A view that would land off 16 bytes raises,
  naming the tensor (`q4k.check_aligned`, also what `Q4KLinear.forward` calls); nothing is ever copied aside.
- **Copies.** Host tensors are page-locked where they lie, so a copy is asynchronous (13.4 GB/s on this PCIe 4 x8
  link: 14 ms for a double block). Before block i runs, the main stream waits on an event for its copy; after it,
  an event marks its slot free and the slot's next user is copied in, ordered after that event.
- **Ring of 3, not 2.** Block i uses slot i % ring. 57 blocks is odd, so with 2 slots the last block (56, slot 0)
  and the next pass's first (0, slot 0) share a slot: block 0's 14 ms copy could only start when block 56 ended,
  and every pass started with a **13.5 ms** stall (seen in the profile). 3 divides 57: block 0's copy runs during
  blocks 55 and 56. Measured A/B (alternating, 3 × 8 passes): **0.869 → 0.859 s** per tile512 pass, for 191 MB more.

## The graphs (`engine/graphs.py`)

- **Capture on first use.** The first time a block meets a key it runs once eagerly (warm-up: lazy initialisation,
  and it shows the output layout), is captured, and the capture is replayed for the result. After that it is only
  replayed. The first pass of a new shape costs ~1.9 s at batch 1 (vs 0.85 s), once per shape.
- **No copies between blocks.** Each key owns two "ping-pong" output buffers per kind (a double block's separate
  text and image streams; a single block's joint stream). Block i writes into buffer (i + 1) % 2, which is exactly
  block i + 1's static input: the next block sees the same tensor objects it was captured with and nothing is
  copied. The fused forwards (Task 6) write their last pass straight into the buffer (`gated_residual(..., out=)`);
  any other forward is captured with one copy at its end (measured +5 ms a pass, so `out=` is used).
- **Inputs.** Block 0's inputs, the timestep embedding and the rotary tables are copied into per-key static buffers,
  skipped when the very same tensor (same object, unchanged) was copied already, as `temb` and the rotary tables
  are for blocks 1..56.
- **One shared memory pool.** Nothing allocated inside a capture outlives it (outputs go to the ping-pong buffers,
  allocated outside), so every graph's intermediates can reuse the same pool memory in any replay order. Four keys
  captured: 2.10 GB reserved with one pool, 2.84 GB with a pool per key. When the last graph is dropped the pool goes
  with it, and the runner takes a fresh handle (capturing into the old one trips an internal assert in torch 2.14's
  caching allocator; the full GPU suite found this, `test_graph_clear_then_recapture` holds it).
- **LRU.** Beyond `FLUX_GRAPH_MAX` = 4 keys the least recently used key's graphs and buffers are dropped (and
  captured again on its next use). A walk uses batches 1..4 (`FLUX_BATCH`).
- **Outside inference mode.** The runner runs blocks under `no_grad` instead of `inference_mode`, so activations
  keep version counters (below).

### Making the invisible visible (from the Task 6 review)

- **Call counters.** `blocks.CALLS` counts fused double / single forwards and norms that attached a quantization;
  `q4k.CALLS` counts w4a8 linears that used the attached quantization ("attached"), refused a stale one ("stale")
  or quantized themselves ("quantized"). They are **Python-time counts**: eager calls and captures add to them, a
  replay runs no Python and adds nothing. `GraphRunner.stats` counts captures, replays, eager calls, evictions.
  `test_graph_matches_eager` checks the capture pass ran every fused forward twice (warm-up + capture), used the
  attached quantization with no stale refusals, and that the next pass replayed all 57 graphs with no Python
  forward at all.
- **Staleness guard.** The fused norm attaches its INT8 quantization to the tensor x_mod. Graphs refill static
  buffers in place, so a tensor can change under an attachment. `attach_quantized` now records `x._version`;
  `quantized_of` returns nothing once the version moved, and nothing for an inference tensor (which tracks no
  version), so a stale quantization is never consumed (`test_stale_attached_quantization_is_never_used`).

## Timelines, before and after (nsys, 3 tile512 passes back to back, per pass)

| | Task 6 (group offload) | slots, eager (`VECWORLD_GRAPHS=0`), ring 2 | slots + graphs, ring 2 | **slots + graphs, ring 3 (default)** |
|---|---|---|---|---|
| GPU span per pass | 940 ms | 858 ms | 859 ms | **853 ms** |
| idle per pass | 102 ms (10.9%) | 10.4 ms (1.2%) | 9.6 ms (1.1%) | **0.76 ms (0.09%)** |
| gaps > 1 ms (3 passes) | 9 | 2 (pass starts) | 2 (pass starts) | **0** |
| median gap between kernels | 0.64 µs | 0.64 µs | 0.67 µs | |

```
 Task 6:  |##########|..|##########|..|##########| ... |####|.......39 ms.......|...28 ms...|####...
           block        ~0.87 ms sync per block            pass end: offload       next pass start
 Task 7:  |##########|##########|##########| ... |####|#|##########|...        (no gap over 0.3 ms)
```

**Eager on slots is already GPU-bound.** Without group offload's syncs the CPU queues kernels far faster than the
GPU runs them (the CPU is a whole pass ahead), so the graphs have no idle time left to remove: graphs and eager
measure the same within noise.

## Numbers (RTX 3050 Ti Laptop, AC; same session)

tile512 (`bench_flux.py --what tile512`, 4 passes, each timed by CUDA events on the compute stream):

| | Task 6 (`ee3baae`) | slots, eager | **slots + graphs (default)** |
|---|---|---|---|
| pass min / median / max s | 0.918 / 0.922 / 0.924 | 0.844 / 0.850 / 0.852 | 0.848 / 0.854 / 0.857 |
| **step s** (median, 2 d.p.) | 0.92 | 0.85 | **0.85** |
| achieved TFLOP/s (bf16-equivalent) | 19.21 | 20.79 | **20.79** |
| % of the 19.1 bf16 peak | 100.5 | 108.8 | 108.8 |

One pass at 1, 2 and 4 windows (`--what peak`: a first pass, which captures with graphs, then the median of 3; each
pass timed with a device sync, which also waits for the next pass's weight prefetch, so slots read ~10 ms high here):

| | Task 6 | slots, eager | **slots + graphs** |
|---|---|---|---|
| **peak allocated GB, batch 1 / 2 / 4** | 1.011 / 1.106 / 1.296 | 1.225 / 1.342 / 1.576 | **1.249 / 1.431 / 1.792** |
| s per window, batch 1 / 2 / 4 | 0.922 / 0.882 / 0.894 | 0.863 / 0.849 / 0.856 | 0.871 / 0.876 / 0.892 |
| first pass of a shape s, batch 1 / 2 / 4 | 1.44 / 1.78 / 3.46 | 0.95 / 1.70 / 3.36 | 1.98 / 3.70 / 7.60 (capture) |

Memory goes up, well under the 3.6 GB bound: the three slots are 574 MB for the whole run (group offload held about
two blocks), and each captured shape keeps its static and ping-pong buffers (~39 MB per window of batch). With all
four walk shapes captured: 1.51 GB allocated, 1.91 GB peak, 2.10 GB reserved.

The per-batch timings above ran one configuration after the other (heat favours the earlier ones). Alternating A/B
at batch 4 (`alt.py`, 5 pairs × 6 passes, both orders): eager medians 3.464 / 3.562 / 3.546 / 3.564 / 3.554 s,
graphs 3.560 / 3.577 / 3.592 / 3.522 / 3.573 s. Whichever runs first in a pair is faster; the mean difference
(+0.8% for graphs) is inside that order effect. nsys at batch 4 shows the same kernels, idle 1.4 ms (eager) and
0.8 ms (graphs) per pass; the compute-bound kernels (GEMMs, flash attention) ran ~1.6% longer in the later
(graphs) run while memory-bound ones did not move: clocks, not launches.

A walk step (`--what step`, `sharp`, 40 window passes in 10 passes of up to 4 windows):

| | Task 6 | slots, eager | **slots + graphs** |
|---|---|---|---|
| walk step s | 39.0 | 38.3 | **38.8** |
| pass min / median / max s (4 windows) | 3.655 / 3.71 / 4.357 | 3.572 / 3.65 / 4.267 | 3.638 / 3.69 / 4.315 |
| peak allocated GB | 1.30 | 1.77 | 1.92 |

At 4 windows a pass, group offload's waits (~100 ms) were already small next to 3.6 s of compute, so the walk
gains ~1%. The first pass of each step is the slow one (VAE encode, text).

**Spread.** The brief expected a narrower per-pass spread. It did not narrow: 6, 8 and 9 ms (max − min over 4 passes)
for Task 6, eager and graphs. On this laptop the spread is set by clocks and heat (passes drift ~2% slower over a
run, more at batch 4), not by launch jitter.

**Where graphs do matter: a busy CPU.** With two Python threads spinning in the same process (GIL contention, as a
busy server could cause), a batch-1 pass took 19.6 / 28.9 / 64.9 s (min / median / max) eager and 7.5 / 23.0 /
74.5 s with graphs. Both collapse: the per-block sequencing (hooks, slot copies of ~20 tensors) is still Python.
Graphs remove most of the Python per block but not all of it. The game's GPU work runs on one worker thread and its
other threads mostly wait on I/O, so this is a stress case, not today's game, but it says where the next CPU cost is.

## Correctness

| Test (`tests/test_graphs.py`, gpu) | Bar | Result |
|---|---|---|
| `test_slot_layout_aligns_every_tensor` | every tensor of every block in slot i % ring, offsets multiple of 256, 16-byte aligned | pass |
| `test_slot_misalignment_is_loud` | a layout off 16 bytes raises, naming the tensor | pass |
| `test_slots_stream_blocks_in_any_order` | in order (twice round), then out of order: = resident blocks | `torch.equal` |
| `test_stale_attached_quantization_is_never_used` | a written-to x drops its attachment; inference tensors never trusted | pass |
| `test_graph_matches_eager` | whole transformer pass, batch 1, capture pass and replay pass vs eager on the same slots; counters | `torch.equal`; counters as expected |
| `test_graph_shape_change` | keys A (1, 1024 + 256), B (3, 1024 + 256), C (1, 1024 + 77), A again: each = eager, A's second run captures nothing | `torch.equal` |
| `test_graph_lru_evicts_beyond_max` | max 2 keys: LRU dropped, recaptured, still = eager | pass |
| `test_graph_clear_then_recapture` | drop every graph twice, recapture: = eager (a released pool's handle must not be reused: torch asserts) | `torch.equal` |
| `test_graph_stop_between_steps` | `Stopped` before pass 3 of a 4-step job, then a full job: = the eager job; allocated memory after a full job, the stopped job and the next job | `torch.equal`; 0.893 / 0.893 / 0.893 GB |
| `test_graph_peak_memory_batch4` | peak allocated with graphs at batch 4, capture pass included, < 3.6 GB | 1.255 GB |
| `test_slots_match_offload` | whole transformer pass, batch 2: slots eager and slots + graphs vs group offload (Task 6's path) | `torch.equal` |
| CPU: `test_graphs_default_on_with_an_engine_kernel`, `test_painter_streams_blocks_from_slots_under_an_engine_kernel` | the `VECWORLD_GRAPHS` rule; slots instead of group offload under engine kernels, group offload under `off` | pass |

## The gate

Relative walk gate (ruling 2026-11-06): no spot worse than the base walk by more than 0.03. Base: `~/walk-archive/t7-base`
(BASE ee3baae, w4a8 + fused, git recorded); ours: `~/walk-archive/t7` at the final commit.

WALK_TABLE

## Tried and not chosen

| Alternative | Measured | Why not |
|---|---|---|
| **Ring of 2 slots** (the brief's default) | tile512 median 0.867 / 0.871 / 0.870 s vs ring 3 0.858 / 0.861 / 0.859 (3 alternating rounds × 8 passes); nsys: a 13.5 ms stall at every pass start, idle 9.6 vs 0.76 ms/pass; batch 4: 3.568 vs 3.553 s | 57 blocks is odd, so block 0 shares block 56's slot and its copy can't overlap. Ring 3 (+191 MB, peak 1.06 → 1.25 GB at batch 1) removes it. Ring 2 stays the class default. |
| **Copy outputs into the next block's static input** (generic forward, no `out=`) | 0.863 / 0.866 / 0.866 s vs `out=` 0.855 / 0.860 / 0.861 (3 rounds) | +5 ms a pass: 57 extra 8 MB copies. The fused forwards write their last pass straight into the buffer; the copy stays as the fallback for unfused forwards. |
| **A memory pool per key** | 4 keys captured: 2.84 GB reserved vs 2.10 GB shared (allocated 1.505 GB both) | +0.74 GB for nothing: nothing outlives a capture, so all graphs can share one pool. |
| **Eager on slots** (`VECWORLD_GRAPHS=0`) as the default | tile512 0.850 vs graphs 0.854 s (one same-session run); batch 4, 5 alternating pairs: eager 3.538 s mean of medians, graphs 3.565 (+0.8%, inside the run-order effect); nsys idle per pass 1.2% vs 1.1% (ring 2), 1.4 vs 0.8 ms (batch 4) | Kept available, not the default: both gates pass and graphs free the CPU (one call per block), which keeps the GPU fed when the CPU is busy (GIL test: median 23.0 vs 28.9 s). Speed alone does not separate them. |
| **Graphs alone, keeping group offload** | not buildable: group offload re-allocates weights per onload (new addresses) | The slots are a precondition, and turned out to be the win (10.9% → 0.09% idle). |
| **A whole-pass graph / a graph per several blocks** | estimate: all weights 6.7 GB > 4 GB card; per-block idle is already 0.09% | Needs weights resident or many slots; no idle left to recover; Task 10 needs the CPU between blocks for SSD reads. |
| **Capture on the second use** (first use eager, second captures) | first pass of a shape: eager 0.87–0.92 s, capture pass (warm-up + capture + replay) 1.92–1.96 s | Same total cost spread over two passes; capturing in the first pass makes every later pass a replay, which is what the bench and the game run. |
| **Measuring per-pass time with a CPU clock at `progress()`** (the bench before) | graphs: first pass 0.01 s, then 0.835–0.866 s | With no offload syncs the CPU runs a pass ahead of the GPU, so the CPU clock timed enqueueing. The bench now times each pass with CUDA events on the compute stream. A device sync per pass was also rejected: it waits for the next pass's weight prefetch too (~14 ms). |
| **Shared ping-pong buffers across keys** (considered, not built) | estimate: static buffers are ~39 MB × batch per key, 393 MB for batches 1..4; sharing would save ~236 MB | Memory is well within budget (peak 1.9 GB with 4 keys); a larger key would force recapturing every key. Left for Task 9/10 if memory gets tight. |

## What to read next

- `learning-ledger.md`, Technique 4 (CUDA graphs) and the roofline: the pass is now GPU-bound at 0.09% idle, so
  from here only faster kernels (the GEMMs are ~85% of it) or less work (block caching) make it shorter.
- Task 8 / 10: BlockSlots becomes BlockStreamer, reading blocks from the SSD instead of page-locked RAM. The
  per-block CPU work between graphs is where those reads go.
