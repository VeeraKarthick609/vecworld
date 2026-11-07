# 09: Window batching: every panorama window through each weight pass (Task 9)

**What this is:** `FLUX_BATCH = 8` (was 4) in `config.py`, `painter.batch` / `VECWORLD_BATCH` (`engine.engine_batch`),
and `flux.painter.rowwise`. A panorama is 8 windows round the cylinder. Since Task 1 the painter has put 4 of them
through each transformer pass, so a 4-step panorama made 8 passes. Now all 8 go through together: 4 passes. A walk
step's active windows (all 8 on the bench step) go in one pass per denoising step: 5 passes instead of 10.

**The result: no speed change today, about half the per-pass work, and output identical to one window a pass.**
The engine is compute-bound since Tasks 5-7, so a window costs the same in a batch of 1 or 8 (0.88-0.94 s per
window-step). Batching now halves what is paid once per pass: weight copies, the per-block Python, the
modulation GEMV's weight read. That is the cost Task 10 grows when the weights stream from the SSD. On the way, batching
was found to be *not* lossless under w4a8 (25.7 dB against batch 1). `rowwise` makes it bit-exact. See "The batch-1
check that failed".

## What batching existed before

| | before (Tasks 1-8) | now |
|---|---|---|
| windows per pass | `FLUX_BATCH = 4` (Task 1; the comment said "Task 6 re-batches", it never did) | `FLUX_BATCH = 8`, `VECWORLD_BATCH` overrides |
| panorama | 8 windows x 4 steps = 32 window passes in **8 passes** (2 per step) | **4 passes** (1 per step) |
| walk step (bench: sharp, focus at 0.5) | 5 sigmas x 8 active windows = 40 window passes in **10 passes** | **5 passes** |
| per-block Python iterations (57 blocks a pass) | panorama 456, walk step 570 | panorama **228**, walk step **285** |
| text embeddings | stacked per window (`torch.cat` in `_forward`) | the same |
| graph keys / cache keys | (batch, 1024, 256) / (job, window ids) | the same, now up to batch 8 |

The loop was already right: `_denoise` collects a step's active windows (`_active`) and runs them in chunks of
`FLUX_BATCH`. Task 9 changes the chunk size, makes it a setting the bench and the walk record, and makes a chunk's
result independent of its size.

## Why the weights are moved and decoded once per pass

A transformer pass walks all 57 blocks' weights once, whatever the batch:

```
 pass over 4 windows                         pass over 8 windows
 [copy blk i+1 | GEMMs for 4 x 1280 rows]    [copy blk i+1 | GEMMs for 8 x 1280 rows                ]
 [copy blk i+2 | ...                    ]    [copy blk i+2 | ...                                    ]
   x 57 blocks, x 2 passes a step              x 57 blocks, x 1 pass a step
```

Per pass, and so per batch, the engine pays for:

- **the weight copies**: 6.66 GB of Q4_K from page-locked RAM into the 3 slots, one block ahead (Task 7). They are
  hidden behind compute at any batch on this machine (a block's copy is ~14 ms, its compute ~15 ms per window).
- **the modulation GEMV**: 1.8 GB of Q4_K read for M = batch rows (11 ms at M = 1, 16 ms at M = 4, Task 5c).
- **the CPU's sequencing**: 57 block forwards in Python, each a dozen fused launches, plus the slots' hooks.

Everything else scales with the windows. Each window's 1280 tokens meet every weight in the token-facing GEMMs
(17.67 TFLOP a window), its attention is its own, and its activations are its own.

## Why batching overlaps with the fused kernel (ledger Technique 3)

In Task 1 the reference dequantized every Q4_K weight to bf16 once per pass and then ran cuBLAS. That dequant was a
per-pass cost, and batching 4 windows cut the per-window step from 2.74 to 1.76 s, almost exactly the dequant saved.
Task 5's kernels decode Q4_K in registers inside the GEMM, per tile of rows. The decode now scales with the rows, and
there is nothing per pass left for batching to spread. Both techniques attack the same cost, so the second one to
arrive finds it gone. That is why the walk step gained only 7% from Task 5 (it already batched), and why batching
gains ~0 now (fusion already removed the cost).

What is left per pass is small on today's machine and big tomorrow. In Task 10 the weights stream from the SSD at
2.0-2.96 GB/s: 6.66 GB is 2.3-3.3 s a pass. At batch 1 that would be 2.3-3.3 s **per window-step**. At batch 8 it is
0.3-0.4 s per window-step, under the ~0.9 s of compute, so it can hide again.

## Activation memory against batch size

`bench/t9-batch-peak.json` (one `_forward` at sigma 1, after a warm-up; eager on the slots):

| windows a pass | 1 | 2 | 4 | 6 | 8 |
|---|---|---|---|---|---|
| peak allocated GB | 1.225 | 1.342 | 1.576 | 1.810 | 2.045 |

That is +0.117 GB a window over a ~1.11 GB floor (the 574 MB of slots, the non-block layers, the inputs). With
graphs (opt-in), batch 8 peaks at 1.82 GB for one key and 2.23 GB with the four largest keys a walk can meet
(8, 7, 6, 5) all kept (`test_graph_peak_memory_at_flux_batch`). The painter's panorama, with the VAE decode, peaks at
2.05 GB (bench) / 2.15 GiB (gpu test). The 3.6 GB limit would allow ~20 windows. A panorama only has 8.

One environment note: the painter runs PyTorch's allocator with expandable segments. Without them, batch-8 graph pools
fragmented and ran out of memory on a third large key in the GPU tests (but not in the game). `tests/conftest.py`
now sets the same allocator.

## The sweep and the choice

| batch | panorama s | passes | s / window-step | walk step s | passes | s / window-step | peak GB (pano / step / one pass) |
|---|---|---|---|---|---|---|---|
| 1 | 30.3 | 32 | 0.891 | 40.3 | 40 | 0.972 | 1.76 / 1.77 / 1.225 |
| 2 | 30.6 | 16 | 0.907 | 39.9 | 20 | 0.962 | 1.76 / 1.77 / 1.342 |
| 4 (Task 8) | 30.3 | 8 | 0.897 | 39.4 | 10 | 0.949 | 1.76 / 1.77 / 1.576 |
| 6 | 30.4 | 8 (6 + 2) | 0.902 | 39.5 | 10 (6 + 2) | 0.951 | 1.81 / 1.82 / 1.810 |
| **8** | **30.2** | **4** | **0.896** | **39.3** | **5** | **0.947** | 2.05 / 2.05 / 2.045 |

(64506e0, AC, one process per cell. s / window-step = the job's GPU pass time / its window passes. The walk step's
first gap includes the VAE encode of the view, so it reads higher.)

The one-pass bench went 1, 2, 4, 6, 8 in one process and read 0.881 -> 0.941 s a window, a rising trend. An
interleaved A/B (`bench/t9-batch-ab.json`, 1 / 4 / 8 alternated over 4 rounds) put it down to heat. Every round was
slower than the one before, whatever the order, and the medians were 0.943 / 0.918 / 0.915 s.

**Chosen: 8.** It ties the others on time, fits with 1.5 GB to spare, and halves the passes and with them the per-pass
costs above. 6 buys nothing: 8 windows at 6 is still 2 passes a step.

## The batch-1 check that failed: cuBLAS picks by M

The brief's gate: the batched panorama within 35 dB of the batch-1 one. Batch 8 gave **25.7 dB**, a visibly
different sample. Bisecting by forward hooks (`/tmp/t9diag.py`, `/tmp/t9op.py`, see the task report):

- all token-facing GEMMs (the engine's), the modulation GEMV, the fused glue and attention give a row the same bits
  at any M: given the same inputs, all 57 blocks were bit-identical at batch 1, 2 and 8;
- `x_embedder` and `context_embedder` happened to be identical too;
- the **time-text embedder** was not: `timestep_embedder.linear_1` and `text_embedder.linear_2` are diffusers'
  bf16 `nn.Linear`s, i.e. cuBLAS, which picks its GEMM algorithm by M, and M here is the batch. At M = 2 vs 1 they
  rounded ~2e-3 apart (one bf16 ulp). The output projection (`proj_out`, M = 1024 x batch) did the same at batch 2.

One ulp in the conditioning reaches every block's modulation. On random inputs it grew to 0.3-5% at the output under
both kernels, about what any one-ulp change does (Task 3's 4.2%). Under w4a8 it also flips per-token INT8 roundings,
and four Euler steps later the panorama is another sample.

`rowwise(transformer)` runs those layers a window at a time: `x_embedder`, `context_embedder`, `time_text_embed`,
`norm_out.linear` and `proj_out`, each split along the batch and concatenated. They are a few MFLOP a window, so the
cost does not show. Now a window's pass is the same in any batch:

- `test_batched_windows_match_single_windows[eager|graphs]`: every window of a batch of 8, 5 and 2 `torch.equal` to
  that window alone, and far from any other window alone (a misplaced embedding would show);
- `test_flux_pano_batched_matches`: the batch-8 panorama vs batch 1, **PSNR inf** (identical);
- `test_batched_equals_unbatched` (CPU fake, ragged batches, a step whose active set changes).

## Correctness under the new sizes

- **Ragged batches (Review Focus 8).** A walk step's active set can be any size up to 8, and `VECWORLD_BATCH` can leave
  a ragged last batch. `test_graph_shape_change` now runs keys at batch 1, 3, 2 (ragged 3 + 3 + 2), 1 with 77 text
  tokens, then 8, 5 (a partial walk step) and 8 again. Each matches eager bit for bit, each new shape captures its
  own graphs, and the repeats replay. The eager slot path needs no key.
- **Cache keys.** These are still the exact (job, window tuple) pairs. With the cache forced on (threshold 100),
  batches of 3 and 8 over a step whose active set changes skip only with the residual stored under their own exact
  tuple, of their own batch size, and a tuple new at a step computes
  (`test_painter_batches_never_reuse_another_compositions_residual`). Bigger batches mean fewer keys, so the cache,
  if ever turned on, has the same one candidate step per batch as in Task 8.
- **Stop between steps.** As before, `check_stop()` runs before every pass, so a stop now waits up to one 7.2 s pass
  (was 3.6 s). That is still between denoising steps, as the global constraint asks.

## The numbers

At the batching commit (48e19c1, AC, `bench/t9-b8-*.json` and `bench/t9-b4-*.json` in the same session):

| | Task 8 (6388713, batch 4) | batch 4 now | **batch 8 now (default)** |
|---|---|---|---|
| tile512 s / pass (1 window) | 0.86 (20.55 TFLOP/s) | - | **0.87** (20.31) |
| panorama s (passes) | 29.7 (8) | 30.2 (8) | **29.8 (4)** |
| panorama s / window-step | - | 0.896 | **0.884** |
| walk step s (passes) | 39.2 (10) | 39.4 (10) | **39.3 (5)** |
| walk step s / window-step | - | 0.950 | **0.946** |
| peak allocated GB (panorama / step) | 1.76 / 1.77 | 1.76 / 1.77 | **2.05 / 2.05** |
| per-block Python iterations (panorama / step) | 456 / 570 | 456 / 570 | **228 / 285** |
| graphs (opt-in) walk step | - | - | 39.6 s, 2.08 GB |

The gate walks agree: per spot, the batch-8 walks paint within -2.3 to +2.0 s of the batch-4 BASE walks (same
machine, same session day). Speed is unchanged, as the sweep said it would be.

Quality: batch 8 is the batch-1 picture exactly (panorama PSNR inf against batch 1, every batched window
`torch.equal` to it alone). Against Task 8's default, which was batch 4 *without* `rowwise`, it is another
sample. See the task report's gate section and its ruling request.

## What to read next

`docs/engine/10-streamer.md` (Task 10): the weights streamed from the SSD, where the per-pass cost this task halved
becomes the floor. The per-block Python is now 228 iterations a panorama, the figure Task 10's CPU / GIL work starts
from.
