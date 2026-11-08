# 12: T5 on the engine, and the engine end to end (Task 10c)

**What this is:** `flux/text.py` (the `engine` path: `_streamed_block`, `t5_file_layout`, `T5Text(engine=...)`),
`FLUX_T5=engine|pread` (config, `engine.engine_t5`), `engine/uring.py` (`BlockFile.extents`: a block read in several
runs of the file; `SsdSource(wrap=False)`), `engine/slots.py` / `engine/stream.py` (`one_pass`, `from_ssd(layout=)`),
two 10b minors (N1, N4), `dream/vae.py` (C5: the FLUX painter no longer borrows `TurboPainter`), and the end-to-end
run of the finished game.

**The result:** a cold T5 encode drops from **6.7 s to 2.3 s** (2.56 s for the first one in a process) with
**bit-identical** embeddings (cosine 1.0, `torch.equal`), and T5 no longer leaves its 3.4 GB file in the page cache.
The walk gate is byte-identical to Task 10b at all 18 spots. In the game (dreamer on, SSD, defaults) a walk step takes
**39-46 s** (median 42.0), against 38.6 s in the bench; 10b measured 43-50 s.

## The problem

T5-XXL turns a prompt into the 256 x 4096 sequence FLUX is conditioned on. Since Task 1 (`engine="pread"`) each
encode read the encoder from its GGUF (Q5_K_M, 3.39 GB) block by block with `pread`, copied each tensor to the GPU,
dequantized it there to bf16 and ran the block. Each new text cost 6-7.5 s cold (file not in the page cache) and 4.7 s
warm, and the game meets one or two new texts per walk step (the base prompt, the focus prompt): up to 13 s of a
step. Profiled (`~/t10c/t5prof.py`):

| piece | time |
|---|---|
| parsing the GGUF's tensor table (`gguf.GGUFReader` maps the file and builds an array per tensor), **every encode** | 2.8-3.2 s |
| `pread` of the 24 blocks, cold, one thread | 2.24 s |
| per block: pageable host -> GPU copy, dequantize (Q5_K: 2.5 ms per 4096 x 4096, 6.0 ms per 4096 x 10240) | 51 ms (1.2 s) |
| the matmuls (256 tokens; 99 GFLOP a block) | ~0.25 s |

Reading the file through the page cache also left it cached: 3.39 GB of "free" memory that pushes the dreamer
toward swap (the reason 10b moved the FLUX blocks to O_DIRECT).

## How it works now

T5's encoder blocks are streamed exactly like FLUX's, by the same code:

```
    T5 GGUF on the SSD   [ blk.0 | blk.1 | ... | blk.5 (first part) | token_embd | blk.10 ... | blk.5 ffn_down ... ]
                              | O_DIRECT reads through io_uring (engine.uring.DirectReader), one per run of the file
                              v
    staging ring         [ buffer 0 ][ buffer 1 ]                      2 x 141 MB page-locked, block i+1 read ahead
                              | one DMA per tensor (a Gather), into the slot layout
                              v
    device slots         [ slot 0 ][ slot 1 ]                          2 x 141 MB, block i in slot i % 2
                              | per linear: dequantize (Q5_K / Q6_K -> bf16) as it is used, multiply, free
                              v
    h (fp32 residual) --> block 0 --> block 1 --> ... --> block 23 --> RMSNorm --> [1, 256, 4096] bf16 on the CPU
```

- **Each encoder block is a module whose buffers are its tensors' file bytes** (`_streamed_block`): a quantized
  linear is `uint8 [out rows, bytes per row]`, a norm its float type. They are built once, as placeholders; the
  streamer points them at its slots for the encode (`BlockStreamer.from_ssd(blocks, path, layout=t5_file_layout(...))`)
  and takes them back after.
- **The linears stay dequantize-then-matmul.** The file's linears are Q5_K and Q6_K; the engine's kernels take Q4_K
  only, and C1 says not to write a Q5_K kernel unless the 5 s bar needs one. It doesn't: the dequant is ~0.67 s of
  GPU time (24 x 28 ms), hidden under the 1.78 s the SSD needs for 3.28 GB. `_block` itself is Task 1's, unchanged; it
  now gets a lazy mapping (`_Weights`) that dequantizes each linear when the block asks for it, so the card holds one
  linear's bf16 copy at a time, not a block's (peak 0.64 GB allocated during an encode, Task 1: 0.66).
- **A block in several runs of the file.** FLUX's GGUF lays each block out contiguously; T5's orders tensors by
  name, so `enc.blk.10` comes before `enc.blk.5.ffn_down`, and blocks 5 and 12 are split in two runs (block 5 spans
  550 MB of the file end to end). `BlockFile.extents` lists the runs; `SsdSource` submits one read per run into the
  same staging buffer and waits for all of them (if one fails, the others are still waited for before the buffer is
  forgotten, so no read lands in a reused buffer). FLUX's single-run BlockFiles take the old path unchanged.
- **One pass, not a loop.** FLUX's streamer is cyclic (the last block prefetches the next pass's first). An encode
  runs the blocks once, so `one_pass` stops both the slot prefetch and the reader's read-ahead at block 23: without it
  every encode read 0.69 GB too much (3.97 GB, 2.54 s).
- **Only the rows the prompt uses** of the 108 MB token embedding are read (one O_DIRECT read per distinct token, ~20
  of them), and the tensor table is parsed once, when `T5Text` is built (adds ~3 s to the painter's load: 7.3 -> 10.4 s).
- **Built per encode, freed after.** The slots (282 MB) and staging buffers (282 MB pinned) exist only during an
  encode; between encodes T5 holds nothing on the card and nothing in RAM but the tokenizer, the tensor table and the
  LRU of finished embeddings. Setup costs ~0.1 s an encode.
- `FLUX_T5=pread` (or `VECWORLD_KERNEL=off`) keeps Task 1's path as the reference and fallback.

## The numbers (AC)

### T5

| | Task 1 path (`pread`) | **engine** |
|---|---|---|
| cold encode, file evicted from the page cache (bench, `t5_encode_cold_s`) | **6.68 s** (`bench/t10c-t5pread-tile512.json`; 10b rows 6.4) | **2.56 s** first in the process, **2.30-2.42 s** after (`bench/t10c-*.json`, in-game probe: 28 misses, 2.32-2.57 s) |
| inside it | table 3 s + pread 2.2 s + copy / dequant 1.2 s | read 3.28 GB at ~1.8 GB/s (`blocks_s` 2.18-2.34 s, of it waiting on the SSD 0.44 s), setup 0.1 s |
| embeddings against Task 1 (3 prompts) | | **cosine 1.000000, every token 1.000000, max diff 0: `torch.equal`** |
| file pages left in the page cache after an encode | 3,387 MB | **0.0 MB** |
| peak allocated during an encode | 0.66 GB | 0.64 GB (two 141 MB slots + one linear's bf16) |
| pinned host memory | none | 0.28 GB, only during an encode |

Tests: `tests/test_flux_text_gpu.py::test_t5_engine_matches_cpu` (cosine >= 0.999; measured identical) and
`::test_t5_encode_fast` (cold <= 5 s on AC and the file not cached; red at 7.45 s before), CPU
`test_t5_file_layout_reads_every_tensor_of_its_block_once`, `test_t5_path_rule`,
`test_ssd_source_reads_a_block_in_several_runs`.

### The walk gate (Global Constraints, C2)

`walk_sheet.py --scenes forest,desert,coast --out ~/walk-archive/t10c` at 3a6db42, then `--relative
~/walk-archive/t10b ~/walk-archive/t10c`:

| scene | spots | worst seam delta | min PSNR | seams (spot 0-5) |
|---|---|---|---|---|
| forest | 6 | +0.000 | inf | 0.740 / 1.079 / 0.815 / 0.855 / 0.668 / 0.443 |
| desert | 6 | +0.000 | inf | 0.288 / 0.385 / 0.705 / 0.250 / 0.512 / 0.435 |
| coast | 6 | +0.000 | inf | 0.443 / 0.736 / 0.649 / 0.671 / 0.671 / **1.192** |

**Verdict PASS**; `cmp`: all 18 spots and 15 `prev` images byte-identical to 10b (bit-identical T5 means bit-identical
pictures). Coast spot 5 is above the fixed 1.1 limit exactly as at Tasks 9, 10a and 10b (same picture), so the walk's
own exit is 1, as before.

### End to end: the game (C7)

The real server (`python -m vecworld` through a probe launcher that only times things: `~/t10c/serve_probe.py`),
dreamer **on**, SSD, every default, driven over HTTP by `~/t10c/journey.py`: start, then 5 moves; each move turns
toward the neighbour nearest the way the spot faces, walks 7.8 s (9 units at 1.15 units/s) and asks for the next spot.
The **fog** is how long that request blocks, the player standing at the rim. Sheets against SD-Turbo (same driver,
`VECWORLD_PAINTER=turbo`): `~/walk-archive/t10c-e2e/flux-vs-turbo-j{0,1,2}.png`.

| journey | first spot (start) | panorama job | walk steps (server) | fogs at the rim | T5 per step | arrived at (game's names) |
|---|---|---|---|---|---|---|
| haunted forest + night | **34.1 s** | 33.2 s | 39.1-45.9 s | 32.3-39.4 s (mean 36.0) | 2.3-4.7 s | ghostly mist (spot 3) |
| desert oasis | 74.5 s (of it ~40 s behind the previous session's pre-dream on the GPU queue) | 33.5 s | 39.1-44.8 s | 34.8-38.3 s (mean 36.3) | 2.3-4.7 s | whispers of sand (spot 3) |
| snowy mountains | 77.2 s (same) | 33.6 s | 40.0-44.5 s | 33.3-38.1 s (mean 36.1) | 2.3-4.8 s | whipped white ridge (spot 3) |

- Every move was served "waited": the pre-dream toward the facing neighbour was already painting, ~42 s against 7.8 s
  of walking. SD-Turbo for comparison: start 2.9-3.7 s, steps 4.0-4.2 s, fogs 0.1-8.1 s.
- Passes: 4 per panorama, 5 per step (batch 8); a batch-8 pass takes 7.04 s of GPU time in the bench
  (`bench/t10c-peak-8.json`), 7.2-7.5 s in the game, 7.8 s while the dreamer generates: 0.88-0.97 s per window,
  **18.2-20.1 bf16-equivalent TFLOP/s** per window-pass. tile512 (one window, batch 1): **3.81 s, 4.64 TFLOP/s**,
  SSD-bound as since 10b (3.64-3.74 there).
- Memory (whole server, 0.5 s samples): RSS **9.75 GB max** in play, VmHWM 11.18 GB (the load), **VmSwap 0**,
  MemAvailable >= 4.54 GB; peak allocated **2.15 GB**, nvidia-smi **2552 MiB** max.
- Depth 0.6 s per spot, after the painter.
- By eye: FLUX is the more beautiful and the more coherent of the two in all three journeys (SD-Turbo's tiles show as
  blocks in forest spots 2 and 4); FLUX's night forest paints several moons a panorama (each window paints its own),
  and the desert journey drifts into waterfalls at spot 5 (the walk's drift toward a renamed neighbour).

### M8: why an in-game step is slower than the bench (attribution, C4)

Bench walk step (server stopped): **38.6 s**, passes 37.22 s of GPU time. In-game 5-pass steps (8 windows every pass;
`~/walk-archive/t10c-e2e/*.events`):

| | wall | T5 | passes (GPU) | the rest (CLIP, VAE, glue) | ticker worst |
|---|---|---|---|---|---|
| bench (server stopped) | 38.6 | (encoded before timing) | 37.22 | 1.4 | |
| game, dreamer **off** (`VECWORLD_DREAMER=off`, n 5) | 41.5 | 3.3 | 36.05 (incl. 5-window passes) | 2.13 | 42-44 ms |
| game, dreamer idle during the step (n 11) | 41.6 | 3.0 | 36.47 | 2.14 | 42-47 ms |
| game, dreamer generating > 50% of the step (n 4) | 43.8 | 4.1 | 37.54 | 2.17 | 72-101 ms |
| game, `VECWORLD_GRAPHS=1`, dreamer idle (n 3) / busy (n 2) | 42.5 / 45.9 | 3.1 / 4.7 | 37.26 / 39.06 | 2.13 / 2.15 | 42 / 73-78 ms |

Where the 3-5 s go, by cause:

| cause | measured | share of the gap |
|---|---|---|
| **T5 inside the step** (the bench encodes before timing; a step meets 1-2 new texts) | 2.3 s per new text; 3.0-4.1 s a step (Task 1 path: 6.7 s per text, 6.7-13.4 s a step: most of 10b's 43-50 s) | the largest part |
| **the dreamer competing for the CPU / GIL** (Qwen on 8 CPU threads, Python in between) | passes +1.07 s (+2.9%) busy vs idle; ticker worst 72-101 ms vs 42-47 | ~1 s, only while it generates (6 calls of 11-29 s in 15 steps) |
| **GPU sharing** (CUDA events per pass) | per-pass GPU time idle 7.46 s in game vs 7.44 s bench | none: the dreamer runs on the CPU, depth and T5 run serially on the same queue |
| **SSD contention** (`/proc/diskstats` vs the reader's bytes) | disk read per step 39.9 GB = FLUX reader 33.3 + T5 2 x 3.28; reader waits 8.2 / 16.4 s in game vs 8.23 / 15.72 bench | none: nothing else reads the drive (no swap-in) |
| the rest (CLIP for new texts, VAE, Python glue) | 2.14 s in game vs ~1.4 s bench | ~0.7 s |
| graphs (`VECWORLD_GRAPHS=1`, one A/B) | passes no faster (37.26 vs 37.3 idle; 39.06 vs 37.54 busy, n 2); first panorama 42.0 vs 33.2 s (captures); peak 2.38 vs 2.15 GB; nvidia-smi 3742 vs 2552 MiB | does not help the eager collapse under the dreamer here; costs memory |

M13 (SSD + graphs, untested at 10b): it runs (the graphs game run above), and the bench walk step with
`VECWORLD_GRAPHS=1` on the SSD paints a byte-identical image to eager (`cmp` of `bench/last_step.png`, 39.0 vs 38.6 s,
`bench/t10c-graphs-step.json`). No default was changed on the basis of M8 (C4).

### Every piece, Task 0 -> 10c

The one-table summary is in `docs/engine/speed.md` ("Task 0 to 10c in one table").

## Tried and not chosen

| technique / setting | measured | why not |
|---|---|---|
| the Task 1 `pread` path (kept as `FLUX_T5=pread`) | 6.68 s cold; leaves 3.39 GB of file cached | 2.9x slower and fills the page cache |
| a Q5_K / Q6_K engine kernel | not built: the dequant is ~0.67 s of GPU time a whole encode, hidden under the SSD's 1.78 s | C1: only if 5 s is missed; it isn't (2.3 s) |
| T5 blocks as single-run BlockFiles (FLUX's layout) | block 5's tensors span 550 MB of the file | would read 0.4 GB of other tensors per encode; `extents` instead |
| 3 staging buffers, read 2 ahead (FLUX's setting) | 2.43 s, 3.42 GB read (a block read twice), +141 MB pinned | slower than 2 / 1 (2.30 s, 3.28 GB) |
| 4 buffers, 3 ahead (before one_pass) | 2.72 s, 4.24 GB | more re-reads |
| 3 device slots (ring 3) | 3.09 s, 4.64 GB read, peak 0.78 GB | read-ahead evicts blocks before use |
| cyclic streaming (no one_pass) | 2.54 s, 3.97 GB read (the next "pass" prefetched) | 0.69 GB read for nothing an encode |
| parsing the tensor table per encode (as Task 1) | +2.8-3.2 s an encode | parsed once at construction (+3 s at load instead) |
| keeping the T5 slots and staging buffers between encodes | would save ~0.1 s an encode | 282 MB of VRAM and 282 MB pinned RAM held all the time, beside FLUX's slots |
| `VECWORLD_GRAPHS=1` in the game (C4's A/B) | see M8: no faster per step, +8.8 s first panorama, +1.2 GB nvidia-smi | not changed (C4: the controller rules) |

## Tests and suite

`uv run pytest -q -p no:warnings -m "gpu or not gpu or ui"`, server stopped, at 618de75: **879 passed in 2011 s**,
exit 0. New: `test_flux_text_gpu.py` (2), `test_t5_path_rule`, `test_t5_file_layout_reads_every_tensor_of_its_block_once`,
`test_ssd_source_reads_a_block_in_several_runs`, `test_uring_hard_reap_error_breaks_the_reader` (N1, red first),
`test_gather_must_cover_the_block_exactly` + a shifted copy (N4, red first),
`test_flux_painter_does_not_borrow_turbo_painter` (C5, red first).

## What to read next

`docs/engine/speed.md` (every task in one table), `docs/engine/11-io-uring.md` (the reader T5 now shares), and the
spec's §2 (`docs/superpowers/specs/2026-10-20-flux-engine-design.md`), now with the finished engine's measured
numbers next to the spike's.
