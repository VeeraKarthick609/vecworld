# 08: First-block cache: skip blocks 1..56 when block 0 barely changed (Task 8)

**What this is:** `engine/cache.py`. Between block 0 and the rest of a transformer pass, a probe compares block 0's
residual with the same window batch's previous step. If it moved little, the pass skips blocks 1..56 and adds the
residual those blocks produced at the batch's last computed step. It is TeaCache / First-Block-Cache, lossy, and gated
by the multi-scene relative walk gate.

**The result: it ships off (`FLUX_CACHE_THRESHOLD = 0.0`).** Every threshold that skipped anything on the gate's
walks failed the gate; every threshold that passed skipped nothing. The machinery stays (tested, one config value
or `VECWORLD_CACHE_THRESHOLD` to turn on), with the measured speed it would buy. See "The sweep" and "Why a 4-step
distilled model has little to skip".

## Residual reuse

A FLUX block adds to the stream it is given: `h_{i+1} = h_i + f_i(h_i)`. So the 56 blocks after block 0 add, in
total, a residual `R = h_56 - h_0`. If the input to them (`h_0`) is almost what it was at the previous step, `R` is
probably almost what it was too, and the pass can use the old one:

```
 computed pass:  x_emb -> [blk 0] -probe-> [blk 1] ... [blk 56] -> norm_out, proj_out      store R = h56 - h0
 skipped pass:   x_emb -> [blk 0] -probe-> h0 + R(stored)          -> norm_out, proj_out      (56 blocks and their
                                                                                              weight copies skipped)
```

A skipped pass costs block 0 (1/57 of the blocks), the embedders, the output layers and one host sync.

## The probe

`r0 = h_0 - x_emb` (block 0's own residual on the image tokens), for the whole window batch. Per batch the probe keeps
the previous step's `r0` and a running sum:

```
 rel(t) = |r0(t) - r0(t-1)|_1 / |r0(t-1)|_1           (fp32 L1 norms, one scalar read on the host)
 acc + rel(t) < threshold  ->  skip, acc += rel(t)
 otherwise                 ->  compute, acc = 0, store the new R
```

Step 0 never skips; a batch without a stored residual computes. `rel` is measured against the previous step whether
or not that step was skipped, so two small moves that add up past the threshold force a recompute
(`test_cache_accumulates_change`). With the threshold at 0 the probe does nothing at all: no norm, no sync.

## Where the decision lives: one pass loop for both block paths

Graphs were off by default from the Task 7 review until Task 10a-g, so the decision is not inside `engine/graphs.py` (the brief's
place) but in `engine.cache.cached_forward`, the transformer's forward written out (diffusers 0.41's, without
controlnet, IP-adapter and gradient checkpointing). It calls each block as a module, so whatever forward the
`GraphRunner` installed runs: graph replays by default (since 10a-g), eager blocks on the weight slots with `VECWORLD_GRAPHS=0`.
Computing, it equals diffusers' forward bit for bit, under both (`test_cached_forward_matches_diffusers`). The
painter calls it only while the cache is active; at threshold 0 it calls the transformer as before.

With graphs, block 0's output is a reused ping-pong buffer that block 2 overwrites, so a computed pass clones `h_0`
before the other blocks run (6 MB per window).

## Skipping saves the copies too

The weight slots prefetch on a forward hook: after block i, its slot's next user (block i + 3) is copied in. A pass
that skips blocks 1..56 would still have started block 3's 14 ms copy after block 0, and the next pass would then find
block 3, not block 0, in slot 0. So `BlockSlots` has `hold`: a block in it records its slot's next copy instead of
issuing it, and `release(i)` issues it. While the cache is active, `cached_forward` holds block 0's:

```
 computed:  [blk 0] probe (sync) release(0) -> copy 3 ... [blk 1][blk 2][blk 3] ...   copies as before, one later
 skipped:   [blk 0] probe (sync)  (nothing released)                                  no copies; block 0 stays in
                                                                                      slot 0 for the next pass
```

Slots 1 and 2 still hold blocks 1 and 2 (prefetched at the end of the last full pass), so the full pass after a skip
starts with no copy at all. `test_full_skip_full_matches_no_cache` runs full, skip, full on the real transformer, eager
and with graphs: the skipped pass issues 0 copies and the full pass after it equals the uncached pass bit for bit;
`test_slots_hold_defers_block0_prefetch_until_released` checks the bookkeeping on small blocks (7 copies per full
pass, 0 per skip). On the computed pass, block 3's copy starts after the probe's sync instead of after block 0; it
still has blocks 1 and 2 (~30 ms at batch 1) to land in.

## Staleness and the cache key (Review Focus 7)

- **Key = (job id, window ids of the batch).** The painter's `_denoise` is one job (a panorama or a walk step); each
  gets a new id from a counter. A batch of windows (0, 1, 2, 3) never uses the residual of (4, 5, 6, 7), nor of
  (0, 1, 2) (`test_cache_keyed_by_windows`).
- **Reset per job, on Stopped, and at the end of a job**, never at step 0: `begin_job` drops everything and accepts
  only keys of the new job (a key of another job raises, it is never served); `_denoise` clears the cache in a
  `finally`, so a job that ends, raises or is stopped leaves no entries (and no residual memory) behind
  (`test_cache_reset_per_job`, `test_painter_stop_clears_the_cache_and_the_next_job_starts_cold`).
- **Batch composition changing mid-job.** A walk step skips background windows at early steps (`_active`), so the
  same window may sit in different batches at different steps. Then the key differs and the batch computes: a miss,
  never a wrong residual. In the gate's walks the batches kept their composition (2 keys per job, 8 probes per 10
  passes). **Task 9** batches windows differently (and may change the batch size mid-job): exact-tuple keys stay
  correct but would miss more. If that matters, key per window (store each window's slice of R and r0, skip a batch
  only when every window in it would skip). Not built: no threshold passes the gate today.

## Why a 4-step distilled model has little to skip

The probe's measured `rel` per step (both batches alike; bench prompt, `bench/t8-probe-pano.json`,
`t8-probe-step.json`, `t8-cache-0.3-tile512.json`; the gpu test's random 512 latent for the last row):

| | step 1 | step 2 | step 3 | step 4 |
|---|---|---|---|---|
| panorama (2 batches, sigmas 1, 0.75, 0.5, 0.25) | 0.83 | 0.23 | 0.29 | |
| walk step `sharp` (2 batches, 5 sigmas) | 0.62 | 0.19 | 0.23 | 0.29 |
| tile512 (1 window, bench prompt) | 0.83 | 0.23 | 0.29 | |
| tile512 (random latent, `test_cache_skips_on_flux`) | 5.18 | 0.23 | 0.55 | |

TeaCache-style thresholds for 25-50-step models (0.05-0.15) sit below every one of these. Schnell is distilled to
take big jumps: each of its 4 steps moves the latent a quarter of the way from noise to image, so block 0's input,
and its residual, change by 20-80% between steps. Only one step per batch is a candidate at all (the 0.19-0.23 step
after the big first jump), and skipping it means reusing a residual from a quarter of the trajectory earlier. That is
the ledger's point (Technique 5: 1.0-1.33x for a 4-step model) and it is what the walks show: a skip happens at most
on 2 of 8 panorama passes and 2-4 of 10 step passes, and those skips are visible.

## The sweep

Gate (ruling for this task): base walks at BASE with the cache off (`~/walk-archive/t8-base-<scene>`, git 4a063b2:
the painter of 970930c plus the gate script; forest byte-identical to t7-r1), three scenes (`walk_sheet.SCENES`:
forest = "haunted forest" seed 7, desert = "red desert" 11, coast = "stormy coast" 23), 6 spots each, kernel w4a8 with
the 8-pattern keep-list, eager slots. Pass: no spot's seam above the same spot of its base by more than 0.03, and the
chosen threshold's minimum per-spot PSNR against the cache-off image at least 30 dB. Every walk records kernel,
keep-list, git sha (6388713, clean) and threshold.

Skips per job: the panorama (spot 0, 8 passes) / each walk step (spots 1-5, 10 passes each). Step s: median walk-step
paint time over spots 1-5 (the forest base has one 80 s outlier at spot 4, so not the mean), and the panorama's.
Worst delta: the largest seam rise over the scene's 6 spots; min PSNR: against the cache-off image of the same spot.

| threshold | scene | skips pano / steps | pano s | walk step s (median) | worst delta | min PSNR dB (spot 0) | gate |
|---|---|---|---|---|---|---|---|
| 0 (base, 4a063b2) | forest | 0 / 0 | 35.8 | 47.8 | | | base |
| | desert | 0 / 0 | 35.8 | 43.6 | | | base |
| | coast | 0 / 0 | 35.6 | 44.1 | | | base |
| 0.3 | forest | 2 / 4, 2, 2, 3, 2 | 28.7 | 35.0 | **+0.150** | **22.56** (30.64) | fail |
| | desert | 2 / 2, 2, 2, 2, 2 | 28.6 | 37.2 | **+0.131** | **24.43** (37.13) | fail |
| | coast | 2 / 2, 3, 3, 2, 2 | 28.4 | 34.4 | **+0.153** | **21.70** (32.33) | fail |
| 0.25 (added) | forest | 2 / 2, 2, 2, 2, 2 | 28.2 | 36.8 | **+0.166** | **23.13** (30.64) | fail |
| | desert | 0 / 2, 2, 2, 2, 2 | 35.9 | 37.3 | **+0.074** | **26.63** (inf) | fail |
| | coast | 1 / 2, 2, 2, 2, 2 | 32.0 | 37.0 | **+0.082** | **22.05** (36.01) | fail |
| 0.2 | forest | 0 / 0 | 35.3 | 44.5 | 0 | inf (byte-identical) | pass, vacuously |
| | desert | 0 / 0 | 36.0 | 43.7 | 0 | inf | pass, vacuously |
| | coast | 0 / 0 | 35.6 | 44.2 | 0 | inf | pass, vacuously |
| 0.15, 0.1, 0.05 | all | 0 / 0 (implied by 0.2) | | | 0 | inf | not walked (pruned) |

Walks: `~/walk-archive/t8-base-*`, `t8-c0.3-*`, `t8-c0.25-*`, `t8-c0.2-*` (gate output in `t8-c<t>.gate`).

Pruning: the sweep ran 0.3 first, then 0.2. At 0.2 not one pass of the 18 jobs skipped, so the three walks are
byte-identical to their bases; on the same trajectory every probe value was at least 0.2, so 0.15, 0.1 and 0.05 can
skip nothing either (identical trajectories, a smaller threshold) and were not walked (0.15 was started and stopped).
0.25, outside the brief's list, was added to look between the passing and the failing threshold.

**Chosen: 0.0 (off).** No threshold that skips passes: 0.3 and 0.25 fail the seam gate (worst +0.15 / +0.17) and
the PSNR floor (21.7 / 22.1 dB). 0.2 (and below) passes only because it never skipped on these walks, and it is not
safe to call that a pass: on the bench's prompt a walk step probes 0.19 at step 2 and would skip there, a case the
gate never saw. A default of 0.2 would buy nothing measured and add a host sync per pass, so the default stays 0.0.

What the failures look like (`~/walk-archive/t8-sheets/`): the 0.3 panorama keeps the composition but goes softer
and hazier in places (PSNR 30.6 / 37.1 / 32.3 dB on spot 0, inside the floor); a walk step then starts from a
different picture, so by spot 1 the walks diverge (22-29 dB): the same layout with different details (forest spot 1
loses the house on the right). Some seams improve (-0.25), some worsen (+0.17); a chained walk turns any change into
noise of this size, so only a threshold that changes nothing passes this gate.

## Speed (what the cache buys when it does skip)

Same session, AC, git 6388713, threshold 0 (off) against 0.3 (`bench/t8-cache-off-*.json`, `t8-cache-0.3-*.json`).
"Step s" here is the job's passes averaged (`s_per_pass_mean`), skipped passes included; the median hides skips.

| | off | 0.3 | speed-up | skips |
|---|---|---|---|---|
| tile512, s per pass (mean of 4) | 0.859 (0.853 / 0.861 / 0.861 / 0.861) | 0.650 (0.854 / 0.866 / **0.021** / 0.859) | **1.32x** | 1 of 4 |
| tile512 achieved TFLOP/s (nominal FLOPs / mean) | 20.55 | 27.19 (above the 19.1 bf16 peak: work skipped, not done faster) | | |
| panorama, s per pass (mean of 8) / wall | 3.49 / 29.7 s | 2.65 / 22.7 s | **1.32x / 1.31x** | 2 of 8 |
| walk step, s per pass (mean of 10) / wall | 3.78 / 39.2 s | 2.32 / 24.6 s | **1.63x / 1.59x** | 4 of 10 |
| peak allocated GB tile512 / pano / step | 1.22 / 1.76 / 1.77 | 1.26 / 1.76 / 1.77 | | |
| a skipped pass | | 21 ms at batch 1, 77-83 ms at batch 4 (block 0, embedders, output, probe) | | |

The probe alone (threshold 1e-9: probes every pass, never skips; `t8-cache-probe-tile512.json`): 0.874 s per pass
against 0.859 / 0.861 for the two cache-off runs around it, about +1.5% (one sync after block 0, the L1 norms and the
`h_0` clone), within reach of this laptop's run-to-run drift but in the direction expected.

Against the expectations: the brief's ~1.5x+ holds only for a walk step at 0.3 (1.63x per pass), and not for the
panorama or tile512 (1.32x). The ledger's 1.0-1.33x for a 4-step distilled model is right for those two. **Neither
ships**: at the threshold that buys it, the gate fails. With the default (off) the step is unchanged from Task 7:
0.86 s, 20.55 TFLOP/s (Task 7 measured 0.85 / 20.79 in its own session).

## Correctness

| Test (`tests/test_cache.py`) | Bar | Result |
|---|---|---|
| `test_cache_never_skips_first_step` | step 0 computes even with an entry and no change | pass |
| `test_cache_threshold_zero_never_skips` | threshold 0: never skips, never probes (no sync) | pass |
| `test_cache_reset_per_job` | begin_job drops entries; another job's key is refused | pass |
| `test_cache_keyed_by_windows` | a different (or partial) window set never gets another batch's residual | pass |
| `test_cache_stop_clears` | clear() empties the cache and ends the job | pass |
| `test_cache_accumulates_change` | 0.06 + 0.06 > 0.1 recomputes; the sum restarts | pass |
| `test_cached_forward_computing_equals_the_plain_loop` (CPU fake FLUX) | the pass loop = the plain loop | `torch.equal` |
| `test_painter_cache_threshold_zero_is_the_plain_transformer` | threshold 0 job = job with no cache object; every block every pass | `torch.equal` |
| `test_painter_skips_at_a_high_threshold_but_never_at_step_0` | a pano-shaped job skips all but each batch's step 0, entries dropped after | pass |
| `test_painter_keys_batches_by_job_and_windows` | keys (job, window ids); new job, new id | pass |
| `test_painter_stop_clears_the_cache_and_the_next_job_starts_cold` | Stopped mid-job: cache empty; next job computes its step 0 | pass |
| `test_engine_cache_threshold_rule` | env / config / off-kernel rule, bad values refused | pass |
| gpu `test_slots_hold_defers_block0_prefetch_until_released` | held prefetch; skip pass 0 copies; full pass after = reference | `torch.equal` |
| gpu `test_cached_forward_matches_diffusers` [eager, graphs] | a computing pass = diffusers' forward | `torch.equal` |
| gpu `test_full_skip_full_matches_no_cache` [eager, graphs] | skip copies nothing; the next full pass = uncached | `torch.equal` |
| gpu `test_cache_skips_on_flux` | 4-step 512 job at threshold 5: 2 of 4 passes skip; at 0: none, = uncached | pass |
| `test_tile512_is_one_cache_job` (bench) | tile512 runs as one cache job and records skips | pass |
| `test_walk_sheet.py` gate tests | scenes, threshold recorded, relative gate (seam delta, PSNR) | pass |

## Tried and not chosen

| Alternative | Measured | Why not |
|---|---|---|
| **Threshold 0.3** (largest of the brief's sweep) | 1.32x pano / 1.63x walk-step passes; gate worst +0.153, min PSNR 21.70 dB | Fails both gate parts in all 3 scenes. Panorama alone (spot 0) 30.6-37.1 dB, but the walk diverges from spot 1 on. |
| **Threshold 0.25** (added between the passing and failing ones) | 0-2 pano skips, 2 per step; worst +0.166, min 22.05 dB | Fails too: one skip per batch per step is enough to change a chained walk. |
| **Threshold 0.2** (passes) | 0 skips in all 18 jobs, byte-identical walks; on the bench prompt a walk step probes 0.19 and would skip | Passes vacuously: no speed measured, an ungated skip possible in the game, +~1.5% probe cost per pass. Default left at 0.0. |
| **0.15 / 0.1 / 0.05** | not walked: at 0.2 every probe on these walks was >= 0.2, so on the same trajectories smaller thresholds skip nothing | Pruned; identical to 0.2 (and to off) on the gate. |
| **Cache decision inside `engine/graphs.py`** (the brief's place) | graphs are off by default since Task 7 | It would only work on the opt-in path; the pass loop (`cached_forward`) serves both (tested both). |
| **Leaving the slot prefetch as it was** (skip only the compute) | a skipped pass would still copy block 3 (14 ms) and the next pass would find block 3 in slot 0 and copy block 0 back on demand (a 14 ms stall) | `hold` / `release` skip the copies: 0 copies on a skip (tested). |
| **Probe metric: the brief's relative L1 of r0, against the previous step** (used) vs **against the last computed step** (ParaAttention FBCache) | not measured separately | The brief's sum of step-to-step changes bounds the change since the last compute from above (triangle inequality, up to the changing denominator), so it is the stricter of the two; the gate fails at the first skip either way. Kept the brief's. |
| **Per-window keys** (skip per window, reuse across batch compositions) | not built | No threshold passes; exact (job, window-tuple) keys are never wrong, only miss. Task 9 note. |
| **Clearing only at begin_job** | | Also cleared at job end and on any exception (`finally`): residuals (6 MB per window, plus r0) don't outlive their job, and a stopped job's entries are gone without waiting for the next begin_job. |
| **Running the cache on the reference path (`VECWORLD_KERNEL=off`)** | | Group offload's prefetch order assumes every block runs; skipping would desynchronise it. The cache is forced off there. |

## What to read next

- `learning-ledger.md`, Technique 5 (block caching): the 1.0-1.33x estimate for a 4-step model holds; the gate says
  even that is not free here.
- Task 9 (batching): the cache key would need to be per window if batches change composition; Task 9 should decide
  whether to keep the cache code at all if it stays off.
- The relative gate (`walk_sheet.py --scenes ... --out`, `--relative BASE CAND --scenes ...`) is the gate for Tasks 9
  and 10.
