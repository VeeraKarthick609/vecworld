"""FLUX bench harness. --painter reference benches the game's FluxPainter (diffusers + GGUF + group offload);
--painter engine benches the same painter with the engine flags taken from the environment (VECWORLD_KERNEL unset
is config.FLUX_KERNEL, as for the painter; --painter reference sets VECWORLD_KERNEL=off). Engine results
also carry the step's FLOP count, achieved TFLOP/s and % of the measured bf16 peak.

  uv run python scripts/bench_flux.py --painter reference --what pano --out bench/x.json
  VECWORLD_KERNEL=naive uv run python scripts/bench_flux.py --painter engine --what tile512 --out bench/t3-naive.json

tile512, pano, step and peak run the real painter (its T5 and CLIP included; T5 is encoded once, before timing, and
its cold encode is recorded). With the first-block cache on (VECWORLD_CACHE_THRESHOLD or config.FLUX_CACHE_THRESHOLD
> 0), tile512, pano and step each run as one cache job and record its passes and skipped passes ("cache"), and
s_per_pass_mean (a job's passes averaged, skipped ones included) is the step metric: the median would hide the
skips. Achieved TFLOP/s is then the nominal pass FLOPs over that mean, so it can exceed the hardware's rate.
blocks and bw measure the bare transformer and the machine; blocks also times the
modulation linears per pass (do_modulation). gemm times
VECWORLD_KERNEL's GEMM alone on the FLUX linear shapes against torch.matmul on the same weight pre-dequantized to
bf16 (the reference's cuBLAS GEMM, without its dequant). Speed numbers are only valid on AC power; on battery
this exits without touching the GPU.
"""
import argparse, glob, json, mmap, os, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT = "a dirt path through a sunlit old forest clearing, open ground in the foreground, big trees, ferns"


def flop_per_step(n_img: int, n_txt: int) -> float:
    """One transformer pass's FLOPs by the exact count (learning ledger section 1.6): 57 blocks x 2 x P weights per
    token (P = 113,246,208, the weights every token meets per block; modulation's one row dropped) plus attention,
    4 N^2 h per block, h = 3072. 17.67e12 at N = 1024 + 256."""
    n = n_img + n_txt
    return 114 * 113_246_208 * n + 700_416 * n ** 2


def bf16_peak() -> float:
    """The measured bf16 matmul peak in TFLOP/s (bench/baseline.json, blocks)."""
    return json.loads((ROOT / "bench/baseline.json").read_text())["results"]["blocks"]["bf16_matmul_tflops"]


def engine_fields(res: dict, n_img: int, n_txt: int, peak_tflops: float | None = None,
                  key: str = "s_per_pass_median") -> dict:
    """res with the pass's FLOP count, the TFLOP/s achieved over res[key] seconds and that as % of the bf16 peak
    (bf16_peak() unless given)."""
    peak = bf16_peak() if peak_tflops is None else peak_tflops
    f = flop_per_step(n_img, n_txt)
    tf = f / res[key] / 1e12
    return {**res, "flop_per_step": f, "achieved_tflops": round(tf, 2), "pct_bf16_peak": round(100 * tf / peak, 1)}


def power():
    on = [open(p).read().strip() for p in glob.glob("/sys/class/power_supply/A*/online")]
    return "ac" if "1" in on else "battery"


def git_sha():
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def smi(fields):
    o = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return [x.strip() for x in o.strip().split(",")]


def rss_gb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS"):
                return int(line.split()[1]) / 1e6
    return 0.0


class Mon:
    """Peak RSS and nvidia-smi memory sampler."""
    def __init__(self, dt=0.25):
        self.dt, self.rss, self.smi, self.stop = dt, 0.0, 0, False
        threading.Thread(target=self.run, daemon=True).start()

    def run(self):
        while not self.stop:
            self.rss = max(self.rss, rss_gb())
            try:
                self.smi = max(self.smi, int(smi("memory.used")[0]))
            except Exception:
                pass
            time.sleep(self.dt)

    def reset(self):
        self.rss, self.smi = 0.0, 0

    def report(self):
        return dict(rss_gb=round(self.rss, 2), smi_mib=self.smi)


def bench(fn, n=5):
    import torch
    fn(); torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(n)]
    for s, e in ev:
        s.record(); fn(); e.record()
    torch.cuda.synchronize()
    return sorted(s.elapsed_time(e) for s, e in ev)[n // 2]


def load_transformer():
    """The bare transformer, its Q4_K linears on the kernel VECWORLD_KERNEL resolves to (vecworld.engine.engine_kernel,
    the painter's own rule: unset is config.FLUX_KERNEL, "off" leaves diffusers), its blocks fused as the painter's
    (vecworld.engine.engine_fuse)."""
    import torch
    from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
    from vecworld.config import FLUX_BASE, FLUX_GGUF
    from vecworld.engine import engine_kernel
    from vecworld.flux.painter import _read_gguf
    kernel = engine_kernel()
    # read as the painter reads it: diffusers' own loader copies each tensor out of a memory map of the whole file, so
    # its peak holds the file's pages and the copies both (~12.8 GB; the GPU tests that load this way got the suite
    # OOM-killed on the 14 GB laptop)
    tr = FluxTransformer2DModel.from_single_file(
        _read_gguf(str(FLUX_GGUF)), quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16, config=FLUX_BASE, subfolder="transformer")
    if kernel is not None:
        from vecworld.engine import keep_for
        from vecworld.engine.swap import gguf_linears, swap_linears
        swap_linears(tr, gguf_linears(tr), kernel, keep_a16=keep_for(kernel))
    from vecworld.engine import engine_fuse
    if engine_fuse(kernel):                       # the painter's rule (VECWORLD_FUSE): fused glue in the blocks
        from vecworld.engine.blocks import patch_blocks
        patch_blocks(tr)
    return tr, str(FLUX_GGUF)


def load_painter(R, mon):
    """The game's FluxPainter, with its prompt's T5 encoded (cold) before anything is timed."""
    from vecworld.config import FLUX_STYLE
    from vecworld.flux.painter import FluxPainter
    passes = []
    t0 = time.time()
    # a CUDA event per pass, on the stream the passes run on: the CPU runs ahead of the GPU (graph replays, and no
    # offload syncs since Task 7), so a CPU clock at progress() would time the enqueueing, not the pass
    p = FluxPainter(progress=lambda k, n: passes.append(_event()))
    R["load_s"] = round(time.time() - t0, 1)
    R["rss_after_setup_gb"] = round(rss_gb(), 2)
    arena = getattr(p, "_arena", None)          # page-locked host memory: the blocks' arena (Task 10a); None when pinned in place (off)
    R["pinned_gb"] = round(arena.nbytes / 1e9, 2) if arena is not None else None
    prompt = [(f"{PROMPT}, {FLUX_STYLE}", 1.0)]
    mon.reset()
    t = time.time(); p.t5.encode(prompt[0][0]); R["t5_encode_cold_s"] = round(time.time() - t, 2)
    R["mon_t5"] = mon.report()
    return p, prompt, passes


def _event():
    import torch
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def timed(fn, passes, mon):
    """Run fn; its wall time, and each pass's GPU time (between the CUDA events progress() recorded)."""
    import torch
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); mon.reset(); passes.clear()
    t = time.time(); t0 = _event(); out = fn(); torch.cuda.synchronize(); dt = time.time() - t
    gaps = [a.elapsed_time(b) / 1000 for a, b in zip([t0] + passes[:-1], passes)]
    return out, dict(s=round(dt, 1), passes=len(passes), s_per_pass_median=round(sorted(gaps)[len(gaps) // 2], 2),
                     s_per_pass_mean=round(sum(gaps) / len(gaps), 3),
                     s_per_pass_min=round(min(gaps), 3), s_per_pass_max=round(max(gaps), 3),
                     s_per_pass=[round(g, 3) for g in gaps],
                     peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), **mon.report())


def run_tile512(p, e, steps, seed=0):
    """One 512x512 window (a token grid FLUX_WIN square, no wrap-around), `steps` Euler steps of one transformer
    pass each. (_denoise can't do this: its windows wrap, so a grid one window wide gets two.)"""
    import torch
    from vecworld.config import FLUX_WIN
    g = torch.randn(1, 64, FLUX_WIN, FLUX_WIN, device=p._dev,
                    generator=torch.Generator(p._dev).manual_seed(seed))
    sig = p._sigmas(steps)
    job, cache = next(p._jobs), p.cache                 # one block cache job, as the painter's _denoise runs one
    s0 = cache.stats["skips"]
    cache.begin_job(job)
    try:
        for k in range(steps):
            g = g + (sig[k + 1] - sig[k]) * p._forward(g, [e], sig[k], key=(job, (0,)), step=k)
            if p.progress is not None:
                p.progress(k + 1, steps)
    finally:
        p.last_cache = {"passes": steps, "skips": cache.stats["skips"] - s0}
        cache.clear()
    return g


def _cache_record(p):
    """The last job's block cache passes and skips, and the probe's relative changes it saw (engine.cache)."""
    return {**(p.last_cache or {}), "rel": [round(x, 3) for x in p.cache.rel]}


def do_tile512(R, mon):
    """One 512x512 window, FLUX_STEPS steps (one transformer pass per step)."""
    import torch
    from vecworld.config import FLUX_STEPS
    p, prompt, passes = load_painter(R, mon)
    with torch.inference_mode():
        e = p._embed(prompt)
        run_tile512(p, e, 1)                                                 # warm-up
        p.cache.rel.clear()
        _, R["tile512"] = timed(lambda: run_tile512(p, e, FLUX_STEPS), passes, mon)
        R["tile512"]["cache"] = _cache_record(p)


def do_peak(R, mon):
    """Peak allocated memory and pass time at 1, 2, 4, 6 and 8 windows per pass (the painter's _forward at sigma 1; 8
    is a panorama step's every window): per batch, a first pass (with graphs: warm-up + capture) and then the median
    of 3 timed passes; the peak covers both."""
    import torch
    from vecworld.config import FLUX_WIN
    p, prompt, _ = load_painter(R, mon)
    with torch.inference_mode():
        e = p._embed(prompt)
        for b in [int(x) for x in os.environ.get("BENCH_PEAK_BATCHES", "1,2,4,6,8").split(",")]:   # a subset: one process each
            g = torch.randn(b, 64, FLUX_WIN, FLUX_WIN, device="cuda", generator=torch.Generator("cuda").manual_seed(b))
            torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            t = time.time(); p._forward(g, [e] * b, 1.0); torch.cuda.synchronize(); first = time.time() - t
            ts = []
            for _ in range(3):
                t = time.time(); p._forward(g, [e] * b, 1.0); torch.cuda.synchronize(); ts.append(time.time() - t)
            med = sorted(ts)[1]
            R[f"batch{b}"] = dict(peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3),
                                  first_pass_s=round(first, 3), s_per_pass=round(med, 3),
                                  s_per_window=round(med / b, 3), passes_s=[round(x, 3) for x in ts])


def _count_windows(p):
    """Count the windows the painter's passes carry (wins[0]); returns the counter."""
    wins, fwd = [0], p._forward

    def counted(tiles, embs, s, **kw):            # a pass carries up to painter.batch windows
        wins[0] += tiles.shape[0]
        return fwd(tiles, embs, s, **kw)

    p._forward = counted
    return wins


def _per_window(res, wins):
    """The job's window passes, and its GPU pass time per window-step (all its passes' GPU time over its window
    passes: what one window costs at one denoising step, the batching metric)."""
    res["window_passes"] = wins
    res["s_per_window_step"] = round(sum(res["s_per_pass"]) / max(wins, 1), 3)


def do_pano(R, mon):
    p, prompt, passes = load_painter(R, mon)
    p.cache.rel.clear()
    wins = _count_windows(p)
    img, R["pano"] = timed(lambda: p.panorama(prompt, seed=21), passes, mon)
    _per_window(R["pano"], wins[0])
    R["pano"]["cache"] = _cache_record(p)
    (ROOT / "bench").mkdir(exist_ok=True); img.save(ROOT / "bench/last_pano.png")


def do_step(R, mon):
    """A walk step as next_spot makes it: the panorama dollied by 1.6, the band it magnified repainted."""
    from vecworld.dream.pano import dolly
    from vecworld.roam.spot import focus_half
    p, prompt, passes = load_painter(R, mon)
    prev = dolly(p.panorama(prompt, seed=21), 1.6)
    wins = _count_windows(p)
    p.cache.rel.clear()
    img, R["step"] = timed(lambda: p.step(prev, prompt, None, 0.5, seed=22, focus_half=focus_half(1.6)), passes, mon)
    _per_window(R["step"], wins[0])
    R["step"]["cache"] = _cache_record(p)
    R["step"]["s_per_window_pass"] = round(R["step"]["s"] / max(wins[0], 1), 2)   # wall time incl. VAE and text
    (ROOT / "bench").mkdir(exist_ok=True); img.save(ROOT / "bench/last_step.png")


def modulation_linears(tr):
    """(name, module) of the modulation linears: the Q4KLinear under each AdaLN norm (a double block's norm1 and
    norm1_context, a single block's norm, the output norm's when it is Q4_K). They turn the per-image conditioning
    into shift / scale / gate, so they see one row per image (Task 5c, docs/engine/05-modulation-gemv.md)."""
    from vecworld.engine.q4k import Q4KLinear
    return [(n, m) for n, m in tr.named_modules()
            if isinstance(m, Q4KLinear) and n.rpartition(".")[2] == "linear" and n.split(".")[-2].startswith("norm")]


# The RTX 3050 Ti Laptop's VRAM bandwidth on paper (GDDR6, 128-bit, 12 Gbps): the brief's "~192 GB/s est.".
VRAM_GBPS_SPEC = 192


def do_modulation(tr, ms=(1, 4)):
    """The modulation linears' cost per transformer pass: each moved to the GPU and timed alone through Q4KLinear's
    own dispatch at M rows (1: a tile512 pass; 4: FLUX_BATCH windows), the better of two 20-call medians, summed.
    GB/s is their packed bytes over that time (each byte is read once a pass), against the 192 GB/s spec and the
    read bandwidth measured here (a 512 MB torch sum)."""
    import torch
    mods = modulation_linears(tr)
    if not mods:
        return None
    buf = torch.ones(512 * 2 ** 20 // 8, dtype=torch.int64, device="cuda")
    read_gbps = 512 * 2 ** 20 / min(bench(lambda: buf.sum(), 20), bench(lambda: buf.sum(), 20)) / 1e6
    del buf
    nbytes = sum(m.qweight.numel() for _, m in mods)
    out = dict(n_linears=len(mods), q4k_gb=round(nbytes / 1e9, 3), vram_read_gbps_measured=round(read_gbps, 1),
               vram_gbps_spec=VRAM_GBPS_SPEC)
    tot = dict.fromkeys(ms, 0.0)
    for _, m in mods:
        m.cuda()
        with torch.inference_mode():
            for M in ms:
                x = torch.randn(M, m.in_features, device="cuda", dtype=torch.bfloat16)
                tot[M] += min(bench(lambda: m(x), 20), bench(lambda: m(x), 20))
        m.cpu()
    for M in ms:
        gbps = nbytes / tot[M] / 1e6
        out[f"m{M}"] = dict(ms_per_pass=round(tot[M], 2), gbps=round(gbps, 1),
                            pct_of_spec=round(100 * gbps / VRAM_GBPS_SPEC, 1),
                            pct_of_measured_read=round(100 * gbps / read_gbps, 1))
    torch.cuda.empty_cache()
    return out


def do_blocks(R, mon):
    import torch
    import torch.nn.functional as F
    from diffusers.quantizers.gguf.utils import dequantize_gguf_tensor
    tr, _ = load_transformer(); dev = "cuda"
    nd, ns = len(tr.transformer_blocks), len(tr.single_transformer_blocks)
    # sustained bf16 matmul throughput on this power state
    a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    mm_ms = bench(lambda: [a @ a for _ in range(10)]) / 10
    tflops = 2 * 4096 ** 3 / mm_ms / 1e9
    R["bf16_matmul_tflops"] = round(tflops, 2)
    for W, H in [(512, 512), (1024, 512)]:
        L, T = (W // 16) * (H // 16), 256
        hs = torch.randn(1, L, 3072, device=dev, dtype=torch.bfloat16)
        es = torch.randn(1, T, 3072, device=dev, dtype=torch.bfloat16)
        temb = torch.randn(1, 3072, device=dev, dtype=torch.bfloat16)
        ids = torch.zeros(T + L, 3, device=dev); ids[T:, 1] = torch.arange(L, device=dev) // (W // 16); ids[T:, 2] = torch.arange(L, device=dev) % (W // 16)
        rot = tr.pos_embed.to(dev)(ids)
        d = tr.transformer_blocks[0].to(dev); s = tr.single_transformer_blocks[0].to(dev)
        with torch.inference_mode():
            td = bench(lambda: d(hidden_states=hs, encoder_hidden_states=es, temb=temb, image_rotary_emb=rot))
            ts = bench(lambda: s(hidden_states=hs, encoder_hidden_states=es, temb=temb, image_rotary_emb=rot))
            q = torch.randn(1, 24, T + L, 128, device=dev, dtype=torch.bfloat16)
            att = bench(lambda: F.scaled_dot_product_attention(q, q, q))
            wd = [p for p in d.parameters() if hasattr(p, "quant_type")]
            ws = [p for p in s.parameters() if hasattr(p, "quant_type")]
            dqd = bench(lambda: [dequantize_gguf_tensor(w) for w in wd])
            dqs = bench(lambda: [dequantize_gguf_tensor(w) for w in ws])
        k = f"{W}x{H}"
        step_s = (nd * td + ns * ts) / 1000
        dq_s = (nd * dqd + ns * dqs) / 1000
        att_s = (nd + ns) * att / 1000  # one SDPA per block
        # GEMM FLOPs: 2 * params * tokens (all linears; double blocks see img+txt tokens, single the joint seq)
        flops = 2 * 11.9e9 * (L + T)
        gemm_s = flops / (tflops * 1e12)
        R[f"{k}"] = dict(
            double_block_ms=round(td, 1), single_block_ms=round(ts, 1),
            double_dequant_ms=round(dqd, 1), single_dequant_ms=round(dqs, 1),
            sdpa_ms_per_block=round(att, 2),
            resident_step_s=round(step_s, 2), dequant_s=round(dq_s, 2),
            dequant_share=round(dq_s / step_s, 3),
            attention_s=round(att_s, 2), attention_share=round(att_s / step_s, 3),
            gemm_glue_s=round(step_s - dq_s - att_s, 2), gemm_glue_share=round(1 - (dq_s + att_s) / step_s, 3),
            gemm_flops_floor_s=round(gemm_s, 2))  # FLOPs / peak matmul TFLOPS: a lower bound, not a measurement
        d.to("cpu"); s.to("cpu"); torch.cuda.empty_cache()
    mod = do_modulation(tr)
    if mod is not None:
        R["modulation"] = mod


def do_bw(R, mon):
    import torch
    R["pcie_link"] = " ".join(smi("pcie.link.gen.current,pcie.link.width.current"))
    n = 256 * 1024 * 1024
    dbuf = torch.empty(n, dtype=torch.uint8, device="cuda")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    for name, pin in [("pageable", False), ("pinned", True)]:
        h = torch.empty(n, dtype=torch.uint8, pin_memory=pin); h.fill_(1)
        for _ in range(50): a @ a   # wake the GPU/link
        ts = []
        for _ in range(20):
            t = time.perf_counter(); dbuf.copy_(h, non_blocking=pin); torch.cuda.synchronize(); ts.append(time.perf_counter() - t)
        ts.sort()
        R[f"h2d_{name}_gbps_median"] = round(n / ts[10] / 1e9, 2)
        R[f"h2d_{name}_gbps_best"] = round(n / ts[0] / 1e9, 2)
        R[f"h2d_{name}_link"] = " ".join(smi("pcie.link.gen.current,pcie.link.width.current"))
    # copy/compute overlap
    h = torch.empty(n, dtype=torch.uint8, pin_memory=True); s = torch.cuda.Stream()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(40): a @ a
    torch.cuda.synchronize(); tc = time.perf_counter() - t
    t = time.perf_counter()
    for _ in range(4): dbuf.copy_(h, non_blocking=True)
    torch.cuda.synchronize(); tm = time.perf_counter() - t
    t = time.perf_counter()
    with torch.cuda.stream(s):
        for _ in range(4): dbuf.copy_(h, non_blocking=True)
    for _ in range(40): a @ a
    torch.cuda.synchronize(); tb = time.perf_counter() - t
    R["overlap_compute_s"], R["overlap_copy_1gb_s"], R["overlap_both_s"] = round(tc, 3), round(tm, 3), round(tb, 3)
    # SSD O_DIRECT sequential read of the transformer GGUF (page cache bypassed)
    from vecworld.config import FLUX_GGUF
    gp = str(FLUX_GGUF)
    chunk = 16 * 1024 * 1024
    buf = mmap.mmap(-1, chunk)  # page-aligned, as O_DIRECT requires
    fd = os.open(gp, os.O_RDONLY | os.O_DIRECT)
    try:
        t = time.perf_counter(); total = 0
        while True:
            r = os.preadv(fd, [buf], total)
            if r <= 0: break
            total += r
        dt = time.perf_counter() - t
    finally:
        os.close(fd)
    R["ssd_odirect_gbps"] = round(total / dt / 1e9, 2)
    R["ssd_bytes_read_gb"] = round(total / 1e9, 2)
    # buffered read twice; the second pass is served from the page cache (warm)
    for key in ("ssd_buffered_first_gbps", "ssd_pagecache_warm_gbps"):
        t = time.perf_counter(); total = 0
        with open(gp, "rb", buffering=0) as f:
            while True:
                b = f.read(chunk)
                if not b: break
                total += len(b)
        R[key] = round(total / (time.perf_counter() - t) / 1e9, 2)


# (M, K, N) of FLUX's token-facing linears: one 512 window with its text (M = 1024 image + 256 text tokens; a double
# block's image and text streams run apart, M = 1024 and 256; a single block runs them joined, M = 1280).
GEMM_SHAPES = [(1024, 3072, 9216), (1024, 3072, 3072), (1024, 3072, 12288), (1024, 12288, 3072),
               (256, 3072, 9216), (256, 3072, 3072), (256, 3072, 12288), (256, 12288, 3072),
               (1280, 3072, 21504), (1280, 15360, 3072)]


def do_gemm(R, mon):
    """Per GEMM_SHAPES entry: VECWORLD_KERNEL's TFLOP/s on a real Q4_K weight of that shape (the first in the GGUF;
    Q4KLinear's own dispatch; the "ours_*" fields, as in bench/t5-gemm.json) against torch.matmul on the same weight
    dequantized to bf16 beforehand: the median of 20 timed calls after warm-up, two rounds each, run alternately, and
    the better round kept. A W4A8 column is always added (Q4KLinear "w4a8", its quantize_act included; quantize_ms
    alone), so --what gemm under VECWORLD_KERNEL=mma compares the two. TFLOP/s are bf16-equivalent (2 M K N / time)."""
    import gguf
    import numpy as np
    import torch
    from vecworld.config import FLUX_GGUF
    from vecworld.engine.q4k import Q4KLinear, quantize_act, read_gguf
    from vecworld.engine import engine_kernel
    kernel = engine_kernel()
    tensors = read_gguf(FLUX_GGUF)
    R["bf16_peak_tflops"] = bf16_peak()
    rows = []
    for M, K, N in GEMM_SHAPES:
        t = next(t for t in tensors.values() if t.type == gguf.GGMLQuantizationType.Q4_K and t.shape == (N, K))
        lin, w8 = Q4KLinear(t, None, kernel).cuda(), Q4KLinear(t, None, "w4a8").cuda()
        wd = torch.from_numpy(gguf.quants.dequantize(np.asarray(t.data), t.type).reshape(N, K)).to("cuda", torch.bfloat16)
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=torch.Generator("cuda").manual_seed(0))
        fns = {"ours": lambda: lin(x), "matmul": lambda: x @ wd.T, "w4a8": lambda: w8(x), "quantize": lambda: quantize_act(x)}
        with torch.inference_mode():
            ms = {k: bench(f, 20) for k, f in fns.items()}
            ms = {k: min(v, bench(fns[k], 20)) for k, v in ms.items()}
        f = 2 * M * K * N
        row = dict(m=M, k=K, n=N, weight=t.name, ours_ms=round(ms["ours"], 3), matmul_ms=round(ms["matmul"], 3),
                   ours_tflops=round(f / ms["ours"] / 1e9, 2), matmul_tflops=round(f / ms["matmul"] / 1e9, 2),
                   pct_of_matmul=round(100 * ms["matmul"] / ms["ours"], 1),
                   w4a8_ms=round(ms["w4a8"], 3), quantize_ms=round(ms["quantize"], 3),
                   w4a8_tflops=round(f / ms["w4a8"] / 1e9, 2),
                   w4a8_pct_of_matmul=round(100 * ms["matmul"] / ms["w4a8"], 1),
                   w4a8_over_ours=round(ms["ours"] / ms["w4a8"], 2))
        rows.append(row)
        print(f"M={M:5d} K={K:5d} N={N:5d}  {kernel} {row['ours_tflops']:6.2f}  matmul {row['matmul_tflops']:6.2f}  "
              f"{row['pct_of_matmul']:5.1f}%   w4a8 {row['w4a8_tflops']:6.2f} (quantize {row['quantize_ms']:.3f} ms) "
              f"{row['w4a8_over_ours']:.2f}x", flush=True)
        del lin, w8, wd
        torch.cuda.empty_cache()
    R["shapes"] = rows
    for name, sel in (("m1024_1280", lambda r: r["m"] != 256), ("m256", lambda r: r["m"] == 256)):
        rs = [r for r in rows if sel(r)]
        R[f"{name}_pct_of_matmul_min"] = min(r["pct_of_matmul"] for r in rs)
        # aggregate: matmul's total time over ours across the shapes (the FLOP-weighted rate ratio)
        R[f"{name}_pct_of_matmul_overall"] = round(100 * sum(r["matmul_ms"] for r in rs) / sum(r["ours_ms"] for r in rs), 1)
        R[f"{name}_w4a8_pct_of_matmul_min"] = min(r["w4a8_pct_of_matmul"] for r in rs)
        R[f"{name}_w4a8_pct_of_matmul_overall"] = round(100 * sum(r["matmul_ms"] for r in rs)
                                                        / sum(r["w4a8_ms"] for r in rs), 1)
        R[f"{name}_w4a8_over_ours_overall"] = round(sum(r["ours_ms"] for r in rs) / sum(r["w4a8_ms"] for r in rs), 2)


def add_engine_fields(R):
    """The FLOP-rate fields on a run's results, per 512 window pass with 256 text tokens: tile512 over its median
    pass; a walk step over its wall time per window pass (its passes batch windows, and its wall time includes the
    VAE and text: a lower bound on the kernel's rate); blocks over each size's resident step."""
    from vecworld.config import FLUX_TEXT_LEN, FLUX_WIN
    if "tile512" in R:
        R["tile512"] = engine_fields(R["tile512"], FLUX_WIN * FLUX_WIN, FLUX_TEXT_LEN)
        if R.get("cache_threshold"):                 # skipped passes are short: the mean is the job's step
            R["tile512"] = engine_fields(R["tile512"], FLUX_WIN * FLUX_WIN, FLUX_TEXT_LEN, key="s_per_pass_mean")
    if "step" in R:
        R["step"] = engine_fields(R["step"], FLUX_WIN * FLUX_WIN, FLUX_TEXT_LEN, key="s_per_window_pass")
    for k, (w, h) in (("512x512", (512, 512)), ("1024x512", (1024, 512))):
        if k in R:
            R[k] = engine_fields(R[k], (w // 16) * (h // 16), 256, key="resident_step_s")


WHATS = dict(tile512=do_tile512, pano=do_pano, step=do_step, blocks=do_blocks, bw=do_bw, gemm=do_gemm, peak=do_peak)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--painter", choices=["reference", "engine"], required=True)
    ap.add_argument("--what", choices=list(WHATS), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from vecworld.engine import engine_kernel
    if a.painter == "engine" and engine_kernel() is None:
        sys.exit("--painter engine with VECWORLD_KERNEL=off is the reference: unset it (FLUX_KERNEL) or name a kernel")
    if a.what == "gemm" and a.painter != "engine":
        sys.exit("--what gemm times an engine kernel: use --painter engine")
    if a.painter == "reference":
        os.environ["VECWORLD_KERNEL"] = "off"         # the reference is diffusers all the way (the painter defaults to an engine kernel)
        os.environ["VECWORLD_FUSE"] = "off"
    pw = power()
    if pw != "ac":
        sys.exit("on battery: speed numbers are invalid; plug in and rerun")
    import torch
    mon = Mon()
    from vecworld.engine import engine_fuse
    from vecworld.engine import engine_batch, engine_cache_threshold, engine_graphs
    R = {"what": a.what, "painter": a.painter, "kernel": engine_kernel(), "fused": engine_fuse(engine_kernel()),
         "batch": engine_batch(engine_kernel()),
         "graphs": engine_graphs(engine_kernel()), "cache_threshold": engine_cache_threshold(engine_kernel()), "gpu_power": " ".join(smi("enforced.power.limit,pcie.link.gen.current,pcie.link.width.current"))}
    t0 = time.time()
    WHATS[a.what](R, mon)
    R["wall_s"] = round(time.time() - t0, 1)
    if a.painter == "engine":
        add_engine_fields(R)
    mon.stop = True
    R["gpu_power_end"] = " ".join(smi("enforced.power.limit,pcie.link.gen.current,pcie.link.width.current"))
    out = {"power": power(), "git": git_sha(), "results": R}
    if out["power"] != "ac":
        sys.exit("power dropped to battery during the run; discarding")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
