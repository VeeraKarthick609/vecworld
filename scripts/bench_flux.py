"""FLUX bench harness. --painter reference benches the game's FluxPainter (diffusers + GGUF + group offload).

  uv run python scripts/bench_flux.py --painter reference --what pano --out bench/x.json

tile512, pano and step run the real painter (its T5 and CLIP included; T5 is encoded once, before timing, and
its cold encode is recorded). blocks and bw measure the bare transformer and the machine. Speed numbers are only
valid on AC power; on battery this exits without touching the GPU.
"""
import argparse, glob, json, mmap, os, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT = "a dirt path through a sunlit old forest clearing, open ground in the foreground, big trees, ferns"


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
    import torch
    from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
    from vecworld.config import FLUX_BASE, FLUX_GGUF
    return FluxTransformer2DModel.from_single_file(
        str(FLUX_GGUF), quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16, config=FLUX_BASE, subfolder="transformer"), str(FLUX_GGUF)


def load_painter(R, mon):
    """The game's FluxPainter, with its prompt's T5 encoded (cold) before anything is timed."""
    from vecworld.config import FLUX_STYLE
    from vecworld.flux.painter import FluxPainter
    passes = []
    t0 = time.time()
    p = FluxPainter(progress=lambda k, n: passes.append(time.time()))
    R["load_s"] = round(time.time() - t0, 1)
    R["rss_after_setup_gb"] = round(rss_gb(), 2)
    prompt = [(f"{PROMPT}, {FLUX_STYLE}", 1.0)]
    mon.reset()
    t = time.time(); p.t5.encode(prompt[0][0]); R["t5_encode_cold_s"] = round(time.time() - t, 2)
    R["mon_t5"] = mon.report()
    return p, prompt, passes


def timed(fn, passes, mon):
    import torch
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); mon.reset(); passes.clear()
    t = time.time(); out = fn(); torch.cuda.synchronize(); dt = time.time() - t
    gaps = [b - a for a, b in zip([t] + passes[:-1], passes)]
    return out, dict(s=round(dt, 1), passes=len(passes), s_per_pass_median=round(sorted(gaps)[len(gaps) // 2], 2),
                     peak_alloc_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), **mon.report())


def run_tile512(p, e, steps, seed=0):
    """One 512x512 window (a token grid FLUX_WIN square, no wrap-around), `steps` Euler steps of one transformer
    pass each. (_denoise can't do this: its windows wrap, so a grid one window wide gets two.)"""
    import torch
    from vecworld.config import FLUX_WIN
    g = torch.randn(1, 64, FLUX_WIN, FLUX_WIN, device=p._dev,
                    generator=torch.Generator(p._dev).manual_seed(seed))
    sig = p._sigmas(steps)
    for k in range(steps):
        g = g + (sig[k + 1] - sig[k]) * p._forward(g, [e], sig[k])
        if p.progress is not None:
            p.progress(k + 1, steps)
    return g


def do_tile512(R, mon):
    """One 512x512 window, FLUX_STEPS steps (one transformer pass per step)."""
    import torch
    from vecworld.config import FLUX_STEPS
    p, prompt, passes = load_painter(R, mon)
    with torch.inference_mode():
        e = p._embed(prompt)
        run_tile512(p, e, 1)                                                 # warm-up
        _, R["tile512"] = timed(lambda: run_tile512(p, e, FLUX_STEPS), passes, mon)


def do_pano(R, mon):
    p, prompt, passes = load_painter(R, mon)
    img, R["pano"] = timed(lambda: p.panorama(prompt, seed=21), passes, mon)
    (ROOT / "bench").mkdir(exist_ok=True); img.save(ROOT / "bench/last_pano.png")


def do_step(R, mon):
    """A walk step as next_spot makes it: the panorama dollied by 1.6, the band it magnified repainted."""
    from vecworld.dream.pano import dolly
    from vecworld.roam.spot import focus_half
    p, prompt, passes = load_painter(R, mon)
    prev = dolly(p.panorama(prompt, seed=21), 1.6)
    img, R["step"] = timed(lambda: p.step(prev, prompt, None, 0.5, seed=22, focus_half=focus_half(1.6)), passes, mon)
    (ROOT / "bench").mkdir(exist_ok=True); img.save(ROOT / "bench/last_step.png")


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


WHATS = dict(tile512=do_tile512, pano=do_pano, step=do_step, blocks=do_blocks, bw=do_bw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--painter", choices=["reference", "engine"], required=True)
    ap.add_argument("--what", choices=list(WHATS), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.painter == "engine":
        sys.exit("painter 'engine' is not built yet")
    pw = power()
    if pw != "ac":
        sys.exit("on battery: speed numbers are invalid; plug in and rerun")
    import torch
    mon = Mon()
    R = {"what": a.what, "gpu_power": " ".join(smi("enforced.power.limit,pcie.link.gen.current,pcie.link.width.current"))}
    t0 = time.time()
    WHATS[a.what](R, mon)
    R["wall_s"] = round(time.time() - t0, 1)
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
