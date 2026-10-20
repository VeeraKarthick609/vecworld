"""FLUX bench harness (Task 0). Loads FLUX the spike's way: diffusers + GGUF + group offload.

  uv run python scripts/bench_flux.py --painter reference --what tile512 --out bench/x.json

Speed numbers are only valid on AC power; on battery this exits without touching the GPU.
Embeddings come from the spike's cache (~/spikes/flux/embeds.pt) so T5 is never loaded.
"""
import argparse, glob, json, mmap, os, subprocess, sys, threading, time
from pathlib import Path

BASE = "unsloth/FLUX.1-schnell"
GGUF_REPO, GGUF_FILE = "city96/FLUX.1-schnell-gguf", "flux1-schnell-Q4_K_S.gguf"
EMBEDS = Path(os.environ.get("FLUX_EMBEDS", "~/spikes/flux/embeds.pt")).expanduser()
ROOT = Path(__file__).resolve().parent.parent


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
    from huggingface_hub import hf_hub_download
    from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
    gp = hf_hub_download(GGUF_REPO, GGUF_FILE)
    return FluxTransformer2DModel.from_single_file(
        gp, quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16, config=BASE, subfolder="transformer"), gp


def load_pipe(R):
    import torch
    from diffusers import FluxPipeline
    t0 = time.time()
    tr, _ = load_transformer()
    pipe = FluxPipeline.from_pretrained(BASE, transformer=tr, text_encoder_2=None, tokenizer_2=None, torch_dtype=torch.bfloat16)
    dev = torch.device("cuda")
    tr.enable_group_offload(onload_device=dev, offload_device=torch.device("cpu"),
                            offload_type="block_level", num_blocks_per_group=1, use_stream=True)
    pipe.vae.to(dev); pipe.text_encoder.to(dev)
    R["load_s"] = round(time.time() - t0, 1)
    R["rss_after_setup_gb"] = round(rss_gb(), 2)
    return pipe, dev


def make_gen(pipe, dev):
    import torch
    E = torch.load(EMBEDS)
    pe0, pp0 = E["forest"]

    def gen(w=512, h=512, steps=4, seed=0, bs=1):
        pe = pe0.repeat(bs, 1, 1).to(dev); pp = pp0.repeat(bs, 1).to(dev)
        torch.cuda.synchronize(); t = time.time()
        img = pipe(prompt_embeds=pe, pooled_prompt_embeds=pp, width=w, height=h, num_inference_steps=steps,
                   guidance_scale=0.0, max_sequence_length=256, generator=torch.Generator("cpu").manual_seed(seed)).images
        torch.cuda.synchronize()
        return img, time.time() - t
    return gen


def do_tile512(R, mon):
    import torch
    pipe, dev = load_pipe(R); gen = make_gen(pipe, dev)
    R["warmup_512_1step_s"] = round(gen(steps=1)[1], 2)
    torch.cuda.reset_peak_memory_stats(); mon.reset()
    _, t4 = gen(steps=4)
    R["t_512_4step_s"] = round(t4, 2)
    R["peak_alloc_512_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    R["mon_512"] = mon.report()
    _, t1 = gen(steps=1)
    R["t_512_1step_s"] = round(t1, 2)
    R["s_per_step_512"] = round((t4 - t1) / 3, 2)
    torch.cuda.reset_peak_memory_stats(); mon.reset()
    _, t = gen(w=1024, h=512)
    R["t_1024x512_4step_s"] = round(t, 2)
    R["peak_alloc_1024x512_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    R["mon_1024x512"] = mon.report()


def do_step(R, mon):
    """Per-step cost at 512 from (4-step - 1-step)/3, three repeats."""
    pipe, dev = load_pipe(R); gen = make_gen(pipe, dev)
    gen(steps=1)
    vals = []
    for i in range(3):
        t4 = gen(steps=4, seed=i)[1]; t1 = gen(steps=1, seed=i)[1]
        vals.append(round((t4 - t1) / 3, 2))
    R["s_per_step_512_runs"] = vals
    R["s_per_step_512"] = sorted(vals)[1]


def do_pano(R, mon):
    import torch
    pipe, dev = load_pipe(R); gen = make_gen(pipe, dev)
    gen(steps=1)
    n = 8
    torch.cuda.reset_peak_memory_stats(); mon.reset()
    ts = [gen(seed=s)[1] for s in (1, 2)]
    R["t_2windows_seq_s"] = round(sum(ts), 2)
    R["pano_8win_seq_s_est"] = round(sum(ts) * n / 2, 1)
    for bs in (2, 4):
        torch.cuda.reset_peak_memory_stats()
        chunks = [gen(bs=bs, seed=10 + i)[1] for i in range(n // bs)]
        R[f"pano_8win_bs{bs}_chunks_s"] = [round(c, 2) for c in chunks]
        R[f"pano_8win_bs{bs}_s"] = round(sum(chunks), 1)  # measured, all 8 windows
        R[f"s_per_step_bs{bs}"] = round(sorted(chunks)[len(chunks) // 2] / 4, 2)
        R[f"peak_alloc_bs{bs}_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    R["mon_pano"] = mon.report()


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
    from huggingface_hub import hf_hub_download
    gp = hf_hub_download(GGUF_REPO, GGUF_FILE)
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
        sys.exit("painter 'engine' is not built yet (Task 1 rewires this bench to FluxPainter)")
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
