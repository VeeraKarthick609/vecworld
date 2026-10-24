"""Task 5b, step 1: how much would per-token INT8 activations hurt each Q4_K linear of FLUX? (docs/engine/04-w4a8.md)

Runs the real FluxPainter on its W4A16 kernel (mma) and, inside every Q4KLinear call, also computes in fp32 what that
linear would output if its input were fake-quantized per token (s = max|x_t| / 127, xq = round(x_t / s), x ~ xq * s),
against the same linear on the unquantized input. The pass itself always continues on the mma output, so each layer's
error is its own, not the compounded error of the layers before it. Two contexts:

  tile512: one real 512 x 512 image (one window, 1024 image + 256 text tokens), FLUX_STEPS passes from seed 0;
  step:    one `sharp` walk step (as bench_flux --what step: the seed-21 panorama dollied 1.6, then repainted).

Per layer: relative L2 error of the output and its max-rel, the activations' max/median |x| ratio per token (median
and max over tokens) and per channel (largest channel's max |x| over the median channel's), and its FLOPs per 512
pass. Layers are ranked by relative L2 (the worse of the two contexts).

  VECWORLD_KERNEL=mma uv run python scripts/w4a8_study.py --out bench/t5b-study.json
  uv run python scripts/w4a8_study.py --share 'transformer_blocks.*.ff.net.2,single_transformer_blocks.*.proj_out'
"""
import argparse
import fnmatch
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STUDY = ROOT / "bench/t5b-study.json"
TOKEN_FACING_M = 256          # the modulation linears (M = windows per pass) are not token-facing
CHUNK = 2048                  # rows per fp32 matmul (VRAM)


def fake_quant(x):
    """Per-token INT8 fake quantization of x [M, K] (fp32): xq * s, s = max|x_t| / 127 (a zero row stays zero)."""
    import torch

    s = x.abs().amax(1, keepdim=True) / 127
    xq = torch.where(s > 0, torch.round(x / torch.where(s > 0, s, 1)).clamp(-127, 127), 0)
    return xq * s


class Stats:
    def __init__(self):
        self.err2 = self.ref2 = 0.0
        self.maxrel = 0.0
        self.tok_med, self.tok_max, self.chan = [], 0.0, 0.0
        self.calls, self.m = 0, None

    def add(self, x, w):
        import torch

        x = x.float()
        self.calls += 1
        self.m = self.m or x.shape[0]
        a = x.abs()
        med = a.median(1).values.clamp_min(1e-12)
        r = a.amax(1) / med
        self.tok_med.append(float(r.median()))
        self.tok_max = max(self.tok_max, float(r.max()))
        cm = a.amax(0)
        self.chan = max(self.chan, float(cm.max() / cm.median().clamp_min(1e-12)))
        e2 = r2 = 0.0
        mx_err = mx_ref = 0.0
        for i in range(0, x.shape[0], CHUNK):
            xc = x[i:i + CHUNK]
            y16 = xc @ w.T
            y8 = fake_quant(xc) @ w.T
            d = y8 - y16
            e2 += float((d * d).sum())
            r2 += float((y16 * y16).sum())
            mx_err, mx_ref = max(mx_err, float(d.abs().max())), max(mx_ref, float(y16.abs().max()))
            del y16, y8, d
        self.err2 += e2
        self.ref2 += r2
        self.maxrel = max(self.maxrel, mx_err / max(mx_ref, 1e-30))
        del a, cm, x
        torch.cuda.empty_cache() if self.calls % 64 == 0 else None

    def row(self):
        return {"rel_l2": (self.err2 / max(self.ref2, 1e-30)) ** 0.5, "max_rel": self.maxrel,
                "token_ratio_median": sorted(self.tok_med)[len(self.tok_med) // 2] if self.tok_med else None,
                "token_ratio_max": self.tok_max, "channel_ratio": self.chan, "calls": self.calls}


def family(name: str) -> str:
    """The layer's name with its block index replaced by *: one keep-list pattern per family."""
    return re.sub(r"(transformer_blocks)\.\d+\.", r"\1.*.", name)


def run(out: Path):
    import torch

    sys.path.insert(0, str(ROOT / "scripts"))
    import bench_flux as bench
    from vecworld.config import FLUX_STEPS
    from vecworld.dream.pano import dolly
    from vecworld.engine.ext import load_ext
    from vecworld.engine.q4k import Q4KLinear
    from vecworld.roam.spot import focus_half

    mon = bench.Mon()
    R = {}
    p, prompt, _ = bench.load_painter(R, mon)
    if p.kernel != "mma":
        sys.exit(f"the study measures against W4A16: run it with VECWORLD_KERNEL=mma (got {p.kernel})")
    ext = load_ext()
    names = {id(m): n for n, m in p.transformer.named_modules() if isinstance(m, Q4KLinear)}
    shapes = {n: (m.in_features, m.out_features) for n, m in p.transformer.named_modules() if isinstance(m, Q4KLinear)}
    stats = {"tile512": {}, "step": {}}
    ctx = [None]

    def hook(mod, inputs, output):
        if ctx[0] is None:
            return
        x = inputs[0].reshape(-1, mod.in_features)
        w = ext.q4k_dequant(mod.qweight, mod.in_features)
        stats[ctx[0]].setdefault(names[id(mod)], Stats()).add(x, w)
        del w

    hooks = [m.register_forward_hook(hook) for m in p.transformer.modules() if isinstance(m, Q4KLinear)]
    t0 = time.time()
    with torch.inference_mode():
        e = p._embed(prompt)
        ctx[0] = "tile512"
        bench.run_tile512(p, e, FLUX_STEPS, seed=0)
        print(f"tile512 done {time.time() - t0:.0f}s", flush=True)
        ctx[0] = None
        prev = dolly(p.panorama(prompt, seed=21), 1.6)
        ctx[0] = "step"
        p.step(prev, prompt, None, 0.5, seed=22, focus_half=focus_half(1.6))
        ctx[0] = None
        print(f"step done {time.time() - t0:.0f}s", flush=True)
    for h in hooks:
        h.remove()
    p.close()

    layers = []
    for n, (k, nn) in shapes.items():
        t, s = stats["tile512"].get(n), stats["step"].get(n)
        if t is None:
            continue
        rt, rs = t.row(), (s.row() if s else None)
        layers.append({"name": n, "family": family(n), "k": k, "n": nn, "m_512": t.m,
                       "flops_512": 2 * t.m * k * nn, "token_facing": t.m >= TOKEN_FACING_M,
                       "tile512": rt, "step": rs,
                       "rel_l2": max(rt["rel_l2"], rs["rel_l2"] if rs else 0.0)})
    layers.sort(key=lambda r: -r["rel_l2"])
    fams = {}
    for r in layers:
        f = fams.setdefault(r["family"], {"family": r["family"], "layers": 0, "flops_512": 0, "rel_l2_max": 0.0,
                                          "rel_l2_sum": 0.0, "token_facing": r["token_facing"]})
        f["layers"] += 1
        f["flops_512"] += r["flops_512"]
        f["rel_l2_max"] = max(f["rel_l2_max"], r["rel_l2"])
        f["rel_l2_sum"] += r["rel_l2"]
    for f in fams.values():
        f["rel_l2_mean"] = f.pop("rel_l2_sum") / f["layers"]
    res = {"power": bench.power(), "git": bench.git_sha(), "wall_s": round(time.time() - t0, 1),
           "contexts": {"tile512": f"one 512 window, {FLUX_STEPS} passes, seed 0",
                        "step": "one sharp walk step (seed-21 panorama dollied 1.6, seed 22)"},
           "families": sorted(fams.values(), key=lambda f: -f["rel_l2_max"]), "layers": layers}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print_table(res)


def print_table(res, top=40):
    print(f"{'family':48s} {'n':>3s} {'rel L2 max':>10s} {'mean':>8s} {'GFLOP/512':>10s}")
    for f in res["families"]:
        print(f"{f['family']:48s} {f['layers']:3d} {f['rel_l2_max']:10.2e} {f['rel_l2_mean']:8.2e} "
              f"{f['flops_512'] / 1e9:10.1f}")
    print(f"\ntop {top} layers")
    for r in res["layers"][:top]:
        s = r["step"] or {}
        print(f"{r['name']:48s} rel L2 512 {r['tile512']['rel_l2']:.2e} step {s.get('rel_l2', 0):.2e}  "
              f"tok {r['tile512']['token_ratio_median']:.1f}/{max(r['tile512']['token_ratio_max'], s.get('token_ratio_max', 0)):.0f}"
              f"  chan {max(r['tile512']['channel_ratio'], s.get('channel_ratio', 0)):.0f}")


def int8_share(keep, study: Path = STUDY) -> float:
    """The share of the token-facing linears' FLOPs (per 512 pass) that runs on INT8 when the layers matching keep
    (fnmatch patterns) stay on mma."""
    layers = [r for r in json.loads(study.read_text())["layers"] if r["token_facing"]]
    tot = sum(r["flops_512"] for r in layers)
    kept = sum(r["flops_512"] for r in layers if any(fnmatch.fnmatchcase(r["name"], p) for p in keep))
    return 1 - kept / tot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=STUDY)
    ap.add_argument("--share", help="comma-separated keep patterns: print the INT8 share of token-facing FLOPs")
    ap.add_argument("--show", action="store_true", help="print the table of an existing study")
    a = ap.parse_args()
    if a.share is not None:
        keep = [p for p in a.share.split(",") if p]
        print(f"{100 * int8_share(keep, a.out):.1f}% of token-facing FLOPs on INT8 with keep {keep}")
        return
    if a.show:
        print_table(json.loads(a.out.read_text()))
        return
    run(a.out)


if __name__ == "__main__":
    main()
