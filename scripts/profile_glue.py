"""Profile one tile512 transformer pass on the painter (VECWORLD_KERNEL / VECWORLD_FUSE as set): GPU kernel time per
op group. Kernels are attributed to the aten op that launched them, and that op to a group: "linear" (everything under
a Q4KLinear: quantize_act, GEMM, GEMV), "rope" (diffusers' apply_rotary_emb, fp32), "qk_rmsnorm", "sdpa", "gelu",
"memcpy" (group offload's copies, on their own stream, overlapped), "glue" (everything else in the blocks: norms,
modulate, gates, residuals, cats, dtype copies), "outside_blocks" (embedders, norm_out, proj_out).
Task 6 (docs/engine/06-glue-fusion.md). VW_PROF_NOQUANT=1 (measurement only) runs the fused glue without its fused
quantize. usage: uv run python scripts/profile_glue.py OUT.json"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_flux as B  # noqa: E402
import diffusers.models.transformers.transformer_flux as tf  # noqa: E402
from vecworld.config import FLUX_STYLE, FLUX_WIN  # noqa: E402
from vecworld.engine.q4k import Q4KLinear  # noqa: E402
from vecworld.flux.painter import FluxPainter  # noqa: E402


def tagged(label, fn):
    def run(*a, **k):
        with record_function(label):
            return fn(*a, **k)
    return run


tf.apply_rotary_emb = tagged("vw:rope", tf.apply_rotary_emb)
Q4KLinear.forward = tagged("vw:linear", Q4KLinear.forward)
if os.environ.get("VW_PROF_NOQUANT"):           # measurement only: fused glue without the fused quantize
    import vecworld.engine.blocks as _b
    _b._int8 = lambda *mods: False
p = FluxPainter()
with torch.inference_mode():
    e = p._embed([(f"{B.PROMPT}, {FLUX_STYLE}", 1.0)])
    g = torch.randn(1, 64, FLUX_WIN, FLUX_WIN, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
    for _ in range(2):
        p._forward(g, [e], 1.0)
    torch.cuda.synchronize()
    # after the warm-up: group offload re-registers its hooks on the first pass, rewriting each block's forward
    for blk in [*p.transformer.transformer_blocks, *p.transformer.single_transformer_blocks]:
        blk.forward = tagged("vw:block", blk.forward)
    walls = []
    for _ in range(5):
        t = time.time(); p._forward(g, [e], 1.0); torch.cuda.synchronize(); walls.append(time.time() - t)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        p._forward(g, [e], 1.0)
        torch.cuda.synchronize()


trace = Path(sys.argv[1]).with_suffix(".trace.json")
prof.export_chrome_trace(str(trace))
T = json.loads(trace.read_text())["traceEvents"]
launch = {}                                  # correlation id -> (pid, tid, ts) of the CUDA runtime launch
cpu = defaultdict(list)                      # (pid, tid) -> [(ts, end, name)] of ops and annotations
for ev in T:
    c = ev.get("cat", "")
    if c == "cuda_runtime" or c == "cuda_driver":
        launch[ev["args"].get("correlation")] = (ev["pid"], ev["tid"], ev["ts"])
    elif c in ("cpu_op", "user_annotation") and "dur" in ev:
        cpu[(ev["pid"], ev["tid"])].append((ev["ts"], ev["ts"] + ev["dur"], ev["name"]))


def context(corr):
    """(innermost cpu op name, set of vw: labels) around the launch of a kernel."""
    if corr not in launch:
        return "?", set()
    pid, tid, ts = launch[corr]
    inside = [(b, e, n) for b, e, n in cpu[(pid, tid)] if b <= ts <= e]
    labs = {n for _, _, n in inside if n.startswith("vw:")}
    ops = [x for x in inside if not x[2].startswith("vw:")]
    inner = min(ops, key=lambda x: x[1] - x[0])[2] if ops else "?"
    return inner, labs


def group(name, inner, lab):
    if "vw:linear" in lab:
        return "linear"
    if "vw:rope" in lab:
        return "rope"
    if "rms_norm" in inner or "rms_norm" in name:
        return "qk_rmsnorm"
    if "flash" in name or "attention" in inner:
        return "sdpa"
    if "gelu" in inner:
        return "gelu"
    if "vw:block" not in lab:
        return "outside_blocks"
    return "glue"


groups, glue_ops, mem = defaultdict(float), defaultdict(lambda: [0.0, 0]), defaultdict(float)
lin_k = defaultdict(lambda: [0.0, 0])
for ev in T:
    if ev.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
        continue
    ms = ev.get("dur", 0) / 1e3
    if ev["cat"] != "kernel":
        mem[ev["name"]] += ms
        continue
    inner, lab = context(ev["args"].get("correlation"))
    gname = group(ev["name"], inner, lab)
    groups[gname] += ms
    if gname == "linear":
        kn = next((t for t in ("quantize_kernel", "gemm_w4a8", "gemm_mma", "gemv", "gemm_naive", "gemm_tiled")
                   if t in ev["name"]), ev["name"][:60])
        lin_k[kn][0] += ms
        lin_k[kn][1] += 1
    if gname == "glue":
        glue_ops[inner][0] += ms
        glue_ops[inner][1] += 1
trace.unlink()
compute = sum(groups.values())
R = dict(kernel=p.kernel, fused=p.fused, wall_s_median=round(sorted(walls)[2], 3), walls=[round(w, 3) for w in walls],
         compute_stream_ms=round(compute, 1),
         groups_ms={k: round(v, 1) for k, v in sorted(groups.items(), key=lambda x: -x[1])},
         glue_ops_ms={k: [round(v[0], 2), v[1]] for k, v in sorted(glue_ops.items(), key=lambda x: -x[1][0])},
         linear_kernels_ms={k: [round(v[0], 2), v[1]] for k, v in sorted(lin_k.items(), key=lambda x: -x[1][0])},
         gpu_clocks=os.popen("nvidia-smi --query-gpu=clocks.sm,power.draw,temperature.gpu --format=csv,noheader").read().strip(),
         memcpy_ms={k: round(v, 1) for k, v in mem.items()})
Path(sys.argv[1]).write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1))
