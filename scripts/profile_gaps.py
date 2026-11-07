"""GPU idle time between kernels over tile512 passes (Task 7, docs/engine/07-cuda-graphs.md).

Run under Nsight Systems (no sudo needed), then analyse the export:

  nsys profile -t cuda,nvtx --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi \
      --capture-range-end=stop -o /tmp/t7-gaps -f true uv run python scripts/profile_gaps.py --passes 3
  uv run python scripts/profile_gaps.py --analyse /tmp/t7-gaps.nsys-rep

The run loads the game's FluxPainter with the flags in the environment, warms up with two tile512 passes (graphs
capture on first use), then profiles --passes passes back to back (no sync between them, as a job runs them) in one
NVTX range "passes". The analysis takes the GPU span from the first to the last kernel in it and reports the time
when no kernel ran (copies on the side stream do not count as busy: they leave the SMs idle), per pass too, the
number of gaps and their size distribution, and the largest gaps.
"""
import argparse
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT = "a dirt path through a sunlit old forest clearing, open ground in the foreground, big trees, ferns"


def run(passes: int, batch: int) -> None:
    import torch
    from vecworld.config import FLUX_STYLE, FLUX_WIN
    from vecworld.flux.painter import FluxPainter

    p = FluxPainter()
    with torch.inference_mode():
        e = p._embed([(f"{PROMPT}, {FLUX_STYLE}", 1.0)])
        g = torch.randn(batch, 64, FLUX_WIN, FLUX_WIN, device="cuda", generator=torch.Generator("cuda").manual_seed(0))
        for _ in range(2):
            p._forward(g, [e] * batch, 1.0)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        torch.cuda.nvtx.range_push("passes")
        for k in range(passes):                 # back to back, as a job runs them (no sync between passes)
            p._forward(g, [e] * batch, 1.0 - k / 4)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
        torch.cuda.cudart().cudaProfilerStop()
    p.close()


def analyse(rep: Path, npass: int) -> dict:
    db = rep.with_suffix(".sqlite")
    if not db.exists():
        subprocess.run(["nsys", "export", "--type", "sqlite", "-f", "true", "-o", str(db), str(rep)], check=True,
                       stdout=subprocess.DEVNULL)
    con = sqlite3.connect(db)
    ranges = con.execute("SELECT start, end FROM NVTX_EVENTS WHERE text = 'passes' ORDER BY start").fetchall()
    kern = con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start").fetchall()
    try:
        cpy = con.execute("SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start").fetchall()
    except sqlite3.OperationalError:
        cpy = []
    out = []
    for a, b in ranges[:1]:
        ks = [(s, e) for s, e in kern if s >= a and e <= b]
        if not ks:
            continue
        span = ks[-1][1] - ks[0][0]
        busy, gaps, cur_s, cur_e = 0, [], ks[0][0], ks[0][1]
        for s, e in ks[1:]:
            if s > cur_e:
                busy += cur_e - cur_s
                gaps.append(s - cur_e)
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        busy += cur_e - cur_s
        idle = span - busy
        big = sorted(gaps, reverse=True)
        cb = sum(n for s, e, n in cpy if s >= a and e <= b)
        out.append(dict(span_ms=round(span / 1e6, 2), kernel_busy_ms=round(busy / 1e6, 2),
                        idle_ms=round(idle / 1e6, 2), idle_pct=round(100 * idle / span, 2), kernels=len(ks),
                        gaps=len(gaps), gaps_over_10us=sum(g > 1e4 for g in gaps),
                        gaps_over_100us=sum(g > 1e5 for g in gaps), gaps_over_1ms=sum(g > 1e6 for g in gaps),
                        idle_in_gaps_over_100us_ms=round(sum(g for g in gaps if g > 1e5) / 1e6, 2),
                        median_gap_us=round(sorted(gaps)[len(gaps) // 2] / 1e3, 2) if gaps else 0,
                        largest_gaps_us=[round(g / 1e3, 1) for g in big[:5]],
                        cpu_wall_ms=round((b - a) / 1e6, 2), memcpy_gb=round(cb / 1e9, 3)))
    return dict(report=str(rep), passes=npass, window=out[0], per_pass_idle_ms=round(out[0]["idle_ms"] / npass, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--analyse", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.analyse:
        r = analyse(a.analyse, a.passes)
        print(json.dumps(r, indent=1))
        if a.out:
            a.out.write_text(json.dumps(r, indent=1))
    else:
        sys.path.insert(0, str(ROOT / "src"))
        run(a.passes, a.batch)


if __name__ == "__main__":
    main()
