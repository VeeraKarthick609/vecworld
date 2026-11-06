"""Microbench the glue kernels at FLUX shapes against the eager torch ops they replace: time per call, and achieved
GB/s on the bytes each pass must move (the fused kernel's own traffic). Task 6 (docs/engine/06-glue-fusion.md).
usage: uv run python scripts/micro_glue.py [label]"""
import json, sys
import torch
from vecworld.engine.glue import qk_norm_rope, residual_norm_modulate, gated_residual
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux import FluxPosEmbed

H, dev, bf = 3072, "cuda", torch.bfloat16


def t(fn, n=50):
    fn(); torch.cuda.synchronize()
    best = []
    for _ in range(3):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record(); torch.cuda.synchronize()
        best.append(s.elapsed_time(e) / n * 1e3)
    return min(best)            # us


R = {"label": sys.argv[1] if len(sys.argv) > 1 else ""}
with torch.inference_mode():
    for M, y_on, q_on in ((1024, False, True), (1024, True, True), (256, True, False), (1280, False, True), (5120, True, False)):
        res = torch.randn(M, H, device=dev, dtype=bf); y = torch.randn(M, H, device=dev, dtype=bf) if y_on else None
        emb = torch.randn(1, 6 * H, device=dev, dtype=bf)
        sh, sc, g = emb[:, :H], emb[:, H:2 * H], emb[:, 2 * H:3 * H]
        us = t(lambda: residual_norm_modulate(res, y, g if y_on else None, sh, sc, quantize=q_on))
        nbytes = M * H * 2 * (1 + 2 * y_on + 1) + (M * H * 1.125 + M * 4 if q_on else 0)
        r = res.view(1, M, H)

        def eager():
            rr = r + g[:, None] * y.view(1, M, H) if y_on else r
            return torch.nn.functional.layer_norm(rr, (H,), eps=1e-6) * (1 + sc[:, None]) + sh[:, None]
        R[f"rnm M={M} y={y_on} q={q_on}"] = dict(us=round(us, 1), GBps=round(nbytes / us / 1e3, 1),
                                                 eager_us=round(t(eager), 1))
    M = 1024
    res, y = torch.randn(M, H, device=dev, dtype=bf), torch.randn(M, H, device=dev, dtype=bf)
    g = torch.randn(1, H, device=dev, dtype=bf)
    us = t(lambda: gated_residual(res, y, g))
    R["gated_residual M=1024"] = dict(us=round(us, 1), GBps=round(M * H * 6 / us / 1e3, 1),
                                      eager_us=round(t(lambda: res + g * y), 1))
    T, L, B = 256, 1024, 1
    ids = torch.zeros(T + L, 3, device=dev); ids[T:, 1] = torch.arange(L, device=dev) // 32; ids[T:, 2] = torch.arange(L, device=dev) % 32
    cos, sin = FluxPosEmbed(theta=10000, axes_dim=[16, 56, 56])(ids)
    x = torch.randn(B, T + L, H, device=dev, dtype=bf); w = torch.ones(128, device=dev, dtype=bf)
    out = torch.empty(B, T + L, 24, 128, device=dev, dtype=bf)
    us = t(lambda: qk_norm_rope(x, w, cos, sin, out, 0, 1e-6))
    nb = (T + L) * H * 2 * 2 + 2 * (T + L) * 128 * 4

    def eager_qk():
        n = torch.nn.functional.rms_norm(x.unflatten(-1, (24, 128)), (128,), w, 1e-6)
        return apply_rotary_emb(n, (cos, sin), sequence_dim=1)
    R["qk_norm_rope S=1280"] = dict(us=round(us, 1), GBps=round(nb / us / 1e3, 1), eager_us=round(t(eager_qk), 1))
print(json.dumps(R, indent=1))
