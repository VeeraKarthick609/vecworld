"""The glue between a FLUX block's GEMMs, fused (Task 6, docs/engine/06-glue-fusion.md): residual add, LayerNorm and
modulate in one pass over each token row, and the attention inputs' RMSNorm + rotary embedding in one pass over
each head (csrc/glue.cu), rounded where the diffusers reference rounds."""
import torch

from vecworld.engine.q4k import attach_quantized


def _rows(t: torch.Tensor) -> torch.Tensor:
    """A [B, H] modulation row tensor as the kernel takes it: rows may be strided (chunks of one modulation output),
    the last dimension contiguous."""
    return t if t.stride(-1) == 1 else t.contiguous()


def residual_norm_modulate(res: torch.Tensor, y: torch.Tensor | None, gate: torch.Tensor | None,
                           shift: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6,
                           quantize: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """res_out = res + gate * y (skipped when y is None: res_out is res itself), then
    x_mod = LayerNorm(res_out) * (1 + scale) + shift (no affine, fp32 statistics); one read of res and y, one write
    of each output. res, y: bf16 [M, H]; gate, shift, scale: bf16 [B, H], one row per image, broadcast over that
    image's M / B tokens.

    quantize: also run W4A8's quantize_act on x_mod in the same pass and attach its outputs to x_mod
    (q4k.attach_quantized), bit-identical to quantize_act(x_mod); a w4a8 Q4KLinear given x_mod uses them, any
    other linear ignores them. Returns (res_out, x_mod)."""
    from vecworld.engine.ext import load_ext

    if (y is None) != (gate is None):
        raise ValueError("residual_norm_modulate: y and gate come together (or neither)")
    out = load_ext().glue_rnm(res, y, None if gate is None else _rows(gate), _rows(shift), _rows(scale), eps,
                              quantize)
    res_out, x = out[0], out[1]
    if quantize:
        attach_quantized(x, (out[2], out[3], out[4]))
    return res_out, x


def gated_residual(res: torch.Tensor, y: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """res + gate * y in one pass (a block's last residual, which no norm follows): res, y bf16 [M, H], gate bf16
    [B, H] broadcast per image as above; gate * y is rounded to bf16 before the add, as the reference rounds."""
    from vecworld.engine.ext import load_ext

    return load_ext().glue_gated_residual(res, y, _rows(gate))


def qk_norm_rope(x: torch.Tensor, weight: torch.Tensor | None, cos: torch.Tensor, sin: torch.Tensor,
                 out: torch.Tensor, offset: int, eps: float) -> torch.Tensor:
    """An attention input's per-head RMSNorm (weight [128] or None, eps) and FLUX's rotary embedding (cos / sin fp32
    [S_out, 128], diffusers' FluxPosEmbed) in one pass: x bf16 [B, S, heads * 128] (a q / k projection's output) is
    written into out[:, offset:offset + S], out bf16 [B, S_out, heads, 128] (text at 0, image after it), at
    positions offset.. of cos / sin. Rounded where torch rounds (rms_norm, then apply_rotary_emb in fp32). Returns
    out."""
    from vecworld.engine.ext import load_ext

    load_ext().glue_qk_norm_rope(x, weight, cos, sin, out, offset, eps)
    return out
