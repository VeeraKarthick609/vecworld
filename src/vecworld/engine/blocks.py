"""FLUX block forwards that run their glue fused (Task 6, docs/engine/06-glue-fusion.md).

diffusers' FluxTransformerBlock and FluxSingleTransformerBlock spend their non-GEMM time in elementwise passes over
the token rows: a LayerNorm, then "* (1 + scale)", then "+ shift"; "gate * out", then "residual + ..."; the q / k
RMSNorm and the rotary embedding (in fp32, ~6 passes); torch.cat of text and image q / k / v, of the single blocks'
streams and of their [attention | MLP]. Each pass reads and writes a whole activation. The forwards here do the same
arithmetic, rounded where diffusers rounds, with one pass per norm (glue.residual_norm_modulate), one per bare
residual (glue.gated_residual) and one per q / k projection (glue.qk_norm_rope, writing into the joint q / k), and
without the q / k and stream cats. Every linear, the attention kernel (dispatch_attention_fn, as the processor calls
it) and the GELU are diffusers'; an attention processor other than the plain FluxAttnProcessor runs as is.

Under w4a8 a norm's output also carries its INT8 quantization when one of its consumers is a w4a8 Q4KLinear, so
to_q / to_k / proj_mlp / ff.net.0 don't each re-read it to quantize it; keep-listed (mma) consumers take the bf16 x.
"""
import types
from collections import Counter

import torch
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor, dispatch_attention_fn

from vecworld.engine.glue import gated_residual, qk_norm_rope, residual_norm_modulate
from vecworld.engine.q4k import Q4KLinear, attach_quantized, quantized_of

# Python-time counts of the fused forwards ("double", "single") and of norms that attached a quantization
# ("quantize"): eager calls and CUDA-graph captures add to them, a graph replay runs no Python and adds nothing
# (Task 7: a test can see the fused blocks were the ones captured).
CALLS: Counter = Counter()


def _int8(*mods) -> bool:
    """Whether any of these linears quantizes its input (a Q4KLinear on w4a8)."""
    return any(isinstance(m, Q4KLinear) and m.kernel == "w4a8" for m in mods)


def _rnm(res, y, gate, shift, scale, quantize):
    """residual_norm_modulate on [B, L, H] activations; returns (res_out, x_mod) in [B, L, H]."""
    shape = res.shape
    r2, x2 = residual_norm_modulate(res.reshape(-1, shape[-1]), None if y is None else y.reshape(-1, shape[-1]),
                                    gate, shift, scale, quantize=quantize)
    x = x2.view(shape)
    q = quantized_of(x2) if quantize else None
    if q is not None:                              # view() makes a new tensor object: carry the quantization over
        attach_quantized(x, q)
        CALLS["quantize"] += 1
    return r2.view(shape), x


def _gres(res, y, gate, out=None):
    """gated_residual on [B, L, H] activations, into out ([B, L, H], contiguous) if given."""
    shape = res.shape
    o = None if out is None else out.view(-1, shape[-1])
    return gated_residual(res.reshape(-1, shape[-1]), y.reshape(-1, shape[-1]), gate, out=o).view(shape)


def _c(t):
    return t if t.is_contiguous() else t.contiguous()


def _plain(attn, kwargs) -> bool:
    """Whether attn runs diffusers' plain FluxAttnProcessor on separate projections with no extra arguments: the
    case _attention reproduces. Anything else (an IP-adapter processor, fused qkv projections, attention kwargs)
    goes through attn itself."""
    return type(attn.processor) is FluxAttnProcessor and not attn.fused_projections and not kwargs


def _attention(attn, x, cx, rot):
    """FluxAttnProcessor's attention with q / k RMSNorm + rotary fused into one pass per projection (qk_norm_rope),
    written straight into the joint [B, T + L, heads, 128] q and k (no torch.cat for them). x [B, L, H] (the image
    or joint stream), cx [B, T, H] or None (a double block's text stream). Returns the attention output [B, T + L,
    H] (text first), as the processor computes it before its output projections."""
    B, L, _ = x.shape
    T = 0 if cx is None else cx.shape[1]
    D, cos, sin = attn.head_dim, rot[0], rot[1]
    q_parts = [(attn.to_q(x), attn.norm_q, T), (attn.to_k(x), attn.norm_k, T)]
    v = attn.to_v(x)
    if cx is not None:
        q_parts += [(attn.add_q_proj(cx), attn.norm_added_q, 0), (attn.add_k_proj(cx), attn.norm_added_k, 0)]
        v = torch.cat([attn.add_v_proj(cx), v], dim=1)
    nh = q_parts[0][0].shape[-1] // D
    q = x.new_empty(B, T + L, nh, D)
    k = x.new_empty(B, T + L, nh, D)
    for i, (t, norm, off) in enumerate(q_parts):
        qk_norm_rope(_c(t), norm.weight, cos, sin, q if i % 2 == 0 else k, off, norm.eps)
    o = dispatch_attention_fn(q, k, v.unflatten(-1, (-1, D)), attn_mask=None,
                              backend=attn.processor._attention_backend,
                              parallel_config=attn.processor._parallel_config)
    return o.flatten(2, 3).to(q.dtype)


def _joint(es, hs):
    """torch.cat([es, hs], dim=1), without the copy when es and hs are the two halves of one [B, T + L, H] buffer
    (a single block's output, handed to the next single block as two slices)."""
    B, T, H = es.shape
    L = hs.shape[1]
    st = ((T + L) * H, H, 1)
    if (es.untyped_storage().data_ptr() == hs.untyped_storage().data_ptr() and es.stride() == hs.stride() == st
            and hs.data_ptr() == es.data_ptr() + T * H * es.element_size() and es.dtype == hs.dtype):
        return es.as_strided((B, T + L, H), st)
    return torch.cat([es, hs], dim=1)


def double_forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor,
                   image_rotary_emb=None, joint_attention_kwargs=None, out=None):
    """FluxTransformerBlock.forward with fused glue (same signature and outputs: (encoder_hidden_states,
    hidden_states)). out: (es, hs) buffers ([B, T, H], [B, L, H], contiguous) the block's last passes write its two
    outputs into, instead of new tensors (a CUDA graph's fixed output buffers, engine/graphs.py)."""
    CALLS["double"] += 1
    hs, es = _c(hidden_states), _c(encoder_hidden_states)
    attn = self.attn
    emb = self.norm1.linear(self.norm1.silu(temb))
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
    cemb = self.norm1_context.linear(self.norm1_context.silu(temb))
    c_shift_msa, c_scale_msa, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = cemb.chunk(6, dim=1)

    _, x = _rnm(hs, None, None, shift_msa, scale_msa, _int8(attn.to_q, attn.to_k, attn.to_v))
    _, cx = _rnm(es, None, None, c_shift_msa, c_scale_msa, _int8(attn.add_q_proj, attn.add_k_proj, attn.add_v_proj))
    if _plain(attn, joint_attention_kwargs) and image_rotary_emb is not None:
        o = _attention(attn, x, cx, image_rotary_emb)
        T = cx.shape[1]
        context_attn_output, attn_output = o[:, :T], o[:, T:]
        outs = (attn.to_out[1](attn.to_out[0](attn_output.contiguous())),
                attn.to_add_out(context_attn_output.contiguous()))
    else:
        outs = attn(hidden_states=x, encoder_hidden_states=cx, image_rotary_emb=image_rotary_emb,
                    **(joint_attention_kwargs or {}))
    attn_output, context_attn_output = outs[0], outs[1]

    ip = len(outs) == 3                            # an IP-adapter's attention output, as diffusers adds it
    hs, x = _rnm(hs, _c(attn_output), gate_msa, shift_mlp, scale_mlp, _int8(self.ff.net[0].proj))
    hs = _gres(hs, _c(self.ff(x)), gate_mlp, None if out is None or ip else out[1])
    if ip:
        hs = hs + outs[2] if out is None else torch.add(hs, outs[2], out=out[1])

    es, cx = _rnm(es, _c(context_attn_output), c_gate_msa, c_shift_mlp, c_scale_mlp,
                  _int8(self.ff_context.net[0].proj))
    es = _gres(es, _c(self.ff_context(cx)), c_gate_mlp, None if out is None else out[0])
    return es, hs


def single_forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor,
                   image_rotary_emb=None, joint_attention_kwargs=None, out=None):
    """FluxSingleTransformerBlock.forward with fused glue (same signature and outputs). out: a [B, T + L, H] buffer
    (contiguous) the last pass writes the joint output into; the outputs are then its two halves."""
    CALLS["single"] += 1
    text_seq_len = encoder_hidden_states.shape[1]
    joint = _joint(encoder_hidden_states, hidden_states)
    attn = self.attn
    emb = self.norm.linear(self.norm.silu(temb))
    shift_msa, scale_msa, gate = emb.chunk(3, dim=1)
    _, x = _rnm(joint, None, None, shift_msa, scale_msa, _int8(attn.to_q, attn.to_k, attn.to_v, self.proj_mlp))
    H = joint.shape[-1]
    # proj_out's input [attention | GELU(mlp)], built in place: the GELU writes its 4H columns directly (diffusers
    # concatenates the two after the fact, one more pass over 5H columns)
    cat = joint.new_empty(*joint.shape[:2], H + self.mlp_hidden_dim)
    torch._C._nn.gelu(self.proj_mlp(x), approximate=self.act_mlp.approximate, out=cat[..., H:])
    if _plain(attn, joint_attention_kwargs) and image_rotary_emb is not None:
        cat[..., :H] = _attention(attn, x, None, image_rotary_emb)
    else:
        cat[..., :H] = attn(hidden_states=x, image_rotary_emb=image_rotary_emb, **(joint_attention_kwargs or {}))
    joint = _gres(joint, self.proj_out(cat), gate, out)
    return joint[:, :text_seq_len], joint[:, text_seq_len:]


def patch_blocks(transformer: torch.nn.Module) -> int:
    """Give each of the transformer's double blocks (transformer_blocks) and single blocks (single_transformer_blocks)
    the fused forward, as an instance attribute (del block.forward restores diffusers'). Call it before group offload:
    its hooks wrap each block's forward as it is when they are installed. Returns how many blocks it patched (a block
    already patched is skipped)."""
    n = 0
    for blocks, fwd in ((getattr(transformer, "transformer_blocks", ()), double_forward),
                        (getattr(transformer, "single_transformer_blocks", ()), single_forward)):
        for blk in blocks:
            f = blk.__dict__.get("forward")
            if f is not None and getattr(f, "__func__", None) is fwd:
                continue
            blk.forward = types.MethodType(fwd, blk)
            n += 1
    return n
