"""The hand-written T5 encoder pieces (vecworld.flux.text) against transformers and gguf (CPU, no models)."""
import gguf
import numpy as np
import pytest
import torch

from vecworld.flux.text import N_BUCKETS, MAX_DIST, _bucket, _block, _Gguf

Q = gguf.GGMLQuantizationType


def test_bucket_matches_transformers():
    from transformers.models.t5.modeling_t5 import T5Attention

    rel = torch.arange(-400, 401)[None, :].repeat(3, 1)
    ref = T5Attention._relative_position_bucket(rel, bidirectional=True, num_buckets=N_BUCKETS,
                                                max_distance=MAX_DIST)
    assert torch.equal(_bucket(rel), ref)


def test_block_matches_transformers_t5block_gated_gelu():
    """One encoder block (with the relative-position bias, as block 0 has) against transformers' T5Block on a
    small random config with T5 v1.1's gated-gelu (gelu_new) feed-forward."""
    from transformers import T5Config
    from transformers.models.t5.modeling_t5 import T5Block

    torch.manual_seed(0)
    cfg = T5Config(d_model=64, d_kv=16, num_heads=4, d_ff=96, feed_forward_proj="gated-gelu",
                   relative_attention_num_buckets=N_BUCKETS, relative_attention_max_distance=MAX_DIST,
                   layer_norm_epsilon=1e-6, is_decoder=False, use_cache=False, dropout_rate=0.0)
    cfg._attn_implementation = "eager"
    assert cfg.dense_act_fn == "gelu_new"
    blk = T5Block(cfg, has_relative_attention_bias=True, layer_idx=0).eval()
    with torch.no_grad():
        for prm in blk.parameters():
            prm.normal_(0, 0.3)
    sa, ff = blk.layer[0].SelfAttention, blk.layer[1].DenseReluDense
    W = {"attn_q": sa.q.weight, "attn_k": sa.k.weight, "attn_v": sa.v.weight, "attn_o": sa.o.weight,
         "attn_norm": blk.layer[0].layer_norm.weight, "ffn_gate": ff.wi_0.weight, "ffn_up": ff.wi_1.weight,
         "ffn_down": ff.wo.weight, "ffn_norm": blk.layer[1].layer_norm.weight}
    L = 150
    h = torch.randn(1, L, 64)
    pos = torch.arange(L)
    bias = sa.relative_attention_bias.weight[_bucket(pos[None, :] - pos[:, None])].permute(2, 0, 1)[None]
    with torch.no_grad():
        ref = blk(h, attention_mask=None, position_bias=None)[0]
        mine = _block(h, {k: v.detach() for k, v in W.items()}, bias.detach(), cfg.num_heads)
    assert torch.allclose(mine, ref, atol=1e-4, rtol=1e-4), (mine - ref).abs().max()


def _random_kquant(rng, rows: int, qt) -> np.ndarray:
    """Random K-quant blocks with finite fp16 scale fields (gguf-py can't quantize K-quants, but any bytes with
    sane scales are a valid block)."""
    bs, ts = gguf.GGML_QUANT_SIZES[qt]
    raw = rng.integers(0, 256, (rows, 2 * (256 // bs) * ts // 2), dtype=np.uint8).reshape(rows, -1, ts)
    scales = rng.uniform(0.01, 0.1, raw.shape[:2]).astype(np.float16).view(np.uint8).reshape(*raw.shape[:2], 2)
    if qt == Q.Q6_K:                       # ql[128] qh[64] scales[16] d(fp16)
        raw[..., -2:] = scales
    else:                                  # Q5_K: d(fp16) dmin(fp16) scales[12] qh[32] qs[128]
        raw[..., 0:2] = scales
        raw[..., 2:4] = scales
    return raw.reshape(rows, -1)


@pytest.fixture
def tiny_gguf(tmp_path):
    rng = np.random.default_rng(1)
    path = str(tmp_path / "t5.gguf")
    w = gguf.GGUFWriter(path, "t5encoder")
    w.add_uint32("t5encoder.block_count", 1)
    w.add_uint32("t5encoder.attention.head_count", 4)
    tensors = {}
    a = rng.standard_normal((6, 512)).astype(np.float32)
    tensors["f32"] = (a, Q.F32)
    for qt in (Q.Q8_0, Q.Q5_0):
        tensors[qt.name] = (gguf.quants.quantize(a, qt), qt)
    for qt in (Q.Q5_K, Q.Q6_K):
        tensors[qt.name] = (_random_kquant(rng, 6, qt), qt)
    for name, (data, qt) in tensors.items():
        if qt == Q.F32:
            w.add_tensor(name, data)
        else:
            w.add_tensor(name, data, raw_dtype=qt)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
    return path, tensors


@pytest.mark.parametrize("name", ["f32", "Q8_0", "Q5_0", "Q5_K", "Q6_K"])
@pytest.mark.parametrize("rows", [None, [3, 0, 5]])
def test_gguf_rows_dequantize_like_gguf_py(tiny_gguf, name, rows):
    path, tensors = tiny_gguf
    data, qt = tensors[name]
    ref = gguf.quants.dequantize(data, qt).astype(np.float32)
    ref = ref if rows is None else ref[rows]
    g = _Gguf(path)
    try:
        assert g.n_blocks == 1 and g.n_heads == 4
        mine = g.tensor(name, "cpu", rows=rows)
    finally:
        g.close()
    assert mine.dtype == torch.bfloat16 and tuple(mine.shape) == ref.shape
    # The dequant runs in bf16 (8-bit mantissa), scales rounded to bf16 first, and Q5_K subtracts dmin * m, so
    # the error is relative to the tensor's magnitude, not each value: allow 2^-7 of the largest.
    err = np.abs(mine.float().numpy() - ref).max()
    assert err <= 2 ** -7 * np.abs(ref).max(), (name, err, np.abs(ref).max())
