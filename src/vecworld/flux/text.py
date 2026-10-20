"""T5-XXL text embeddings for FLUX, run block by block on the GPU straight from the GGUF.

The spike's route (transformers loading the GGUF and dequantizing all of it to bf16 on the CPU) peaked at 12.4 GB
of RAM, which can't sit next to FLUX in 14 GB. Here nothing is kept between encodes but the tokenizer and an LRU
of finished embeddings: on a miss each of the 24 encoder blocks is read from the file (pread, no mmap, so the
weights never count toward the process RSS), dequantized on the GPU, run, and freed before the next one.
"""
import math
import os
import threading
from collections import OrderedDict

import numpy as np

from vecworld.config import FLUX_BASE, FLUX_T5_GGUF, FLUX_TEXT_LEN

N_BUCKETS, MAX_DIST = 32, 128     # T5 v1.1 relative attention
EPS = 1e-6


def _bucket(rel):
    """T5's bidirectional relative-position bucket for key_pos - query_pos."""
    import torch

    half = N_BUCKETS // 2
    out = (rel > 0).long() * half
    n = rel.abs()
    exact = half // 2
    big = exact + (torch.log(n.float().clamp(min=1) / exact) / math.log(MAX_DIST / exact) * (half - exact)).long()
    return out + torch.where(n < exact, n, big.clamp(max=half - 1))


class _Gguf:
    """Tensor metadata from the GGUF; bytes are read on demand with pread, never mapped."""

    def __init__(self, path):
        import gguf

        r = gguf.GGUFReader(path)
        self.meta = {t.name: (t.tensor_type, [int(d) for d in reversed(t.shape)], int(t.data_offset), int(t.n_bytes))
                     for t in r.tensors}
        self.n_blocks = int(r.fields["t5encoder.block_count"].parts[-1][0])
        self.n_heads = int(r.fields["t5encoder.attention.head_count"].parts[-1][0])
        del r                                       # drops the reader's memmap of the whole file
        self.fd = os.open(path, os.O_RDONLY)

    def close(self):
        os.close(self.fd)

    def raw(self, name) -> np.ndarray:
        _, _, off, n = self.meta[name]
        buf = bytearray(n)
        mv = memoryview(buf)
        got = 0
        while got < n:
            got += os.preadv(self.fd, [mv[got:]], off + got)
        return np.frombuffer(buf, np.uint8)

    def tensor(self, name, dev, rows=None):
        """The tensor dequantized to bf16 on dev (rows: only these rows of a 2-D tensor)."""
        import gguf
        import torch
        from diffusers.quantizers.gguf.utils import GGML_QUANT_SIZES, dequantize_functions

        qt, shape, _, _ = self.meta[name]
        raw = self.raw(name)
        if qt in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
            dt = {gguf.GGMLQuantizationType.F32: np.float32, gguf.GGMLQuantizationType.F16: np.float16}.get(qt)
            t = (torch.from_numpy(raw.view(dt)) if dt is not None
                 else torch.from_numpy(raw.view(np.int16)).view(torch.bfloat16)).reshape(shape)
            t = t if rows is None else t[rows]
            return t.to(dev, torch.bfloat16)
        bs, ts = GGML_QUANT_SIZES[qt]
        q = torch.from_numpy(raw).reshape(shape[0], -1)          # one quantized row per output row
        if rows is not None:
            q, shape = q[rows], [len(rows), shape[1]]
        q = q.to(dev)
        out = dequantize_functions[qt](q.reshape(-1, ts), bs, ts, torch.bfloat16)
        return out.reshape(shape)


def _rms(x, w):
    import torch

    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS)).to(w.dtype) * w


def _block(h, W, bias, n_heads):
    """One T5 encoder block. h is the fp32 residual stream [1, L, d]; matmuls run in bf16."""
    import torch

    x = _rms(h, W["attn_norm"])
    L, d = x.shape[1], x.shape[2]

    def heads(t):
        return t.view(1, L, n_heads, d // n_heads).transpose(1, 2)

    q, k, v = (heads(x @ W[n].T) for n in ("attn_q", "attn_k", "attn_v"))
    # T5 attention is unscaled, with the relative-position bias added to the logits
    a = torch.softmax((q @ k.transpose(-1, -2)).float() + bias, dim=-1).to(v.dtype) @ v
    h = h + (a.transpose(1, 2).reshape(1, L, d) @ W["attn_o"].T).float()
    x = _rms(h, W["ffn_norm"])
    g = torch.nn.functional.gelu(x @ W["ffn_gate"].T, approximate="tanh") * (x @ W["ffn_up"].T)
    return h + (g @ W["ffn_down"].T).float()


class T5Text:
    """encode(text) -> [1, FLUX_TEXT_LEN, 4096] bf16 on the CPU, cached per text (least recently used dropped)."""

    def __init__(self, cache: int = 64, path=None, device: str = "cuda"):
        from transformers import T5TokenizerFast

        self.cache, self.path, self.device = cache, str(path or FLUX_T5_GGUF), device
        self.tok = T5TokenizerFast.from_pretrained(FLUX_BASE, subfolder="tokenizer_2")
        self._embs: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.misses = 0

    def encode(self, text: str):
        with self._lock:
            if text in self._embs:
                self._embs.move_to_end(text)
                return self._embs[text]
            e = self._run(text)
            self.misses += 1
            self._embs[text] = e
            while len(self._embs) > self.cache:
                self._embs.popitem(last=False)
            return e

    def _run(self, text: str):
        import torch

        ids = self.tok(text, padding="max_length", max_length=FLUX_TEXT_LEN, truncation=True,
                       return_tensors="pt").input_ids[0]
        g, dev = _Gguf(self.path), self.device
        try:
            with torch.inference_mode():
                uniq, inv = torch.unique(ids, return_inverse=True)
                h = g.tensor("token_embd.weight", dev, rows=uniq.tolist())[inv.to(dev)][None].float()
                L = ids.shape[0]
                pos = torch.arange(L, device=dev)
                rel_w = g.tensor("enc.blk.0.attn_rel_b.weight", dev).float()       # [buckets, heads]
                bias = rel_w[_bucket(pos[None, :] - pos[:, None])].permute(2, 0, 1)[None]
                del rel_w
                names = ("attn_q", "attn_k", "attn_v", "attn_o", "attn_norm", "ffn_gate", "ffn_up", "ffn_down",
                         "ffn_norm")
                for i in range(g.n_blocks):
                    W = {n: g.tensor(f"enc.blk.{i}.{n}.weight", dev) for n in names}
                    h = _block(h, W, bias, g.n_heads)
                    del W
                h = _rms(h, g.tensor("enc.output_norm.weight", dev))
                out = h.to(torch.bfloat16).cpu()
        finally:
            g.close()
            if dev != "cpu":
                torch.cuda.empty_cache()
        return out
