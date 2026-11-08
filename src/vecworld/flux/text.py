"""T5-XXL text embeddings for FLUX, run block by block on the GPU straight from the GGUF.

The spike's route (transformers loading the GGUF and dequantizing all of it to bf16 on the CPU) peaked at 12.4 GB
of RAM, which can't sit next to FLUX in 14 GB. Here nothing is kept between encodes but the tokenizer, the file's
tensor table and an LRU of finished embeddings. Two paths (FLUX_T5 / engine.engine_t5), the same arithmetic:

- "engine" (Task 10c, the default, docs/engine/12-t5-and-results.md): the 24 encoder blocks are streamed from the
  file on the SSD by the FLUX engine's streamer (engine.stream.BlockStreamer fed by engine.uring.SsdSource: O_DIRECT
  reads through io_uring into page-locked staging buffers, one DMA per tensor into fixed device slots, the next blocks
  read while one runs). Each linear is dequantized on the GPU as it is used (the file's linears are Q5_K / Q6_K: no
  engine kernel takes those), then multiplied. The token embedding reads only the rows the prompt uses.
- "pread" (Task 1, the reference and fallback): each block's tensors are read with pread into host memory (no mmap,
  so they never count toward the RSS, though the page cache keeps the file), copied to the GPU, dequantized, run and
  freed before the next block; the file's tensor table is parsed again per encode.
"""
import math
import os
import threading
from collections import OrderedDict

import numpy as np

from vecworld.config import FLUX_BASE, FLUX_T5_GGUF, FLUX_TEXT_LEN

from vecworld.engine import T5_PATHS     # ("engine", "pread")

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


NAMES = ("attn_q", "attn_k", "attn_v", "attn_o", "attn_norm", "ffn_gate", "ffn_up", "ffn_down", "ffn_norm")


def _read_meta(path):
    """(name -> (quant type, shape [out, in], file offset, bytes), blocks, heads): the GGUF's tensor table. gguf-py
    maps the file and builds an array per tensor, ~3 s for this one, so the engine path parses it once."""
    import gguf

    r = gguf.GGUFReader(path)
    meta = {t.name: (t.tensor_type, [int(d) for d in reversed(t.shape)], int(t.data_offset), int(t.n_bytes))
            for t in r.tensors}
    n_blocks = int(r.fields["t5encoder.block_count"].parts[-1][0])
    n_heads = int(r.fields["t5encoder.attention.head_count"].parts[-1][0])
    del r                                           # drops the reader's memmap of the whole file
    return meta, n_blocks, n_heads


def _float_types():
    import gguf

    Q = gguf.GGMLQuantizationType
    return {Q.F32: np.float32, Q.F16: np.float16, Q.BF16: None}


def _dequant(qt, q, shape):
    """bf16 [shape] from the tensor's file bytes q (on the device: uint8 rows, or the float tensor itself)."""
    import torch
    from diffusers.quantizers.gguf.utils import GGML_QUANT_SIZES, dequantize_functions

    if qt in _float_types():
        return q.reshape(shape).to(torch.bfloat16)
    bs, ts = GGML_QUANT_SIZES[qt]
    return dequantize_functions[qt](q.reshape(-1, ts), bs, ts, torch.bfloat16).reshape(shape)


class _Gguf:
    """Tensor metadata from the GGUF; bytes are read on demand with pread, never mapped."""

    def __init__(self, path):
        self.meta, self.n_blocks, self.n_heads = _read_meta(path)
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


class _Weights:
    """A streamed block's weights by name, each dequantized to bf16 when _block asks for it (and freed after its use):
    one linear's bf16 copy at a time on the card, not the block's."""

    def __init__(self, blk):
        self.blk = blk

    def __getitem__(self, name):
        qt, shape = self.blk.kinds[name]
        return _dequant(qt, getattr(self.blk, name), shape)


def _streamed_block(meta, i):
    """Encoder block i as a module whose buffers are its tensors' file bytes, for engine.stream.BlockStreamer to point
    into a device slot: a quantized linear as uint8 [out rows, bytes per row], a norm as its float type. Built with
    placeholders (no bytes); the streamer fills the slot from the file before each forward."""
    import torch

    class StreamedBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.kinds = {}
            for n in NAMES:
                qt, shape, _, nb = meta[f"enc.blk.{i}.{n}.weight"]
                ft = _float_types()
                if qt in ft:
                    dt = torch.bfloat16 if ft[qt] is None else torch.from_numpy(np.zeros(0, ft[qt])).dtype
                    ph = torch.zeros((), dtype=dt).expand(shape)
                else:
                    if nb % shape[0]:
                        raise ValueError(f"enc.blk.{i}.{n}: {nb} bytes don't split into {shape[0]} rows")
                    ph = torch.zeros((), dtype=torch.uint8).expand(shape[0], nb // shape[0])
                if ph.numel() * ph.element_size() != nb:
                    raise ValueError(f"enc.blk.{i}.{n}: {qt.name} {shape} is not {nb} bytes")
                self.register_buffer(n, ph, persistent=False)
                self.kinds[n] = (qt, shape)

        def forward(self, h, bias, n_heads):
            return _block(h, _Weights(self), bias, n_heads)

    return StreamedBlock()


def t5_file_layout(blocks, meta, gap: int = 1 << 16):
    """Each streamed block's engine.uring.BlockFile: every tensor of encoder block i copied from its range of the file,
    nothing resident. The GGUF orders tensors by name ("enc.blk.10" before "enc.blk.5.ffn_down"), so a block's tensors
    lie in one to three runs of the file: tensors closer than `gap` bytes share a read (the gap read with them)."""
    from vecworld.engine.slots import block_layout
    from vecworld.engine.uring import ALIGN, BlockFile

    out = []
    for i, blk in enumerate(blocks):
        lay, _ = block_layout(blk)
        ts = sorted((meta[f"enc.blk.{i}.{n}.weight"][2], meta[f"enc.blk.{i}.{n}.weight"][3], lay[n]) for n in NAMES)
        runs = [[ts[0]]]
        for t in ts[1:]:
            last = runs[-1][-1]
            if t[0] - (last[0] + last[1]) <= gap:
                runs[-1].append(t)
            else:
                runs.append([t])
        extents, copies, bo = [], [], 0
        for run in runs:
            lo, hi = run[0][0], max(f + n for f, n, _ in run)
            alo, ahi = lo // ALIGN * ALIGN, -(-hi // ALIGN) * ALIGN
            extents.append((alo, ahi - alo, hi - alo, bo))
            copies += [(so, bo + f - alo, n) for f, n, so in run]
            bo += ahi - alo
        out.append(BlockFile(lo=extents[0][0], nbytes=bo, need=bo, copies=copies, extents=extents))
    return out


class T5Text:
    """encode(text) -> [1, FLUX_TEXT_LEN, 4096] bf16 on the CPU, cached per text (least recently used dropped).
    engine: "engine" or "pread" (see the module docstring); None: engine.engine_t5's rule (FLUX_T5 / config.FLUX_T5)."""

    def __init__(self, cache: int = 64, path=None, device: str = "cuda", engine: str | None = None):
        from transformers import T5TokenizerFast

        if engine is None:
            from vecworld.engine import engine_t5

            engine = engine_t5()
        if engine not in T5_PATHS:
            raise ValueError(f"T5Text engine={engine!r}; use one of {T5_PATHS}")
        self.cache, self.path, self.device, self.engine = cache, str(path or FLUX_T5_GGUF), device, engine
        self.tok = T5TokenizerFast.from_pretrained(FLUX_BASE, subfolder="tokenizer_2")
        self._embs: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self.misses = 0
        self._meta = self._blocks = None
        if engine == "engine":                  # the tensor table once, here (~3 s), not per encode
            self._meta = _read_meta(self.path)
            self._blocks = [_streamed_block(self._meta[0], i) for i in range(self._meta[1])]
        self.last: dict = {}                    # the last encode's timings (engine path), in seconds

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

    def _ids(self, text: str):
        return self.tok(text, padding="max_length", max_length=FLUX_TEXT_LEN, truncation=True,
                        return_tensors="pt").input_ids[0]

    def _run(self, text: str):
        return self._run_engine(text) if self.engine == "engine" else self._run_pread(text)

    def _run_engine(self, text: str):
        import time

        import torch

        from vecworld.config import FLUX_T5_AHEAD, FLUX_T5_BUFFERS, FLUX_T5_RING, FLUX_SSD_CHUNK, FLUX_SSD_QUEUE_DEPTH
        from vecworld.engine.stream import BlockStreamer
        from vecworld.engine.uring import DirectReader

        t0 = time.perf_counter()
        meta, _, n_heads = self._meta
        dev = self.device
        ids = self._ids(text)
        uniq, inv = torch.unique(ids, return_inverse=True)

        def rows(name, which=None):
            """Tensor `name` (only rows `which` of it) from the file with O_DIRECT, dequantized to bf16 on dev."""
            qt, shape, off, nb = meta[name]
            if which is None:
                host = torch.empty(nb, dtype=torch.uint8)
                reader.read_into(off, nb, host)
            else:
                bpr = nb // shape[0]
                host = torch.empty(len(which), bpr, dtype=torch.uint8)
                for k, r in enumerate(which):
                    reader.read_into(off + r * bpr, bpr, host[k])
                shape = [len(which), shape[1]]
            ft = _float_types()
            q = host.to(dev)
            if qt in ft:
                q = q.view(torch.bfloat16 if ft[qt] is None else torch.from_numpy(np.zeros(0, ft[qt])).dtype)
            return _dequant(qt, q, shape)

        with torch.inference_mode():
            reader = DirectReader(self.path)
            try:
                h = rows("token_embd.weight", uniq.tolist())[inv.to(dev)][None].float()
                L = ids.shape[0]
                pos = torch.arange(L, device=dev)
                rel_w = rows("enc.blk.0.attn_rel_b.weight").float()                 # [buckets, heads]
                bias = rel_w[_bucket(pos[None, :] - pos[:, None])].permute(2, 0, 1)[None]
                out_norm = rows("enc.output_norm.weight")
            finally:
                reader.close()
            del rel_w
        t1 = time.perf_counter()
        # outside inference mode: install() repoints the blocks' (normal) buffers at the slots
        st = BlockStreamer.from_ssd(self._blocks, self.path, layout=t5_file_layout(self._blocks, meta),
                                    ring=FLUX_T5_RING, device=dev, buffers=FLUX_T5_BUFFERS, ahead=FLUX_T5_AHEAD,
                                    one_pass=True, queue_depth=FLUX_SSD_QUEUE_DEPTH, chunk=FLUX_SSD_CHUNK)
        try:
            t2 = time.perf_counter()
            with torch.inference_mode():
                for blk in self._blocks:
                    h = blk(h, bias, n_heads)
                h = _rms(h, out_norm)
                out = h.to(torch.bfloat16).cpu()
            t3 = time.perf_counter()
            stats = dict(st.reader.stats)
        finally:
            try:
                st.close()
            finally:
                st.reader.close()
                torch.cuda.empty_cache()
        self.last = dict(prep_s=round(t1 - t0, 3), setup_s=round(t2 - t1, 3), blocks_s=round(t3 - t2, 3),
                         total_s=round(time.perf_counter() - t0, 3), read_wait_s=round(stats["read_wait_s"], 3),
                         buffer_wait_s=round(stats["buffer_wait_s"], 3), gb=round(stats["bytes"] / 1e9, 2))
        return out

    def _run_pread(self, text: str):
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
