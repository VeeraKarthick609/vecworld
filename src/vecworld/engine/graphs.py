"""The FLUX blocks replayed as CUDA graphs, one per block and input shape (Task 7, docs/engine/07-cuda-graphs.md).

A CUDA graph records a sequence of kernel launches once (capture) and submits it again with a single call (replay):
no Python, no PyTorch dispatch, no per-kernel driver call, and the GPU gets the whole dependency chain at once. The
price is that a graph is frozen: every kernel's grid and every device address it reads or writes are baked in. So

- the weights must sit at fixed addresses: engine.slots.BlockSlots (block i always in slot i % ring);
- inputs and outputs live in fixed buffers per shape: key = (batch, image tokens, text tokens), one set of graphs
  per key, least-recently-used keys dropped beyond max_graphs;
- the CPU still runs between blocks, to sequence the slot copies (outside the graphs, so Task 10's SSD reads fit).

Buffers: block 0's inputs, the timestep embedding and the rotary tables are copied into per-key static buffers
(skipped when the very same tensor was copied already, e.g. temb for blocks 1..56). Every block writes its outputs
into one of two per-key ping-pong buffers per kind (a double block's separate text / image streams, a single
block's joint stream), which are exactly the next block's static inputs: no copies between blocks. The fused
forwards (engine.blocks) write their last pass straight into them (out=); any other forward is captured with a copy.

All graphs capture from one shared memory pool. Nothing allocated inside a capture outlives it (outputs go to the
ping-pong buffers, allocated outside the pool), so every graph's intermediates may reuse the same pool memory,
whatever order the graphs replay in.
"""
import inspect
import threading
import weakref
from collections import Counter, OrderedDict

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vecworld.config import FLUX_GRAPH_MAX, FLUX_SLOT_RING
from vecworld.engine.slots import BlockSlots, _tensors

BLOCK_LISTS = ("transformer_blocks", "single_transformer_blocks")


class _Shared:
    """A static input buffer and (weakly) the source last copied into it. Block 0 always copies (force): a caller may
    hand the same tensor object again with new values written in place, and an inference tensor has no version
    counter to tell. Later blocks of the same pass skip the copy while the very same tensor comes again (temb and the
    rotary tables, for blocks 1..56)."""

    def __init__(self, like: torch.Tensor):
        self.t = torch.empty(like.shape, dtype=like.dtype, device=like.device)
        self.src, self.ver = None, None

    def fill(self, src: torch.Tensor, force: bool = False) -> torch.Tensor:
        ver = None if src.is_inference() else src._version
        if force or self.src is None or self.src() is not src or ver != self.ver:
            self.t.copy_(src)
            self.src, self.ver = weakref.ref(src), ver
        return self.t


def _arg(func, args, kwargs, name):
    for k, a in enumerate(func._schema.arguments):
        if a.name == name:
            if name in kwargs:
                return kwargs[name]
            return args[k] if k < len(args) else a.default_value
    return None


class _NoRandom(TorchDispatchMode):
    """Raises on any op that draws random numbers (torch.Tag.nondeterministic_seeded), except attention and dropout
    with a dropout probability of 0 (FLUX's attention dispatches with dropout_p = 0).

    The runner replays graphs with the extension's graph_launch, which skips CUDAGraph.replay()'s RNG prologue (the
    philox seed / offset update): a graph that draws random numbers would draw the capture's numbers on every replay,
    silently. Run over the eager warm-up that precedes each capture, it makes such a block fail at capture, naming it.
    It sees PyTorch ops only: a custom extension kernel drawing its own random numbers (cuRAND) is not caught."""

    def __init__(self, i: int):
        super().__init__()
        self.i = i

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if torch.Tag.nondeterministic_seeded in func.tags:
            p = _arg(func, args, kwargs, "dropout_p")
            if p is None and func.overloadpacket is torch.ops.aten.native_dropout:
                p = 0.0 if not _arg(func, args, kwargs, "train") else _arg(func, args, kwargs, "p")
            if p is None or float(p) != 0.0:
                raise RuntimeError(f"block {self.i}: {func} draws random numbers, and its CUDA graph would repeat the "
                                   "capture's numbers on every replay (graph_launch skips replay()'s RNG prologue); "
                                   "run it eagerly (VECWORLD_GRAPHS=0)")
        return func(*args, **kwargs)


class _Key:
    """One input shape's graphs and fixed buffers."""

    def __init__(self, n: int):
        self.graphs: list[torch.cuda.CUDAGraph | None] = [None] * n
        self.execs: list[int] = [0] * n                 # each graph's cudaGraphExec_t, launched by the extension
        self.ins: list[tuple | None] = [None] * n       # (es, hs) graph i reads
        self.outs: list[tuple | None] = [None] * n      # (es, hs) block i returns (views of its output buffer)
        self.shared: dict[str, _Shared] = {}
        self.bufs: dict[tuple, tuple] = {}              # (kind, parity) -> output buffers


class _Forward:
    """A block's forward routed through the runner (an instance attribute; nn.Module calls it)."""

    def __init__(self, runner, i, inner):
        self.runner, self.i, self.inner = runner, i, inner

    def __call__(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None):
        return self.runner._run(self.i, hidden_states, encoder_hidden_states, temb, image_rotary_emb,
                                joint_attention_kwargs)


class GraphRunner:
    """A CUDA graph per (block index, input shape key) for blocks on BlockSlots.

    Each block's forward is wrapped (install after patch_blocks: it wraps the forward as it is). The first time a
    block meets a key it runs eagerly once (warm-up: lazy initialisation, the output layout), is captured into a
    graph from the shared pool, and that graph is replayed for its output; afterwards it is only replayed. With
    enabled False (VECWORLD_GRAPHS=0) the blocks run eagerly on the same slots. Calls a graph can't take (attention
    kwargs, no rotary embedding, not bf16 CUDA) run eagerly too.

    Blocks run outside inference mode (under no_grad), so their activations track versions and the fused glue's
    attached quantization can be checked for staleness (q4k.attach_quantized).

    stats counts "captures", "replays", "eager" block calls and key "evictions".

    With graphs, a block's returned outputs are the key's fixed ping-pong buffers, the same tensor objects on every
    call: they are overwritten by the next call of the same block, and within a pass by block i + 2 (same parity).
    The transformer consumes each output before then; a caller that keeps a block output (a residual cache, a debug
    hook) must clone it. Eager calls return fresh tensors. All calls must come from one thread (the first caller's).

    Off by default in the painter since the Task 7 review (VECWORLD_GRAPHS=1 to opt in): eager on the slots is as fast
    and the graphs cost captures on each new shape and memory (docs/engine/07-cuda-graphs.md)."""

    def __init__(self, blocks: list[torch.nn.Module], slots: BlockSlots | None, max_graphs: int = FLUX_GRAPH_MAX,
                 enabled: bool = True):
        self.blocks, self.slots, self.max_graphs, self.enabled = list(blocks), slots, max_graphs, enabled
        self.stats: Counter = Counter()
        self.pool = torch.cuda.graph_pool_handle()
        from vecworld.engine.ext import load_ext

        # a replay is the extension's cudaGraphLaunch, which keeps the GIL (Task 10a, docs/engine/10-streamer.md)
        self._launch = load_ext().graph_launch
        self._keys: OrderedDict[tuple, _Key] = OrderedDict()
        self._owner: int | None = None
        self._inner, self._own = [], []
        for i, b in enumerate(self.blocks):
            self._own.append("forward" in b.__dict__)
            inner = b.forward
            self._inner.append((inner, "out" in inspect.signature(inner).parameters))
            b.forward = _Forward(self, i, inner)

    def keys(self) -> list[tuple]:
        return list(self._keys)

    def _key(self, key: tuple) -> _Key:
        k = self._keys.get(key)
        if k is None:
            k = self._keys[key] = _Key(len(self.blocks))
            while len(self._keys) > max(self.max_graphs, 1):
                self._drop(self._keys.popitem(last=False)[1])
                self.stats["evictions"] += 1
        self._keys.move_to_end(key)
        return k

    def _drop(self, k: _Key) -> None:
        torch.cuda.synchronize()
        for g in k.graphs:
            if g is not None:
                g.reset()
        k.graphs, k.execs, k.ins, k.outs, k.shared, k.bufs = [], [], [], [], {}, {}
        if not any(g is not None for kk in self._keys.values() for g in kk.graphs):
            # The last graph of the pool is gone, and with it the pool: capturing into its old handle again trips
            # the caching allocator ("use_count > 0" internal assert, torch 2.14). Start a fresh pool.
            self.pool = torch.cuda.graph_pool_handle()

    def _run(self, i, hs, es, temb, rot, kwargs):
        inner, takes_out = self._inner[i]
        me = threading.get_ident()                          # one thread, as BlockSlots (unsynchronised state)
        if self._owner is None:
            self._owner = me
        elif self._owner != me:
            raise RuntimeError(f"GraphRunner belongs to thread {self._owner}; block {i} was run from thread {me}")
        with torch.inference_mode(False), torch.no_grad():
            if (not self.enabled or kwargs or rot is None or hs.dtype != torch.bfloat16 or not hs.is_cuda):
                self.stats["eager"] += 1
                return inner(hidden_states=hs, encoder_hidden_states=es, temb=temb, image_rotary_emb=rot,
                             joint_attention_kwargs=kwargs)
            k = self._key((hs.shape[0], hs.shape[1], es.shape[1]))
            if k.graphs[i] is None:
                return self._capture(k, i, inner, takes_out, hs, es, temb, rot)
            ins = k.ins[i]
            if not (es is ins[0] and hs is ins[1]):
                if i > 0 and k.ins[i] is k.outs[i - 1]:     # adopted from block i - 1, handed something else
                    ins[0].copy_(es)
                    ins[1].copy_(hs)
                else:
                    k.shared["es"].fill(es, force=True)
                    k.shared["hs"].fill(hs, force=True)
            first = i == 0                                  # a new pass: copy everything again (see _Shared)
            k.shared["temb"].fill(temb, first)
            k.shared["cos"].fill(rot[0], first)
            k.shared["sin"].fill(rot[1], first)
            self._launch(k.execs[i])
            self.stats["replays"] += 1
            return k.outs[i]

    def _static(self, k: _Key, name: str, src: torch.Tensor) -> torch.Tensor:
        s = k.shared.get(name)
        if s is None:
            s = k.shared[name] = _Shared(src)
        return s.fill(src, force=True)

    def _capture(self, k, i, inner, takes_out, hs, es, temb, rot):
        prev = k.outs[i - 1] if i > 0 else None
        if prev is not None and es is prev[0] and hs is prev[1]:
            ins = prev                                      # block i - 1's output buffers: no copy
        else:
            ins = (self._static(k, "es", es), self._static(k, "hs", hs))
        args = dict(hidden_states=ins[1], encoder_hidden_states=ins[0], temb=self._static(k, "temb", temb),
                    image_rotary_emb=(self._static(k, "cos", rot[0]), self._static(k, "sin", rot[1])))
        with _NoRandom(i):
            w = inner(**args)                               # warm-up, eager; also tells the output layout
        joint = w[0].untyped_storage().data_ptr() == w[1].untyped_storage().data_ptr()
        T, parity = w[0].shape[1], (i + 1) % 2
        if joint:
            key = ("joint", parity)
            if key not in k.bufs:
                j = torch.empty(w[0].shape[0], T + w[1].shape[1], w[0].shape[2], dtype=w[0].dtype,
                                device=w[0].device)
                k.bufs[key] = (j, (j[:, :T], j[:, T:]))
            dest, outs = k.bufs[key]
        else:
            key = ("double", parity)
            if key not in k.bufs:
                pair = (torch.empty_like(w[0], memory_format=torch.contiguous_format),
                        torch.empty_like(w[1], memory_format=torch.contiguous_format))
                k.bufs[key] = (pair, pair)
            dest, outs = k.bufs[key]
        del w
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
            res = inner(**args, out=dest) if takes_out else inner(**args)
            for r, o in zip(res, outs):
                if r.data_ptr() != o.data_ptr() or r.stride() != o.stride():
                    o.copy_(r)                              # a forward without out=: one copy, in the graph
            del res
        k.graphs[i], k.ins[i], k.outs[i] = g, ins, outs
        k.execs[i] = g.raw_cuda_graph_exec()
        self.stats["captures"] += 1
        self._launch(k.execs[i])
        self.stats["replays"] += 1
        return outs

    def clear(self) -> None:
        """Drop every key's graphs and buffers (they are captured again on their next use)."""
        while self._keys:
            self._drop(self._keys.popitem()[1])

    def close(self) -> None:
        """clear(), and give each block back the forward it had (safe twice)."""
        self.clear()
        for b, own, (inner, _) in zip(self.blocks, self._own, self._inner):
            if isinstance(b.__dict__.get("forward"), _Forward):
                if own:
                    b.forward = inner
                else:
                    del b.forward


def stream_blocks(transformer: torch.nn.Module, device="cuda", ring: int = FLUX_SLOT_RING, graphs: bool = True,
                  max_graphs: int = FLUX_GRAPH_MAX, pack: bool = False, ssd: str | None = None,
                  reader: dict | None = None) -> tuple[BlockSlots, GraphRunner]:
    """Put a FLUX transformer on the engine's block streaming, in place of diffusers' group offload: everything but
    the blocks moves to `device` for good (embedders, output norm and projection: ~57 MB), the blocks view into `ring`
    slots, and a GraphRunner replays them (graphs=False: eagerly). Call it after swap_linears and patch_blocks.

    pack=False: the blocks' current (host, page-locked) tensors become BlockSlots' host copies (Task 7). pack=True (the
    painter, Task 10a): their (not page-locked) host tensors are moved into one page-locked arena, block by block, and
    an engine.stream.BlockStreamer streams each block with one copy from it (its arena: the streamer's .arena).
    ssd=<the GGUF's path> (the painter under FLUX_WEIGHTS_SOURCE=ssd, Task 10b): the BlockStreamer reads each block
    from the file instead (BlockStreamer.from_ssd; reader: its SsdSource settings; the reader: the streamer's .reader)
    and the blocks' host tensors are dropped."""
    for name, child in transformer.named_children():
        if name not in BLOCK_LISTS:
            child.to(device)
    blocks = [b for name in BLOCK_LISTS for b in getattr(transformer, name)]
    if ssd is not None:
        from vecworld.engine.stream import BlockStreamer

        slots = BlockStreamer.from_ssd(blocks, ssd, ring=ring, device=device, **(reader or {}))
    elif pack:
        from vecworld.engine.stream import BlockStreamer

        slots = BlockStreamer.packed(blocks, ring=ring, device=device)
    else:
        host = [{n: t.data for n, t in _tensors(b).items()} for b in blocks]
        slots = BlockSlots(blocks, lambda i: host[i], ring=ring, device=device)
    try:
        return slots, GraphRunner(blocks, slots, max_graphs=max_graphs, enabled=graphs)
    except BaseException:
        if pack or ssd is not None:  # the arena / reader is registered and nobody else owns it (the caller never got
            try:                     # `slots`)
                slots.close()       # the blocks back on their arena views (placeholders from the SSD), then unregistered
            finally:
                (slots.reader if ssd is not None else slots.arena).close()
        raise
