"""The FLUX engine: hand-written CUDA kernels that replace pieces of the diffusers reference path.

ext.load_ext builds the CUDA extension (csrc/) once and caches it; q4k.Q4KLinear runs a linear straight from its
packed Q4_K super-blocks; swap.swap_linears puts it in place of a model's Q4_K linears (docs/engine/).
"""
