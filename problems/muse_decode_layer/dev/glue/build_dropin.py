"""Export the same kernels as an importable serving-shaped module."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
s=(ROOT/'problems/muse_decode_layer/cand_r3_best5_glue.py').read_text();t=ast.parse(s)
helpers='\n\n'.join(ast.get_source_segment(s,n) for n in t.body if isinstance(n,ast.FunctionDef) and n.name in ('_pdl_start','_pdl_wait','_launch'))
# Source segments exclude decorators; restore the two JIT helper decorators.
helpers=helpers.replace('def _pdl_start','@triton.jit\ndef _pdl_start').replace('def _pdl_wait','@triton.jit\ndef _pdl_wait')
out='''"""Experimental decode glue; exact candidate bodies, H200 latency pending.

boundary((partials, splits), stream, alpha, beta, 1+prew, postw=1+postw)
route_and_norm((partials,splits), {router_bias,pre_expert_w}, {local_start})
All matrix products and expert work are caller-owned. No weights are expanded.
"""
import torch
import triton
import triton.language as tl

PDL=True

'''+helpers+'\n\n'+(HERE/'kernels.py.inc').read_text()
(HERE/'dropin.py').write_text(out)
