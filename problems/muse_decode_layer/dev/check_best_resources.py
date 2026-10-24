"""Compile the token-tiled PREFIX for sm90 and report actual shared-memory metadata."""
from pathlib import Path
import sys
import json
import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
import local_eval
mod=local_eval.load_module('best_resources',ROOT/'problems/muse_decode_layer/cand_r3_best.py')
rows=[]
for t in (1,8,16,32,128,256):
    tb=min(32,max(16,triton.next_power_of_2(t)))
    for fn,names,constants in (
        (mod._o_stats_kernel,{'X':'*bf16','W':'*bf16','STREAM':'*fp32','ALPHA':'*fp32','BETA':'*fp32','A':'*bf16',
         'STATS':'*fp32','COEFF':'*fp32','CNT':'*i32'},dict(T=t,TB=tb,BN=64,BK=128,USE_PDL=True)),
        (mod._pool_boundary_kernel,{'A':'*bf16','COEFF':'*fp32','STREAM':'*fp32','STREAM1':'*fp32','ALPHA':'*fp32',
         'BETA':'*fp32','PREW':'*fp32','W':'*bf16','P':'*fp32'},dict(T=t,TB=tb,BN=64,KS=2048,BK=128,USE_PDL=True))):
        src=ASTSource(fn,signature=names, constexprs=constants)
        kernel=triton.compile(src,target=GPUTarget('cuda',90,32),options={'num_warps':4,'num_stages':2})
        shared=kernel.metadata.shared
        assert shared<=232448,(t,fn.__name__,shared)
        rows.append(dict(t=t,kernel=fn.__name__,tb=tb,smem=shared))
print(json.dumps(rows,indent=2))
(ROOT/'problems/muse_decode_layer/dev/best_resources_local.json').write_text(json.dumps(rows,indent=2)+'\n')
