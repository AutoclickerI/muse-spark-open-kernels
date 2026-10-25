"""Hopper resource audit of87dc's exact dense-chain device body."""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
source=ROOT/'problems/muse_decode_layer/cand_r3_best7_reset.py'
for n in ast.parse(source.read_text()).body:
    if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='DENSE_CHAIN_CUDA' for t in n.targets):cuda=ast.literal_eval(n.value)
text=cuda[:cuda.index('template <int NT, bool PREFIX>\nvoid launch(')]+'\n}\n'
text=text.replace('#include <torch/extension.h>','').replace('#include <ATen/cuda/CUDAContext.h>','')
for nt in (1,2,4):
    text+=f'template __global__ void persistent::dense_island<{nt},true>(persistent::Params);\n'
out=ROOT/'runs/screen';path=out/'megakernel_prior90.cu';obj=path.with_suffix('.o');path.write_text(text)
r=subprocess.run(['/usr/local/cuda/bin/nvcc','-arch=sm_90a','-std=c++20','-O3','-lineinfo','--ptxas-options=-v',
    '-c',str(path),'-o',str(obj)],capture_output=True,text=True,check=True)
(out/'megakernel_prior90.compile').write_text(r.stderr)
rows=[]
for match in re.finditer(r"Compiling entry function '([^']+)'.*?Used (\d+) registers",r.stderr,re.S):
    name,regs=match[1],int(match[2]);nt=int(re.search(r'ILi(\d+)E',name)[1])
    smem=8*(8*nt)*20*4;register_cta=regs*256
    rows.append(dict(nt=nt,registers_per_thread=regs,dynamic_smem_bytes=smem,grid_ctas=132,
        registers_per_cta=register_cta,register_limit_ctas_per_sm=65536//register_cta,
        resident_warps_with_actual_grid=8,actual_resident_warp_fraction=8/64))
assert len(rows)==3,rows
result=dict(cuda_sha256=hashlib.sha256(cuda.encode()).hexdigest(),target='sm_90a',rows=rows,
    instruction='source unchanged; strip host launch/includes and explicitly instantiate the three prefix specializations')
(HERE/'prior_sm90a.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
