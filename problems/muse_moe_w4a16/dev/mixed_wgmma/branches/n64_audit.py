"""Recompile exact 858ff2fa CUDA's lean N32/N64 specializations, locally.

The old comparison was not an occupancy ablation. These resources explain a
credible mechanism, not a causal measurement of each individual cost.
"""
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
H=Path(__file__).resolve().parent;R=H.parents[4]
source=R/'runs/muse_moe_w4a16/20261025T023627Z_submit_858ff2faa7d84e33.py'
tree=ast.parse(source.read_text())
payload=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_LAB_PAYLOADS')[0][-1]
cu=next(ast.literal_eval(n.value) for n in ast.parse(payload).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='BR_HH_CUDA')
original_sha=hashlib.sha256(cu.encode()).hexdigest()
cu='#define BR_KINDS(F) F(20,32,true,2,0,0,0) F(21,32,false,2,0,0,0) F(24,64,true,2,0,0,0) F(25,64,false,2,0,0,0)\n'+cu.split('\n',1)[1]
p=Path('/tmp/p3b_n64_old.cu');p.write_text(cu)
r=subprocess.run(['nvcc','-std=c++20','-O3','-lineinfo','-gencode=arch=compute_90a,code=sm_90a','-cubin',str(p),'-o','/tmp/p3b_n64_old.cubin','-Xptxas','-v'],capture_output=True,text=True,check=True)
log=r.stdout+r.stderr;rows=[]
for x in log.split('Compiling entry function ')[1:]:
 if 'mwbr4gemmI' not in x.splitlines()[0]:continue
 nt=int(re.search(r'gemmILi(\d+)',x)[1]);first='ELb1E' in x.splitlines()[0]
 regs=int(re.search(r'Used (\d+) registers',x)[1]);sm=re.search(r'(\d+) bytes smem',x);static=int(sm[1]) if sm else 0
 dynamic=1023+4*(64*128//2+64*128//16)+4*nt*128*2
 rows.append(dict(nt=nt,first=first,registers=regs,static_smem=static,dynamic_smem=dynamic,
                  smem_limited_CTAs=232448//(static+dynamic)))
out=dict(source=str(source.relative_to(R)),original_cuda_sha256=original_sha,rows=rows,
         observation='N64 S4 doubles accumulator count and activation stage storage; 4 -> 2 smem-limited CTAs/SM. It does not share activation tiles across output-channel CTAs; 64-token padding can exceed N32 padding. Two heavy classes also add launches. No one-cause ablation was run.')
(H/'n64_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
