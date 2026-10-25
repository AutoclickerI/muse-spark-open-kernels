"""Compile every Triton PDL consumer in S14 and retain its actual SASS/PTX."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import torch
import triton
from triton.backends.compiler import GPUTarget

from audit_native import HERE, REPO, load


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,default=REPO/'problems/muse_decode_layer/cand_r3_best7_s14_final.py')
    ap.add_argument('--label',default='baseline');a=ap.parse_args()
    m=load(a.candidate);folder=HERE/'native'/a.label;folder.mkdir(parents=True,exist_ok=True)
    pointer_names=set('X W P CNT Y G BR STREAM SOUT OUT POSTW ALPHA BETA PREW PR BIAS LSTART IDS WTS LIDS SUMS POS LOC KC VC KSCALE VSCALE INV_FREQ QOUT GATE Q PT SEQ OP MP LP STREAM_OUT'.split())
    bf16=set('X W Y G OUT QOUT Q'.split());i64={'POS','LOC'};i32={'IDS','LIDS','LSTART','CNT','PT','SEQ'}
    buffers={d:torch.empty(1,device='cuda',dtype=d) for d in (torch.bfloat16,torch.float32,torch.int32,torch.int64,torch.float8_e4m3fn)}
    default=dict(T=1,x_stride=8192,N=8192,K=8192,KS=1024,BN=64,BK=128,TB=16,PREFETCH=True,USE_PDL=True,
        S=6,NS=4,PS=8192,RS=8192,BS=8192,ps_stride=8192,bs_stride=8192,br_stride=8192,pr_stride=4608,
        NB=8192,HAS_POSTW=True,REPLICAS=1,post_eps=1e-8,pre_eps=1e-5,eps=1e-5,H=8192,C=1,D=64,
        HQ=16,HKV=2,ROPE=True,Q_FP8=False,q_mult=5.473,NE=256,NLOCAL=32,mult=0.1767766952966369,
        BLOCK=512,pt_stride=80008,sm_scale=0.125,n_splits=4,L=1,LB=1,PAGE=1,WINDOW=2048,SPLIT=512,
        BLOCK_N=64,SPLITS_P2=4,TRANS=False,CG=False)
    overrides={'_best7_old_boundary_kernel':dict(C=16),'_best7_old_route_norm_kernel':dict(D=4096,K=8,C=1),
        '_attn_sumcast_kernel':dict(N=2304,S=7),'_gate_split_kernel':dict(N=1024,S=7,NS=3),
        '_gi_kernel':dict(N=4608,S=6,NS=2),'_gb_kernel':dict(N=4608,S=6)}
    names=[]
    for n in ast.parse(a.candidate.read_text()).body:
        if isinstance(n,ast.FunctionDef) and n.name not in ('_pdl_wait','_pdl_start'):
            s=ast.unparse(n)
            if ('_pdl_wait(' in s or 'gdc_wait()' in s) and n.name.startswith('_'):names.append(n.name)
    rows=[];old=triton.runtime.driver.active.get_current_target
    triton.runtime.driver.active.get_current_target=lambda:GPUTarget('cuda',90,32)
    try:
        for name in names:
            kernel=getattr(m,name);values=dict(default,**overrides.get(name,{}));args=[]
            for arg in kernel.arg_names:
                if arg in pointer_names:
                    dtype=torch.bfloat16 if arg in bf16 else torch.int64 if arg in i64 else torch.int32 if arg in i32 else torch.float8_e4m3fn if arg in ('KC','VC') else torch.float32
                    args.append(buffers[dtype])
                else:args.append(values[arg])
            c=kernel.warmup(*args,grid=(1,1,1),num_warps=4,launch_pdl=True)
            p=folder/(name+'.ptx');p.write_text(c.asm['ptx'])
            cubin=p.with_suffix('.cubin')
            subprocess.run(['ptxas','-arch=sm_90a','-O3',str(p),'-o',str(cubin)],check=True,capture_output=True,text=True)
            usage=subprocess.run(['cuobjdump','--dump-resource-usage',str(cubin)],capture_output=True,text=True,check=True).stdout
            nregs=int(re.search(r'REG:(\d+)',usage).group(1))
            sass=subprocess.run(['nvdisasm','-g',str(cubin)],capture_output=True,text=True,check=True).stdout
            p.with_suffix('.sass').write_text(sass)
            ins=[s.strip() for s in sass.splitlines() if re.search(r'/\*[0-9a-f]+\*/',s) and ';' in s and not '.byte' in s]
            waits=[i for i,s in enumerate(ins) if 'ACQBULK' in s]
            pre=[s for s in ins[:waits[0]] if re.search(r'\b(?:LDG|LDGSTS|CCTL)\b',s)] if waits else []
            row=dict(name=name,source_sha256=hashlib.sha256(kernel.src.encode()).hexdigest(),waits=waits,
                independent_candidates_before_first_wait=pre,registers=nregs,shared=c.metadata.shared,
                local_loads=c.asm['ptx'].count('ld.local'),local_stores=c.asm['ptx'].count('st.local'))
            rows.append(row);print(name,'waits',waits,'preload instructions',len(pre),'registers',nregs,flush=True)
    finally:triton.runtime.driver.active.get_current_target=old
    (HERE/(a.label+'_triton.json')).write_text(json.dumps(rows,indent=2)+'\n')
    print('PASS Triton PDL kernels',len(rows),flush=True)


if __name__=='__main__':main()
