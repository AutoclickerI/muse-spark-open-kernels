"""Compile producer epilogues for sm90 without launching a Hopper cubin locally."""
import json
from pathlib import Path
import subprocess
import sys

import torch
import triton
from triton.backends.compiler import GPUTarget

import kernels as V
from check import state

HERE=Path(__file__).resolve().parent
rows=[]
parallel='--parallel' in sys.argv
fast_route='--fast-route' in sys.argv
stats='--stats' in sys.argv or parallel or fast_route
if stats:
    if fast_route:import stats_fast_route as S
    elif parallel:import stats_parallel as S
    else:import stats_kernel as S
kernel=S._ps_gemv if stats else V._pg_gemv


class Warm:
    def __getitem__(self,grid):
        def run(*args,**kw):
            compiled=kernel.warmup(*args,grid=grid,**kw)
            p=HERE/'native.cubin';p.write_bytes(compiled.asm['cubin'])
            usage=subprocess.run(['cuobjdump','--dump-resource-usage',str(p)],capture_output=True,text=True,check=True).stdout
            rows.append(dict(t=kw['T'],n=kw['N'],k=kw['K'],role=kw['ROLE'],meta=kw['EMIT_META'],
                             fuse=kw['FUSE'],trace=kw['TRACE'],shared=compiled.metadata.shared,
                             local_loads=compiled.asm['ptx'].count('ld.local'),
                             local_stores=compiled.asm['ptx'].count('st.local'),usage=usage))
            assert compiled.metadata.shared<=232448
            print({k:v for k,v in rows[-1].items() if k!='usage'},flush=True)
            return compiled
        return run


target=triton.runtime.driver.active.get_current_target
triton.runtime.driver.active.get_current_target=lambda:GPUTarget('cuda',90,32)
if stats:S._ps_gemv=Warm()
else:V._pg_gemv=Warm()
try:
    dev='cuda';gen=torch.Generator(device=dev).manual_seed(1);st=state(dev,gen)
    for t in (1,8,16,32):
        for role in ('b1','route','b2'):
            n,k,s=(8192,1024,2) if role=='b1' else ((4608,8192,6) if role=='route' else (1024 if t<=8 else 8192,4096,8 if t<=8 else 2))
            w=torch.empty((n,k),device=dev,dtype=torch.bfloat16)
            x=torch.empty((t,k),device=dev,dtype=torch.bfloat16);stream=torch.empty((t,8192),device=dev)
            start=torch.empty(1,device=dev,dtype=torch.int32)
            for fuse,meta,trace in ((True,True,False),(True,False,False),(True,True,True),(False,False,False)):
                factory=S.ProducerStats if stats else V.ProducerGlue
                factory(w,t,s,role,fuse=fuse,emit_meta=meta,trace=trace)(x,st,stream,start)
finally:
    triton.runtime.driver.active.get_current_target=target
    if stats:S._ps_gemv=kernel
    else:V._pg_gemv=kernel
(HERE/('fast_route_compile_hopper.json' if fast_route else 'parallel_compile_hopper.json' if parallel else 'stats_compile_hopper.json' if stats else 'compile_hopper.json')).write_text(json.dumps(rows,indent=2)+'\n')
print('PASS sm90',len(rows),'cells')
