"""Compile the exact embedded JIT definitions for Hopper, without launching."""
import ast
import hashlib
import json
import linecache
from pathlib import Path
import types

import torch
import triton
from triton.backends.compiler import GPUTarget
from check import state

HERE=Path(__file__).resolve().parent
source=ast.literal_eval(next(n.value for n in ast.parse((HERE/'lab_producer_hybrid.py').read_text()).body
    if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='PG_SOURCE'))
name='<producer_glue_s11>'
linecache.cache[name]=(len(source),None,source.splitlines(True),name)
host=types.ModuleType('producer_glue_s11');exec(compile(source,name,'exec'),host.__dict__)
target=triton.runtime.driver.active.get_current_target
kernel=host._ps_gemv
rows=[]


class Warm:
    def __getitem__(self,grid):
        def run(*args,**kw):
            compiled=kernel.warmup(*args,grid=grid,**kw)
            assert 'ld.local' not in compiled.asm['ptx'] and 'st.local' not in compiled.asm['ptx']
            rows.append(dict(t=kw['T'],n=kw['N'],s=kw['S'],shared=compiled.metadata.shared))
            return compiled
        return run


triton.runtime.driver.active.get_current_target=lambda:GPUTarget('cuda',90,32)
host._ps_gemv=Warm()
try:
    st=state('cuda',torch.Generator(device='cuda').manual_seed(2))
    for role,n,k,s in (('route',4608,8192,6),('b1',8192,1024,2),('b1',8192,1024,1),('b2',1024,4096,8)):
        w=torch.empty((n,k),device='cuda',dtype=torch.bfloat16)
        op=host.ProducerStats(w,1,s,role)
        op(torch.empty((1,k),device='cuda',dtype=torch.bfloat16),st,
           torch.empty((1,8192),device='cuda'),torch.empty(1,device='cuda',dtype=torch.int32))
        print('PASS EMBED',role,s,flush=True)
finally:
    triton.runtime.driver.active.get_current_target=target
    host._ps_gemv=kernel
(HERE/'embedded_validation.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
    torch=torch.__version__,triton=triton.__version__,target='cuda90',cells=rows),indent=2)+'\n')
