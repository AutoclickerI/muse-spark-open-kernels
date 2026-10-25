"""Run attention TASK output/cache checks on fresh graph inputs for both exponent approximations."""
import sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_attention_decode'),str(HERE.parent/'vq8')]
from local_eval import load_module,copy_into
import vq8
import build
ref=load_module('reference',ROOT/'problems/muse_attention_decode/reference.py');ref._fa3_scheduler_metadata=lambda *a,**kw:None
CASES=[dict(batch=b,ctx=ctx,heads='tp8',layer='full',q_len=8,page=page,dist=dist,spread='agentic' if b>1 else 'fixed',kv_scale=scale)
 for b,ctx,page,dist,scale in ((1,20,64,'soft',1.),(3,8000,1,'edges',1.),(3,4000,1,'soft',1/64),(16,80000,1,'coherent',1.),(32,80000,1,'coherent',1.))]
fails=0
for d in (2,3):
    vq8.VQ8_CUDA=build.source(d);vq8._EXT.clear()
    for i,case in enumerate(CASES):
        data=ref.generate_input(**case,seed=1200+i)
        fn=lambda:vq8.vq8_attention(data,stages=6,min_tiles=8)
        fn();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=fn()
        copy_into(data,ref.generate_input(**case,seed=4200+i));graph.replay();torch.cuda.synchronize()
        ok,msg=ref.check_implementation(data,ref.collect(data,out));fails+=not ok
        print('PASS' if ok else 'FAIL','degree',d,case,msg[:160],flush=True)
print('FAILS',fails);raise SystemExit(bool(fails))
