"""Check the compressed host at the same user-visible projection ABI."""
import sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer/dev/p12_components')]
import local_eval,static_cache
static_cache.install(local_eval);load=local_eval.load_module
def module(name,path):
    m=load(name,path)
    if name=='submission':
        m._P12_FORCE=True
        class VD:
            def __getitem__(self,grid):
                def run(x,w,p,xd,wd,t,n,k,s,**kw):p.zero_();p[0].copy_(m.torch.mm(x,w.T,out_dtype=m.torch.float32))
                return run
        m._vdsk_gemm=VD()
    return m
local_eval.load_module=module
out=H/'screen/compact';out.mkdir(parents=True,exist_ok=True)
sys.argv=['local_eval.py',str(H/'compact_host.py'),'--no-baseline','--reps','2','--inputs','3','--json',str(out/'result.json')]
local_eval.main()
