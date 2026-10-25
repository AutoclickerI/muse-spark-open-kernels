"""Official projection TASK checks, with explicit Hopper launch stand-ins."""
import importlib.util,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools')]
import local_eval
spec=importlib.util.spec_from_file_location('p12_static',R/'problems/muse_decode_layer/dev/p12_components/static_cache.py')
cache=importlib.util.module_from_spec(spec);spec.loader.exec_module(cache);cache.install(local_eval)
forced='--forced' in sys.argv;loader=local_eval.load_module;calls={}
def load(label,path):
    m=loader(label,path)
    if label!='submission':return m
    m.DC_FORCE=forced
    if forced:
        class VD:
            def __init__(self,sparse):self.sparse=sparse
            def __getitem__(self,grid):
                def run(x,w,p,xd,wd,t,n,k,s,**kw):
                    assert p.shape==(s,t,n) and grid==(m.triton.cdiv(t,128)*(n//64),s)
                    if self.sparse:
                        raw=m.torch.empty((n,k),device=x.device,dtype=m.torch.bfloat16)
                        m.dc_unpack(dict(coded=w,N=n,K=k),raw)
                    else:raw=w
                    p.zero_();p[0].copy_(m.torch.mm(x,raw.T,out_dtype=m.torch.float32))
                    key='sparse_vdsk' if self.sparse else 'control_vdsk';calls[key]=calls.get(key,0)+1
                return run
        m._dc_vdsk_gemm=VD(True);m._vdsk_gemm=VD(False)
    for name in ('dc_project','dc_nf_project','dc_ol_project','dc_library_project','dc_vdsk_project'):
        fn=getattr(m,name)
        def count(*a,_fn=fn,_name=name,**kw):
            key=_name+('_graph' if m.torch.cuda.is_current_stream_capturing() else '_eager')
            calls[key]=calls.get(key,0)+1;return _fn(*a,**kw)
        setattr(m,name,count)
    return m
local_eval.load_module=load
out=H/'screen'/('forced' if forced else 'normal');out.mkdir(parents=True,exist_ok=True)
sys.argv=['local_eval.py',str(H/'candidate.py'),'--no-baseline','--reps','2','--inputs','3','--json',str(out/'result.json')]
try:local_eval.main()
finally:(out/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
