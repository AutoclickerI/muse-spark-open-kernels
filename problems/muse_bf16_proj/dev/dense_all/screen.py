"""All coded projections under the unchanged official projection checker."""
import json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer/dev/p12_components')]
import local_eval,static_cache
import portable
static_cache.install(local_eval);load=local_eval.load_module;outputs=[];seen={};forced='--forced' in sys.argv
def module(name,path):
    m=load(name,path)
    if name!='submission':return m
    m.DA_LOCAL_MM=portable.mm
    m._P12_FORCE=forced
    if forced:
        class VD:
            def __getitem__(self,grid):
                def run(x,w,p,xd,wd,t,n,k,s,**kw):p.zero_();p[0].copy_(m.torch.mm(x,w.T,out_dtype=m.torch.float32))
                return run
        m._vdsk_gemm=VD()
    old=m.prepare;call=m.kernel_fn
    def prepare(weights):
        st=old(weights);st['da']=m.da_prepare(weights,True,True);return st
    def kernel(data,st):
        outputs.clear();x=data['x'];t=x.shape[0];role=data['config']['layer'];bank=st['da']
        if role!='router' and (t<=32 or t in (128,256)):
            actual='pool_router' if role=='pooler' else role
            roles=[actual]+(['qkv3','gate'] if role=='qkv_gate' else [])
            for name in roles:
                key='unpooler_slice' if name=='unpooler' and t<=8 else name;w=bank['roles'][key]
                for arm in range(len(m.DA_SMALL if t<=32 else m.DA_WIDE)):
                    out=m.da_materialize(m.da_project(x,name,bank,arm));d=dict(x=x,config=dict(layer='qkv_gate',m=t,n=w.shape[0],k=w.shape[1]));outputs.append((d,dict(qkv_gate=w),arm,out));seen[str((t,name,arm))]=seen.get(str((t,name,arm)),0)+1
            # Official ABI for pooler is N4096, while the extra check above uses
            # the layer's full N4608 pool+router bank.
            return outputs[0][3][:,:4096] if role=='pooler' else outputs[0][3]
        return call(data,st)
    m.prepare=prepare;m.kernel_fn=kernel;return m
local_eval.load_module=module;check=local_eval.Problem.check
def verify(self,data,out):
    for d,w,arm,value in outputs:
        ok,msg=self.ref.check_implementation(d,value,w)
        if not ok:return False,f'dense arm{arm}: '+msg
    return check(self,data,out)
local_eval.Problem.check=verify;folder=H/'screen'/('forced' if forced else 'normal');folder.mkdir(parents=True,exist_ok=True)
sys.argv=['local_eval.py',str(H/'adapter.py'),'--no-baseline','--reps','4','--inputs','3','--json',str(folder/'result.json')]
try:local_eval.main()
finally:(folder/'arms.json').write_text(json.dumps(seen,indent=2)+'\n')
