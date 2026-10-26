"""Published TASK checker for every coded arm; only DSM is replaced locally."""
import sys,json,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer/dev/p12_components')]
import local_eval,static_cache
static_cache.install(local_eval);load=local_eval.load_module;outputs=[];seen={}
forced='--forced' in sys.argv
def module(name,path):
    m=load(name,path)
    if name!='submission':return m
    m._P12_FORCE=forced
    if forced:
        class VD:
            def __getitem__(self,grid):
                def run(x,w,p,xd,wd,t,n,k,s,**kw):p.zero_();p[0].copy_(m.torch.mm(x,w.T,out_dtype=m.torch.float32))
                return run
        m._vdsk_gemm=VD()
    prepare=m.prepare;original=m.kernel_fn
    def init(weights):
        st=prepare(weights);st['sgq']=m.sgq_prepare(weights,True);return st
    def call(data,st):
        outputs.clear();t=data['x'].shape[0]
        if data['config']['layer']=='qkv_gate' and t in (8,16):
            for n in (2304,1280):
                d=dict(data,config=dict(data['config'],n=n))
                for arm in range(len(m.SGQ_CONFIGS)):
                    out=m._p12_materialize(m.sgq_project(data['x'],n,arm,st['sgq']));outputs.append((d,arm,out));key=str((t,n,arm));seen[key]=seen.get(key,0)+1
            return outputs[0][2]
        return original(data,st)
    m.prepare=init;m.kernel_fn=call;return m
local_eval.load_module=module;check=local_eval.Problem.check
def verify(self,data,out):
    for d,arm,value in outputs:
        ok,msg=check(self,d,value)
        if not ok:return False,f'QKV {d["config"]["n"]} arm{arm}: '+msg
    return check(self,data,out)
local_eval.Problem.check=verify;folder=H/'screen'/('qkv_forced' if forced else 'qkv_normal');folder.mkdir(parents=True,exist_ok=True)
sys.argv=['local_eval.py',str(H/'qkv_candidate.py'),'--no-baseline','--reps','4','--inputs','3','--json',str(folder/'result.json')]
try:local_eval.main()
finally:(folder/'arms.json').write_text(json.dumps(dict(candidate_sha256=hashlib.sha256((H/'qkv_candidate.py').read_bytes()).hexdigest(),standin='S1 CTA-local reduction for S2/S4 DSM; exact decoder/MMA/metadata execute',counts=seen),indent=2)+'\n')
