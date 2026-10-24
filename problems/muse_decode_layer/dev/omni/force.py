"""Cold native compilation plus full Hopper glue on sm120 for cand_r3_omni.

Only native attention/CUTLASS projection launches use portable stand-ins.
The mixed MoE engine is quarantined; NT2/NT4 execute natively.
"""
from pathlib import Path
import sys,argparse,json
import torch
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'tools'))
sys.path.insert(0,str(ROOT/'problems/muse_decode_layer/dev'))
import local_eval
import verify_best



def install(mod,cold):
    calls=verify_best.install(mod,cold)
    vgc_factory=mod._omni_vgc_module
    branch={'cutlass':0,'nt2':0,'nt4':0}
    class PortableVgcPlan:
        def __init__(self,shape,dtype,recipe):self.shape=shape;self.dtype=dtype;self.recipe=recipe
        def run(self,x,w,out):
            assert x.shape==(self.shape[0],self.shape[2]) and out.dtype==self.dtype
            torch.mm(x,w.t(),out_dtype=self.dtype,out=out) if self.dtype==torch.float32 else torch.mm(x,w.t(),out=out)
            branch['cutlass']+=1
            return out
        def close(self):pass
    class PortableVgc:
        def plan(self,m,n,k,dtype,recipe,device):return PortableVgcPlan((m,n,k),dtype,recipe)
    def vgc():
        if cold:vgc_factory()
        mod._OMNI_VGC=PortableVgc();return mod._OMNI_VGC
    mod._omni_vgc_module=vgc
    reuse=mod.omni_reuse_call
    def reuse_call(data,bank):
        branch['nt'+str(bank['nt'])]+=1
        return reuse(data,bank)
    mod.omni_reuse_call=reuse_call
    original=mod.prepare
    def prepare(weights):
        state=original(weights)
        # Force all new projection recipes, then keep CUTLASS selected in the full graph screen.
        for t in (128,256):
            for name in ('qkv_gate','o','pool_router','unpooler'):
                w=state[name];dtype=torch.float32 if name=='pool_router' else torch.bfloat16
                for i,recipe in enumerate(mod._omni_proj_recipes(t,name)):
                    key=(t,name,i)
                    if key not in state['omni_cutlass']:state['omni_cutlass'][key]=mod._OMNI_VGC.plan(t,*w.shape,dtype,recipe,w.device)
                    for j in range(3):
                        x=torch.randn((t,w.shape[1]),device=w.device,dtype=torch.bfloat16)
                        actual=mod._materialize(mod._project_variant(x,w,name,state,('cutlass',i),dtype))
                        ref=x.float()@w.float().t()
                        assert mod._omni_error(actual,ref)<.01,(t,name,recipe)
                state['proj_plan'][t,name]=('cutlass',0)
        oracle_weights=dict(weights,post_norm_w=weights['post_expert_norm'])
        for t in (128,256):
            for j in range(3):
                data=mod.fx_make_fixture(weights,t,6301+t+7919*j)
                want=mod.rr_reuse_fp32(data,oracle_weights)
                for choice in ('nt2','nt4'):
                    state['omni_moe_plan'][t]=choice
                    out=mod.omni_experts(data['hidden_states'],data['topk_weights'],data['topk_ids'],state)
                    assert mod._omni_error(out,want)<.01,(t,choice,mod._omni_error(out,want))
        state['omni_moe_plan'].update({16:'nt4',32:'nt4',128:'nt2',256:'nt2'})
        # Make port_verify a complete layer choice (best's wrapper already forces XQA).
        for b in (1,16,32):
            state['attention_plan'][b,8,True]=('port_verify',None)
        print('OMNI_FORCE_BRANCHES',json.dumps(branch),flush=True)
        return state
    mod.prepare=prepare


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cold-native',action='store_true');args=ap.parse_args()
    original=local_eval.load_module
    def load(name,path):
        mod=original(name,path)
        if name=='submission':install(mod,args.cold_native)
        return mod
    local_eval.load_module=load
    sys.argv=[sys.argv[0],str(ROOT/'problems/muse_decode_layer/cand_r3_omni.py'),'--no-baseline',
        '--json',str(ROOT/'problems/muse_decode_layer/dev/omni_forced_local.json')]
    local_eval.main()


if __name__=='__main__':main()
