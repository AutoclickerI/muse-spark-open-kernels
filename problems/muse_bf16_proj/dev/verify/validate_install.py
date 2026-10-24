"""Exercise shortlist setup, tuple dispatch, rejection and nearby-shape routing."""
import importlib.util
import time
from pathlib import Path

import torch

path = Path(__file__).with_name('projection_dropin.py')
spec = importlib.util.spec_from_file_location('dropin',path)
vp = importlib.util.module_from_spec(spec);spec.loader.exec_module(vp)


def old(x,w,name,state,out_dtype=torch.bfloat16):
    out = torch.mm(x,w.T,**({'out_dtype':torch.float32} if out_dtype==torch.float32 else {}))
    return out.unsqueeze(0),1


def main():
    state={n:torch.randn(512,1024,device='cuda',dtype=torch.bfloat16)
           for n in ('qkv_gate','pool_router','unpooler','o')}
    c=(64,1,1,4,1,2,1,1)
    recipes={(m,name):[('old',),('tma',c),('grouped',(64,1,1,3,2,2,1,1,2)),('lt',0,0),('lt',1,0)]
             for m in (128,256) for name in state}
    timing=vp.vp_pair;seen=[]
    def functional_pair(control,trial,x,banks,flush,repeats=20):
        for fresh in banks:
            x.copy_(fresh);control.graph.replay();trial.graph.replay()
            vp.vp_check(trial.out,control.out)
        seen.append(repeats)
        # Force selection through every family without asserting local speed.
        return 1.,1./(1+len(seen)),1./(1+len(seen))
    try:
        vp.vp_pair=functional_pair
        rows=vp.verify_proj_prepare(state,old,time.monotonic()+240,recipes)
    finally:
        vp.vp_pair=timing
    assert len(state['_verify_proj']['plan'])==8,rows
    assert sum('unknown projection family' in r for r in rows)==8,rows
    assert len(seen)==32 and set(seen)=={20},(seen,rows)
    project=vp.verify_proj_bind(old)
    x=torch.randn(128,1024,device='cuda',dtype=torch.bfloat16)
    vp.vp_check(project(x,state['o'],'o',{})[0][0],old(x,state['o'],'o',{})[0][0])
    count=0
    for m in (127,128,129,256,257):
        for name,w in ((n,state[n]) for n in ('qkv_gate','pool_router','unpooler','o')):
            dtype=torch.float32 if name=='pool_router' else torch.bfloat16
            for stride in (1,2):
                x=torch.randn(m,1024*stride,device='cuda',dtype=torch.bfloat16)[:,::stride]
                with vp._PgGraph(lambda:vp._vp_materialize(project(x,w,name,state,out_dtype=dtype))) as graph:
                    for _ in range(3):
                        x.normal_();graph.graph.replay();vp.vp_check(graph.out,old(x,w,name,state,out_dtype=dtype)[0][0]);count+=1
            # A new weight object with the same shape must use that new weight.
            w2=w*2;x=torch.randn(m,1024,device='cuda',dtype=torch.bfloat16)
            vp.vp_check(vp.verify_project(x,w2,name,state,out_dtype=dtype)[0][0],old(x,w2,name,state,out_dtype=dtype)[0][0]);count+=1
    expired=dict(state)
    rows=vp.verify_proj_prepare(expired,old,time.monotonic()-1,recipes)
    assert not expired['_verify_proj']['plan'] and any('BUDGET' in r for r in rows)
    rows=vp.verify_proj_prepare(expired,old,time.monotonic()+20,{(128,'o'):[('tma',c)]})
    assert not expired['_verify_proj']['plan'] and any('BUDGET compile' in r for r in rows)
    # Both module and graph-construction failures leave an executable incumbent.
    constructor=vp.vp_tma_module
    try:
        def broken():raise AttributeError('intentional missing native entrypoint')
        vp.vp_tma_module=broken
        rows=vp.verify_proj_prepare(expired,old,time.monotonic()+240,{(128,'o'):[('tma',c)]})
        assert not expired['_verify_proj']['plan'] and any('AttributeError' in r for r in rows)
    finally:vp.vp_tma_module=constructor
    assert vp._PgGraph.active==0
    print('VP_INSTALL_VALIDATION',dict(fresh_checks=count,optional_trials=len(seen),
          unknown_family_rejections=8,expired_budget=True,module_failure=True),flush=True)


if __name__=='__main__':main()
