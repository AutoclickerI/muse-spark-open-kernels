"""Check the multicast module's raw ABI and TMA/MMA stand-in on native sm120.
Hopper WGMMA execution still requires the judge.
"""
import importlib.util
import json
from pathlib import Path
import sys
import torch
P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P))
import reference as ref
spec=importlib.util.spec_from_file_location('max_proj',P/'cand_portfolio_max.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
mod=c._module();weights=ref.generate_weights(ref.WEIGHTS_SEED)
r=weights['router'];hi=r.bfloat16();lo=(r-hi.float()).bfloat16();pair=torch.cat((hi,lo),1).contiguous()
props=torch.cuda.get_device_properties(0);cases=0
for bt,ns,nc,st,mc in c.WIDE_SPECS:
    br=64*ns*nc;smem=max(st*(br+bt)*128+16*st+1024,64*(br+4)*4+1024)
    if smem>props.shared_memory_per_block_optin:continue
    for layer in ('router','qkv_gate'):
        data=ref.generate_input(m=bt,layer=layer,seed=7319,dist='outliers');x=data['x'];n=data['config']['n']
        if n%br or (n//br)%mc:continue
        w=pair if layer=='router' else weights[layer]
        out=torch.empty((bt,n),device='cuda',dtype=torch.float32 if layer=='router' else torch.bfloat16)
        for ks in (1,4):
            cfg=(bt,ns,nc,st,ks,mc)
            graph,out=c._graph(lambda:(mod.wide(w,x,out,*cfg),out)[1])
            for seed in (7319,53327,98411):
                fresh=ref.generate_input(m=bt,layer=layer,seed=seed,dist='outliers')
                x.copy_(fresh['x']);graph.replay();torch.cuda.synchronize()
                ok,msg=ref.check_implementation(data,out,weights)
                assert ok,(layer,cfg,msg);cases+=1
print(json.dumps({'native_wide_tma_mma_replay_cases':cases,'ok':True}),flush=True)
