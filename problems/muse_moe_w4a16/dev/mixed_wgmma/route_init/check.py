"""Random-byte overwrite proof with the same captured plan through route transitions."""
import hashlib
import json
import runpy
from pathlib import Path
import torch

H=Path(__file__).resolve().parent;R=H.parents[4]


def main():
    p=R/'prof/lab/moe_route_init_m256.py';m=runpy.run_path(str(p));results=[];state_checks=0
    for seed in range(12):
        w=m['dc_weights'](5353+7919*seed,'cuda');bank=m['wi_prepare'](w);real=m['dc_fixtures'](w,256);base=real[0]
        none=dict(base,topk_ids=torch.full_like(base['topk_ids'],255))
        hot=torch.full_like(base['topk_ids'],255);hot[:,0]=0
        balanced=torch.full_like(hot,255);balanced[:,0]=torch.arange(256,device='cuda')%32
        one=torch.full_like(hot,255);one[17,3]=31
        ds=[none,real[0],none,dict(base,topk_ids=hot),none,real[1],dict(base,topk_ids=balanced),none,dict(base,topk_ids=one),real[2],none]
        wants=m['dc_oracles'](real,w);static={k:v.clone() if torch.is_tensor(v) else v for k,v in base.items()};plans=[m['ri_plan'](static,bank,s) for s in range(3)];graphs=[]
        for plan in plans:
            steps=plan['steps']
            for f in steps:out=f()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for f in steps:out=f()
            graphs.append(g)
        gen=torch.Generator(device='cuda').manual_seed(67201+seed)
        for i,d in enumerate(ds):
            for k in ('hidden_states','topk_ids','topk_weights'):static[k].copy_(d[k])
            outputs=[]
            for stage,(plan,g) in enumerate(zip(plans,graphs)):
                m['ri_poison'](plan,gen);before=plan['v2'][18].clone();g.replay();torch.cuda.synchronize();out=plan['v2'][19].clone();outputs.append(out)
                assert bool(torch.isfinite(out).all()),(seed,i,stage,'nonfinite')
                if stage:assert torch.equal(before,plan['v2'][18]),(seed,i,stage,'dead state modified');state_checks+=1
            for stage in (1,2):assert torch.equal(outputs[0],outputs[stage]),(seed,i,stage,'not bit exact')
            if d is none:
                for out in outputs:assert bool((out.view(torch.uint8)==0).all()),(seed,i,'none output not positive zero')
            results.append(dict(seed=seed,transition=i,bit_exact=True))
        for j,d in enumerate(real):
            for stage in range(3):
                got=m['ri_call'](d,bank,stage);e,r=m['ri_error'](got,wants[j]);assert e<=.015 and r<=.075,(seed,j,stage,e,r)
        print('ROUTE_POISON_SEED_PASS',seed,len(results),flush=True)
    # Nearby shapes, nonzero local expert range and strided original input IDs.
    extras=0
    for size in (128,192,257):
        d=m['dc_fixtures'](w,size)[0];want=m['dc_oracles']([d],w)[0];outs=[]
        for stage in range(3):
            dd=dict(d,topk_ids=d['topk_ids']+64,config=dict(local_start=64));plan=m['ri_plan'](dd,bank,stage);m['ri_poison'](plan,gen)
            for f in plan['steps']:out=f()
            e,r=m['ri_error'](out,want);assert e<=.015 and r<=.075;outs.append(out.clone());extras+=1
        assert all(torch.equal(outs[0],out) for out in outs[1:])
    report=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),weight_seeds=12,transitions_per_seed=11,transition_checks=len(results),captured_variant_replays=len(results)*3,FP32_checks=12*3*3+extras,dead_state_unchanged_checks=state_checks,random_byte_poison=True,none_some_none=True,all_output_channels_bit_exact=True,all_remote_positive_zero=True,nearby_M=[128,192,257],local_start64=True)
    (H/'agreement.json').write_text(json.dumps(report,indent=2)+'\n');print('ROUTE_OVERWRITE_PASS',report,flush=True)


if __name__=='__main__':main()
