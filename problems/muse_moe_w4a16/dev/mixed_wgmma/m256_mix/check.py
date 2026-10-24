"""Independent FP32, changing-route capture, threshold edges and pointer-owner checks."""
import json
import runpy
from pathlib import Path
import torch

H=Path(__file__).resolve().parent;R=H.parents[4]


def main():
    m=runpy.run_path(str(R/'prof/lab/moe_m256_mix.py'));rows=[]
    def gate(out,want,label):
        e,r=m['mx_error'](out,want);assert e<.01 and r<.05,(label,e,r);rows.append((e,r))
    for seed in range(12):
        w=m['dc_weights'](5353+7919*seed,'cuda');bank=m['mx_prepare'](w);ds=m['dc_fixtures'](w,256);wants=m['dc_oracles'](ds,w)
        for name,a,b,q,g,c in m['MX_RECIPES']:
            graph=m['MWGraph'](lambda d,a=a,b=b,q=q,g=g,c=c:m['mx_call'](d,bank,a,b,q,g,c),ds[0])
            for d,want in zip(ds,wants):graph.refresh(d);gate(graph.replay(),want,(seed,name))
        print('MIX_SEED_PASS',seed,len(rows),flush=True)
    # A changed route is valid even if it represents a router cutoff flip.
    # Both arms and the independent oracle consume its actual IDs and weights.
    w=m['dc_weights'](38471,'cuda')
    for key in ('w13_sf','w2_sf'):
        sf=w[key].view(torch.uint8);sf[...,::3]^=128;sf[...,1::37]=0;sf[...,2::41]=1
    bank=m['mx_prepare'](w);base=m['dc_fixtures'](w,256)[0];ds=[]
    hot=(0,1,7,8,9,15,16,17,31,32,33,63,64,65,119,128,129,255,256)
    for count in hot:
        ids=torch.full_like(base['topk_ids'],255);ids[:count,0]=0
        for col,n in enumerate((8,9,16,17,31,32,33),1):ids[:n,col]=col
        ds.append(dict(base,hidden_states=torch.randn_like(base['hidden_states']),topk_ids=ids))
    ds.append(dict(base,topk_ids=torch.full_like(base['topk_ids'],255)))
    ids=torch.full_like(base['topk_ids'],255);ids[:,0]=torch.arange(256,device='cuda')%32;ds.append(dict(base,topk_ids=ids))
    wants=m['dc_oracles'](ds,w)
    for decoder in (0,1,2):
        for a in m['MX_FIRST']:
            for b in m['MX_SECOND']:
                graph=m['MWGraph'](lambda d,a=a,b=b,q=decoder:m['mx_call'](d,bank,a,b,q),ds[0])
                for d,want in zip(ds,wants):graph.refresh(d);gate(graph.replay(),want,(decoder,a,b,'edge'))
        print('MIX_DECODER_EDGES_PASS',decoder,len(rows),flush=True)
    # Drop the plan before eager calls and allocator churn, poison every active
    # producer output, then verify no missing tile or route-prefix write is hidden.
    for a in m['MX_FIRST']:
        for b in m['MX_SECOND']:
            data=ds[-3];want=wants[-3];plan=m['mx_plan'](data,bank,a,b,2);steps=plan['steps'];v2=plan['v2'];tasks=plan['tasks'];del plan
            churn=[torch.empty((256,4096),device='cuda').fill_(77) for _ in range(3)]
            steps[0]();v2[9].fill_(float('nan'));v2[15].fill_(float('nan'));steps[1]();steps[2]();out=steps[3]();torch.cuda.synchronize()
            cnt=v2[4].cpu().tolist();off=v2[5].cpu().tolist()
            assert int(tasks[0][-1])==sum((n+a[1]-1)//a[1] for n in cnt if n>a[0])
            assert int(tasks[1][-1])==sum((n+b[1]-1)//b[1] for n in cnt if n>b[0])
            for e,n in enumerate(cnt):
                if n:
                    for idx in (9,15,16):assert bool(torch.isfinite(v2[idx][off[e]:off[e]+n]).all()),(a,b,e,idx)
            gate(out,want,(a,b,'poison'))
            shifted=dict(data,topk_ids=data['topk_ids']+64,config=dict(local_start=64))
            graph=m['MWGraph'](lambda d,a=a,b=b:m['mx_call'](d,bank,a,b,2),shifted);gate(graph.replay(),want,(a,b,'local_start64'))
    for size in (128,192,257):
        data=m['dc_fixtures'](w,size)[0];want=m['dc_oracles']([data],w)[0]
        for a in m['MX_FIRST']:
            for b in m['MX_SECOND']:gate(m['mx_call'](data,bank,a,b,2),want,(a,b,size))
    for key in ('w13_sf','w2_sf'):w[key].zero_()
    bank=m['mx_prepare'](w);data=ds[-1]
    for a in m['MX_FIRST']:
        for b in m['MX_SECOND']:gate(m['mx_call'](data,bank,a,b,2),torch.zeros_like(data['hidden_states']),(a,b,'allzero'))
    result=dict(checks=len(rows),max_L2=max(e for e,r in rows),max_row=max(r for e,r in rows),seeds=12,fixtures_per_seed=3,thresholds=[8,16,32],heavy_N=[32,64],decoders=[0,1,2],hot_counts=hot,
                signed_zero_subnormal=True,allzero_scale=True,independent_gate_up=True,actual_route_oracle=True,poisoned_active_outputs=True,scratch_owner_churn=True,nonzero_local_start=True,nearby_M=[128,192,257])
    (H/'agreement.json').write_text(json.dumps(result,indent=2)+'\n');print('MIX_ALL_PASS',result,flush=True)


if __name__=='__main__':main()
