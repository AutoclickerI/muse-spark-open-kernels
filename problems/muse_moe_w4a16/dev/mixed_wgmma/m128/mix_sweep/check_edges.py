"""Check every threshold and N64 edge with independent output/scratch gates."""
import importlib.util
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def main():
    spec=importlib.util.spec_from_file_location('sw_edges',ROOT/'prof/lab/moe_m128_mix_sweep.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    w=m.dc_weights(38471,'cuda')
    for key in ('w13_sf','w2_sf'):
        sf=w[key].view(torch.uint8);sf[...,::3]^=128;sf[...,1::37]=0;sf[...,2::41]=1
    bank=m.hh_prepare(w);ds=m.dc_fixtures(w,128);hot=(0,1,7,8,9,15,16,17,23,24,25,31,32,33,63,64,65,80,128)
    for count in hot:
        ids=torch.full_like(ds[0]['topk_ids'],255);ids[:count,0]=0
        # All four cuts are simultaneously crossed by other experts.
        for col,n in enumerate((8,9,16,17,24,25,32),1):ids[:n,col]=col
        ds.append(dict(ds[0],hidden_states=torch.randn_like(ds[0]['hidden_states']),topk_ids=ids))
    wants=m.dc_oracles(ds,w);rows=[]
    for name,a,n,b,k in m.SW_CHOICES:
        graph=m.MWGraph(lambda d,a=a,n=n,b=b,k=k:m.sw_call(d,bank,a,n,b,k),ds[0])
        for i,(data,want) in enumerate(zip(ds,wants)):
            graph.refresh(data);out=graph.replay();torch.cuda.synchronize();e,r=m.pm_error(out,want)
            assert e<=.01 and r<=.05,(name,i,e,r);rows.append(dict(name=name,fixture=i,L2=e,row=r))
        plan=m.sw_plan(ds[-1],bank,a,n,b,k);steps=plan['steps'];v2=plan['v2'];tasks=plan['tasks'];del plan
        churn=[torch.empty((128,4096),device='cuda').fill_(77) for _ in range(3)]
        steps[0]();v2[9].fill_(float('nan'));v2[15].fill_(float('nan'))
        steps[1]();steps[2]();out=steps[3]();torch.cuda.synchronize()
        c=v2[4].cpu().tolist();off=v2[5].cpu().tolist()
        assert int(tasks[0][-1])==sum((cnt+n-1)//n for cnt in c if cnt>a),(name,'W13 tasks')
        assert int(tasks[1][-1])==sum((cnt+k-1)//k for cnt in c if cnt>b),(name,'W2 tasks')
        for e,cnt in enumerate(c):
            if cnt:
                for idx in (9,15,16):assert bool(torch.isfinite(v2[idx][off[e]:off[e]+cnt]).all()),(name,e,idx)
        e,r=m.pm_error(out,wants[-1]);assert e<=.01 and r<=.05,(name,'poison',e,r)
        shifted=dict(ds[0],topk_ids=ds[0]['topk_ids']+64,config=dict(ds[0]['config'],local_start=64))
        graph=m.MWGraph(lambda d,a=a,n=n,b=b,k=k:m.sw_call(d,bank,a,n,b,k),shifted);out=graph.replay();torch.cuda.synchronize()
        e,r=m.pm_error(out,wants[0]);assert e<=.01 and r<=.05,(name,'local_start64',e,r)
        rows.append(dict(name=name,fixture='local_start64',L2=e,row=r))
    report=dict(checks=len(rows),poisoned_active_ACT_Y_SS_complete=True,task_prefixes_exact=True,signed_zero_subnormal_scales=True,separate_gate_up_globals=True,
        thresholds=[8,16,24,32],hot_counts=hot,max_L2=max(v['L2'] for v in rows),max_row=max(v['row'] for v in rows),rows=rows)
    (HERE/'edges.json').write_text(json.dumps(report,indent=2)+'\n');print('SWEEP_EDGES_PASS',len(rows),report['max_L2'],report['max_row'],flush=True)


if __name__=='__main__':main()
