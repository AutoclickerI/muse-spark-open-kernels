"""Check threshold boundaries, ring wrap, complete writes and scale exactness."""
import importlib.util
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def main():
    spec=importlib.util.spec_from_file_location('pc_edges',ROOT/'prof/lab/moe_m128_producer_consumer.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    w=m.dc_weights(38471,'cuda')
    for key in ('w13_sf','w2_sf'):
        sf=w[key].view(torch.uint8);sf[...,::3]^=128;sf[...,1::37]=0;sf[...,2::41]=1
    bank=m.hh_prepare(w);rows=[]
    for M in (16,32,128):
        ds=m.dc_fixtures(w,M);counts=[n for n in (0,1,7,8,15,16,17,31,32,33,63,64,65,80,128) if n<=M]
        for count in counts:
            ids=torch.full_like(ds[0]['topk_ids'],255);ids[:count,0]=0
            ds.append(dict(ds[0],hidden_states=torch.randn_like(ds[0]['hidden_states']),topk_ids=ids))
        wants=m.dc_oracles(ds,w);cut=0 if M<128 else 16
        choices=[(f'cfg{cfg}',cfg,528,3,cut) for cfg in range(4)]
        if M==128:choices=list(m.PC_CHOICES)
        for name,cfg,grid,mask,pcut in choices:
            graph=m.MWGraph(lambda d,c=cfg,g=grid,p=mask,q=pcut:m.pc_call(d,bank,c,g,p,cut,q),ds[0])
            # Replay long graphs in reverse fixture order as well, so counters
            # and barrier phases must reset after heavy/empty route transitions.
            for fi in list(range(len(ds)))+list(range(len(ds)-1,-1,-1)):
                graph.refresh(ds[fi]);out=graph.replay();torch.cuda.synchronize();e,r=m.pm_error(out,wants[fi])
                assert e<=.01 and r<=.05,(M,name,fi,e,r);rows.append(dict(M=M,name=name,fixture=fi,L2=e,row=r))
            plan=m.pc_plan(ds[-1],bank,cfg,grid,mask,cut,pcut);steps=plan['steps'];v2=plan['v2'];tasks=plan['tasks'];del plan
            churn=[torch.empty((M,4096),device='cuda').fill_(77) for _ in range(3)]
            steps[0]();v2[9].fill_(float('nan'));v2[15].fill_(float('nan'))
            steps[1]();steps[2]();out=steps[3]();torch.cuda.synchronize()
            c=v2[4].cpu().tolist();off=v2[5].cpu().tolist();assert int(tasks[1][-1])==sum(n>pcut for n in c)
            assert int(tasks[3][-1])==sum((n+31)//32 for n in c if cut<n<=pcut)
            for expert,n in enumerate(c):
                if n:
                    for idx in (9,15,16):assert bool(torch.isfinite(v2[idx][off[expert]:off[expert]+n]).all()),(M,name,expert,idx)
            e,r=m.pm_error(out,wants[-1]);assert e<=.01 and r<=.05,(M,name,'poison',e,r)
            shifted=dict(ds[0],topk_ids=ds[0]['topk_ids']+64,config=dict(ds[0]['config'],local_start=64))
            graph=m.MWGraph(lambda d,c=cfg,g=grid,p=mask,q=pcut:m.pc_call(d,bank,c,g,p,cut,q),shifted);out=graph.replay();torch.cuda.synchronize()
            e,r=m.pm_error(out,wants[0]);assert e<=.01 and r<=.05,(M,name,'start64',e,r)
            rows.append(dict(M=M,name=name,fixture='start64',L2=e,row=r))
    report=dict(checks=len(rows),signed_zero_subnormal_scales=True,separate_gate_up_globals=True,poisoned_active_ACT_Y_SS_complete=True,
        expert_wide_tasks_one_per_selected_expert=True,rings_wrapped_many_times=True,max_L2=max(v['L2'] for v in rows),max_row=max(v['row'] for v in rows),rows=rows)
    (HERE/'edges.json').write_text(json.dumps(report,indent=2)+'\n');print('PC_EDGES_PASS',len(rows),report['max_L2'],report['max_row'],flush=True)


if __name__=='__main__':main()
