"""Gate the finalizer port and real shared-A reuse across hot N64/N128 boundaries."""
import importlib.util
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def main():
    spec=importlib.util.spec_from_file_location('one_read_edges',ROOT/'prof/lab/moe_m128_no_reread.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    w=m.dc_weights(38471,'cuda')
    for key in ('w13_sf','w2_sf'):
        sf=w[key].view(torch.uint8);sf[...,::3]^=128;sf[...,1::37]=0;sf[...,2::41]=1
    bank=m.issue_prepare(w);records=[]
    ds=m.dc_fixtures(w,128)
    for count in (0,1,16,32,50,64,65,80,128):
        ids=torch.full_like(ds[0]['topk_ids'],255)
        ids[:count,0]=0
        ds.append(dict(ds[0],hidden_states=torch.randn_like(ds[0]['hidden_states']),topk_ids=ids))
    wants=m.dc_oracles(ds,w)
    for name,prefix,cfg,grid in m.OR_CHOICES:
        graph=m.MWGraph(lambda d,p=prefix,c=cfg,z=grid:m.or_call(d,bank,p,c,z),ds[0])
        for i,(data,want) in enumerate(zip(ds,wants)):
            graph.refresh(data);out=graph.replay();torch.cuda.synchronize();e,r=m.or_error(out,want)
            assert e<=.01 and r<=.05,(name,i,e,r)
            records.append(dict(variant=name,M=128,fixture=i,L2=e,row=r))
    # Shape is public metadata, so the same expert-wide CTA gets capacity256
    # or384. Q/SF loads remain outside the shared-A subgroup loop even when a
    # single expert crosses N128 or N256. No expert token set is reread.
    for tokens in (129,257):
        data=m.dc_fixtures(w,tokens)[0];ids=torch.full_like(data['topk_ids'],255);ids[:,0]=0
        data=dict(data,topk_ids=ids);want=m.dc_oracles([data],w)[0]
        for prefix in ('or_ss64','or_ss128'):
            plan=m.or_plan(data,bank,prefix,0,528);steps=plan['steps'];v13=plan['v13'];v2=plan['v2'];nt=plan['NT'];del plan
            # Retain only launch closures; force allocator reuse before eager
            # and graph capture to audit raw-pointer tensor ownership.
            churn=[torch.empty((tokens,4096),device='cuda').fill_(77) for _ in range(4)]
            steps[0]()
            for index in (9,15,16):v2[index].fill_(float('nan'))
            steps[1]();steps[2]();out=steps[3]();torch.cuda.synchronize()
            for index in (9,15,16):assert bool(torch.isfinite(v2[index][:tokens]).all()),(prefix,tokens,index)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for fn in steps:out=fn()
            graph.replay();torch.cuda.synchronize();e,r=m.or_error(out,want)
            counts=v13[4].cpu();assert int(((counts+nt-1)//nt).sum())==1
            assert e<=.01 and r<=.05,(prefix,tokens,e,r)
            records.append(dict(variant=prefix,M=tokens,capacity=nt,logical_weight_read_factor=1.,L2=e,row=r))
    # Nonzero local_start is fixed metadata for each captured graph.
    for prefix,cfg in (('or_rs',3),('or_ss64',0),('or_ss128',0)):
        original=ds[0];data=dict(original,topk_ids=original['topk_ids']+64,config=dict(original['config'],local_start=64))
        g=m.MWGraph(lambda d,p=prefix,c=cfg:m.or_call(d,bank,p,c,528),data);out=g.replay();torch.cuda.synchronize()
        e,r=m.or_error(out,wants[0]);assert e<=.01 and r<=.05,(prefix,e,r)
        records.append(dict(variant=prefix,M=128,local_start=64,L2=e,row=r))
    report=dict(checks=len(records),signed_zero_subnormal_scales=True,separate_gate_up_global_scales=True,
        hot_counts=[0,1,16,32,50,64,65,80,128,129,257],max_L2=max(r['L2'] for r in records),max_row=max(r['row'] for r in records),results=records)
    (HERE/'edges.json').write_text(json.dumps(report,indent=2)+'\n');print('ONE_READ_EDGES_PASS',len(records),report['max_L2'],report['max_row'],flush=True)


if __name__=='__main__':main()
