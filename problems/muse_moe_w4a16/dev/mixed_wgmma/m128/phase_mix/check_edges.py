"""Audit dynamic threshold transitions, phase ownership and active scratch writes."""
import hashlib
import json
from pathlib import Path
import torch
from quick import load

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def main():
    m=load('phase_mix_edges',ROOT/'prof/lab/moe_m128_phase_mix.py');w=m.dc_weights(38471,'cuda')
    for key in ('w13_sf','w2_sf'):
        sf=w[key].view(torch.uint8);sf[...,::3]^=128;sf[...,1::37]=0;sf[...,2::41]=1
    base=m.issue_prepare(w);word=m.issue_word_bank(base);native=m.issue_nt_prepare(w)
    ds=m.dc_fixtures(w,128)
    for count in (0,1,8,15,16,17,31,32,33,64,65,80,128):
        ids=torch.full_like(ds[0]['topk_ids'],255);ids[:count,0]=0
        # Adjacent disjoint experts straddle the branch at the same time.
        ids[:16,1]=1;ids[:17,2]=2
        ds.append(dict(ds[0],hidden_states=torch.randn_like(ds[0]['hidden_states']),topk_ids=ids))
    wants=m.dc_oracles(ds,w);rows=[]
    for name,a,b in m.PM_CHOICES:
        graph=m.MWGraph(lambda d,a=a,b=b:m.pm_call(d,base,word,native,a,b),ds[0])
        for i,(data,want) in enumerate(zip(ds,wants)):
            graph.refresh(data);out=graph.replay();torch.cuda.synchronize();e,r=m.pm_error(out,want)
            assert e<=.01 and r<=.05,(name,i,e,r);rows.append(dict(name=name,fixture=i,L2=e,row=r))
        # Keep only closures to check tensor lifetime despite dropping plan.
        data=ds[-1];plan=m.pm_plan(data,base,word,native,a,b);steps=plan['steps'];v2=plan['v2'];tasks=plan['tasks'];del plan
        churn=[torch.empty((128,4096),device='cuda').fill_(77) for _ in range(3)]
        steps[0]();v2[9].fill_(float('nan'));v2[15].fill_(float('nan'))
        steps[1]();steps[2]();out=steps[3]();torch.cuda.synchronize()
        counts=v2[4].cpu().tolist();off=v2[5].cpu().tolist();order=v2[6].cpu().tolist()
        assert [int(t[-1]) for t in tasks]==[sum((n+31)//32 for n in counts),sum((n+31)//32 for n in counts if n>16),sum((n+127)//128 for n in counts),sum((n+15)//16 for n in counts)]
        for e,n in enumerate(counts):
            if n:
                for idx in (9,15,16):assert bool(torch.isfinite(v2[idx][off[e]:off[e]+n]).all()),(name,e,n,idx)
        err,row=m.pm_error(out,wants[-1]);assert err<=.01 and row<=.05,(name,err,row)
        shifted=dict(ds[0],topk_ids=ds[0]['topk_ids']+64,config=dict(ds[0]['config'],local_start=64))
        graph=m.MWGraph(lambda d,a=a,b=b:m.pm_call(d,base,word,native,a,b),shifted);out=graph.replay();torch.cuda.synchronize()
        e,r=m.pm_error(out,wants[0]);assert e<=.01 and r<=.05,(name,'local_start64',e,r)
        rows.append(dict(name=name,fixture='local_start64',L2=e,row=r))
    # Public fixed handoff is the exact D32g wrapper used by integration.
    cell=load('d32g_handoff_edge',HERE/'d32g_cell.py');bank=cell.d32g_prepare(w)
    graph=m.MWGraph(lambda d:cell.d32g_experts(d['hidden_states'],d['topk_weights'],d['topk_ids'],bank),ds[0])
    for i,(data,want) in enumerate(zip(ds,wants)):
        graph.refresh(data);out=graph.replay();torch.cuda.synchronize();e,r=m.pm_error(out,want)
        assert e<=.01 and r<=.05,('D32g_cell',i,e,r);rows.append(dict(name='D32g_cell',fixture=i,L2=e,row=r))
    report=dict(checks=len(rows),signed_zero_subnormal_scales=True,separate_gate_up_global_scales=True,
        threshold_counts=[0,1,8,15,16,17,31,32,33,64,65,80,128],active_ACT_Y_SS_finite=True,
        cell_sha256=hashlib.sha256((HERE/'d32g_cell.py').read_bytes()).hexdigest(),max_L2=max(r['L2'] for r in rows),max_row=max(r['row'] for r in rows),rows=rows)
    (HERE/'edges.json').write_text(json.dumps(report,indent=2)+'\n');print('PHASE_MIX_EDGES_PASS',len(rows),report['max_L2'],report['max_row'],flush=True)


if __name__=='__main__':main()
