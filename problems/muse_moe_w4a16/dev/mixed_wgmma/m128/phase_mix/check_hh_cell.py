"""Exercise the actual fixed handoff wrapper across changed routes and scales."""
import hashlib
import json
from pathlib import Path
import importlib.util
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    m=load('hh_control',ROOT/'prof/lab/moe_m128_phase_mix.py');cell=load('hh_fixed',HERE/'hh_cell.py')
    w=m.dc_weights(38471,'cuda')
    for key in ('w13_sf','w2_sf'):
        sf=w[key].view(torch.uint8);sf[...,::3]^=128;sf[...,1::37]=0;sf[...,2::41]=1
    bank=cell.hh_prepare(w);ds=m.dc_fixtures(w,128)
    for count in (0,1,15,16,17,32,65,80,128):
        ids=torch.full_like(ds[0]['topk_ids'],255);ids[:count,0]=0
        ds.append(dict(ds[0],hidden_states=torch.randn_like(ds[0]['hidden_states']),topk_ids=ids))
    wants=m.dc_oracles(ds,w);g=m.MWGraph(lambda d:cell.hh_experts(d['hidden_states'],d['topk_weights'],d['topk_ids'],bank),ds[0]);rows=[]
    for repeat in range(2):
        for i in (range(len(ds)) if repeat==0 else range(len(ds)-1,-1,-1)):
            g.refresh(ds[i]);out=g.replay();torch.cuda.synchronize();e,r=m.pm_error(out,wants[i]);assert e<=.01 and r<=.05,(i,e,r)
            rows.append(dict(fixture=i,repeat=repeat,L2=e,row=r))
    shifted=dict(ds[0],topk_ids=ds[0]['topk_ids']+64,config=dict(ds[0]['config'],local_start=64))
    g=m.MWGraph(lambda d:cell.hh_experts(d['hidden_states'],d['topk_weights'],d['topk_ids'],bank,64),shifted)
    out=g.replay();torch.cuda.synchronize();e,r=m.pm_error(out,wants[0]);assert e<=.01 and r<=.05,(e,r)
    rows.append(dict(fixture='local_start64',repeat=0,L2=e,row=r))
    report=dict(cell_sha256=hashlib.sha256((HERE/'hh_cell.py').read_bytes()).hexdigest(),checks=len(rows),max_L2=max(v['L2'] for v in rows),max_row=max(v['row'] for v in rows),rows=rows)
    (HERE/'hh_agreement.json').write_text(json.dumps(report,indent=2)+'\n');print('HH_CELL_PASS',len(rows),report['max_L2'],report['max_row'],flush=True)


if __name__=='__main__':main()
