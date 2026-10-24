"""Fresh graph gates for signed/zero/subnormal block-scale metadata and the cell."""
import importlib.util
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    m=load('issue_lab_edges',ROOT/'prof/lab/moe_m128_issue_limit.py')
    cell=load('issue_nt1_cell',HERE/'dropin_nt1_lop.py')
    w=m.dc_weights(38471,'cuda');w['w13_gs_gate']=w['w13_gs'];w['w13_gs_up']=w['w13_gs']
    for key in ('w13_sf','w2_sf'):
        raw=w[key].view(torch.uint8);raw[...,::3]^=128;raw[...,1::37]=0;raw[...,2::41]=1
    base=m.issue_prepare(w);word=m.issue_word_bank(base);native=cell.m128_nt1lop_issue_nt_prepare(w)
    ds=m.dc_fixtures(w,128)
    for kind in ('empty','single','hot','balanced'):
        ids=torch.full_like(ds[0]['topk_ids'],255)
        if kind=='single':ids[3,2]=31
        if kind=='hot':ids[:,0]=0
        if kind=='balanced':ids[:,0]=torch.arange(128,device='cuda')%32
        ds.append(dict(ds[0],hidden_states=torch.randn_like(ds[0]['hidden_states']),topk_ids=ids))
    wants=m.dc_oracles(ds,w);results=[]
    functions=[('standaloneNT1lop',lambda d:cell.m128_nt1lop_issue_nt_call(d,native))]
    functions += [(name,lambda d,p=p,c=c,z=z,b=base if p=='ibase' else word:m.issue_call(d,b,p,c,z)) for name,p,c,z in m.ISSUE_CHOICES]
    for name,fn in functions:
        graph=m.MWGraph(fn,ds[0])
        for i,(data,want) in enumerate(zip(ds,wants)):
            graph.refresh(data);out=graph.replay();torch.cuda.synchronize();e,r=m.issue_error(out,want)
            assert e<=.01 and r<=.05,(name,i,e,r)
            results.append(dict(variant=name,fixture=i,L2=e,row=r))
    report=dict(checks=len(results),max_L2=max(r['L2'] for r in results),max_row=max(r['row'] for r in results),
        signed_zero_subnormal_scales=True,route_transitions=['real0','real1','real2','empty','single','hot','balanced'],results=results)
    (HERE/'edges.json').write_text(json.dumps(report,indent=2)+'\n');print('ISSUE_EDGES_PASS',len(results),report['max_L2'],report['max_row'])


if __name__=='__main__':main()
