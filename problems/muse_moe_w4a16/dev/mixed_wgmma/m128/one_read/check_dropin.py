"""Verify the fixed best7 cell's actual public wrapper and recoded bank."""
import importlib.util
import json
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[5]


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    control=load('one_read_dropin_control',ROOT/'prof/lab/moe_m128_no_reread.py')
    cell=load('one_read_dropin',HERE/'dropin.py');w=control.dc_weights(38471,'cuda')
    bank=cell.m128_one_prepare(w);ds=control.dc_fixtures(w,128)
    for count in (0,1,65,80,128):
        ids=torch.full_like(ds[0]['topk_ids'],255);ids[:count,0]=0
        ds.append(dict(ds[0],hidden_states=torch.randn_like(ds[0]['hidden_states']),topk_ids=ids))
    wants=control.dc_oracles(ds,w)
    graph=control.MWGraph(lambda d:cell.m128_one_experts(d['hidden_states'],d['topk_weights'],d['topk_ids'],bank),ds[0]);results=[]
    for i,(data,want) in enumerate(zip(ds,wants)):
        graph.refresh(data);out=graph.replay();torch.cuda.synchronize();e,r=control.or_error(out,want)
        assert e<=.01 and r<=.05,(i,e,r);results.append(dict(fixture=i,L2=e,row=r))
    report=dict(checks=len(results),separate_gate_up_global_scales=True,max_L2=max(r['L2'] for r in results),max_row=max(r['row'] for r in results),results=results)
    (HERE/'dropin_agreement.json').write_text(json.dumps(report,indent=2)+'\n');print('ONE_READ_DROPIN_PASS',len(results),report['max_L2'],report['max_row'])


if __name__=='__main__':main()
