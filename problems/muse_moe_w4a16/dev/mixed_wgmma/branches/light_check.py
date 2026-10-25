"""Captured route-class transitions with NaN scratch, including the 8/9/16/17 boundaries."""
import json
import runpy
from pathlib import Path
import torch

H=Path(__file__).resolve().parent


def main():
    m=runpy.run_path(str(H/'light_payload.py'),init_globals={'lab_cold_l2':lambda:None});m['op_build']()
    w=m['dc_weights'](4242,'cuda');bank=m['op_prepare'](w);checks=0;worst=[0.,0.]
    for size in (127,128,255,256,257):
        base=m['dc_fixtures'](w,size)[0]
        cases=[]
        for n in (0,8,9,16,17,0,size,0):
            ids=torch.full_like(base['topk_ids'],-1);ids[:n,0]=5
            cases.append(dict(base,topk_ids=ids))
        wants=m['dc_oracles'](cases,w)
        static={k:v.clone() if torch.is_tensor(v) else v for k,v in base.items()}
        plans=[m['op_plan'](static,bank,a) for a in range(4)];graphs=[]
        for p in plans:
            for f in p['steps']:f()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for f in p['steps']:f()
            graphs.append(g)
        for data,want in zip(cases,wants):
            for k in ('hidden_states','topk_ids','topk_weights'):static[k].copy_(data[k])
            for p,g in zip(plans,graphs):
                h13=p['owners'][0]
                for t in (h13[10],h13[11],h13[12],p['out']):t.fill_(float('nan'))
                g.replay();torch.cuda.synchronize();o=p['out']
                assert torch.isfinite(o).all(),(size,'nonfinite')
                e,r=m['op_error'](o,want)
                assert e<=.015 and r<=.075,(size,e,r)
                worst=[max(worst[0],e),max(worst[1],r)];checks+=1
        print('SIZE',size,'checks',checks,'worst',worst,flush=True)
        del plans,graphs
    result=dict(checks=checks,worst_L2_row=worst,sizes=[127,128,255,256,257],route_counts=[0,8,9,16,17,0,'M',0],poison='ACT/Y/SS/OUT NaN before every replay')
    (H/'light_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
