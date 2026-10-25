"""Capture, poison, route transitions and PHASE/DAG equality on real portable math."""
import json
from pathlib import Path
import runpy
import torch
H=Path(__file__).resolve().parent

def main():
    ns=runpy.run_path(str(H/'shared_payload.py'),init_globals={'lab_cold_l2':lambda:None})
    ns['op_build']();w=ns['dc_weights'](4242,'cuda');bank=ns['op_prepare'](w);checks=0;equal=0;worst=[0.,0.]
    for size in (127,128,255,256,257):
        base=ns['dc_fixtures'](w,size)[0];cases=[]
        for n in (0,8,9,16,17,32,33,64,65,0,size,0):
            ids=torch.full_like(base['topk_ids'],-1);ids[:n,0]=5
            # A second expert completes while the first still has many tiles.
            ids[:min(9,size),1]=7
            cases.append(dict(base,topk_ids=ids))
        cases+=ns['dc_fixtures'](w,size)
        wants=ns['dc_oracles'](cases,w)
        static={k:v.clone() if torch.is_tensor(v) else v for k,v in base.items()}
        plans=[ns['op_plan'](static,bank,a) for a in (2,3)];graphs=[]
        for p in plans:
            for f in p['steps']:f()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for f in p['steps']:f()
            graphs.append(g)
        for data,want in zip(cases,wants):
            for k in ('hidden_states','topk_ids','topk_weights'):static[k].copy_(data[k])
            outs=[]
            for p,g in zip(plans,graphs):
                h=p['owners'][0]
                for t in (h[10],h[11],h[12],p['out']):t.fill_(float('nan'))
                p['owners'][-1][2][33:].fill_(-123456)
                for rep in range(3):
                    g.replay();torch.cuda.synchronize();o=p['out']
                    assert torch.isfinite(o).all(),(size,'nonfinite')
                    e,r=ns['op_error'](o,want);assert e<=.015 and r<=.075,(size,e,r)
                    worst=[max(worst[0],e),max(worst[1],r)];checks+=1
                outs.append(o.clone())
            assert torch.equal(*outs),(size,'PHASE/DAG difference');equal+=1
        print('SIZE',size,'checks',checks,'equal',equal,'worst',worst,flush=True)
    out=dict(checks=checks,phase_dag_equal=equal,worst_L2_row=worst,sizes=[127,128,255,256,257],
             poison='ACT/Y/SS/OUT NaN and every queue state counter invalid before captured route',replays=3)
    (H/'shared_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

if __name__=='__main__':main()
