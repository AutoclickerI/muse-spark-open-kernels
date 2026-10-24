"""M128 real-router / R0 and fresh-route graphs against the independent FP32 oracle."""
import ast
import sys
from pathlib import Path
import json
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from adapter import prepare,work,module,CONFIGS

ROOT=HERE.parents[4]


def controls():
    # A plain namespace, without modifying any imported module.
    p=ROOT/'prof/lab/moe_pipeline_seed.py'
    tree=ast.parse(p.read_text())
    wanted=('dc_weights','dc_fixtures','dc_oracles')
    nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) or
        isinstance(n,ast.FunctionDef) and (n.name.startswith(('rg_','fx_')) or n.name in wanted) or
        isinstance(n,ast.Assign) and any(getattr(t,'id','').startswith(('rg_','fx_','CAPTURED_R0')) for t in n.targets)]
    ns={};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
    return ns


def main():
    ns=controls();torch.manual_seed(5544)
    worst=rowmax=0.;total=0;results=[]
    for seed in (5353,13272,21191):
        w=ns['dc_weights'](seed,'cuda');bank=prepare(w)
        ds=ns['dc_fixtures'](w,128,r0=seed==5353)
        if seed==5353:
            # Completion tickets must be fresh across route transitions, including
            # all remote, one pair, and all tokens sent to one expert.
            for kind in ('empty','single','hot','balanced'):
                ids=torch.full((128,8),255,device='cuda',dtype=torch.int32)
                if kind=='single':ids[3,2]=31
                if kind=='hot':ids[:,0]=0
                if kind=='balanced':ids[:,0]=torch.arange(128,device='cuda')%32
                ds.append(dict(ds[0],hidden_states=torch.randn_like(ds[0]['hidden_states']),topk_ids=ids))
        wants=ns['dc_oracles'](ds,w)
        for which,cfg in enumerate(CONFIGS):
            for grid in (264,528,1056):
                static={k:v.clone() if torch.is_tensor(v) else v for k,v in ds[0].items()}
                fs=work(static,bank,which,grid)
                for f in fs:got=f()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for f in fs:got=f()
                err=row=0.
                for d,want in zip(ds,wants):
                    for key in ('hidden_states','topk_ids','topk_weights'):static[key].copy_(d[key])
                    graph.replay();torch.cuda.synchronize();delta=got.float()-want.float()
                    e=float(delta.norm()/want.float().norm().clamp_min(1e-12));den=want.float().norm(dim=-1)
                    r=float((delta.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max())
                    assert bool(torch.isfinite(got).all()) and e<=.01 and r<=.05,(seed,which,grid,e,r)
                    err=max(err,e);row=max(row,r);total+=1
                worst=max(worst,err);rowmax=max(rowmax,row)
                results.append(dict(seed=seed,config=cfg,grid=grid,l2=err,row=row))
            print('M128_PASS',seed,which,cfg,err,row,flush=True)
    info=[(module().m128_info(i,1),module().m128_info(i,0)) for i in range(len(CONFIGS))]
    report=dict(checks=total,max_relative_L2=worst,max_row=rowmax,results=results,local_resource_info=info)
    (HERE/'agreement.json').write_text(json.dumps(report,indent=2)+'\n')
    print('M128_TOTAL',total,worst,rowmax,flush=True)


if __name__=='__main__':main()
