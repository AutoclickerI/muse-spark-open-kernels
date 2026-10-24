"""Fresh route transitions exercise all completion counters and the Hopper B layout."""
import ast
from pathlib import Path
import json
import torch
from validate import make_weights
from fused import ft_prepare,ft_work,FT_CONFIGS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def main():
    torch.manual_seed(5544);w=make_weights()
    w['w13_gs_gate']=w['w13_gs'].clone();w['w13_gs_up']=w['w13_gs']*torch.exp(.55*torch.randn(32,device='cuda'))
    bank=ft_prepare(w);t=ast.parse((ROOT/'prof/lab/moe_pipeline_seed.py').read_text())
    fn=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='dc_oracles')
    ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'oracle','exec'),ns);ref=ns['dc_oracles']
    m=256;ds=[]
    for mode in ('balanced','random','hot','empty','single','random','empty','balanced'):
        ids=torch.full((m,8),255,device='cuda',dtype=torch.int32)
        if mode=='balanced':ids[:,0]=torch.arange(m,device='cuda')%32
        if mode=='hot':ids[:,0]=0
        if mode=='single':ids[7,2]=31
        if mode=='random':ids=torch.rand((m,256),device='cuda').topk(8,-1).indices.int()
        ds.append(dict(hidden_states=torch.randn((m,4096),device='cuda').bfloat16(),topk_ids=ids,
            topk_weights=torch.softmax(torch.randn((m,8),device='cuda'),-1),config={'local_start':0}))
    wants=ref(ds,w);worst=0.;rowmax=0.;num=0
    for which in range(len(FT_CONFIGS)):
        for grid in (264,528,1056):
            static={k:v.clone() if torch.is_tensor(v) else v for k,v in ds[0].items()}
            fs=ft_work(static,bank,which,grid)
            for f in fs:got=f()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for f in fs:got=f()
            for d,want in zip(ds,wants):
                for key in ('hidden_states','topk_ids','topk_weights'):static[key].copy_(d[key])
                graph.replay();torch.cuda.synchronize();delta=got.float()-want.float()
                err=float(delta.norm()/want.float().norm().clamp_min(1e-12));den=want.float().norm(dim=-1)
                row=float((delta.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max())
                assert bool(torch.isfinite(got).all()) and err<=.01 and row<=.05,(which,grid,num,err,row)
                worst=max(worst,err);rowmax=max(rowmax,row);num+=1
            print('FUSED_GRAPH_PASS',which,grid,worst,rowmax,flush=True)
    # Global expert offsets and finite non-normal scale bytes must also work.
    for key in ('w13','w2'):
        raw=torch.randint(256,w[key+'_sf'].shape,device='cuda',dtype=torch.uint8);raw[(raw&127)==127]=126
        w[key+'_sf']=raw.view(torch.float8_e4m3fn)
    bank=ft_prepare(w);wide=ds[:3];wants=ref(wide,w)
    for which in (0,2,3,5,6,7):
        for d,want in zip(wide,wants):
            dd=dict(d,topk_ids=d['topk_ids']+64,config={'local_start':64})
            for f in ft_work(dd,bank,which,528):got=f()
            torch.cuda.synchronize();err=float((got.float()-want.float()).norm()/want.float().norm())
            assert bool(torch.isfinite(got).all()) and err<=.01,(which,err)
            worst=max(worst,err);num+=1
        print('FUSED_WIDE_PASS',which,worst,flush=True)
    (HERE/'diagnostics/fused_agreement.json').write_text(json.dumps({'checks':num,'max_relative_L2':worst,'max_row':rowmax,'graph_route_transitions':192,'independent_gate_up_GS':True},indent=2)+'\n')
    print('FUSED_TOTAL',num,worst,rowmax,flush=True)


if __name__=='__main__':main()
