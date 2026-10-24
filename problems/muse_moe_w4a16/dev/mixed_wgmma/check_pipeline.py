"""Local actual async-ring/split-K math with refreshed inputs and independent GS."""
import ast
from pathlib import Path
import torch
from validate import make_weights
from pipeline import pipe_prepare,pipe_work,PIPE_CONFIGS

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def oracle():
    tree=ast.parse((ROOT/'prof/lab/moe_decode_adapters.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dc_oracles')
    ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'oracle','exec'),ns)
    return ns['dc_oracles']


def main():
    torch.manual_seed(773);w=make_weights();w['w13_gs_gate']=w['w13_gs'].clone()
    w['w13_gs_up']=w['w13_gs']*torch.exp(.55*torch.randn(32,device='cuda'))
    bank=pipe_prepare(w);ref=oracle();worst=0.;checked=0
    for m in (8,128,256):
        ds=[]
        for j in range(3):
            ids=torch.rand((m,256),device='cuda').topk(8,-1).indices.int()
            ids[:,0]=0;ids[:,1:]=torch.where(ids[:,1:]==0,255,ids[:,1:])
            ds.append(dict(hidden_states=torch.randn((m,4096),device='cuda').bfloat16(),topk_ids=ids,
                 topk_weights=torch.softmax(torch.randn((m,8),device='cuda'),-1),config={'local_start':0}))
        wants=ref(ds,w)
        for which,cfg in enumerate(PIPE_CONFIGS):
            static={k:v.clone() if torch.is_tensor(v) else v for k,v in ds[0].items()}
            fs=pipe_work(static,bank,which,1056 if cfg[3]>1 else 528)
            for f in fs:got=f()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for f in fs:got=f()
            for data,want in zip(ds,wants):
                for key in ('hidden_states','topk_ids','topk_weights'):static[key].copy_(data[key])
                graph.replay();torch.cuda.synchronize()
                err=float((got.float()-want.float()).norm()/want.float().norm().clamp_min(1e-12))
                den=want.float().norm(dim=-1);den=den.clamp_min(den.mean()*.1+1e-12)
                row=float(((got.float()-want.float()).norm(dim=-1)/den).max())
                assert bool(torch.isfinite(got).all()) and err<=.01 and row<=.05,(m,which,err,row)
                checked+=1;worst=max(worst,err)
            print('PIPE_GRAPH_PASS',m,which,cfg,err,row,flush=True)
    for key in ('w13','w2'):
        raw=torch.randint(256,w[key+'_sf'].shape,device='cuda',dtype=torch.uint8)
        raw[(raw&127)==127]=126;w[key+'_sf']=raw.view(torch.float8_e4m3fn)
    bank=pipe_prepare(w);m=128
    ds=[]
    for mode in ('empty','hot','sparse'):
        ids=torch.full((m,8),255,device='cuda',dtype=torch.int32)
        if mode=='hot':ids[:,0]=0
        if mode=='sparse':ids[3,2]=31
        ds.append(dict(hidden_states=torch.randn((m,4096),device='cuda').bfloat16(),topk_ids=ids,
               topk_weights=torch.softmax(torch.randn((m,8),device='cuda'),-1),config={'local_start':0}))
    wants=ref(ds,w)
    for which in range(len(PIPE_CONFIGS)):
        static={k:v.clone() if torch.is_tensor(v) else v for k,v in ds[0].items()}
        fs=pipe_work(static,bank,which,1056)
        for f in fs:got=f()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for f in fs:got=f()
        for data,want in zip(ds,wants):
            for key in ('hidden_states','topk_ids','topk_weights'):static[key].copy_(data[key])
            graph.replay();torch.cuda.synchronize()
            err=float((got.float()-want.float()).norm()/want.float().norm().clamp_min(1e-12))
            assert bool(torch.isfinite(got).all()) and err<=.01,(which,err)
            checked+=1;worst=max(worst,err)
        print('PIPE_WIDE_ROUTE_PASS',which,err,flush=True)
    print('PIPE_TOTAL',checked,'worstL2',worst,flush=True)


if __name__=='__main__':main()
