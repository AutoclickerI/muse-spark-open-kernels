"""FP32 checks for the actual register layout, raw-scale bank and async ring on local SM120."""
import torch
from validate import make_weights,load_helpers
from vector import vec_prepare,vec_work


def main():
    w=make_weights();bank=vec_prepare(w);ref=load_helpers()
    for m in (8,128,256):
        ids=torch.rand((m,256),device='cuda').topk(8,-1).indices.int();ids[:,0]=0
        ids[:,1:]=torch.where(ids[:,1:]==0,255,ids[:,1:])
        d=dict(hidden_states=torch.randn((m,4096),device='cuda').bfloat16(),topk_ids=ids,
               topk_weights=torch.softmax(torch.randn((m,8),device='cuda'),-1),config={'local_start':0})
        for nt in (16,32,64):
            phases=vec_work(d,bank,nt)
            for f in phases:got=f()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for f in phases:got=f()
            for j in range(3):
                d['hidden_states'].normal_();want=ref(d,w);graph.replay();torch.cuda.synchronize()
                err=float((got.float()-want.float()).norm()/want.float().norm().clamp_min(1e-12))
                row=(got.float()-want.float()).norm(dim=-1)/want.float().norm(dim=-1).clamp_min(want.float().norm(dim=-1).mean()*.1)
                print('VEC',m,nt,j,err,float(row.max()),flush=True)
                assert err<.01 and float(row.max())<.05
    # Exercise the raw-byte format lookup on signed, zero and subnormal scales,
    # plus zero-local-work and single-expert hot routing. No prepared dequant bank.
    for name in ('w13','w2'):
        raw=torch.randint(256,w[name+'_sf'].shape,device='cuda',dtype=torch.uint8)
        raw[(raw&127)==127]=126;w[name+'_sf']=raw.view(torch.float8_e4m3fn)
    bank=vec_prepare(w);m=128
    for hot in (False,True):
        ids=torch.full((m,8),255,device='cuda',dtype=torch.int32)
        if hot:ids[:,0]=0
        d=dict(hidden_states=torch.randn((m,4096),device='cuda').bfloat16(),topk_ids=ids,
               topk_weights=torch.softmax(torch.randn((m,8),device='cuda'),-1),config={'local_start':0})
        for nt in (16,32,64):
            fs=vec_work(d,bank,nt)
            for f in fs:got=f()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for f in fs:got=f()
            for j in range(3):
                d['hidden_states'].normal_();want=ref(d,w);graph.replay();torch.cuda.synchronize()
                err=float((got.float()-want.float()).norm()/want.float().norm().clamp_min(1e-12))
                assert err<.01,err
                print('VEC_WIDE_EXTREME',hot,nt,j,err,flush=True)


if __name__=='__main__':main()
