"""Producer publication, norm/router math and changed-input graph replay checks."""
import argparse
import json
import time
from pathlib import Path

import torch
from kernels import ProducerGlue

HERE=Path(__file__).resolve().parent


def state(dev,gen):
    r=lambda n:torch.randn(n,device=dev,generator=gen)
    return dict(alpha1=torch.sigmoid(r(8192))+.3,alpha2=torch.sigmoid(r(8192))+.3,
        beta1=torch.sigmoid(r(8192)),beta2=torch.sigmoid(r(8192)),
        pre_ffn_w=1+.1*r(8192),next_pre_attn_w=1+.1*r(8192),post_ffn_w=1+.1*r(8192),
        pre_expert_w=1+.1*r(4096),router_bias=.05*r(256))


def reference(op,x,st,stream,start):
    # Match the producer's split accumulation; this isolates publication and
    # the epilogue from library GEMM's distinct reduced-precision path.
    p=op.p.float().sum(0)
    if op.role=='route':
        a=p[:,:4096];logits=(op.p[:,:,4096:4352]+op.p[:,:,4352:]).sum(0)
        out=(a*(a.square().mean(-1,keepdim=True)+1.e-5).rsqrt()*st['pre_expert_w']).bfloat16()
        scores=(logits*.1767766952966369).sigmoid()
        ids=(scores+st['router_bias']).argsort(dim=-1,descending=True,stable=True)[:,:8].int()
        picked=scores.gather(-1,ids.long());wts=picked/(picked.sum(-1,keepdim=True)+1.e-15)
        lids=(ids-start+256)%256;lids=torch.where(lids<32,lids,-1)
        return out,ids,wts,lids
    if op.n<8192:p=torch.nn.functional.pad(p,(0,8192-op.n))
    post=op.role=='b2'
    if post:p=p*st['post_ffn_w']
    p=p*(p.square().mean(-1,keepdim=True)+1.e-8).rsqrt()
    a,b=(st['alpha2'],st['beta2']) if post else (st['alpha1'],st['beta1'])
    sout=a*stream+b*p
    out=(sout*(sout.square().mean(-1,keepdim=True)+1.e-5).rsqrt()*
         st['next_pre_attn_w' if post else 'pre_ffn_w']).bfloat16()
    return out,sout


def check_out(got,want):
    worst=0.
    for a,b in zip(got,want):
        if a.dtype==torch.int32:
            assert torch.equal(a,b), 'routing fields differ'
        else:
            assert bool(torch.isfinite(a).all())
            e=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1.e-12))
            worst=max(worst,e)
            assert e<.0005,(a.shape,e)
    return worst


def check_meta(op,lids,start):
    flat=lids.flatten().cpu().tolist()
    lists=[[i for i,e in enumerate(flat) if e==expert] for expert in range(32)]
    count=[len(row) for row in lists];offset=[0]
    for n in count:offset.append(offset[-1]+n)
    blocks=[(n+7)//8 for n in count];u13=[0];u2=[0]
    for n in blocks:u13.append(u13[-1]+n*512);u2.append(u2[-1]+n*256)
    pairs=[i for row in lists for i in row];pos=[-1]*256
    for j,i in enumerate(pairs):pos[i]=j
    tc=[sum(e>=0 for e in flat[t*8:t*8+8]) if t<op.t else 0 for t in range(32)]
    want=[1,op.t,int(start.cpu()[0]),len(pairs)]+count+offset+blocks+u13+u2+tc+pairs+[-1]*(256-len(pairs))+pos
    assert op.meta[:711].cpu().tolist()==want,'metadata differs'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');ap.add_argument('--stats',action='store_true');ap.add_argument('--parallel',action='store_true');ap.add_argument('--fast-route',action='store_true');a=ap.parse_args()
    factory=ProducerGlue
    if a.stats or a.parallel or a.fast_route:
        if a.fast_route:from stats_fast_route import ProducerStats
        elif a.parallel:from stats_parallel import ProducerStats
        else:from stats_kernel import ProducerStats
        a.stats=True
        factory=ProducerStats
    dev='cuda';gen=torch.Generator(device=dev).manual_seed(52419);st=state(dev,gen)
    rows=[]
    for t in ((1,) if a.quick else (1,2,3,8,16,31,32) if a.parallel else (1,2,8,16,31,32)):
        for role in ('b2','route','b1'):
            n,k,s=(1024 if t<=8 else 8192,4096,8 if t<=8 else 2) if role=='b2' else ((4608,8192,6) if role=='route' else (8192,1024,2))
            w=(torch.randn((n,k),device=dev,generator=gen)/k**.5).bfloat16()
            x=torch.randn((t,k),device=dev,generator=gen).bfloat16()
            stream=torch.randn((t,8192),device=dev,generator=gen)
            start=torch.tensor([96],device=dev,dtype=torch.int32)
            for owner in ((0,) if a.quick else (0,1)):
                op=factory(w,t,s,role,owner=owner)
                fn=lambda:op(x,st,stream,start)
                got=fn();torch.cuda.synchronize()
                worst=check_out(got,reference(op,x,st,stream,start))
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):got=fn()
                for i in range(32):
                    x.normal_(generator=gen);stream.normal_(generator=gen)
                    if i%3==0:x.zero_()
                    op.p.fill_(float('nan'))
                    op.meta.fill_(-777)
                    if a.stats:
                        op.a.fill_(float('nan'));op.b.fill_(float('nan'));op.stats.fill_(float('nan'))
                    graph.replay();torch.cuda.synchronize()
                    worst=max(worst,check_out(got,reference(op,x,st,stream,start)))
                    if role=='route':check_meta(op,got[3],start)
                    assert int(op.count.cpu()[0])==int(op.gen.cpu()[0])*getattr(op,'joiners',op.ctas)
                    assert bool((op.gen==op.gen[0]).all())
                # Cross both uint32 and uint64 boundaries. Counters and CTA
                # generations use the same modulo-2^64 arithmetic.
                def seed_u64(dst,v):
                    v&=(1<<64)-1
                    dst.view(torch.int64).fill_(v if v<(1<<63) else v-(1<<64))
                for epoch in ((1<<32)-2,(1<<64)-2):
                    seed_u64(op.gen,epoch);seed_u64(op.count,epoch*getattr(op,'joiners',op.ctas))
                    if a.stats:seed_u64(op.splits,epoch*s)
                    seed_u64(op.route_count,epoch*t)
                    for i in range(4):
                        x.normal_(generator=gen);graph.replay();torch.cuda.synchronize()
                        worst=max(worst,check_out(got,reference(op,x,st,stream,start)))
                        if role=='route':check_meta(op,got[3],start)
                if role=='route':
                    bias=st['router_bias'].clone()
                    st['router_bias'].fill_(-1);st['router_bias'][:8]=1
                    x.zero_()
                    for ep_start in (96,0,96):
                        start.fill_(ep_start);graph.replay();torch.cuda.synchronize()
                        check_meta(op,got[3],start)
                    st['router_bias'].copy_(bias);start.fill_(96)
                rows.append(dict(t=t,role=role,owner=owner,replays=40,epoch_u32_crossing=True,
                                 epoch_u64_crossing=True,none_some_none_routes=role=='route',worst_l2=worst))
                print(rows[-1],flush=True)
    prefix='fast_route_' if a.fast_route else 'parallel_' if a.parallel else 'stats_' if a.stats else ''
    (HERE/(prefix+('quick_validation.json' if a.quick else 'validation.json'))).write_text(json.dumps(rows,indent=2)+'\n')


if __name__=='__main__':main()
