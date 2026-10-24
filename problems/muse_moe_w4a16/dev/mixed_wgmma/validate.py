"""Local arithmetic/graph checks plus an SM90a source build; no H200 timing claims."""
import argparse
import importlib.util
from pathlib import Path
import sys

import torch
import tilelang
import tilelang.language as T

from engine import MW_PRELUDE, mw_prepare, mw_work, mw_compile, mw_add_planes, mw_repack_planes

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'prof/lab'))
from build_moe_real_routing import namespace


def load_helpers():
    src=(ROOT/'problems/muse_moe_w4a16/dev/v9_reuse/reuse.py').read_text()
    src=src[src.index('def _reuse_fp32_impl'):]
    scope={'torch':torch}
    exec(src,scope)
    return scope['reuse_fp32']


def make_weights():
    gen=torch.Generator(device='cuda').manual_seed(555)
    w={}
    for key,n in (('w13',8192),('w2',4096)):
        w[key]=torch.randint(256,(32,n,2048),device='cuda',dtype=torch.uint8,generator=gen)
        w[key+'_sf']=torch.exp2(7+.4*torch.randn((32,n,256),device='cuda',generator=gen)).clamp(8,448).to(torch.float8_e4m3fn)
        w[key+'_gs']=(torch.exp(.35*torch.randn(32,device='cuda',generator=gen))*1.6e-5).float()
    w['post_norm_w']=(.1*torch.randn(4096,device='cuda',generator=gen)).bfloat16()
    return w


@T.prim_func
def conversion(Y:T.Tensor((131072,),'bfloat16')):
    with T.Kernel(512,threads=256,prelude=MW_PRELUDE) as b:
        t=T.get_thread_binding()
        i=b*256+t
        Y[i]=T.call_pure_extern('bfloat16','mw_decode',
                               T.cast(i//512,'uint32'),T.cast((i//2)%256,'uint32'),i%2)


def check_conversion():
    f=tilelang.compile(conversion)
    out=torch.empty(131072,device='cuda',dtype=torch.bfloat16);f(out)
    i=torch.arange(131072,device='cuda')
    q=i//512;s=((i//2)%256).to(torch.uint8);odd=i%2
    code=(q>>(4*odd))&15
    lut=torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device='cuda')
    sf=s.view(torch.float8_e4m3fn).float()
    v=torch.where((code&8)!=0,-lut[code&7],lut[code&7])*sf
    good=torch.isfinite(sf)
    assert torch.equal(out.view(torch.int16)[good],v.bfloat16().view(torch.int16)[good])
    print('CONVERT PASS',int(good.sum()),'finite E2M1/E4M3 pairs, bit exact, signed/subnormal/zero',flush=True)


@T.prim_func
def plane_conversion(A:T.Tensor((256,),'int32'),B:T.Tensor((256,),'int32'),Y:T.Tensor((1048576,),'bfloat16')):
    with T.Kernel(4096,threads=256,prelude=MW_PRELUDE) as b:
        t=T.get_thread_binding()
        i=b*256+t
        q=i//4096
        Y[i]=T.call_pure_extern('bfloat16','mw_plane',T.cast(A[q],'uint32'),T.cast(B[q],'uint32'),
                               T.cast((i//16)%256,'uint32'),(i//2)%8,i%2)


def check_plane_conversion():
    q=torch.arange(256,device='cuda',dtype=torch.int32).to(torch.uint8)
    pack=mw_repack_planes(q[:,None,None].expand(256,16,16).contiguous())
    a=pack[:,0,0,0,0].contiguous();b=pack[:,0,0,0,1].contiguous()
    out=torch.empty(1048576,device='cuda',dtype=torch.bfloat16)
    tilelang.compile(plane_conversion)(a,b,out)
    i=torch.arange(out.numel(),device='cuda');code=((i//4096)>>(4*(i%2)))&15
    sf=((i//16)%256).to(torch.uint8).view(torch.float8_e4m3fn).float()
    lut=torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device='cuda')
    want=(torch.where((code&8)!=0,-lut[code&7],lut[code&7])*sf/256).bfloat16()
    good=torch.isfinite(sf)
    assert torch.equal(out.view(torch.int16)[good],want.view(torch.int16)[good])
    print('PLANE CONVERT PASS',int(good.sum()),'finite slot/E2M1/E4M3 pairs, bit exact',flush=True)


def check_extremes(bank,ref,w,choice):
    m=256
    data=dict(hidden_states=torch.randn((m,4096),device='cuda').bfloat16(),
              topk_ids=torch.full((m,8),255,device='cuda',dtype=torch.int32),
              topk_weights=torch.softmax(torch.randn((m,8),device='cuda'),-1),config={'local_start':0})
    for value in (255,0):
        data['topk_ids'].fill_(255)
        if value==0:data['topk_ids'][:,0]=0
        phases=mw_work(data,bank,choice)
        for f in phases:got=f()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for f in phases:got=f()
        for j in range(3):
            data['hidden_states'].normal_()
            want=ref(data,w);graph.replay();torch.cuda.synchronize()
            torch.testing.assert_close(got,want,rtol=.03,atol=.03)
        print('EXTREME GRAPH PASS',value,choice,flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--compile90',action='store_true');ap.add_argument('--planes',action='store_true');args=ap.parse_args()
    choices=((64,16,128,0,132,True),(64,16,128,2,264,True),(64,32,128,2,264,True),(64,16,128,2,264,False))
    if args.planes:
        choices=((64,16,128,2,264,True,True),(64,32,128,2,264,True,True),
                 (64,64,128,3,264,True,True),(128,32,128,2,264,True,True))
    if args.compile90:
        for choice in choices:
            for first,n in ((True,8192),(False,4096)):
                f=mw_compile(128,n,4096,*choice[:5],first,choice[5],target='sm_90a',PLANES=args.planes)
                code=f.get_kernel_source()
                assert 'wgmma' in code,code[:1000]
                path=Path(__file__).parent/(('plane' if args.planes else '')+('rs' if choice[5] else 'ss')+str(choice[0]))
                path=path.with_name(path.name+f'_n{choice[1]}_s{choice[3]}_{n}.cu')
                path.write_text(code)
                print('SM90a PASS',choice,first,len(code),flush=True)
        return
    check_conversion()
    if args.planes:check_plane_conversion()
    w=make_weights();bank=mw_prepare(w);ref=load_helpers()
    if args.planes:mw_add_planes(bank)
    for m in (8,128,256):
        ids=torch.randint(256,(m,8),device='cuda',dtype=torch.int32)
        # Unique top-k IDs, with one hot local expert, cold experts and remote slots.
        ids=torch.rand((m,256),device='cuda').topk(8,-1).indices.int()
        ids[:,0]=0
        ids[:,1:]=torch.where(ids[:,1:]==0,255,ids[:,1:])
        data=dict(hidden_states=torch.randn((m,4096),device='cuda').bfloat16(),topk_ids=ids,
                  topk_weights=torch.softmax(torch.randn((m,8),device='cuda'),-1),config={'local_start':0})
        for choice in choices:
            phases=mw_work(data,bank,choice)
            for f in phases:got=f()
            torch.cuda.synchronize()
            want=ref(data,w)
            err=float((got.float()-want.float()).norm()/want.float().norm())
            print('EAGER',m,choice,err,flush=True)
            assert err<.01
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for f in phases:got=f()
            for j in range(3):
                data['hidden_states'].normal_()
                data['topk_weights'].copy_(torch.softmax(torch.randn_like(data['topk_weights']),-1))
                want=ref(data,w)
                graph.replay();torch.cuda.synchronize()
                err=float((got.float()-want.float()).norm()/want.float().norm())
                den=want.float().norm(dim=-1).clamp_min(want.float().norm(dim=-1).mean()*.1)
                row=float(((got.float()-want.float()).norm(dim=-1)/den).max())
                assert err<.01 and row<.05,(m,choice,j,err,row)
            print('GRAPH PASS',m,choice,'L2',err,'row',row,flush=True)
    # No-input and skew extremes exercise prefix zero intervals and empty-route clearing.
    check_extremes(bank,ref,w,choices[0])


if __name__=='__main__':main()
