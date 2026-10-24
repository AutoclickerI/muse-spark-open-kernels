"""Force new Hopper glue, normalized expert consumers, and tuner rejection paths."""
import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time

import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT/'problems/muse_prefill_layer'))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


class PortableLt:
    def __init__(self):self.options=[];self.calls=[]
    def lt_options(self,w,x,y):
        assert w.ndim==x.ndim==y.ndim==2 and x.shape[1]==w.shape[1]
        assert y.shape==(x.shape[0],w.shape[0])
        self.options.append((x.shape[0],w.shape[0],w.shape[1]))
        return torch.zeros((8,128),device=x.device,dtype=torch.uint8)
    def lt_run(self,w,x,y,workspace,codes,ordinal):
        assert codes.shape==(8,128) and 0<=ordinal<8
        kwargs={'out_dtype':torch.float32} if y.dtype==torch.float32 else {}
        torch.mm(x,w.t(),out=y,**kwargs)
        self.calls.append((x.shape[0],w.shape[0],w.shape[1],ordinal))


class RouterStandin:
    def __init__(self,kernel):self.kernel=kernel;self.calls=[]
    def __getitem__(self,grid):
        def launch(x,w,bias,ls,ids,probs,se,sp,counts,*,M,K,BM,BK,SORT,num_warps,num_stages):
            assert grid==((M+BM-1)//BM,) and K==8192 and BK==64
            assert num_warps==4 and num_stages==3 and BM in (8,16,32)
            assert w.shape==(512,8192) and ids.shape==probs.shape==se.shape==sp.shape==(M,8)
            self.calls.append((M,BM,SORT))
            return self.kernel[grid](x,w,bias,ls,ids,probs,se,sp,counts,M=M,K=K,BM=BM,BK=BK,SORT=SORT,num_warps=4,num_stages=1)
        return launch


def main():
    ap=argparse.ArgumentParser();ap.add_argument('file',nargs='?',default=str(ROOT/'runs/screen/glue_prefill/work.py'))
    ap.add_argument('--faults-only',action='store_true');args=ap.parse_args()
    g=load('glue_safety',Path(args.file));dev='cuda';summary={}
    if not args.faults_only:
        ref=load('glue_ref',ROOT/'problems/muse_prefill_layer/reference.py')
        weights=ref.generate_weights(ref.WEIGHTS_SEED,device=dev)
        st=g._glue_base_prepare(weights)
        st.update(glue_choices={},glue_scratch={},_glue_router_original=weights['router'])
        # The new Hopper-only control seed is forced with a signature-only library shim.
        native=st['grouped_mod'];portable=PortableLt()
        saved_loader=g._grouped_module
        g._grouped_module=lambda:portable
        g.IS_HOPPER=True
        forced=g._glue_base_prepare(weights)
        assert forced['expert_dispatch']==g.GLUE_H200_EXPERTS
        assert forced['dense_choices']==g.GLUE_H200_DENSE
        assert forced['attention_choices']==g.GLUE_H200_ATTENTION
        summary['forced_hopper_prepare']=True
        g.IS_HOPPER=False;g._grouped_module=saved_loader
        del forced
        portable=PortableLt();st['grouped_mod']=portable
        g._glue_seed_controls(st)
        for role,m in g.GLUE_H200_DENSE:
            x=torch.randn(m,st[role].shape[1],device=dev,dtype=torch.bfloat16)
            out=g._score_dense(x,st,role)
            assert out.shape==(m,st[role].shape[0]) and bool(torch.isfinite(out).all())
        summary['lt_seed_options']=len(portable.options);summary['lt_seed_forward']=len(portable.calls)
        st['grouped_mod']=native;st['dense_choices']={}
        # Force each three-stage Hopper router branch through real one-stage BF16 math.
        standin=RouterStandin(g._glue_router_gemm);g._glue_router_gemm=standin
        g.IS_HOPPER=True
        x=torch.randn(33,8192,device=dev,dtype=torch.bfloat16);ls=torch.tensor(64,device=dev,dtype=torch.int32)
        oracle=g._glue_route_reference(x,st)
        for bm in (8,16,32):
            for sort in (False,True):
                graph=g._ScoreGraph(lambda bm=bm,sort=sort:g._glue_route(x,ls,st,('gemm',bm,sort)))
                g._glue_check_route(graph.output,oracle,ls)
                x.normal_();oracle=g._glue_route_reference(x,st);graph.replay()
                g._glue_check_route(graph.output,oracle,ls)
        summary['hopper_router_configs']=len(set(standin.calls))
        g.IS_HOPPER=False;g._glue_router_gemm=standin.kernel
        # Every runtime family the proven H200 dispatch uses consumes normalized input
        # through its actual clone. Include raw11, the portable complete-layer default.
        m=33;p=torch.randn(m,4096,device=dev,dtype=torch.bfloat16)
        pn=(p.float()*torch.rsqrt(p.float().square().mean(-1,keepdim=True)+1e-5)*st['pre_expert']).bfloat16()
        se=torch.full((m,8),-1,device=dev,dtype=torch.int32);se[:,0]=3
        sp=torch.zeros_like(se);sp[:,0]=torch.arange(m,device=dev,dtype=torch.int32)
        counts=torch.zeros(32,device=dev,dtype=torch.int32);counts[3]=m
        probs=torch.ones(m,8,device=dev)/8
        expert_checks=[]
        for name in ('humming_masked','dg_w8','raw11'):
            old=g._ScoreGraph(lambda:g._score_expert(p,probs,se,sp,counts,st,name))
            new=g._ScoreGraph(lambda:g._glue_ready_score_expert(pn,probs,se,sp,counts,st,name))
            rel=g._glue_rel(new.output,old.output,'ready '+name,.01)
            expert_checks.append((name,rel))
        summary['ready_experts']=expert_checks
        del weights,st,old,new
    # Real graph capture with portable math; inject errors at each optional phase.
    st_base={'qkv':torch.empty(1,device=dev)}
    g._glue_base_prepare=lambda w:dict(st_base)
    g._fused_flush_l2=lambda flush:None
    def fixture(st,role,m,gen):
        x=torch.ones(4,4,device=dev,dtype=torch.bfloat16)
        def run(c):return x.clone()
        def check(out):return g._glue_rel(out,x,'injected',.01)
        def refresh(i):x.fill_(1+i%3)
        return run,check,refresh
    def timing(old,new,refresh,flush,repeats,deadline):return 100.,90.,.9,repeats
    g._glue_case=fixture;g._glue_pair=timing
    phases=[]
    for phase in ('case','capture','numeric','timing','final','tuple'):
        oldcase=g._glue_case;oldpair=g._glue_pair
        if phase=='case':
            def bad(*args):raise AttributeError('missing entry point')
            g._glue_case=bad
        if phase in ('capture','numeric','tuple'):
            def patched(st,role,m,gen,phase=phase):
                run,check,refresh=fixture(st,role,m,gen)
                def altered(c):
                    if c==('stock',):return run(c)
                    if phase=='capture':raise AttributeError('trial entry point')
                    if phase=='tuple':return (run(c),)
                    return run(c)*2
                return altered,check,refresh
            g._glue_case=patched
        if phase in ('timing','final'):
            def badtime(old,new,refresh,flush,repeats,deadline,phase=phase):
                if phase=='timing' or repeats==20:raise ValueError('trial timing')
                return timing(old,new,refresh,flush,repeats,deadline)
            g._glue_pair=badtime
        st=dict(st_base,glue_choices={},glue_scratch={},glue_summary=[],glue_rows=[])
        g._glue_tune(st,{'router':torch.empty(1,device=dev)},time.monotonic())
        assert not st['glue_choices'],phase
        assert any(s.startswith('GLUE_REJECT') for s in st['glue_summary']),phase
        phases.append(phase);g._glue_case=oldcase;g._glue_pair=oldpair
    st=dict(st_base,glue_choices={},glue_scratch={},glue_summary=[],glue_rows=[])
    g._glue_tune(st,{'router':torch.empty(1,device=dev)},time.monotonic()-240)
    assert not st['glue_choices'] and all('GLUE_BUDGET' in l for l in st['glue_summary'])
    def failwrapper(*args):raise AttributeError('optional wrapper')
    g._glue_tune=failwrapper
    with contextlib.redirect_stdout(io.StringIO()):st=g.prepare({})
    assert not st['glue_choices'] and st['qkv'] is st_base['qkv']
    summary.update(rejection_phases=phases,expired_budget=True,wrapper_retains_incumbent=True,
        source_sha256=hashlib.sha256(Path(args.file).read_bytes()).hexdigest())
    print('GLUE_SAFETY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
