"""Validate the decode projection port, including forced Hopper Python ABIs.

Only this development script substitutes Hopper-only kernels; the candidate does
not import it. Native tests exercise every eligible projection with fresh inputs
and graph replay, including strided pool/router output stores.
"""
import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import time
import types

import torch

ROOT = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument('--build-only', choices=('9.0a', '12.0'))
ap.add_argument('--forced-only', action='store_true')
args = ap.parse_args()
path = ROOT / 'problems/muse_decode_layer/cand_r3_proj.py'
spec = importlib.util.spec_from_file_location('decode_projection_validation', path)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
if args.build_only:
    c.IS_HOPPER = args.build_only == '9.0a'
    started = time.monotonic()
    c._dp_module()
    print(json.dumps(dict(arch=args.build_only, build='ok', seconds=time.monotonic()-started)), flush=True)
    sys.exit(0)

torch.manual_seed(8315)
device = torch.device('cuda')
native = c._dp_module()
pool = torch.randn((4096, 8192), device=device, dtype=torch.bfloat16) * .01
router = torch.randn((256, 8192), device=device) * .01
hi = router.to(torch.bfloat16)
lo = (router-hi.float()).to(torch.bfloat16)
unpool = torch.randn((8192, 4096), device=device, dtype=torch.bfloat16) * .01
s = dict(qkv_gate=torch.randn((2304, 8192), device=device, dtype=torch.bfloat16)*.01,
         o=torch.randn((8192, 1024), device=device, dtype=torch.bfloat16)*.01,
         pool_router=torch.cat((pool, hi, lo)), unpooler=unpool, unpooler_slice=unpool[:1024],
         dp_mod=native, dp_codes={}, dp_scratch={},
         dp_workspace=torch.empty(512 << 20, device=device, dtype=torch.uint8),
         dp_pool=pool, dp_pool_t=pool.T.contiguous(), dp_router_hi=hi,
         dp_router_lo=lo, dp_router_pair=torch.cat((hi, lo), 1),
         dp_mix_hi=torch.cat((pool, hi)),
         inv_freq=1/(c.ROPE_THETA**(torch.arange(0, c.HD, 2, device=device).float()/c.HD)))
names=('qkv_gate', 'o', 'pool_router', 'unpooler_slice', 'unpooler')
s['dp_transposed']={n:s[n].T.contiguous() for n in names}


def shape_state(t, name):
    w=s[name];k=w.shape[1];n=w.shape[0]
    x=torch.randn((t,k),device=device,dtype=torch.bfloat16)
    kind=torch.float32 if name=='pool_router' else torch.bfloat16
    for label,rows in ((name,n),)+((('router',256),) if name=='pool_router' else ()):
        s['dp_scratch'][(t,label)]=(torch.empty(((t+63)//64)*64*rows*17,device=device),
                                  torch.zeros(((t+63)//64)*(rows//64),device=device,dtype=torch.uint32))
    for label,weight,rows in ((name,w,n),)+((('pool_only',pool,4096),) if name=='pool_router' else ()):
        out=torch.empty((t,n),device=device,dtype=kind)[:,:rows]
        for nn in (0,1):
            wt=(s['dp_pool_t'] if label=='pool_only' else s['dp_transposed'][name]) if nn else weight
            s['dp_codes'][(t,label,nn)]=native.options(wt,x,out,nn,s['dp_workspace'])
    return x,kind


count=Counter();worst={};native_graphs=0
if not args.forced_only:
    for t in (1,8,16,32,128,256,3,24,264):
        for name in names:
            if name=='unpooler_slice' and t>8 or name=='unpooler' and t<=8:continue
            x,kind=shape_state(t,name);w=s[name]
            # Nearby tokens use the public dispatch's validated default. They
            # never enumerate the exact-token synthetic autotune registry.
            configs=list(c._dp_variants(s,name,t,w,x)) if t in c._TUNE_TOKENS else ['custom','cublas']
            for cfg in configs:
                fn=lambda:c._project_variant(x,w,name,s,cfg,kind)
                # A graph must preserve counters and overwrite zero-tail storage
                # on every replay, with different activation values each time.
                graph=c._DPGraph();graph.stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(graph.stream):fn()
                graph.stream.synchronize()
                with torch.cuda.graph(graph.graph,stream=graph.stream):out=fn()
                for seed in range(3):
                    fresh=torch.randn_like(x)
                    if seed==1:fresh[:,::257]*=32
                    if seed==2:
                        fresh.zero_();fresh[:,63]=1
                    x.copy_(fresh);graph.replay()
                    y=c._dp_canonical(out,name)
                    control=c._dp_canonical(c._project_variant(x,w,name,s,'cublas',kind),name)
                    assert c._dp_agree(y,control),(t,name,cfg,'native correctness')
                    if name=='pool_router' and cfg[0] in ('mix','pool_split'):
                        assert not bool(c._materialize(out)[:,4352:].count_nonzero()),(t,cfg,'zero tail')
                family=cfg if isinstance(cfg,str) else cfg[0]
                count[family]+=1;native_graphs+=1
            print(json.dumps(dict(native_tokens=t,name=name,variants=len(configs))),flush=True)
            # Scratch is only needed until this shape's replay completes.
            s['dp_scratch'].clear()


class PortableHopper:
    """Exact full Hopper call signature: no *args that could hide ABI bugs."""
    def __getattr__(self,name):return getattr(native,name)

    def hop(self,w,x,out,scratch,cfg,zero_tail=False):
        work,counts=scratch
        bt,bk,st,mode,split,threads=cfg
        k=x.shape[1]
        y=torch.mm(x,w[:,:k].T,out_dtype=torch.float32)
        if w.shape[1]==2*k:y+=torch.mm(x,w[:,k:].T,out_dtype=torch.float32)
        out.copy_(y)
        if zero_tail:
            out.as_strided(out.shape,out.stride(),out.storage_offset()+out.shape[1]).zero_()


# Force every paired Hopper call through its full portable ABI, including the
# two-router-product output and strided zero tail.
c.IS_HOPPER=True;s['dp_mod']=PortableHopper();forced=0
for t in c._TUNE_TOKENS:
    for name in names:
        if name=='unpooler_slice' and t>8 or name=='unpooler' and t<=8:continue
        x,kind=shape_state(t,name)
        for cfg in c._dp_variants(s,name,t,s[name],x):
            if not isinstance(cfg,tuple):continue
            if cfg[0]!='hop' and not (cfg[0]=='pool_split' and cfg[3]=='hop'):continue
            out=c._project_variant(x,s[name],name,s,cfg,kind)
            want=c._dp_canonical(c._project_variant(x,s[name],name,s,'cublas',kind),name)
            assert c._dp_agree(c._dp_canonical(out,name),want),(t,name,cfg)
            if cfg[0]=='pool_split':assert not bool(c._materialize(out)[:,4352:].count_nonzero())
            forced+=1
        s['dp_scratch'].clear()


# Force all 27 FA3 settings, both documented tuple/tensor return forms, and
# get_scheduler_metadata on every shape. One injected bad setting must be logged
# and rejected without aborting the tuner.
fake=types.ModuleType('sgl_kernel.flash_attn');seen=set();returns_tuple=False;inject=False
def metadata(**kw):
    assert kw['num_heads']==c.HQ and kw['num_heads_k']==c.HKV
    seen.add((kw['batch_size'],kw['window_size'],kw['num_splits'],kw['pack_gqa']))
    return kw
def flash(q,**kw):
    out=q.to(torch.bfloat16)
    if inject and kw['num_splits']==128 and kw['pack_gqa'] is False:
        out=out.clone();out[...,::2]=-out[...,::2]
    return (out,None) if returns_tuple else out
fake.get_scheduler_metadata=metadata;fake.flash_attn_with_kvcache=flash
sys.modules['sgl_kernel.flash_attn']=fake
c._fa3_on_device=lambda device:True
def small_synthetic(batch,window,device):
    L=8;t=batch*L
    q=torch.randn((t,c.HQ,c.HD),device=device,dtype=torch.float16).to(c.FP8)
    gate=torch.randn((t,c.Q_SIZE),device=device)
    data=dict(cache_seqlens=torch.full((batch,),16,device=device,dtype=torch.int32),
              cu_seqlens_q=torch.arange(0,t+1,L,device=device,dtype=torch.int32),
              page_table=torch.arange(batch*16,device=device,dtype=torch.int32).reshape(batch,16),
              k_cache=torch.zeros((batch*16,1,c.HKV,c.HD),device=device,dtype=c.FP8),
              v_cache=torch.zeros((batch*16,1,c.HKV,c.HD),device=device,dtype=c.FP8),
              k_scale=torch.ones((),device=device),v_scale=torch.ones((),device=device),
              config=dict(batch=batch,q_len=L,tokens=t,page_size=1,window=window,rope_theta=c.ROPE_THETA,
                          sm_scale=.125,qk_scale_factor=5.473,kv_scale=1.0))
    data['positions']=torch.arange(t,device=device);data['cache_loc']=torch.arange(t,device=device)
    return q,gate,data
c._fa3_synthetic=small_synthetic
flush=torch.zeros(128<<20,device=device,dtype=torch.uint8)
rejected=[];native_reject=c._dp_reject
def record_reject(family,key,variant,exc):
    rejected.append((family,key,variant));native_reject(family,key,variant,exc)
c._dp_reject=record_reject
for returns_tuple in (False,True):
    inject=True
    plans=c._tune_fa3(flush,device,{'_tune_deadline':time.monotonic()+60})
    assert len(plans)==6
    assert all(v!=(128,False) for v in plans.values())
    forced+=6*27
assert len(seen)==6*27
assert len([r for r in rejected if r[0]=='FA3' and r[2]==(128,False)])==12

# Tuple regression: execute the actual q_len=1 native custom attention, and
# q_len=8 Hopper FA3 dispatch with portable library output, from split partials.
c.ATTN_BACKEND='fa3';inject=False;returns_tuple=True
for L in (1,8):
    q,gate,data=small_synthetic(1,0,device);data['config']['q_len']=L
    data['config']['tokens']=L;data['config']['qk_scale_factor']=5.473
    data.update(positions=torch.arange(L,device=device),cache_loc=torch.arange(L,device=device),
                seq_lens=torch.tensor([16],device=device,dtype=torch.int32))
    for splits in (1,4):
        partial=(torch.randn((splits,L,2304),device=device),splits)
        out=c.stage_attention(partial,data,s)
        assert out.shape==(L,1024) and bool(torch.isfinite(out).all())
        forced+=1

# Bad-variant rejection and expired-budget defaults without allocating a bank.
expired={'_tune_deadline':time.monotonic()-1}
plan=c._tune_projections(expired,flush,device)
assert len(plan)==24 and all(v=='cublas' for v in plan.values())
assert c._tune_fa3(flush,device,expired)=={}
assert c._tune_pdl(expired,flush,device)=={}
assert all(v=='v9' for v in c._tune_moe(expired,device,flush).values())
old_variants=c._dp_variants;old_measure=c._measure_graph;old_tokens=c._TUNE_TOKENS
c._TUNE_TOKENS=(1,)
c._dp_module=lambda:PortableHopper()
c._dp_variants=lambda *args:iter(('cublas',('invalid_injected',)))
c._measure_graph=lambda *args,**kw:1.0
s.update(alpha1=torch.ones(8192,device=device),beta1=torch.ones(8192,device=device),
         alpha2=torch.ones(8192,device=device),beta2=torch.ones(8192,device=device),
         pre_ffn_w=torch.ones(8192,device=device),post_ffn_w=torch.ones(8192,device=device),
         next_pre_attn_w=torch.ones(8192,device=device),pre_expert_w=torch.ones(4096,device=device),
         router_bias=torch.zeros(256,device=device),_tune_deadline=time.monotonic()+60)
plan=c._tune_projections(s,flush,device)
assert len(plan)==4 and all(v=='cublas' for v in plan.values())
assert len([r for r in rejected if r[0]=='PROJ' and r[2]==('invalid_injected',)])==4
c._dp_variants=old_variants;c._measure_graph=old_measure;c._TUNE_TOKENS=old_tokens
torch.cuda.synchronize()
print(json.dumps(dict(ok=True,device=torch.cuda.get_device_name(),native_graphs=native_graphs,
                      native_families=dict(count),forced_hopper_cases=forced,
                      injected_rejections=len(rejected),budget_and_reject='passed')),flush=True)
