# Driver is kept separate for review; build_moe_real_routing.py appends it to the standalone payload.
import math as lab_math
import statistics as lab_stats
import time as lab_time
import torch
import torch as lab_torch

LAB_MS = (128, 256, 8, 16, 32, 512, 753)


def lab_experts(device):
    gen = lab_torch.Generator(device=device).manual_seed(5252)
    w = {}
    mean_sq = 4.0**7 * lab_math.exp(2*(lab_math.log(2)*.4)**2)
    gs = (8.6*mean_sq*4096)**-.5
    for name,n in (('w13',8192),('w2',4096)):
        w[name] = lab_torch.randint(0,256,(32,n,2048),device=device,dtype=lab_torch.uint8,generator=gen)
        w[name+'_sf'] = lab_torch.exp2(7+.4*lab_torch.randn((32,n,256),device=device,generator=gen)).clamp(8,448).to(lab_torch.float8_e4m3fn)
        w[name+'_gs'] = lab_torch.full((32,),gs,device=device)
    w['post_norm_w'] = (.1*lab_torch.randn(4096,device=device,generator=gen)).bfloat16()
    return w


def lab_router(weights,device):
    if all(k in weights for k in ('router','router_bias','pre_ffn_norm','pooler','pre_expert_norm')):
        return weights,'loaded'
    # Proj's router is iid N(0,1/sqrt(8192)), whereas decode uses .045 noise,
    # .07 mean alignment and .055 bias. Do not mislabel the proj matrix as a real router.
    gen=lab_torch.Generator(device=device).manual_seed(5353)
    mean,_=fx__basis(device)
    norm=(.1*lab_torch.randn(8192,generator=gen,device=device)).bfloat16()
    router=.045*lab_torch.randn(256,8192,generator=gen,device=device)
    router += .07*lab_torch.randn(256,1,generator=gen,device=device)*mean/lab_math.sqrt(8192)
    return dict(router=router,router_bias=.055*lab_torch.randn(256,generator=gen,device=device),
        pre_ffn_norm=norm,pooler=weights.get('pooler',lab_torch.empty(0,device=device)),
        pre_expert_norm=(.1*lab_torch.randn(4096,generator=gen,device=device)).bfloat16()),'model_seed5353'


def lab_board(m,seed,device):
    gen=lab_torch.Generator(device=device).manual_seed(seed)
    x=lab_torch.randn(m,4096,device=device,generator=gen).bfloat16()
    scores=lab_torch.rand(m,256,device=device,generator=gen)
    z=lab_torch.randn(256,device=device,generator=gen)
    u=lab_torch.rand(256,device=device,generator=gen).clamp_min(1e-12)
    routed=(scores+.03*z/(-u.log()).sqrt()).topk(8,-1)
    counts=lab_torch.bincount((routed.indices//32).flatten(),minlength=8)
    ids=((routed.indices-32*counts.argmax())%256).int()
    return dict(hidden_states=x,topk_ids=ids,topk_weights=(routed.values/routed.values.sum(-1,keepdim=True)).float(),config={})


def lab_datasets(rw,m,route,device):
    ds=[]
    for j in range(3):
        seed=9000+7919*j+m
        d=fx_make_fixture(rw,m,seed,mode='real') if route=='R' else lab_board(m,seed,device)
        # External MoE APIs index a global 256-element map, so -1 becomes remote expert 255.
        d['topk_ids']=lab_torch.where(d['topk_ids']<0,255,d['topk_ids']).int()
        d['config']=dict(bs=m,d_hidden=4096,d_expert=4096,n_experts=256,local_start=0,local_count=32,top_k=8,sharding='ep8')
        ds.append({k:d[k] for k in ('hidden_states','topk_ids','topk_weights','config')})
    return ds


class LabGraph:
    def __init__(self,fn,data):
        self.data={k:v.clone() if lab_torch.is_tensor(v) else v for k,v in data.items()}
        fn(self.data)
        lab_torch.cuda.synchronize()
        self.graph=lab_torch.cuda.CUDAGraph()
        with lab_torch.cuda.graph(self.graph):self.out=fn(self.data)
    def refresh(self,data):
        for k in ('hidden_states','topk_ids','topk_weights'):self.data[k].copy_(data[k])
    def replay(self):
        self.graph.replay();return self.out


def lab_pair(old,new,ds,flush,repeats=6):
    ot,nt,ratios=[],[],[]
    a,b=lab_torch.cuda.Event(enable_timing=True),lab_torch.cuda.Event(enable_timing=True)
    for i in range(repeats):
        old.refresh(ds[i%len(ds)]);new.refresh(ds[i%len(ds)])
        pair=[]
        for g in ((old,new) if i%2==0 else (new,old)):
            flush.sum()
            a.record();g.replay();b.record();b.synchronize()
            pair.append(float(a.elapsed_time(b))*1000)
        x,y=pair if i%2==0 else pair[::-1]
        ot.append(x);nt.append(y);ratios.append(y/x)
    # Median paired ratio, anchored by the median reference time, as in the judge.
    return lab_stats.median(ot)*lab_stats.median(ratios),lab_stats.median(nt),lab_stats.median(ot)


def lab_error(out,name,exc):
    out.append('ERR '+name+' '+type(exc).__name__+' '+str(exc).splitlines()[0][:100])


def lab_local_glue(device):
    # Force Hopper Python launch branches with portable native stand-ins. These values
    # are never reported as H200 measurements. Actual packed arithmetic is judge-only.
    def meta(shape,dtype):return lab_torch.empty(1,device=device,dtype=dtype).expand(shape)
    class Native:
        smem_optin=227328
        def smem_needed(self,*a):return 100000
        def resident_blocks(self,*a):return 1
        def run(self,*a):
            assert len(a) in (17,19,20),len(a)
            assert a[0].dtype==lab_torch.bfloat16 and a[1].dtype==lab_torch.int32
            out=a[12] if len(a)==17 else a[14]
            assert out.shape==a[0].shape
            out.copy_(a[0])
    class Tma:
        def tg_capacity(self,*a):return 1
        def tg_run(self,*a):
            assert len(a)==19,len(a)
            assert a[14]==0 and a[15] in (32,64) and a[16] in (3,5)
            a[13].copy_(a[0])
    native=Native()
    bank=dict(mod=native,w13p=meta((32,512,36864),lab_torch.uint8),w2p=meta((32,256,36864),lab_torch.uint8),
       gs13=lab_torch.ones(32,device=device),gs2=lab_torch.ones(32,device=device),pnw=lab_torch.zeros(4096,device=device,dtype=lab_torch.bfloat16),cnt=lab_torch.zeros(64,device=device,dtype=lab_torch.int32),grid=132)
    tg=dict(mod=Tma(),w13=meta((32,64,64,2,128,16),lab_torch.uint8),w13_sf=meta((32,64,64,2,32,4,2),lab_torch.uint8),
       w2=meta((32,32,64,2,128,16),lab_torch.uint8),w2_sf=meta((32,32,64,2,32,4,2),lab_torch.uint8),
       gs13=bank['gs13'],gs2=bank['gs2'],pnw=bank['pnw'],counter=lab_torch.zeros(2,device=device,dtype=lab_torch.int32),sms=132,grids={})
    class Grouped:
        def route(self,ids,counts,offs,curs,pairs,pos,tasks,start,fixed):
            assert start==0 and fixed==0
            counts.zero_();offs.zero_();curs.zero_();pairs.zero_();pos.fill_(-1);tasks.zero_()
        def gemm(self,w,gs,x,out,pairs,offs,tasks,first,fixed,grid,stages):
            assert first in (0,1) and fixed==0 and stages==2
            out.zero_()
    class Route:
        def route(self,ids,local,sort,experts,padded,start,bm):
            assert start==0
            local.copy_(lab_torch.where(ids<32,ids,-1));sort.zero_();experts.zero_();padded.zero_()
    class Ops:
        def launch_kernel(self,**kw):
            assert kw['top_k'] in (1,8) and kw['expert_ids'].dtype==lab_torch.int32
            kw['outputs'].zero_()
    class Experts:
        count=32;d_expert=4096;d_hidden=4096
        def __init__(self):
            self.expert_map=lab_torch.where(lab_torch.arange(256,device=device)<32,lab_torch.arange(256,device=device),-1).int()
            self.locks=lab_torch.zeros(1024,device=device,dtype=lab_torch.int32)
        def plan(self,m):return m,{'w13':(16,lab_torch.zeros(4,dtype=lab_torch.int64)),'w2':(16,lab_torch.zeros(4,dtype=lab_torch.int64))}
        def weight(self,name):return dict(weight=meta((1,),lab_torch.uint8),weight_scale=meta((1,),lab_torch.uint8),weight_scale_2=meta((1,),lab_torch.float32))
    wide=Experts();del wide.expert_map
    hu_state=dict(banks={(0,32):wide},glue=Route(),norm_w=lab_torch.ones(4096,device=device,dtype=lab_torch.bfloat16))
    ho_state=dict(hu_state,banks={(0,32):Experts()})
    global hu_humming_ops,ho_humming_ops
    old_hu,old_ho=hu_humming_ops,ho_humming_ops
    hu_humming_ops=ho_humming_ops=Ops()
    group=dict(bank,grouped_mod=Grouped())
    count=0
    for m in (8,16,32,128,256,512,753):
        d=lab_board(m,9000+m,device)
        d['config']=dict(local_start=0,local_count=32,d_hidden=4096,d_expert=4096)
        paths=[lambda q:v9_kernel_fn(q,bank),lambda q:ph__kernel_v9(q,bank),lambda q:v19__kernel_v9(q,bank)]
        paths += [lambda q,c=c:tg__tg_run(q,tg,c) for c in tg_TG_CHOICES]
        for fn in paths:
            got=fn(d);lab_torch.testing.assert_close(got,d['hidden_states'],rtol=0,atol=0)
            count+=1
        for fn in (lambda q:gr__grouped_call(q,group),lambda q:hu__kernel_humming(q,hu_state),lambda q:ho_kernel_fn(q,ho_state)):
            got=fn(d);assert got.shape==d['hidden_states'].shape and bool((got==0).all())
            graph=LabGraph(fn,d);graph.refresh(d);assert bool((graph.replay()==0).all())
            count+=1
        for st,original in ((hu_state,False),(ho_state,True)):
            fs=lab_phase_humming(d,st,original)
            for f in fs:
                graph=lab_torch.cuda.CUDAGraph()
                with lab_torch.cuda.graph(graph):f()
                graph.replay();count+=1
            assert bool((fs[-1]()==0).all())
    hu_humming_ops,ho_humming_ops=old_hu,old_ho
    return [f'LOCAL_GLUE {count} launch paths passed, grouped/Humming captured; H200 arithmetic unverified']


def lab_bf16_bank(w):
    # Deliberately reference-only: the deployable candidates retain their NVFP4 bank.
    lut=lab_torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=w['w13'].device)
    banks={}
    for name,n in (('w13',8192),('w2',4096)):
        dst=lab_torch.empty((32,n,4096),device=lut.device,dtype=lab_torch.bfloat16)
        for e in range(32):
            q=w[name][e]
            code=lab_torch.stack((q&15,q>>4),-1).flatten(-2).long()
            v=lut[code&7]*lab_torch.where((code&8)!=0,-1.,1.)
            dst[e]=(v*w[name+'_sf'][e].float().repeat_interleave(16,-1)*w[name+'_gs'][e]).bfloat16()
        banks[name]=dst.transpose(-1,-2)
    banks['norm']=(1+w['post_norm_w'].float()).bfloat16()
    return banks


def lab_bf16(data,bank):
    x=data['hidden_states'];m=x.shape[0]
    ids=data['topk_ids'].flatten().long()
    local=lab_torch.where(ids<32,ids,32)
    order=local.argsort(stable=True)
    counts=lab_torch.zeros(32,device=x.device,dtype=lab_torch.int32).scatter_add_(0,local.clamp_max(31),(local<32).to(lab_torch.int32))
    ends=counts.cumsum(0,dtype=lab_torch.int32)
    xp=x[order//8]
    h=lab_torch._grouped_mm(xp,bank['w13'],offs=ends)
    a=(lab_torch.nn.functional.silu(h[:,:4096].float())*h[:,4096:].float()).bfloat16()
    y=lab_torch._grouped_mm(a,bank['w2'],offs=ends)
    restore=lab_torch.empty_like(y);restore[order]=lab_torch.where((local[order]<32)[:,None],y,0)
    return hu__combine(restore.view(m,8,4096),data['topk_weights'],bank['norm'],lab_torch.where(data['topk_ids']<32,data['topk_ids'],-1))


def lab_phase_grouped(data,bank):
    # Route work is outside these phase-only graphs, and included in the end-to-end graph.
    x=data['hidden_states'];m=x.shape[0];dev=x.device;p=m*8
    counts=lab_torch.empty(32,device=dev,dtype=lab_torch.int32)
    offs=lab_torch.empty(33,device=dev,dtype=lab_torch.int32)
    curs=lab_torch.empty_like(counts);pairs=lab_torch.empty(p,device=dev,dtype=lab_torch.int32)
    pos=lab_torch.empty_like(pairs);tasks=lab_torch.empty(165,device=dev,dtype=lab_torch.int32)
    a=lab_torch.empty(p,4096,device=dev,dtype=lab_torch.bfloat16);y=lab_torch.empty_like(a)
    out=lab_torch.empty_like(x);mod=bank['grouped_mod'];grid=bank['grid']*4
    mod.route(data['topk_ids'],counts,offs,curs,pairs,pos,tasks,0,0)
    fs=[lambda:mod.gemm(bank['w13p'],bank['gs13'],x,a,pairs,offs,tasks,1,0,grid,2),
        lambda:mod.gemm(bank['w2p'],bank['gs2'],a,y,pairs,offs,tasks,0,0,grid,2),
        lambda:gr__combine_canonical[(m,)](y,pos,data['topk_weights'],bank['pnw'],out,HIDDEN=4096,num_warps=8)]
    for f in fs:f()
    return fs


def lab_phase_humming(data,st,original=False):
    # Both GEMMs and combine retain their actual library launch settings.
    bank=st['banks'][(0,32)];x=data['hidden_states'];m=x.shape[0]
    routes,gemms=bank.plan(m)
    h=lab_torch.empty(m*8,8192,device=x.device,dtype=x.dtype)
    a=lab_torch.empty(m*8,4096,device=x.device,dtype=x.dtype);y=lab_torch.empty_like(a)
    aligns={}
    local=bank.expert_map[data['topk_ids']] if original else None
    for name,(bm,launch) in gemms.items():
        if bm not in aligns:
            aligns[bm]=(local,*ho_moe_align_block_size(local,bm,32,ignore_invalid_expert=True)) if original else hu__route(data,bm,st['glue'])
    local=next(iter(aligns.values()))[0]
    ops=ho_humming_ops if original else hu_humming_ops
    silu=ho_silu_and_mul if original else hu_silu_and_mul
    def gemm(name,inp,out,topk):
        bm,launch=gemms[name];ids,sort,experts,padded=aligns[bm]
        ops.launch_kernel(configs=launch,inputs=inp,outputs=out,**bank.weight(name),locks=bank.locks,
            sorted_ids=sort,expert_ids=experts,num_tokens_padded=padded,top_k=topk,valid_shape_m=routes)
    def first():
        gemm('w13',x,h,8);silu(h,a,expert_ids=local.view(-1),expert_step=1)
    def second():gemm('w2',a,y,1)
    def combine():return hu__combine(y.view(m,8,4096),data['topk_weights'],st['norm_w'],local)
    fs=[first,second,combine]
    for f in fs:f()
    return fs


def lab_phase_time(fs,flush):
    out=[]
    for f in fs:
        g=lab_torch.cuda.CUDAGraph()
        with lab_torch.cuda.graph(g):f()
        a,b=lab_torch.cuda.Event(enable_timing=True),lab_torch.cuda.Event(enable_timing=True)
        vals=[]
        for _ in range(6):
            flush.sum();a.record();g.replay();b.record();b.synchronize();vals.append(float(a.elapsed_time(b))*1000)
        out.append(lab_stats.median(vals))
    return '/'.join(f'{x:.1f}' for x in out)


def lab(deadline,weights):
    device=next(iter(weights.values())).device
    props=lab_torch.cuda.get_device_properties(device)
    if props.major>=12:return lab_local_glue(device)
    out=[f'META {props.name} sm={props.major}{props.minor} R=decode_model B=board us=paired_graph_raw flush=128MiB n=6']
    rw,router_source=lab_router(weights,device)
    if rw['pooler'].numel()==0:
        gen=lab_torch.Generator(device=device).manual_seed(872)
        rw['pooler']=(lab_torch.randn(4096,8192,device=device,generator=gen)/lab_math.sqrt(8192)).bfloat16()
    out.append('ROUTER '+router_source+(' exact_loaded_decode_weights' if router_source=='loaded' else ' proj_router_matches_decode=False; generated model seed'))
    w=dict(weights) if 'w13' in weights else lab_experts(device)
    if 'post_norm_w' not in w:w['post_norm_w']=w['post_expert_norm']
    fixtures={}
    for m in LAB_MS:
        for r in ('R','B'):
            ds=lab_datasets(rw,m,r,device);fixtures[m,r]=ds
            ids=ds[0]['topk_ids'];local=ids[ids<32]
            counts=lab_torch.bincount(local.long(),minlength=32)
            out.append(f'ROUTE {r} {m} pairs={local.numel()} E={int((counts>0).sum())} hot={int(counts.max())}')
    # Preserve a compact real-route fixture so later lab shards can use spare proj
    # quota if compilation prevents the complete matrix from fitting in this run.
    import base64,zlib
    packed=b''.join(bytes(fixtures[m,'R'][0]['topk_ids'].byte().cpu().flatten().tolist()) for m in (8,16,32,128,256))
    out.append('PACK_R ms=8,16,32,128,256 first_fixture zlib_u8='+base64.b64encode(zlib.compress(packed,9)).decode())
    flush=lab_torch.zeros((128<<20)//4,device=device)
    states={};baselines={};rows={}
    def run_family(name,init,call,phases=None):
        if lab_time.monotonic()>deadline-12:
            out.append('BUDGET '+name);return
        try:st=init();states[name]=st
        except Exception as exc:lab_error(out,'init:'+name,exc);return
        for m in LAB_MS:
            for r in ('R','B'):
                if lab_time.monotonic()>deadline-5:
                    out.append('BUDGET '+name);return
                ds=fixtures[m,r]
                try:
                    g=LabGraph(lambda d:call(d,st),ds[0])
                    if (m,r) not in baselines:
                        baselines[m,r]=LabGraph(lambda d:v9_kernel_fn(d,states['v9']),ds[0])
                    ref=baselines[m,r]
                    max_rel=0.
                    for d in ds:
                        ref.refresh(d);want=ref.replay().clone()
                        g.refresh(d);got=g.replay()
                        lab_torch.cuda.synchronize()
                        rel=float((got.float()-want.float()).norm()/want.float().norm().clamp_min(1e-12))
                        row=((got.float()-want.float()).norm(dim=-1)/want.float().norm(dim=-1).clamp_min(.1*want.float().norm(dim=-1).mean()).clamp_min(1e-12)).max()
                        if not bool(lab_torch.isfinite(got).all()) or rel>.015 or float(row)>.075:
                            raise ValueError(f'agreement rel={rel:.4g} row={float(row):.4g}')
                        max_rel=max(max_rel,rel)
                    paired,median,anchor=lab_pair(ref,g,ds,flush)
                    # One compact line per kernel/M/routing; suffix p= is W13+SwiGLU/W2/combine.
                    line=f'T {name} {m} {r} {paired:.1f}'
                    if phases and m in (128,256) and lab_time.monotonic()<deadline-12:
                        try:line+=' p='+lab_phase_time(phases(ds[0],st),flush)
                        except Exception as exc:line+=' pERR='+type(exc).__name__
                    out.append(line);rows[name,m,r]=paired
                    del g
                except Exception as exc:lab_error(out,f'{name}/{m}/{r}',exc)
    run_family('v9',lambda:v9_prepare(w),v9_kernel_fn)
    run_family('grouped',lambda:dict(states['v9'],grouped_mod=gr__grouped_module()),gr__grouped_call,lab_phase_grouped)
    tg=None
    if lab_time.monotonic()<deadline-15:
        try:tg=tg__tg_prepare_bank(w)
        except Exception as exc:lab_error(out,'init:tg',exc)
    if tg is not None:
        for i,c in enumerate(tg_TG_CHOICES):
            run_family('tg'+str(i),lambda:tg,lambda d,s,c=c:tg__tg_run(d,s,c))
    run_family('v9p',lambda:dict(states['v9'],mod=ph__module('native')),ph__kernel_v9)
    run_family('v19',lambda:dict(states['v9'],mod=v19__module('native')),v19__kernel_v9)
    run_family('v11',lambda:hu__prepare_humming(w),hu__kernel_humming,lab_phase_humming)
    run_family('humming',lambda:ho_prepare(w),ho_kernel_fn,lambda d,s:lab_phase_humming(d,s,True))
    run_family('marlin',lambda:ma_prepare(w),ma_kernel_fn)
    run_family('bf16ref',lambda:lab_bf16_bank(w),lab_bf16)
    for m in LAB_MS:
        for r in ('R','B'):
            ranked=sorted((us,k) for (k,mm,rr),us in rows.items() if mm==m and rr==r and k!='bf16ref')
            if ranked:out.append(f'WIN {m} {r} '+','.join(f'{k}:{us:.1f}' for us,k in ranked[:1]))
    # Known API constraints from the image's own implementations. No FP4/FP8 activation
    # timings are mixed into the BF16 admissible ranking.
    import importlib.util
    import sgl_kernel
    import flashinfer
    out.append('LIB vllm='+str(importlib.util.find_spec('vllm') is not None)+' sgl_machete='+str(any('machete' in x for x in dir(sgl_kernel)))+' marlin=measured')
    out.append('LIB flashinfer grouped_mm_fp4=FP4_activations; mm_bf16_fp4=SM100+; cutlass W4A16 mixed-MXFP4 scale support needs separate adapter')
    out.append(f'END left_s={deadline-lab_time.monotonic():.1f} rows={len(rows)} tg_choices={tg_TG_CHOICES}')
    return out
