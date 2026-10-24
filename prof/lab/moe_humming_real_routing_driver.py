# Appended to the main lab and copied Humming variants by build_moe_humming_real_routing.py.
PROBE_MS=(8,16,32,128,256)


def lab_humming_glue(device):
    """Force all seven additional Humming launch paths on a portable zero GEMM."""
    from types import SimpleNamespace
    launch=lab_torch.zeros(4,dtype=lab_torch.int64)
    class Ops:
        def launch_kernel(self,**kw):
            assert kw['inputs'].dtype==lab_torch.bfloat16
            assert kw['outputs'].dtype==lab_torch.bfloat16
            assert kw['sorted_ids'].dtype==kw['expert_ids'].dtype==lab_torch.int32
            assert kw['top_k'] in (1,8)
            kw['outputs'].zero_()
    class Route:
        def route(self,ids,lids,*args):
            if len(args)==5:
                sort,experts,padded,start,bm=args
                assert start==0 and bm>0
                sort.zero_();experts.zero_();padded.zero_()
            else:
                assert len(args)==10,len(args)
                s0,e0,p0,s1,e1,p1,start,b0,b1,threshold=args
                assert start==0 and b0>0 and b1>=b0
                for t in (s0,e0,p0,s1,e1,p1):t.zero_()
            lids.copy_(lab_torch.where(ids<32,ids,-1))
    class Bank:
        count=32;d_expert=4096;d_hidden=4096
        def __init__(self,bucket):
            self.bucket=bucket
            self.locks=lab_torch.zeros(1024,device=device,dtype=lab_torch.int32)
        def weight(self,name):
            q=lab_torch.empty(1,device=device,dtype=lab_torch.uint8)
            return dict(weight=q,weight_scale=q,weight_scale_2=lab_torch.ones(1,device=device))
        def plan(self,m):
            p=[(8,launch),(64,launch)] if self.bucket else (16,launch)
            return m,dict(w13=p,w2=p)
    count=0
    for name,prefix,prepare_fn,kernel_fn in PROBE_HUMMING:
        globals()[prefix+'humming_ops']=Ops()
        bucket=name in ('h5','h6','h6k')
        bank=Bank(bucket)
        state=dict(glue=Route(),norm_w=lab_torch.ones(4096,device=device,dtype=lab_torch.bfloat16))
        state.update({'bank':bank} if bucket else {'banks':{(0,32):bank}})
        for m in PROBE_MS:
            d=lab_board(m,9000+m,device)
            d['config']=dict(local_start=0,local_count=32,d_hidden=4096,d_expert=4096)
            g=LabGraph(lambda q:kernel_fn(q,state),d)
            g.refresh(d);got=g.replay();assert bool((got==0).all())
            count+=1
    return [f'LOCAL_HUMMING {count} graph paths passed; native Humming arithmetic remains judge-only']


def lab_probe_datasets(rw,m,r,device):
    ds=lab_datasets(rw,m,r,device)
    if r=='R':
        import base64,zlib
        raw=zlib.decompress(base64.b64decode(PROBE_PACK_R))
        start=8*sum(n for n in PROBE_MS if PROBE_MS.index(n)<PROBE_MS.index(m))
        ids=lab_torch.tensor(list(raw[start:start+8*m]),device=device,dtype=lab_torch.int32).view(m,8)
        for d in ds:d['topk_ids']=ids.clone()
    return ds


def lab(deadline,weights):
    device=next(iter(weights.values())).device
    if lab_torch.cuda.get_device_capability(device)[0]>=12:
        return lab_local_glue(device)+lab_humming_glue(device)
    out=[f'META CAPTURED_ROUTING {PROBE_ROUTE_JOB} fixture0; R=fixed loaded IDs with three refreshed activations; B=board; raw paired graph us n=6 flush128MiB']
    rw,_=lab_router(weights,device)
    w=lab_experts(device)
    # Decode varies FP32 global scales independently per expert; constant GS hid
    # the loaded-weight Humming rejection. Preserve that stress on proj quota.
    gen=lab_torch.Generator(device=device).manual_seed(555)
    for name in ('w13','w2'):
        w[name+'_gs']=w[name+'_gs']*lab_torch.exp(lab_torch.randn(32,device=device,generator=gen))
    out.append('SCALES per_expert_log_normal seed555; canonical FP32 unchanged')
    ds={(m,r):lab_probe_datasets(rw,m,r,device) for m in PROBE_MS for r in ('R','B')}
    for (m,r),sets in ds.items():
        ids=sets[0]['topk_ids'];v=ids[ids<32];c=lab_torch.bincount(v.long(),minlength=32)
        out.append(f'ROUTE {r} {m} pairs={v.numel()} E={int((c>0).sum())} hot={int(c.max())}')
    flush=lab_torch.zeros((128<<20)//4,device=device)
    native=v9_prepare(w)
    controls={(m,r):LabGraph(lambda d:v9_kernel_fn(d,native),sets[0]) for (m,r),sets in ds.items()}
    def family(name,init,call,reference=False,phases=None):
        if lab_time.monotonic()>deadline-35:
            out.append('BUDGET '+name);return
        try:state=init()
        except Exception as exc:lab_error(out,'init:'+name,exc);return
        for (m,r),sets in ds.items():
            if lab_time.monotonic()>deadline-5:
                out.append('BUDGET '+name);break
            try:
                old=controls[m,r];new=LabGraph(lambda d:call(d,state),sets[0])
                for d in sets:
                    old.refresh(d);want=old.replay().clone();new.refresh(d);got=new.replay();lab_torch.cuda.synchronize()
                    if not bool(lab_torch.isfinite(got).all()):raise ValueError('nonfinite')
                    if not reference:
                        rel=float((got.float()-want.float()).norm()/want.float().norm().clamp_min(1e-12))
                        row=float(((got.float()-want.float()).norm(dim=-1)/want.float().norm(dim=-1).clamp_min(.1*want.float().norm(dim=-1).mean()).clamp_min(1e-12)).max())
                        if rel>.015 or row>.075:raise ValueError(f'agreement {rel:.4g}/{row:.4g}')
                us,raw,anchor=lab_pair(old,new,sets,flush)
                line=f'T {name} {m} {r} {us:.1f}'
                if phases and m in (128,256):
                    try:line+=' p='+lab_phase_time(phases(sets[0],state),flush)
                    except Exception as exc:
                        line+=' pERR='+type(exc).__name__
                        lab_error(out,f'phase:{name}/{m}/{r}',exc)
                out.append(line);del new
            except Exception as exc:lab_error(out,f'{name}/{m}/{r}',exc)
    family('v9',lambda:native,v9_kernel_fn)
    family('v11',lambda:hu__prepare_humming(w),hu__kernel_humming,phases=lab_phase_humming)
    family('humming',lambda:ho_prepare(w),ho_kernel_fn)
    family('marlin',lambda:ma_prepare(w),ma_kernel_fn)
    family('bf16ref',lambda:lab_bf16_bank(w),lab_bf16)
    if 'v9p' in PROBE_PREVIOUS:
        family('v9p',lambda:dict(native,mod=ph__module('native')),ph__kernel_v9)
    if 'v19' in PROBE_PREVIOUS:
        family('v19',lambda:dict(native,mod=v19__module('native')),v19__kernel_v9)
    if 'grouped' in PROBE_PREVIOUS:
        family('grouped',lambda:dict(native,grouped_mod=gr__grouped_module()),gr__grouped_call)
    tg_names=[n for n in PROBE_PREVIOUS if n.startswith('tg')]
    if tg_names and lab_time.monotonic()<deadline-60:
        try:
            tg_bank=tg__tg_prepare_bank(w)
            for name in tg_names:
                choice=tg_TG_CHOICES[int(name[2:])]
                family(name,lambda:tg_bank,lambda d,s,c=choice:tg__tg_run(d,s,c))
        except Exception as exc:lab_error(out,'init:prior_tg',exc)
    for name,prefix,prepare_fn,kernel_fn in PROBE_HUMMING:
        family(name,lambda f=prepare_fn:f(w),kernel_fn)
    out.append('ALIAS h3=v11: same TMA/warp-specialized indexed kernel and fused route/combine; compile shape inventory differs only')
    out.append(f'END left_s={deadline-lab_time.monotonic():.1f}')
    return out
