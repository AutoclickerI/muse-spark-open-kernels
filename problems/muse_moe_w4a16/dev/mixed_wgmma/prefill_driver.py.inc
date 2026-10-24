"""Same already-normalized x, weights and refreshed routes for every prefill producer."""


class MWLocalDGStandin:
    """Force the Hopper-only Python ABI locally; never a remote timing contender."""
    def __init__(self):self.calls=0
    def m_grouped_bf16_gemm_nt_contiguous(self,a,b,c,ends,**kw):
        assert a.dtype==b.dtype==c.dtype==torch.bfloat16
        assert a.shape[1]==b.shape[2] and c.shape[1]==b.shape[1] and ends.shape==(32,)
        assert kw['use_psum_layout']
        self.calls+=1;c.zero_()


def mw_prefill_state(w):
    dev=w['w13'].device
    st={'pre_expert':torch.ones(4096,device=dev),'ones':torch.ones(32,device=dev),
        'post_expert':(1+w['post_norm_w'].float()).bfloat16(),
        'post_expert_bf16':(1+w['post_norm_w'].float()).bfloat16(),
        'sms':torch.cuda.get_device_properties(dev).multi_processor_count,
        'gs13_gate':w['w13_gs'],'gs13_up':w['w13_gs'],'gs2':w['w2_gs'],
        'w13_gs':w['w13_gs'],'w2_gs':w['w2_gs']}
    st['w13p']=pf_repack_v6(w['w13'],w['w13_sf'],True)
    st['w2p']=pf_repack_v6(w['w2'],w['w2_sf'],False)
    raw=dict(st);raw['tiled_mod']=pf__RawModule(pf__grouped_module(),'raw17')
    for key,gated in (('w13',True),('w2',False)):
        b,s=pf__raw_repack(w[key],w[key+'_sf'],gated)
        out=torch.empty_like(b);so=torch.empty_like(s)
        pf__encode_blob_fast[(b.numel()//2048,)](b.view(torch.int32),s,out.view(torch.int32),num_warps=4)
        pf__encode_sf_fast[(triton.cdiv(s.numel(),1024),)](s,so,s.numel(),BLOCK=1024,num_warps=4)
        raw['blob13' if gated else 'blob2']=out
        raw['sfb13' if gated else 'sfb2']=so
    st['raw17']=raw
    hum=pf__Experts(w,0,32);hum.gemm_type=pf_GemmType.GROUPED_MASKED
    hum.kernels={'w13':{},'w2':{}};hum.plans={};hum.compile([512,1024,4096,16384]);st['humming']=hum
    if torch.cuda.get_device_capability(dev)==(9,0):
        import deep_gemm
        st['_dg']=deep_gemm
    else:st['_dg']=MWLocalDGStandin()
    return st


def mw_prefill_call(data,state,name):
    x=data['hidden_states'];m=x.shape[0];dev=x.device
    counts=torch.empty(32,device=dev,dtype=torch.int32)
    slots=torch.empty((32,m),device=dev,dtype=torch.int32)
    pos=torch.empty((m,8),device=dev,dtype=torch.int32)
    mw_route[(32,)](data['topk_ids'],counts,slots,pos,m,0,triton.next_power_of_2(m),num_warps=4 if m<2048 else 16)
    local=torch.where(data['topk_ids']<32,data['topk_ids'],-1).int()
    args=(x,data['topk_weights'],local,pos,counts)
    if name=='raw17':return pf__raw_experts_tiled(*args,state['raw17'])
    if name=='masked':return pf__experts_humming_masked(*args,state)
    return pf__run_canonical_dg(*args,state,'dg_w8')


def lab(deadline,weights,state=None):
    dev=next(v.device for v in weights.values() if torch.is_tensor(v))
    lines=['META prefill controls from cand_mid_sweep raw17/humming_masked/dg_w8; gather-only producer adapters, already-normalized x; public decode generator5353; n6 paired cold graphs, FP32 L2<=.01 row<=.05; local DG stand-in never timed']
    w=rg_generate_weights(5353,device=dev,router_seed=5353);w['post_norm_w']=w['post_expert_norm']
    bank=mw_add_planes(mw_prepare(w));controls=mw_prefill_state(w);marlin=marlin_prepare(w)
    flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
    sms=torch.cuda.get_device_properties(dev).multi_processor_count
    choice=(64,64,128,3,sms*2,True,True)
    for m in (512,753,3012,16384):
        if time.monotonic()>deadline-15:
            lines.append(f'BUDGET M={m}');break
        ds=[fx_make_fixture(w,m,9000+m+7919*j) for j in range(3)]
        for d in ds:
            d['topk_ids']=torch.where(d['topk_ids']<0,255,d['topk_ids'])
            d['config']=dict(d['config'],bs=m,n_experts=256,top_k=8,sharding='ep8')
        wants=[reuse_fp32(d,w) for d in ds]
        old=MWGraph(lambda d:marlin_kernel_fn(d,marlin),ds[0])
        err,row=mw_gate(old,ds,wants)
        if err>.01 or row>.05:raise ValueError(f'Marlin gate {m} {err} {row}')
        candidates=[('plane64',lambda d:mw_call(d,bank,choice))]
        candidates += [(name,lambda d,n=name:mw_prefill_call(d,controls,n)) for name in ('raw17','masked','dg_w8')]
        for name,fn in candidates:
            if time.monotonic()>deadline-10:
                lines.append(f'BUDGET M={m} {name}');break
            if name=='dg_w8' and isinstance(controls['_dg'],MWLocalDGStandin):
                g=MWGraph(fn,ds[0]);g.replay();torch.cuda.synchronize()
                lines.append(f'LOCAL_GLUE dg_w8 M={m} ABI_PASS calls={controls["_dg"].calls}');del g
                continue
            try:
                new=MWGraph(fn,ds[0]);err,row=mw_gate(new,ds,wants)
                a,b,ratio=mw_pair(old,new,ds,flush)
                lines.append(f'T {name} M={m} us={b:.1f} Marlin={a:.1f} ratio={ratio:.4f} fp32={err:.5f} row={row:.5f} '+('PASS' if err<=.01 and row<=.05 else 'REJECT'))
                del new
            except Exception as e:
                lines.append(f'ERROR {name} M={m} {type(e).__name__}: {str(e).splitlines()[0][:200]}')
        del old,wants,ds
    lines.append(f'END left_s={deadline-time.monotonic():.1f}')
    return lines
