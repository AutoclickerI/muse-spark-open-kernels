import triton
import triton.language as tl


@triton.jit
def gd_start(USE:tl.constexpr):
    if USE:tl.extra.cuda.gdc_launch_dependents()


@triton.jit
def gd_wait(USE:tl.constexpr):
    if USE:tl.extra.cuda.gdc_wait()


@triton.jit
def gd_prefetch(ptrs):
    return tl.inline_asm_elementwise('prefetch.global.L2 [$1]; mov.u32 $0,0;',
        '=r,l',[ptrs],dtype=tl.int32,is_pure=False,pack=1)


@triton.jit
def gd_read():
    return tl.inline_asm_elementwise('mov.u64 $0, %clock64; mov.u64 $1, %globaltimer; mov.u32 $2, %smid;',
        '=l,=l,=r,~{memory}',[],dtype=(tl.int64,tl.int64,tl.int32),is_pure=False,pack=1)


@triton.jit
def gd_save(STAMPS,c0,g0,c1,g1,c2,g2,c3,g3,c4,g4,c5,g5,c6,g6,sm):
    pid=tl.program_id(0)+tl.num_programs(0)*(tl.program_id(1)+tl.num_programs(1)*tl.program_id(2))
    tl.store(STAMPS+pid*16+0,c0);tl.store(STAMPS+pid*16+1,g0)
    tl.store(STAMPS+pid*16+2,c1);tl.store(STAMPS+pid*16+3,g1)
    tl.store(STAMPS+pid*16+4,c2);tl.store(STAMPS+pid*16+5,g2)
    tl.store(STAMPS+pid*16+6,c3);tl.store(STAMPS+pid*16+7,g3)
    tl.store(STAMPS+pid*16+8,c4);tl.store(STAMPS+pid*16+9,g4)
    tl.store(STAMPS+pid*16+10,c5);tl.store(STAMPS+pid*16+11,g5)
    tl.store(STAMPS+pid*16+12,c6);tl.store(STAMPS+pid*16+13,g6)
    tl.store(STAMPS+pid*16+14,sm)


def gd_project(m,x,w,stamps):
    t,k=x.shape;n=w.shape[0];bn,ks,bk,pf=m._GEMV_CHOICES[t,n,k]
    out=torch.empty((k//ks,t,n),device=x.device,dtype=torch.float32)
    gd_gemv_kernel[(n//bn,k//ks)](stamps,x,w,out,t,x.stride(0),N=n,K=k,KS=ks,BN=bn,BK=bk,
        TB=max(16,triton.next_power_of_2(t)),PREFETCH=pf,USE_PDL=True,launch_pdl=True,num_warps=4,num_stages=4)
    return out,k//ks


def gd_consumer(m,value,role,data,state,plan,stamps=None):
    p,s=value;t=p.shape[1];device=p.device
    if role=='qkv_gate':
        if stamps is None:return m.qk_prep(value,data,state,q_fp8=True)
        q=torch.empty((t,16,64),device=device,dtype=torch.float8_e4m3fn)
        gate=torch.empty((t,1024),device=device,dtype=torch.float32)
        gd_qk_prep_kernel[(t,)](stamps,p,s,p.stride(0),data['positions'],data['cache_loc'],data['k_cache'],data['v_cache'],
            data['k_scale'],data['v_scale'],state['inv_freq'],q,gate,5.4730064863875,1.e-5,
            HQ=16,HKV=2,D=64,ROPE=False,Q_FP8=True,USE_PDL=True,launch_pdl=True,num_warps=4)
        return q,gate
    if role=='pool_router':
        if stamps is None:return m._best7_route(value,state,data,plan,gf_sync(t,device))
        warps,reps=plan['geometry'].get('route',(8,1))
        x=torch.empty((t,4096),device=device,dtype=torch.bfloat16)
        ids=torch.empty((t,8),device=device,dtype=torch.int32);lids=torch.empty_like(ids)
        wts=torch.empty((t,8),device=device,dtype=torch.float32)
        gd_route_fast[(t,reps)](stamps,p,s,p.stride(0),p.stride(1),state['router_bias'],state['pre_expert_w'],
            data['local_start'],x,ids,wts,lids,REPLICAS=reps,USE_PDL=True,launch_pdl=True,num_warps=warps)
        return x,ids,wts,lids
    phase='b1' if role=='o' else 'b2'
    if stamps is None:return m._best7_boundary(value,data['stream'],state,plan,phase,gf_sync(t,device))
    post=phase=='b2';warps,reps=plan['geometry'].get(phase,(8,1))
    out=torch.empty((t,8192),device=device,dtype=torch.bfloat16);stream=torch.empty_like(data['stream'])
    a=state['alpha2' if post else 'alpha1'];b=state['beta2' if post else 'beta1']
    pw=state['next_pre_attn_w' if post else 'pre_ffn_w']
    gd_boundary_fast[(t,reps)](stamps,p,s,p.stride(0),p.stride(1),data['stream'],stream,out,
        state['post_ffn_w'] if post else a,a,b,pw,NB=p.shape[2],HAS_POSTW=post,
        REPLICAS=reps,USE_PDL=True,launch_pdl=True,num_warps=warps)
    return out,stream


def gd_summary(stamps,anchor=None):
    # Only differences on the SAME CTA use clock64. Cross-SM dispatch and tail
    # use globaltimer. Instrumented figures are diagnostic, never speedups.
    a=stamps.detach().cpu().double();end=a[:,13];start=a[:,1]
    valid=(start>0)&(end>=start)
    if not bool(valid.all()):raise RuntimeError('incomplete CTA timestamp coverage')
    cycles=lambda i,j:float((a[:,2*j]-a[:,2*i]).median())
    out=dict(ctas=len(a),entry_first_cycles=cycles(0,1),wait_cycles=cycles(2,3),
        main_cycles=cycles(3,4),reduce_cycles=cycles(4,5),consumer_load_cycles=cycles(1,4),
        active_us=float(end.max()-start.min())/1000.,
        tail_us=float(end.max()-end.quantile(.95))/1000.,
        last_wave_us=float(end.max()-start.max())/1000.,
        start_spread_us=float(start.max()-start.min())/1000.,
        clocks_per_us=float(((a[:,12]-a[:,0])/(end-start).clamp_min(1)*1000.).median()))
    if anchor is not None:out['dispatch_first_us']=float(a[:,3].min()-anchor[1].item())/1000.
    return out
