"""Grouped canonical expert portfolio: no extra resident weight representation."""
# Copied into the single-file candidate by build_expert_portfolio.py.

_mod=None

def _grouped_module():
    global _mod
    if _mod is None:
        import os,hashlib
        from torch.utils.cpp_extension import load_inline
        cap=torch.cuda.get_device_capability()
        previous=os.environ.get('TORCH_CUDA_ARCH_LIST')
        os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' if cap==(9,0) else f'{cap[0]}.{cap[1]}'
        try:
            _mod=load_inline(name='prefill_canonical_portfolio_'+hashlib.sha256(GROUPED_SRC.encode()).hexdigest()[:12],
                cpp_sources='void gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);\nvoid moe_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);',
                cuda_sources=GROUPED_SRC,functions=['gemm','moe_gemm'],extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','--ptxas-options=-v'],verbose=False)
        finally:
            if previous is None:os.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:os.environ['TORCH_CUDA_ARCH_LIST']=previous
    return _mod


@triton.jit
def _portfolio_plan(COUNTS,OFFSETS,TASKS,WIDTH:tl.constexpr):
    e=tl.arange(0,32)
    c=tl.load(COUNTS+e)
    inclusive=tl.cumsum(c,0)
    tl.store(OFFSETS+e,inclusive-c)
    tl.store(OFFSETS+32,tl.sum(c,0))
    chosen=WIDTH if WIDTH else tl.where(c<=8,8,tl.where(c<=16,16,tl.where(c<=32,32,tl.where(c<=64,64,128))))
    for i in tl.static_range(6):
        n=8<<i
        tiles=tl.where(chosen==n,(c+n-1)//n,0)
        inc=tl.cumsum(tiles,0)
        tl.store(TASKS+i*33+e,inc-tiles)
        tl.store(TASKS+i*33+32,tl.sum(tiles,0))


# Width, row blocks, shared stages, persistent grid waves.
_GROUPED_CHOICES={
 'adaptive128':(0,2,2,4),
 'bn64_r1':(64,1,2,4),
 'bn64_r2':(64,2,2,4),
 'bn64_r1_g8':(64,1,2,8),
 'bn128_r1':(128,1,2,4),
 'bn128_r2':(128,2,2,4),
 'bn128_s3':(128,2,3,4),
 'bn256_s2':(256,1,2,4),
 'bn256_s3':(256,1,3,4),
}


def _run_grouped(xn,probs,slot_e,slot_pos,counts,st,name):
    width,rb,stages,waves=_GROUPED_CHOICES[name]
    T,D=xn.shape;dev=xn.device
    offsets=torch.empty(33,dtype=torch.int32,device=dev)
    tasks=torch.empty(6*33,dtype=torch.int32,device=dev)
    pairs=torch.empty(T*8,dtype=torch.int32,device=dev)
    positions=torch.empty_like(pairs)
    _portfolio_plan[(1,)](counts,offsets,tasks,WIDTH=width,num_warps=1)
    _canonical_pairs[(triton.cdiv(T*8,256),)](slot_e,slot_pos,offsets,pairs,positions,SIZE=T*8,BLOCK=256)
    inter=torch.empty((T*8,D),dtype=torch.bfloat16,device=dev)
    y=torch.empty_like(inter)
    mod=st['grouped_mod'];grid=st['sms']*waves
    mod.gemm(st['w13p'],st['gs13_gate'],st['gs13_up'],xn,inter,pairs,offsets,tasks,1,width,grid,stages,rb)
    mod.gemm(st['w2p'],st['gs2'],st['gs2'],inter,y,pairs,offsets,tasks,0,width,grid,stages,rb)
    out=torch.empty_like(xn)
    _canonical_finalize[(T,)](y,positions,probs,st['post_expert'],out,D=D,num_warps=8)
    return out


@triton.jit
def _portfolio_swiglu(H,A,STARTS,COUNTS,GATE_GS,UP_GS,D:tl.constexpr,BR:tl.constexpr):
    e=tl.program_id(0);n=tl.load(COUNTS+e)
    r=tl.program_id(1)*BR+tl.arange(0,BR)
    if tl.program_id(1)*BR>=n:return
    row=tl.load(STARTS+e)+r
    c=tl.arange(0,D)
    g=tl.load(H+row[:,None].to(tl.int64)*(2*D)+c[None,:],r[:,None]<n,0).to(tl.float32)*tl.load(GATE_GS+e)
    u=tl.load(H+row[:,None].to(tl.int64)*(2*D)+D+c[None,:],r[:,None]<n,0).to(tl.float32)*tl.load(UP_GS+e)
    g=g.to(tl.bfloat16).to(tl.float32);u=u.to(tl.bfloat16).to(tl.float32)
    tl.store(A+row[:,None].to(tl.int64)*D+c[None,:],(g*tl.sigmoid(g)*u).to(tl.bfloat16),r[:,None]<n)


def _run_canonical_dg(pooled,probs,slot_e,slot_pos,counts,st,name):
    import deep_gemm
    T,D=pooled.shape;dev=pooled.device;cap=T*8+32*128
    starts=torch.empty(32,dtype=torch.int32,device=dev)
    psum=torch.empty_like(starts)
    _offsets_kernel[(1,)](counts,starts,psum,ALIGN=128)
    a=torch.empty((cap,D),dtype=torch.bfloat16,device=dev)
    row_of=torch.empty((T,8),dtype=torch.int32,device=dev)
    _gather_norm_kernel[(T,)](pooled,st['pre_expert'],slot_e,slot_pos,starts,a,row_of,D=D,num_warps=8)
    # Sequential lifetime aliases: one weight workspace and one output workspace.
    weight_scratch=torch.empty((32,8192,4096),dtype=torch.bfloat16,device=dev)
    w2=weight_scratch.view(-1)[:32*4096*4096].view(32,4096,4096)
    h=torch.empty((cap,2*D),dtype=torch.bfloat16,device=dev)
    y=h.view(-1)[:cap*D].view(cap,D)
    kb=2048 if 'k2048' in name else 1024
    _canonical_expand_coalesced[(32,512,4096//kb)](st['w13p'],st['w13p'].view(torch.int32),weight_scratch.view(torch.int32),counts,
        ROWS=8192,GATED=True,KB=kb,num_warps=8)
    hint={} if not name.startswith('dg_hint') else {'expected_m_for_psum_layout':max(1,triton.cdiv(2*T,32))}
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a,weight_scratch,h,psum,use_psum_layout=True,**hint)
    _portfolio_swiglu[(32,triton.cdiv(T,4))](h,a,starts,counts,st['gs13_gate'],st['gs13_up'],D=D,BR=4,num_warps=4)
    _canonical_expand_coalesced[(32,256,4096//kb)](st['w2p'],st['w2p'].view(torch.int32),w2.view(torch.int32),counts,
        ROWS=4096,GATED=False,KB=kb,num_warps=8)
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a,w2,y,psum,use_psum_layout=True,**hint)
    out=torch.empty_like(pooled)
    _combine_kernel[(T,)](y,row_of,slot_e,probs,st['gs2'],st['post_expert'],out,D=D,num_warps=8)
    return out


def _expert_bucket(T):
    return 753 if T<2048 else 3012 if T<8192 else 16384


def _experts_selected(pooled,probs,slot_e,slot_pos,counts,st):
    name=st['expert_dispatch'][_expert_bucket(pooled.shape[0])]
    if name=='pipeline':
        return _experts_pipeline(pooled,probs,slot_e,slot_pos,counts,st)
    if name.startswith('dg'):
        return _run_canonical_dg(pooled,probs,slot_e,slot_pos,counts,st,name)
    xn=torch.empty_like(pooled)
    _pre_expert_norm_kernel[(pooled.shape[0],)](pooled,st['pre_expert'],xn,D=D_EXPERT,num_warps=4)
    return _run_grouped(xn,probs,slot_e,slot_pos,counts,st,name)


def _tune_portfolio(st):
    import json,math
    dev=st['w13p'].device
    flush=torch.empty(128*1024*1024,dtype=torch.uint8,device=dev)
    selected={}
    for T in (753,3012,16384):
        gen=torch.Generator(device=dev).manual_seed(15191+T)
        x=torch.randn((T,D_EXPERT),device=dev,dtype=torch.bfloat16,generator=gen)
        measurements={};errors={};references={}
        for distribution in ('flat','busy'):
            n=1 if distribution=='flat' else 2
            e=torch.full((T,8),-1,dtype=torch.int32,device=dev)
            pos=torch.zeros_like(e);probs=torch.zeros((T,8),device=dev)
            r=torch.arange(T*n,device=dev,dtype=torch.int32).reshape(T,n)
            e[:,:n]=r%32;pos[:,:n]=r//32;probs[:,:n]=1./n
            counts=(torch.full((32,),T*n//32,device=dev,dtype=torch.int32)+(torch.arange(32,device=dev)<T*n%32).int())
            for name in tuple(_GROUPED_CHOICES)+('pipeline','dg','dg_hint','dg_k2048','dg_hint_k2048'):
                st['expert_dispatch']={753:name,3012:name,16384:name}
                fn=lambda:_experts_selected(x,probs,e,pos,counts,st)
                for _ in range(2):out=fn()
                # Native stream per tuning graph avoids PyTorch cuBLAS stream-workspace reset races.
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    graph=torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):out=fn()
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    samples=[]
                    for _ in range(3):
                        flush.fill_(1);start.record();graph.replay();end.record();end.synchronize()
                        samples.append(start.elapsed_time(end)*1000)
                torch.cuda.current_stream().wait_stream(stream)
                out32=out.float()
                if name=='adaptive128':references[distribution]=out32.clone()
                rel=float((out32-references[distribution]).norm()/references[distribution].norm().clamp_min(1e-12))
                if not math.isfinite(rel) or rel>.015:raise RuntimeError(f'Expert portfolio disagrees T={T}, {name}: {rel}')
                measurements[name+':'+distribution]=sorted(samples)[1];errors[name+':'+distribution]=rel
                del graph,out,stream,out32
        costs={name:math.sqrt(measurements[name+':flat']*measurements[name+':busy']) for name in tuple(_GROUPED_CHOICES)+('pipeline','dg','dg_hint','dg_k2048','dg_hint_k2048')}
        selected[T]=min(costs,key=costs.get)
        print('PREFILL_EXPERT_PORTFOLIO',json.dumps(dict(tokens=T,selected=selected[T],us=measurements,relative_l2=errors),sort_keys=True),flush=True)
    st['expert_dispatch']=selected
