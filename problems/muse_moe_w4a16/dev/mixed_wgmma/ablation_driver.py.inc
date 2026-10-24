"""LAB ONLY: reinterpret/constant/no-MMA outputs are deliberately wrong and never returned."""


def ab_module():
    import hashlib
    from torch.utils.cpp_extension import load_inline
    return load_inline(name='mw_w13_ab_'+hashlib.sha256(AB_CUDA.encode()).hexdigest()[:12],
        cpp_sources='void ab_run(std::vector<torch::Tensor>,int64_t);int64_t ab_hostcheck(std::vector<torch::Tensor>,int64_t);',cuda_sources=AB_CUDA,functions=['ab_run','ab_hostcheck'],
        extra_include_paths=['/opt/sglang/lib/python3.12/site-packages/tilelang/src',
                             '/opt/sglang/lib/python3.12/site-packages/tilelang/3rdparty/cutlass/include'],
        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_90a,code=sm_90a',
          '-U__CUDA_NO_HALF_OPERATORS__','-U__CUDA_NO_HALF_CONVERSIONS__',
          '-U__CUDA_NO_BFLOAT16_CONVERSIONS__','-U__CUDA_NO_HALF2_OPERATORS__'],verbose=False)


def ab_capture(fn):
    fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):fn()
    return g


def lab(deadline,weights,state=None):
    mod=ab_module();dev=next(v.device for v in weights.values() if torch.is_tensor(v))
    local=torch.cuda.get_device_capability(dev)!=(9,0)
    if local:
        native=mod
        class Portable:
            def ab_run(self,args,index):
                native.ab_hostcheck(args,index)
                m,kind,grid=AB_CASES[index]
                mw_compile(m,8192,BN=32,GRID=grid)(*args)
        mod=Portable()
    w=rg_generate_weights(5353,device=dev,router_seed=5353)
    raw=zlib.decompress(base64.b64decode(CAPTURED_R0));lines=['META W13 ONLY, R0 exact IDs, cold128MiB,6 graph replays/3 fresh x. Wrong-value ablations are timing-only. B/s is nominal requested Q+SF bytes, not measured DRAM traffic.']
    flush=torch.empty(128<<20,device=dev,dtype=torch.uint8)
    for m in (128,256):
        start=8*sum((8,16,32) if m==128 else (8,16,32,128))
        ids=torch.tensor(list(raw[start:start+8*m]),device=dev,dtype=torch.int32).view(m,8)
        ids=torch.where(ids<32,ids,255)
        cnt=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32)
        order=torch.empty_like(cnt);tasks=torch.empty_like(off)
        slots=torch.empty((32,m),device=dev,dtype=torch.int32);pos=torch.empty_like(ids)
        mw_route[(32,)](ids,cnt,slots,pos,m,0,triton.next_power_of_2(m));mw_prefix[(1,)](cnt,off,order,tasks,32)
        counts=cnt.cpu().tolist();offsets=off.cpu().tolist()
        requested=sum((n+31)//32 for n in counts)*8192*4096*9//16
        unique=sum(n>0 for n in counts)*8192*4096*9//16
        xx=[torch.randn((m,4096),device=dev).bfloat16() for _ in range(3)];x=xx[0].clone()
        y=torch.empty((m*8+1024,8192),device=dev,dtype=torch.float32)
        args=[w['w13'],w['w13_sf'].view(torch.uint8),w['w13_gs'],x,cnt,off,order,tasks,slots,y]
        # Independent baseline W13 agreement includes all nonempty experts.
        valid=[];refs=[]
        lut=torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=dev)
        old_tf=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
        try:
            for e,n in enumerate(counts):
                if not n:continue
                tok=torch.where(ids==e)[0]
                q=w['w13'][e];c=torch.stack((q&15,q>>4),-1).flatten(-2).long()
                ww=lut[c&7]*torch.where((c&8)!=0,-1.,1.)*w['w13_sf'][e].float().repeat_interleave(16,-1)
                valid.extend(range(offsets[e],offsets[e]+n))
                refs.append([z[tok].float()@ww.t()*w['w13_gs'][e] for z in xx])
            wants=[torch.cat([rr[j] for rr in refs]) for j in range(3)]
        finally:torch.backends.cuda.matmul.allow_tf32=old_tf
        ix=torch.tensor(valid,device=dev)
        for i,(mm,kind,grid) in enumerate(AB_CASES):
            if mm!=m:continue
            graph=ab_capture(lambda v=i:mod.ab_run(args,v));values=[];error=0.
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            for j in range(6):
                x.copy_(xx[j%3]);flush.zero_();a.record();graph.replay();b.record();b.synchronize()
                values.append(a.elapsed_time(b)*1000.)
                if kind=='base':
                    got=y[ix];want=wants[j%3]
                    error=max(error,float((got-want).norm()/want.norm()))
            us=statistics.median(values)
            if kind=='base' and error>.01:raise ValueError(f'baseline W13 gate {m}/{grid} {error}')
            lines.append(f'{"LOCAL_ABI" if local else "AB"} M={m} kind={kind} grid={grid} us={us:.2f} nominal_GBs={requested/us/1e3:.2f} unique_GBs={unique/us/1e3:.2f} req_MB={requested/1e6:.2f} fp32={error:.6f} '+('PASS' if kind=='base' else 'TIMING_ONLY'))
    return lines
