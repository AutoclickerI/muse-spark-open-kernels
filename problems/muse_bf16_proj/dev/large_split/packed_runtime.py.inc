PS_LOCAL_MM=None

def ps_prepare(state):
    import time,hashlib
    start=time.monotonic();tag=hashlib.sha256(PS_CUDA.encode()).hexdigest()[:12];mod=load_inline(name='p10_packed_shared_'+tag,cpp_sources=PS_CPP,cuda_sources=PS_CUDA,functions=['launch','resource'],extra_cuda_cflags=['-O3','-std=c++20',_nf_vendor()['_nf_arch_flag']()],extra_ldflags=['-lcuda'],no_implicit_headers=True,with_pytorch_error_handling=False);compile_s=time.monotonic()-start;banks={};pack_s={}
    for role in ('qkv_gate','o','pool_router','unpooler'):
        w=state[role];n,k=w.shape
        for bw in (64,128):
            if not any(c[1]==bw for c in PS_CONFIGS):continue
            t0=time.monotonic();banks[role,bw]=w.reshape(n//bw,bw,k//64,64).permute(2,0,1,3).contiguous();torch.cuda.synchronize();pack_s[str((role,bw))]=time.monotonic()-t0
    return dict(module=mod,banks=banks,compile_s=compile_s,pack_s=pack_s,prepare_s=time.monotonic()-start,resources=[mod.resource(i) for i in range(len(PS_CONFIGS))])

def ps_project(x,w,role,bank,idx,s):
    t,k=x.shape;n=w.shape[0];p=torch.empty((s,t,n),device=x.device,dtype=torch.float32)
    if PS_LOCAL_MM:p.zero_();PS_LOCAL_MM(x,w,p[0])
    else:
        wp=bank['banks'][role,PS_CONFIGS[idx][1]];rc=bank['module'].launch(idx,x.data_ptr(),wp.data_ptr(),p.data_ptr(),t,n,k,s,torch.cuda.current_stream().cuda_stream)
        if rc:raise RuntimeError('packed SS '+str((t,role,idx,s,rc)))
    return p,s
