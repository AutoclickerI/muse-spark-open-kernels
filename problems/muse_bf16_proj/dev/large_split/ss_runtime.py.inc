SS_LOCAL_MM=None

def ss_prepare(state):
    import time,hashlib
    start=time.monotonic();tag=hashlib.sha256(SS_CUDA.encode()).hexdigest()[:12]
    module=load_inline(name='p10_shared_shared_'+tag,cpp_sources=SS_CPP,cuda_sources=SS_CUDA,functions=['launch','resource'],extra_cuda_cflags=['-O3','-std=c++20',_nf_vendor()['_nf_arch_flag']()],extra_ldflags=['-lcuda'],no_implicit_headers=True,with_pytorch_error_handling=False)
    return dict(module=module,prepare_s=time.monotonic()-start,resources=[module.resource(i) for i in range(len(SS_CONFIGS))])

def ss_project(x,w,bank,idx,s):
    t,k=x.shape;n=w.shape[0];p=torch.empty((s,t,n),device=x.device,dtype=torch.float32)
    if SS_LOCAL_MM:p.zero_();SS_LOCAL_MM(x,w,p[0])
    else:
        rc=bank['module'].launch(idx,x.data_ptr(),w.data_ptr(),p.data_ptr(),t,n,k,s,torch.cuda.current_stream().cuda_stream)
        if rc:raise RuntimeError('SS launch '+str((t,n,k,idx,s,rc)))
    return p,s
