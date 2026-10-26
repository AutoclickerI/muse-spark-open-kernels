LS_CACHE={}
LS_LOCAL_MM=None

def ls_prepare(m,weights):
    import time,hashlib
    ns=m['_nf_vendor']();modules={};build={}
    for mode in LS_MODES:
        src=ls_source(ns,mode);t0=time.monotonic();tag=hashlib.sha256(src.encode()).hexdigest()[:12]
        modules[mode]=load_inline(name='p10_ls_'+mode+'_'+tag,cpp_sources=LS_CPP,cuda_sources=src,functions=['launch','resource'],extra_cuda_cflags=['-O3','-std=c++20',ns['_nf_arch_flag']()],extra_ldflags=['-lcuda'],no_implicit_headers=True,with_pytorch_error_handling=False);build[mode]=time.monotonic()-t0
    roles={n:weights[n] for n in ('qkv_gate','o','unpooler')};roles['pool_router']=torch.cat((weights['pooler'],weights['router'].bfloat16(),(weights['router']-weights['router'].bfloat16().float()).bfloat16()))
    packed={};packing={}
    for name,w in roles.items():
        t0=time.monotonic();packed[name]=ns['dd_decodediet_tt_mw_pack_weight'](w,64,1,1,False)['w'];torch.cuda.synchronize();packing[name]=time.monotonic()-t0
    return dict(modules=modules,roles=roles,packed=packed,build_s=build,pack_s=packing)

def ls_run(x,role,bank,mode,pad=0):
    t,k=x.shape;n=bank['roles'][role].shape[0];bn,s=LS_CONFIG[t,role];idx=0 if bn==128 else 1
    p=torch.empty((s,t,n),device='cuda',dtype=torch.float32)
    sink=torch.empty((s*(n//64)*((t+bn-1)//bn)*128,),device='cuda',dtype=torch.int32)
    if LS_LOCAL_MM:
        p.zero_();LS_LOCAL_MM(x,bank['roles'][role],p[0])
        if mode=='bf16':
            # Same physical per-split allocation/stride as native BF16 stores.
            copy=p[0].clone();y=p.view(torch.bfloat16).view(s,2*t,n)[:,:t];y.zero_();y[0].copy_(copy);return y,s
    else:
        rc=bank['modules'][mode].launch(idx,x.data_ptr(),bank['packed'][role].data_ptr(),sink.data_ptr(),p.data_ptr(),t,n,k,s,pad,torch.cuda.current_stream().cuda_stream)
        if rc:raise RuntimeError('oracle launch '+str((t,role,mode,pad,rc)))
    return (p.view(torch.bfloat16).view(s,2*t,n)[:,:t] if mode=='bf16' else p),s

@triton.jit
def _ls_warm(X,Y,N:tl.constexpr,HOT:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    if HOT:v=tl.load(X+i,i<N,0,cache_modifier='.cg').to(tl.float32)
    else:v=tl.full((B,),0,tl.float32)
    tl.store(Y+tl.program_id(0),tl.sum(v,0))
