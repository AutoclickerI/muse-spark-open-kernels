    # Compile before capture, by exact source hash. A probe uses the same native
    # K-major helper as the timed kernel; portable N64 stand-ins are local only.
    from concurrent.futures import ThreadPoolExecutor
    def native(name,ns):
        if name=='kmajor':
            arch='90a' if torch.cuda.get_device_capability()[0]==9 else ''.join(map(str,torch.cuda.get_device_capability()))
            cu=ns['ATTN_PORT_CUDA_SRC']+_BUNDLE['probe'];tag=hashlib.sha256(cu.encode()).hexdigest()[:12]
            mod=ns['load_inline'](name='p5b_kmajor_probe_'+tag+'_'+arch,
                cpp_sources='#include <pybind11/pybind11.h>\n'+ns['ATTN_PORT_CPP_SRC']+'\nvoid kmajor_probe(int64_t,int64_t,int64_t,int64_t);',cuda_sources=cu,
                functions=['attn_forward','max_cluster','occupancy','kmajor_probe'],
                no_implicit_headers=True,with_pytorch_error_handling=False,
                extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
            ns['_ATTN_PORT_EXT']=mod
            return mod
        if torch.cuda.get_device_capability()[0]==9:
            cu=ns['VQ8_CUDA'];tag=hashlib.sha256(cu.encode()).hexdigest()[:12]
            mod=ns['load_inline'](name='p5b_vq8_native_'+tag,
                cpp_sources='#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n'+ns['VQ8_CPP'],
                cuda_sources=cu,functions=['vq8_forward','vq8_resources'],
                no_implicit_headers=True,with_pytorch_error_handling=False,
                extra_cuda_cflags=['-O3','-std=c++20','-gencode=arch=compute_90a,code=sm_90a'])
            ns['_VQ8_MOD']=mod
            return mod
        return ns['_vq8_module']()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs={name:pool.submit(native,name,ns) for name,ns in spaces.items() if name not in ('A','A2')}
        for name,f in jobs.items():compiled[name]=f.result();states[name]=dict(state)
    probe_rows=[]
    mod=compiled['kmajor']
    for seed in (79137,89273,94973):
        gen=torch.Generator(device='cuda').manual_seed(seed)
        # Small integer operands make the layout/accumulator probe exact even
        # with tensor-core reduced partial-product precision.
        k=torch.randint(-2,3,(64,64),generator=gen,device='cuda').to(torch.float8_e4m3fn)
        q=torch.randint(-2,3,(8,64),generator=gen,device='cuda').to(torch.float8_e4m3fn)
        kr=torch.zeros((64,128),dtype=torch.uint8,device='cuda')
        idx=torch.arange(64,device='cuda');col=torch.arange(64,device='cuda')
        sw=((((idx[:,None]>>1)&3)<<1)^((idx[:,None]&1)<<2))
        offsets=((col[None,:]>>4)^sw)*16+(col[None,:]&15)
        kr.scatter_(1,offsets.long(),k.view(torch.uint8))
        y=torch.empty((64,8),device='cuda')
        mod.kmajor_probe(kr.data_ptr(),q.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream)
        want=k.float()@q.float().T
        error=float((y-want).abs().max());assert error==0,('K-major layout',seed,error)
        probe_rows.append([seed,error])
