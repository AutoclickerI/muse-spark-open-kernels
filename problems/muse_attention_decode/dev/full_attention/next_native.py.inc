    # One native proof module also supplies V/V2's actual timed executable.
    # On sm120 the explicitly installed attention launch stand-in remains;
    # the isolated denominator/rescale probe executes its real portable form.
    v=spaces['V'];arch='90a' if torch.cuda.get_device_capability()[0]==9 else ''.join(map(str,torch.cuda.get_device_capability()))
    cu=v['VQ8_CUDA']+_BUNDLE['probe'];tag=hashlib.sha256(cu.encode()).hexdigest()[:12]
    mod=v['load_inline'](name='pvsum_lab_'+tag+'_'+arch,cpp_sources=v['VQ8_CPP']+'\nvoid pv_den_probe(int64_t,int64_t,int64_t,int64_t);',
        cuda_sources=cu,functions=['vq8_forward','vq8_resources','pv_den_probe'],
        extra_cuda_cflags=['-O3','-std=c++20',f'-gencode=arch=compute_{arch},code=sm_{arch}'])
    for name in ('V','V2'):spaces[name]['_VQ8_MOD']=mod
    probe_rows=[]
    for seed in (68137,77051,94973):
        gen=torch.Generator(device='cuda').manual_seed(seed)
        p=(torch.rand((2,64,128),device='cuda',generator=gen)*448).to(torch.float8_e4m3fn)
        alpha=torch.where(torch.arange(64,device='cuda')%3==0,.5,1.).float();y=torch.empty((128,4),device='cuda')
        mod.pv_den_probe(p.data_ptr(),alpha.data_ptr(),y.data_ptr(),torch.cuda.current_stream().cuda_stream)
        lane=torch.arange(128,device='cuda')%32;w=torch.arange(128,device='cuda')//32;row=16*w+lane//4
        ref=p[0].float().double().sum(-1)*alpha.double()+p[1].float().double().sum(-1)
        want=torch.stack((ref[row],ref[row],ref[row+8],ref[row+8]),-1)
        error=float(((y.double()-want).abs()/want.abs().clamp_min(1e-12)).max());assert error<1e-6,('native denominator',seed,error)
        probe_rows.append([seed,error])
    for name,ns in spaces.items():
        if name in ('A','A2'):continue
        states[name]=dict(state);compiled[name]=True
