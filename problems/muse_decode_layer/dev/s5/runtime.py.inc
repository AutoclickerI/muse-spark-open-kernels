@functools.lru_cache(None)
def pm_module():
    arch='90a' if torch.cuda.get_device_capability()==(9,0) else '120'
    mod=load_inline(name='s5_hh_'+hashlib.sha256(S5_HH_CUDA.encode()).hexdigest()[:12],
        cpp_sources=S5_HH_CPP,cuda_sources=S5_HH_CUDA,functions=S5_HH_FUNCTIONS,
        no_implicit_headers=True,with_pytorch_error_handling=False,
        extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','-gencode=arch=compute_'+arch+',code=sm_'+arch],verbose=False)
    for prefix in ('idist','pm_nt'):ISSUE_OPTS[prefix]=getattr(mod,prefix+'_init')()
    return mod
