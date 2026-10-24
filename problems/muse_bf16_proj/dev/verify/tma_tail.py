_vp_tma_mods={}


def vp_tma_module(arch=None):
    from torch.utils.cpp_extension import load_inline
    target=arch or ('9.0a' if torch.cuda.get_device_capability()[0]==9 else '12.0')
    tag=hashlib.sha256(VP_TMA_SOURCE.encode()).hexdigest()[:12]+target.replace('.','_')
    if tag not in _vp_tma_mods:
        previous=os.environ.get('TORCH_CUDA_ARCH_LIST');os.environ['TORCH_CUDA_ARCH_LIST']=target
        try:
            _vp_tma_mods[tag]=load_inline(name='muse_verify_tma_'+tag,
                cpp_sources='''#include <torch/extension.h>
void wg_gemm(torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,bool);''',
                cuda_sources=VP_TMA_SOURCE,functions=['wg_gemm'],
                extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','--split-compile=16'],verbose=False)
        finally:
            if previous is None:os.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:os.environ['TORCH_CUDA_ARCH_LIST']=previous
    return _vp_tma_mods[tag]


def vp_tma(x,w,choice,out_dtype=torch.bfloat16,pdl=False,mod=None):
    bt,nsub,ncw,st,mc,ks,waves,pipe=choice
    out=torch.empty((x.shape[0],w.shape[0]),device=x.device,dtype=out_dtype)
    native=vp_tma_module() if mod is None else mod
    native.wg_gemm(w,x,out,w.shape[0],bt,nsub,ncw,st,ks,mc,waves,pipe,pdl)
    return out


def vp_tma_choices(m,n,k):
    for bt,nsub,ncw,st,mc in VP_TMA_INSTANCES:
        if bt>m*2:continue
        br=64*nsub*ncw
        if (n//br)%mc or n%br:continue
        # Match the logical work to132SMs; measure the neighbouring K partitions.
        tiles=(n//br)*((m+bt-1)//bt)
        splits=sorted(set((1,2,4, min(8,max(1,2**max(0,(132//max(1,tiles)).bit_length()-1))))))
        for ks in splits:
            if ks*mc>8 or k//64<ks:continue
            for waves in (0,1,2):
                yield (bt,nsub,ncw,st,mc,ks,waves,1)
