"""Lossless register-A WGMMA experiment; local sm_120 uses the same TMA layout and mma.sync stand-in."""
import torch
from coded import CD_CUDA, cd_encode, cd_decode

# Filled by build_wg.py from wg_kernel.cu.
WG_KERNEL = None


def wg_atoms():
    lines=[]
    for mt in (8,16,32):
        regs=mt//2
        outs=', '.join(f'%{i}' for i in range(regs))
        args=', '.join(f'%{regs+i}' for i in range(4))
        cons=', '.join(f'"+f"(d[{i}])' for i in range(regs))
        ins=', '.join(f'"r"(a[{i}])' for i in range(4))
        lines.append(f'''{'if' if mt==8 else 'else if'} constexpr(MT=={mt}) {{
  asm volatile("{{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n{mt}k16.f32.bf16.bf16 {{{outs}}}, {{{args}}}, %{regs+4}, p,1,1,0; }}" : {cons} : {ins}, "l"(desc));
}}''')
    return '\n'.join(lines)


def wg_source():
    prefix=CD_CUDA[:CD_CUDA.index('constexpr int CB =')]
    return '#include <cuda.h>\n#include <cudaTypedefs.h>\n'+prefix+WG_KERNEL.replace('WG_ATOMS',wg_atoms())+r'''
} // namespace cdg
static CUtensorMap wg_map(void*ptr,int K,int M,int stride,int MT) {
  static PFN_cuTensorMapEncodeTiled_v12000 enc=nullptr;
  if(!enc){cudaDriverEntryPointQueryResult q;auto e=cudaGetDriverEntryPoint("cuTensorMapEncodeTiled",(void**)&enc,cudaEnableDefault,&q);
    if(e!=cudaSuccess||q!=cudaDriverEntryPointSuccess||!enc)throw std::runtime_error("TMA entrypoint");}
  CUtensorMap map;cuuint64_t dims[2]={(uint64_t)K,(uint64_t)M},strides[1]={(uint64_t)stride*2};
  cuuint32_t box[2]={64,(unsigned)MT},es[2]={1,1};
  if(enc(&map,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,ptr,dims,strides,box,es,CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
       CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)!=CUDA_SUCCESS)throw std::runtime_error("TMA encode");
  return map;
}
template<int MT,int WB> static void wg_launch(int NW,const uint4*wc,const float*sc,const int*eo,const int2*es,
 const __nv_bfloat16*x,int stride,int M,float*p,int N,int K,int S,int rs,int ks,int prew,int pdl,cudaStream_t stream) {
  const int KB=K/64,nb=KB/S,bytes=nb*MT*128+((nb+3)/4)*8;
  auto fn=cdg::wg_gemv<MT,WB>;
  auto e=cudaFuncSetAttribute(fn,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
  if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
  CUtensorMap map=wg_map((void*)x,K,M,stride,MT);
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(N/(NW*16),S);cfg.blockDim=dim3(NW*32);cfg.dynamicSmemBytes=bytes;cfg.stream=stream;
  cudaLaunchAttribute at{};if(pdl){at.id=cudaLaunchAttributeProgrammaticStreamSerialization;at.val.programmaticStreamSerializationAllowed=1;cfg.attrs=&at;cfg.numAttrs=1;}
  e=cudaLaunchKernelEx(&cfg,fn,wc,sc,eo,es,x,stride,M,p,N,KB,S,rs,ks,prew,map);
  if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
void wg_run(int64_t wc,int64_t sc,int64_t eo,int64_t es,int64_t x,int64_t stride,int64_t M,int64_t p,
 int64_t N,int64_t K,int64_t S,int64_t NW,int64_t rs,int64_t ks,int64_t prew,int64_t pdl,int64_t WB,int64_t stream) {
 if(M<1||M>32||(NW!=4&&NW!=8)||N%(NW*16)||K%(S*128))throw std::runtime_error("WGMMA shape");
 int MT=M<=8?8:M<=16?16:32;
 #define WG_CASE(T,B) if(MT==T&&WB==B){wg_launch<T,B>(NW,(const uint4*)wc,(const float*)sc,(const int*)eo,(const int2*)es,(const __nv_bfloat16*)x,stride,M,(float*)p,N,K,S,rs,ks,prew,pdl,(cudaStream_t)stream);return;}
 WG_CASE(8,1) WG_CASE(16,1) WG_CASE(32,1) WG_CASE(8,2) WG_CASE(16,2) WG_CASE(32,2)
 throw std::runtime_error("WGMMA instance");
}
'''

_WG_MODULE=[]


def wg_module():
    if not _WG_MODULE:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        cap=torch.cuda.get_device_capability();arch='90a' if cap==(9,0) else f'{cap[0]}{cap[1]}'
        src=wg_source();tag=hashlib.sha256(src.encode()).hexdigest()[:12]
        cpp='#include <cstdint>\nvoid wg_run('+','.join(['int64_t']*18)+');'
        _WG_MODULE.append(load_inline(name=f'muse_coded_wg_{arch}_{tag}',cpp_sources=cpp,cuda_sources=src,functions=['wg_run'],
            extra_cuda_cflags=['-O3','-std=c++20','-lineinfo',f'-gencode=arch=compute_{arch},code=sm_{arch}'],verbose=False))
    return _WG_MODULE[0]


def wg_perm(device):
    v=torch.arange(64,device=device)
    return ((v>>2)&3)*16+((v>>4)&3)*2+((v>>1)&1)*8+(v&1)


def wg_encode(w,kmajor=False):
    N,K=w.shape
    c=cd_encode(w.reshape(N,K//64,64)[:,:,wg_perm(w.device)].reshape(N,K).contiguous(),kmajor=kmajor)
    return c


def wg_decode(c):
    w=cd_decode(c);N,K=w.shape
    return w.reshape(N,K//64,64)[:,:,wg_perm(w.device).argsort()].reshape(N,K).contiguous()


def wg_gemv(x,c,S,NW=4,WB=2,prew=1,pdl=True,out=None):
    M,K=x.shape;N=c['N']
    out=torch.empty((S,M,N),device=x.device,dtype=torch.float32) if out is None else out
    wg_module().wg_run(c['coded'].data_ptr(),c['scales'].data_ptr(),c['eoff'].data_ptr(),c['esc'].data_ptr(),
        x.data_ptr(),x.stride(0),M,out.data_ptr(),N,K,S,NW,c['rs'],c['ks'],prew,int(pdl),WB,torch.cuda.current_stream().cuda_stream)
    return out,S
