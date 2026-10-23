"""Lossless mixed-output pooler/router TMA+WGMMA; reuse the projection lane's pipeline."""
from pathlib import Path
import ast
P=Path(__file__).resolve().parents[1]
s=Path('problems/muse_bf16_proj/cand_portfolio.py').read_text();tree=ast.parse(s)
values={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('HOP_HELPERS','HOP_MAP','HOP_KERNEL')}
n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_wgmma_fn');ns={};exec('\n'.join(s.splitlines()[n.lineno-1:n.end_lineno]),ns)
helpers=values['HOP_HELPERS'].replace('namespace muse_wg','namespace prefill_mixed').replace('WGMMA_FNS',''.join(ns['_wgmma_fn'](n) for n in (64,128,256)))
# Copy only the actual producer/consumer pipeline; mixed-output epilogue is new.
k=values['HOP_KERNEL'];a=k.index('    if (tid == 0) {');b=k.index('    auto store =',a)
pipeline=k[a:b]
mma=' else '.join(f'if constexpr(BT=={n}) wgmma_m64n{n}(acc,da+2*(kk%4),db+2*(kk%4),1);' for n in (64,128,256))
pipeline=pipeline.replace('HOP_MMA',mma)
source='''#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <stdint.h>
'''+helpers+'''
template<int BT,int BK,int ST>
__global__ void __launch_bounds__(256) mixed_kernel(
 const __grid_constant__ CUtensorMap tmW,const __grid_constant__ CUtensorMap tmX,
 __nv_bfloat16* __restrict__ Pool,float* __restrict__ Logits,int M){
#ifdef MP_TMA
 constexpr int BR=64,AB=BR*64*2,CB=(BR+BT)*64*2,SB=CB*(BK/64);
 extern __shared__ __align__(1024) unsigned char smem[];
 uint64_t* full=reinterpret_cast<uint64_t*>(smem+ST*SB);uint64_t* empty=full+ST;
 const int tid=threadIdx.x,consumer=tid<128;
 const int tile=blockIdx.x,row0=(tile%72)*64,tok0=(tile/72)*BT;
 const int steps=8192/BK,kb0=0,kx=8192;
'''+pipeline+'''
 if(consumer){
  int warp=tid>>5,lane=tid&31,row=warp*16+(lane>>2),col=2*(lane&3);
#pragma unroll
  for(int j=0;j<BT/8;++j){
#pragma unroll
   for(int q=0;q<4;++q){
    int r=row+(q>=2?8:0),t=j*8+col+(q&1),feature=row0+r;
    if(tok0+t<M){
     float v=acc[j*4+q];
     if(feature<4096)Pool[(size_t)(tok0+t)*4096+feature]=__float2bfloat16_rn(v);
     else Logits[(size_t)(tok0+t)*512+feature-4096]=v;
    }
   }
  }
 }
#endif
}
'''+values['HOP_MAP']+'''
template<int BT,int BK,int ST>
void mixed_launch(torch::Tensor W,torch::Tensor X,torch::Tensor Pool,torch::Tensor Logits){
 TORCH_CHECK(W.size(0)==4608 && W.size(1)==8192 && X.size(1)==8192,"mixed TMA shape");
 int M=X.size(0);size_t smem=ST*(64+BT)*BK*2+ST*16;
 auto kern=mixed_kernel<BT,BK,ST>;
 static bool initialized=false;
 if(!initialized){
  TORCH_CHECK(cudaFuncSetAttribute(kern,cudaFuncAttributeMaxDynamicSharedMemorySize,smem)==cudaSuccess,"mixed TMA shared memory");
  initialized=true;
 }
 CUtensorMap mw=make_map(W.data_ptr(),4608,8192,64,BK),mx=make_map(X.data_ptr(),M,8192,BT,BK);
 kern<<<72*((M+BT-1)/BT),256,smem,at::cuda::getCurrentCUDAStream()>>>(mw,mx,Pool.data_ptr<at::BFloat16>()?reinterpret_cast<__nv_bfloat16*>(Pool.data_ptr()):nullptr,Logits.data_ptr<float>(),M);
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace prefill_mixed
void mixed(torch::Tensor W,torch::Tensor X,torch::Tensor Pool,torch::Tensor Logits,int64_t code){
 using namespace prefill_mixed;
 if(code==0)mixed_launch<64,64,4>(W,X,Pool,Logits);
 else if(code==1)mixed_launch<128,64,4>(W,X,Pool,Logits);
 else if(code==2)mixed_launch<128,128,2>(W,X,Pool,Logits);
 else if(code==3)mixed_launch<256,64,2>(W,X,Pool,Logits);
 else TORCH_CHECK(false,"mixed TMA config");
}
'''
source=source.replace('Pool.data_ptr<at::BFloat16>()?reinterpret_cast<__nv_bfloat16*>(Pool.data_ptr()):nullptr','reinterpret_cast<__nv_bfloat16*>(Pool.data_ptr())')
(P/'dev/dense_tma.cu').write_text(source)
(P/'dev/dense_tma_paths.py').write_text('''# TMA mixed-output GEMM: a lossless BF16 concatenated bank, BF16 pooler and FP32 router.
_DENSE_TMA_SRC = '''+repr(source)+'''
_dense_tma_mod=None

def _dense_tma_module():
    global _dense_tma_mod
    if _dense_tma_mod is None:
        import hashlib
        from torch.utils.cpp_extension import load_inline
        cap=torch.cuda.get_device_capability();previous=os.environ.get('TORCH_CUDA_ARCH_LIST')
        os.environ['TORCH_CUDA_ARCH_LIST']='9.0a' if cap==(9,0) else f'{cap[0]}.{cap[1]}'
        try:
            _dense_tma_mod=load_inline(name='prefill_mixed_tma_'+hashlib.sha256(_DENSE_TMA_SRC.encode()).hexdigest()[:12],
                cpp_sources='void mixed(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);',
                cuda_sources=_DENSE_TMA_SRC,functions=['mixed'],extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr','--ptxas-options=-v'],verbose=False)
        finally:
            if previous is None:os.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:os.environ['TORCH_CUDA_ARCH_LIST']=previous
    return _dense_tma_mod
''')
print('Built TMA mixed pipeline',len(source))
