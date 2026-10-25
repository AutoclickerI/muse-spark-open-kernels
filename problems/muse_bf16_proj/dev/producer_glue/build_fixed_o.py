"""Preserve S11's exact config6 O producer; add tile statistics/row apply."""
import ast
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent;REPO=HERE.parents[3]
src=(REPO/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text()
assert hashlib.sha256(src.encode()).hexdigest().startswith('acda6c62')
tree=ast.parse(src)
cuda=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id',None)=='GF_CUDA'))
cuda=cuda[:cuda.index('__global__ void anchor')]
cuda=cuda.replace('namespace gf {','namespace pgfo {',1)
anchor='// All K partitions stay inside one CTA.'
i=cuda.index(anchor)
cuda=cuda[:i]+'}\n'+(HERE/'fixed_o_epilogue.cuh').read_text()+'\nnamespace pgfo {\n'+cuda[i:]
cuda=cuda.replace('template<int R,int T,int KW,int DEPTH,bool TRACE>','template<int R,int T,int KW,int DEPTH,bool FUSE,bool PROFILE>')
cuda=cuda.replace('uint64_t* stamps,int M,int N,int K,bool fp32,bool pdl){',
    'uint64_t* stamps,int M,int N,int K,bool fp32,bool pdl, uint64_t* count,uint64_t* completed,\n'
    'const float* stream,float* sout,B* out,float* av,float* bv,float4* stats,\n'
    'const float* alpha,const float* beta,const float* prew){')
cuda=cuda.replace('  uint64_t clocks[7],times[7];\n','')
cuda=cuda.replace('#define MARK(I) if constexpr(TRACE){if(tid==0){clocks[I]=ck();times[I]=gt();}}','#define MARK(I)')
cuda=cuda.replace('  MARK(3);','  MARK(3);\n  uint64_t generation=0;\n  if constexpr(FUSE)generation=acquire64(completed+blockIdx.x)+1;')
start=cuda.index('  if constexpr(TRACE)__syncthreads();')
end=cuda.index('#undef MARK',start)
cuda=cuda[:start]+'''  if constexpr(FUSE) {
    __syncthreads();
    boundary_finish<PROFILE>((const B*)y,stamps,count,completed,generation,stream,sout,out,av,bv,stats,alpha,beta,prew,sums);
  }
'''+cuda[end:]
cuda+='''
} // namespace pgfo
void run(std::vector<int64_t> v,int64_t fuse,int64_t profile,int64_t stream) {
  if(v.size()!=15)throw std::runtime_error("fixed O boundary ABI");
  cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed=1;
  cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(512);cfg.blockDim=128;cfg.stream=(cudaStream_t)stream;
  cfg.attrs=&attr;cfg.numAttrs=1;
  auto launch=[&]<bool F,bool P>() {
    return cudaLaunchKernelEx(&cfg,pgfo::direct<16,8,4,8,F,P>,
      (const pgfo::B*)v[0],(const pgfo::B*)v[1],(void*)v[2],(uint64_t*)v[3],1,8192,1024,false,true,
      (uint64_t*)v[4],(uint64_t*)v[5],(const float*)v[6],(float*)v[7],(pgfo::B*)v[8],
      (float*)v[9],(float*)v[10],(float4*)v[11],(const float*)v[12],(const float*)v[13],(const float*)v[14]);
  };
  cudaError_t e=!fuse?launch.template operator()<false,false>():profile?
    launch.template operator()<true,true>():launch.template operator()<true,false>();
  if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
'''
cuda='#include <vector>\n'+cuda
(HERE/'fixed_o.cu').write_text(cuda)
print('fixed_o.cu',hashlib.sha256(cuda.encode()).hexdigest())
