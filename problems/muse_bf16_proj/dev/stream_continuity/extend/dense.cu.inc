#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <stdint.h>
#include <vector>
#include <stdexcept>
namespace ew_dense {
using namespace nvcuda;
__device__ __forceinline__ unsigned sp(const void*p){return (unsigned)__cvta_generic_to_shared(p);}
__device__ __forceinline__ void copy16(void*d,const void*s,int valid){asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(sp(d)),"l"(s),"r"(valid?16:0):"memory");}
template<int BM,bool EARLY>__global__ __launch_bounds__(128,2) void run(const __nv_bfloat16*X,const __nv_bfloat16*W,float*Y,int M,int N,int K){
 asm volatile("griddepcontrol.launch_dependents;":::"memory");
 constexpr int BK=64,BN=64,WB=BN*BK,XB=BM*BK,SB=WB+XB;
 extern __shared__ __align__(128) __nv_bfloat16 sm[];
 const int tid=threadIdx.x,warp=tid/32,mt=blockIdx.y*BM,nt=blockIdx.x*BN;
 auto weights=[&](int slot,int k){for(int i=tid;i<WB/8;i+=128){int n=i/(BK/8),c=i%(BK/8)*8;copy16(sm+slot*SB+i*8,W+(long long)(nt+n)*K+k+c,nt+n<N);}};
 auto inputs=[&](int slot,int k){for(int i=tid;i<XB/8;i+=128){int r=i/(BK/8),c=i%(BK/8)*8;copy16(sm+slot*SB+WB+i*8,X+(long long)min(mt+r,M-1)*K+k+c,mt+r<M);}};
 if constexpr(EARLY){weights(0,0);asm volatile("cp.async.commit_group;":::"memory");}
 asm volatile("griddepcontrol.wait;":::"memory");
 if constexpr(!EARLY)weights(0,0);
 inputs(0,0);asm volatile("cp.async.commit_group;cp.async.wait_group 0;":::"memory");__syncthreads();
 constexpr int NC=BM==64?4:1;
 wmma::fragment<wmma::accumulator,16,16,16,float> acc[NC];
 #pragma unroll
 for(int j=0;j<NC;j++)wmma::fill_fragment(acc[j],0.f);
 for(int k=0;k<K;k+=BK){int slot=(k/BK)&1;
  if(k+BK<K){weights(slot^1,k+BK);inputs(slot^1,k+BK);asm volatile("cp.async.commit_group;":::"memory");}
  #pragma unroll
  for(int q=0;q<BK;q+=16){
   wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a;
   wmma::load_matrix_sync(a,sm+slot*SB+WB+(BM==64?warp*16*BK:0)+q,BK);
   #pragma unroll
   for(int j=0;j<NC;j++){
    wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> b;
    wmma::load_matrix_sync(b,sm+slot*SB+(BM==64?j:warp)*16*BK+q,BK);
    wmma::mma_sync(acc[j],a,b,acc[j]);
   }
  }
  asm volatile("cp.async.wait_group 0;":::"memory");__syncthreads();
 }
 float* tmp=(float*)sm;
 #pragma unroll
 for(int j=0;j<NC;j++)wmma::store_matrix_sync(tmp+(BM==64?warp*16*BN+j*16:warp*16),acc[j],BN,wmma::mem_row_major);
 __syncthreads();
 for(int i=tid;i<BM*BN;i+=128)if(mt+i/BN<M && nt+i%BN<N)Y[(long long)(mt+i/BN)*N+nt+i%BN]=tmp[i];
}
template<int BM,bool E>std::vector<int64_t> info(){cudaFuncAttributes a{};int n;cudaFuncGetAttributes(&a,run<BM,E>);cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,run<BM,E>,128,2*(64+BM)*64*2);return {a.numRegs,(int64_t)a.sharedSizeBytes,2*(64+BM)*64*2,n,(int64_t)a.localSizeBytes};}
}
std::vector<int64_t> ew_info(int64_t bm,int64_t early){return bm==64?(early?ew_dense::info<64,true>():ew_dense::info<64,false>()):(early?ew_dense::info<16,true>():ew_dense::info<16,false>());}
void ew_dense_run(int64_t x,int64_t w,int64_t y,int64_t m,int64_t n,int64_t k,int64_t early,int64_t stream){
 cudaLaunchConfig_t c{};c.gridDim=dim3((n+63)/64,(m+(m>32?63:15))/(m>32?64:16));c.blockDim=dim3(128);c.dynamicSmemBytes=2*(64+(m>32?64:16))*64*2;c.stream=(cudaStream_t)stream;
 cudaLaunchAttribute a{};a.id=cudaLaunchAttributeProgrammaticStreamSerialization;a.val.programmaticStreamSerializationAllowed=1;c.attrs=&a;c.numAttrs=1;
 cudaError_t e;
 #define GO(B,E) cudaLaunchKernelEx(&c,ew_dense::run<B,E>,(const __nv_bfloat16*)x,(const __nv_bfloat16*)w,(float*)y,(int)m,(int)n,(int)k)
 if(m>32)e=early?GO(64,true):GO(64,false);else e=early?GO(16,true):GO(16,false);
 #undef GO
 if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));
}
