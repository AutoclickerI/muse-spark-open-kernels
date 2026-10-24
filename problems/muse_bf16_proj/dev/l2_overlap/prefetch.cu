#include <cuda_runtime.h>
#include <stdint.h>

// Pure byte-address ABI. All pointer arithmetic happens on a CUDA allocation,
// explicitly converted to .global before either PTX instruction consumes it.
// Normal eviction only: no cache set-aside or state surviving a call is used.
namespace p7_l2 {
__device__ __forceinline__ uint4 read16(const char* ptr) {
  uint4 v;
  unsigned long long p=__cvta_generic_to_global(ptr);
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
      :"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"l"(p):"memory");
  return v;
}
template<int MODE>
__global__ void prefetch(const char* a,size_t na,const char* b,size_t nb,unsigned* sink) {
  const size_t tid=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
  const size_t step=(size_t)gridDim.x*blockDim.x;
  unsigned sum=0;
  if constexpr(MODE==0) {
    // A bounded number of CTAs runs full 16-byte loads. The checksum makes
    // every byte observable, and each CTA writes a private sink location.
    for(size_t i=tid*16;i<na+nb;i+=step*16*8) {
      uint4 v[8];
#pragma unroll
      for(int u=0;u<8;++u) {
        size_t j=i+u*step*16;
        v[u]=j<na?read16(a+j):j<na+nb?read16(b+j-na):uint4{};
      }
#pragma unroll
      for(int u=0;u<8;++u) sum^=v[u].x^v[u].y^v[u].z^v[u].w;
    }
    sink[tid]=sum;
  } else {
    // Each lane issues an independent 64 KiB bulk prefetch. This is a hint;
    // correctness never depends on the request completing before the join.
    constexpr size_t chunk=65536;
    for(size_t i=tid*chunk;i<na+nb;i+=step*chunk) {
      const char* ptr=i<na?a+i:b+i-na;
      size_t remain=i<na?na-i:na+nb-i;
      unsigned count=(unsigned)(remain<chunk?remain:chunk);
      unsigned long long address=__cvta_generic_to_global(ptr);
      asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"::"l"(address),"r"(count):"memory");
    }
  }
}
}

extern "C" int p7_prefetch(const void* a,size_t na,const void* b,size_t nb,
                            void* sink,int ctas,int mode,void* stream) {
  if(((uintptr_t)a%16)||(na%16)||((uintptr_t)b%16)||(nb%16)||ctas<1||mode<0||mode>1)
    return (int)cudaErrorInvalidValue;
  // Separate regions must end on a chunk boundary for the bulk grid mapping.
  if(mode && ((na%65536)||(nb%65536))) return (int)cudaErrorInvalidValue;
  if(mode) p7_l2::prefetch<1><<<ctas,128,0,(cudaStream_t)stream>>>
      ((const char*)a,na,(const char*)b,nb,(unsigned*)sink);
  else p7_l2::prefetch<0><<<ctas,128,0,(cudaStream_t)stream>>>
      ((const char*)a,na,(const char*)b,nb,(unsigned*)sink);
  return (int)cudaGetLastError();
}
