#include <cuda_runtime.h>
#include <stdint.h>
__device__ __forceinline__ uint32_t addr(const void*p){return (uint32_t)__cvta_generic_to_shared(p);}
__global__ void lut64(const uint32_t* words,uint32_t* output){
 extern __shared__ __align__(256) uint32_t table[];int tid=threadIdx.x,lane=tid&31;
 for(int i=tid;i<256*64;i+=blockDim.x){uint32_t u=i>>6;table[i]=((u&7)<<6)|((u&8)<<12)|((u&112)<<18)|((u&128)<<24);}
 __syncthreads();int gt=blockIdx.x*blockDim.x+tid;uint32_t p=addr(table)+lane*4;uint32_t z=words[gt];
#pragma unroll
 for(int k=0;k<4;++k){uint32_t a=__byte_perm(z,p,0x7604+(k<<4))+(p&0xff00u);uint32_t v;asm volatile("ld.shared.u32 %0,[%1];":"=r"(v):"r"(a));output[gt*4+k]=v;}
}
