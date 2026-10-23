import torch
from torch.utils.cpp_extension import load_inline
c=r'''
#include <cuda_runtime.h>
__global__ void k(const unsigned char* p,unsigned char* o){extern __shared__ unsigned char s[]; __shared__ unsigned long long bar; if(threadIdx.x==0){unsigned b=(unsigned)__cvta_generic_to_shared(&bar); asm volatile("mbarrier.init.shared.b64 [%0], 1;"::"r"(b):"memory"); asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], 64;"::"r"(b):"memory"); asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], 64, [%2];"::"r"((unsigned)__cvta_generic_to_shared(s)),"l"(p),"r"(b):"memory"); asm volatile("{ .reg .pred %%p; BULK_WAIT: mbarrier.try_wait.parity.shared.b64 %%p, [%0], 0; @!%%p bra BULK_WAIT; }"::"r"(b):"memory");} __syncthreads();if(threadIdx.x<64)o[threadIdx.x]=s[threadIdx.x];}
void go(long p,long o,long s){k<<<1,128,64,(cudaStream_t)s>>>((const unsigned char*)p,(unsigned char*)o);}
'''
cpp='void go(long p,long o,long s);'
e=load_inline(name='bulk_test_sm120',cpp_sources=cpp,cuda_sources=c,functions=['go'],extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr','-lineinfo'],verbose=False)
a=torch.arange(64,dtype=torch.uint8,device='cuda');b=torch.zeros_like(a);e.go(a.data_ptr(),b.data_ptr(),torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize();print(a,b,torch.equal(a,b))
