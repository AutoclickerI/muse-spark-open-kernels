import torch
from torch.utils.cpp_extension import load_inline
cpp = r'''
#include <torch/extension.h>
void launch(torch::Tensor out);
'''
src = r'''
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <stdint.h>
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n" : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
__global__ void kk(float* out, int mode){
 int lane=threadIdx.x&31, g=lane>>2, q4=lane&3;
 uint32_t a[4], b[2];
 // Modes 0..3 isolate A registers; modes 4..5 isolate B registers.  Every
 // nonzero byte is one, so each C register identifies its fragment source.
 for (int z=0; z<4; ++z) a[z] = (mode < 4) ? ((z == mode) ? 0x38383838u : 0u) : 0x38383838u;
 b[0] = (mode == 4 || mode < 4) ? 0x38383838u : 0u;
 b[1] = (mode == 5 || mode < 4) ? 0x38383838u : 0u;
 float c[4]={0,0,0,0}; mma(c,a,b[0],b[1]);
 for(int e=0;e<4;e++) out[lane*4+e]=c[e];
}
void launch(torch::Tensor out){
 for (int mode=0; mode<6; ++mode) kk<<<1,32>>>(out.data_ptr<float>() + mode*32*4, mode);
}
'''
ext=load_inline('mma_map_probe3', cpp_sources=cpp, cuda_sources=src, functions=['launch'], verbose=False)
out=torch.empty((6,32,4),device='cuda',dtype=torch.float32)
ext.launch(out)
torch.cuda.synchronize()
for mode, row in enumerate(out.cpu()):
 print(f"mode {mode} (A{mode} if <4, B{mode-4} otherwise):")
 print(row)
# A complete matrix product proves the register-to-coordinate mapping, including K.
cpp2 = '#include <torch/extension.h>\nvoid product(torch::Tensor a, torch::Tensor b, torch::Tensor c);'
src2 = src[:src.index('__global__ void kk')] + r'''
__global__ void product_kernel(const uint8_t* A, const uint8_t* B, float* C) {
 int lane=threadIdx.x, g=lane>>2, q=lane&3;
 uint32_t a[4];
 a[0]=*reinterpret_cast<const uint32_t*>(A+g*32+4*q);
 a[1]=*reinterpret_cast<const uint32_t*>(A+(g+8)*32+4*q);
 a[2]=*reinterpret_cast<const uint32_t*>(A+g*32+16+4*q);
 a[3]=*reinterpret_cast<const uint32_t*>(A+(g+8)*32+16+4*q);
 uint32_t b0=*reinterpret_cast<const uint32_t*>(B+g*32+4*q);
 uint32_t b1=*reinterpret_cast<const uint32_t*>(B+g*32+16+4*q);
 float c[4]={0,0,0,0}; mma(c,a,b0,b1);
 C[g*8+2*q]=c[0]; C[g*8+2*q+1]=c[1];
 C[(g+8)*8+2*q]=c[2]; C[(g+8)*8+2*q+1]=c[3];
}
void product(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
 product_kernel<<<1,32>>>(reinterpret_cast<uint8_t*>(a.data_ptr()), reinterpret_cast<uint8_t*>(b.data_ptr()), c.data_ptr<float>());
}
'''
ext2=load_inline('mma_map_product',cpp_sources=cpp2,cuda_sources=src2,functions=['product'],verbose=False)
torch.manual_seed(7)
a=torch.randint(-3,4,(16,32),device='cuda').to(torch.float8_e4m3fn)
b=torch.randint(-3,4,(8,32),device='cuda').to(torch.float8_e4m3fn)
c=torch.empty((16,8),device='cuda',dtype=torch.float32)
ext2.product(a,b,c)
torch.testing.assert_close(c,a.float()@b.float().T,rtol=0,atol=0)
print('PASS: exact m16n8k32 FP8 matrix product; A0/A1 K0..15, A2/A3 K16..31; C=(g,2q),(g,2q+1),(g+8,2q),(g+8,2q+1)')
