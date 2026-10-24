#include <cstdint>
__global__ void probe(const uint32_t* p,float* out){
 uint32_t a0=p[threadIdx.x*6],a1=p[threadIdx.x*6+1],a2=p[threadIdx.x*6+2],a3=p[threadIdx.x*6+3],b0=p[threadIdx.x*6+4],b1=p[threadIdx.x*6+5];
 float x=0,y=0,z=0,w=0;
 asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};":"+f"(x),"+f"(y),"+f"(z),"+f"(w):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
 out[threadIdx.x*4]=x;out[threadIdx.x*4+1]=y;out[threadIdx.x*4+2]=z;out[threadIdx.x*4+3]=w;
}
