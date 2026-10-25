#include <cuda_runtime.h>
__global__ void plain(float* p){*p=1;}
__global__ void with_wait(float* p){asm volatile("griddepcontrol.wait;":::"memory");*p=1;}
__global__ void with_release(float* p){asm volatile("griddepcontrol.launch_dependents;":::"memory");*p=1;}
__global__ void ordinary_before(float* p,const float* __restrict__ w){float x=*w;asm volatile("griddepcontrol.wait;":::"memory");*p=x;}
__global__ void volatile_before(float* p,const float* w){float x=*reinterpret_cast<const volatile float*>(w);asm volatile("griddepcontrol.wait;":::"memory");*p=x;}
__global__ void asm_before(float* p,const float* w){float x;asm volatile("ld.global.cg.f32 %0,[%1];":"=f"(x):"l"(w):"memory");asm volatile("griddepcontrol.wait;":::"memory");*p=x;}
__global__ void asm_volatile_before(float* p,const float* w){float x;asm volatile("ld.volatile.global.f32 %0,[%1];":"=f"(x):"l"(w):"memory");asm volatile("griddepcontrol.wait;":::"memory");*p=x;}
